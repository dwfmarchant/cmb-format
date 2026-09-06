"""Encode arrays and descriptors, and read CMB headers and array data.

See ``docs/binary-format.md`` for the format and ``_file`` for file assembly.
"""

import hashlib
import json
import os
import struct

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cmb_format._padding import (
    normalize_default_padding,
    normalize_integer_array,
    validate_default_padding_shape,
)

__all__ = [
    "DTYPE_TO_NUMPY",
    "INT8_MAX",
    "KIND_ITEMSIZE_TO_DTYPE_NAME",
    "MAGIC",
    "READABLE_FORMAT_VERSIONS",
    "WRITTEN_FORMAT_VERSION",
    "array_dtype_name",
    "base_mesh_descriptor",
    "descriptor_shape",
    "padding_as_json",
    "padding_belongs_to_base_mesh",
    "read_array",
    "read_arrays",
    "read_header",
    "resolve_reference_n_cells",
    "resolve_shared_padding",
    "serialize_array",
    "serialize_arrays",
    "serialize_mesh",
    "sha256_hex",
    "shape_from_mesh_arrays",
    "summarize_models",
    "to_le_bytes",
]

MAGIC = b"CELLMODB"

# An integer counter of incompatible wire changes, unrelated to this
# package's own version. See docs/binary-format.md's Versioning section.
WRITTEN_FORMAT_VERSION = 1
READABLE_FORMAT_VERSIONS = frozenset({1})

DTYPE_TO_NUMPY = {
    "float64": "<f8",
    "float32": "<f4",
    "int64": "<i8",
    "int32": "<i4",
    "int16": "<i2",
    "int8": "<i1",
}
KIND_ITEMSIZE_TO_DTYPE_NAME = {
    ("f", 8): "float64",
    ("f", 4): "float32",
    ("i", 8): "int64",
    ("i", 4): "int32",
    ("i", 2): "int16",
    ("i", 1): "int8",
}

INT8_MAX = np.iinfo(np.int8).max


def array_dtype_name(arr: NDArray) -> str:
    key = (arr.dtype.kind, arr.dtype.itemsize)
    if key not in KIND_ITEMSIZE_TO_DTYPE_NAME:
        raise TypeError(f"unsupported array dtype for CMB: {arr.dtype}")
    return KIND_ITEMSIZE_TO_DTYPE_NAME[key]


def to_le_bytes(arr: ArrayLike, dtype_name: str) -> bytes:
    numpy_dtype = np.dtype(DTYPE_TO_NUMPY[dtype_name])
    return np.ascontiguousarray(arr, dtype=numpy_dtype).tobytes()


def sha256_hex(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def serialize_array(arr: ArrayLike, buffer: bytearray) -> dict:
    """Append an array's little-endian bytes to buffer and return its descriptor.

    Accepts array-like input with a supported dtype. Mutates the supplied
    bytearray; the descriptor's offset is the buffer length before appending.
    The descriptor records dtype, shape, byte length, and a SHA-256 checksum.
    """
    arr = np.asarray(arr)
    dtype_name = array_dtype_name(arr)
    raw = to_le_bytes(arr, dtype_name)
    offset = len(buffer)
    buffer.extend(raw)
    return {
        "dtype": dtype_name,
        "shape": list(arr.shape),
        "offset": offset,
        "length": len(raw),
        "checksum": sha256_hex(raw),
    }


def serialize_arrays(arrays: dict, buffer: bytearray) -> dict:
    return {name: serialize_array(arrays[name], buffer) for name in sorted(arrays)}


def shape_from_mesh_arrays(
    mesh_class: str, arrays: dict
) -> tuple[int, int, int] | None:
    """Derive an axis shape from a raw mesh descriptor's geometry arrays."""
    if mesh_class not in ("TensorMesh", "UniformTensorMesh"):
        return None
    if not isinstance(arrays, dict):
        raise ValueError(f"{mesh_class} descriptor arrays must be a mapping")
    if mesh_class == "TensorMesh":
        widths = []
        for name in ("h_x", "h_y", "h_z"):
            if name not in arrays:
                raise ValueError(
                    f"TensorMesh descriptor arrays missing required key {name!r}"
                )
            try:
                width = np.asarray(arrays[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"TensorMesh descriptor arrays[{name!r}] must be a 1D array"
                ) from exc
            if width.ndim != 1 or width.size == 0:
                raise ValueError(
                    f"TensorMesh descriptor arrays[{name!r}] must be a non-empty "
                    "1D array"
                )
            widths.append(width)
        return tuple(int(width.size) for width in widths)
    if mesh_class == "UniformTensorMesh":
        if "shape" not in arrays:
            raise ValueError(
                "UniformTensorMesh descriptor arrays missing required key 'shape'"
            )
        shape_name = "UniformTensorMesh descriptor arrays['shape']"
        shape = normalize_integer_array(
            arrays["shape"],
            name=shape_name,
            shape=(3,),
            minimum=1,
            value_description="three positive integer values",
        )
        return tuple(int(value) for value in shape)
    return None


def base_mesh_descriptor(mesh_dict: dict) -> dict | None:
    """Return a nested base-mesh descriptor, or None when absent.

    Checks that a supplied value is a mapping containing ``mesh_class``;
    it does not validate the full geometry schema.
    """
    if "base_mesh" not in mesh_dict:
        return None
    base = mesh_dict["base_mesh"]
    mode = mesh_dict.get("mode")
    mesh_class = mesh_dict.get("mesh_class")
    if mode == "embedded" and mesh_class == "OctreeMesh":
        owner = "OctreeMesh"
    elif mode == "reference":
        owner = "reference mesh"
    else:
        owner = "mesh"
    if not isinstance(base, dict):
        raise ValueError(f"{owner} descriptor 'base_mesh' must be a mapping")
    if "mesh_class" not in base:
        raise ValueError(
            f"{owner} descriptor 'base_mesh' missing required key 'mesh_class'"
        )
    return base


def descriptor_shape(mesh_dict: dict) -> tuple[int, int, int] | None:
    """Derive an axis shape when this embedded/reference descriptor has one."""
    mode = mesh_dict.get("mode")
    mesh_class = mesh_dict.get("mesh_class")
    if mode == "embedded" and mesh_class == "OctreeMesh":
        base = base_mesh_descriptor(mesh_dict)
        if base is None:
            raise ValueError("OctreeMesh descriptor missing required key 'base_mesh'")
    elif mode == "reference":
        base = base_mesh_descriptor(mesh_dict)
        if base is None:
            return None
    else:
        # Nested base_mesh descriptors omit mode when serialized, but their
        # mesh_class/arrays shape is still enough to derive the cell counts.
        if mesh_class in ("TensorMesh", "UniformTensorMesh"):
            return shape_from_mesh_arrays(mesh_class, mesh_dict.get("arrays", {}))
        return None

    return shape_from_mesh_arrays(base.get("mesh_class"), base.get("arrays", {}))


def padding_belongs_to_base_mesh(mesh_dict: dict) -> bool:
    """Whether a descriptor's canonical padding field belongs on its base."""
    base = mesh_dict.get("base_mesh")
    return isinstance(base, dict) and (
        mesh_dict.get("mode") == "reference"
        or mesh_dict.get("mesh_class") == "OctreeMesh"
    )


def padding_as_json(
    value: ArrayLike | None, shape: tuple[int, int, int] | None = None
) -> list[int] | None:
    """Normalize a padding field and validate it against an optional shape."""
    if value is None:
        return None
    padding = normalize_default_padding(value)
    if shape is not None:
        validate_default_padding_shape(padding, shape)
    return padding.tolist()


def resolve_shared_padding(
    outer: ArrayLike | None,
    nested: ArrayLike | None,
    shape: tuple[int, int, int] | None = None,
) -> list[int] | None:
    """Resolve outer and nested padding to one JSON list.

    The nested base mesh owns the canonical setting. Use its value when
    present, or the outer value as a fallback. If both are present, their
    normalized values must match. An optional shape bounds each opposing pair.
    """
    outer_padding = normalize_default_padding(outer)
    nested_padding = normalize_default_padding(nested)
    if (
        outer_padding is not None
        and nested_padding is not None
        and not np.array_equal(outer_padding, nested_padding)
    ):
        raise ValueError("outer and nested default_padding values must match")
    padding = nested_padding if nested_padding is not None else outer_padding
    if padding is not None and shape is not None:
        validate_default_padding_shape(padding, shape)
    return None if padding is None else padding.tolist()


def serialize_mesh(mesh_dict: dict, buffer: bytearray) -> dict:
    """Append a mesh's geometry arrays to buffer and return its descriptor.

    Embedded input contains raw arrays; the result replaces them with array
    descriptors. Reference input supplies ``n_cells`` and may include a base
    mesh. Base-mesh arrays are appended to the same buffer.

    Normalizes padding and checks it against available axis shapes. This
    routine does not perform full geometric validation or reorder cells.
    """
    mode = mesh_dict.get("mode")
    if mode == "embedded":
        header = {
            "mode": "embedded",
            "mesh_class": mesh_dict["mesh_class"],
            "arrays": serialize_arrays(mesh_dict["arrays"], buffer),
        }
    elif mode == "reference":
        # Reference mode omits mesh_class; optional base geometry is added below.
        header = {"mode": "reference", "n_cells": mesh_dict["n_cells"]}
    else:
        raise NotImplementedError(f"unsupported mesh mode for serialization: {mode!r}")

    base = base_mesh_descriptor(mesh_dict)
    has_shared_padding = padding_belongs_to_base_mesh(mesh_dict)
    if has_shared_padding:
        # An octree and its base mesh share one padding setting, stored on
        # the base descriptor. A value found on the outer descriptor instead
        # is moved there.
        shared_padding = resolve_shared_padding(
            mesh_dict.get("default_padding"),
            base.get("default_padding"),
            descriptor_shape(mesh_dict),
        )
        padding = None
    else:
        shared_padding = None
        padding = padding_as_json(
            mesh_dict.get("default_padding"), descriptor_shape(mesh_dict)
        )
    if padding is not None:
        header["default_padding"] = padding
    if isinstance(base, dict):
        base_header = {
            "mesh_class": base["mesh_class"],
            "arrays": serialize_arrays(base["arrays"], buffer),
        }
        if has_shared_padding:
            base_padding = shared_padding
        else:
            base_padding = padding_as_json(
                base.get("default_padding"), descriptor_shape(base)
            )
        if base_padding is not None:
            base_header["default_padding"] = base_padding
        header["base_mesh"] = base_header
    return header


def resolve_reference_n_cells(mesh_n_cells: int | None, normalized_models: dict) -> int:
    """Determine a reference-mode file's cell count.

    All model arrays must be one-dimensional and have equal lengths. Their
    length must match ``mesh_n_cells`` when supplied. Without models, use
    ``mesh_n_cells``; raise ValueError if neither source is available.
    """
    lengths = {}
    for name, entry in normalized_models.items():
        arr = np.asarray(entry["array"])
        if arr.ndim != 1:
            raise ValueError(
                f"model {name!r} must be a 1D array (docs/binary-format.md's "
                f"models schema), got shape {arr.shape}"
            )
        lengths[name] = arr.shape[0]
    if not lengths:
        if mesh_n_cells is None:
            raise ValueError(
                "a reference-mode file needs either a mesh or at least one "
                "model to determine n_cells"
            )
        return mesh_n_cells
    distinct = set(lengths.values())
    if len(distinct) > 1:
        raise ValueError(f"models disagree on cell count: {lengths}")
    (n_cells,) = distinct
    if mesh_n_cells is not None and n_cells != mesh_n_cells:
        raise ValueError(
            f"mesh has n_cells={mesh_n_cells}, but models have {n_cells} cells"
        )
    return n_cells


def read_array(f, data_start: int, descriptor: dict) -> NDArray:
    """Read an array from an open binary file and verify its checksum.

    Seeks to ``data_start + descriptor["offset"]`` and reads the descriptor's
    byte length. Raises ValueError for a short read or checksum mismatch.
    Returns a read-only NumPy array with the recorded dtype and shape.
    The file position changes; this function does not validate the full header.
    """
    f.seek(data_start + descriptor["offset"])
    raw = f.read(descriptor["length"])
    if len(raw) != descriptor["length"]:
        raise ValueError("truncated CMB file: array data shorter than declared length")
    if sha256_hex(raw) != descriptor["checksum"]:
        raise ValueError("CMB checksum mismatch: array data is corrupted")
    numpy_dtype = np.dtype(DTYPE_TO_NUMPY[descriptor["dtype"]])
    return np.frombuffer(raw, dtype=numpy_dtype).reshape(tuple(descriptor["shape"]))


def read_arrays(f, data_start: int, descriptors: dict) -> dict:
    """Read a mapping of named array descriptors from an open binary file.

    Returns a dictionary of NumPy arrays. Each read uses `read_array`, including
    its offset handling and checksum verification, and changes the file position.
    """
    return {name: read_array(f, data_start, d) for name, d in descriptors.items()}


def read_header(f) -> tuple[dict, int]:
    """Read a CMB file's trailing JSON header from an open binary file.

    Checks the minimum file size, trailing magic, header-length bounds, and
    format version. Parses the JSON without validating the full header schema
    or reading array data.

    Returns ``(header, data_start)``, where ``data_start`` is byte 8 and array
    offsets are relative to it. Changes the file position.
    """
    f.seek(0, os.SEEK_END)
    total_length = f.tell()
    if total_length < len(MAGIC) + 16:
        raise ValueError("file too small to be a valid CMB file")

    f.seek(total_length - 8)
    if f.read(8) != MAGIC:
        raise ValueError("invalid or corrupt CMB file: trailing magic not found")

    f.seek(total_length - 16)
    (header_length,) = struct.unpack("<Q", f.read(8))
    header_start = total_length - 16 - header_length
    data_start = len(MAGIC)
    if header_start < data_start:
        raise ValueError("invalid CMB file: header length exceeds file size")

    f.seek(header_start)
    header = json.loads(f.read(header_length).decode("utf-8"))
    version = header.get("format_version")
    if version not in READABLE_FORMAT_VERSIONS:
        readable = ", ".join(str(v) for v in sorted(READABLE_FORMAT_VERSIONS))
        raise ValueError(
            f"unsupported CMB format_version: {version!r}; "
            f"this build of cmb_format reads {{{readable}}}"
        )
    return header, data_start


def summarize_models(header: dict) -> dict:
    """Return each model's metadata, dtype, and shape from a parsed header.

    Returns a dictionary keyed by model name. Does not read array data or
    verify checksums.
    """
    return {
        name: {
            "metadata": model_header.get("metadata", {}),
            "dtype": model_header["array"]["dtype"],
            "shape": model_header["array"]["shape"],
        }
        for name, model_header in header.get("models", {}).items()
    }
