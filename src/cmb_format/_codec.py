"""Encode arrays and descriptors, and read CMB headers and array data.

See ``docs/binary-format.md`` for the format and ``_file`` for file assembly.
"""

import hashlib
import json
import math
import os
import re
import struct
from numbers import Integral

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
    "raw_mesh_shape",
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
    "validate_model_lengths",
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

# Maximum value representable by the int8 dtype used for octree levels.

_MESH_CLASSES = ("TensorMesh", "UniformTensorMesh", "OctreeMesh")
_GEOMETRY_KEYS = {
    "TensorMesh": frozenset({"origin", "h_x", "h_y", "h_z"}),
    "UniformTensorMesh": frozenset({"origin", "cell_size", "shape"}),
    "OctreeMesh": frozenset({"level", "position"}),
}
_ARRAY_FIELDS = ("dtype", "shape", "offset", "length", "checksum")
_CHECKSUM_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)


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


def _nonnegative_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _raw_array(arr, *, name: str, dtypes: tuple[str, ...] | None = None):
    try:
        result = np.asarray(arr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an array") from exc
    if dtypes is not None:
        try:
            dtype_name = array_dtype_name(result)
        except TypeError as exc:
            raise ValueError(
                f"{name} has an unsupported dtype: {result.dtype}"
            ) from exc
        if dtype_name not in dtypes:
            expected = ", ".join(dtypes)
            raise ValueError(f"{name} must have dtype {expected}; got {dtype_name}")
    return result


def _validate_raw_geometry_arrays(mesh_class: str, arrays: dict, *, context: str):
    if not isinstance(arrays, dict):
        raise ValueError(f"{context} arrays must be a mapping")
    expected = _GEOMETRY_KEYS[mesh_class]
    actual = set(arrays)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing required key(s) {missing}")
        if extra:
            detail.append(f"unexpected key(s) {extra}")
        raise ValueError(
            f"{context} arrays must contain exactly the required keys "
            f"({'; '.join(detail)})"
        )

    if mesh_class == "TensorMesh":
        origin = _raw_array(
            arrays["origin"], name=f"{context} arrays['origin']", dtypes=("float64",)
        )
        if origin.shape != (3,):
            raise ValueError(f"{context} arrays['origin'] must have shape (3,)")
        shape = []
        for name in ("h_x", "h_y", "h_z"):
            width = _raw_array(
                arrays[name], name=f"{context} arrays['{name}']", dtypes=("float64",)
            )
            if width.ndim != 1 or width.size == 0:
                raise ValueError(
                    f"{context} arrays['{name}'] must be a non-empty 1D array"
                )
            shape.append(int(width.size))
        return tuple(shape)

    if mesh_class == "UniformTensorMesh":
        for name in ("origin", "cell_size"):
            value = _raw_array(
                arrays[name], name=f"{context} arrays['{name}']", dtypes=("float64",)
            )
            if value.shape != (3,):
                raise ValueError(f"{context} arrays['{name}'] must have shape (3,)")
        raw_shape = _raw_array(
            arrays["shape"],
            name=f"{context} arrays['shape']",
            dtypes=("int32", "int64"),
        )
        if raw_shape.shape != (3,):
            raise ValueError(f"{context} arrays['shape'] must have shape (3,)")
        values = normalize_integer_array(
            raw_shape,
            name=f"{context} arrays['shape']",
            shape=(3,),
            minimum=1,
            value_description="three positive integer values",
        )
        return tuple(int(value) for value in values)

    level = _raw_array(
        arrays["level"], name=f"{context} arrays['level']", dtypes=("int8",)
    )
    position = _raw_array(
        arrays["position"],
        name=f"{context} arrays['position']",
        dtypes=("int32", "int64"),
    )
    if level.ndim != 1 or position.ndim != 1:
        raise ValueError(f"{context} octree level and position must be 1D arrays")
    if level.size != position.size:
        raise ValueError(
            f"{context} octree level and position arrays must have equal lengths"
        )
    return int(level.size)


def _validate_power_of_two_shape(shape: tuple[int, int, int], *, context: str) -> None:
    if any(value <= 0 or value & (value - 1) for value in shape):
        raise ValueError(f"{context} base-grid dimensions must each be powers of two")


def _validate_raw_base_mesh(base: dict, *, context: str) -> tuple[int, int, int]:
    if not isinstance(base, dict):
        raise ValueError(f"{context} must be a mapping")
    if base.get("mesh_class") != "UniformTensorMesh":
        raise ValueError(f"{context} mesh_class must be 'UniformTensorMesh'")
    shape = _validate_raw_geometry_arrays(
        "UniformTensorMesh", base.get("arrays"), context=f"{context}"
    )
    _validate_power_of_two_shape(shape, context=context)
    padding_as_json(base.get("default_padding"), shape)
    return shape


def _validate_raw_mesh(mesh: dict) -> tuple[int, tuple[int, int, int] | None]:
    if not isinstance(mesh, dict):
        raise ValueError("mesh descriptor must be a mapping")
    mode = mesh.get("mode")
    if mode == "embedded":
        mesh_class = mesh.get("mesh_class")
        if not isinstance(mesh_class, str) or mesh_class not in _MESH_CLASSES:
            raise ValueError(f"unsupported embedded mesh_class: {mesh_class!r}")
        shape_or_count = _validate_raw_geometry_arrays(
            mesh_class, mesh.get("arrays"), context=mesh_class
        )
        if mesh_class == "OctreeMesh":
            if "base_mesh" not in mesh:
                raise ValueError(
                    "OctreeMesh descriptor missing required key 'base_mesh'"
                )
            base_shape = _validate_raw_base_mesh(
                mesh["base_mesh"], context="OctreeMesh base_mesh"
            )
            resolve_shared_padding(
                mesh.get("default_padding"),
                mesh["base_mesh"].get("default_padding"),
                base_shape,
            )
            return int(shape_or_count), base_shape
        if "base_mesh" in mesh:
            raise ValueError(f"{mesh_class} does not support base_mesh")
        shape = tuple(shape_or_count)
        padding_as_json(mesh.get("default_padding"), shape)
        return math.prod(shape), shape
    if mode == "reference":
        n_cells = _nonnegative_integer(mesh.get("n_cells"), name="reference n_cells")
        base = mesh.get("base_mesh")
        if "base_mesh" not in mesh:
            padding_as_json(mesh.get("default_padding"))
            return n_cells, None
        if base is None:
            raise ValueError("reference mesh 'base_mesh' must be a mapping")
        shape = _validate_raw_base_mesh(base, context="reference base_mesh")
        resolve_shared_padding(
            mesh.get("default_padding"), base.get("default_padding"), shape
        )
        return n_cells, shape
    raise ValueError(f"unsupported mesh mode for serialization: {mode!r}")


def serialize_array(arr: ArrayLike, buffer: bytearray) -> dict:
    """Append an array's little-endian bytes to buffer and return its descriptor.

    Accepts array-like input with a supported dtype. Mutates the supplied
    bytearray; the descriptor's offset is the buffer length before appending.
    The descriptor records dtype, shape, byte length, and a SHA-256 checksum.
    """
    arr = np.asarray(arr)
    if arr.ndim > 1:
        raise ValueError(f"arrays must be at most 1D, got shape {arr.shape}")
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


def raw_mesh_shape(mesh_dict: dict) -> tuple[int, int, int] | None:
    """Return the axis shape represented by raw mesh arrays.

    This helper accepts writer input containing NumPy arrays. Parsed array
    descriptors from a file are not valid inputs; use their recorded shapes
    or read the arrays instead.
    """
    if not isinstance(mesh_dict, dict):
        raise ValueError("mesh descriptor must be a mapping")
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
        # Nested base meshes omit mode. Keep this helper's original limited
        # shape derivation; full geometry checks happen on I/O paths.
        if mesh_class in ("TensorMesh", "UniformTensorMesh"):
            return shape_from_mesh_arrays(mesh_class, mesh_dict.get("arrays", {}))
        return None
    return shape_from_mesh_arrays(base.get("mesh_class"), base.get("arrays", {}))


def descriptor_shape(mesh_dict: dict) -> tuple[int, int, int] | None:
    """Compatibility wrapper for :func:`raw_mesh_shape`.

    The name is retained for callers of the original API. It accepts raw
    writer mesh dictionaries, not parsed array descriptors.
    """
    return raw_mesh_shape(mesh_dict)


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
    _validate_raw_mesh(mesh_dict)
    mode = mesh_dict["mode"]
    if mode == "embedded":
        header = {
            "mode": "embedded",
            "mesh_class": mesh_dict["mesh_class"],
            "arrays": serialize_arrays(mesh_dict["arrays"], buffer),
        }
    else:
        # Reference mode omits mesh_class; optional base geometry is added below.
        header = {
            "mode": "reference",
            "n_cells": _nonnegative_integer(
                mesh_dict["n_cells"], name="reference n_cells"
            ),
        }

    base = base_mesh_descriptor(mesh_dict)
    has_shared_padding = padding_belongs_to_base_mesh(mesh_dict)
    if has_shared_padding:
        # An octree and its base mesh share one padding setting, stored on
        # the base descriptor. A value found on the outer descriptor instead
        # is moved there.
        shared_padding = resolve_shared_padding(
            mesh_dict.get("default_padding"),
            base.get("default_padding"),
            raw_mesh_shape(mesh_dict),
        )
        padding = None
    else:
        shared_padding = None
        padding = padding_as_json(
            mesh_dict.get("default_padding"), raw_mesh_shape(mesh_dict)
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
                base.get("default_padding"), raw_mesh_shape(base)
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
    if mesh_n_cells is not None:
        mesh_n_cells = _nonnegative_integer(mesh_n_cells, name="reference n_cells")
    if not isinstance(normalized_models, dict):
        raise ValueError("models must be a mapping")
    lengths = {}
    for name, entry in normalized_models.items():
        if not isinstance(entry, dict) or "array" not in entry:
            raise ValueError(f"model {name!r} must be an object with an 'array' field")
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
                "a reference-mode file needs n_cells or at least one model "
                "to determine n_cells"
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


def _validate_array_descriptor(
    descriptor: dict, *, context: str, data_size: int | None = None
) -> tuple[str, list[int], int, int]:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{context} must be an object")
    missing = [name for name in _ARRAY_FIELDS if name not in descriptor]
    if missing:
        raise ValueError(f"{context} missing required field(s): {missing}")
    dtype = descriptor["dtype"]
    if not isinstance(dtype, str) or dtype not in DTYPE_TO_NUMPY:
        raise ValueError(f"{context}.dtype has unsupported value {dtype!r}")
    shape = descriptor["shape"]
    if not isinstance(shape, list) or len(shape) > 1:
        raise ValueError(f"{context}.shape must contain zero or one dimension")
    dimensions = []
    for index, dimension in enumerate(shape):
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(
            dimension, Integral
        ):
            raise ValueError(f"{context}.shape[{index}] must be a non-negative integer")
        dimension = int(dimension)
        if dimension < 0:
            raise ValueError(f"{context}.shape[{index}] must be a non-negative integer")
        dimensions.append(dimension)
    offset = _nonnegative_integer(descriptor["offset"], name=f"{context}.offset")
    length = _nonnegative_integer(descriptor["length"], name=f"{context}.length")
    expected = (dimensions[0] if dimensions else 1) * np.dtype(
        DTYPE_TO_NUMPY[dtype]
    ).itemsize
    if length != expected:
        raise ValueError(
            f"{context}.length={length} does not match shape {shape!r} "
            f"and dtype {dtype!r} "
            f"({expected} bytes expected)"
        )
    checksum = descriptor["checksum"]
    if not isinstance(checksum, str) or _CHECKSUM_RE.fullmatch(checksum) is None:
        raise ValueError(f"{context}.checksum must be a sha256: hexadecimal digest")
    if data_size is not None and offset + length > data_size:
        raise ValueError(
            f"{context} extends beyond the data section "
            f"({offset + length} > {data_size})"
        )
    return dtype, dimensions, offset, length


def read_array(f, data_start: int, descriptor: dict) -> NDArray:
    """Read and checksum-verify one array descriptor.

    Descriptor structure and byte-length agreement are checked before reading.
    Standalone buffers have no CMB trailer, so file bounds are checked by
    :func:`read_header`; callers reading a normal CMB file should use the
    descriptors returned by that function. The result is read-only, and the
    file position changes.
    """
    dtype, shape, offset, length = _validate_array_descriptor(
        descriptor, context="array descriptor"
    )
    f.seek(data_start + offset)
    raw = f.read(length)
    if len(raw) != length:
        raise ValueError("truncated CMB file: array data shorter than declared length")
    if sha256_hex(raw).casefold() != descriptor["checksum"].casefold():
        raise ValueError("CMB checksum mismatch: array data is corrupted")
    numpy_dtype = np.dtype(DTYPE_TO_NUMPY[dtype])
    return np.frombuffer(raw, dtype=numpy_dtype).reshape(tuple(shape))


def read_arrays(f, data_start: int, descriptors: dict) -> dict:
    """Read a mapping of named array descriptors from an open binary file.

    Returns a dictionary of NumPy arrays. Each read uses `read_array`, including
    its offset handling and checksum verification, and changes the file position.
    """
    return {name: read_array(f, data_start, d) for name, d in descriptors.items()}


def read_header(f) -> tuple[dict, int]:
    """Read a CMB file's trailing JSON header from an open binary file.

    Checks the file framing, header schema, array descriptor bounds, mesh
    geometry, and that every model is one value per cell. It reads and
    checksum-verifies a three-element ``shape`` array when a
    ``UniformTensorMesh`` needs its values for cell counts and padding checks
    (12 bytes for int32 or 24 bytes for int64), so this function may raise an
    array checksum error. Other array payloads remain unread and unverified.

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
    if not isinstance(header, dict):
        raise ValueError("CMB header must be a JSON object")
    version = header.get("format_version")
    if (
        isinstance(version, (bool, np.bool_))
        or not isinstance(version, Integral)
        or version not in READABLE_FORMAT_VERSIONS
    ):
        readable = ", ".join(str(v) for v in sorted(READABLE_FORMAT_VERSIONS))
        raise ValueError(
            f"unsupported CMB format_version: {version!r}; "
            f"this build of cmb_format reads {{{readable}}}"
        )
    models = header.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("CMB header 'models' must be an object")
    mesh = header.get("mesh")
    if not isinstance(mesh, dict):
        raise ValueError("CMB header 'mesh' must be an object")
    data_size = header_start - data_start
    for name, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"model {name!r} must be an object")
        if "array" not in entry:
            raise ValueError(f"model {name!r} missing required field 'array'")
        _, shape, _, _ = _validate_array_descriptor(
            entry["array"], context=f"model {name!r} array", data_size=data_size
        )
        if len(shape) != 1:
            raise ValueError(f"model {name!r} must be a 1D array, got shape {shape!r}")
    n_cells = _validate_parsed_mesh(f, mesh, data_start, data_size)
    validate_model_lengths(models, n_cells)
    return header, data_start


def _parsed_geometry_arrays(
    f,
    arrays: dict,
    mesh_class: str,
    data_start: int,
    data_size: int | None,
    *,
    context: str,
):
    if not isinstance(arrays, dict):
        raise ValueError(f"{context} arrays must be an object")
    expected = _GEOMETRY_KEYS[mesh_class]
    actual = set(arrays)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing required key(s) {missing}")
        if extra:
            detail.append(f"unexpected key(s) {extra}")
        raise ValueError(
            f"{context} arrays must contain exactly the required keys "
            f"({'; '.join(detail)})"
        )
    descriptors = {}
    for name in expected:
        descriptors[name] = _validate_array_descriptor(
            arrays[name], context=f"{context} arrays[{name!r}]", data_size=data_size
        )

    def require(name, *, dtype, shape=None, rank=None):
        actual_dtype, actual_shape, _, _ = descriptors[name]
        if actual_dtype not in dtype:
            raise ValueError(f"{context} arrays[{name!r}] must have dtype {dtype}")
        if shape is not None and actual_shape != shape:
            raise ValueError(
                f"{context} arrays[{name!r}] must have shape {shape}, "
                f"got {actual_shape}"
            )
        if rank is not None and len(actual_shape) != rank:
            raise ValueError(f"{context} arrays[{name!r}] must be {rank}D")

    if mesh_class == "TensorMesh":
        require("origin", dtype=("float64",), shape=[3])
        shape = []
        for name in ("h_x", "h_y", "h_z"):
            require(name, dtype=("float64",), rank=1)
            dimension = descriptors[name][1][0]
            if dimension == 0:
                raise ValueError(f"{context} arrays[{name!r}] must be non-empty")
            shape.append(dimension)
        return tuple(shape)
    if mesh_class == "UniformTensorMesh":
        require("origin", dtype=("float64",), shape=[3])
        require("cell_size", dtype=("float64",), shape=[3])
        require("shape", dtype=("int32", "int64"), shape=[3])
        values = read_array(f, data_start, arrays["shape"])
        normalized = normalize_integer_array(
            values,
            name=f"{context} arrays['shape']",
            shape=(3,),
            minimum=1,
            value_description="three positive integer values",
        )
        return tuple(int(value) for value in normalized)
    require("level", dtype=("int8",), rank=1)
    require("position", dtype=("int32", "int64"), rank=1)
    level_length = descriptors["level"][1][0]
    if level_length != descriptors["position"][1][0]:
        raise ValueError(
            f"{context} octree level and position arrays must have equal lengths"
        )
    return level_length


def _validate_parsed_base_mesh(
    f, base: dict, data_start: int, data_size: int | None, *, context: str
) -> tuple[int, int, int]:
    if not isinstance(base, dict):
        raise ValueError(f"{context} must be an object")
    if base.get("mesh_class") != "UniformTensorMesh":
        raise ValueError(f"{context}.mesh_class must be 'UniformTensorMesh'")
    shape = _parsed_geometry_arrays(
        f,
        base.get("arrays"),
        "UniformTensorMesh",
        data_start,
        data_size,
        context=context,
    )
    _validate_power_of_two_shape(shape, context=context)
    padding_as_json(base.get("default_padding"), shape)
    return shape


def _validate_parsed_mesh(f, mesh: dict, data_start: int, data_size: int | None) -> int:
    mode = mesh.get("mode")
    if mode == "reference":
        n_cells = _nonnegative_integer(mesh.get("n_cells"), name="reference n_cells")
        base = mesh.get("base_mesh")
        if "base_mesh" not in mesh:
            padding_as_json(mesh.get("default_padding"))
            return n_cells
        if base is None:
            raise ValueError("reference mesh 'base_mesh' must be an object")
        shape = _validate_parsed_base_mesh(
            f, base, data_start, data_size, context="reference base_mesh"
        )
        resolve_shared_padding(
            mesh.get("default_padding"), base.get("default_padding"), shape
        )
        return n_cells
    if mode != "embedded":
        raise ValueError(f"unsupported mesh mode: {mode!r}")
    mesh_class = mesh.get("mesh_class")
    if not isinstance(mesh_class, str) or mesh_class not in _MESH_CLASSES:
        raise ValueError(f"unsupported embedded mesh_class: {mesh_class!r}")
    value = _parsed_geometry_arrays(
        f, mesh.get("arrays"), mesh_class, data_start, data_size, context=mesh_class
    )
    if mesh_class == "OctreeMesh":
        if "base_mesh" not in mesh:
            raise ValueError("OctreeMesh descriptor missing required key 'base_mesh'")
        shape = _validate_parsed_base_mesh(
            f, mesh["base_mesh"], data_start, data_size, context="OctreeMesh base_mesh"
        )
        resolve_shared_padding(
            mesh.get("default_padding"), mesh["base_mesh"].get("default_padding"), shape
        )
        return value
    if "base_mesh" in mesh:
        raise ValueError(f"{mesh_class} does not support base_mesh")
    shape = tuple(value)
    padding_as_json(mesh.get("default_padding"), shape)
    return math.prod(shape)


def validate_model_lengths(models: dict, n_cells: int | None) -> None:
    """Raise if any model array descriptor is not one value per cell."""
    if not isinstance(models, dict):
        raise ValueError("models must be an object")
    for name, entry in models.items():
        descriptor = entry.get("array") if isinstance(entry, dict) else None
        shape = descriptor.get("shape") if isinstance(descriptor, dict) else None
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError(f"model {name!r} must be a 1D array, got shape {shape!r}")
        if (
            not isinstance(shape[0], Integral)
            or isinstance(shape[0], (bool, np.bool_))
            or shape[0] < 0
        ):
            raise ValueError(f"model {name!r} must have a non-negative integer shape")
        if n_cells is not None and shape != [n_cells]:
            raise ValueError(
                f"model {name!r} has shape {list(shape)}, but the mesh has "
                f"{n_cells} cells; models are one value per cell"
            )


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
