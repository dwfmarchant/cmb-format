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

from cmb_format._compat import normalize_header_padding
from cmb_format._padding import (
    _normalize_default_padding,
    _validate_normalized_padding_shape,
    normalize_integer_array,
)

__all__ = [
    "KIND_ITEMSIZE_TO_DTYPE_NAME",
    "MAGIC",
    "READABLE_FORMAT_VERSIONS",
    "WRITTEN_FORMAT_VERSION",
    "array_dtype_name",
    "read_array",
    "read_arrays",
    "read_header",
    "sha256_hex",
    "summarize_models",
    "validate_model_lengths",
]

MAGIC = b"CELLMODB"

# An integer counter of incompatible wire changes, unrelated to this
# package's own version. See docs/binary-format.md's Versioning section.
WRITTEN_FORMAT_VERSION = 2
READABLE_FORMAT_VERSIONS = frozenset({1, 2})

_DTYPE_TO_NUMPY = {
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


def _validate_raw_base_mesh(base: dict, *, context: str) -> dict[str, int] | None:
    if not isinstance(base, dict):
        raise ValueError(f"{context} must be a mapping")
    if base.get("mesh_class") != "UniformTensorMesh":
        raise ValueError(f"{context} mesh_class must be 'UniformTensorMesh'")
    shape = _validate_raw_geometry_arrays(
        "UniformTensorMesh", base.get("arrays"), context=f"{context}"
    )
    _validate_power_of_two_shape(shape, context=context)
    padding = _normalize_default_padding(base.get("default_padding"), shape)
    return padding


def _validate_raw_mesh(
    mesh: dict,
) -> tuple[int, dict[str, int] | None]:
    """Validate raw mesh input and return its count and owner padding."""
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
            padding = _validate_raw_base_mesh(
                mesh["base_mesh"], context="OctreeMesh base_mesh"
            )
            return int(shape_or_count), padding
        if "base_mesh" in mesh:
            raise ValueError(f"{mesh_class} does not support base_mesh")
        shape = tuple(shape_or_count)
        padding = _normalize_default_padding(mesh.get("default_padding"), shape)
        return math.prod(shape), padding
    if mode == "reference":
        n_cells = _nonnegative_integer(mesh.get("n_cells"), name="reference n_cells")
        if "base_mesh" not in mesh:
            padding = _normalize_default_padding(mesh.get("default_padding"))
            return n_cells, padding
        padding = _validate_raw_base_mesh(
            mesh["base_mesh"], context="reference base_mesh"
        )
        return n_cells, padding
    raise ValueError(f"unsupported mesh mode for serialization: {mode!r}")


def _serialize_array(arr: ArrayLike, buffer: bytearray) -> dict:
    """Append one supported array as little-endian bytes and return its descriptor."""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        raise ValueError(f"arrays must be at most 1D, got shape {arr.shape}")
    dtype_name = array_dtype_name(arr)
    numpy_dtype = np.dtype(_DTYPE_TO_NUMPY[dtype_name])
    raw = np.ascontiguousarray(arr, dtype=numpy_dtype).tobytes()
    offset = len(buffer)
    buffer.extend(raw)
    return {
        "dtype": dtype_name,
        "shape": list(arr.shape),
        "offset": offset,
        "length": len(raw),
        "checksum": sha256_hex(raw),
    }


def _serialize_arrays(arrays: dict, buffer: bytearray) -> dict:
    return {name: _serialize_array(arrays[name], buffer) for name in sorted(arrays)}


def _serialize_mesh(
    mesh_dict: dict, buffer: bytearray, *, padding: dict[str, int] | None
) -> dict:
    """Append validated mesh geometry and return its parsed descriptor."""
    mode = mesh_dict["mode"]
    if mode == "embedded":
        header = {
            "mode": "embedded",
            "mesh_class": mesh_dict["mesh_class"],
            "arrays": _serialize_arrays(mesh_dict["arrays"], buffer),
        }
    else:
        header = {"mode": "reference", "n_cells": mesh_dict["n_cells"]}

    base = mesh_dict.get("base_mesh")
    if padding is not None and not isinstance(base, dict):
        header["default_padding"] = padding
    if isinstance(base, dict):
        header["base_mesh"] = {
            "mesh_class": base["mesh_class"],
            "arrays": _serialize_arrays(base["arrays"], buffer),
        }
        if padding is not None:
            header["base_mesh"]["default_padding"] = padding
    return header


def _validate_array_descriptor(
    descriptor: dict, *, context: str, data_size: int | None = None
) -> tuple[str, list[int], int, int]:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{context} must be an object")
    missing = [name for name in _ARRAY_FIELDS if name not in descriptor]
    if missing:
        raise ValueError(f"{context} missing required field(s): {missing}")
    dtype = descriptor["dtype"]
    if not isinstance(dtype, str) or dtype not in _DTYPE_TO_NUMPY:
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
        _DTYPE_TO_NUMPY[dtype]
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
    numpy_dtype = np.dtype(_DTYPE_TO_NUMPY[dtype])
    return np.frombuffer(raw, dtype=numpy_dtype).reshape(tuple(shape))


def read_arrays(f, data_start: int, descriptors: dict) -> dict:
    """Read a mapping of named array descriptors from an open binary file.

    Returns a dictionary of NumPy arrays. Each read uses `read_array`, including
    its offset handling and checksum verification, and changes the file position.
    """
    return {name: read_array(f, data_start, d) for name, d in descriptors.items()}


def read_header(
    f,
    *,
    read_shape_payload: bool = True,
) -> tuple[dict, int]:
    """Read and structurally validate a CMB file's trailing JSON header.

    The returned header normalizes recognized v1 list padding and v2 object
    padding to complete named dictionaries and reports ``format_version`` 2.
    Unrecognized fields remain unchanged. The default
    ``read_shape_payload=True`` checks the file framing, header
    schema, array descriptor bounds, mesh descriptors, and that every model is one
    value per cell. It reads and checksum-verifies a three-element ``shape``
    array when a ``UniformTensorMesh`` needs its values for cell counts and
    padding checks (12 bytes for int32 or 24 bytes for int64), so the default
    call may raise an array checksum error. Other array payloads remain unread
    and unverified.

    With ``read_shape_payload=False``, structural header and descriptor checks
    still run without reading any shape payload. Shape values and validations
    dependent on them, such as uniform padding and model counts, are deferred
    to the caller; counts available directly from the header remain checked.
    Array payloads are not read or checksum-verified. This mode supports cheap
    header inspection.

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
    normalize_header_padding(header)
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
    n_cells = _validate_parsed_mesh(
        f,
        mesh,
        data_start,
        data_size,
        read_shape_payload=read_shape_payload,
    )
    if n_cells is not None:
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
    read_shape_payload: bool,
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
        if not read_shape_payload:
            return None
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
    f,
    base: dict,
    data_start: int,
    data_size: int | None,
    *,
    context: str,
    read_shape_payload: bool,
) -> tuple[int, int, int] | None:
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
        read_shape_payload=read_shape_payload,
    )
    if shape is not None:
        _validate_power_of_two_shape(shape, context=context)
    if shape is not None and base.get("default_padding") is not None:
        _validate_normalized_padding_shape(base["default_padding"], shape)
    return shape


def _validate_parsed_mesh(
    f,
    mesh: dict,
    data_start: int,
    data_size: int | None,
    *,
    read_shape_payload: bool,
) -> int | None:
    mode = mesh.get("mode")
    if mode == "reference":
        n_cells = _nonnegative_integer(mesh.get("n_cells"), name="reference n_cells")
        base = mesh.get("base_mesh")
        if "base_mesh" not in mesh:
            return n_cells
        if base is None:
            raise ValueError("reference mesh 'base_mesh' must be an object")
        _validate_parsed_base_mesh(
            f,
            base,
            data_start,
            data_size,
            context="reference base_mesh",
            read_shape_payload=read_shape_payload,
        )
        return n_cells
    if mode != "embedded":
        raise ValueError(f"unsupported mesh mode: {mode!r}")
    mesh_class = mesh.get("mesh_class")
    if not isinstance(mesh_class, str) or mesh_class not in _MESH_CLASSES:
        raise ValueError(f"unsupported embedded mesh_class: {mesh_class!r}")
    value = _parsed_geometry_arrays(
        f,
        mesh.get("arrays"),
        mesh_class,
        data_start,
        data_size,
        context=mesh_class,
        read_shape_payload=read_shape_payload,
    )
    if mesh_class == "OctreeMesh":
        if "base_mesh" not in mesh:
            raise ValueError("OctreeMesh descriptor missing required key 'base_mesh'")
        _validate_parsed_base_mesh(
            f,
            mesh["base_mesh"],
            data_start,
            data_size,
            context="OctreeMesh base_mesh",
            read_shape_payload=read_shape_payload,
        )
        return value
    if "base_mesh" in mesh:
        raise ValueError(f"{mesh_class} does not support base_mesh")
    shape = None if value is None else tuple(value)
    if shape is not None and mesh.get("default_padding") is not None:
        _validate_normalized_padding_shape(mesh["default_padding"], shape)
    return None if shape is None else math.prod(shape)


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
