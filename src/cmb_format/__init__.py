"""Python I/O for Cell Model Binary (CMB) meshes and models.

The API accepts dictionaries of NumPy arrays. Files can contain mesh
geometry and named per-cell models, or models for a separately stored mesh.
See ``docs/binary-format.md`` for the normative format specification.
"""

from cmb_format._codec import (
    DTYPE_TO_NUMPY,
    INT8_MAX,
    MAGIC,
    READABLE_FORMAT_VERSIONS,
    WRITTEN_FORMAT_VERSION,
    base_mesh_descriptor,
    descriptor_shape,
    padding_as_json,
    padding_belongs_to_base_mesh,
    read_array,
    read_arrays,
    read_header,
    resolve_reference_n_cells,
    resolve_shared_padding,
    serialize_array,
    serialize_mesh,
    summarize_models,
)
from cmb_format._detect import is_cmb_file
from cmb_format._file import build_file_bytes, write_file
from cmb_format._padding import (
    normalize_default_padding,
    validate_default_padding_shape,
)

__all__ = [
    "DTYPE_TO_NUMPY",
    "INT8_MAX",
    "MAGIC",
    "READABLE_FORMAT_VERSIONS",
    "WRITTEN_FORMAT_VERSION",
    "base_mesh_descriptor",
    "build_file_bytes",
    "descriptor_shape",
    "is_cmb_file",
    "normalize_default_padding",
    "padding_as_json",
    "padding_belongs_to_base_mesh",
    "read_array",
    "read_arrays",
    "read_header",
    "resolve_reference_n_cells",
    "resolve_shared_padding",
    "serialize_array",
    "serialize_mesh",
    "summarize_models",
    "validate_default_padding_shape",
    "write_file",
]
