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
    padding_as_json,
    padding_belongs_to_base_mesh,
    raw_mesh_shape,
    read_array,
    read_arrays,
    read_header,
    resolve_reference_n_cells,
    serialize_array,
    serialize_mesh,
    summarize_models,
)
from cmb_format._detect import is_cmb_file
from cmb_format._file import (
    build_file_bytes,
    list_models,
    read_contents,
    read_file,
    write_file,
)
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
    "is_cmb_file",
    "list_models",
    "normalize_default_padding",
    "padding_as_json",
    "padding_belongs_to_base_mesh",
    "raw_mesh_shape",
    "read_array",
    "read_arrays",
    "read_contents",
    "read_file",
    "read_header",
    "resolve_reference_n_cells",
    "serialize_array",
    "serialize_mesh",
    "summarize_models",
    "validate_default_padding_shape",
    "write_file",
]
