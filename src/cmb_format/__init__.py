"""Python I/O for Cell Model Binary (CMB) meshes and models.

The API accepts dictionaries of NumPy arrays. Files can contain mesh
geometry and named per-cell models, or models for a separately stored mesh.
See ``docs/binary-format.md`` for the normative format specification.
"""

from importlib.metadata import version as _version

from cmb_format._codec import (
    DTYPE_TO_NUMPY,
    MAGIC,
    READABLE_FORMAT_VERSIONS,
    WRITTEN_FORMAT_VERSION,
    base_mesh_descriptor,
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

__version__ = _version("cmb-format")

__all__ = [
    "DTYPE_TO_NUMPY",
    "MAGIC",
    "READABLE_FORMAT_VERSIONS",
    "WRITTEN_FORMAT_VERSION",
    "__version__",
    "base_mesh_descriptor",
    "build_file_bytes",
    "is_cmb_file",
    "list_models",
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
    "write_file",
]
