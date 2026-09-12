"""Python I/O for Cell Model Binary (CMB) meshes and models.

The API accepts dictionaries of NumPy arrays. Files can contain mesh
geometry and named per-cell models, or models for a separately stored mesh.
See ``docs/binary-format.md`` for the normative format specification.
"""

from importlib import metadata as _metadata

from cmb_format._codec import (
    MAGIC,
    READABLE_FORMAT_VERSIONS,
    WRITTEN_FORMAT_VERSION,
    read_array,
    read_arrays,
    read_header,
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

try:
    __version__ = _metadata.version("cmb-format")
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "MAGIC",
    "READABLE_FORMAT_VERSIONS",
    "WRITTEN_FORMAT_VERSION",
    "__version__",
    "build_file_bytes",
    "is_cmb_file",
    "list_models",
    "read_array",
    "read_arrays",
    "read_contents",
    "read_file",
    "read_header",
    "summarize_models",
    "write_file",
]
