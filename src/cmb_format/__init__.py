"""CMB (Cell Model Binary): a file format for cell-based meshes and models.

A CMB file holds one mesh's geometry plus any number of named models — arrays
of per-cell values — with arbitrary metadata at both the file and the model
level. Arrays are stored raw and uncompressed; a JSON header at the end of
the file gives each one's dtype, shape, byte offset and checksum, so a reader
can seek straight to a single array without touching the rest.

``docs/binary-format.md`` is the normative specification.

Everything here works in *array dicts*: plain dicts of numpy arrays, keyed
the way the specification's descriptors are. There is no mesh or model
object. A consumer converts its own classes to and from array dicts.
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
