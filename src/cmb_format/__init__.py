"""CMB (Cell Model Binary): a file format for cell-based meshes and models.

A CMB file holds one mesh's geometry plus any number of named models — arrays
of per-cell values — in a single file, with arbitrary metadata at both the
file and the model level. Arrays are stored raw and uncompressed, described by
a JSON header at the end of the file, so a reader can seek straight to an
array without parsing anything proportional to its size.

``docs/binary-format.md`` is the normative specification; this package is a
reference implementation of it, not its definition. The format is deliberately
implementable from that document alone, in any language, using nothing more
exotic than file I/O and a JSON parser.

This package works in *array dicts*: plain dicts of numpy arrays, keyed the
way the spec's descriptors are. It has no notion of a mesh or model object.
Converting a consumer's own classes to and from array dicts is that
consumer's job — which is what keeps this package free of any one project's
type system.
"""

from cmb_format._codec import (
    DTYPE_TO_NUMPY,
    INT8_MAX,
    KIND_ITEMSIZE_TO_DTYPE_NAME,
    MAGIC,
    array_dtype_name,
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
    serialize_arrays,
    serialize_mesh,
    sha256_hex,
    shape_from_mesh_arrays,
    summarize_models,
    to_le_bytes,
)
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
    "array_dtype_name",
    "base_mesh_descriptor",
    "descriptor_shape",
    "normalize_default_padding",
    "normalize_integer_array",
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
    "validate_default_padding_shape",
]
