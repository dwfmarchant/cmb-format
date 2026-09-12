"""Check the package exports against the expected public API."""

import cmb_format

EXPECTED_PUBLIC_API = [
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


def test_public_api_is_exactly_this():
    assert sorted(cmb_format.__all__) == EXPECTED_PUBLIC_API


def test_all_is_sorted():
    assert list(cmb_format.__all__) == sorted(cmb_format.__all__)


def test_every_exported_name_resolves():
    unresolved = [n for n in cmb_format.__all__ if not hasattr(cmb_format, n)]
    assert unresolved == []


def test_no_module_objects_leak_into_the_namespace():
    import types

    assert [
        n
        for n in cmb_format.__all__
        if isinstance(getattr(cmb_format, n), types.ModuleType)
    ] == []
