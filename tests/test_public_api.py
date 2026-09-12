"""Check the package exports against the expected public API."""

import importlib
import importlib.metadata as metadata

import cmb_format

EXPECTED_PUBLIC_API = [
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


def test_version_falls_back_when_distribution_metadata_is_missing(monkeypatch):
    real_version = metadata.version

    def missing_version(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing_version)
    try:
        assert importlib.reload(cmb_format).__version__ == "0+unknown"
    finally:
        monkeypatch.setattr(metadata, "version", real_version)
        importlib.reload(cmb_format)
