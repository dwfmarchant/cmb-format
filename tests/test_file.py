"""Test the complete-file reader against reference files and corrupt payloads."""

from pathlib import Path

import numpy as np
import pytest

import cmb_format as cmb
from cases import _REFERENCE_CELLS, CASES
from test_helpers import frame, named_padding, unpack_case

GOLDENS = Path(__file__).parent / "goldens" / "v1"


def assert_array_loaded(actual, expected):
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert not actual.flags.writeable


def assert_mesh_loaded(actual, expected):
    assert isinstance(actual, dict)
    for key in ("mode", "mesh_class"):
        if key in expected:
            assert actual[key] == expected[key]
    assert actual.get("default_padding") == expected.get("default_padding")
    if expected.get("mode") == "reference":
        assert actual["n_cells"] == expected.get("n_cells", _REFERENCE_CELLS)
    if "arrays" in expected:
        assert actual["arrays"].keys() == expected["arrays"].keys()
        for name, array in expected["arrays"].items():
            assert_array_loaded(actual["arrays"][name], array)
    if "base_mesh" in expected:
        assert_mesh_loaded(actual["base_mesh"], expected["base_mesh"])
    else:
        assert "base_mesh" not in actual


@pytest.mark.parametrize("case_name", sorted(CASES))
@pytest.mark.parametrize("path_type", [str, Path])
def test_read_file_loads_golden_and_can_be_written_back(tmp_path, case_name, path_type):
    source = GOLDENS / f"{case_name}.cmb"
    case = CASES[case_name]
    mesh, models, metadata = cmb.read_file(path_type(source))

    assert_mesh_loaded(mesh, case["mesh"])
    for name, expected_model in case["models"].items():
        assert isinstance(models[name], dict)
        assert models[name]["metadata"] == expected_model["metadata"]
        assert_array_loaded(models[name]["array"], expected_model["array"])
    assert list(models) == list(case["models"])
    assert metadata == case["metadata"]

    # Feed back only what read_file returned; sourcing any argument from the
    # fixture instead would hide a field the reader fails to recover.
    output = tmp_path / "rewritten.cmb"
    cmb.write_file(output, mesh, models, metadata)
    assert output.read_bytes() == cmb.build_file_bytes(mesh, models, metadata)


def test_read_file_handles_omitted_models_and_metadata(tmp_path):
    header, data = unpack_case("tensor_embedded")
    del header["models"]
    path = tmp_path / "no-models.cmb"
    path.write_bytes(frame(header, data))
    _, models, _ = cmb.read_file(path)
    assert models == {}

    header, data = unpack_case("tensor_with_models")
    del header["models"]["rho"]["metadata"]
    path.write_bytes(frame(header, data))
    _, models, _ = cmb.read_file(path)
    assert models["rho"]["metadata"] == {}


@pytest.mark.parametrize("location", ["mesh", "base_mesh", "model"])
def test_read_file_verifies_every_kind_of_array(tmp_path, location):
    header, data = unpack_case("octree_base_padding_models")
    descriptors = {
        "mesh": header["mesh"]["arrays"]["position"],
        "base_mesh": header["mesh"]["base_mesh"]["arrays"]["origin"],
        "model": header["models"]["rho"]["array"],
    }
    corrupted = bytearray(data)
    corrupted[descriptors[location]["offset"]] ^= 0xFF
    path = tmp_path / "corrupted.cmb"
    path.write_bytes(frame(header, corrupted))

    with pytest.raises(ValueError, match="checksum mismatch"):
        cmb.read_file(path)


def test_read_file_rejects_invalid_file(tmp_path):
    path = tmp_path / "invalid.cmb"
    path.write_bytes(b"not a CMB file")
    with pytest.raises(ValueError, match="too small"):
        cmb.read_file(path)


def test_read_file_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        cmb.read_file(tmp_path / "missing.cmb")


def test_read_file_preserves_unknown_descriptor_fields(tmp_path):
    header, data = unpack_case("reference_with_base_mesh")
    header["mesh"]["future_mesh_field"] = "kept"
    header["mesh"]["base_mesh"]["future_base_field"] = True
    header["models"]["rho"]["future_model_field"] = 42
    path = tmp_path / "extensions.cmb"
    path.write_bytes(frame(header, data))

    mesh, models, _ = cmb.read_file(path)
    assert mesh["future_mesh_field"] == "kept"
    assert mesh["base_mesh"]["future_base_field"] is True
    assert models["rho"]["future_model_field"] == 42

    header, data = unpack_case("reference_explicit_n_cells")
    header["mesh"]["arrays"] = {"future": "ignored"}
    path.write_bytes(frame(header, data))
    mesh, _, _ = cmb.read_file(path)
    assert mesh["arrays"] == {"future": "ignored"}


class _PayloadReadSpy:
    """Reject reads from the CMB data section except explicitly allowed ranges."""

    def __init__(self, raw, data_end, allowed):
        self._raw = raw
        self._data_end = data_end
        self._allowed = set(allowed)

    def read(self, size=-1):
        start = self._raw.tell()
        end = float("inf") if size < 0 else start + size
        overlaps = start < self._data_end and end > 8
        if overlaps and (start, size) not in self._allowed:
            raise AssertionError(
                f"unexpected CMB payload read at {start} for {size} bytes"
            )
        return self._raw.read(size)

    def __enter__(self):
        self._raw.__enter__()
        return self

    def __exit__(self, *args):
        return self._raw.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _header_and_data_end(path):
    import json
    import struct

    raw = path.read_bytes()
    header_length = struct.unpack("<Q", raw[-16:-8])[0]
    header_start = len(raw) - 16 - header_length
    return json.loads(raw[header_start:-16]), header_start


def _spy_payload_reads(monkeypatch, path, allowed):
    import builtins
    import os

    _header, data_end = _header_and_data_end(path)
    original_open = builtins.open
    target = os.fspath(path)

    def open_spy(file_name, mode="r", *args, **kwargs):
        raw = original_open(file_name, mode, *args, **kwargs)
        if os.fspath(file_name) == target and "rb" in mode:
            return _PayloadReadSpy(raw, data_end, allowed)
        return raw

    monkeypatch.setattr(builtins, "open", open_spy)


def _descriptor_ranges(header, names=()):
    ranges = []
    mesh = header["mesh"]
    if mesh.get("mode") == "embedded":
        ranges.extend(
            (8 + item["offset"], item["length"]) for item in mesh["arrays"].values()
        )
    if "base_mesh" in mesh:
        ranges.extend(
            (8 + item["offset"], item["length"])
            for item in mesh["base_mesh"]["arrays"].values()
        )
    ranges.extend(
        (
            8 + header["models"][name]["array"]["offset"],
            header["models"][name]["array"]["length"],
        )
        for name in names
    )
    return ranges


@pytest.mark.parametrize(
    "case_name",
    [
        "uniform_padding_models",
        "octree_base_padding_models",
        "reference_with_base_mesh",
    ],
)
def test_list_models_reads_no_array_payload(tmp_path, monkeypatch, case_name):
    path = tmp_path / f"{case_name}.cmb"
    path.write_bytes((GOLDENS / f"{case_name}.cmb").read_bytes())
    header, _ = _header_and_data_end(path)
    _spy_payload_reads(monkeypatch, path, [])

    listing = cmb.list_models(path)

    expected = {
        name: {
            "metadata": entry.get("metadata", {}),
            "dtype": entry["array"]["dtype"],
            "shape": entry["array"]["shape"],
        }
        for name, entry in header["models"].items()
    }
    assert listing == expected


@pytest.mark.parametrize(
    "case_name, expected_type, expected_cells, allowed_name",
    [
        ("uniform_padding_models", "UniformTensorMesh", 32, "shape"),
        ("octree_base_padding_models", "OctreeMesh", 15, None),
        ("reference_with_base_mesh", None, 3, None),
    ],
)
def test_read_contents_reads_only_top_level_uniform_shape(
    tmp_path, monkeypatch, case_name, expected_type, expected_cells, allowed_name
):
    path = tmp_path / f"{case_name}.cmb"
    path.write_bytes((GOLDENS / f"{case_name}.cmb").read_bytes())
    header, _ = _header_and_data_end(path)
    allowed = []
    if allowed_name is not None:
        descriptor = header["mesh"]["arrays"][allowed_name]
        allowed = [(8 + descriptor["offset"], descriptor["length"])]
    _spy_payload_reads(monkeypatch, path, allowed)

    contents = cmb.read_contents(path)

    assert contents["has_mesh"] is (expected_type is not None)
    assert contents["mesh_type"] == expected_type
    assert contents["n_cells"] == expected_cells
    assert contents["has_base_mesh"] is (case_name != "uniform_padding_models")
    assert contents["models"] == cmb.list_models(path)


def test_list_models_is_empty_for_mesh_only_file(tmp_path, monkeypatch):
    path = tmp_path / "mesh-only.cmb"
    path.write_bytes((GOLDENS / "tensor_embedded.cmb").read_bytes())
    _spy_payload_reads(monkeypatch, path, [])

    assert cmb.list_models(path) == {}


@pytest.mark.parametrize(
    "case_name, expected",
    [
        (
            "tensor_embedded",
            {
                "has_mesh": True,
                "mesh_type": "TensorMesh",
                "has_base_mesh": False,
                "n_cells": 24,
            },
        ),
        (
            "reference_models_only",
            {
                "has_mesh": False,
                "mesh_type": None,
                "has_base_mesh": False,
                "n_cells": 3,
            },
        ),
    ],
)
def test_read_contents_reports_tensor_and_bare_reference(tmp_path, case_name, expected):
    path = tmp_path / f"{case_name}.cmb"
    path.write_bytes((GOLDENS / f"{case_name}.cmb").read_bytes())

    contents = cmb.read_contents(path)

    assert {key: contents[key] for key in expected} == expected
    assert contents["models"] == cmb.list_models(path)


def test_inspectors_report_empty_tensor_and_reference_models(tmp_path):
    tensor_path = tmp_path / "tensor.cmb"
    tensor_path.write_bytes((GOLDENS / "tensor_embedded.cmb").read_bytes())
    assert cmb.list_models(tensor_path) == {}
    assert cmb.read_contents(tensor_path)["models"] == {}

    header, data = unpack_case("reference_explicit_n_cells")
    header["models"] = {}
    reference_path = tmp_path / "reference.cmb"
    reference_path.write_bytes(frame(header, data))

    assert cmb.list_models(reference_path) == {}
    contents = cmb.read_contents(reference_path)
    assert contents["has_mesh"] is False
    assert contents["mesh_type"] is None
    assert contents["has_base_mesh"] is False
    assert contents["n_cells"] == 3
    assert contents["models"] == {}


def test_inspectors_ignore_corrupt_model_payloads(tmp_path):
    path = tmp_path / "corrupt-model.cmb"
    path.write_bytes((GOLDENS / "uniform_padding_models.cmb").read_bytes())
    header, _ = _header_and_data_end(path)
    model = header["models"]["rho"]["array"]
    raw = bytearray(path.read_bytes())
    raw[8 + model["offset"]] ^= 0xFF
    path.write_bytes(bytes(raw))

    assert cmb.list_models(path)["rho"]["shape"] == [32]
    assert cmb.read_contents(path)["n_cells"] == 32


def test_read_contents_verifies_embedded_uniform_shape_payload(tmp_path):
    path = tmp_path / "corrupt-shape.cmb"
    path.write_bytes((GOLDENS / "uniform_embedded.cmb").read_bytes())
    header, _ = _header_and_data_end(path)
    shape = header["mesh"]["arrays"]["shape"]
    raw = bytearray(path.read_bytes())
    raw[8 + shape["offset"]] ^= 0xFF
    path.write_bytes(bytes(raw))

    assert cmb.list_models(path) == {}
    with pytest.raises(ValueError, match="checksum mismatch"):
        cmb.read_contents(path)


@pytest.mark.parametrize(
    "case_name", ["octree_base_padding_models", "reference_with_base_mesh"]
)
def test_inspectors_skip_nested_base_shape_payloads(tmp_path, case_name):
    path = tmp_path / f"corrupt-{case_name}.cmb"
    path.write_bytes((GOLDENS / f"{case_name}.cmb").read_bytes())
    header, _ = _header_and_data_end(path)
    shape = header["mesh"]["base_mesh"]["arrays"]["shape"]
    raw = bytearray(path.read_bytes())
    raw[8 + shape["offset"]] ^= 0xFF
    path.write_bytes(bytes(raw))

    assert cmb.list_models(path)
    assert cmb.read_contents(path)["has_base_mesh"] is True
    with pytest.raises(ValueError, match="checksum mismatch"):
        cmb.read_file(path)


@pytest.mark.parametrize(
    "case_name",
    [
        "uniform_padding_models",
        "octree_base_padding_models",
        "reference_models_only",
        "reference_with_base_mesh",
    ],
)
@pytest.mark.parametrize("selection", [[], ["rho"]])
def test_read_file_reads_only_geometry_and_selected_model_payloads(
    tmp_path, monkeypatch, case_name, selection
):
    path = tmp_path / f"{case_name}.cmb"
    path.write_bytes((GOLDENS / f"{case_name}.cmb").read_bytes())
    header, _ = _header_and_data_end(path)
    _spy_payload_reads(monkeypatch, path, _descriptor_ranges(header, selection))

    _mesh, models, _metadata = cmb.read_file(path, models=selection)

    assert list(models) == selection


def test_read_file_selection_preserves_stored_order_and_skips_unselected_bytes(
    tmp_path,
):
    source = GOLDENS / "tensor_with_models.cmb"
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    header, _ = _header_and_data_end(path)
    model = header["models"]["sus"]["array"]
    raw = bytearray(path.read_bytes())
    raw[8 + model["offset"]] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="checksum mismatch"):
        cmb.read_file(path)
    _mesh, models, _metadata = cmb.read_file(path, models=["rho", "rho"])
    assert list(models) == ["rho"]


def test_read_file_selection_rejects_string_and_unknown_names(tmp_path):
    path = tmp_path / "tensor_with_models.cmb"
    path.write_bytes((GOLDENS / path.name).read_bytes())

    with pytest.raises(TypeError, match="not a single string"):
        cmb.read_file(path, models="rho")
    with pytest.raises(ValueError, match="unknown model"):
        cmb.read_file(path, models=["missing"])

    _mesh, models, _metadata = cmb.read_file(path, models=["sus", "rho"])
    assert list(models) == ["rho", "sus"]


_MISSING = object()


@pytest.mark.parametrize(
    "case_name, outer, nested, expected_outer, expected_nested",
    [
        (
            "tensor_embedded",
            named_padding([1.0] * 6),
            _MISSING,
            named_padding([1] * 6),
            _MISSING,
        ),
        ("uniform_embedded", None, _MISSING, _MISSING, _MISSING),
        (
            "octree_embedded",
            named_padding([1.0] * 6),
            _MISSING,
            named_padding([1.0] * 6),
            _MISSING,
        ),
        ("octree_embedded", None, None, None, _MISSING),
        (
            "octree_embedded",
            named_padding([1.0] * 6),
            named_padding([1.0] * 6),
            named_padding([1.0] * 6),
            named_padding([1] * 6),
        ),
        (
            "reference_models_only",
            named_padding([1.0] * 6),
            _MISSING,
            named_padding([1] * 6),
            _MISSING,
        ),
        (
            "reference_with_base_mesh",
            named_padding([1.0] * 6),
            _MISSING,
            named_padding([1.0] * 6),
            _MISSING,
        ),
        ("reference_with_base_mesh", None, None, None, _MISSING),
        (
            "reference_with_base_mesh",
            named_padding([1.0] * 6),
            named_padding([1.0] * 6),
            named_padding([1.0] * 6),
            named_padding([1] * 6),
        ),
    ],
)
def test_read_file_normalizes_padding_by_descriptor_owner(
    tmp_path, case_name, outer, nested, expected_outer, expected_nested
):
    header, data = unpack_case(case_name)
    mesh = header["mesh"]
    if outer is _MISSING:
        mesh.pop("default_padding", None)
    else:
        mesh["default_padding"] = outer
    if "base_mesh" in mesh:
        base = mesh["base_mesh"]
        if nested is _MISSING:
            base.pop("default_padding", None)
        else:
            base["default_padding"] = nested

    path = tmp_path / f"{case_name}-padding.cmb"
    path.write_bytes(frame(header, data))
    loaded_mesh, _, _ = cmb.read_file(path, models=[])

    if expected_outer is _MISSING:
        assert "default_padding" not in loaded_mesh
    elif expected_outer is None:
        assert loaded_mesh["default_padding"] is None
    else:
        assert loaded_mesh["default_padding"] == expected_outer
    if expected_nested is _MISSING:
        assert "default_padding" not in loaded_mesh.get("base_mesh", {})
    else:
        actual = loaded_mesh["base_mesh"]["default_padding"]
        assert actual == expected_nested
        assert all(type(value) is int for value in actual.values())


@pytest.mark.parametrize(
    "version, case_name",
    [
        (1, "octree_embedded"),
        (1, "reference_with_base_mesh"),
        (2, "octree_embedded"),
        (2, "reference_with_base_mesh"),
    ],
)
def test_outer_shared_padding_is_unknown_across_versions(tmp_path, version, case_name):
    header, data = unpack_case(case_name)
    header["format_version"] = version
    outer = {"west": True}
    base_values = [1, 2, 1, 2, 1, 2]
    base_padding = base_values if version == 1 else named_padding(base_values)
    header["mesh"]["default_padding"] = outer
    header["mesh"]["base_mesh"]["default_padding"] = base_padding
    path = tmp_path / f"v{version}-{case_name}-outer-unknown.cmb"
    path.write_bytes(frame(header, data))

    expected_base = named_padding(base_values)
    for read_shape_payload in (True, False):
        with path.open("rb") as stream:
            parsed, _ = cmb.read_header(stream, read_shape_payload=read_shape_payload)
        assert parsed["mesh"]["default_padding"] == outer
        assert parsed["mesh"]["base_mesh"]["default_padding"] == expected_base

    loaded_mesh, _, _ = cmb.read_file(path, models=[])
    assert loaded_mesh["default_padding"] == outer
    assert loaded_mesh["base_mesh"]["default_padding"] == expected_base
    assert set(cmb.list_models(path)) == set(header["models"])
    contents = cmb.read_contents(path)
    assert contents["has_base_mesh"] is True
    assert set(contents["models"]) == set(header["models"])


def test_v1_outer_only_shared_padding_is_not_relocated(tmp_path):
    header, data = unpack_case("octree_embedded")
    header["format_version"] = 1
    header["mesh"]["default_padding"] = [1, 2, 1, 2, 1, 2]
    header["mesh"]["base_mesh"].pop("default_padding", None)
    path = tmp_path / "v1-outer-only.cmb"
    path.write_bytes(frame(header, data))

    with path.open("rb") as stream:
        parsed, _ = cmb.read_header(stream)
    assert parsed["mesh"]["default_padding"] == [1, 2, 1, 2, 1, 2]
    assert "default_padding" not in parsed["mesh"]["base_mesh"]
    loaded_mesh, _, _ = cmb.read_file(path, models=[])
    assert loaded_mesh["default_padding"] == [1, 2, 1, 2, 1, 2]
    assert "default_padding" not in loaded_mesh["base_mesh"]


def test_read_contents_validates_uniform_padding_after_shape_read(tmp_path):
    header, data = unpack_case("uniform_embedded")
    header["mesh"]["default_padding"] = named_padding([5, 0, 0, 0, 0, 0])
    path = tmp_path / "bad-padding.cmb"
    path.write_bytes(frame(header, data))

    with pytest.raises(ValueError, match="opposing sides"):
        cmb.read_contents(path)


def test_read_contents_uses_python_integer_product_for_large_uniform_shapes(tmp_path):
    mesh = {
        "mode": "embedded",
        "mesh_class": "UniformTensorMesh",
        "arrays": {
            "origin": np.zeros(3),
            "cell_size": np.ones(3),
            "shape": np.array([2**32, 2**32, 1], dtype=np.int64),
        },
    }
    path = tmp_path / "large-uniform.cmb"
    path.write_bytes(cmb.build_file_bytes(mesh))

    assert cmb.read_contents(path)["n_cells"] == 2**64
