"""Test the complete-file reader against reference files and corrupt payloads."""

from pathlib import Path

import numpy as np
import pytest

import cmb_format as cmb
from cases import _REFERENCE_CELLS, CASES
from test_helpers import frame, unpack_case

GOLDENS = Path(__file__).parent / "goldens"


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
    assert output.read_bytes() == source.read_bytes()


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
