"""Check committed files and the Python writer's exact serialization.

Each golden has a parsed JSON header sidecar. Tests compare v1 compatibility
and generated v2 bytes with committed files and check format constants against
literal values.

Regenerate v2 fixtures with ``python tests/generate_goldens.py`` only when
intentionally updating serialization, and review both binary and sidecar diffs.
The historical v1 fixtures are kept in their own directory. A serialization
change does not necessarily change the format.
"""

import io
import json
import pathlib
import struct

import numpy as np
import pytest

import cmb_format as cmb
from cases import CASES, build_bytes
from test_helpers import named_padding

GOLDENS = pathlib.Path(__file__).parent / "goldens"
V1_GOLDENS = GOLDENS / "v1"
V2_GOLDENS = GOLDENS / "v2"
CASE_NAMES = sorted(CASES)


@pytest.mark.parametrize("name", sorted(p.stem for p in V1_GOLDENS.glob("*.cmb")))
def test_v1_golden_files_sidecars_and_arrays_are_unchanged(name):
    expected = (V1_GOLDENS / f"{name}.cmb").read_bytes()

    with io.BytesIO(expected) as f:
        header, data_start = cmb.read_header(f)
        # Keep the stored v1 sidecar consistent with its binary fixture.
        header_start = _header_start(expected)
        stored = json.loads(expected[header_start:-16].decode("utf-8"))
        assert stored == json.loads((V1_GOLDENS / f"{name}.header.json").read_text())
        if "default_padding" in stored["mesh"]:
            assert header["mesh"]["default_padding"] == named_padding(
                stored["mesh"]["default_padding"]
            )
        if (
            "base_mesh" in stored["mesh"]
            and "default_padding" in stored["mesh"]["base_mesh"]
        ):
            assert header["mesh"]["base_mesh"]["default_padding"] == named_padding(
                stored["mesh"]["base_mesh"]["default_padding"]
            )
        if "arrays" in header["mesh"]:
            cmb.read_arrays(f, data_start, header["mesh"]["arrays"])
        if "base_mesh" in header["mesh"]:
            cmb.read_arrays(f, data_start, header["mesh"]["base_mesh"]["arrays"])
        for model in header["models"].values():
            # Reading also verifies the stored SHA-256 checksum.
            cmb.read_array(f, data_start, model["array"])
        n_cells = _cell_count(f, header, data_start)
        wrong = {
            model_name: entry["array"]["shape"][0]
            for model_name, entry in header["models"].items()
            if entry["array"]["shape"] != [n_cells]
        }
    assert wrong == {}, f"{name} has {n_cells} cells but models {wrong}"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_v2_serialization_preserves_data_and_uses_named_padding(name):
    v1_path = V1_GOLDENS / f"{name}.cmb"
    v2_raw = (V2_GOLDENS / f"{name}.cmb").read_bytes()
    assert build_bytes(CASES[name]) == v2_raw
    with io.BytesIO(v2_raw) as f:
        header, data_start = cmb.read_header(f)
    header_start = _header_start(v2_raw)
    stored = json.loads(v2_raw[header_start:-16].decode("utf-8"))
    assert stored == json.loads((V2_GOLDENS / f"{name}.header.json").read_text())
    assert header == stored
    assert header["format_version"] == 2
    if v1_path.is_file():
        v1_raw = v1_path.read_bytes()
        assert (
            v2_raw[data_start : _header_start(v2_raw)]
            == v1_raw[8 : _header_start(v1_raw)]
        )
    for descriptor in (header["mesh"], header["mesh"].get("base_mesh", {})):
        if "default_padding" in descriptor:
            assert set(descriptor["default_padding"]) == {
                "west",
                "east",
                "south",
                "north",
                "bottom",
                "top",
            }


def _header_start(raw):
    (header_length,) = struct.unpack("<Q", raw[-16:-8])
    return len(raw) - 16 - header_length


@pytest.mark.parametrize("version, directory", [(1, V1_GOLDENS), (2, V2_GOLDENS)])
def test_magic_is_literal_and_trailer_uses_byte_eight_offsets(version, directory):
    raw = (directory / "tensor_with_models.cmb").read_bytes()
    assert raw[:8] == b"CELLMODB"
    assert raw[-8:] == b"CELLMODB"
    (declared,) = struct.unpack("<Q", raw[-16:-8])
    header_start = len(raw) - 16 - declared
    assert (
        json.loads(raw[header_start:-16].decode("utf-8"))["format_version"] == version
    )
    # Use the format's literal offset independently of the MAGIC constant.
    _, data_start = cmb.read_header(io.BytesIO(raw))
    assert data_start == 8


def test_dtype_tokens_are_exactly_the_documented_set():
    # Literal tokens catch renames that a shared writer/reader table would hide.
    expected = {
        "float64",
        "float32",
        "int64",
        "int32",
        "int16",
        "int8",
    }
    assert set(cmb.DTYPE_TO_NUMPY) == expected
    header = json.loads((V2_GOLDENS / "all_dtypes.header.json").read_text())
    assert {m["array"]["dtype"] for m in header["models"].values()} == expected


def test_header_key_names_are_stable():
    header = json.loads(
        (V2_GOLDENS / "octree_base_padding_models.header.json").read_text()
    )
    assert set(header) == {"format_version", "mesh", "metadata", "models"}
    assert set(header["mesh"]) == {"arrays", "base_mesh", "mesh_class", "mode"}
    assert set(header["mesh"]["base_mesh"]) == {
        "arrays",
        "default_padding",
        "mesh_class",
    }
    assert set(header["mesh"]["base_mesh"]["default_padding"]) == {
        "west",
        "east",
        "south",
        "north",
        "bottom",
        "top",
    }
    assert set(header["mesh"]["arrays"]["level"]) == {
        "checksum",
        "dtype",
        "length",
        "offset",
        "shape",
    }
    assert header["mesh"]["arrays"]["level"]["checksum"].startswith("sha256:")


def test_reference_mode_carries_no_mesh_class():
    # The minimal reference descriptor identifies the mode and cell count.
    header = json.loads((V2_GOLDENS / "reference_models_only.header.json").read_text())
    assert header["mesh"]["mode"] == "reference"
    assert "mesh_class" not in header["mesh"]
    assert header["mesh"]["n_cells"] == 3


def test_every_case_has_current_v2_golden_and_sidecar():
    # Keep current fixture definitions and generated files in sync.
    expected = {
        filename
        for name in CASE_NAMES
        for filename in (f"{name}.cmb", f"{name}.header.json")
    }
    actual = {path.name for path in V2_GOLDENS.iterdir() if path.is_file()}
    assert actual == expected
    for path in V1_GOLDENS.glob("*.cmb"):
        assert (V1_GOLDENS / f"{path.stem}.header.json").is_file(), (
            f"missing sidecar for {path.stem}"
        )


def _cell_count(f, header, data_start):
    """Number of cells a golden's mesh descriptor describes."""
    mesh = header["mesh"]
    if mesh.get("mode") == "reference":
        return mesh["n_cells"]
    arrays = mesh["arrays"]
    if mesh["mesh_class"] == "TensorMesh":
        return (
            arrays["h_x"]["shape"][0]
            * arrays["h_y"]["shape"][0]
            * arrays["h_z"]["shape"][0]
        )
    if mesh["mesh_class"] == "UniformTensorMesh":
        return int(np.prod(cmb.read_array(f, data_start, arrays["shape"])))
    if mesh["mesh_class"] == "OctreeMesh":
        return arrays["level"]["shape"][0]
    raise AssertionError(f"unhandled mesh_class {mesh['mesh_class']!r}")
