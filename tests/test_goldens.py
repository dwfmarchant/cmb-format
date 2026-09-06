"""Conformance against committed reference files.

Each golden in ``goldens/`` is a complete ``.cmb`` file whose exact bytes are
the assertion, paired with a ``.header.json`` sidecar holding its parsed
header. Together they pin the wire: dtype tokens, header key names, byte
order, offsets and the magic.

Regenerate with ``python tests/generate_goldens.py`` after an intentional
format change, and commit the result.

Values here are spelled out literally rather than imported from the code
under test.
"""

import io
import json
import pathlib
import struct

import pytest

import cmb_format as cmb
from cases import CASES, build_bytes

GOLDENS = pathlib.Path(__file__).parent / "goldens"
CASE_NAMES = sorted(CASES)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_golden_bytes_are_unchanged(name):
    assert build_bytes(CASES[name]) == (GOLDENS / f"{name}.cmb").read_bytes()


@pytest.mark.parametrize("name", CASE_NAMES)
def test_sidecar_matches_the_golden_it_documents(name):
    # The sidecar exists so review can read a format change. It is only
    # trustworthy if it is actually derived from the bytes beside it.
    raw = (GOLDENS / f"{name}.cmb").read_bytes()
    header, _ = cmb.read_header(io.BytesIO(raw))
    assert header == json.loads((GOLDENS / f"{name}.header.json").read_text())


@pytest.mark.parametrize("name", CASE_NAMES)
def test_golden_arrays_read_back_and_verify(name):
    raw = (GOLDENS / f"{name}.cmb").read_bytes()
    with io.BytesIO(raw) as f:
        header, data_start = cmb.read_header(f)
        if "arrays" in header["mesh"]:
            cmb.read_arrays(f, data_start, header["mesh"]["arrays"])
        if "base_mesh" in header["mesh"]:
            cmb.read_arrays(f, data_start, header["mesh"]["base_mesh"]["arrays"])
        for model in header["models"].values():
            # read_array verifies the stored sha256 as a side effect
            cmb.read_array(f, data_start, model["array"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_writing_is_deterministic(name):
    # Byte-exact goldens are only meaningful if writing the same input twice
    # produces the same bytes -- dict ordering, JSON separators and float
    # formatting all have to be stable.
    assert build_bytes(CASES[name]) == build_bytes(CASES[name])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_magic_is_literally_CELLMODB_at_both_ends(name):
    raw = (GOLDENS / f"{name}.cmb").read_bytes()
    assert raw[:8] == b"CELLMODB"
    assert raw[-8:] == b"CELLMODB"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_trailer_is_little_endian_uint64_header_length(name):
    raw = (GOLDENS / f"{name}.cmb").read_bytes()
    (declared,) = struct.unpack("<Q", raw[-16:-8])
    header_start = len(raw) - 16 - declared
    assert json.loads(raw[header_start:-16].decode("utf-8"))["format_version"] == 1


@pytest.mark.parametrize("name", CASE_NAMES)
def test_array_offsets_are_relative_to_byte_eight(name):
    # The spec fixes the data section as starting immediately after the
    # leading magic. read_header returns that base; it must be 8, not
    # "however long MAGIC happens to be".
    raw = (GOLDENS / f"{name}.cmb").read_bytes()
    _, data_start = cmb.read_header(io.BytesIO(raw))
    assert data_start == 8


def test_dtype_tokens_are_exactly_the_documented_set():
    # The wire spells dtypes as these strings. Renaming one is invisible to a
    # round trip and breaks every other implementation -- so pin them here.
    assert set(cmb.DTYPE_TO_NUMPY) == {
        "float64",
        "float32",
        "int64",
        "int32",
        "int16",
        "int8",
    }
    header = json.loads((GOLDENS / "all_dtypes.header.json").read_text())
    assert {m["array"]["dtype"] for m in header["models"].values()} == {
        "float64",
        "float32",
        "int64",
        "int32",
        "int16",
        "int8",
    }


def test_header_key_names_are_stable():
    header = json.loads(
        (GOLDENS / "octree_base_padding_models.header.json").read_text()
    )
    assert set(header) == {"format_version", "mesh", "metadata", "models"}
    assert set(header["mesh"]) == {"arrays", "base_mesh", "mesh_class", "mode"}
    assert set(header["mesh"]["base_mesh"]) == {
        "arrays",
        "default_padding",
        "mesh_class",
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
    # Reference mode describes no geometry, so it has no mesh_class -- a
    # reader keys off `mode`, and this is easy to regress by making the
    # writer uniform across modes.
    header = json.loads((GOLDENS / "reference_models_only.header.json").read_text())
    assert header["mesh"]["mode"] == "reference"
    assert "mesh_class" not in header["mesh"]
    assert header["mesh"]["n_cells"] == 3


def test_every_case_has_both_a_golden_and_a_sidecar():
    # Guards the matrix itself: adding a case to cases.py without
    # regenerating leaves a case silently unpinned.
    for name in CASE_NAMES:
        assert (GOLDENS / f"{name}.cmb").is_file(), f"missing golden for {name}"
        assert (GOLDENS / f"{name}.header.json").is_file(), (
            f"missing sidecar for {name}"
        )
    stray = {p.stem for p in GOLDENS.glob("*.cmb")} - set(CASE_NAMES)
    assert not stray, f"goldens with no case in cases.py: {sorted(stray)}"
