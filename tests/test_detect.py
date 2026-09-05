"""is_cmb_file: identification by magic, and what it deliberately does not do."""

import pathlib

import pytest

import cmb_format as cmb
from cases import CASES, build_bytes

GOLDENS = pathlib.Path(__file__).parent / "goldens"


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_golden_identifies_as_cmb(tmp_path, name):
    # Covers embedded and reference mode, every mesh class, padding and
    # models -- the whole matrix, not just the common case.
    path = tmp_path / "sample"  # no extension at all
    path.write_bytes(build_bytes(CASES[name]))

    assert cmb.is_cmb_file(path) is True


def test_identifies_a_golden_read_straight_from_disk():
    assert cmb.is_cmb_file(GOLDENS / "tensor_embedded.cmb") is True


def test_true_for_matching_magic_with_malformed_remainder(tmp_path):
    # Identification, not validation. This file can never be read, and
    # still identifies as CMB -- that is the contract.
    path = tmp_path / "truncated.cmb"
    path.write_bytes(cmb.MAGIC + b"\x00\x01 not a real header")

    assert cmb.is_cmb_file(path) is True
    with pytest.raises(ValueError), open(path, "rb") as f:
        cmb.read_header(f)


def test_extension_is_irrelevant(tmp_path):
    cmb_named_txt = tmp_path / "actually_cmb.txt"
    cmb_named_txt.write_bytes(build_bytes(CASES["tensor_embedded"]))
    other_named_cmb = tmp_path / "actually_text.cmb"
    other_named_cmb.write_bytes(b"2 2 2\n0 0 10\n1 1\n1 1\n1 1\n")

    assert cmb.is_cmb_file(cmb_named_txt) is True
    assert cmb.is_cmb_file(other_named_cmb) is False


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(cmb.MAGIC[:-1], id="one_byte_short_of_the_magic"),
        pytest.param(cmb.MAGIC[::-1], id="magic_reversed"),
        pytest.param(b"\x89PNG\r\n\x1a\n\x00\x00", id="arbitrary_binary"),
        pytest.param(b"MESHMODB and then some", id="a_superseded_magic"),
    ],
)
def test_false_for_non_cmb_content(tmp_path, content):
    path = tmp_path / "candidate.cmb"
    path.write_bytes(content)

    assert cmb.is_cmb_file(path) is False


def test_filesystem_errors_propagate(tmp_path):
    with pytest.raises(FileNotFoundError):
        cmb.is_cmb_file(tmp_path / "nope.cmb")


def test_accepts_str_and_pathlike(tmp_path):
    path = tmp_path / "mesh.cmb"
    path.write_bytes(build_bytes(CASES["tensor_embedded"]))

    assert cmb.is_cmb_file(path) is True
    assert cmb.is_cmb_file(str(path)) is True
