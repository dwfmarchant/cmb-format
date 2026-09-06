"""Codec behaviour on malformed input.

Bad magic, truncation, a header length past end of file, an unreadable
format version, checksum mismatch, and dtypes the format cannot express.
"""

import io
import json
import struct

import numpy as np
import pytest

import cmb_format as cmb
from cases import CASES, build_bytes
from cmb_format._codec import array_dtype_name

VALID = build_bytes(CASES["tensor_with_models"])


def _reader(raw):
    return io.BytesIO(raw)


def test_rejects_a_file_whose_trailing_magic_is_wrong():
    raw = bytearray(VALID)
    raw[-8:] = b"NOTMAGIC"
    with pytest.raises(ValueError, match="trailing magic"):
        cmb.read_header(_reader(bytes(raw)))


def test_rejects_a_file_too_small_to_hold_a_trailer():
    with pytest.raises(ValueError, match="too small"):
        cmb.read_header(_reader(b"CELLMODB"))


def test_rejects_a_header_length_larger_than_the_file():
    raw = bytearray(VALID)
    raw[-16:-8] = struct.pack("<Q", len(VALID) * 10)
    with pytest.raises(ValueError, match="header length exceeds file size"):
        cmb.read_header(_reader(bytes(raw)))


def test_rejects_an_unsupported_format_version():
    # Hand-assembled: the package stamps WRITTEN_FORMAT_VERSION by design,
    # so a file claiming a version it does not write has to be forged.
    raw = bytearray(build_bytes(CASES["tensor_embedded"]))
    (length,) = struct.unpack("<Q", raw[-16:-8])
    header = json.loads(raw[len(raw) - 16 - length : -16])
    header["format_version"] = 999
    blob = json.dumps(header).encode("utf-8")
    forged = (
        cmb.MAGIC
        + bytes(raw[8 : len(raw) - 16 - length])
        + blob
        + struct.pack("<Q", len(blob))
        + cmb.MAGIC
    )
    with pytest.raises(ValueError, match="unsupported CMB format_version"):
        cmb.read_header(_reader(forged))


def test_version_constants_are_self_consistent():
    # Whatever this build writes, it must also be able to read.
    assert cmb.WRITTEN_FORMAT_VERSION in cmb.READABLE_FORMAT_VERSIONS


def test_written_files_carry_the_packages_version(tmp_path):
    # The point of write_file owning the stamp: a consumer cannot forget it
    # or drift from it, because it never supplies it.
    path = tmp_path / "stamped.cmb"
    cmb.write_file(path, mesh=CASES["tensor_embedded"]["mesh"])
    with open(path, "rb") as f:
        header, _ = cmb.read_header(f)
    assert header["format_version"] == cmb.WRITTEN_FORMAT_VERSION


def test_detects_a_corrupted_array_via_its_checksum():
    raw = bytearray(VALID)
    with _reader(bytes(raw)) as f:
        header, data_start = cmb.read_header(f)
    descriptor = header["models"]["rho"]["array"]
    flip = data_start + descriptor["offset"]
    raw[flip] ^= 0xFF
    with _reader(bytes(raw)) as f:
        _, data_start = cmb.read_header(f)
        with pytest.raises(ValueError, match="checksum mismatch"):
            cmb.read_array(f, data_start, descriptor)


def test_detects_array_data_shorter_than_declared():
    with _reader(VALID) as f:
        header, data_start = cmb.read_header(f)
    descriptor = dict(header["models"]["rho"]["array"])
    descriptor["length"] = descriptor["length"] * 100
    with (
        _reader(VALID) as f,
        pytest.raises(ValueError, match="shorter than declared"),
    ):
        cmb.read_array(f, data_start, descriptor)


def test_rejects_an_array_dtype_the_format_cannot_express():
    with pytest.raises(TypeError, match="unsupported array dtype"):
        array_dtype_name(np.array([1 + 2j]))


@pytest.mark.parametrize("dtype_name", sorted(cmb.DTYPE_TO_NUMPY))
def test_every_declared_dtype_round_trips(dtype_name):
    values = np.arange(4, dtype=np.dtype(cmb.DTYPE_TO_NUMPY[dtype_name]))
    buffer = bytearray()
    descriptor = cmb.serialize_array(values, buffer)
    assert descriptor["dtype"] == dtype_name
    raw = cmb.MAGIC + bytes(buffer)
    with _reader(raw) as f:
        back = cmb.read_array(f, 8, descriptor)
    np.testing.assert_array_equal(back, values)


def test_padding_must_fit_the_mesh_it_describes():
    with pytest.raises(ValueError, match="exceed the mesh shape"):
        cmb.validate_default_padding_shape(
            cmb.normalize_default_padding([5, 5, 0, 0, 0, 0]), (3, 3, 3)
        )


def test_unknown_header_keys_are_ignored_at_every_level(tmp_path):
    # The spec's forward-compatibility rule, exercised. If this ever fails,
    # every future optional field becomes a breaking change and the format
    # can only be replaced, never grown.
    raw = bytearray(build_bytes(CASES["octree_base_padding_models"]))
    (length,) = struct.unpack("<Q", raw[-16:-8])
    header = json.loads(raw[len(raw) - 16 - length : -16])

    header["a_field_from_the_future"] = {"anything": [1, 2, 3]}
    header["mesh"]["unknown_mesh_key"] = "ignored"
    header["mesh"]["base_mesh"]["unknown_base_key"] = 42
    header["mesh"]["arrays"]["level"]["unknown_array_key"] = True
    for model in header["models"].values():
        model["unknown_model_key"] = None
        model["array"]["another_unknown"] = "x"

    blob = json.dumps(header).encode("utf-8")
    forged = (
        cmb.MAGIC
        + bytes(raw[8 : len(raw) - 16 - length])
        + blob
        + struct.pack("<Q", len(blob))
        + cmb.MAGIC
    )
    path = tmp_path / "from_the_future.cmb"
    path.write_bytes(forged)

    assert cmb.is_cmb_file(path) is True
    with open(path, "rb") as f:
        parsed, data_start = cmb.read_header(f)
        arrays = cmb.read_arrays(f, data_start, parsed["mesh"]["arrays"])
        cmb.read_arrays(f, data_start, parsed["mesh"]["base_mesh"]["arrays"])
        for model in parsed["models"].values():
            cmb.read_array(f, data_start, model["array"])
    # Known fields still parse, and the unknown ones rode along untouched
    # rather than being rejected or silently dropped.
    assert set(arrays) == {"level", "position"}
    assert parsed["mesh"]["mesh_class"] == "OctreeMesh"
    assert parsed["mesh"]["base_mesh"]["default_padding"] == [1, 1, 1, 1, 1, 1]
    assert parsed["a_field_from_the_future"] == {"anything": [1, 2, 3]}
    assert parsed["mesh"]["unknown_mesh_key"] == "ignored"
    assert set(cmb.summarize_models(parsed)) == set(header["models"])
