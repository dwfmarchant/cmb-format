"""Codec behaviour: the error paths a reader has to get right.

The goldens pin what a *valid* file looks like. These pin what happens to an
invalid one -- a reader that accepts corruption silently is worse than one
that rejects a good file loudly.
"""

import io
import json
import struct

import numpy as np
import pytest

import cmb_format as cmb
from cases import CASES, build_bytes, build_header_and_buffer

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
    header, buffer = build_header_and_buffer(CASES["tensor_embedded"])
    header["format_version"] = 2
    blob = json.dumps(header).encode("utf-8")
    raw = cmb.MAGIC + bytes(buffer) + blob + struct.pack("<Q", len(blob)) + cmb.MAGIC
    with pytest.raises(ValueError, match="unsupported CMB format_version"):
        cmb.read_header(_reader(raw))


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
        cmb.array_dtype_name(np.array([1 + 2j]))


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
