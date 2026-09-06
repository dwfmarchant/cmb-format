"""Test array encoding, malformed-input handling, and header compatibility."""

import io
import json
import math
import struct

import numpy as np
import pytest

import cmb_format as cmb
from cases import CASES, build_bytes
from cmb_format._codec import array_dtype_name, sha256_hex

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
    # Replace the header because the writer always stamps WRITTEN_FORMAT_VERSION.
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


def test_written_files_carry_the_written_format_version(tmp_path):
    # The writer supplies the format version without caller input.
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
        header, _ = cmb.read_header(f)
    descriptor = dict(header["models"]["rho"]["array"])
    short = b"CELLMODB" + b"\0" * (descriptor["offset"] + descriptor["length"] - 1)
    with (
        _reader(short) as f,
        pytest.raises(ValueError, match="shorter than declared"),
    ):
        cmb.read_array(f, 8, descriptor)


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
    # Optional fields must not prevent reading known geometry and model data.
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
    # Parsing preserves unknown fields while known fields remain usable.
    assert set(arrays) == {"level", "position"}
    assert parsed["mesh"]["mesh_class"] == "OctreeMesh"
    assert parsed["mesh"]["base_mesh"]["default_padding"] == [1, 1, 1, 1, 1, 1]
    assert parsed["a_field_from_the_future"] == {"anything": [1, 2, 3]}
    assert parsed["mesh"]["unknown_mesh_key"] == "ignored"
    assert set(cmb.summarize_models(parsed)) == set(header["models"])


def _forge_model_shape(case_name, shape):
    """A valid file with model bytes and declared shape replaced consistently."""
    raw = bytearray(build_bytes(CASES[case_name]))
    (length,) = struct.unpack("<Q", raw[-16:-8])
    header = json.loads(raw[len(raw) - 16 - length : -16])
    data = bytes(raw[8 : len(raw) - 16 - length])
    rebuilt = bytearray(
        data[: next(iter(header["models"].values()))["array"]["offset"]]
    )
    for index, entry in enumerate(header["models"].values()):
        descriptor = entry["array"]
        start = descriptor["offset"]
        stop = start + descriptor["length"]
        payload = data[start:stop]
        if index == 0:
            dtype = np.dtype(cmb.DTYPE_TO_NUMPY[descriptor["dtype"]])
            count = math.prod(shape)
            payload = np.arange(count, dtype=dtype).tobytes()
            descriptor["shape"] = shape
        descriptor["offset"] = len(rebuilt)
        descriptor["length"] = len(payload)
        descriptor["checksum"] = sha256_hex(payload)
        rebuilt.extend(payload)
    data = bytes(rebuilt)
    blob = json.dumps(header).encode("utf-8")
    return cmb.MAGIC + data + blob + struct.pack("<Q", len(blob)) + cmb.MAGIC


def test_write_rejects_a_model_that_is_not_one_value_per_cell(tmp_path):
    with pytest.raises(ValueError, match="one value per cell"):
        cmb.write_file(
            tmp_path / "bad.cmb",
            CASES["tensor_embedded"]["mesh"],
            {"rho": {"metadata": {}, "array": np.arange(3.0)}},
        )


@pytest.mark.parametrize(
    "case_name",
    ["tensor_with_models", "uniform_padding_models", "octree_base_padding_models"],
)
def test_read_rejects_a_model_that_is_not_one_value_per_cell(case_name):
    # UniformTensorMesh states cell counts as array values, so this also
    # covers the path that reads the shape array to derive them.
    with pytest.raises(ValueError, match="one value per cell"):
        cmb.read_header(_reader(_forge_model_shape(case_name, [7])))


def test_reference_mode_model_length_is_checked_too():
    with pytest.raises(ValueError, match="one value per cell"):
        cmb.read_header(_reader(_forge_model_shape("reference_models_only", [99])))


@pytest.mark.parametrize(
    "case_name",
    [
        "tensor_with_models",
        "uniform_padding_models",
        "octree_base_padding_models",
        "reference_models_only",
    ],
)
@pytest.mark.parametrize("ndim", [0, 2], ids=["scalar", "matrix"])
@pytest.mark.parametrize("writer", ["write_file", "build_file_bytes"])
def test_writers_reject_non_1d_models(tmp_path, case_name, ndim, writer):
    case = CASES[case_name]
    values = next(iter(case["models"].values()))["array"]
    invalid = np.array(1.0) if ndim == 0 else values.reshape(1, -1)
    models = {"rho": {"array": invalid}}
    with pytest.raises(ValueError, match="must be a 1D array"):
        if writer == "write_file":
            cmb.write_file(tmp_path / "bad.cmb", case["mesh"], models)
        else:
            cmb.build_file_bytes(case["mesh"], models)


@pytest.mark.parametrize(
    "case_name",
    [
        "tensor_with_models",
        "uniform_padding_models",
        "octree_base_padding_models",
        "reference_models_only",
    ],
)
@pytest.mark.parametrize("ndim", [0, 2], ids=["scalar", "matrix"])
def test_read_rejects_non_1d_models(case_name, ndim):
    values = next(iter(CASES[case_name]["models"].values()))["array"]
    # The matrix declares the correct total number of values but the wrong rank.
    shape = [] if ndim == 0 else [1, len(values)]
    with pytest.raises(
        ValueError, match=r"must (be a 1D array|contain zero or one dimension)"
    ):
        cmb.read_header(_reader(_forge_model_shape(case_name, shape)))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_uniform_shape_integer_widths_round_trip(dtype):
    original = CASES["uniform_padding_models"]
    mesh = {
        **original["mesh"],
        "arrays": {
            **original["mesh"]["arrays"],
            "shape": original["mesh"]["arrays"]["shape"].astype(dtype),
        },
    }
    with _reader(cmb.build_file_bytes(mesh, original["models"])) as f:
        header, start = cmb.read_header(f)
        shape = cmb.read_array(f, start, header["mesh"]["arrays"]["shape"])
        rho = cmb.read_array(f, start, header["models"]["rho"]["array"])
    assert shape.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(shape, mesh["arrays"]["shape"])
    np.testing.assert_array_equal(rho, original["models"]["rho"]["array"])
