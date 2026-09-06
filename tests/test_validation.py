"""Regression tests for the v1 writer and reader validation contract."""

import io
import json
import struct

import numpy as np
import pytest

import cmb_format as cmb
from cases import CASES
from cmb_format._codec import sha256_hex


def _unpack(case_name="tensor_with_models"):
    raw = _build(case_name)
    length = struct.unpack("<Q", raw[-16:-8])[0]
    return (
        json.loads(raw[-16 - length : -16]),
        raw[8 : -16 - length],
    )


def _build(case_name):
    case = CASES[case_name]
    return cmb.build_file_bytes(case["mesh"], case["models"], case["metadata"])


def _frame(header, data):
    blob = json.dumps(header).encode()
    return cmb.MAGIC + data + blob + struct.pack("<Q", len(blob)) + cmb.MAGIC


def _mutated(case_name, path, value):
    header, data = _unpack(case_name)
    target = header
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return _frame(header, data)


def _tensor():
    return {
        "mode": "embedded",
        "mesh_class": "TensorMesh",
        "arrays": {
            "origin": np.zeros(3),
            "h_x": np.ones(2),
            "h_y": np.ones(2),
            "h_z": np.ones(2),
        },
    }


def _uniform(shape=(4, 4, 2)):
    return {
        "mode": "embedded",
        "mesh_class": "UniformTensorMesh",
        "arrays": {
            "origin": np.zeros(3),
            "cell_size": np.ones(3),
            "shape": np.asarray(shape, dtype=np.int32),
        },
    }


def _base(shape=(4, 4, 4)):
    result = _uniform(shape)
    result.pop("mode")
    return result


def _octree(base_shape=(4, 4, 4)):
    return {
        "mode": "embedded",
        "mesh_class": "OctreeMesh",
        "arrays": {
            "level": np.array([0], dtype=np.int8),
            "position": np.array([0], dtype=np.int32),
        },
        "base_mesh": _base(base_shape),
    }


@pytest.mark.parametrize("writer", ["build", "write"])
@pytest.mark.parametrize(
    "mesh_class", ["TensorMesh", "UniformTensorMesh", "OctreeMesh"]
)
def test_writer_rejects_wrong_geometry_keys(tmp_path, writer, mesh_class):
    mesh = (
        _tensor()
        if mesh_class == "TensorMesh"
        else _uniform()
        if mesh_class == "UniformTensorMesh"
        else _octree()
    )
    mesh["arrays"]["extra"] = np.array([1.0])
    with pytest.raises(ValueError, match="required keys"):
        if writer == "build":
            cmb.build_file_bytes(mesh)
        else:
            cmb.write_file(tmp_path / "bad.cmb", mesh)


@pytest.mark.parametrize("writer", ["build", "write"])
@pytest.mark.parametrize(
    "mesh_class", ["TensorMesh", "UniformTensorMesh", "OctreeMesh"]
)
def test_writer_rejects_missing_geometry_keys(tmp_path, writer, mesh_class):
    mesh = (
        _tensor()
        if mesh_class == "TensorMesh"
        else _uniform()
        if mesh_class == "UniformTensorMesh"
        else _octree()
    )
    del mesh["arrays"][next(iter(mesh["arrays"]))]
    with pytest.raises(ValueError, match="required keys"):
        if writer == "build":
            cmb.build_file_bytes(mesh)
        else:
            cmb.write_file(tmp_path / "bad.cmb", mesh)


@pytest.mark.parametrize("value", ["Tensormesh", [], {}, None])
def test_writer_rejects_unknown_mesh_class(value):
    mesh = _tensor()
    mesh["mesh_class"] = value
    with pytest.raises(ValueError, match="mesh_class"):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize("value", ["other", [], {}, None])
def test_writer_rejects_unknown_mesh_mode(value):
    mesh = _tensor()
    mesh["mode"] = value
    with pytest.raises(ValueError, match="mode"):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize("field", ["origin", "cell_size"])
def test_writer_rejects_wrong_uniform_geometry_dtype(field):
    mesh = _uniform()
    mesh["arrays"][field] = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match=field):
        cmb.build_file_bytes(mesh)


def test_writer_rejects_wrong_tensor_geometry_shape():
    mesh = _tensor()
    mesh["arrays"]["origin"] = np.zeros((3, 1))
    with pytest.raises(ValueError, match="origin"):
        cmb.build_file_bytes(mesh)


def test_writer_rejects_wrong_octree_geometry_dtype_and_lengths():
    mesh = _octree()
    mesh["arrays"]["level"] = np.array([0], dtype=np.int16)
    with pytest.raises(ValueError, match="level"):
        cmb.build_file_bytes(mesh)
    mesh = _octree()
    mesh["arrays"]["position"] = np.array([0, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="equal lengths"):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize("shape", [(6, 4, 4), (3, 3, 3), (12, 4, 4)])
@pytest.mark.parametrize("writer", ["build", "write"])
def test_writer_rejects_non_power_of_two_octree_bases(tmp_path, shape, writer):
    meshes = [
        _octree(shape),
        {"mode": "reference", "n_cells": 0, "base_mesh": _base(shape)},
    ]
    for index, mesh in enumerate(meshes):
        with pytest.raises(ValueError, match="powers of two"):
            if writer == "build":
                cmb.build_file_bytes(mesh)
            else:
                cmb.write_file(tmp_path / f"bad-{index}.cmb", mesh)


def test_standalone_uniform_mesh_does_not_require_power_of_two_dimensions():
    mesh = _uniform((3, 3, 3))
    assert cmb.build_file_bytes(mesh)


@pytest.mark.parametrize("value", [True, False, 3.0, -1, "3", [], {}, None])
def test_reference_count_is_strict_on_writer(value):
    with pytest.raises(ValueError, match="n_cells"):
        cmb.build_file_bytes({"mode": "reference", "n_cells": value})


def test_numpy_reference_count_is_normalized_for_json():
    header = cmb.serialize_mesh(
        {"mode": "reference", "n_cells": np.int64(3)}, bytearray()
    )
    assert header["n_cells"] == 3
    assert type(header["n_cells"]) is int


def test_zero_reference_count_is_valid():
    raw = cmb.build_file_bytes({"mode": "reference", "n_cells": np.int64(0)}, {})
    with io.BytesIO(raw) as f:
        header, _ = cmb.read_header(f)
    assert header["mesh"]["n_cells"] == 0


def test_reference_count_can_be_inferred_from_models():
    raw = cmb.build_file_bytes(
        {"mode": "reference"}, {"rho": {"array": np.arange(3.0)}}
    )
    with io.BytesIO(raw) as f:
        header, _ = cmb.read_header(f)
    assert header["mesh"]["n_cells"] == 3


@pytest.mark.parametrize("value", [True, False, 3.0, -1, "3", [], {}, None])
def test_reference_count_is_strict_on_reader(value):
    header, data = _unpack("reference_explicit_n_cells")
    header["mesh"]["n_cells"] = value
    with pytest.raises(ValueError, match="n_cells"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


@pytest.mark.parametrize(
    "case_name", ["tensor_embedded", "uniform_embedded", "octree_embedded"]
)
def test_reader_rejects_extra_geometry_keys(case_name):
    with pytest.raises(ValueError, match="required keys"):
        cmb.read_header(
            io.BytesIO(_mutated(case_name, ["mesh", "arrays", "extra"], {}))
        )


@pytest.mark.parametrize(
    "case_name", ["tensor_embedded", "uniform_embedded", "octree_embedded"]
)
def test_reader_rejects_missing_geometry_keys(case_name):
    header, data = _unpack(case_name)
    del header["mesh"]["arrays"][next(iter(header["mesh"]["arrays"]))]
    with pytest.raises(ValueError, match="required keys"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


def test_reader_rejects_invalid_geometry_dtype_and_shape():
    header, data = _unpack("tensor_embedded")
    descriptor = header["mesh"]["arrays"]["origin"]
    start = descriptor["offset"]
    descriptor.update(
        dtype="float32", length=12, checksum=sha256_hex(data[start : start + 12])
    )
    with pytest.raises(ValueError, match="origin"):
        cmb.read_header(io.BytesIO(_frame(header, data)))
    header, data = _unpack("uniform_embedded")
    descriptor = header["mesh"]["arrays"]["origin"]
    start = descriptor["offset"]
    descriptor.update(
        shape=[2], length=16, checksum=sha256_hex(data[start : start + 16])
    )
    with pytest.raises(ValueError, match="origin"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


def test_reader_rejects_unequal_octree_geometry_lengths():
    header, data = _unpack("octree_embedded")
    descriptor = header["mesh"]["arrays"]["position"]
    start = descriptor["offset"]
    payload = data[start : start + 8]
    descriptor.update(shape=[2], length=8, checksum=sha256_hex(payload))
    with pytest.raises(ValueError, match="equal lengths"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


def test_reader_rejects_wrong_octree_geometry_dtype():
    header, data = _unpack("octree_embedded")
    descriptor = header["mesh"]["arrays"]["level"]
    start = descriptor["offset"]
    length = 2 * descriptor["shape"][0]
    descriptor.update(
        dtype="int16", length=length, checksum=sha256_hex(data[start : start + length])
    )
    with pytest.raises(ValueError, match="must have dtype"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


@pytest.mark.parametrize(
    "case_name, field",
    [
        ("tensor_embedded", "h_x"),
        ("tensor_embedded", "h_y"),
        ("tensor_embedded", "h_z"),
        ("octree_embedded", "level"),
        ("octree_embedded", "position"),
    ],
)
def test_reader_rejects_scalar_geometry_descriptors(case_name, field):
    header, data = _unpack(case_name)
    descriptor = header["mesh"]["arrays"][field]
    width = np.dtype(cmb.DTYPE_TO_NUMPY[descriptor["dtype"]]).itemsize
    start = descriptor["offset"]
    descriptor.update(
        shape=[],
        length=width,
        checksum=sha256_hex(data[start : start + width]),
    )
    with pytest.raises(ValueError, match="must be 1D"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


def test_reader_rejects_missing_or_invalid_octree_base():
    header, data = _unpack("octree_embedded")
    del header["mesh"]["base_mesh"]
    with pytest.raises(ValueError, match="base_mesh"):
        cmb.read_header(io.BytesIO(_frame(header, data)))
    header, data = _unpack("octree_embedded")
    header["mesh"]["base_mesh"]["mesh_class"] = "TensorMesh"
    with pytest.raises(ValueError, match="UniformTensorMesh"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


def test_reader_rejects_reference_count_and_inference_failures():
    header, data = _unpack("reference_explicit_n_cells")
    header["mesh"]["n_cells"] = True
    with pytest.raises(ValueError, match="n_cells"):
        cmb.read_header(io.BytesIO(_frame(header, data)))
    with pytest.raises(ValueError, match="n_cells"):
        cmb.resolve_reference_n_cells(None, {})
    with pytest.raises(ValueError, match="disagree"):
        cmb.resolve_reference_n_cells(
            None,
            {"a": {"array": np.zeros(2)}, "b": {"array": np.zeros(3)}},
        )


def test_reader_rejects_bad_padding_values_and_conflicts():
    header, data = _unpack("tensor_embedded")
    header["mesh"]["default_padding"] = [True, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="padding"):
        cmb.read_header(io.BytesIO(_frame(header, data)))
    header, data = _unpack("octree_embedded")
    header["mesh"]["default_padding"] = [1, 1, 1, 1, 1, 1]
    header["mesh"]["base_mesh"]["default_padding"] = [2, 2, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="match"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


@pytest.mark.parametrize("shape", [(6, 4, 4), (3, 3, 3), (12, 4, 4)])
def test_reader_rejects_non_power_of_two_embedded_and_reference_bases(shape):
    for case_name in ("octree_embedded", "reference_with_base_mesh"):
        header, data = _unpack(case_name)
        descriptor = header["mesh"]["base_mesh"]["arrays"]["shape"]
        payload = np.asarray(shape, dtype="<i4").tobytes()
        start = descriptor["offset"]
        data = data[:start] + payload + data[start + len(payload) :]
        descriptor.update(checksum=sha256_hex(payload))
        with pytest.raises(ValueError, match="powers of two"):
            cmb.read_header(io.BytesIO(_frame(header, data)))


def test_reader_rejects_non_object_header_and_models():
    with pytest.raises(ValueError, match="JSON object"):
        cmb.read_header(io.BytesIO(_frame([], b"")))
    with pytest.raises(ValueError, match="models"):
        cmb.read_header(io.BytesIO(_mutated("tensor_embedded", ["models"], [])))


@pytest.mark.parametrize(
    "field, value",
    [
        ("dtype", "float16"),
        ("shape", [True]),
        ("offset", -1),
        ("length", 1.0),
        ("checksum", "sha256:bad"),
    ],
)
def test_reader_rejects_malformed_descriptor(field, value):
    with pytest.raises(ValueError, match=r"descriptor|array"):
        cmb.read_header(
            io.BytesIO(
                _mutated("tensor_with_models", ["models", "rho", "array", field], value)
            )
        )


class _NoArrayAccess:
    def seek(self, *args):
        raise AssertionError("descriptor was not rejected before seek")

    def read(self, *args):
        raise AssertionError("descriptor was not rejected before read")


@pytest.mark.parametrize(
    "mutation",
    ["negative_offset", "missing_field", "dtype", "shape", "inflated_length"],
)
def test_read_array_rejects_malformed_descriptors_before_io(mutation):
    buffer = bytearray()
    descriptor = cmb.serialize_array(np.arange(4.0), buffer)
    if mutation == "negative_offset":
        descriptor["offset"] = -1
    elif mutation == "missing_field":
        del descriptor["checksum"]
    elif mutation == "dtype":
        descriptor["dtype"] = "float16"
    elif mutation == "shape":
        descriptor["shape"] = [2]
    else:
        descriptor["length"] *= 2
    with pytest.raises(ValueError):
        cmb.read_array(_NoArrayAccess(), 0, descriptor)


@pytest.mark.parametrize("field", ["dtype", "shape", "offset", "length", "checksum"])
def test_reader_rejects_missing_descriptor_fields(field):
    header, data = _unpack("tensor_with_models")
    del header["models"]["rho"]["array"][field]
    with pytest.raises(ValueError, match="missing required"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


def test_reader_rejects_negative_offset_even_with_matching_magic_checksum():
    header, data = _unpack("reference_models_only")
    header["mesh"]["n_cells"] = 1
    descriptor = header["models"]["rho"]["array"]
    descriptor.update(
        offset=-8,
        length=8,
        shape=[1],
        checksum=sha256_hex(cmb.MAGIC),
    )
    with pytest.raises(ValueError, match="offset"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


class _RejectDataReads(io.BytesIO):
    def __init__(self, raw, data_end):
        super().__init__(raw)
        self.data_end = data_end

    def read(self, size=-1):
        if 8 <= self.tell() < self.data_end:
            raise AssertionError("payload read occurred before descriptor bounds check")
        return super().read(size)


def test_reader_checks_model_bounds_before_reading_uniform_shape_payload():
    header, data = _unpack("uniform_embedded")
    header["models"] = {
        "rho": {
            "array": {
                "dtype": "float64",
                "shape": [1],
                "offset": len(data),
                "length": 8,
                "checksum": sha256_hex(b"\0" * 8),
            }
        }
    }
    raw = _frame(header, data)
    with pytest.raises(ValueError, match="data section"):
        cmb.read_header(_RejectDataReads(raw, 8 + len(data)))


def test_reader_checks_nested_base_descriptor_bounds():
    header, data = _unpack("octree_embedded")
    header["mesh"]["base_mesh"]["arrays"]["shape"]["offset"] = len(data)
    with pytest.raises(ValueError, match="data section"):
        cmb.read_header(io.BytesIO(_frame(header, data)))


def test_serialize_array_rejects_multidimensional_without_mutating_buffer():
    buffer = bytearray(b"prefix")
    with pytest.raises(ValueError, match="at most 1D"):
        cmb.serialize_array(np.ones((2, 2)), buffer)
    assert buffer == bytearray(b"prefix")


def test_scalar_standalone_array_round_trips():
    buffer = bytearray()
    descriptor = cmb.serialize_array(np.array(2.0), buffer)
    with io.BytesIO(bytes(buffer)) as f:
        result = cmb.read_array(f, 0, descriptor)
    assert result.shape == ()
    assert result.flags.writeable is False


def test_padding_conflict_is_rejected():
    mesh = _octree()
    mesh["default_padding"] = [1, 1, 1, 1, 1, 1]
    mesh["base_mesh"]["default_padding"] = [2, 2, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="must match"):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize(
    "padding",
    [
        [True, 0, 0, 0, 0, 0],
        [1, 2, 3],
        [np.inf] * 6,
        [np.nan] * 6,
        [0.5] * 6,
        [-1] * 6,
        [2**63] * 6,
        ["0"] * 6,
    ],
)
def test_writer_rejects_bad_padding_values(padding):
    mesh = _tensor()
    mesh["default_padding"] = padding
    with pytest.raises(ValueError, match="padding"):
        cmb.build_file_bytes(mesh)


def test_raw_mesh_shape_and_compatibility_alias_accept_raw_partial_inputs():
    raw = {"mesh_class": "UniformTensorMesh", "arrays": {"shape": np.array([3, 2, 4])}}
    assert cmb.raw_mesh_shape(raw) == (3, 2, 4)
    assert cmb.descriptor_shape(raw) == (3, 2, 4)


@pytest.mark.parametrize(
    "path, value",
    [
        (["format_version"], True),
        (["format_version"], 1.0),
        (["format_version"], []),
        (["mesh"], None),
        (["mesh", "mode"], "unknown"),
        (["mesh", "mode"], {}),
        (["mesh", "mesh_class"], "Tensormesh"),
        (["mesh", "mesh_class"], []),
        (["mesh", "arrays"], []),
        (["models"], None),
        (["models"], False),
        (["models", "rho"], None),
        (["models", "rho"], {}),
        (["models", "rho", "array"], []),
        (["models", "rho", "array", "dtype"], {}),
        (["models", "rho", "array", "shape"], [-1]),
        (["models", "rho", "array", "shape"], [24.0]),
        (["models", "rho", "array", "offset"], True),
        (["models", "rho", "array", "offset"], "0"),
        (["models", "rho", "array", "length"], -1),
    ],
)
def test_malformed_schema_values_raise_format_errors(path, value):
    with pytest.raises(ValueError):
        cmb.read_header(io.BytesIO(_mutated("tensor_with_models", path, value)))


def test_null_header_raises_format_error():
    with pytest.raises(ValueError, match="JSON object"):
        cmb.read_header(io.BytesIO(_frame(None, b"")))


@pytest.mark.parametrize("writer", ["build", "write"])
def test_uniform_cell_count_does_not_overflow_int64_on_write(tmp_path, writer):
    mesh = _uniform()
    mesh["arrays"]["shape"] = np.array([2**32, 2**32, 1], dtype=np.int64)
    # An int64 product wraps this count to zero. No model data is allocated.
    models = {"rho": {"array": np.empty(0)}}
    with pytest.raises(ValueError, match="18446744073709551616 cells"):
        if writer == "build":
            cmb.build_file_bytes(mesh, models)
        else:
            cmb.write_file(tmp_path / "overflow.cmb", mesh, models)


def test_uniform_cell_count_does_not_overflow_int64_on_read():
    header, data = _unpack("uniform_embedded")
    payload = np.array([2**32, 2**32, 1], dtype="<i8").tobytes()
    header["mesh"]["arrays"]["shape"].update(
        dtype="int64", offset=len(data), length=24, checksum=sha256_hex(payload)
    )
    header["models"] = {
        "rho": {
            "array": {
                "dtype": "float64",
                "shape": [0],
                "offset": 0,
                "length": 0,
                "checksum": sha256_hex(b""),
            }
        }
    }
    with pytest.raises(ValueError, match="18446744073709551616 cells"):
        cmb.read_header(io.BytesIO(_frame(header, data + payload)))


@pytest.mark.parametrize("entry", [None, {}, {"metadata": {}}])
def test_public_reference_count_helper_rejects_malformed_models(entry):
    with pytest.raises(ValueError, match="array"):
        cmb.resolve_reference_n_cells(None, {"rho": entry})


@pytest.mark.parametrize("value", [True, 3.0, "3", -1])
def test_supplied_reference_count_type_is_checked_even_with_matching_models(value):
    with pytest.raises(ValueError, match="n_cells"):
        cmb.build_file_bytes(
            {"mode": "reference", "n_cells": value},
            {"rho": {"array": np.ones(1 if value is True else 3)}},
        )


def test_writer_rejects_missing_octree_base_before_touching_output(tmp_path):
    mesh = _octree()
    del mesh["base_mesh"]
    path = tmp_path / "existing.cmb"
    path.write_bytes(b"existing contents")
    with pytest.raises(ValueError, match="base_mesh"):
        cmb.write_file(path, mesh)
    assert path.read_bytes() == b"existing contents"


@pytest.mark.parametrize("dtype", sorted(cmb.DTYPE_TO_NUMPY))
def test_big_endian_arrays_are_stored_as_little_endian(dtype):
    values = np.arange(3, dtype=np.dtype(cmb.DTYPE_TO_NUMPY[dtype]).newbyteorder(">"))
    buffer = bytearray()
    descriptor = cmb.serialize_array(values, buffer)
    assert (
        bytes(buffer)
        == np.arange(3, dtype=np.dtype(cmb.DTYPE_TO_NUMPY[dtype])).tobytes()
    )
    np.testing.assert_array_equal(
        cmb.read_array(io.BytesIO(buffer), 0, descriptor), values
    )


def test_octree_position_and_base_shape_support_int64():
    mesh = _octree()
    mesh["arrays"]["position"] = mesh["arrays"]["position"].astype(np.int64)
    mesh["base_mesh"]["arrays"]["shape"] = np.array([4, 4, 4], dtype=np.int64)
    raw = cmb.build_file_bytes(mesh)
    with io.BytesIO(raw) as f:
        header, start = cmb.read_header(f)
        position = cmb.read_array(f, start, header["mesh"]["arrays"]["position"])
        shape = cmb.read_array(f, start, header["mesh"]["base_mesh"]["arrays"]["shape"])
    assert position.dtype.itemsize == shape.dtype.itemsize == 8
    np.testing.assert_array_equal(position, mesh["arrays"]["position"])
    np.testing.assert_array_equal(shape, [4, 4, 4])


def test_padding_shape_check_requires_three_axes():
    padding = cmb.normalize_default_padding([0] * 6)
    with pytest.raises(ValueError, match="mesh shape must have shape"):
        cmb.validate_default_padding_shape(padding, [2, 2])
