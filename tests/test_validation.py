"""Regression tests for the v1 writer and reader validation contract."""

import io

import numpy as np
import pytest

import cmb_format as cmb
from cmb_format._codec import sha256_hex
from test_helpers import frame, fresh_mesh, mutate, named_padding, unpack_case

MESH_CASES = {
    "TensorMesh": "tensor_embedded",
    "UniformTensorMesh": "uniform_embedded",
    "OctreeMesh": "octree_embedded",
}


@pytest.mark.parametrize(
    "mesh_class, mutation",
    [
        pytest.param(mesh_class, mutation, id=f"{mesh_class}-{mutation}")
        for mesh_class in ("TensorMesh", "UniformTensorMesh", "OctreeMesh")
        for mutation in ("extra", "missing")
    ],
)
def test_writer_rejects_wrong_geometry_keys(mesh_class, mutation):
    mesh = fresh_mesh(MESH_CASES[mesh_class])
    if mutation == "extra":
        mesh["arrays"]["extra"] = np.array([1.0])
    else:
        del mesh["arrays"][next(iter(mesh["arrays"]))]
    with pytest.raises(ValueError, match="required keys"):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("mesh_class", value, id=f"mesh_class-{value!r}")
        for value in ("Tensormesh", [], {}, None)
    ]
    + [
        pytest.param("mode", value, id=f"mode-{value!r}")
        for value in ("other", [], {}, None)
    ],
)
def test_writer_rejects_unknown_mesh_values(field, value):
    mesh = fresh_mesh("tensor_embedded")
    mesh[field] = value
    with pytest.raises(ValueError, match=field):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize(
    "case_name, field, value, match",
    [
        ("uniform_embedded", "origin", np.zeros(3, dtype=np.float32), "origin"),
        ("uniform_embedded", "cell_size", np.zeros(3, dtype=np.float32), "cell_size"),
        (
            "uniform_embedded",
            "shape",
            np.array([0, 4, 2], dtype=np.int32),
            "positive integer values",
        ),
        ("tensor_embedded", "origin", np.zeros((3, 1)), "origin"),
        ("octree_embedded", "level", np.array([0], dtype=np.int16), "level"),
        (
            "octree_embedded",
            "position",
            np.array([0, 1], dtype=np.int32),
            "equal lengths",
        ),
    ],
)
def test_writer_rejects_bad_geometry(case_name, field, value, match):
    mesh = fresh_mesh(case_name)
    mesh["arrays"][field] = value
    with pytest.raises(ValueError, match=match):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize("shape", [(6, 4, 4), (3, 3, 3), (12, 4, 4)])
def test_writer_rejects_non_power_of_two_octree_bases(shape):
    octree = fresh_mesh("octree_embedded")
    octree["base_mesh"]["arrays"]["shape"] = np.asarray(shape, dtype=np.int32)
    reference = fresh_mesh("reference_with_base_mesh")
    reference["n_cells"] = 0
    reference["base_mesh"]["arrays"]["shape"] = np.asarray(shape, dtype=np.int32)
    meshes = [
        octree,
        reference,
    ]
    for mesh in meshes:
        with pytest.raises(ValueError, match="powers of two"):
            cmb.build_file_bytes(mesh)


@pytest.mark.parametrize("case_name", ["tensor_embedded", "uniform_embedded"])
def test_writer_rejects_non_octree_base_mesh(case_name):
    mesh = fresh_mesh(case_name)
    mesh["base_mesh"] = {}
    with pytest.raises(ValueError, match="does not support base_mesh"):
        cmb.build_file_bytes(mesh)


def test_standalone_uniform_mesh_does_not_require_power_of_two_dimensions():
    mesh = fresh_mesh("uniform_embedded")
    mesh["arrays"]["shape"] = np.array([3, 3, 3], dtype=np.int32)
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


def test_reference_count_must_match_model_length():
    with pytest.raises(
        ValueError, match=r"mesh has n_cells=3, but models have 1 cells"
    ):
        cmb.build_file_bytes(
            {"mode": "reference", "n_cells": 3},
            {"rho": {"array": np.ones(1)}},
        )


@pytest.mark.parametrize("value", [True, False, 3.0, -1, "3", [], {}, None])
def test_reference_count_is_strict_on_reader(value):
    header, data = unpack_case("reference_explicit_n_cells")
    header["mesh"]["n_cells"] = value
    with pytest.raises(ValueError, match="n_cells"):
        cmb.read_header(io.BytesIO(frame(header, data)))


@pytest.mark.parametrize(
    "case_name, mutation",
    [
        pytest.param(case_name, mutation, id=f"{case_name}-{mutation}")
        for case_name in ("tensor_embedded", "uniform_embedded", "octree_embedded")
        for mutation in ("extra", "missing")
    ],
)
def test_reader_rejects_wrong_geometry_keys(case_name, mutation):
    if mutation == "extra":
        raw = mutate(case_name, ["mesh", "arrays", "extra"], {})
    else:
        header, data = unpack_case(case_name)
        del header["mesh"]["arrays"][next(iter(header["mesh"]["arrays"]))]
        raw = frame(header, data)
    with pytest.raises(ValueError, match="required keys"):
        cmb.read_header(io.BytesIO(raw))


@pytest.mark.parametrize(
    "case_name, field, updates, match",
    [
        ("tensor_embedded", "origin", {"dtype": "float32", "length": 12}, "origin"),
        ("uniform_embedded", "origin", {"shape": [2], "length": 16}, "origin"),
        ("octree_embedded", "position", {"shape": [2], "length": 8}, "equal lengths"),
        (
            "octree_embedded",
            "level",
            {"dtype": "int16", "length": 30},
            "must have dtype",
        ),
    ],
)
def test_reader_rejects_bad_geometry(case_name, field, updates, match):
    header, data = unpack_case(case_name)
    descriptor = header["mesh"]["arrays"][field]
    start = descriptor["offset"]
    descriptor.update(
        **updates,
        checksum=sha256_hex(data[start : start + updates["length"]]),
    )
    with pytest.raises(ValueError, match=match):
        cmb.read_header(io.BytesIO(frame(header, data)))


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
    with pytest.raises(ValueError, match="must be 1D"):
        header, data = unpack_case(case_name)
        descriptor = header["mesh"]["arrays"][field]
        width = np.dtype(cmb.DTYPE_TO_NUMPY[descriptor["dtype"]]).itemsize
        descriptor.update(
            shape=[],
            length=width,
            checksum=sha256_hex(
                data[descriptor["offset"] : descriptor["offset"] + width]
            ),
        )
        cmb.read_header(io.BytesIO(frame(header, data)))


@pytest.mark.parametrize(
    "mutation, match",
    [("missing", "base_mesh"), ("wrong_class", "UniformTensorMesh")],
)
def test_reader_rejects_missing_or_invalid_octree_base(mutation, match):
    header, data = unpack_case("octree_embedded")
    if mutation == "missing":
        del header["mesh"]["base_mesh"]
    else:
        header["mesh"]["base_mesh"]["mesh_class"] = "TensorMesh"
    with pytest.raises(ValueError, match=match):
        cmb.read_header(io.BytesIO(frame(header, data)))


@pytest.mark.parametrize("case_name", ["tensor_embedded", "uniform_embedded"])
def test_reader_rejects_non_octree_base_mesh(case_name):
    header, data = unpack_case(case_name)
    header["mesh"]["base_mesh"] = {}
    with pytest.raises(ValueError, match="does not support base_mesh"):
        cmb.read_header(io.BytesIO(frame(header, data)))


def test_reader_rejects_reference_count_and_inference_failures():
    header, data = unpack_case("reference_explicit_n_cells")
    header["mesh"]["n_cells"] = True
    with pytest.raises(ValueError, match="n_cells"):
        cmb.read_header(io.BytesIO(frame(header, data)))
    with pytest.raises(ValueError, match="n_cells"):
        cmb.resolve_reference_n_cells(None, {})
    with pytest.raises(ValueError, match="disagree"):
        cmb.resolve_reference_n_cells(
            None,
            {"a": {"array": np.zeros(2)}, "b": {"array": np.zeros(3)}},
        )


def test_reader_validates_recognized_padding_and_ignores_outer_octree_data():
    header, data = unpack_case("tensor_embedded")
    header["mesh"]["default_padding"] = named_padding([True, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="padding"):
        cmb.read_header(io.BytesIO(frame(header, data)))

    header, data = unpack_case("octree_embedded")
    header["mesh"]["default_padding"] = {"west": True}
    header["mesh"]["base_mesh"]["default_padding"] = named_padding([2, 2, 1, 1, 1, 1])
    parsed, _ = cmb.read_header(io.BytesIO(frame(header, data)))
    assert parsed["mesh"]["default_padding"] == {"west": True}
    assert parsed["mesh"]["base_mesh"]["default_padding"] == named_padding(
        [2, 2, 1, 1, 1, 1]
    )


@pytest.mark.parametrize("shape", [(6, 4, 4), (3, 3, 3), (12, 4, 4)])
def test_reader_rejects_non_power_of_two_embedded_and_reference_bases(shape):
    for case_name in ("octree_embedded", "reference_with_base_mesh"):
        header, data = unpack_case(case_name)
        descriptor = header["mesh"]["base_mesh"]["arrays"]["shape"]
        payload = np.asarray(shape, dtype="<i4").tobytes()
        start = descriptor["offset"]
        data = data[:start] + payload + data[start + len(payload) :]
        descriptor.update(checksum=sha256_hex(payload))
        with pytest.raises(ValueError, match="powers of two"):
            cmb.read_header(io.BytesIO(frame(header, data)))


def test_reader_rejects_non_object_header_and_models():
    with pytest.raises(ValueError, match="JSON object"):
        cmb.read_header(io.BytesIO(frame([], b"")))
    with pytest.raises(ValueError, match="models"):
        cmb.read_header(io.BytesIO(mutate("tensor_embedded", ["models"], [])))


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
                mutate("tensor_with_models", ["models", "rho", "array", field], value)
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
    header, data = unpack_case("tensor_with_models")
    del header["models"]["rho"]["array"][field]
    with pytest.raises(ValueError, match="missing required"):
        cmb.read_header(io.BytesIO(frame(header, data)))


def test_reader_rejects_negative_offset_even_with_matching_magic_checksum():
    header, data = unpack_case("reference_models_only")
    header["mesh"]["n_cells"] = 1
    descriptor = header["models"]["rho"]["array"]
    descriptor.update(
        offset=-8,
        length=8,
        shape=[1],
        checksum=sha256_hex(cmb.MAGIC),
    )
    with pytest.raises(ValueError, match="offset"):
        cmb.read_header(io.BytesIO(frame(header, data)))


class _RejectDataReads(io.BytesIO):
    def __init__(self, raw, data_end):
        super().__init__(raw)
        self.data_end = data_end

    def read(self, size=-1):
        if 8 <= self.tell() < self.data_end:
            raise AssertionError("payload read occurred before descriptor bounds check")
        return super().read(size)


def test_reader_checks_model_bounds_before_reading_uniform_shape_payload():
    header, data = unpack_case("uniform_embedded")
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
    raw = frame(header, data)
    with pytest.raises(ValueError, match="data section"):
        cmb.read_header(_RejectDataReads(raw, 8 + len(data)))


def test_reader_checks_nested_base_descriptor_bounds():
    header, data = unpack_case("octree_embedded")
    header["mesh"]["base_mesh"]["arrays"]["shape"]["offset"] = len(data)
    with pytest.raises(ValueError, match="data section"):
        cmb.read_header(io.BytesIO(frame(header, data)))


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


def test_writer_ignores_outer_octree_padding_and_keeps_base_value():
    mesh = fresh_mesh("octree_embedded")
    mesh["default_padding"] = {"west": True}
    mesh["base_mesh"]["default_padding"] = named_padding([2, 2, 1, 1, 1, 1])

    raw = cmb.build_file_bytes(mesh)
    header, _ = cmb.read_header(io.BytesIO(raw))

    assert "default_padding" not in header["mesh"]
    assert header["mesh"]["base_mesh"]["default_padding"] == named_padding(
        [2, 2, 1, 1, 1, 1]
    )


@pytest.mark.parametrize(
    "padding, match",
    [
        (named_padding([True, 0, 0, 0, 0, 0]), "not booleans"),
        ([1, 2, 3], "mapping"),
        (named_padding([np.inf] * 6), "finite integer values"),
        (named_padding([np.nan] * 6), "finite integer values"),
        (named_padding([0.5] * 6), "finite integer values"),
        (named_padding([-1] * 6), "non-negative"),
        (named_padding([2**63] * 6), "int64 range"),
        (named_padding(["0"] * 6), "integer values"),
    ],
)
def test_writer_rejects_bad_padding_values(padding, match):
    mesh = fresh_mesh("tensor_embedded")
    mesh["default_padding"] = padding
    with pytest.raises(ValueError, match=match):
        cmb.build_file_bytes(mesh)


@pytest.mark.parametrize("value", [[1, 2], (1, 2)])
def test_padding_values_must_be_scalar(value):
    mesh = fresh_mesh("tensor_embedded")
    mesh["default_padding"] = named_padding([value, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="must be a scalar integer"):
        cmb.build_file_bytes(mesh)


def test_raw_mesh_shape_accepts_raw_partial_inputs():
    raw = {"mesh_class": "UniformTensorMesh", "arrays": {"shape": np.array([3, 2, 4])}}
    assert cmb.raw_mesh_shape(raw) == (3, 2, 4)
    raw["mode"] = "embedded"
    assert cmb.raw_mesh_shape(raw) == (3, 2, 4)


def test_raw_mesh_shape_rejects_octree_without_base_mesh():
    mesh = fresh_mesh("octree_embedded")
    del mesh["base_mesh"]
    with pytest.raises(ValueError, match="missing required key 'base_mesh'"):
        cmb.raw_mesh_shape(mesh)


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
        cmb.read_header(io.BytesIO(mutate("tensor_with_models", path, value)))


def test_null_header_raises_format_error():
    with pytest.raises(ValueError, match="JSON object"):
        cmb.read_header(io.BytesIO(frame(None, b"")))


def test_uniform_cell_count_does_not_overflow_int64_on_write():
    mesh = fresh_mesh("uniform_embedded")
    mesh["arrays"]["shape"] = np.array([2**32, 2**32, 1], dtype=np.int64)
    # An int64 product wraps this count to zero. No model data is allocated.
    models = {"rho": {"array": np.empty(0)}}
    with pytest.raises(ValueError, match="18446744073709551616 cells"):
        cmb.build_file_bytes(mesh, models)


def test_uniform_cell_count_does_not_overflow_int64_on_read():
    header, data = unpack_case("uniform_embedded")
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
        cmb.read_header(io.BytesIO(frame(header, data + payload)))


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
    mesh = fresh_mesh("octree_embedded")
    del mesh["base_mesh"]
    path = tmp_path / "existing.cmb"
    path.write_bytes(b"existing contents")
    with pytest.raises(ValueError, match="base_mesh"):
        cmb.write_file(path, mesh)
    assert path.read_bytes() == b"existing contents"


def test_octree_position_and_base_shape_support_int64():
    mesh = fresh_mesh("octree_embedded")
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
