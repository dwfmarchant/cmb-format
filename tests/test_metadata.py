"""Test JSON serialization of file and model metadata, including NumPy scalars."""

import copy
import math

import numpy as np
import pytest

import cmb_format as cmb

MESH = {"mode": "reference", "n_cells": 2}


def models_with(metadata):
    return {"m": {"metadata": metadata, "array": np.array([1.0, 2.0])}}


def numpy_metadata():
    return {
        "threshold": np.float32(0.1),
        "half": np.float16(0.5),
        "double": np.float64(0.25),
        "air_value": np.float32("nan"),
        "ceiling": np.float32("inf"),
        "count": np.int32(-7),
        "big": np.int64(2**53 + 1),
        "unsigned": np.uint64(2**64 - 1),
        "settings": {"active": np.bool_(True), "levels": [np.int8(1), np.False_]},
    }


EXPECTED = {
    "threshold": float(np.float32(0.1)),
    "half": 0.5,
    "double": 0.25,
    "air_value": math.nan,
    "ceiling": math.inf,
    "count": -7,
    "big": 2**53 + 1,
    "unsigned": 2**64 - 1,
    "settings": {"active": True, "levels": [1, False]},
}


def assert_builtin_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_builtin_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_builtin_equal(a, e)
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def test_numpy_scalars_round_trip_as_python_scalars(tmp_path):
    file_metadata = numpy_metadata()
    model_metadata = numpy_metadata()
    file_before = copy.deepcopy(file_metadata)
    model_before = copy.deepcopy(model_metadata)
    path = tmp_path / "scalars.cmb"

    cmb.write_file(path, MESH, models_with(model_metadata), file_metadata)
    data = cmb.build_file_bytes(MESH, models_with(model_metadata), file_metadata)

    assert path.read_bytes() == data
    _, models, metadata = cmb.read_file(path)
    assert_builtin_equal(metadata, EXPECTED)
    assert_builtin_equal(models["m"]["metadata"], EXPECTED)
    assert_builtin_equal(cmb.list_models(path)["m"]["metadata"], EXPECTED)
    # Caller-owned metadata keeps its NumPy scalars.
    for after, before in ((file_metadata, file_before), (model_metadata, model_before)):
        assert type(after["threshold"]) is np.float32
        assert type(after["settings"]["active"]) is np.bool_
        assert after.keys() == before.keys()


def test_narrow_floats_widen_exactly():
    data = cmb.build_file_bytes(MESH, metadata={"x": np.float32(0.1)})
    assert b'"x": 0.10000000149011612' in data


def test_numpy_scalars_serialize_like_python_scalars():
    python = {"a": 0.5, "b": 3, "c": True, "d": [math.nan, None, "s"]}
    numpy = {
        "a": np.float32(0.5),
        "b": np.int64(3),
        "c": np.True_,
        "d": [np.float16("nan"), None, "s"],
    }
    assert cmb.build_file_bytes(MESH, metadata=numpy) == cmb.build_file_bytes(
        MESH, metadata=python
    )


@pytest.mark.parametrize(
    "value",
    [
        np.complex64(1 + 2j),
        np.datetime64("2026-01-01"),
        np.array([1.0, 2.0]),
        np.array(1.0),
        object(),
    ],
    ids=["complex", "datetime64", "array", "0d-array", "object"],
)
@pytest.mark.parametrize("level", ["file", "model"])
def test_unsupported_metadata_values_raise_before_writing(tmp_path, value, level):
    path = tmp_path / "existing.cmb"
    path.write_bytes(b"original")
    metadata = {"nested": [{"value": value}]}
    if level == "model":
        kwargs = {"models": models_with(metadata)}
    else:
        kwargs = {"metadata": metadata}

    with pytest.raises(TypeError, match="not JSON serializable"):
        cmb.build_file_bytes(MESH, **kwargs)
    with pytest.raises(TypeError, match="not JSON serializable"):
        cmb.write_file(path, MESH, **kwargs)
    assert path.read_bytes() == b"original"
    missing = tmp_path / "missing.cmb"
    with pytest.raises(TypeError):
        cmb.write_file(missing, MESH, **kwargs)
    assert not missing.exists()


@pytest.mark.skipif(
    issubclass(np.longdouble, np.float64),
    reason="longdouble is float64 on this platform",
)
def test_longdouble_is_rejected():
    with pytest.raises(TypeError, match="not JSON serializable"):
        cmb.build_file_bytes(MESH, metadata={"x": np.longdouble(1)})


def test_numpy_scalar_keys_remain_unsupported():
    with pytest.raises(TypeError, match="keys must be"):
        cmb.build_file_bytes(MESH, metadata={np.int64(1): "value"})
