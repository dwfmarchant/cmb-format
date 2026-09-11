"""Small, concrete helpers shared by codec and validation tests."""

import copy
import json
import math
import struct

import numpy as np

import cmb_format as cmb
from cases import CASES, build_bytes
from cmb_format._codec import sha256_hex

_PADDING_NAMES = ("west", "east", "south", "north", "bottom", "top")


def named_padding(values):
    return dict(zip(_PADDING_NAMES, values, strict=True))


def fresh_case(case_name):
    """Return an independent copy of a named golden case."""
    return copy.deepcopy(CASES[case_name])


def fresh_mesh(case_name):
    """Return an independent copy of a named case's mesh."""
    return fresh_case(case_name)["mesh"]


def unpack_case(case_name="tensor_with_models"):
    """Return a case header and its raw data section."""
    raw = build_bytes(CASES[case_name])
    length = struct.unpack("<Q", raw[-16:-8])[0]
    return json.loads(raw[-16 - length : -16]), raw[8 : -16 - length]


def frame(header, data):
    """Frame a JSON header and data section as a complete CMB file."""
    blob = json.dumps(header).encode()
    return cmb.MAGIC + data + blob + struct.pack("<Q", len(blob)) + cmb.MAGIC


def mutate(case_name, path, value):
    """Return a valid case with one header value replaced."""
    header, data = unpack_case(case_name)
    target = header
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return frame(header, data)


def forge_model_shape(case_name, shape):
    """Return a valid file with model bytes and the first shape replaced."""
    header, data = unpack_case(case_name)
    descriptor = next(iter(header["models"].values()))["array"]
    dtype = np.dtype(cmb.DTYPE_TO_NUMPY[descriptor["dtype"]])
    payload = np.arange(math.prod(shape), dtype=dtype).tobytes()
    descriptor["offset"] = len(data)
    descriptor["length"] = len(payload)
    descriptor["shape"] = shape
    descriptor["checksum"] = sha256_hex(payload)
    return frame(header, data + payload)
