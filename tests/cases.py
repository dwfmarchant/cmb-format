"""The golden matrix: one case per shape the specification allows.

Both `generate_goldens.py` and `test_goldens.py` build from these, so a
golden and the test that checks it can never drift apart -- and regenerating
is a deliberate act (run the generator) rather than something a test does
quietly on failure.

Arrays are chosen to be small, exactly representable in binary floating
point, and distinguishable from each other, so a golden's bytes are readable
in a hex dump when something goes wrong.
"""

import json
import struct

import numpy as np

import cmb_format as cmb


def _tensor(padding=None):
    d = {
        "mode": "embedded",
        "mesh_class": "TensorMesh",
        "arrays": {
            "origin": np.array([0.0, 10.0, -5.0]),
            "h_x": np.array([1.0, 2.0, 4.0]),
            "h_y": np.array([1.0, 1.0]),
            "h_z": np.array([0.5, 0.5, 0.5, 0.5]),
        },
    }
    if padding is not None:
        d["default_padding"] = padding
    return d


def _uniform(padding=None):
    d = {
        "mode": "embedded",
        "mesh_class": "UniformTensorMesh",
        "arrays": {
            "origin": np.array([0.0, 0.0, 0.0]),
            "cell_size": np.array([2.0, 2.0, 1.0]),
            "shape": np.array([4, 4, 2], dtype=np.int32),
        },
    }
    if padding is not None:
        d["default_padding"] = padding
    return d


def _base_mesh(padding=None):
    d = {
        "mesh_class": "UniformTensorMesh",
        "arrays": {
            "origin": np.array([0.0, 0.0, 0.0]),
            "cell_size": np.array([1.0, 1.0, 1.0]),
            "shape": np.array([4, 4, 4], dtype=np.int32),
        },
    }
    if padding is not None:
        d["default_padding"] = padding
    return d


# A real octree: eight level-0 cells filling the (0,0,0) 2x2x2 block, and
# seven level-1 cells covering the remaining corner blocks of a 4x4x4 base
# grid -- 15 cells, a complete and valid subdivision. Derived once from a
# constructed OctreeMesh and hard-coded here, so these goldens stay free of
# any consumer's mesh classes. Positions are raveled base-grid indices.
_OCTREE_LEVEL = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
_OCTREE_POSITION = np.array(
    [0, 1, 2, 4, 5, 8, 10, 16, 17, 20, 21, 32, 34, 40, 42], dtype=np.int32
)


def _octree(base_padding=None):
    return {
        "mode": "embedded",
        "mesh_class": "OctreeMesh",
        "arrays": {"level": _OCTREE_LEVEL, "position": _OCTREE_POSITION},
        "base_mesh": _base_mesh(base_padding),
    }


_RHO = {"metadata": {"units": "ohm-m"}, "array": np.array([1.0, 10.0, 100.0])}
_RHO15 = {"metadata": {"units": "ohm-m"}, "array": np.arange(1.0, 16.0)}
_SUS = {
    "metadata": {"units": "SI", "air_value": -100.0},
    "array": np.array([0.0, 0.5, 1.5]),
}

# Every dtype the format defines a token for. This case is what pins those
# token strings -- a rename of them is a wire change that round-tripping
# cannot see, because the writer and reader share the same lookup table.
_ALL_DTYPES = {
    f"m_{name}": {"metadata": {}, "array": np.arange(3, dtype=np.dtype(numpy_code))}
    for name, numpy_code in cmb.DTYPE_TO_NUMPY.items()
}

CASES = {
    "tensor_embedded": {"mesh": _tensor(), "metadata": {}, "models": {}},
    "tensor_with_models": {
        "mesh": _tensor(),
        "metadata": {"survey": "demo"},
        "models": {"rho": _RHO, "sus": _SUS},
    },
    "tensor_padding": {
        "mesh": _tensor([1, 1, 1, 1, 2, 2]),
        "metadata": {},
        "models": {},
    },
    "uniform_embedded": {"mesh": _uniform(), "metadata": {}, "models": {}},
    "uniform_padding_models": {
        "mesh": _uniform([2, 2, 1, 1, 0, 0]),
        "metadata": {},
        "models": {"rho": _RHO},
    },
    "octree_embedded": {"mesh": _octree(), "metadata": {}, "models": {}},
    "octree_base_padding_models": {
        "mesh": _octree([1, 1, 1, 1, 1, 1]),
        "metadata": {"note": "octree"},
        "models": {"rho": _RHO},
    },
    "reference_models_only": {
        "mesh": {"mode": "reference"},
        "metadata": {},
        "models": {"rho": _RHO},
    },
    "reference_explicit_n_cells": {
        "mesh": {"mode": "reference", "n_cells": 3},
        "metadata": {},
        "models": {"rho": _RHO},
    },
    "reference_with_base_mesh": {
        "mesh": {"mode": "reference", "base_mesh": _base_mesh()},
        "metadata": {},
        "models": {"rho": _RHO},
    },
    "all_dtypes": {"mesh": _tensor(), "metadata": {}, "models": _ALL_DTYPES},
}


def build_header_and_buffer(case):
    """Assemble a case into (header dict, data buffer).

    Mirrors the field order a writer must use; the header's key order is part
    of the bytes, so it is part of what the goldens pin.
    """
    buffer = bytearray()
    mesh_dict = case["mesh"]
    if mesh_dict.get("mode") == "reference":
        mesh_dict = {
            **mesh_dict,
            "n_cells": cmb.resolve_reference_n_cells(
                mesh_dict.get("n_cells"), case["models"]
            ),
        }
    header = {
        "format_version": 1,
        "mesh": cmb.serialize_mesh(mesh_dict, buffer),
        "metadata": case["metadata"],
        "models": {
            name: {
                "metadata": entry["metadata"],
                "array": cmb.serialize_array(entry["array"], buffer),
            }
            for name, entry in case["models"].items()
        },
    }
    return header, buffer


def build_bytes(case):
    """Assemble a case into the exact bytes of a `.cmb` file."""
    header, buffer = build_header_and_buffer(case)
    blob = json.dumps(header).encode("utf-8")
    return cmb.MAGIC + bytes(buffer) + blob + struct.pack("<Q", len(blob)) + cmb.MAGIC
