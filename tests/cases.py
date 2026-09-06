"""Shared inputs for golden-file generation and serialization tests.

Includes embedded and reference modes, all three mesh classes, nested base
meshes, padding, multiple models, and the supported dtypes. Small arrays
with exactly representable values make the binary fixtures easier to inspect.
"""

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


# Cell counts of the meshes above. Models must be one value per cell.
_TENSOR_CELLS = 3 * 2 * 4
_UNIFORM_CELLS = 4 * 4 * 2
_OCTREE_CELLS = len(_OCTREE_LEVEL)
_REFERENCE_CELLS = 3


def _rho(n):
    return {"metadata": {"units": "ohm-m"}, "array": np.arange(1.0, n + 1.0)}


def _sus(n):
    return {
        "metadata": {"units": "SI", "air_value": -100.0},
        "array": np.arange(n, dtype=np.float64) * 0.5,
    }


def _all_dtypes(n):
    """One model per dtype the format defines a token for."""
    return {
        f"m_{name}": {"metadata": {}, "array": np.arange(n, dtype=np.dtype(code))}
        for name, code in cmb.DTYPE_TO_NUMPY.items()
    }


CASES = {
    "tensor_embedded": {"mesh": _tensor(), "metadata": {}, "models": {}},
    "tensor_with_models": {
        "mesh": _tensor(),
        "metadata": {"survey": "demo"},
        "models": {"rho": _rho(_TENSOR_CELLS), "sus": _sus(_TENSOR_CELLS)},
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
        "models": {"rho": _rho(_UNIFORM_CELLS)},
    },
    "octree_embedded": {"mesh": _octree(), "metadata": {}, "models": {}},
    "octree_base_padding_models": {
        "mesh": _octree([1, 1, 1, 1, 1, 1]),
        "metadata": {"note": "octree"},
        "models": {"rho": _rho(_OCTREE_CELLS)},
    },
    "reference_models_only": {
        "mesh": {"mode": "reference"},
        "metadata": {},
        "models": {"rho": _rho(_REFERENCE_CELLS)},
    },
    "reference_explicit_n_cells": {
        "mesh": {"mode": "reference", "n_cells": _REFERENCE_CELLS},
        "metadata": {},
        "models": {"rho": _rho(_REFERENCE_CELLS)},
    },
    "reference_with_base_mesh": {
        "mesh": {"mode": "reference", "base_mesh": _base_mesh()},
        "metadata": {},
        "models": {"rho": _rho(_REFERENCE_CELLS)},
    },
    "all_dtypes": {
        "mesh": _tensor(),
        "metadata": {},
        "models": _all_dtypes(_TENSOR_CELLS),
    },
}


def build_bytes(case):
    """Serialize a case using the package writer exercised by the golden tests."""
    return cmb.build_file_bytes(case["mesh"], case["models"], case["metadata"])
