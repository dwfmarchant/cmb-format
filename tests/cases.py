"""Shared inputs for golden-file generation and serialization tests.

Includes embedded and reference modes, all three mesh classes, nested base
meshes, padding, multiple models, and the supported dtypes. Small arrays
with exactly representable values make the binary fixtures easier to inspect.
"""

import numpy as np

import cmb_format as cmb

PADDING_NAMES = ("west", "east", "south", "north", "bottom", "top")
_DTYPE_ITEMS = (
    ("float64", "<f8"),
    ("float32", "<f4"),
    ("int64", "<i8"),
    ("int32", "<i4"),
    ("int16", "<i2"),
    ("int8", "<i1"),
)


def named_padding(values):
    return dict(zip(PADDING_NAMES, values, strict=True))


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
        d["default_padding"] = named_padding(padding)
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
        d["default_padding"] = named_padding(padding)
    return d


def _base_mesh(padding=None, shape=(4, 4, 4)):
    d = {
        "mesh_class": "UniformTensorMesh",
        "arrays": {
            "origin": np.array([0.0, 0.0, 0.0]),
            "cell_size": np.array([1.0, 1.0, 1.0]),
            "shape": np.array(shape, dtype=np.int32),
        },
    }
    if padding is not None:
        d["default_padding"] = named_padding(padding)
    return d


# A real octree: eight level-0 cells filling the (0,0,0) 2x2x2 block, and
# seven level-1 cells covering the remaining corner blocks of a 4x4x4 base
# grid -- 15 cells, a complete subdivision in root-local Morton order.
# Positions retain their x-fastest base-grid encoding. Fixtures are independent
# of consumer mesh classes and their sorting implementations.
_OCTREE_LEVEL = np.array([0] * 8 + [1] * 7, dtype=np.int8)
_OCTREE_POSITION = np.array(
    [0, 1, 4, 5, 16, 17, 20, 21, 2, 8, 10, 32, 34, 40, 42], dtype=np.int32
)
# Preserve each cell's original resistivity while reordering the leaves.
_OCTREE_RHO = np.array([1, 2, 4, 5, 8, 9, 10, 11, 3, 6, 7, 12, 13, 14, 15], dtype=float)

# An 8x4x2 base grid has eight 2x2x2 roots. Refine the first root into eight
# level-0 leaves, then visit the seven remaining roots in x-fastest order.
# Root corners 4 and 6 precede 16 here; a global Morton sort reverses that.
_RECTANGULAR_POSITION = np.array(
    [0, 1, 8, 9, 32, 33, 40, 41, 2, 4, 6, 16, 18, 20, 22], dtype=np.int32
)


def _octree(base_padding=None):
    return {
        "mode": "embedded",
        "mesh_class": "OctreeMesh",
        "arrays": {"level": _OCTREE_LEVEL, "position": _OCTREE_POSITION},
        "base_mesh": _base_mesh(base_padding),
    }


def _rectangular_octree():
    return {
        "mode": "embedded",
        "mesh_class": "OctreeMesh",
        "arrays": {
            "level": np.array([0] * 8 + [1] * 7, dtype=np.int8),
            "position": _RECTANGULAR_POSITION,
        },
        "base_mesh": _base_mesh(shape=(8, 4, 2)),
    }


# Cell counts of the meshes above. Models must be one value per cell.
_TENSOR_CELLS = 3 * 2 * 4
_UNIFORM_CELLS = 4 * 4 * 2
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
        for name, code in _DTYPE_ITEMS
    }


# These twelve existing names have frozen v1 fixtures and must not be renamed
# or changed. Add new scenarios under new names when only v2 coverage is needed.
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
        "models": {"rho": {"metadata": {"units": "ohm-m"}, "array": _OCTREE_RHO}},
    },
    "octree_rectangular_models": {
        "mesh": _rectangular_octree(),
        "metadata": {},
        "models": {
            "rho": {
                "metadata": {"units": "ohm-m"},
                "array": _RECTANGULAR_POSITION.astype(np.float64) + 0.5,
            }
        },
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
