"""Check octree golden ordering using a recursive traversal of the roots."""

import pathlib

import numpy as np
import pytest

import cmb_format as cmb

GOLDENS = pathlib.Path(__file__).parent / "goldens" / "v1"
OCTREE_CASES = [
    "octree_embedded",
    "octree_base_padding_models",
    "octree_rectangular_models",
]


@pytest.mark.parametrize("name", OCTREE_CASES)
def test_octree_goldens_follow_root_local_morton_order(name):
    with open(GOLDENS / f"{name}.cmb", "rb") as f:
        header, start = cmb.read_header(f)
        mesh = header["mesh"]
        arrays = cmb.read_arrays(f, start, mesh["arrays"])
        shape = cmb.read_array(f, start, mesh["base_mesh"]["arrays"]["shape"])

    nx, ny, nz = (int(value) for value in shape)
    root_size = min(nx, ny, nz)
    levels = arrays["level"]
    positions = arrays["position"]
    by_position = {int(position): index for index, position in enumerate(positions)}
    assert len(by_position) == len(positions) == len(levels)
    visited = []

    def visit(i, j, k, span):
        position = i + nx * (j + ny * k)
        index = by_position.get(position)
        if index is not None and 2 ** int(levels[index]) == span:
            visited.append(index)
            return
        assert span > 1, f"Missing leaf at base cell {(i, j, k)}"
        half = span // 2
        for child in range(8):
            visit(
                i + (child & 1) * half,
                j + ((child >> 1) & 1) * half,
                k + ((child >> 2) & 1) * half,
                half,
            )

    # Traverse roots and children directly, independently of a Morton-code sort.
    for k in range(0, nz, root_size):
        for j in range(0, ny, root_size):
            for i in range(0, nx, root_size):
                visit(i, j, k, root_size)
    assert visited == list(range(len(levels)))


def test_reordered_octree_preserves_model_cell_associations():
    with open(GOLDENS / "octree_base_padding_models.cmb", "rb") as f:
        header, start = cmb.read_header(f)
        position = cmb.read_array(f, start, header["mesh"]["arrays"]["position"])
        rho = cmb.read_array(f, start, header["models"]["rho"]["array"])
    # The original fixture assigned 1..15 in ascending base-grid position order.
    np.testing.assert_array_equal(rho[np.argsort(position)], np.arange(1.0, 16.0))


def test_rectangular_octree_model_values_identify_their_cells():
    with open(GOLDENS / "octree_rectangular_models.cmb", "rb") as f:
        header, start = cmb.read_header(f)
        position = cmb.read_array(f, start, header["mesh"]["arrays"]["position"])
        rho = cmb.read_array(f, start, header["models"]["rho"]["array"])
    np.testing.assert_array_equal(rho, position + 0.5)
