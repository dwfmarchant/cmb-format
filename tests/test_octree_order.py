"""Check root-local Morton ordering keys and their enforcement on octree IO."""

import pathlib

import numpy as np
import pytest

import cmb_format as cmb
from cases import CASES
from test_helpers import fresh_case, fresh_mesh, reorder_octree_payloads

GOLDENS = pathlib.Path(__file__).parent / "goldens"
VERSIONS = ["v1", "v2"]
WRITERS = ["write_file", "build_file_bytes"]
OCTREE_CASES = [
    "octree_embedded",
    "octree_base_padding_models",
    "octree_rectangular_models",
]
ORDER_ERROR = "root-local Morton order"

# octree_rectangular_models sorted by the global Morton code of each lower
# corner. Global codes rank (0, 2, 0) at position 16 before (4, 0, 0) at
# position 4, whereas root-local order visits the 4x2x1 roots x fastest.
RECTANGULAR_GLOBAL_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 9, 10, 13, 14]
NONCANONICAL_ORDERS = [
    ("octree_rectangular_models", "global"),
    ("octree_rectangular_models", "position"),
    ("octree_rectangular_models", "reversed"),
    ("octree_rectangular_models", "shuffled"),
    # A single cubic root makes global Morton order the required order.
    ("octree_embedded", "position"),
    ("octree_embedded", "reversed"),
    ("octree_embedded", "shuffled"),
]


def _write(writer, path, mesh, models=None, metadata=None):
    if writer == "write_file":
        cmb.write_file(path, mesh, models, metadata)
    else:
        path.write_bytes(cmb.build_file_bytes(mesh, models, metadata))


def _permutation(case_name, order):
    position = CASES[case_name]["mesh"]["arrays"]["position"]
    if order == "global":
        return np.array(RECTANGULAR_GLOBAL_ORDER)
    if order == "position":
        return np.argsort(position)
    if order == "reversed":
        return np.arange(position.size)[::-1]
    return np.random.default_rng(0).permutation(position.size)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("name", OCTREE_CASES)
def test_octree_goldens_follow_root_local_morton_order(version, name):
    with open(GOLDENS / version / f"{name}.cmb", "rb") as f:
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
        assert span > 1, f"Missing cell at base cell {(i, j, k)}"
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
    # The shared ordering keys agree with the traversal.
    assert np.all(np.diff(cmb.octree_order_keys(positions, shape)) > 0)


def test_keys_number_roots_x_fastest_then_local_morton():
    # 8x4x2 has L = 2, so eight 2x2x2 roots form a 4x2x1 grid and each root
    # spans eight keys. The first root's level-0 cells take local Morton codes
    # x_bit + 2*y_bit + 4*z_bit; the other roots start at keys 8, 16, ... 56.
    position = CASES["octree_rectangular_models"]["mesh"]["arrays"]["position"]
    keys = cmb.octree_order_keys(position, (8, 4, 2))
    assert keys.dtype == np.int64
    np.testing.assert_array_equal(
        keys, [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56]
    )


@pytest.mark.parametrize(
    "shape, corner, key",
    [
        # One cubic root: local (3, 0, 1) interleaves to 0b001_101.
        ((4, 4, 4), (3, 0, 1), 0b001_101),
        # L = 4 and 4x2x1 roots: root (1, 1, 0) is root 1 + 4 * 1 = 5, and
        # local (1, 2, 3) interleaves to 0b110_101.
        ((16, 8, 4), (5, 6, 3), 5 * 4**3 + 0b110_101),
        # L = 2 with roots stacked along y and z: root (0, 1, 0) is root 1.
        ((2, 4, 8), (0, 2, 0), 1 * 2**3),
        # L = 1: each root is a single base cell, so keys equal positions.
        ((4, 4, 1), (3, 2, 0), 3 + 4 * 2),
    ],
)
def test_keys_match_hand_computed_values(shape, corner, key):
    i, j, k = corner
    nx, ny, _ = shape
    assert cmb.octree_order_keys([i + nx * (j + ny * k)], shape).tolist() == [key]


@pytest.mark.parametrize(
    "shape, first, second",
    [
        # 4x2x1 roots: root-local order reaches (4, 0, 0) before (0, 2, 0),
        # but global Morton codes rank (0, 2, 0) first (16 versus 64).
        ((8, 4, 2), (4, 0, 0), (0, 2, 0)),
        # L = 1 gives x-fastest order, while global Morton codes rank
        # (0, 1, 0) before (2, 0, 0) (2 versus 8).
        ((4, 4, 1), (2, 0, 0), (0, 1, 0)),
    ],
)
def test_root_local_order_differs_from_global_morton(shape, first, second):
    nx, ny, _ = shape
    position = [i + nx * (j + ny * k) for i, j, k in (first, second)]
    first_key, second_key = cmb.octree_order_keys(position, shape)
    assert first_key < second_key


@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (4, 4, 1), (4, 4, 4), (8, 4, 2), (2, 4, 8), (16, 2, 2)]
)
def test_keys_number_every_base_cell_once(shape):
    n_cells = int(np.prod(shape))
    keys = cmb.octree_order_keys(np.arange(n_cells), shape)
    np.testing.assert_array_equal(np.sort(keys), np.arange(n_cells))


def test_keys_follow_input_order_without_modifying_inputs():
    position = np.array([22, 0, 16, 2], dtype=np.int32)
    position.setflags(write=False)
    shape = np.array([8, 4, 2], dtype=np.int64)
    keys = cmb.octree_order_keys(position, shape)

    assert keys.dtype == np.int64
    assert keys.tolist() == [56, 0, 32, 8]
    assert position.tolist() == [22, 0, 16, 2]
    assert position.dtype == np.int32
    assert shape.tolist() == [8, 4, 2]
    for same in ([22, 0, 16, 2], position.astype(np.uint16), position.astype(">i8")):
        assert cmb.octree_order_keys(same, [8, 4, 2]).tolist() == [56, 0, 32, 8]


@pytest.mark.parametrize(
    "position, shape, match",
    [
        ([0], (4, 4), "shape"),
        ([0], (4, 4, 0), "positive"),
        ([0], (4, 4, 3), "powers of two"),
        ([0], (4, 4, True), "booleans"),
        ([0], (2**21, 2**21, 2**21), r"2\*\*62"),
        ([-1], (4, 4, 4), "lie in"),
        ([64], (4, 4, 4), "lie in"),
        (np.array([2**64 - 1], dtype=np.uint64), (4, 4, 4), "lie in"),
        ([0.0], (4, 4, 4), "integer array"),
        ([True], (4, 4, 4), "integer array"),
        ([[0]], (4, 4, 4), "one-dimensional"),
    ],
)
def test_keys_reject_invalid_positions_and_shapes(position, shape, match):
    with pytest.raises(ValueError, match=match):
        cmb.octree_order_keys(position, shape)


def test_keys_limit_local_not_global_coordinates():
    # This domain exceeds 2**21 cells along x, but its roots are 2x2x2, so
    # local Morton coordinates stay below 2. The last root spans the last
    # eight keys.
    nx = 2**22
    last_x = nx - 1
    keys = cmb.octree_order_keys([0, last_x, last_x + nx * 3], (nx, 2, 2))
    last_root = (nx // 2 - 1) * 8
    assert keys.tolist() == [0, last_root + 1, last_root + 7]


def test_keys_cover_the_largest_supported_base_grid():
    # 2**62 base cells: L = 2**20 and a 2x2x1 grid of roots. The final corner
    # has the final key; larger grids are rejected before computing keys.
    shape = (2**21, 2**21, 2**20)
    n_cells = 2**62
    assert cmb.octree_order_keys([0, n_cells - 1], shape).tolist() == [
        0,
        n_cells - 1,
    ]
    with pytest.raises(ValueError, match="lie in"):
        cmb.octree_order_keys([n_cells], shape)


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("case_name", OCTREE_CASES)
def test_writers_accept_canonical_octrees_without_changing_inputs(
    tmp_path, writer, case_name
):
    case = fresh_case(case_name)
    arrays = dict(case["mesh"]["arrays"])
    copies = {name: values.copy() for name, values in arrays.items()}
    model_arrays = {name: entry["array"] for name, entry in case["models"].items()}
    path = tmp_path / "out.cmb"
    _write(writer, path, case["mesh"], case["models"], case["metadata"])

    assert path.read_bytes() == (GOLDENS / "v2" / f"{case_name}.cmb").read_bytes()
    for name, values in case["mesh"]["arrays"].items():
        assert values is arrays[name]
        np.testing.assert_array_equal(values, copies[name], strict=True)
    for name, entry in case["models"].items():
        assert entry["array"] is model_arrays[name]


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("case_name, order", NONCANONICAL_ORDERS)
def test_writers_reject_noncanonical_octree_order(tmp_path, writer, case_name, order):
    permutation = _permutation(case_name, order)
    assert not np.array_equal(permutation, np.arange(permutation.size))
    case = fresh_case(case_name)
    mesh = case["mesh"]
    mesh["arrays"] = {
        name: values[permutation] for name, values in mesh["arrays"].items()
    }
    models = {
        name: {**entry, "array": entry["array"][permutation]}
        for name, entry in case["models"].items()
    }
    with pytest.raises(ValueError, match=ORDER_ERROR):
        _write(writer, tmp_path / "out.cmb", mesh, models)


@pytest.mark.parametrize("writer", WRITERS)
def test_writers_reject_repeated_octree_positions(tmp_path, writer):
    mesh = fresh_mesh("octree_embedded")
    mesh["arrays"] = {
        name: np.append(values, values[-1]) for name, values in mesh["arrays"].items()
    }
    with pytest.raises(ValueError, match="position 42 is repeated at indices 14 and"):
        _write(writer, tmp_path / "out.cmb", mesh)


def test_order_error_explains_the_required_permutation():
    mesh = fresh_mesh("octree_rectangular_models")
    mesh["arrays"] = {name: values[::-1] for name, values in mesh["arrays"].items()}
    with pytest.raises(ValueError) as excinfo:
        cmb.build_file_bytes(mesh)
    message = str(excinfo.value)
    assert "position 20 at index 1 belongs before position 22 at index 0" in message
    assert "octree_order_keys" in message
    assert "same permutation to level, position, and every model" in message


def test_rejected_write_leaves_destination_and_inputs_untouched(tmp_path):
    case = fresh_case("octree_rectangular_models")
    mesh = case["mesh"]
    mesh["arrays"] = {name: values[::-1] for name, values in mesh["arrays"].items()}
    arrays = dict(mesh["arrays"])
    copies = {name: values.copy() for name, values in arrays.items()}
    rho = case["models"]["rho"]["array"]
    rho_copy = rho.copy()
    path = tmp_path / "existing.cmb"
    path.write_bytes(b"existing contents")

    with pytest.raises(ValueError, match=ORDER_ERROR):
        cmb.write_file(path, mesh, case["models"])
    assert path.read_bytes() == b"existing contents"
    for name, values in mesh["arrays"].items():
        assert values is arrays[name]
        np.testing.assert_array_equal(values, copies[name], strict=True)
    assert case["models"]["rho"]["array"] is rho
    np.testing.assert_array_equal(rho, rho_copy, strict=True)


@pytest.mark.parametrize("selection", [None, ["rho"], []])
def test_read_file_rejects_noncanonical_octree_files(tmp_path, selection):
    path = tmp_path / "unordered.cmb"
    path.write_bytes(
        reorder_octree_payloads("octree_rectangular_models", RECTANGULAR_GLOBAL_ORDER)
    )
    with pytest.raises(ValueError, match=ORDER_ERROR):
        cmb.read_file(path, models=selection)


@pytest.mark.parametrize(
    "selection, expected_models", [(None, ["rho"]), (["rho"], ["rho"]), ([], [])]
)
@pytest.mark.parametrize("version", VERSIONS)
def test_read_file_returns_canonical_octrees_as_stored(
    version, selection, expected_models
):
    case = CASES["octree_base_padding_models"]
    path = GOLDENS / version / "octree_base_padding_models.cmb"
    mesh, models, metadata = cmb.read_file(path, models=selection)

    for name, values in case["mesh"]["arrays"].items():
        np.testing.assert_array_equal(mesh["arrays"][name], values, strict=True)
        assert not mesh["arrays"][name].flags.writeable
    base_padding = case["mesh"]["base_mesh"]["default_padding"]
    assert mesh["base_mesh"]["default_padding"] == base_padding
    assert list(models) == expected_models
    for name in expected_models:
        expected = case["models"][name]
        assert models[name]["metadata"] == expected["metadata"]
        np.testing.assert_array_equal(
            models[name]["array"], expected["array"], strict=True
        )
    assert metadata == case["metadata"]


def test_low_level_readers_return_noncanonical_arrays_unchanged(tmp_path):
    order = np.arange(15)[::-1]
    path = tmp_path / "unordered.cmb"
    path.write_bytes(reorder_octree_payloads("octree_rectangular_models", order))
    expected = CASES["octree_rectangular_models"]

    # Header inspection neither reads nor certifies octree cell order.
    assert cmb.list_models(path)["rho"]["shape"] == [15]
    assert cmb.read_contents(path)["n_cells"] == 15
    with open(path, "rb") as f:
        header, start = cmb.read_header(f)
        arrays = cmb.read_arrays(f, start, header["mesh"]["arrays"])
        rho = cmb.read_array(f, start, header["models"]["rho"]["array"])
    for name, values in expected["mesh"]["arrays"].items():
        np.testing.assert_array_equal(arrays[name], values[order], strict=True)
    np.testing.assert_array_equal(
        rho, expected["models"]["rho"]["array"][order], strict=True
    )


@pytest.mark.parametrize("writer", WRITERS)
def test_reference_models_keep_their_supplied_order(tmp_path, writer):
    base_mesh = fresh_mesh("octree_rectangular_models")["base_mesh"]
    # Reference files carry no geometry, so no cell order can be checked.
    rho = np.arange(15, dtype=np.float64)[::-1] ** 2
    path = tmp_path / "reference.cmb"
    _write(
        writer,
        path,
        {"mode": "reference", "base_mesh": base_mesh},
        {"rho": {"metadata": {"units": "ohm-m"}, "array": rho}},
    )

    mesh, models, _ = cmb.read_file(path)
    assert mesh["n_cells"] == 15
    assert models["rho"]["metadata"] == {"units": "ohm-m"}
    np.testing.assert_array_equal(models["rho"]["array"], rho, strict=True)


@pytest.mark.parametrize(
    "case_name", ["octree_base_padding_models", "octree_rectangular_models"]
)
def test_migration_reorders_cells_and_models_together(tmp_path, case_name):
    case = CASES[case_name]
    source = tmp_path / "unordered.cmb"
    source.write_bytes(reorder_octree_payloads(case_name, np.arange(15)[::-1]))
    with pytest.raises(ValueError, match=ORDER_ERROR):
        cmb.read_file(source)

    # The README's migration recipe.
    with open(source, "rb") as f:
        header, data_start = cmb.read_header(f)
        mesh = header["mesh"]
        mesh["arrays"] = cmb.read_arrays(f, data_start, mesh["arrays"])
        base = mesh["base_mesh"]
        base["arrays"] = cmb.read_arrays(f, data_start, base["arrays"])
        models = {
            name: {**entry, "array": cmb.read_array(f, data_start, entry["array"])}
            for name, entry in header.get("models", {}).items()
        }
    keys = cmb.octree_order_keys(mesh["arrays"]["position"], base["arrays"]["shape"])
    order = np.argsort(keys, kind="stable")
    mesh["arrays"] = {name: values[order] for name, values in mesh["arrays"].items()}
    for entry in models.values():
        entry["array"] = entry["array"][order]
    target = tmp_path / "ordered.cmb"
    cmb.write_file(target, mesh, models, header.get("metadata", {}))

    migrated_mesh, migrated_models, _ = cmb.read_file(target)
    labels = dict(
        zip(
            case["mesh"]["arrays"]["position"].tolist(),
            case["models"]["rho"]["array"].tolist(),
            strict=True,
        )
    )
    # Each model value still describes the cell it was stored with.
    assert migrated_models["rho"]["array"].tolist() == [
        labels[position] for position in migrated_mesh["arrays"]["position"].tolist()
    ]
    # Cells, models, padding, and metadata match the canonical file exactly.
    assert target.read_bytes() == (GOLDENS / "v2" / f"{case_name}.cmb").read_bytes()


@pytest.mark.parametrize("version", VERSIONS)
def test_reordered_octree_preserves_model_cell_associations(version):
    path = GOLDENS / version / "octree_base_padding_models.cmb"
    mesh, models, _ = cmb.read_file(path)
    position = mesh["arrays"]["position"]
    # The original fixture assigned 1..15 in ascending base-grid position order.
    np.testing.assert_array_equal(
        models["rho"]["array"][np.argsort(position)], np.arange(1.0, 16.0)
    )


@pytest.mark.parametrize("version", VERSIONS)
def test_rectangular_octree_model_values_identify_their_cells(version):
    path = GOLDENS / version / "octree_rectangular_models.cmb"
    mesh, models, _ = cmb.read_file(path)
    np.testing.assert_array_equal(
        models["rho"]["array"], mesh["arrays"]["position"] + 0.5
    )
