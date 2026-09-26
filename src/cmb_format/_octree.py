"""Compute the root-local Morton order required for embedded octree cells.

See ``docs/binary-format.md`` for the convention. The writers and
``read_file`` reject octree geometry whose ordering keys do not strictly
increase.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cmb_format._padding import normalize_integer_array

__all__ = ["octree_order_keys"]

# Keys number base cells, so every key must fit in int64. With power-of-two
# dimensions this admits base grids of up to 2**62 cells.
_MAX_BASE_CELLS = int(np.iinfo(np.int64).max)


def _validate_power_of_two_shape(shape: tuple[int, int, int], *, context: str) -> None:
    if any(value <= 0 or value & (value - 1) for value in shape):
        raise ValueError(f"{context} base-grid dimensions must each be powers of two")


def octree_order_keys(position: ArrayLike, shape: ArrayLike) -> NDArray[np.int64]:
    """Return the root-local Morton ordering key of each octree cell.

    Embedded octree cells are stored in strictly increasing key order. The
    base grid is divided into cubic roots of ``L = min(nx, ny, nz)`` base cells
    per axis, numbered x fastest, then y, then z. A cell's key is
    ``root_id * L**3`` plus the Morton code of its lower corner within that
    root. The code interleaves local coordinate bits from least significant,
    so each three-bit group is ``x_bit + 2*y_bit + 4*z_bit``. Distinct
    positions have distinct keys.

    Parameters
    ----------
    position : array_like of int
        One-dimensional base-grid indices ``i + nx * (j + ny * k)`` of cell
        lower corners, each in ``[0, nx * ny * nz)``.
    shape : sequence of int
        Base-grid shape ``(nx, ny, nz)``. Each dimension must be a positive
        power of two, and the grid may contain at most ``2**62`` cells.

    Returns
    -------
    numpy.ndarray
        ``int64`` keys in ``[0, nx * ny * nz)``, one for each position, in
        input order.

    Notes
    -----
    Inputs are neither sorted nor modified. ``numpy.argsort(keys,
    kind="stable")`` gives the permutation into CMB order; apply that same
    permutation to ``level``, ``position``, and every model array. Sorting
    cannot repair model values that already describe the wrong cells.
    """
    nx, ny, nz = (
        int(value)
        for value in normalize_integer_array(
            shape,
            name="octree base-grid shape",
            shape=(3,),
            minimum=1,
            value_description="three positive integer values",
        )
    )
    _validate_power_of_two_shape((nx, ny, nz), context="octree")
    n_base_cells = nx * ny * nz
    if n_base_cells > _MAX_BASE_CELLS:
        raise ValueError(
            f"octree base grid {(nx, ny, nz)} exceeds 2**62 cells, so its "
            "ordering keys would not fit in int64"
        )

    values = np.asarray(position)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("octree position must be a one-dimensional integer array")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= n_base_cells):
        raise ValueError(
            f"octree position values must lie in [0, {n_base_cells}) for "
            f"base-grid shape {(nx, ny, nz)}"
        )
    values = values.astype(np.int64, copy=False)

    # Power-of-two dimensions let shifts and masks replace division. Local
    # coordinates are below L <= 2**20, since L**3 <= nx * ny * nz <= 2**62.
    x_bits = nx.bit_length() - 1
    y_bits = ny.bit_length() - 1
    root_bits = min(nx, ny, nz).bit_length() - 1
    i = values & (nx - 1)
    j = (values >> x_bits) & (ny - 1)
    k = values >> (x_bits + y_bits)
    root_id = (i >> root_bits) + (nx >> root_bits) * (
        (j >> root_bits) + (ny >> root_bits) * (k >> root_bits)
    )
    local_mask = (1 << root_bits) - 1
    local = (
        _spread_bits(i & local_mask)
        | (_spread_bits(j & local_mask) << 1)
        | (_spread_bits(k & local_mask) << 2)
    )
    return (root_id << (3 * root_bits)) | local.astype(np.int64)


def _spread_bits(values: NDArray[np.int64]) -> NDArray[np.uint64]:
    """Move bit ``b`` of each value below ``2**21`` to bit ``3 * b``."""
    spread = values.astype(np.uint64)
    spread = (spread | (spread << 32)) & 0x1F00000000FFFF
    spread = (spread | (spread << 16)) & 0x1F0000FF0000FF
    spread = (spread | (spread << 8)) & 0x100F00F00F00F00F
    spread = (spread | (spread << 4)) & 0x10C30C30C30C30C3
    return (spread | (spread << 2)) & 0x1249249249249249


def _validate_octree_order(
    position: ArrayLike, shape: ArrayLike, *, context: str
) -> None:
    """Raise unless octree cells are stored in strictly increasing key order."""
    keys = octree_order_keys(position, shape)
    unordered = keys[1:] <= keys[:-1]
    if not unordered.any():
        return
    index = int(unordered.argmax()) + 1
    values = np.asarray(position)
    current, previous = int(values[index]), int(values[index - 1])
    if current == previous:
        problem = f"position {current} is repeated at indices {index - 1} and {index}"
    else:
        problem = (
            f"position {current} at index {index} belongs before position "
            f"{previous} at index {index - 1}"
        )
    raise ValueError(
        f"{context} cells must be stored in root-local Morton order: {problem}. "
        "Reorder with numpy.argsort(octree_order_keys(position, shape), "
        "kind='stable') and apply the same permutation to level, position, "
        "and every model."
    )
