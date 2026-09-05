"""Validation and normalization for the six-value visualization padding.

Padding is part of the CMB format -- a file's header may carry
``default_padding`` -- so its canonical representation and validation live
here rather than in any consumer.

Always in canonical order ``[west, east, south, north, bottom, top]``.
"""

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "normalize_default_padding",
    "normalize_integer_array",
    "validate_default_padding_shape",
]


def normalize_integer_array(
    values: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    minimum: int | None = None,
    value_description: str = "scalar integers",
) -> NDArray[np.integer]:
    """Normalize scalar integer values with an optional shape/range check."""
    try:
        values_arr = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be {value_description}") from exc
    if shape is not None and values_arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values_arr.shape}")

    int64_info = np.iinfo(np.int64)
    int64_min = int(int64_info.min)
    int64_max = int(int64_info.max)
    normalized: list[int] = []
    for value in values_arr.tolist():
        # bool is an Integral subclass, but is explicitly not valid here.
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} values must be integers, not booleans")
        if isinstance(value, Integral):
            integer = int(value)
        elif isinstance(value, Real):
            number = float(value)
            if not np.isfinite(number) or not number.is_integer():
                raise ValueError(f"{name} values must be finite integer values")
            integer = int(number)
        else:
            raise ValueError(f"{name} values must be {value_description}")
        if integer < int64_min or integer > int64_max:
            raise ValueError(f"{name} values exceed int64 range")
        if minimum is not None and integer < minimum:
            if minimum == 0:
                raise ValueError(f"{name} values must be non-negative")
            raise ValueError(f"{name} values must be {value_description}")
        normalized.append(integer)

    return np.asarray(normalized, dtype=np.int64)


def normalize_default_padding(
    default_padding: ArrayLike | None,
) -> NDArray[np.integer] | None:
    """Normalize padding to a read-only int64 array in canonical order.

    Canonical order is ``[west, east, south, north, bottom, top]``, defined
    by ``docs/binary-format.md`` -- the format's own convention, not any
    consumer's.
    """
    if default_padding is None:
        return None
    normalized = normalize_integer_array(
        default_padding,
        name="default_padding",
        shape=(6,),
        minimum=0,
    )

    # Keep immutable backing storage as well as marking the view read-only:
    # callers cannot opt back into writability with ``setflags(write=True)``
    # on the returned public array.
    result = np.frombuffer(
        np.asarray(normalized, dtype=np.int64).tobytes(), dtype=np.int64
    )
    result.setflags(write=False)
    return result


def validate_default_padding_shape(
    default_padding: NDArray[np.integer] | None, shape: ArrayLike
) -> None:
    """Validate opposing padding sides against a three-axis shape."""
    if default_padding is None:
        return
    shape_arr = np.asarray(shape)
    if shape_arr.shape != (3,):
        raise ValueError(f"mesh shape must have shape (3,), got {shape_arr.shape}")
    for side_a, side_b, axis, n_cells in zip(
        default_padding[::2],
        default_padding[1::2],
        ("x", "y", "z"),
        shape_arr,
        strict=True,
    ):
        if int(side_a) + int(side_b) > int(n_cells):
            raise ValueError(
                f"default_padding {axis}-axis opposing sides exceed "
                f"the mesh shape ({int(n_cells)} cells)"
            )
