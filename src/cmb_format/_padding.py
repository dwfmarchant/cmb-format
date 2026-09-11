"""Normalize and validate named visualization padding.

Padding mappings use west, east, south, north, bottom, and top names. Values
are normalized to non-negative Python integers with shared shape validation.
"""

from collections.abc import Mapping
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
    """Normalize integral numeric values to int64 with optional shape and bounds.

    Accepts integers and finite, integer-valued real numbers. Rejects booleans
    and values outside the int64 range.
    """
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


_PADDING_NAMES = ("west", "east", "south", "north", "bottom", "top")
_PADDING_NAME_SET = frozenset(_PADDING_NAMES)


def _normalize_padding_value(value, *, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"{name} must be a scalar integer")
        value = value.item()
    normalized = normalize_integer_array(
        [value], name=name, shape=(1,), minimum=0, value_description="integer values"
    )
    return int(normalized[0])


def normalize_default_padding(
    default_padding: Mapping[str, object] | None,
) -> dict[str, int] | None:
    """Return a fresh complete named padding mapping, or ``None``.

    Mapping values are finite, integer-valued, non-negative numbers. Omitted
    names default to zero; unknown names and positional sequences are rejected.
    """
    if default_padding is None:
        return None
    if not isinstance(default_padding, Mapping):
        raise ValueError(
            "default_padding must be a mapping with named west/east/south/north/"
            "bottom/top values"
        )
    unknown = [name for name in default_padding if name not in _PADDING_NAME_SET]
    if unknown:
        raise ValueError(f"default_padding has unknown name(s): {unknown!r}")
    return {
        name: _normalize_padding_value(
            default_padding.get(name, 0), name=f"default_padding[{name!r}]"
        )
        for name in _PADDING_NAMES
    }


def validate_default_padding_shape(
    default_padding: Mapping[str, object] | None, shape: ArrayLike
) -> None:
    """Validate opposing named padding sides against a three-axis shape."""
    normalized = normalize_default_padding(default_padding)
    if normalized is None:
        return
    shape_arr = np.asarray(shape)
    if shape_arr.shape != (3,):
        raise ValueError(f"mesh shape must have shape (3,), got {shape_arr.shape}")
    for side_a, side_b, axis, n_cells in zip(
        (normalized["west"], normalized["south"], normalized["bottom"]),
        (normalized["east"], normalized["north"], normalized["top"]),
        ("x", "y", "z"),
        shape_arr,
        strict=True,
    ):
        if side_a + side_b > int(n_cells):
            raise ValueError(
                f"default_padding {axis}-axis opposing sides exceed "
                f"the mesh shape ({int(n_cells)} cells)"
            )
