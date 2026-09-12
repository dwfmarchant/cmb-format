"""Normalize stored CMB header padding and legacy header representations."""

from collections.abc import Mapping

from cmb_format._padding import (
    _PADDING_NAMES,
    _normalize_default_padding,
)


def _normalize_field(value, *, format_version: int, context: str):
    if value is None:
        return None
    if format_version == 1:
        if not isinstance(value, list) or len(value) != len(_PADDING_NAMES):
            raise ValueError(f"{context} must be a six-value list in CMB v1")
        value = dict(zip(_PADDING_NAMES, value, strict=True))
    elif format_version == 2:
        if not isinstance(value, Mapping):
            raise ValueError(f"{context} must be a named object in CMB v2")
        names = set(value)
        expected = set(_PADDING_NAMES)
        missing = sorted(expected - names)
        unknown = sorted(names - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing names {missing}")
            if unknown:
                details.append(f"unknown names {unknown}")
            raise ValueError(f"{context} has invalid names: {', '.join(details)}")
    else:  # pragma: no cover - read_header checks the version first.
        raise ValueError(f"unsupported CMB format version: {format_version}")
    try:
        return _normalize_default_padding(value)
    except ValueError as exc:
        raise ValueError(f"invalid {context}: {exc}") from exc


def _padding_owner(mesh: dict) -> dict:
    base = mesh.get("base_mesh")
    if isinstance(base, dict) and (
        mesh.get("mode") == "reference" or mesh.get("mesh_class") == "OctreeMesh"
    ):
        return base
    return mesh


def normalize_header_padding(header: dict) -> None:
    """Normalize recognized padding and expose v1 headers in v2 form.

    The recognized owner is the mesh descriptor for tensor, uniform, and bare
    reference meshes, and the base descriptor for meshes that include one.
    """
    mesh = header["mesh"]
    owner = _padding_owner(mesh)
    if "default_padding" in owner:
        context = (
            "base_mesh.default_padding" if owner is not mesh else "mesh.default_padding"
        )
        owner["default_padding"] = _normalize_field(
            owner["default_padding"],
            format_version=header["format_version"],
            context=context,
        )
    if header["format_version"] == 1:
        # Keep this migration target literal and independent of future writer bumps.
        header["format_version"] = 2
