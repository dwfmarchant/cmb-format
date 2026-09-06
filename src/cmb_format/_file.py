"""Assemble CMB files from mesh and model array dictionaries.

Writers stamp ``WRITTEN_FORMAT_VERSION`` and buffer the raw array data before
writing the header and trailer. See ``docs/binary-format.md`` for the layout.
"""

import json
import os
import struct

import numpy as np

from cmb_format._codec import (
    MAGIC,
    WRITTEN_FORMAT_VERSION,
    _validate_raw_mesh,
    resolve_reference_n_cells,
    serialize_array,
    serialize_mesh,
)

__all__ = ["build_file_bytes", "write_file"]


def _check_model_lengths(models: dict, n_cells: int) -> None:
    """Raise if any model is not one value per cell."""
    for name, entry in models.items():
        if not isinstance(entry, dict) or "array" not in entry:
            raise ValueError(f"model {name!r} must be an object with an 'array' field")
        arr = np.asarray(entry["array"])
        if arr.ndim != 1:
            raise ValueError(
                f"model {name!r} must be a 1D array, got shape {arr.shape}"
            )
        length = arr.shape[0]
        if length != n_cells:
            raise ValueError(
                f"model {name!r} has {length} values, but the mesh has "
                f"{n_cells} cells; models are one value per cell"
            )


def _assemble(mesh: dict, models: dict | None, metadata: dict | None):
    """Build the header dictionary and its array-data buffer.

    Insertion order preserves the Python writer's exact serialization, which
    is checked by golden files. The format does not require this JSON key order.
    """
    if not isinstance(mesh, dict):
        raise ValueError("mesh must be a mapping")
    if models is None:
        models = {}
    elif not isinstance(models, dict):
        raise ValueError("models must be a mapping")
    buffer = bytearray()

    if mesh.get("mode") == "reference":
        # Derive the count from model lengths and cross-check any supplied count.
        mesh = {
            **mesh,
            "n_cells": resolve_reference_n_cells(mesh.get("n_cells"), models),
        }

    # Validate geometry and model cardinality before mutating the data buffer.
    n_cells, _ = _validate_raw_mesh(mesh)
    _check_model_lengths(models, n_cells)
    mesh_header = serialize_mesh(mesh, buffer)

    header = {
        "format_version": WRITTEN_FORMAT_VERSION,
        "mesh": mesh_header,
        "metadata": metadata or {},
        "models": {
            name: {
                "metadata": entry.get("metadata", {}),
                "array": serialize_array(entry["array"], buffer),
            }
            for name, entry in models.items()
        },
    }
    return header, buffer


def write_file(
    file_name: str | os.PathLike,
    mesh: dict,
    models: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """Write a complete CMB file.

    Buffers the full data section in memory, then writes the file's sections
    separately. This avoids the additional complete-file byte string assembled
    by `build_file_bytes`.

    Parameters
    ----------
    file_name : str or os.PathLike
        Output path. Overwritten if it exists.
    mesh : dict
        Mesh description containing raw geometry arrays for embedded mode,
        or ``{"mode": "reference"}`` for models stored separately. Embedded
        octrees require ``base_mesh``. See ``docs/binary-format.md`` for the
        descriptor fields.
    models : dict, optional
        ``{name: {"metadata": {...}, "array": <ndarray>}}``. In reference mode,
        model lengths determine ``n_cells`` and must match any supplied count.
        If there are no models, the reference descriptor must supply ``n_cells``.
        Named model entries are serialized in caller insertion order.
    metadata : dict, optional
        File-level metadata.
    """
    header, buffer = _assemble(mesh, models, metadata)
    blob = json.dumps(header).encode("utf-8")
    with open(file_name, "wb") as f:
        f.write(MAGIC)
        f.write(buffer)
        f.write(blob)
        f.write(struct.pack("<Q", len(blob)))
        f.write(MAGIC)


def build_file_bytes(
    mesh: dict,
    models: dict | None = None,
    metadata: dict | None = None,
) -> bytes:
    """Return a complete CMB file as bytes.

    Accepts the same mesh, model, and metadata dictionaries as `write_file`.
    Holds both the array-data buffer and assembled file in memory; `write_file`
    avoids assembling the complete-file byte string.
    """
    header, buffer = _assemble(mesh, models, metadata)
    blob = json.dumps(header).encode("utf-8")
    return MAGIC + bytes(buffer) + blob + struct.pack("<Q", len(blob)) + MAGIC
