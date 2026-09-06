"""Assembling a whole CMB file.

Layout, in order::

    magic          8 bytes, CELLMODB
    data           every array's bytes, back to back
    header         UTF-8 JSON
    header length  uint64, little-endian
    magic          8 bytes again

A reader seeks to the end, reads the last 16 bytes to get the header length,
then jumps back to the header. Array offsets in the header are relative to
byte 8, immediately after the leading magic.

``format_version`` is stamped from ``WRITTEN_FORMAT_VERSION``; callers do not
supply it. Models arrive normalized as
``{name: {"metadata": {...}, "array": <ndarray>}}``.
"""

import json
import os
import struct

from cmb_format._codec import (
    MAGIC,
    WRITTEN_FORMAT_VERSION,
    resolve_reference_n_cells,
    serialize_array,
    serialize_mesh,
)

__all__ = ["build_file_bytes", "write_file"]


def _assemble(mesh: dict, models: dict | None, metadata: dict | None):
    """Build the header dict and the data buffer it describes.

    Key insertion order here is the header's key order on disk, so it is
    part of the format's bytes, not a detail.
    """
    models = models or {}
    buffer = bytearray()

    if mesh.get("mode") == "reference":
        # A reference-mode file declares its cell count rather than carrying
        # geometry; resolve it from the models when not given explicitly.
        mesh = {
            **mesh,
            "n_cells": resolve_reference_n_cells(mesh.get("n_cells"), models),
        }

    header = {
        "format_version": WRITTEN_FORMAT_VERSION,
        "mesh": serialize_mesh(mesh, buffer),
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

    Streams the data section straight out rather than concatenating it, so
    peak memory stays at roughly one copy of the arrays. Prefer this over
    `build_file_bytes` for anything large.

    Parameters
    ----------
    file_name : str or os.PathLike
        Path to write to. Overwritten if it exists.
    mesh : dict
        An embedded-mode mesh dict (``mode``, ``mesh_class``, ``arrays``,
        optionally ``base_mesh`` and ``default_padding``) or a
        reference-mode one (``{"mode": "reference"}``, optionally with
        ``n_cells`` or ``base_mesh``). See ``docs/binary-format.md``.
    models : dict, optional
        ``{name: {"metadata": {...}, "array": <ndarray>}}``.
    metadata : dict, optional
        Arbitrary file-level metadata.
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
    """Assemble a complete CMB file in memory.

    Same layout as `write_file`, returned rather than written -- for tests,
    round-tripping through a buffer, or handing bytes to something that
    isn't a file. Holds the whole file in memory; use `write_file` for
    anything large.
    """
    header, buffer = _assemble(mesh, models, metadata)
    blob = json.dumps(header).encode("utf-8")
    return MAGIC + bytes(buffer) + blob + struct.pack("<Q", len(blob)) + MAGIC
