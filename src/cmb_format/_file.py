"""Read and write CMB files as mesh and model array dictionaries.

Writers stamp ``WRITTEN_FORMAT_VERSION`` and buffer the raw array data before
writing the header and trailer. See ``docs/binary-format.md`` for the layout.
"""

import json
import math
import os
import struct
from collections.abc import Sequence

import numpy as np

from cmb_format._codec import (
    MAGIC,
    WRITTEN_FORMAT_VERSION,
    _nonnegative_integer,
    _serialize_array,
    _serialize_mesh,
    _validate_raw_mesh,
    read_array,
    read_arrays,
    read_header,
    summarize_models,
    validate_model_lengths,
)
from cmb_format._compat import _padding_owner
from cmb_format._padding import (
    _validate_normalized_padding_shape,
    normalize_integer_array,
)

__all__ = [
    "build_file_bytes",
    "list_models",
    "read_contents",
    "read_file",
    "write_file",
]


def _select_model_names(
    model_headers: dict, requested: Sequence[str] | None
) -> list[str]:
    """Resolve a model selection in stored header order."""
    if requested is None:
        return list(model_headers)
    if isinstance(requested, str):
        raise TypeError(
            "models must be a sequence of names, not a single string -- "
            f"pass [{requested!r}] to read only that model"
        )
    requested_names = list(requested)
    unknown = [name for name in requested_names if name not in model_headers]
    if unknown:
        have = ", ".join(repr(name) for name in model_headers) or "no models"
        raise ValueError(
            f"unknown model(s): {', '.join(repr(name) for name in unknown)}; "
            f"file has {have}"
        )
    wanted = set(requested_names)
    return [name for name in model_headers if name in wanted]


def read_file(
    file_name: str | os.PathLike,
    *,
    models: Sequence[str] | None = None,
) -> tuple[dict, dict, dict]:
    """Read a CMB file into mesh, selected model, and metadata dictionaries.

    The three results match ``write_file``'s ``mesh``, ``models``, and
    ``metadata`` parameters, so passing them straight back preserves the mesh
    geometry, model arrays, and metadata. ``models=None`` reads all model
    payloads. A sequence reads only the named payloads, in file order; duplicate
    names collapse and an empty sequence skips every model payload. A bare string
    and unknown names raise an exception before any selected model is read.

    Parameters
    ----------
    file_name : str or os.PathLike
        Input path.
    models : sequence of str, optional
        Model payloads to load. Geometry and any nested base-mesh arrays are
        always loaded.

    Returns
    -------
    mesh : dict
        Mesh descriptor. For an embedded mesh, and for any ``base_mesh``,
        the geometry ``arrays`` are loaded as read-only NumPy arrays. A
        reference descriptor includes its ``n_cells``; no external mesh is
        loaded, and any unrecognized ``arrays`` key it carries is passed
        through unconverted. Recognized padding on tensor and uniform meshes,
        bare references, and ``base_mesh`` descriptors is returned as a complete
        dictionary of Python integers; an explicit ``null`` there is omitted.
        Unrecognized descriptor fields pass through unchanged on reads and are
        ignored by writers.
    models : dict
        ``{name: {"metadata": {...}, "array": <ndarray>}}`` with read-only
        NumPy arrays and stored model metadata. Empty if no model was selected.
        Opaque fields in selected model entries are preserved.
    metadata : dict
        File-level metadata. Empty if the file records none.

    Notes
    -----
    Validates the header and checksum-verifies all geometry and selected model
    arrays. Stored geometry and model order is preserved; no consumer mesh
    ordering is applied. The file is closed before returning.
    """
    with open(file_name, "rb") as f:
        header, data_start = read_header(f)
        model_headers = header.get("models", {})
        selected = _select_model_names(model_headers, models)
        mesh = header["mesh"]
        if mesh["mode"] == "embedded":
            mesh["arrays"] = read_arrays(f, data_start, mesh["arrays"])
        if "base_mesh" in mesh:
            base = mesh["base_mesh"]
            base["arrays"] = read_arrays(f, data_start, base["arrays"])
        owner = _padding_owner(mesh)
        if owner.get("default_padding") is None:
            owner.pop("default_padding", None)
        model_entries = {
            name: {
                **model_headers[name],
                "metadata": model_headers[name].get("metadata", {}),
                "array": read_array(f, data_start, model_headers[name]["array"]),
            }
            for name in selected
        }
    return mesh, model_entries, header.get("metadata", {})


def list_models(file_name: str | os.PathLike) -> dict:
    """List model metadata and shapes without reading any array payloads.

    The returned mapping is ``{name: {"metadata": {...}, "dtype": "float64",
    "shape": [n]}}`` in stored model order. Header descriptors and payload
    bounds are validated, but
    no array bytes are read or checksum-verified. In particular, a corrupt model
    or geometry payload does not affect this inspection result.
    """
    with open(file_name, "rb") as f:
        header, _ = read_header(f, read_shape_payload=False)
    return summarize_models(header)


def read_contents(file_name: str | os.PathLike) -> dict:
    """Summarize a CMB file without loading model payloads.

    Returns a mapping with ``has_mesh`` (bool), ``mesh_type`` (the embedded
    mesh class or ``None``), ``has_base_mesh`` (bool), ``n_cells`` (int), and
    ``models`` (the same mapping returned by :func:`list_models`). Header
    descriptors, payload bounds, and scalar padding syntax are validated. The
    only payload read is the top-level ``shape`` array of an embedded
    ``UniformTensorMesh`` (three values needed to compute ``n_cells``); nested
    base-mesh shape arrays and all model payloads remain untouched.
    """
    with open(file_name, "rb") as f:
        header, data_start = read_header(f, read_shape_payload=False)
        mesh_header = header["mesh"]
        mode = mesh_header.get("mode")
        if mode == "embedded":
            mesh_type = mesh_header["mesh_class"]
            arrays = mesh_header["arrays"]
            if mesh_type == "TensorMesh":
                n_cells = math.prod(
                    arrays[name]["shape"][0] for name in ("h_x", "h_y", "h_z")
                )
            elif mesh_type == "OctreeMesh":
                n_cells = arrays["level"]["shape"][0]
            elif mesh_type == "UniformTensorMesh":
                shape_values = read_array(f, data_start, arrays["shape"])
                shape = normalize_integer_array(
                    shape_values,
                    name="UniformTensorMesh arrays['shape']",
                    shape=(3,),
                    minimum=1,
                    value_description="three positive integer values",
                )
                if mesh_header.get("default_padding") is not None:
                    _validate_normalized_padding_shape(
                        mesh_header["default_padding"],
                        tuple(int(value) for value in shape),
                    )
                n_cells = math.prod(int(value) for value in shape)
            else:  # pragma: no cover - read_header validates this first.
                raise ValueError(f"unknown mesh_class: {mesh_type!r}")
            has_mesh = True
        elif mode == "reference":
            has_mesh = False
            mesh_type = None
            n_cells = mesh_header["n_cells"]
        else:  # pragma: no cover - read_header validates this first.
            raise ValueError(f"unsupported mesh mode for reading: {mode!r}")

        # Uniform shape values were deferred by
        # read_header(..., read_shape_payload=False).
        # Once available, retain the full-read model/cardinality validation.
        validate_model_lengths(header.get("models", {}), n_cells)

    return {
        "has_mesh": has_mesh,
        "mesh_type": mesh_type,
        "has_base_mesh": "base_mesh" in mesh_header,
        "n_cells": n_cells,
        "models": summarize_models(header),
    }


def _resolve_model_count(mesh: dict, models: dict, expected: int | None) -> int:
    """Validate raw model cardinality and return the mesh cell count."""
    supplied = mesh.get("n_cells") if mesh.get("mode") == "reference" else None
    if supplied is not None:
        supplied = _nonnegative_integer(supplied, name="reference n_cells")
    lengths = {}
    for name, entry in models.items():
        if not isinstance(entry, dict) or "array" not in entry:
            raise ValueError(f"model {name!r} must be an object with an 'array' field")
        arr = np.asarray(entry["array"])
        if arr.ndim != 1:
            raise ValueError(
                f"model {name!r} must be a 1D array, got shape {arr.shape}"
            )
        lengths[name] = arr.shape[0]
    if expected is not None:
        for name, length in lengths.items():
            if length != expected:
                raise ValueError(
                    f"model {name!r} has {length} values, but the mesh has "
                    f"{expected} cells; models are one value per cell"
                )
        return expected
    if not lengths:
        if supplied is None:
            raise ValueError(
                "a reference-mode file needs n_cells or at least one model "
                "to determine n_cells"
            )
        return supplied
    distinct = set(lengths.values())
    if len(distinct) > 1:
        raise ValueError(f"models disagree on cell count: {lengths}")
    (n_cells,) = distinct
    if supplied is not None and n_cells != supplied:
        raise ValueError(
            f"mesh has n_cells={supplied}, but models have {n_cells} cells"
        )
    return n_cells


def _assemble(mesh: dict, models: dict | None, metadata: dict | None):
    """Build the header dictionary and its array-data buffer."""
    if not isinstance(mesh, dict):
        raise ValueError("mesh must be a mapping")
    if models is None:
        models = {}
    elif not isinstance(models, dict):
        raise ValueError("models must be a mapping")

    if mesh.get("mode") == "reference":
        n_cells = _resolve_model_count(mesh, models, None)
        mesh = {**mesh, "n_cells": n_cells}
        validation = _validate_raw_mesh(mesh)
    else:
        validation = _validate_raw_mesh(mesh)
        _resolve_model_count(mesh, models, validation[0])

    buffer = bytearray()
    mesh_header = _serialize_mesh(mesh, buffer, padding=validation[1])
    header = {
        "format_version": WRITTEN_FORMAT_VERSION,
        "mesh": mesh_header,
        "metadata": metadata or {},
        "models": {
            name: {
                "metadata": entry.get("metadata", {}),
                "array": _serialize_array(entry["array"], buffer),
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
