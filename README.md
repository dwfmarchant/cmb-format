# cmb-format

**CMB (Cell Model Binary)** is a binary file format and Python I/O library
for storing UBC GIF–style tensor and octree meshes and their associated
models. It provides an alternative to the ASCII mesh and model files used
by UBC GIF software. CMB supports uniform and variable-spacing tensor
meshes, octree meshes, multiple named models, and file- and model-level
metadata.

CMB stores mesh geometry and model values as typed arrays, with metadata in
JSON, so consumers can avoid parsing millions of numbers from text. Large
octree consumers that need only arrays can also avoid allocating a full
consumer mesh; see the [discretize round trips and benchmarks](https://github.com/dwfmarchant/cmb-format/blob/main/docs/discretize.md).

## Capabilities

- Store mesh geometry and per-cell model arrays together in a `.cmb` file.
- Store models separately from geometry to avoid duplicating large meshes.
- Read selected model payloads without loading unrequested models.
- Inspect model metadata and file contents without loading model payloads.
- Verify each array's integrity with a SHA-256 checksum.

## Installation

Requires Python 3.11 or newer. NumPy is the only runtime dependency.
From a local checkout:

```bash
python -m pip install .
```

## Usage

The API accepts dictionaries of NumPy arrays describing meshes and models.
CMB uses a different cell ordering from UBC GIF; these routines do not
convert between the two. See
[Cell numbering / ordering](https://github.com/dwfmarchant/cmb-format/blob/main/docs/binary-format.md#cell-numbering-ordering)
in the format specification.

Write a four-cell tensor mesh and a resistivity model, then read them back:

```python
import numpy as np

import cmb_format as cmb

mesh = {
    "mode": "embedded",
    "mesh_class": "TensorMesh",
    "arrays": {
        "origin": np.zeros(3),
        "h_x": np.array([1.0, 2.0]),
        "h_y": np.array([1.0, 1.0]),
        "h_z": np.array([3.0]),
    },
}
models = {
    "rho": {
        "metadata": {"units": "ohm-m"},
        "array": np.array([10.0, 20.0, 30.0, 40.0]),
    }
}
cmb.write_file("example.cmb", mesh, models)

mesh, models, metadata = cmb.read_file("example.cmb")
rho = models["rho"]["array"]

# Load geometry and only the requested model. File order is preserved.
mesh, models, metadata = cmb.read_file("example.cmb", models=["rho"])
```

`read_file` loads and checksum-verifies all geometry and selected model arrays,
including nested base-mesh geometry. Its three results match `write_file`'s
`mesh`, `models`, and `metadata` parameters, so passing them straight back
preserves the mesh geometry, model arrays, and metadata. Valid padding is
returned as integer lists, with shared octree/reference padding on the nested
base descriptor. The NumPy arrays are read-only; use `.copy()` if you need to
modify them. `models=None` loads every model, `models=[]` loads none, and
duplicate selections collapse in stored file order.

For inexpensive inspection, use the raw header summaries:

```python
model_summaries = cmb.list_models("example.cmb")
contents = cmb.read_contents("example.cmb")
```

`list_models` returns mappings such as
`{"rho": {"metadata": {"units": "ohm-m"}, "dtype": "float64", "shape": [4]}}`
and reads no array payloads. `read_contents` returns
`{"has_mesh": bool, "mesh_type": str | None, "has_base_mesh": bool,
"n_cells": int, "models": dict}`; it reads only an embedded uniform mesh's
three-value shape array to compute `n_cells`. Nested base-mesh and model
payloads remain unread. Both helpers preserve model metadata, dtype, shape, and
stored order.

To read individual arrays without loading the whole file:

```python
with open("example.cmb", "rb") as f:
    header, data_start = cmb.read_header(f)
    metadata = header["metadata"]
    geometry = cmb.read_arrays(f, data_start, header["mesh"]["arrays"])
    rho = cmb.read_array(f, data_start, header["models"]["rho"]["array"])
```

`read_header` accepts the keyword `read_shape_payload`. Its default `True`
performs full header validation, including the embedded uniform mesh shape
checksum. Set it to `False` for structural header checks without reading shape
payloads; shape values and dependent uniform padding or model-count checks are
deferred.

For measured large-octree and tensor round trips and timing methodology, see
[the discretize interoperability notes](https://github.com/dwfmarchant/cmb-format/blob/main/docs/discretize.md). On the measured
2.18-million-leaf sample, the generated CMB file is 10.4 MiB versus 28.8 MiB
for UBC, and conversion plus CMB writing is about 21× faster.

## Development

From a local checkout, with pip 25.1 or newer:

```bash
python -m pip install --group dev -e .
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Committed reference files in `tests/goldens/` test compatibility with the
binary format alongside round-trip tests.

The [format specification](https://github.com/dwfmarchant/cmb-format/blob/main/docs/binary-format.md) defines the file layout
and mesh schemas. Package and format versions are independent; see
[versioning](https://github.com/dwfmarchant/cmb-format/blob/main/docs/binary-format.md#versioning) and the
[package changelog](https://github.com/dwfmarchant/cmb-format/blob/main/CHANGELOG.md).
