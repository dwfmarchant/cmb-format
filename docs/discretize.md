# Interoperating with `discretize`

To run this demo, install `discretize` separately:

```bash
python -m pip install discretize
```

Replace `large_octree_mesh.msh` with a local UBC octree file; the benchmark
dataset is not bundled with this repository. The conversion assumes an octree
with a uniform base grid. A CMB octree stores each leaf as one `level` and
`position` value; `position` is a Fortran-order, x-fastest index into the
base-grid shape. Model arrays stay in that same leaf row order. The mesh and
model values are typed arrays; file metadata is JSON header data, not a
checksummed array payload.

## Complete `TreeMesh` ↔ CMB dictionary demo

```python
import numpy as np
import cmb_format as cmb
from discretize import TreeMesh


# discretize.TreeMesh -> CMB dictionary. The base grid must be uniform.
dis_mesh = TreeMesh.read_UBC("large_octree_mesh.msh")
if any(not np.allclose(axis, axis[0]) for axis in dis_mesh.h):
    raise ValueError("CMB UniformTensorMesh bases require uniform TreeMesh axes")

centers, tree_levels = dis_mesh.__getstate__()
# Keep level at int64 for the bit shift; narrow only the stored CMB array.
level = (dis_mesh.max_level - tree_levels).astype(np.int64)
ijk = (centers - (1 << level[:, None])) // 2
shape = np.array([len(axis) for axis in dis_mesh.h], dtype=np.int64)
position = np.ravel_multi_index(ijk.T, tuple(shape), order="F")
if position.size == 0 or position.max() <= np.iinfo(np.int32).max:
    position = position.astype(np.int32)

cmb_mesh_data = {
    "mode": "embedded",
    "mesh_class": "OctreeMesh",
    "arrays": {"level": level.astype(np.int8), "position": position},
    "base_mesh": {
        "mesh_class": "UniformTensorMesh",
        "arrays": {
            "origin": dis_mesh.origin.copy(),
            "cell_size": np.array([axis[0] for axis in dis_mesh.h], dtype=np.float64),
            "shape": shape,
        },
    },
}

cmb.write_file("octree_mesh.cmb", cmb_mesh_data)

# Read the file just written and reconstruct the same mesh.
mesh_dict, _, _ = cmb.read_file("octree_mesh.cmb")
base = mesh_dict["base_mesh"]
shape = tuple(int(value) for value in base["arrays"]["shape"])
cell_size = np.asarray(base["arrays"]["cell_size"], dtype=np.float64)
tree = TreeMesh(
    [np.full(n, h) for n, h in zip(shape, cell_size)],
    origin=base["arrays"]["origin"],
    diagonal_balance=False,
)
ijk = np.column_stack(
    np.unravel_index(mesh_dict["arrays"]["position"], shape, order="F")
)
level = mesh_dict["arrays"]["level"].astype(np.int64)
tree.__setstate__((2 * ijk + (1 << level[:, None]), tree.max_level - level))
assert dis_mesh.equals(tree)
```

## TensorMesh with named models

To run this example, install `discretize` separately as shown above. Replace
the `Tensor/fine.*` paths with local UBC tensor mesh and model files; those
inputs are not bundled with this repository. `read_model_UBC` handles UBC's
model ordering, and the returned arrays are passed unchanged to CMB. The
dictionary keeps each model name and its array aligned with the tensor mesh.

```python
import numpy as np
import cmb_format as cmb
from discretize import TensorMesh


dis_tensor = TensorMesh.read_UBC("Tensor/fine.msh")
con_model = dis_tensor.read_model_UBC("Tensor/fine.con")
chg_model = dis_tensor.read_model_UBC("Tensor/fine.chg")

cmb_tensor_mesh_data = {
    "mode": "embedded",
    "mesh_class": "TensorMesh",
    "arrays": {
        "h_x": dis_tensor.h[0],
        "h_y": dis_tensor.h[1],
        "h_z": dis_tensor.h[2],
        "origin": dis_tensor.origin,
    },
}
cmb_models = {
    "con": {"metadata": {"units": "S/m"}, "array": con_model},
    "chg": {"metadata": {"units": "mV/V"}, "array": chg_model},
}
cmb.write_file("fine.cmb", cmb_tensor_mesh_data, cmb_models)

mesh_dict, model_dict, _ = cmb.read_file("fine.cmb")
tensor = TensorMesh(
    h=[mesh_dict["arrays"][name] for name in ("h_x", "h_y", "h_z")],
    origin=mesh_dict["arrays"]["origin"],
)
np.testing.assert_array_equal(model_dict["con"]["array"], con_model)
np.testing.assert_array_equal(model_dict["chg"]["array"], chg_model)
assert dis_tensor.equals(tensor)
```

## TensorMesh timings

The measurements below use a 825 × 951 × 52 tensor mesh with 40,797,900
cells and two float64 model arrays (`con` and `chg`). These are equivalent
full-load scopes; the CMB timing includes checksums, `TensorMesh`
reconstruction, and both model arrays.

| Path | Operation | Median wall time |
| --- | --- | ---: |
| UBC | Read the text mesh and both text models | 5.07 s |
| CMB | Read, checksum, reconstruct `TensorMesh`, and read both models | 218 ms |

The reconstructed mesh equals the UBC mesh and both model arrays compare
exactly. The CMB full load is about 23.3× faster; the benefit comes mainly from
the large model arrays, since the UBC mesh geometry itself is only 9,233 bytes.
The UBC inputs total 909,064,079 bytes (867.0 MiB); the generated CMB file is
652,782,280 bytes (622.5 MiB).

Each value is the median of three wall-clock runs after one warmup, with a warm
OS cache and without `fsync`. The environment was Python 3.12.14, NumPy 2.5.3,
SciPy 1.18.1, `discretize` 0.12.0, and macOS 26.5.2 arm64. Results vary with
the local machine and environment.

## Large-octree timings

The measurements below use the same 2,180,697-leaf octree with a
1024 × 1024 × 1024 base grid and geometry only (no model arrays). Array-only
reads avoid building the tree hierarchy; full-tree timings include
reconstruction.

| Path | Operation | Median wall time |
| --- | --- | ---: |
| UBC | Read the text mesh and construct `TreeMesh` | 11.45 s |
| CMB | Read and checksum arrays into dictionaries | 4.82 ms |
| CMB | Read arrays and reconstruct `TreeMesh` | 6.62 s |

The CMB path reconstructs the same tree (`dis_mesh.equals(tree)` is true), so
the full-tree load is about 1.73× faster. Conversion plus CMB writing is about
21× faster than UBC writing:

| Path | Operation | Median wall time |
| --- | --- | ---: |
| UBC | Write from `TreeMesh` | 0.980 s |
| CMB | Convert `TreeMesh` to a dictionary | 37.1 ms |
| CMB | Write a prebuilt dictionary | 7.68 ms |
| CMB | Convert and write from `TreeMesh` | 46.9 ms |

The UBC source is 30,152,484 bytes (28.76 MiB); the generated CMB file is
10,904,609 bytes (10.40 MiB), about 2.77× smaller. The CMB file uses int8
levels, int32 positions, an int64 base shape, and no compression.

Each value is the median of three wall-clock runs after one warmup, with a warm
OS cache and without `fsync`. The environment was Python 3.12.14, NumPy 2.5.3,
SciPy 1.18.1, `discretize` 0.12.0, and macOS 26.5.2 arm64. Results vary with
the local machine and environment.
