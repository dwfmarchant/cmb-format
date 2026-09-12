# Cell Model Binary format (CMB) — v2 specification

CMB stores cell-based meshes and their models in `.cmb` files. This document
is the normative specification for CMB version 2, independent of any
implementation.

## Design goals

- **Portable:** raw binary arrays and a UTF-8 JSON header.
- **Selective reads:** explicit byte offsets allow readers to load individual
  arrays without reading the entire file.
- **Flexible metadata:** user-defined metadata at the file and model levels.
- **Bundling:** mesh geometry and multiple named models can share one file.
- **Efficient appends:** a trailing header allows new models to be added
  without rewriting existing array data.

Arrays are uncompressed.

## File layout

The JSON header follows the array data. Let `H` be `header_length` and
`D = file_length - 16 - H` be the header's absolute byte offset.

| Byte range (inclusive) | Content |
| --- | --- |
| `0..7` | Leading magic: 8 ASCII bytes, `CELLMODB` |
| `8..D-1` | Raw array data, back to back |
| `D..D+H-1` | UTF-8 JSON header, exactly `H` bytes |
| `D+H..D+H+7` | `header_length`: little-endian uint64 |
| `D+H+8..D+H+15` | Trailing magic: 8 ASCII bytes, `CELLMODB` |

To read the header:

1. Read the last 16 bytes of the file.
2. Verify that the last 8 bytes equal `CELLMODB`.
3. Interpret the preceding 8 bytes as `header_length` (little-endian uint64).
4. Seek to `file_length - 16 - header_length` and read exactly
   `header_length` bytes as UTF-8 JSON.

Every array's `offset` is relative to byte 8, the start of the data section.
The leading magic identifies the file; parsing uses the trailing magic and
header length. There is no padding or alignment between arrays, or between
the data section and the header.

### Appending and removing models

To append a model, read and verify the existing trailer to locate `D`.
Truncate the file at `D`, removing the old header and trailer. Write the
new model's raw bytes there; its array offset is `D - 8`. Then write a new
header containing the unchanged `mesh`, `metadata`, and existing `models`
entries plus the new model, followed by the new header length and trailing
magic. Existing array bytes remain unchanged.

This in-place operation is **not atomic**. A failure after truncation and
before the new header and trailer are complete can leave an unreadable
file. Per-array checksums do not provide crash recovery.

To remove a model, write a new file containing the raw array bytes to keep,
with updated offsets and a new header. The arrays can be copied without
decoding and re-encoding them. The format has no free-space bookkeeping
for in-place deletion.

### Byte order

All multibyte binary values, including array elements and the header length,
are **little-endian**. There is no per-file or per-array byte-order field.
Implementations on big-endian systems must convert byte order as needed.

## Header schema

The header is a single JSON object with `format_version` set to `2`:

```jsonc
{
  "format_version": 2,
  "mesh": { /* mesh descriptor, embedded or reference — see below */ },
  "metadata": {},                        // arbitrary user key/values, any JSON
  "models": {
    "resistivity": {
      "metadata": {"units": "ohm-m"},    // arbitrary user key/values
      "array": { /* array descriptor */ }
    }
  }
}
```

### Mesh descriptor

A mesh descriptor has two modes:

```jsonc
// mode "embedded": the mesh's own geometry arrays are stored in this file.
{
  "mode": "embedded",
  "mesh_class": "OctreeMesh",             // "TensorMesh" | "UniformTensorMesh" | "OctreeMesh"
  "arrays": { "...": { /* array descriptor, see below */ } },
  // optional visualization metadata belongs on base_mesh for OctreeMesh
  "base_mesh": { /* only for mesh_class == "OctreeMesh"; see below */ }
}

// mode "reference": models are paired with a separate embedded-mode mesh file.
{
  "mode": "reference",
  "n_cells": 12345,                       // required non-negative integer cell count
  "base_mesh": {                          // optional -- OctreeMesh reference only, see below
    "mesh_class": "UniformTensorMesh",
    "arrays": {...},
    "default_padding": {                 // optional shared padding
      "west": 0, "east": 0, "south": 0,
      "north": 0, "bottom": 0, "top": 0
    }
  }
}
```

In embedded mode, `mesh_class` identifies the geometry schema and `arrays`
contains its geometry array descriptors. An embedded octree also requires
`base_mesh` (see [Mesh classes](#mesh-classes)).

In reference mode, `n_cells` is required in addition to `mode`. It allows
readers to check model lengths without opening a separate mesh file.
`mesh_class` and `base_mesh` are absent by default. A reference descriptor
without `base_mesh` may include `default_padding`. The caller is responsible
for pairing the models with the correct mesh.

Reference mode is useful for octrees, whose per-leaf geometry can be large.
Tensor meshes may also use reference mode, although their smaller geometry
arrays usually make embedding practical.

An octree reference may include `base_mesh` to describe its domain and
finest cell size. Readers can compare it with the paired mesh's base grid,
but equal base grids do not establish that the octree refinements match.

### Default padding

`TensorMesh` and `UniformTensorMesh` descriptors support optional
visualization metadata through `default_padding`. It is stored as a complete
JSON object with six non-negative integer fields:

```json
{
  "west": 0,
  "east": 0,
  "south": 0,
  "north": 0,
  "bottom": 0,
  "top": 0
}
```

All six names are required, and unknown names are invalid. The fields have no
positional order. A missing `default_padding` field or explicit JSON `null`
means no default padding; an object containing six zeros remains an explicit
setting. Padding is not a binary array and does not alter the geometry array
key sets.

Where an axis shape is available, the two opposing padding values may sum
to that axis's cell count but may not exceed it. A bare reference descriptor
has no shape for this check.

### Array descriptor

Geometry and model arrays use the same descriptor:

```jsonc
{
  "dtype": "float64",
  "shape": [3],
  "offset": 0,
  "length": 24,
  "checksum": "sha256:af04c1..."  // digest abbreviated for illustration
}
```

| Field | Meaning |
| --- | --- |
| `dtype` | `float64`, `float32`, `int64`, `int32`, `int16`, or `int8` |
| `shape` | Non-negative integer dimensions: `[]` for a scalar or `[n]` for a one-dimensional array |
| `offset` | Non-negative integer byte offset relative to the start of the data section (byte 8) |
| `length` | Non-negative integer byte length, equal to `product(shape) * dtype_byte_width` |
| `checksum` | Required SHA-256 checksum of the raw stored array bytes |

`checksum` is the hexadecimal SHA-256 digest of the array's raw stored bytes
at data-section offsets `[offset, offset + length)`, prefixed with `sha256:`.
A reader verifies arrays it reads; it need not verify arrays it does not read.
Verification of one array does not require reading any other array.

Arrays are at most one-dimensional: `shape` has zero elements for a scalar
or one element for a one-dimensional array. Multidimensional arrays are not
supported. Cell ordering is described [below](#cell-numbering-ordering).

### Mesh classes

The geometry arrays for each embedded `mesh_class` are:

| Mesh class | Array | Shape | Dtype |
| --- | --- | --- | --- |
| `TensorMesh` | `origin` | `[3]` | `float64` |
| `TensorMesh` | `h_x`, `h_y`, `h_z` | `[nx]`, `[ny]`, `[nz]`, respectively | `float64` |
| `UniformTensorMesh` | `origin`, `cell_size` | `[3]` each | `float64` |
| `UniformTensorMesh` | `shape` | `[3]` | `int32` or `int64` |
| `OctreeMesh` | `level` | `[n_leaves]` | `int8` |
| `OctreeMesh` | `position` | `[n_leaves]` | `int32` or `int64` |

`int32` is the recommended default for `position` and `shape`. Both also
permit `int64`, and readers must support either type. Use `int64` where the
stored values exceed the `int32` range. `level` uses `int8` independently of
the position type.

For `OctreeMesh`, `base_mesh` describes the finest-resolution grid using
`{"mesh_class": "UniformTensorMesh", "arrays": {...}}`, with optional
`default_padding`. It has no `mode`, `metadata`, or `models` fields. Its
geometry is always stored inline, including when it appears in a reference
file. It is required for embedded octrees and optional for references.

<a id="cell-numbering-ordering"></a>

### Cell numbering / ordering

Coordinates increase eastward (`x`), northward (`y`), and upward (`z`).
`origin` is the bottom-south-west corner of the mesh.

**Tensor meshes:** `TensorMesh` and `UniformTensorMesh` number cells from
zero, advancing `x` fastest, then `y`, then `z`. For cell coordinates
`(i, j, k)` and axis cell counts `(nx, ny, nz)`, the linear index is
`i + nx * (j + ny * k)`. Axis-width arrays run in increasing coordinate order.

**Octree position encoding:** each `position` is the same x-fastest linear
index into the finest-resolution `base_mesh` grid. It identifies the base
cell at the leaf's bottom-south-west corner. A leaf at `level = 0` has the
base cell size; at level `l` it spans `2**l` base cells along each axis.
Position encoding is independent of the order in which leaves are stored.

**Octree leaf order:** `level` and `position` use **root-local Morton order**.
For base-grid shape `(nx, ny, nz)`, let `L = min(nx, ny, nz)`:

1. Partition the base grid into roots of `L` base cells along each axis.
2. Visit roots in x-fastest order, then y, then z, starting at the origin.
3. Within each root, recursively visit children in ascending order of
   `x_bit + 2*y_bit + 4*z_bit`. Each bit selects the lower (`0`) or upper
   (`1`) half along its axis. Visit all descendants of a child before
   advancing to the next child, emitting a cell when a leaf is reached.

Root-local traversal can differ from a single global Morton sort on a
rectangular base grid. All octree model arrays use the same leaf order as
`level` and `position`, including in reference-mode files.

Each octree base-grid dimension must be a positive power of two. Consequently
`L` is a power of two that divides each base-grid dimension, making the root
partition well-defined.

The Python I/O routines preserve supplied array order. Callers must supply
geometry and model values in the appropriate CMB order.

### Models

`models` maps user-chosen names to entries containing `metadata` and one
array descriptor. Each model is a one-dimensional array of length
`n_cells`, using the mesh's cell order. A file may have zero or more models.

## Validation a reader should perform

- Trailing `magic` matches exactly; reject otherwise.
- `format_version` is the integer `2`.
- A non-null `default_padding` value is an object with all six named fields,
  no unknown names, and non-negative integer values. Validate opposing counts
  against the axis shape where available (see [Default padding](#default-padding)).
- **Unknown keys are ignored.** Readers MUST tolerate unrecognized fields
  in the header, mesh descriptor, `base_mesh`, model entries, and array
  descriptors. This allows optional fields to be added without a version
  bump.
- Every array's `length` equals `product(shape) * dtype_byte_width`, and
  `offset + length` does not exceed the data section's actual size.
- Every array that is read has a `checksum` matching the SHA-256 of its raw
  stored bytes (see [Array descriptor](#array-descriptor)). Arrays that are
  not read may be left unverified.
- If `mode` is `"embedded"`, `mesh_class` is one of the three known
  values, and `arrays` contains exactly the keys that class requires
  (see table above) — no more, no fewer. Reference mode has no
  `mesh_class` to validate (see Mesh descriptor above).
- In reference mode, `n_cells` is a non-negative integer. An octree
  `base_mesh` has positive power-of-two dimensions on every axis.
- Every model's array is one-dimensional with length equal to the mesh's
  cell count, in **both** modes. In reference mode that count is the stated
  `n_cells`. In embedded mode it comes from the geometry: the product of the
  `h_x`/`h_y`/`h_z` lengths for `TensorMesh`, the product of the `shape`
  values for `UniformTensorMesh`, and the `level` length for `OctreeMesh`.
- When reading a reference-mode file with its paired embedded mesh, compare
  their base grids if the reference carries `base_mesh`. A match is only a
  consistency check; pairing the correct mesh and models remains the
  caller's responsibility.

Geometric validation, such as positive widths or cell sizes and octree leaf
alignment and tiling, is the receiving application's responsibility.

## Versioning

`format_version` is an integer counter, independent of package versions.
See the [format changelog](../FORMAT_CHANGELOG.md) for the format-only history.

A format-version bump is required when removing, renaming, or retyping a
required field; changing the byte layout, offsets, or trailer; changing
dtype tokens; or changing the meaning of an existing field.

A bump is not required for optional fields that readers may ignore, or for
user-defined entries in file or model `metadata`. Experimental metadata
belongs in those dictionaries.

### Relationship to implementation versions

Package releases and format versions are independent. Do not infer format
support from a package version. The Python implementation exposes it
explicitly:

```python
cmb_format.WRITTEN_FORMAT_VERSION  # the version write_file stamps
cmb_format.READABLE_FORMAT_VERSIONS  # the versions read_header accepts
```

Other implementations should expose equivalent information. A reader that
rejects an unsupported format version should report the versions it accepts.

## Implementations

- **`cmb_format`** is the Python reference implementation in this repository.
  It accepts dictionaries of NumPy arrays and depends on NumPy alone.
- **[`earthbasis-mesh`](https://github.com/dwfmarchant/earthbasis-mesh)** integrates
  the format with mesh and model objects through `earthbasis_mesh.io.cmb`.

`tests/goldens/v2/` contains committed reference files and parsed JSON
header sidecars. They test reading known files and preserve the Python
writer's exact serialization. A golden-file change requires review, but does
not necessarily imply a format change: JSON whitespace or key order can
change without changing the format's meaning.

## Legacy versions

### CMB v1

A v1 file sets `format_version` to `1`. When present and non-null,
`default_padding` is a six-element JSON list in this order:

```text
[west, east, south, north, bottom, top]
```

Each value is a non-negative integer. Named objects are not valid v1 padding.
Padding placement, missing or null padding, explicit all-zero padding, and
opposing-count limits have the same meaning as in v2. File framing, array
descriptors, payload bytes, checksums, geometry schemas, and cell ordering are
unchanged.

The Python reference implementation reads v1 and v2 and writes only v2.
When reading v1, it converts padding lists to named dictionaries before
applying shared validation. `read_header` returns these normalized fields
while retaining the stored `format_version` of `1`; its result is not a
verbatim copy of the stored JSON. Reading does not modify the file.

To convert a v1 file to v2, translate any padding lists to named objects, set
`format_version` to `2`, and rewrite the JSON header and its length. Geometry
and model array bytes, offsets, and checksums can be retained.

Historical reference files and their header sidecars are preserved in
`tests/goldens/v1/`. The fixture generator writes only the current v2 fixtures.
