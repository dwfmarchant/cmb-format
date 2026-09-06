# Cell Model Binary format (CMB) — v1 specification

CMB stores cell-based meshes and their models in `.cmb` files. This document
is the normative v1 specification, independent of any implementation.

## Design goals, and why

- **Portable:** raw binary arrays and a UTF-8 JSON header.
- **Selective reads:** explicit byte offsets allow readers to load individual
  arrays without reading the entire file.
- **Flexible metadata:** user-defined metadata at the file and model levels.
- **Bundling:** mesh geometry and multiple named models can share one file.
- **Efficient appends:** a trailing header allows new models to be added
  without rewriting existing array data.

Arrays are uncompressed in v1.

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
decoding and re-encoding them; v1 has no free-space bookkeeping for in-place
deletion.

### Byte order

All multibyte binary values, including array elements and the header length,
are **little-endian**. There is no per-file or per-array byte-order field.
Implementations on big-endian systems must convert byte order as needed.

## Header schema

The header is a single JSON object:

```jsonc
{
  "format_version": 1,
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
  "n_cells": 12345,                       // required cell count
  "base_mesh": {                          // optional -- OctreeMesh reference only, see below
    "mesh_class": "UniformTensorMesh",
    "arrays": {...},
    "default_padding": [0, 0, 0, 0, 0, 0] // optional shared padding
  }
}
```

In embedded mode, `mesh_class` identifies the geometry schema and `arrays`
contains its geometry array descriptors. An embedded octree also requires
`base_mesh` (see [Mesh classes](#mesh-classes)).

In reference mode, `n_cells` is required in addition to `mode`. It allows
readers to check model lengths without opening a separate mesh file.
`mesh_class` and `base_mesh` are absent by default. The caller is responsible
for pairing the models with the correct mesh.

Reference mode is useful for octrees, whose per-leaf geometry can be large.
Tensor meshes may also use reference mode, although their smaller geometry
arrays usually make embedding practical.

An octree reference may include `base_mesh` to describe its domain and
finest cell size. Readers can compare it with the paired mesh's base grid,
but equal base grids do not establish that the octree refinements match.

### Default padding

`default_padding` is optional visualization metadata, stored as a JSON list
of six non-negative integers in the order
`[west, east, south, north, bottom, top]`, also written
`[x-, x+, y-, y+, z-, z+]`. Missing or explicit JSON `null` means no default
padding. It is not a binary array and does not alter the geometry array
key sets.

For an embedded octree, padding belongs to the nested `base_mesh`; the
outer descriptor does not repeat it. A reference descriptor with a
`base_mesh` uses the same placement. For compatibility, readers/writers may
accept an outer-only field and move it to the nested base. If both fields
are supplied, they must normalize to the same six values or the descriptor
is rejected. A bare reference without `base_mesh` may retain its outer
padding field.

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
| `shape` | Integer dimensions: `[]` for a scalar or `[n]` for a one-dimensional array |
| `offset` | Byte offset relative to the start of the data section (byte 8) |
| `length` | Byte length, equal to `product(shape) * dtype_byte_width` |
| `checksum` | Required SHA-256 checksum of the raw stored array bytes |

`checksum` is the hexadecimal SHA-256 digest of the array's raw stored bytes
at data-section offsets `[offset, offset + length)`, prefixed with `sha256:`.
A reader verifies arrays it reads; it need not verify arrays it does not read.
Verification of one array does not require reading any other array.

All arrays in v1 are at most one-dimensional: `shape` has zero elements for
a scalar or one element for a one-dimensional array. Multidimensional
arrays are outside v1. Cell ordering is described
[below](#cell-numbering-ordering).

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

Per-cell arrays (`level`/`position` and every model array) use the same cell order as
everywhere else in this ecosystem — index 0 is the bottom-south-west
cell, advancing `x` fastest, then `y`, then `z` slowest. This format does
not re-derive or re-check that ordering; it stores whatever order the
writer's own cell indexing already uses.

### Models

`models` maps user-chosen names to entries containing `metadata` and one
array descriptor. Each model is a one-dimensional array of length
`n_cells`, using the mesh's cell order. A file may have zero or more models.

## Validation a reader should perform

- Trailing `magic` matches exactly; reject otherwise.
- `format_version` is a version the reader understands.
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
  (see table above) — no more, no fewer. If `mode` is `"reference"`,
  every model's array length equals the stated `n_cells` (there is no
  `mesh_class` to validate in this mode — see Mesh descriptor above).
- When reading a reference-mode file with its paired embedded mesh, compare
  their base grids if the reference carries `base_mesh`. A match is only a
  consistency check; pairing the correct mesh and models remains the
  caller's responsibility.

Geometric validation, such as positive widths or cell sizes and octree leaf
alignment and tiling, is the receiving application's responsibility.

## Versioning

`format_version` is an integer counter, independent of package versions.

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
- **[`subcrop-mesh`](https://github.com/dwfmarchant/subcrop-mesh)** integrates
  the format with mesh and model objects through `subcrop_mesh.io.cmb`.

`tests/goldens/` contains committed reference files and parsed JSON header
sidecars. They test reading known files and preserve the Python writer's
exact serialization. A golden-file change requires review, but does not
necessarily imply a format change: JSON whitespace or key order can change
without changing the format's meaning.
