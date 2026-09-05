# Cell Model Binary format (CMB) — v1 specification

A cross-language file format for cell-based meshes and the models defined
on them. This document is normative: it describes the *format*, not any
particular implementation of it, and a reader/writer can be built in any
language from this document alone. See the note at the end for the
implementations that exist.

Files use the **`.cmb`** extension.

## Design goals, and why

- **Portable** — readable with nothing more exotic than "read bytes from
  a file" and "parse JSON," available in essentially every language.
  Deliberately **not** a zip/archive-based format (unlike `.npz`): a zip
  reader is an extra dependency in every target language, whereas raw
  file I/O plus JSON is universal. Modeled on the same pattern as
  `glTF`'s `.glb` and `safetensors`: a JSON header describing byte
  offsets into a following flat binary data section.
- **Fast** — arrays are stored as raw, uncompressed binary, not ASCII
  text. No parsing cost proportional to array size; a reader can `seek`
  directly to an array's offset.
- **Flexible metadata** — arbitrary user key/values at the file level and
  per-model, unlike UBC's fixed-field text formats.
- **Bundling** — a mesh's geometry and one or more named models live in a
  single file, rather than separate mesh/model files.
- **No compression, by design, for v1** — compression is a legitimate
  future addition (see Future wishlist), but it's a strict opt-in on top
  of this base format, not part of it: it would require every language
  implementation to also link a compression library, working against the
  portability goal. Left out until there's a concrete need for it.
- **Cheap to append a model to an existing file** — adding a model
  shouldn't require rewriting a mesh's entire (potentially huge, for a
  large octree) existing data. See File layout: the header sits at the
  *end* of the file for exactly this reason.

## File layout

The header is at the **end** of the file, not the start — the same
pattern ZIP and Parquet use, and for the same reason: it lets a writer
append new data (e.g. a new model) without touching anything already
written, since nothing before the header needs to move.

```
byte 0..7          magic:          8 bytes, ASCII, always "CELLMODB" (identification only; not required to parse the file — see below)
byte 8..D-1        data:           raw array bytes, back to back (D = total file length - 16 - header_length)
byte D..D+H-1      header:         UTF-8 JSON, exactly H = header_length bytes
byte D+H..D+H+7    header_length:  uint64, little-endian (= H)
byte D+H+8..D+H+15 magic:          8 bytes, ASCII, "CELLMODB" again (trailer confirmation)
```

A reader: seek to end-of-file, read the last 16 bytes, verify the last 8
equal the magic, interpret the 8 before that as `header_length` (little-
endian uint64), seek back `16 + header_length` bytes from the end, read
and parse that many bytes as the JSON header. Every array's `offset` in
the header (see below) is relative to byte 8 — the start of the data
section, immediately after the leading magic.

There is no padding/alignment between arrays in the data section, or
between the data section and the header — every array's exact position
is given by an explicit byte offset in the header, so a reader never
needs to compute one from array sizes.

**Appending a model** to an existing file: read and verify the existing
trailer as above to find `D` (truncating the file there removes the old
header without touching the data section), append the new model's raw
bytes starting at offset `D - 8`, then write a new header — the old
`mesh`/`metadata`/other `models` entries unchanged, plus the new model's
entry — followed by a new `header_length` and trailing magic. Only the
new model's bytes and a new (small) header are written; the mesh
geometry and every other model's bytes are untouched.

**Removing a model** is not a cheap in-place operation in general (bytes
after the removed one can't be un-appended without shifting them) — spec
it as "write a new file, copying the raw bytes of everything kept"
(no decode/re-encode needed, just byte copies) rather than adding
free-space bookkeeping to the format to support true in-place deletion.

**Byte order**: everything in this format — the fixed-width integers in
the file layout above, and every array's data — is **little-endian**,
unconditionally. There is no per-file or per-array byte-order field.
Virtually all consumer and server hardware today is little-endian; this
trades big-endian portability (rare in practice) for removing
byte-order handling from every reader/writer implementation.

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

**Mesh descriptor** — has two modes:

```jsonc
// mode "embedded": the mesh's own geometry arrays are stored in this file.
{
  "mode": "embedded",
  "mesh_class": "OctreeMesh",             // "TensorMesh" | "UniformTensorMesh" | "OctreeMesh"
  "arrays": { "...": { /* array descriptor, see below */ } },
  // optional visualization metadata belongs on base_mesh for OctreeMesh
  "base_mesh": { /* only for mesh_class == "OctreeMesh"; see below */ }
}

// mode "reference": no per-leaf geometry in this file — a model-only
// file, paired with a separate embedded-mode file holding the actual mesh.
{
  "mode": "reference",
  "n_cells": 12345,                       // the only field every reference-mode descriptor has -- lets a reader validate model array lengths without opening the referenced mesh file
  "base_mesh": {                          // optional -- OctreeMesh reference only, see below
    "mesh_class": "UniformTensorMesh",
    "arrays": {...},
    "default_padding": [0, 0, 0, 0, 0, 0] // optional shared padding
  }
}
```

Reference mode exists mainly for `OctreeMesh`: its geometry
(`level`/`position`, one entry per leaf) can be large on its own, so
duplicating it into every model file is wasteful. `TensorMesh` and
`UniformTensorMesh` geometry is a handful of small arrays regardless of
cell count — cheap enough to always embed, so in practice their model
files should just use `"mode": "embedded"` rather than bothering with a
reference. Nothing in the format *forbids* a tensor mesh reference,
though — the mode is a property of the file, not tied to a specific
`mesh_class`.

`n_cells` is the only thing every reference-mode descriptor is
guaranteed to have. `mesh_class` and `base_mesh` are both
absent by default — a bare reference descriptor doesn't tell a reader
what kind of mesh it refers to, or let it verify the pairing at all,
and that's intentional, not an oversight: keeping a model paired with
the right mesh is the *caller's* responsibility, the same trust model
`subcrop_mesh`'s own `Model` class already uses for exactly this problem
(`Model` doesn't hold a mesh reference or any mesh-identifying
information either — see its docstring). This format doesn't try to
enforce a guarantee that the reference implementation of it doesn't
itself check.

`default_padding` is optional and does not change the format version. When
present it is a JSON list of exactly six non-negative integer values in
canonical order `[west, east, south, north, bottom, top]` (also
written `[x-, x+, y-, y+, z-, z+]`). Missing (or explicit JSON `null`) means
no default padding (`None`). For an embedded `OctreeMesh`, the one shared
setting is owned by its nested `base_mesh` descriptor and is stored there;
the outer descriptor does not repeat it. A reference descriptor carrying a
`base_mesh` follows the same nested canonical placement. For compatibility,
readers/writers may accept an outer-only field as a fallback and move it to
the nested base; if both fields are supplied they must normalize to the same
six values, otherwise the descriptor is rejected. A bare reference without a
`base_mesh` may retain its sole outer field because it has no base owner.
It is visualization metadata rather than geometry: it is not a binary array,
and does not alter the per-class geometry `arrays` key sets. As with mesh
construction, opposing
sides may sum to an axis's cell count but may not exceed it whenever an axis
shape is available; a bare reference descriptor has no shape for this check.

That said, `base_mesh` is still fully specified, for a writer or reader
that *does* want a sanity check — building a validation layer on top of
`subcrop_mesh`, or a different language's implementation, for instance. A
writer may include it whenever it's cheap or useful to:

- `base_mesh` (only meaningful for an `OctreeMesh` reference — see
  table below) is just `origin`/`cell_size`/`shape`, 9 numbers, cheap
  regardless of how many leaves the actual mesh has. When present, it
  buys two things: a reader can compare it directly against the paired
  mesh file's own `base_mesh` as an instant, free sanity check (no
  hashing required) before touching anything else, and a reader can
  learn the model's domain/resolution without opening the paired mesh
  file at all. It is only a sanity check, though — two different
  refinements of the *same* base grid (a realistic scenario: re-running
  octree generation with different refinement parameters but the same
  domain, e.g. while iterating on an inversion) would have identical
  `base_mesh` values despite being genuinely different meshes. Pairing a
  model with the right mesh remains the caller's responsibility.

**Array descriptor** — every array (geometry or model) is described the
same way:

```jsonc
{
  "dtype": "float64",   // "float64" | "float32" | "int64" | "int32" | "int16" | "int8"
  "shape": [3],          // array of integers; e.g. [] for a scalar, [3] for a vector, [n] for a per-cell array
  "offset": 0,           // byte offset from the start of the data section (not the start of the file)
  "length": 24,           // byte length; must equal the product of shape * dtype's byte width
  "checksum": "sha256:af04c1..."  // SHA-256 of exactly the raw bytes at [offset, offset+length); required
}
```

`checksum` is **required** on every array descriptor (geometry or
model), not optional — this is v1's corruption/truncation detection
mechanism. It's cheap: a writer computes it incrementally while writing
each array's bytes anyway, and a reader that's already reading an
array's bytes (which it has to, to use them) gets verification for
essentially free by hashing what it already read. Unlike a single
whole-file or whole-data-section hash, per-array checksums stay
consistent with the rest of this format's "read only what you need"
design — verifying one model doesn't require touching any other array's
bytes.

This also makes **appending** (see File layout) effectively atomic from
a reader's perspective: if a writer crashes after appending new array
bytes but before writing the new trailer, the old trailer (and its
checksums) is still exactly what it was — a reader sees a complete,
valid, unchanged file, never a half-appended one. A crash after the new
trailer is written is verifiable the normal way, via the new checksums.

All arrays in v1 are at most 1-dimensional (`shape` has 0 or 1 elements)
— there is no row/column-major ("C order"/"Fortran order") ambiguity to
resolve, since ordering only matters for arrays of 2 or more dimensions.
Multi-dimensional arrays (e.g. a vector-valued model) are out of scope
for v1; see Future wishlist.

**`arrays`** (within an `"mode": "embedded"` mesh descriptor), by
`mesh_class`:

| `mesh_class`         | `arrays` keys |
|-----------------------|---------------|
| `TensorMesh`           | `origin` (shape `[3]`, dtype `float64`), `h_x`, `h_y`, `h_z` (each shape `[n]` for that axis's cell count, dtype `float64`) |
| `UniformTensorMesh`    | `origin` (shape `[3]`, dtype `float64`), `cell_size` (shape `[3]`, dtype `float64`), `shape` (shape `[3]`, dtype `int32`) |
| `OctreeMesh`            | `level` (shape `[n_leaves]`, dtype `int8`), `position` (shape `[n_leaves]`, dtype `int32`) — plus a `base_mesh` object |

**Integer widths for `shape`/`level`/`position`**: `int32` is the
recommended default for `position` and `shape` — it covers any base grid
up to `2**31 - 1` cells total (~1290³ cube; real-world base grids run
from ~512³ common up to ~1024×1024×512 on the larger end, both with
massive headroom under that ceiling). A writer whose base grid actually
exceeds it may use `int64` for `position` instead — the array descriptor
already states `dtype` per array, so this is just a different valid
value, not a format change; a reader must handle either.

`level` doesn't need anywhere near that range — its maximum possible
value is `log2` of the base grid's largest per-axis dimension (around 11
for a ~2048-per-axis grid), nowhere close to even `int8`'s 127. Use
`int8` for `level` regardless of what width `position` needs; they're
separate arrays, so there's no cost to sizing them independently. The
same logic would apply to `i`/`j`/`k` individually if a future version
ever stored per-axis indices instead of one combined `position` value —
each only needs to cover that axis's extent (`int16` comfortably covers
up to 32767 per axis), not the full `nx*ny*nz` product `position` does.

For `OctreeMesh`, `base_mesh` is itself `{"mesh_class": "UniformTensorMesh",
"arrays": {...}}` — the same `UniformTensorMesh` schema, nested (including
the optional `default_padding` that is the octree/base pair's sole setting), mirroring
`OctreeMesh` composing a `UniformTensorMesh` for its finest-resolution
grid. It has no `mode`, `metadata`, or `models` of its own — it is always
inline data, never itself a reference, regardless of whether it's nested
inside an `"embedded"` `OctreeMesh` descriptor (mandatory there) or a
`"reference"` one (optional — see Mesh descriptor above).

**Cell numbering / ordering**: per-cell arrays (`h_x`/`h_y`/`h_z`,
`level`/`position`, and every model array) use the same cell order as
everywhere else in this ecosystem — index 0 is the bottom-south-west
cell, advancing `x` fastest, then `y`, then `z` slowest. This format does
not re-derive or re-check that ordering; it stores whatever order the
writer's own cell indexing already uses.

**Models**: each entry in `models` is a name (chosen by whoever wrote the
file — not a fixed vocabulary) mapping to its own `metadata` and a single
1D `array` of length `n_cells`, in the same cell order as above. A file
may have zero or more models.

## Validation a reader should perform

- Trailing `magic` matches exactly; reject otherwise.
- `format_version` is a version the reader understands.
- **Unknown keys are ignored, not rejected.** A reader MUST tolerate header
  keys it does not recognise, at every level — file, mesh descriptor,
  `base_mesh`, model entry, array descriptor. This is what allows the
  format to gain optional fields without a version bump, and a reader that
  rejects them makes every future addition a breaking change. Ignore what
  you do not understand; validate what you do.
- Every array's `length` equals `product(shape) * dtype_byte_width`, and
  `offset + length` does not exceed the data section's actual size.
- Every array's `checksum` matches a fresh SHA-256 of its actual bytes
  (see Array descriptor). A reader that isn't going to use a given array
  at all may skip verifying it — the point of per-array checksums is
  that verification cost is scoped to what's actually read.
- If `mode` is `"embedded"`, `mesh_class` is one of the three known
  values, and `arrays` contains exactly the keys that class requires
  (see table above) — no more, no fewer. If `mode` is `"reference"`,
  every model's array length equals the stated `n_cells` (there is no
  `mesh_class` to validate in this mode — see Mesh descriptor above).
- When reading a `"reference"`-mode model file together with the
  embedded-mode mesh file it's meant to pair with, and the model file
  happens to carry `base_mesh` and/or `checksum` (both optional — see
  the reference-mode discussion above): compare `base_mesh` directly as
  a cheap first check, and/or verify `checksum` against the mesh file's
  own checksum for the stronger guarantee. Neither is guaranteed to be
  present — a reader has no way to validate the pairing of a bare
  `{"mode": "reference", "n_cells": ...}` descriptor at all, and that's
  by design (see above).
- Per-class geometric validation (positive widths, positive `cell_size`,
  `OctreeMesh` leaf alignment/tiling) is **not** this format's
  responsibility — that's the receiving mesh class's own constructor
  validation (in whatever language is reading the file), the same as for
  any other mesh construction path.

## Future wishlist

Deliberately not in v1 — not open questions blocking this spec, just
ideas worth keeping in mind so a later version can add them without
breaking v1 files:

1. **Compression.** Left out of v1 entirely (see Design goals). Worth
   revisiting once there's a concrete large-file performance need —
   likely as a per-array opt-in flag in the array descriptor (e.g. a
   `"compression": "none" | "..."` field) rather than a whole-file
   toggle, so uncompressed and compressed arrays can coexist and readers
   that don't support a given scheme can at least fail on just that
   array rather than the whole file.
2. **Multi-dimensional models.** v1 only supports flat, scalar-per-cell
   models (`shape` has 0 or 1 elements). A vector-valued model (e.g. a
   per-cell 3-vector field) would need `shape` to allow more than one
   element and a stated element order (C vs. Fortran) — addable in a
   backwards-compatible way whenever it comes up, since `shape` is
   already an arbitrary-length list of integers in the schema; v1 just
   restricts it by convention, not by structure. Existing v1 files and
   readers are unaffected either way.
3. **Mandatory (or at least written) reference-mode pairing data.**
   `base_mesh` is fully specified (see the reference-mode discussion
   above), but v1 doesn't require it — a bare
   `{"mode": "reference", "n_cells": ...}` descriptor is entirely valid,
   and a reader can't validate its pairing at all.
   This is deliberate for now, not just unfinished: it mirrors
   `subcrop_mesh`'s own `Model` class, which likewise holds no mesh
   reference or mesh-identifying information and leaves pairing to the
   caller (see its docstring) — there's no reader-side consumer in
   `subcrop_mesh` today that the field would actually feed. Worth
   revisiting if that changes: a validation layer built on top of
   `subcrop_mesh`, a different language's implementation, or real
   multi-file (mesh + separately-distributed model files) usage that
   wants the guarantee in practice.

## Versioning

`format_version` is a plain integer counter, not semver. There is no minor
or patch component, because there is no such thing as a patch release of a
byte layout: a file either parses under a version's rules or it does not.

**A bump is required for** removing, renaming or retyping a field a reader
must read; changing the byte layout, offsets, or the trailer; changing the
dtype token strings; changing the meaning of an existing field.

**A bump is not required for** adding an optional field a reader may ignore
(see the unknown-key rule above), or anything placed in the arbitrary
`metadata` dicts at file or model level. `metadata` is the intended home
for experimental fields — put one there until it has earned promotion into
the header proper, and no version has to move in the meantime.

### Relationship to implementation versions

None. `format_version` versions *the bytes*; an implementation's own
version — `cmb-format`'s package version, or any other reader's — versions
*its code*, and the two are deliberately unrelated. A release of
`cmb-format` never implies a format change, and a format change never
implies a major release of any particular implementation. Do not infer one
from the other, and do not align them: implementations change far more
often than the wire does, and tying them means no implementation can make a
breaking API change without falsely implying that existing files are stale.

The `cmb-format` package states which versions a given build handles as
data rather than leaving it to be inferred:

```python
cmb_format.WRITTEN_FORMAT_VERSION  # the version write_file stamps
cmb_format.READABLE_FORMAT_VERSIONS  # the versions read_header accepts
```

Other implementations should expose something equivalent. A reader that
declines a file should say which versions it does handle.

## Implementations

This document is the format. Implementations follow it; none of them define
it.

- **`cmb_format`** (this repository) is the Python reference implementation.
  It works in *array dicts* — plain dicts of numpy arrays keyed the way the
  descriptors above are — and depends on numpy alone. It has no notion of a
  mesh or model object, which is what keeps it free of any one consumer's
  type system.
- **[`subcrop-mesh`](https://github.com/dwfmarchant/subcrop-mesh)** is the first
  consumer. Its `subcrop_mesh.io.cmb` module converts its own mesh and model
  classes to and from array dicts and is worth reading as a worked example
  of that layer. Its `docs/mesh-classes.md` and `docs/io-ubc.md` describe
  the prior art this format's conventions — cell numbering, coordinate
  system — are inherited from.

`tests/goldens/` holds committed reference files covering every shape
described above. They are the format's fixed point: round-trip tests only
prove an implementation agrees with itself, so a change to anything on the
wire — a dtype token, a header key, a byte order — is caught there and
nowhere else. A golden changing is the signal that this document needs to
change too.
