# Code review findings — 0.1.0 release candidate

A working document, not part of the format specification. It records issues
found reviewing `79d4455` on `release-0.1.0`, to be resolved or consciously
deferred before tagging v0.1.0. Delete it once the list is closed.

Baseline at the time of review: 179 tests passing, `ruff check` and
`ruff format --check` clean, goldens regenerating byte-identically, the
README example running, 82% line coverage.

The central theme: `binary-format.md` specifies a reader-validation contract
that the reference implementation does not implement. The gaps compound —
each missing check silently disables a check that *is* implemented, so an
invalid file is accepted rather than rejected.

## 1. A typo in `mesh_class` disables every model check

Both the writer and the reader accept an unrecognized `mesh_class`, though
[the spec](binary-format.md#validation-a-reader-should-perform) requires it be
one of the three known values. Because `header_cell_count` (`_codec.py:419`)
and `_mesh_cell_count` (`_file.py:24`) return `None` for an unknown class, and
`validate_model_lengths` skips its check when the count is `None`, an unknown
class turns model validation *off* rather than failing:

```text
mesh_class="Tensormesh"  (typo), 2x2x2 mesh = 8 cells, model with 5 values
  build_file_bytes  -> accepted, 1074 bytes
  read_header       -> accepted, model shape [5]

mesh_class="TensorMesh"  (correct spelling)
  build_file_bytes  -> ValueError: model 'rho' has 5 values, but the mesh
                       has 8 cells; models are one value per cell
```

Not an attacker scenario — a one-character typo produces a silently invalid
file.

## 2. Array key sets are never checked against the mesh class

The spec requires `arrays` contain "exactly the keys that class requires — no
more, no fewer". Nothing enforces this. All of the following write and read
without complaint:

- `OctreeMesh` with `arrays: {"foo": ...}` and no `level` or `position`. The
  missing `level` also disables the cell-count check, so a 999-value model on
  that mesh passes both writer and reader.
- `TensorMesh` missing `origin`.
- `UniformTensorMesh` with only `shape`, no `origin` and no `cell_size`.
- Any mesh carrying extra, non-spec array keys.

Tensor classes have their `h_x`/`h_y`/`h_z`/`shape` keys checked, but only
incidentally, via `descriptor_shape` being evaluated as an argument at
`_codec.py:292`. `origin` and `cell_size` are never required.

## 3. The writer emits a non-integer `n_cells`; the reader then skips validation

`header_cell_count` gates on `isinstance(n_cells, int)` (`_codec.py:429`). For
a models-free reference mesh, `resolve_reference_n_cells` returns the caller's
value unchanged (`_codec.py:336`), so the writer emits it verbatim:

```text
{"mode": "reference", "n_cells": 3.0}   -> JSON  "n_cells": 3.0
{"mode": "reference", "n_cells": True}  -> JSON  "n_cells": true
```

Reading such a file back, `header_cell_count` returns `None`, and a model
declaring `shape: [999]` against `n_cells: 3.0` is accepted. The reference
implementation can produce a file that defeats its own documented validation.
`isinstance(True, int)` is also why `n_cells: true` survives the writer.

The spec calls `n_cells` a "required cell count"; it should be required to be
an integer, and the writer should coerce or reject.

## 4. `read_array` performs none of the required descriptor validation

The spec requires that `length` equal `product(shape) * dtype_byte_width` and
that `offset + length` not exceed the data section. Neither is checked in
`read_array` (`_codec.py:348`):

| Malformed descriptor | Actual behaviour | Expected |
| --- | --- | --- |
| `length` inflated to 1e9 | reads, fails late on the short read | reject up front |
| `offset: -8` | seeks into the leading magic; only the checksum catches it | reject |
| `shape` doubled, `length` unchanged | `ValueError: cannot reshape array of size 24` | reject as a format error |
| `dtype: "float16"` | `KeyError: 'float16'` | `ValueError` |
| `checksum` or `offset` key missing | `KeyError` | `ValueError` |

The checksum does the real defensive work, so no *wrong data* is returned. But
the error types leak implementation internals, and callers catching
`ValueError` — as the rest of the library's behaviour implies they should —
will not catch these.

The `offset + length` bound is not currently checkable: `read_array` receives
only `(f, data_start, descriptor)`, and `read_header` never returns
`header_start`, so the data section's size is not available to it. That is an
API-shape decision worth making before v1 freezes.

## 5. A non-object JSON header raises `AttributeError`

`header.get(...)` at `_codec.py:405` assumes the parsed header is an object:

```text
header = b"[]"    -> AttributeError: 'list' object has no attribute 'get'
header = b"null"  -> AttributeError: 'NoneType' object has no attribute 'get'
```

Every other corrupt-file path raises `ValueError`; `JSONDecodeError` and
`UnicodeDecodeError` are both `ValueError` subclasses, so those are fine.
`tests/test_detect.py:31` asserts `pytest.raises(ValueError)` on a corrupt
file, so this contract is real and this case breaks it.

## 6. Writer and reader disagree on `base_mesh` for embedded octrees

The spec says `base_mesh` is required for embedded octrees. The writer
enforces it (`_codec.py:194`); the reader accepts a file with `base_mesh`
deleted. A file this library refuses to write, it will happily read.

## 7. `serialize_array` writes multidimensional geometry arrays

The spec states that all v1 arrays are at most one-dimensional. Models are
guarded by `_check_model_lengths`, but geometry arrays are not: a `(3, 1)`
`origin` serializes as `"shape": [3, 1]` and reads back as `(3, 1)`.

A single `ndim > 1` check in `serialize_array` (`_codec.py:91`) closes this
for every array kind at once.

## 8. `descriptor_shape` is public, named for descriptors, and rejects them

`descriptor_shape` (`_codec.py:187`) accepts only the writer's raw-ndarray
form. Passed an actual parsed header it fails — with a message that says
"descriptor":

```text
descriptor_shape(writer mesh dict)    -> (3, 2, 4)
descriptor_shape(parsed header mesh)  -> ValueError: TensorMesh descriptor
                                         arrays['h_x'] must be a non-empty
                                         1D array
```

The trap is that its neighbours in the same public API — `base_mesh_descriptor`,
`padding_belongs_to_base_mesh`, `summarize_models` — all do accept parsed
headers. Either rename it or make it handle both shapes. Best decided before
the API is public.

## 9. Test coverage: the validation paths are the untested ones

The 179 tests concentrate on goldens and round-trips. Nearly every `raise` in
the library has no coverage, including rules the spec states explicitly:

| Location | Untested rule |
| --- | --- |
| `_codec.py:248` | outer and nested `default_padding` values must match |
| `_codec.py:194` | embedded octree missing `base_mesh` |
| `_codec.py:339` | models disagree on cell count |
| `_codec.py:342` | model length mismatches the declared `n_cells` |
| `_codec.py:331-336` | reference mode with neither `n_cells` nor models |
| `_codec.py:276` | unsupported mesh mode |
| `_padding.py` (63%) | booleans, non-finite, non-integer, out-of-int64-range and negative values; the `shape != (3,)` guard |

All of these were verified by hand to behave correctly. They are simply
unprotected against regression — which matters most for exactly the rules a
future contributor would be most likely to "simplify".

## 10. Smaller items

- **Dead code.** `_codec.py:158` is unreachable: the guard at `_codec.py:120`
  restricts `mesh_class` to two values and both branches return.
- **Specification gap.** The octree root partition `L = min(nx, ny, nz)` is
  well defined only when `L` divides every axis, which implicitly requires
  power-of-two base dimensions. The spec never says so, and the writer accepts
  6x4x4 and 3x3x3 base grids where the traversal is undefined.
- **Ordering asymmetry.** `serialize_arrays` sorts geometry keys, but
  `_assemble` iterates models in insertion order, so the same two models in a
  different dict order produce different bytes (`rho` at offset 96 versus
  288). Legal per the format, but it makes golden stability depend on caller
  dict ordering, and `test_writing_is_deterministic` compares the same dict
  twice, so it would not catch a regression here.
- **Triplicated logic.** Cell counting exists at `_codec.py:419`,
  `_file.py:24` and `tests/test_goldens.py:143`, with subtly different input
  assumptions. This is the root cause of findings 1 and 3 each appearing
  twice.
- **Misleading message.** `resolve_reference_n_cells` (`_codec.py:333`) says
  "needs either a mesh or at least one model" when it means `n_cells`; in
  reference mode there *is* a mesh.
- **Docstring gap.** `read_header` does not mention that the
  `UniformTensorMesh` shape read verifies a checksum, so `read_header` can
  raise "checksum mismatch".
- **Style.** `tests/generate_goldens.py:25` has `import io` inside the loop
  body.
- **Undocumented export.** `INT8_MAX` is public and genuinely used downstream
  (`subcrop-mesh`, `io/_cmb_convert.py:52`), but appears in neither the README
  nor the specification.

## Suggested order

1. **Findings 1-3** let invalid files through silently and are cheap to fix. A
   single shared, strict `cell_count(mesh)` that raises on an unknown
   `mesh_class` or a missing required array, plus an integer coercion for
   `n_cells`, closes all three and most of the duplication in finding 10.
2. **Findings 4-7** are the remaining specification-versus-code divergences.
   They matter more than usual because format v1 is about to freeze.
3. **Finding 8** is an API naming decision, best made before the API is
   public.
4. **Finding 9** should follow whatever changes 1-8 produce, so the new checks
   land with tests.

## What holds up well

Worth not regressing:

- `_padding.py` rejects ragged lists, strings, `None`s, dicts, booleans,
  non-finite floats and out-of-range integers, each with a clear `ValueError`.
- The double-copy in `normalize_default_padding` (`_padding.py:87`) genuinely
  resists `setflags(write=True)` on the returned array.
- Big-endian input converts correctly on write; `array_dtype_name` ignores
  byte order and `to_le_bytes` pins the little-endian dtype.
- `write_file` serializes the JSON header before opening the output, so a
  metadata serialization failure cannot truncate an existing file.
- Goldens regenerate byte-identically, and the golden suite covers all three
  mesh classes, both modes, nested base meshes, padding and every dtype token.
