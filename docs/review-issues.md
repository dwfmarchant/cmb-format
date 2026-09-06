# Review findings — 0.1.0 release candidate

A working document, not part of the format specification. It records review
findings for the v0.1.0 release. Items A–E are resolved below; item F remains
open for release-time follow-up and eventual deletion of this file.

Findings from the initial review of `79d4455` have been removed now that they
are fixed; they remain in git history. What follows is the cloud multi-agent
review of `2959bb4`, plus the historical items now resolved below.

## Ultrareview of `2959bb4`

The observations and line references in this section describe the code at
`2959bb4`, before the cleanup revisions below.

A multi-agent cloud review of `release-0.1.0` against `main` — 41 files,
+2688/-849. It returned **three findings, all severity "nit", and no
correctness bugs**, which is a meaningful result for a change of that size.

All three were verified against the code. Each is real, but two are
mischaracterized in ways that change what is worth doing about them.

| # | Finding | Verdict |
| --- | --- | --- |
| 1 | Duplicate validation on the write path | Real; negligible in the measured NumPy-array case |
| 2 | Padding validated twice | Real; the outlier branch is misidentified |
| 3 | `header_cell_count` is orphaned | Real; provenance misstated |

### 1. Duplicate validation on the write path

**Verdict: real duplication; negligible in the measured NumPy-array case. Optional.**

`_assemble` calls `_validate_raw_mesh` (`_file.py:65`), then `serialize_mesh`
calls it again (`_codec.py:465`). Instrumented, the mesh is validated exactly
twice per write and models are walked two or three times: once in
`resolve_reference_n_cells` (reference mode only), once in
`_check_model_lengths`, and once in `serialize_array`.

The review claims writes "pay 2x-3x the validation cost of the actual work"
and that the cost "scales with model count on the hot write path". Measured on
a 500,000-cell octree with 10 models, about 42 MB of array payload:

```text
full build_file_bytes              19.229 ms   100%
one _validate_raw_mesh pass         0.003 ms     0.017%
one _check_model_lengths pass       0.001 ms     0.003%
```

The redundant pass costs 3 microseconds, 0.017% of the write. Validation is
O(number of arrays); the write is O(bytes), dominated by `to_le_bytes` and
SHA-256. For existing NumPy arrays, `np.asarray` does not copy, so the
repeated model walks are close to free; list inputs can allocate and walk
their contents. The measured NumPy-array case does not support the performance
argument.

There is also a reason for the duplication that the review did not account
for: `serialize_mesh` is public API, and its `_validate_raw_mesh` call is what
makes a direct call safe on its own. The suggested fix — validate once, then
serialize with validation suppressed — would open a hole in that public entry
point unless the suppression is internal only.

Resolution: retain the duplicate validation as a clarity tradeoff. The
measured cost is negligible in the existing NumPy-array case.

### 2. Padding validated twice

**Verdict: real redundancy, but the outlier is a different branch.**

The review reports that the embedded-octree branch validates padding twice
while "reference-with-base" validates once. In fact three of the four branches
validate twice, and the single outlier is the *parsed* reference branch:

| Branch | `padding_as_json(outer, shape)` | `resolve_shared_padding` |
| --- | --- | --- |
| raw octree (`_codec.py:246`) | yes | yes |
| raw reference + base (`_codec.py:267`) | yes | yes |
| parsed octree (`_codec.py:830`) | yes | yes |
| parsed reference + base (`_codec.py:812`) | **no** | yes |

So this is a raw-versus-parsed inconsistency, not embedded-versus-reference.
`resolve_shared_padding` does validate the resolved value against the shape,
so the extra `padding_as_json` calls are genuinely redundant.

One consequence the review did not note: removing them changes error
precedence. Before B, an outer padding that both mismatched the nested value
and exceeded the shape reported the shape error; after B it reports the
mismatch error. No existing test required the old ordering.

Resolution: remove the three redundant calls in the raw-octree,
raw-reference-with-base, and parsed-octree branches. Keep
`resolve_shared_padding` as the shared agreement and shape check. A combined
mismatch and overflow now reports the mismatch first, which is accepted.

### 3. `header_cell_count` is orphaned

**Verdict: real, and the one worth acting on.**

`header_cell_count` (`_codec.py:842`) has no callers in `src`, in `tests`, or
downstream in `subcrop-mesh` and `subcrop-view`. It is listed in
`_codec.__all__` but not in the package's public API, so `test_public_api.py`
does not pin it. It is a thin wrapper around
`_validate_parsed_mesh(f, mesh, data_start, None)`.

The review calls it "newly defined". It is not: it existed at `79d4455`, where
`read_header` called it. The validation rework replaced that call with a direct
`_validate_parsed_mesh` call and orphaned it. The accurate framing is that it
is a leftover from the rework rather than new surface — the same class of
issue as the unreachable `return` removed from `shape_from_mesh_arrays`.

Resolution: delete it, along with its `_codec.__all__` entry (A,
`eb732ef`).

## Decisions and resolution record

The initial review listed six historical `raise` statements as untested. Five
were reachable and are now covered by C: non-octree meshes carrying a base
mesh on both I/O paths, a missing octree base in `raw_mesh_shape`, a reference
count mismatch, and the positive-minimum dimension check. The sixth was the
unsupported-mode branch inside `serialize_mesh`. It was already unreachable,
including for direct calls, because `_validate_raw_mesh` rejects the mode
first; A removed that redundant branch while retaining the public entry
validation.

The missing-base check in `raw_mesh_shape` and the `_padding.py:61` positive-
minimum fallback both predate the validation work at `79d4455`; they were not
introduced by the later validation pass.

Model entries continue to serialize in caller insertion order. This is
documented in the Python API only; model names are not sorted and the format
specification is unchanged. `descriptor_shape` remains as a compatibility
wrapper because it has three actual callers in `subcrop-mesh`.

| Item | Resolution | Revision |
| --- | --- | --- |
| A. Remove unused and unreachable code | Completed | `eb732ef` |
| B. Consolidate shared padding validation | Completed | `8017445` |
| C. Cover the remaining meaningful validation gaps | Completed | `06185cd` |
| D. Document named-model ordering in the Python API | Completed | `432c70d` |
| E. Correct and close this review record | Completed in this document edit | this document edit |
| F. Verify the cleanup before release | Open release-time follow-up | — |

## Verification state

At the latest verified code revision, `432c70d`:

- 204 tests pass on Python 3.12.14 and 3.14.7; combined statement/branch
  coverage is 91%.
- C covers the five previously unexecuted lines identified by the historical
  audit, with no prior coverage lost.
- `ruff check` and `ruff format --check` are clean.
- The golden suite passes for all 12 binary goldens and parsed header sidecars,
  byte-identically.
- No golden files or format-specification changes were made after `2959bb4`.
- The independent 125-case malformed-input audit passed at `2959bb4`, again
  after B (`8017445`), and again after D (`432c70d`). This is evidence for
  those 125 cases, not a claim that every possible malformed input has been
  tested.

## What holds up well

Worth not regressing:

- `_padding.py` rejects ragged lists, strings, `None`s, dicts, booleans,
  non-finite floats and out-of-range integers, each with a clear `ValueError`.
- The double-copy in `normalize_default_padding` genuinely resists
  `setflags(write=True)` on the returned array.
- Big-endian input converts correctly on write; `array_dtype_name` ignores
  byte order and `to_le_bytes` pins the little-endian dtype.
- `write_file` serializes the JSON header before opening the output, so a
  metadata serialization failure cannot truncate an existing file.
- Goldens regenerate byte-identically, and the golden suite covers all three
  mesh classes, both modes, nested base meshes, padding and every dtype token.

## F. Verify the cleanup before release

Before tagging, rerun the full verification on the release environment,
review any newly missed coverage paths, and rerun the independent malformed-
input audit after any release-time changes. Confirm that the golden bytes and
sidecars remain unchanged, record the final results, then delete this working
review file. This item remains open.
