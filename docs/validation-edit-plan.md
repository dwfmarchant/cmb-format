# Release validation implementation instructions

This plan addresses `review-issues.md` without changing the v1 byte layout,
cell ordering, or geometry conventions. Implement the work in this repository
only. Do not modify downstream projects. Leave commits to the reviewing agent.

## Scope and compatibility

- Preserve valid golden files byte for byte, including geometry/model ordering.
- Preserve `read_header(f) -> (header, data_start)` and the existing `read_array`
  call signature. Low-level array reads must still work on standalone buffers.
- Preserve support for int32 and int64 uniform shapes and octree positions,
  reference counts inferred from models, scalar standalone arrays, all six dtype
  tokens, and unknown optional fields in schema objects.
- Reject extra names specifically in mesh geometry `arrays`, because the spec
  requires exact geometry key sets. Do not reject unknown optional descriptor
  fields, or interpret arbitrary metadata as schema.
- The user confirmed that UBC octree base-grid dimensions must each be powers
  of two. Document and enforce this for octree base meshes, including optional
  reference base meshes. Do not apply it to standalone tensor meshes. Do not
  validate leaf tiling/alignment, reorder cells, sort model names, or bump the
  format version.
- Do not broadly catch exceptions and disguise programming or I/O failures as
  format errors. Malformed schema and unsupported field values should raise
  informative `ValueError`s with the field/array context.

## 1. Shared structural checks and writer validation

Read the complete codec, writer, padding helpers, spec, and relevant tests first.
Prefer a small private validation module if it reduces duplication or avoids
cycles. Share rules between raw-array and parsed-descriptor paths; these have
different representations, so do not force both through a misleading generic
cell-count function. Keep independent expected-count calculations in tests.

Validate the mesh mode and supported class before counting cells. Embedded
geometry requires exactly these names, shapes and dtypes:

| Class | Arrays |
| --- | --- |
| TensorMesh | float64 origin `[3]`; float64 h_x/h_y/h_z, each 1D |
| UniformTensorMesh | float64 origin/cell_size `[3]`; int32 or int64 shape `[3]` |
| OctreeMesh | int8 level and int32/int64 position, both 1D with equal lengths |

An embedded octree requires a UniformTensorMesh base descriptor with the same
uniform geometry checks, with each base-grid dimension a positive power of two.
Validate optional reference base descriptors likewise. Retain existing shape
and padding value checks; no other new geometric policies.
Ensure unsupported classes and missing geometry cannot silently produce an
unknown count and bypass model length checks. Every model is 1D with exactly
the count derived from valid mesh geometry or the reference count.

Require explicit reference `n_cells` to be a non-negative integer; reject
booleans, floats (including 3.0), strings, and negatives. Accept NumPy integer
inputs on the writer and normalize to Python int for JSON. Do not truncate or
silently coerce floats/strings. Zero remains allowed. Counts inferred from
models still work; supplied counts must agree with every model. Apply this rule
to public `resolve_reference_n_cells` and direct `serialize_mesh` paths too.

Reject arrays with more than one dimension in `serialize_array` before mutating
its buffer. Scalar arrays remain supported by this low-level API; models and
geometry receive their stricter schema checks. Keep valid serialization and
existing public helper use cases working.

## 2. Reader schema and descriptor checks

Validate the parsed header is an object, with a valid integer format_version,
mesh object and models mapping when present. Reject bool/float version values
that compare equal to version 1. Apply mesh structural rules above before
deriving a count. An absent models field may retain the existing empty-model
behavior. Do not restrict arbitrary JSON metadata unnecessarily.

For every geometry/base/model array descriptor, require:

- An object with the required dtype, shape, offset, length and checksum fields.
- A supported dtype token.
- A shape list containing zero or one non-negative integer dimensions (bools
  and floats are not integers for this schema).
- Non-negative integer offset and length, excluding booleans.
- Byte length exactly equal to product(shape) times dtype width; the scalar
  product is 1. Use Python integer arithmetic to avoid NumPy overflow.
- A SHA-256 checksum string with the required prefix and 64 hexadecimal digits.
- When the file data boundary is available, offset + length <= data size.

`read_header` already knows header_start and data_start: validate bounds there
for every known array descriptor, including arrays never read. Do this before
reading uniform shape payloads. Do not change its return tuple or eagerly read
all array payloads/checksums. Actual checksums remain verified on array reads.

`read_array` independently checks descriptor structure and byte-length agreement
before seeking/reading. It must reject negative offsets and malformed lengths,
but standalone buffers have no CMB trailer, so do not require one. Document that
file bounds are validated by `read_header` and its descriptors should be used
for normal file reads. Preserve checksum verification and read-only results.

Read uniform shape values as needed for existing shape/count/padding validation;
use Python integer multiplication for counts. Check padding consistently using
existing normalization/placement rules where shape is available. Unknown
optional fields must survive reading; avoid rewriting metadata or discarding
unknown fields during validation.

## 3. Public helper clarity and small cleanup

Add a clearly named `raw_mesh_shape` helper for the raw-array input operation
currently named `descriptor_shape`. Keep `descriptor_shape` as a documented
compatibility alias/wrapper with the same behavior, without runtime warnings.
Use the clearer name internally and update the exact public API test. Explicitly
document that parsed array descriptors are not inputs to this helper.

Remove the unreachable trailing return in shape_from_mesh_arrays if refactoring
makes it obsolete. Correct the reference-count error to mention n_cells rather
than 'a mesh'. Explain uniform shape reads and checksum errors in read_header's
docstring. Move import io out of the golden-generation loop. Describe INT8_MAX
briefly near its definition as the maximum int8 value; no README expansion is
needed merely for this constant. Preserve model insertion order deliberately.

Clarify in the existing spec text that n_cells is a non-negative integer and
offset/length/dimensions are non-negative integers. These make the intended
field types explicit; do not expand into an unrelated format redesign.
Document the user-confirmed power-of-two requirement for each octree base-grid
dimension alongside the root-local Morton convention. Explain that this makes
L a power of two dividing each dimension; do not claim divisibility alone implies
power-of-two dimensions.

## 4. Regression tests

Add parameterized tests organized by behavior, not one test per implementation
helper. Cover writer build_file_bytes and write_file, and reader inputs whose
JSON headers are independently mutated/reframed so reader tests do not rely on
the writer accepting invalid inputs. Preserve meaningful existing tests.

Cover unknown classes/modes; missing/extra geometry keys; wrong geometry dtype
or shape; unequal octree level/position lengths; missing/invalid octree base;
invalid explicit counts (with and without models); disagreeing model counts;
missing reference count without models; scalar/multidimensional model rejection;
and multidimensional standalone serialization rejection without buffer mutation.
Reject octree bases 6x4x4, 3x3x3 and 12x4x4 on writer and reader paths, in embedded
and reference modes. Accept existing cubic/rectangular power-of-two bases and
standalone uniform/tensor meshes whose dimensions are not powers of two.

Reader cases include non-object headers and malformed mesh/model structures;
missing descriptor fields; invalid dtype/checksum format; invalid shape, offset
or length types/values; inconsistent byte lengths; negative offsets with a
matching checksum for the leading magic; and descriptors extending into the
header or trailer. Confirm invalid descriptors are rejected before an attempted
large data read. Cover nested base descriptors and unread models in bounds tests.

Positive regressions include unknown optional fields at each extensible schema
level; both integer widths; big-endian writer input; zero reference count;
NumPy integer count input; valid scalar standalone arrays; and selective reads
that do not verify an unread model's payload checksum.

Add missing existing-validation tests listed in review item 9, especially
padding conflicts, bad padding values, and reference count inference failures.
Do not duplicate test matrices without adding a distinct behavior.

## 5. Verification and handoff

Use /Users/dmarchant/dev/.venv/bin/python with PYTHONPATH=src for tests and ruff.
Python 3.14 is also available at /private/tmp/cmb-ci-review-py314/bin/python.
Run focused tests while developing, then full pytest, ruff check and ruff
format --check. Generate golden outputs only into a temporary directory and
compare them against the tracked fixtures; do not update goldens to hide a
serialization change. Report any blocked checks or compatibility concerns.

Update review-issues.md with concise resolution/deferred notes rather than
deleting the user's review. Correct the mistaken claims about every model check,
bool counts, checksum/bounds safety, and power-of-two implications. Record the
user-confirmed octree dimension policy as addressed; keep ordering stable and note
why test-side cell counting remains independent. Do not claim complete coverage
or full geometry validation. Report changed files and test results for review.
