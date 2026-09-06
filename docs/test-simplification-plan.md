# Test simplification plan

## Objective and scope

Simplify the suite at `b378ad8` while retaining its distinct behavioral
guarantees. The baseline is 320 collected cases from 90 test functions,
1,294 lines in `test_*.py`, and 90% combined statement/branch coverage.
Production code, public API, format specification, fixture inputs, and binary
goldens must remain unchanged. Do not modify downstream repositories.

The problem is maintenance cost, not execution time. Aim to remove roughly a
quarter of the test code, counting any new helper modules in the result. This
is a guide, not a quota: do not compress readable code, weaken assertions, or
hide hundreds of cases in a bespoke testing framework to meet a number.

## Implementation steps

1. **Share small, concrete setup helpers.** Consolidate header unpacking,
   framing, field mutation, and replacement of array payloads into one small
   test helper module used by codec and validation tests. Reuse deep copies of
   `CASES` rather than redeclaring tensor/uniform/octree/base constructors.
   A helper may accept a case name and return a fresh copy; it must not mutate
   shared cases. Keep helpers independent of production validators. Avoid
   generic callback dispatch, elaborate path languages, or a fixture framework.

2. **Simplify the golden suite.** Keep byte-for-byte comparison for every
   fixture, parsed-sidecar equality, reading/checksumming every stored array,
   and independent model cardinality assertions. Related checks may share one
   parameterized per-fixture test, with assertion context naming the fixture
   and array. Remove the separate same-input-twice determinism test: comparison
   against fixed golden bytes already covers it. Check literal magic, trailer
   encoding, and byte-eight data start once on a representative fixture rather
   than multiplying each invariant across all twelve cases. Keep the literal
   dtype/schema assertions and the check for missing or stray fixtures.

3. **Reduce writer cross-products.** Detailed invalid-input matrices should
   normally call `build_file_bytes`, because `write_file` shares `_assemble`.
   Retain a small explicit integration test that `write_file` produces the same
   valid bytes and rejects invalid geometry and models. Retain the check that
   validation failure does not overwrite an existing file. Do not duplicate
   every dtype, mesh class, shape, and invalid scalar across both wrappers.

4. **Consolidate validation by behavior.** Merge repeated missing/extra-key,
   malformed-header, reference-count, and padding setup into readable tables
   with meaningful pytest IDs. Keep separate writer and reader coverage where
   their validation paths differ. Reduce model-rank combinations to cover the
   shared embedded path and the separate reference inference path; retain cell
   count checks for each mesh class, since counting differs. Remove duplicate
   examples that only repeat a previously covered rule. Do not remove bool,
   integer-valued float, malformed container, or overflow cases merely because
   their values look unusual: these caught real gaps in this review.

5. **Simplify detection and array positives.** Leading-magic detection is
   independent of mesh class; use a representative valid file instead of all
   golden cases. Preserve short/incorrect magic, extension independence,
   matching magic with an invalid remainder, str/Path inputs, and propagation
   of filesystem errors. Native/big-endian dtype round trips may be combined
   into one clear test covering all six tokens and literal little-endian bytes.
   Keep int32/int64 geometry support and scalar standalone array behavior.

6. **Keep specialized tests recognizable.** Morton traversal tests, the
   negative-offset/matching-checksum regression, bounds-before-read I/O guards,
   integer-overflow regressions, and output-preservation checks should remain
   directly understandable. Do not bury them in generic mutation tables.
   Public API checks can stay small and explicit; sorting `__all__` is not a
   reason to undertake unrelated API changes.

## Rules for preserving meaningful coverage

- The malformed-file fixtures must remain internally consistent except for
  the rule under test. When changing dtype, shape or payload length, update
  length/checksum/offsets as necessary. An octree length-mismatch test should
  reach the equal-leaf-length check, not a descriptor-byte-length error.
- Keep context-specific error assertions where they distinguish the intended
  check. Do not replace all assertions with bare `raises(ValueError)`.
- Keep independent expected values and traversal/count calculations; do not
  reuse production validation logic to manufacture an expected result.
- Writer and reader unknown-class/mode rejection, exact geometry schemas,
  required octree bases, and power-of-two bases must stay covered. Preserve
  acceptance of non-power-of-two standalone tensor meshes.
- Preserve strict reference counts (including inference and disagreement),
  scalar/matrix model rejection, geometry rank/dtype/length checks, required
  descriptor fields, invalid types, non-negative dimensions/offsets/lengths,
  byte-length consistency, unsupported versions, and non-object headers.
- Preserve full-file bounds checks for unread models and nested base arrays,
  standalone `read_array` validation before I/O, checksum mismatch/short reads,
  and selective reads that leave unread payload checksums unverified.
- Preserve unknown optional fields, endian conversion, int64 count arithmetic,
  supported integer widths, zero reference counts, NumPy integer input,
  padding errors/conflicts/bounds, and the legacy shape-helper alias.

## Verification and handoff

Read applicable AGENTS instructions and the complete current tests before
editing. Do not commit; the parent agent will review and commit the patch.
The parent owns this plan and any final review notes.

Run the full suite with Python 3.12 and 3.14, not just focused subsets:

```sh
PYTHONPATH=src /Users/dmarchant/dev/.venv/bin/python -m pytest -q
PYTHONPATH=src /private/tmp/cmb-ci-review-py314/bin/python -m pytest -q
/Users/dmarchant/dev/.venv/bin/python -m ruff check .
/Users/dmarchant/dev/.venv/bin/python -m ruff format --check .
git diff --check
```

Compare coverage against `/private/tmp/cmb-simplify-before.json`, using a
temporary COVERAGE_FILE and JSON report path. Inspect newly missed statements
and branches, not just the rounded percentage. A lost distinct validation path
must be restored or explicitly explained to the parent before handoff. Do not
add coverage exclusions. Do not regenerate or update goldens.

Report before/after test cases, functions, and total test-code lines including
new helpers, and explain which redundancy was removed. Name any behavior that
could not be retained. Finish only after full tests and checks pass. Send a
brief progress update during the work; do not report a partial focused-test
run as completion.

## Reviewed result

Luna High implemented the simplification, followed by independent review.

| Measure | Before | After |
| --- | --- | --- |
| Collected test cases | 320 | 197 |
| Test functions | 90 | 76 |
| Test-code lines, including new helpers | 1,294 | 1,148 |
| Combined statement/branch coverage | 90% | 90% |

The reduction comes from consolidating per-fixture golden checks, removing
redundant determinism and framing combinations, reducing duplicate writer
tests, and sharing setup and mutation helpers. Specialized bounds, overflow,
checksum and Morton-order regressions remain explicit. The code reduction is
smaller than the initial guide; further compression was not required at the
expense of readable tests.

The exact executed statements and branch arcs match the baseline, with no
newly missed lines or branches. All 197 cases pass on Python 3.12 and 3.14 and
against the installed wheel outside the checkout. Ruff, formatting and
whitespace checks pass. Production code, fixture inputs and golden files are
unchanged.

Independent checks in temporary copies confirmed that both suites detect all
ten injected defects: disabled bounds or checksum checks, overflowing cell
counts, omitted writer or reader model-length checks, missing power-of-two
checks, accepted boolean counts, missing geometry-key checks, accepted array
rank violations, and broken endian conversion. These checks supplement the
coverage comparison; they do not assert that every possible defect is tested.
