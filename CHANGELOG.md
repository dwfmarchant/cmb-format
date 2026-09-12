# Changelog

Notable changes to `cmb-format`, the Python reference implementation.

This file tracks the **package**. The CMB **format** versions separately —
see the [format changelog](FORMAT_CHANGELOG.md) and `docs/binary-format.md`'s
Versioning section. Which format versions a build handles is stated in code,
as `WRITTEN_FORMAT_VERSION` and `READABLE_FORMAT_VERSIONS`.

## [Unreleased]

- Add selective model loading to `read_file`.
- Add `list_models` and `read_contents` for inspection without loading model
  payloads, with only an embedded uniform mesh shape read when needed for its
  cell count.
- Preserve raw array order and normalize recognized descriptor padding on full reads.
- Replace positional padding with named mappings and write CMB v2 headers.
- Read CMB v1 padding through a single header-boundary compatibility adapter.
- Upgrade v1 headers to the current written format version when returned by `read_header`.
- Recognize padding only on the nested base for octree and reference descriptors with a base; remove the reconciliation helper.

## [0.1.0]

Initial release.

[Unreleased]: https://github.com/dwfmarchant/cmb-format/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dwfmarchant/cmb-format/releases/tag/v0.1.0
