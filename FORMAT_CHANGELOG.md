# CMB Format Changelog

This file records changes to the CMB file format only. Package APIs and
implementation behavior are tracked in [CHANGELOG.md](CHANGELOG.md).

## [Unreleased]

### v2

- Change `default_padding` from the positional six-item JSON list
  `[west, east, south, north, bottom, top]` to a complete named JSON object
  with those six keys.
- Keep array payload bytes, offsets, ordering, checksums, and the trailer
  unchanged from v1.

## v1

- Initial CMB format with positional six-item `default_padding` JSON lists.

[Unreleased]: https://github.com/dwfmarchant/cmb-format/compare/v0.1.0...HEAD
