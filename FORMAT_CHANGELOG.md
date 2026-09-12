# CMB Format Changelog

This file records changes to the CMB file format only. Package APIs and
implementation behavior are tracked in [CHANGELOG.md](CHANGELOG.md).

## v2 (Released in v0.2.0)

- Change `default_padding` from the positional six-item JSON list
  `[west, east, south, north, bottom, top]` to a complete named JSON object
  with those six keys.
- Keep array payload bytes, offsets, ordering, checksums, and the trailer
  unchanged from v1.

## v1 (Released in v0.1.0)

- Initial CMB format with positional six-item `default_padding` JSON lists.
