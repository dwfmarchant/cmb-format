# cmb-format

**CMB — Cell Model Binary.** A file format for cell-based meshes and the
models defined on them, plus a Python reference implementation.

A CMB file holds one mesh's geometry and any number of named models — arrays
of per-cell values — in a single file, with arbitrary metadata at both the
file and the model level.

[`docs/binary-format.md`](docs/binary-format.md) is the normative
specification. This package implements it; it does not define it.

## Design, in brief

- **Portable** — raw binary array data plus a JSON header. No zip container,
  no compression, nothing that needs a library beyond "read bytes" and "parse
  JSON" in the target language. Modelled on the same pattern as glTF's `.glb`
  and safetensors.
- **Fast** — arrays are stored raw and uncompressed. A reader seeks straight
  to an array's byte offset; nothing is parsed in proportion to array size.
- **Appendable** — the header sits at the *end* of the file, so adding a model
  to an existing file writes only the new bytes and a new header, without
  rewriting a potentially very large mesh.
- **Self-describing** — dtypes, shapes, offsets and checksums all live in the
  header.

Files use the `.cmb` extension and begin and end with the 8-byte ASCII magic
`CELLMODB`.

## Scope

This package works in **array dicts**: plain dicts of numpy arrays, keyed the
way the specification's descriptors are. It has no notion of a mesh or model
*object*, and depends on numpy alone.

Converting your own classes to and from array dicts is your code's job. That
boundary is the point — it is what keeps the format independent of any one
project's type system, and what makes the "implementable in any language from
the spec alone" claim in the specification true rather than aspirational.

[`mesh-core`](https://github.com/dwfmarchant/mesh-core) is the first consumer;
its `mesh_core.io.cmb` module is worth reading as a worked example of that
conversion layer.

## Versioning

Two version numbers exist and they are **unrelated**:

| | What it versions | Shape |
| --- | --- | --- |
| `format_version` in the file | the bytes | integer counter |
| this package's version | this code | semver |

A release of `cmb-format` never implies a format change, and a format change
never implies a major release. They move at completely different rates — the
code changes many times for every time the wire does — so tying them would
mean no API fix without falsely implying existing files are stale.

Which versions a build handles is stated as data, not inferred:

```python
cmb_format.WRITTEN_FORMAT_VERSION  # 1 -- what write_file stamps
cmb_format.READABLE_FORMAT_VERSIONS  # {1} -- what read_header accepts
```

`format_version` is stamped by this package, never by a consumer, so a
writer cannot omit it or drift from it. See the specification's Versioning
section for what does and does not warrant a bump; the short version is that
optional additions do not, because readers must ignore keys they do not
recognise.

## Golden files

`tests/goldens/` holds committed reference files — the bytes themselves are
the test. Round-trip tests only prove the implementation agrees with itself;
the goldens are what make a change to the wire format fail loudly, and they
are deliberately not gitignored.

Each golden is paired with a `.header.json` sidecar holding its parsed
header, so that an intentional format change shows up as a readable diff in
review rather than as "binary file changed".

## Usage

```python
import json, struct
import numpy as np
import cmb_format as cmb

buffer = bytearray()
header = {"format_version": 1, "metadata": {}}
header["mesh"] = cmb.serialize_mesh(
    {
        "mode": "embedded",
        "mesh_class": "TensorMesh",
        "arrays": {
            "origin": np.zeros(3),
            "h_x": np.array([1.0, 2.0]),
            "h_y": np.array([1.0, 1.0]),
            "h_z": np.array([3.0]),
        },
    },
    buffer,
)
header["models"] = {
    "rho": {
        "metadata": {"units": "ohm-m"},
        "array": cmb.serialize_array(np.arange(4.0), buffer),
    }
}

blob = json.dumps(header).encode("utf-8")
path.write_bytes(
    cmb.MAGIC + bytes(buffer) + blob + struct.pack("<Q", len(blob)) + cmb.MAGIC
)

with open(path, "rb") as f:
    header, data_start = cmb.read_header(f)
    arrays = cmb.read_arrays(f, data_start, header["mesh"]["arrays"])
```

## Development

Member of the `uv` workspace rooted at the parent `dev/` directory.

```bash
uv sync
uv run pytest
uv run ruff check .
```
