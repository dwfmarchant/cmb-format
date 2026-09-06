"""Regenerate tests/goldens/.

    python tests/generate_goldens.py

Writes one ``.cmb`` file and one ``.header.json`` sidecar per case in
``cases.py``. Run after an intentional format change and commit the result.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cmb_format as cmb
from cases import CASES, build_bytes

GOLDENS = pathlib.Path(__file__).parent / "goldens"


def main():
    GOLDENS.mkdir(exist_ok=True)
    for name, case in CASES.items():
        raw = build_bytes(case)
        (GOLDENS / f"{name}.cmb").write_bytes(raw)
        import io

        header, _ = cmb.read_header(io.BytesIO(raw))
        (GOLDENS / f"{name}.header.json").write_text(
            json.dumps(header, indent=2, sort_keys=True) + "\n"
        )
        print(f"  {name}.cmb  ({len(raw)} bytes)")
    print(f"wrote {len(CASES)} goldens")


if __name__ == "__main__":
    main()
