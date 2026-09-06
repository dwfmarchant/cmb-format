"""Identifying a CMB file on disk.

The only entry point here that takes a path; everything else works on open
files or in-memory array dicts.
"""

import os

from cmb_format._codec import MAGIC

__all__ = ["is_cmb_file"]


def is_cmb_file(file_name: str | os.PathLike) -> bool:
    """Whether a file is CMB, by its leading magic bytes.

    Identification, not validation. Reads the first 8 bytes and compares
    them; nothing else is parsed or checked, so a truncated or corrupted CMB
    file still returns `True`. `read_header` and `read_array` validate, and
    raise.

    Content, not filename: a CMB file named `.msh` is still CMB, and a file
    of something else named `.cmb` is still not.

    True for every CMB file regardless of mode -- an embedded-mode file
    carrying a mesh and a reference-mode (model-only) file share the same
    magic.

    Constant cost: reads 8 bytes and stops, whatever the file's size. Safe
    to call on every keystroke of a path field in a GUI.

    Filesystem errors propagate. A missing path raises `FileNotFoundError`
    and an unreadable one `PermissionError`, rather than being flattened
    into `False` -- "this file is not CMB" and "this file could not be read"
    are worth telling apart, and only the caller knows which of them is
    worth suppressing.

    Parameters
    ----------
    file_name : str or os.PathLike
        Path to the file to test.

    Returns
    -------
    bool
        True if the file's first bytes are CMB's magic. False for any other
        content, including files shorter than the magic.
    """
    with open(file_name, "rb") as f:
        return f.read(len(MAGIC)) == MAGIC
