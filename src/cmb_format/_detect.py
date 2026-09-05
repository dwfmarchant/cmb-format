"""Identifying a CMB file on disk.

The only path-level entry point in this package. Everything else here works
in already-open files or in-memory array dicts, deliberately -- but "is this
a CMB file?" is a question about the format, so the format package is where
it belongs. A consumer should not have to import ``MAGIC`` and hand-roll the
comparison, and every consumer that did would write the same three lines.
"""

import os

from cmb_format._codec import MAGIC

__all__ = ["is_cmb_file"]


def is_cmb_file(file_name: str | os.PathLike) -> bool:
    """Whether a file is CMB, by its leading magic bytes.

    Identification, **not validation.** A match means the file begins the
    way every CMB file begins; it says nothing about whether the rest of the
    file is well-formed. Nothing here parses the header, checks the trailer,
    reads arrays, or verifies checksums -- a truncated or corrupted CMB file
    still returns `True`, and should, because "what kind of file is this"
    and "is this file intact" are different questions with different costs.
    `read_header` and `read_array` answer the second one, and raise when the
    answer is no.

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
