"""Identify CMB files by their leading magic bytes."""

import os

from cmb_format._codec import MAGIC

__all__ = ["is_cmb_file"]


def is_cmb_file(file_name: str | os.PathLike) -> bool:
    """Check whether a file starts with the eight CMB magic bytes.

    Uses file contents rather than the filename extension. A truncated or
    corrupted file can return True if its leading magic remains intact.
    The header, trailer, and arrays are not validated.

    Filesystem errors, including FileNotFoundError and PermissionError,
    propagate to the caller.

    Parameters
    ----------
    file_name : str or os.PathLike
        Path to the file to test.

    Returns
    -------
    bool
        True if the first eight bytes match the magic; otherwise False,
        including for shorter files.
    """
    with open(file_name, "rb") as f:
        return f.read(len(MAGIC)) == MAGIC
