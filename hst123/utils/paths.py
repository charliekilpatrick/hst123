"""
Path helpers for FITS files (stable across :func:`os.chdir`).

The pipeline changes the working directory to ``--work-dir``; relative paths
must be normalized to absolute before that.
"""
import os


def normalize_work_and_raw_dirs(work_dir, raw_dir):
    """
    Resolve CLI ``--work-dir`` and ``--raw-dir`` to absolute paths.

    Default raw directory is ``<work-dir>/raw``. Call once after parsing
    arguments so paths remain valid after ``os.chdir(work_dir)``.

    Parameters
    ----------
    work_dir : str or None
        ``--work-dir`` value, or None for current directory.
    raw_dir : str or None
        ``--raw-dir``; if ``None``, ``"."``, or ``"./"``, use ``<work>/raw``.

    Returns
    -------
    tuple of str
        ``(work_abs, raw_abs)`` absolute paths.
    """
    if work_dir:
        work_abs = os.path.abspath(os.path.expanduser(work_dir))
    else:
        work_abs = os.path.abspath(os.getcwd())
    if raw_dir in (None, "", ".", "./"):
        raw_abs = os.path.join(work_abs, "raw")
    else:
        raw_abs = os.path.abspath(os.path.expanduser(raw_dir))
    return work_abs, raw_abs


def normalize_fits_path(path: str) -> str:
    """
    Return an absolute, normalized ``path`` for FITS references.

    Parameters
    ----------
    path : str
        User-supplied path; empty string is returned unchanged.

    Returns
    -------
    str
        ``os.path.normpath`` of the expanded absolute path.

    Notes
    -----
    After alignment, the process cwd is ``--work-dir``; unresolved relative
    paths would double-resolve (e.g. ``test_data/foo.fits`` under ``work_dir``).
    """
    if not path:
        return path
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))
