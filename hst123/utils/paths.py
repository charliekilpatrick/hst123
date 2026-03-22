"""Path helpers for FITS files (stable across os.chdir)."""
import os


def normalize_work_and_raw_dirs(work_dir, raw_dir):
    """
    Resolve CLI ``--work-dir`` and ``--raw-dir`` to absolute paths.

    Default raw directory is ``<work-dir>/raw``. Must run once after parsing
    args so paths stay valid after ``os.chdir(work_dir)`` (no ``test_data/``
    doubling).
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
    Return absolute, normalized path so it stays valid after chdir(work_dir).

    TweakReg and related code change the process cwd to ``--work-dir``; relative
    paths like ``test_data/foo.fits`` would then incorrectly resolve under
    ``work_dir/test_data/...``.
    """
    if not path:
        return path
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))
