"""Suppress stdout/stderr (e.g. during drizzlepac/astroquery import)."""
import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """Redirect stdout and stderr to devnull; restores them on exit."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextmanager
def suppress_stdout_fd():
    """
    Redirect OS file descriptor 1 (stdout) to devnull.

    Use for C extensions (e.g. drizzlepac ``photeq``) that printf to the C runtime
    stdout; Python :func:`suppress_stdout` does not catch those. Leave fd 2
    (stderr) alone so ``logging`` handlers attached to stderr still work.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    saved = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)
