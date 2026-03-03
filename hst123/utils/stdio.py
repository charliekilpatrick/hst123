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
