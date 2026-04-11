"""
Context managers to silence stdout/stderr during noisy imports or C I/O.

Use :func:`suppress_stdout` for Python-level streams and
:func:`suppress_stdout_fd` when C libraries write to file descriptor 1.

:func:`limit_blas_threads_when_parallel` avoids BLAS/OpenMP oversubscription when
DrizzlePac uses multiple workers (``num_cores`` > 1).
"""
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def suppress_stdout() -> Iterator[None]:
    """
    Redirect ``sys.stdout`` and ``sys.stderr`` to ``os.devnull`` until exit.

    Yields
    ------
    None

    Notes
    -----
    Does not affect the C runtime ``stdout``; see :func:`suppress_stdout_fd`.
    """
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
def suppress_stdout_fd() -> Iterator[None]:
    """
    Redirect OS file descriptor 1 (stdout) to ``os.devnull``.

    Yields
    ------
    None

    Notes
    -----
    For C extensions (e.g. DrizzlePac ``photeq``) that write to the C stdio
    ``stdout``. Python :func:`suppress_stdout` does not catch those writes.
    Stderr (fd 2) is left unchanged so :mod:`logging` handlers on stderr still work.
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


_BLAS_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextmanager
def limit_blas_threads_when_parallel(num_cores: int) -> Iterator[None]:
    """
    When *num_cores* > 1, set common thread env vars to ``1`` for the duration.

    Multi-worker DrizzlePac plus multi-threaded NumPy/BLAS often oversubscribes
    CPUs; capping worker-local BLAS threads usually improves wall time without
    changing outputs.
    """
    if num_cores <= 1:
        yield
        return
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in _BLAS_THREAD_KEYS}
    try:
        for k in _BLAS_THREAD_KEYS:
            os.environ[k] = "1"
        yield
    finally:
        for k in _BLAS_THREAD_KEYS:
            v = saved[k]
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
