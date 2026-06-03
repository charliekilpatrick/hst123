"""
Context managers to silence stdout/stderr during noisy imports or C I/O.

Use :func:`suppress_stdout` for Python-level streams and
:func:`suppress_stdout_fd` when C libraries write to file descriptor 1.

:func:`limit_blas_threads_when_parallel` avoids BLAS/OpenMP oversubscription when
DrizzlePac uses multiple workers (``num_cores`` > 1).
"""
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator


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


@contextmanager
def tee_stdout_fd_to_logger(
    logger,
    *,
    prefix: str,
    level: int,
) -> Iterator[None]:
    """
    Tee OS file descriptor 1 (stdout) into *logger* while the context is active.

    This captures C-level and Python-level writes that ultimately hit fd 1, and
    is useful for libraries that print progress directly (JHAT, compiled tools).

    Notes
    -----
    - Does **not** touch fd 2 (stderr) so logging handlers on stderr keep working.
    - Also writes through to the original stdout so interactive CLI output is preserved.
    """
    # Create a pipe and swap stdout (fd 1) to the pipe writer.
    rfd, wfd = os.pipe()
    saved = os.dup(1)
    os.dup2(wfd, 1)
    os.close(wfd)

    stop = threading.Event()

    def _reader():
        buf = ""
        try:
            while not stop.is_set():
                try:
                    chunk = os.read(rfd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:
                    text = str(chunk)
                buf += text
                # Emit complete lines.
                while True:
                    if "\n" not in buf and "\r" not in buf:
                        break
                    # Normalize CR-only progress to newlines.
                    buf = buf.replace("\r\n", "\n").replace("\r", "\n")
                    line, _, rest = buf.partition("\n")
                    buf = rest
                    s = " ".join(line.split())
                    if s:
                        logger.log(level, "%s%s", prefix, s)
        finally:
            # Flush any tail without newline.
            tail = " ".join(buf.replace("\r", "\n").replace("\r\n", "\n").split())
            if tail:
                logger.log(level, "%s%s", prefix, tail)

    t = threading.Thread(target=_reader, name="hst123-tee-stdout", daemon=True)
    t.start()
    try:
        yield
    finally:
        # Restore stdout; closing the pipe causes reader to exit.
        os.dup2(saved, 1)
        os.close(saved)
        stop.set()
        try:
            os.close(rfd)
        except OSError:
            pass
        # Give the reader a moment to flush buffered lines.
        t.join(timeout=1.0)


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
