"""
Logging setup (LogConfig, get_logger, make_banner) and status formatting.

**CLI output policy (hst123 package)**

All **user-facing** messages from this package should go through the ``logging``
API using loggers under the ``hst123`` namespace, typically::

    from hst123.utils.logging import get_logger
    log = get_logger(__name__)

Use ``log.info`` / ``log.warning`` / ``log.error`` for normal output; ``log.debug``
for detail. Section headers use ``make_banner`` (compact single-line markers).
The helper
``ensure_cli_logging_configured()`` attaches a formatted handler (default
**stderr**) the first time the CLI runs so messages are visible even if nothing
else configured logging.

**Work-directory session log:** after ``--work-dir`` is resolved, the pipeline
calls ``attach_work_dir_log_file()`` to create ``<work-dir>/logs/`` and add a
``FileHandler`` so formatted ``hst123.*`` records (including
``hst123.third_party`` forwarded from DrizzlePac/stpipe) match stderr. By default,
``attach_work_dir_log_file()`` also wraps ``sys.stdout`` and ``sys.stderr`` (by
default) so Python-level writes to both streams are **appended** to the same
session file via one shared append handle (disable with
``HST123_SESSION_LOG_STDOUT=0`` / ``HST123_SESSION_LOG_STDERR=0``). Because the
CLI ``StreamHandler`` writes formatted records to stderr, those lines can appear
**twice** in the session file (once from the ``FileHandler``, once from the
stderr mirror). Console stream handlers are repointed to the tee when mirrors
install so piped logging matches the terminal. C extensions that bypass Python
``sys.stdout`` / ``sys.stderr``
may still need ``tee_stdout_fd_to_logger`` for fd 1. At exit, hooks flush the
handler and restore streams. Compare logs by matching the ``Session log: …``
line at startup to the filename and PID.

**External programs** (``dolphot``, ``calcsky``, ``make``, …): use
``run_external_command()`` so their stdout/stderr is merged, streamed to the
logger (INFO), and optionally copied to a file (e.g. DOLPHOT ``.output``).

Third-party libraries (e.g. drizzlepac, astroquery) may still write to stdout/stderr
directly; some call sites use ``suppress_stdout`` where needed. AstroDrizzle and photeq runfiles live under ``<work-dir>/.hst123_runfiles/``, are replayed into
loggers ``hst123.astrodrizzle`` / ``hst123.photeq``, then removed. The pipeline **always** replays those runfiles into the session log with compact whitespace (not controlled by ``HST123_REPLAY_SUBLOGS``). For other uses of :func:`ingest_text_file_to_logger`, default (**``HST123_REPLAY_SUBLOGS``** unset or ``1``) is full replay; set ``HST123_REPLAY_SUBLOGS=0`` or ``summary`` for a one-line summary only.

Environment variables: ``HST123_SESSION_LOG_STDOUT`` / ``HST123_SESSION_LOG_STDERR``
(default ``1``: mirror each stream into the session log; set ``0`` to disable),
``HST123_LOG_LEVEL`` (use ``DEBUG`` to show ``@log_calls``
entry/exit lines), ``HST123_LOG_ENABLE_STDOUT``, ``HST123_LOG_ENABLE_FILE``,
``HST123_LOG_DIR``, ``HST123_REPLAY_SUBLOGS``, ``HST123_LOG_FULL_NAMES`` (set to
``1`` to show full logger names in ``[…]`` instead of compressed tags),
``HST123_LOG_PROPAGATE_ROOT`` (set to ``1`` to propagate the ``hst123`` logger to
the root logger; default is off so DrizzlePac/stpipe handlers on root do not
duplicate every line). For DOLPHOT scrape ``parse_phot`` batch progress, use
``HST123_NO_SCRAPE_PROGRESS=1`` to disable the bar and periodic lines, or
``HST123_SCRAPE_PROGRESS_LOG_ONLY=1`` to force INFO lines (and ETA) instead of a
stderr progress bar when running in a terminal.
"""
from __future__ import annotations

import atexit
import functools
import logging
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

_QUEUE = None
_LISTENER = None
_CONFIGURED = False
_CLI_LOGGING_INSTALLED = False
_WORK_DIR_LOG_HANDLER = None
_WORK_DIR_LOG_ATEXIT_REGISTERED = False
_SESSION_STREAM_MIRROR_STATE: dict | None = None
_SESSION_STREAM_MIRROR_ATEXIT_REGISTERED = False


class _StreamTee:
    """Write to a primary stream and append a copy to a second file-like object."""

    __slots__ = ("_primary", "_copy", "_lock")

    def __init__(self, primary, copy_fh, *, shared_lock: threading.Lock | None = None) -> None:
        self._primary = primary
        self._copy = copy_fh
        self._lock = shared_lock if shared_lock is not None else threading.Lock()

    def write(self, s: str) -> int:
        n = self._primary.write(s)
        if s:
            with self._lock:
                self._copy.write(s)
                self._copy.flush()
        if isinstance(n, int):
            return n
        return len(s) if s else 0

    def flush(self) -> None:
        self._primary.flush()
        with self._lock:
            self._copy.flush()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def _is_console_stream_handler(h: logging.Handler) -> bool:
    """True for terminal StreamHandlers (not FileHandler or rotating log files)."""
    return isinstance(h, logging.StreamHandler) and not isinstance(
        h,
        (logging.FileHandler, RotatingFileHandler),
    )


def _iter_all_handlers() -> list[logging.Handler]:
    """Handlers on root, named loggers, and the multiprocessing QueueListener if any."""
    seen: set[int] = set()
    out: list[logging.Handler] = []

    def add(h: logging.Handler) -> None:
        hid = id(h)
        if hid not in seen:
            seen.add(hid)
            out.append(h)

    for h in logging.root.handlers:
        add(h)
    for name in logging.root.manager.loggerDict:
        obj = logging.root.manager.loggerDict[name]
        if isinstance(obj, logging.Logger):
            for h in obj.handlers:
                add(h)
    lst = _LISTENER
    if lst is not None:
        try:
            for h in lst.handlers:
                add(h)
        except Exception:
            pass
    return out


def _retarget_console_stream_handlers(stream_map: dict) -> None:
    """
    Point StreamHandler.stream from old file-like objects to new ones.

    Used when ``sys.stdout`` / ``sys.stderr`` are wrapped for session mirroring:
    handlers installed earlier still reference the original streams and would
    otherwise bypass the tee.
    """
    if not stream_map:
        return
    for h in _iter_all_handlers():
        if not _is_console_stream_handler(h):
            continue
        st = getattr(h, "stream", None)
        if st is None:
            continue
        new_st = stream_map.get(st)
        if new_st is None:
            continue
        h.acquire()
        try:
            h.stream = new_st
        finally:
            h.release()


def _restore_session_stream_mirrors() -> None:
    """Restore ``sys.stdout`` / ``sys.stderr`` and close the shared session append handle."""
    global _SESSION_STREAM_MIRROR_STATE
    st = _SESSION_STREAM_MIRROR_STATE
    if st is None:
        return
    # Repoint logging handlers that target the tee wrappers back to the raw
    # streams before closing the session append handle.
    restore_map: dict = {}
    if st.get("stdout_tee") is not None:
        restore_map[st["stdout_tee"]] = st["orig_stdout"]
    if st.get("stderr_tee") is not None:
        restore_map[st["stderr_tee"]] = st["orig_stderr"]
    if restore_map:
        _retarget_console_stream_handlers(restore_map)
    try:
        if st.get("stdout_wrapped"):
            sys.stdout = st["orig_stdout"]  # type: ignore[assignment]
    except Exception:
        pass
    try:
        if st.get("stderr_wrapped"):
            sys.stderr = st["orig_stderr"]  # type: ignore[assignment]
    except Exception:
        pass
    try:
        st["copy_fh"].flush()
        st["copy_fh"].close()
    except Exception:
        pass
    _SESSION_STREAM_MIRROR_STATE = None


def _restore_session_stdout_tee() -> None:
    """Backward-compatible alias for :func:`_restore_session_stream_mirrors`."""
    _restore_session_stream_mirrors()


def _install_session_log_stream_mirrors(log_path: str) -> None:
    """
    Append ``sys.stdout`` and/or ``sys.stderr`` writes to *log_path* (session log).

    Uses one shared append file object and lock so interleaved output stays coherent.
    """
    global _SESSION_STREAM_MIRROR_STATE, _SESSION_STREAM_MIRROR_ATEXIT_REGISTERED
    if _SESSION_STREAM_MIRROR_STATE is not None:
        return
    want_out = os.environ.get("HST123_SESSION_LOG_STDOUT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    want_err = os.environ.get("HST123_SESSION_LOG_STDERR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if not want_out and not want_err:
        return
    try:
        copy_fh = open(log_path, "a", encoding="utf-8", errors="replace")
    except OSError:
        return
    lock = threading.Lock()
    orig_out = sys.stdout
    orig_err = sys.stderr
    state: dict = {
        "copy_fh": copy_fh,
        "orig_stdout": orig_out,
        "orig_stderr": orig_err,
    }
    if want_out:
        tee_out = _StreamTee(orig_out, copy_fh, shared_lock=lock)
        sys.stdout = tee_out  # type: ignore[assignment]
        state["stdout_wrapped"] = True
        state["stdout_tee"] = tee_out
    if want_err:
        tee_err = _StreamTee(orig_err, copy_fh, shared_lock=lock)
        sys.stderr = tee_err  # type: ignore[assignment]
        state["stderr_wrapped"] = True
        state["stderr_tee"] = tee_err
    # CLI StreamHandlers still reference the pre-wrap stdout/stderr objects.
    retarget: dict = {}
    if want_out:
        retarget[orig_out] = sys.stdout
    if want_err:
        retarget[orig_err] = sys.stderr
    if retarget:
        _retarget_console_stream_handlers(retarget)
    _SESSION_STREAM_MIRROR_STATE = state
    if not _SESSION_STREAM_MIRROR_ATEXIT_REGISTERED:
        atexit.register(_restore_session_stream_mirrors)
        _SESSION_STREAM_MIRROR_ATEXIT_REGISTERED = True


def _flush_work_dir_session_log() -> None:
    """Best-effort flush so NFS/editors see complete session logs after exit or interrupt."""
    h = _WORK_DIR_LOG_HANDLER
    if h is None:
        return
    try:
        h.flush()
        stream = getattr(h, "stream", None)
        if stream is not None and hasattr(stream, "flush"):
            stream.flush()
    except Exception:
        pass


RUN_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
ROOT_LOGGER = "hst123"
LOG_CAPTURE_PACKAGES = []
DEFAULT_FILENAME_PREFIX = "hst123_log"


def _apply_hst123_propagate_policy() -> None:
    """
    By default, disable propagation from the ``hst123`` logger to ``logging.root``.

    DrizzlePac/stpipe and other stacks often attach a second handler to the root
    logger, which repeats every ``hst123.*`` message in a second format (e.g.
    ``… - stpipe - INFO - …``). User-facing output should use the compact
    formatter on the ``hst123`` logger only.
    """
    lg = logging.getLogger(ROOT_LOGGER)
    if os.environ.get("HST123_LOG_PROPAGATE_ROOT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        lg.propagate = True
    else:
        lg.propagate = False


def suppress_third_party_root_stream_handlers() -> None:
    """
    Remove any StreamHandlers attached to the *root* logger.

    Some third-party stacks (notably DrizzlePac/stpipe) attach a root StreamHandler
    with their own formatter, producing extra console lines like::

        2026-... - stpipe - INFO - ...

    ``hst123`` logs via the ``hst123`` logger (and does not propagate to root by
    default), so removing root stream handlers keeps stdout/stderr clean while
    leaving our session log intact.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            try:
                root.removeHandler(h)
            except Exception:
                continue


_ROOT_ADDHANDLER_ORIG = None
_STPIPE_ADDHANDLER_ORIG = None
_LOGGER_ADDHANDLER_ORIG = None


def _normalize_third_party_log_text(text: str) -> str | None:
    """
    Per-line strip; drop empty lines; return None if nothing remains.

    Used when forwarding root-logger records into ``hst123.third_party`` so
    DrizzlePac/stpipe noise (blank lines, padding whitespace) does not clutter
    the session log.
    """
    if not isinstance(text, str):
        text = str(text)
    kept: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            kept.append(s)
    if not kept:
        return None
    return "\n".join(kept)


class _ForwardRootRecordsToHst123(logging.Handler):
    """
    Root logger handler that forwards third-party records into ``hst123`` logs.

    This lets us *capture* messages from stacks that log on the root logger
    (e.g. DrizzlePac/stpipe), while still keeping console output under hst123's
    control.
    """

    def __init__(self):
        super().__init__()
        self._dst = logging.getLogger(f"{ROOT_LOGGER}.third_party")
        # Suppress redundant duplicates: stpipe often mirrors messages emitted by
        # other third-party loggers (via its own wrappers / delegation). Keep a
        # small recent-message cache so we can drop identical "stpipe" copies.
        self._recent: list[tuple[float, int, str, str]] = []  # (t, level, name, msg)
        self._recent_max = 200

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Avoid loops: do not re-emit hst123 records back into itself.
            if str(getattr(record, "name", "")).startswith(ROOT_LOGGER):
                return
            name = str(getattr(record, "name", ""))
            raw_msg = record.getMessage()
            msg = _normalize_third_party_log_text(raw_msg)
            if msg is None:
                return

            # De-duplicate: if stpipe logs the exact same message/level that was
            # just logged by a non-stpipe third-party logger, drop the stpipe copy.
            # Use a small time window to avoid hiding legitimate repeated stpipe logs.
            try:
                import time

                now = time.time()
                # Prune old entries (>2s)
                self._recent = [x for x in self._recent if (now - x[0]) <= 2.0]
                if name.startswith("stpipe"):
                    for (t0, lvl0, n0, m0) in reversed(self._recent):
                        if lvl0 == record.levelno and m0 == msg and not n0.startswith("stpipe"):
                            if (now - t0) <= 1.0:
                                return
                self._recent.append((now, int(record.levelno), name, msg))
                if len(self._recent) > self._recent_max:
                    self._recent = self._recent[-self._recent_max :]
            except Exception:
                pass

            # Preserve the original logger name for debugging.
            self._dst.log(record.levelno, "[%s] %s", name, msg)
        except Exception:
            return


def capture_root_logging_to_hst123(*, block_stream_handlers: bool = True) -> None:
    """
    Capture third-party root-logger output into ``hst123`` and suppress their console handlers.

    - Removes existing root StreamHandlers (so ``stpipe - INFO - ...`` lines stop).
    - Installs a forwarding handler on root so root logger records show up in the
      hst123 session log (as ``[third_party] [orig.logger] message``).
    - Optionally blocks future attempts to attach StreamHandlers to the root
      logger (common in stpipe/drizzlepac).
    """
    root = logging.getLogger()

    # Remove any existing root StreamHandlers.
    suppress_third_party_root_stream_handlers()

    # stpipe installs a DelegationHandler on the *root* logger at import time which
    # re-emits records under the "stpipe" logger name, creating redundant duplicates
    # (e.g. one line from "stsci.skypac.utils" and the same line again from "stpipe").
    # Remove it so we only capture the original third-party logger records.
    try:
        for h in list(root.handlers):
            if h.__class__.__name__ == "DelegationHandler" and h.__class__.__module__.startswith("stpipe"):
                root.removeHandler(h)
    except Exception:
        pass

    # Ensure our forwarder exists (only once).
    if not any(isinstance(h, _ForwardRootRecordsToHst123) for h in root.handlers):
        root.addHandler(_ForwardRootRecordsToHst123())

    # Block future StreamHandler additions to root (stpipe likes to attach one).
    global _ROOT_ADDHANDLER_ORIG
    if block_stream_handlers and _ROOT_ADDHANDLER_ORIG is None:
        _ROOT_ADDHANDLER_ORIG = root.addHandler

        def _add_handler_blocking_stream(h):
            if isinstance(h, logging.StreamHandler):
                return
            return _ROOT_ADDHANDLER_ORIG(h)

        root.addHandler = _add_handler_blocking_stream  # type: ignore[assignment]

    # Some libraries attach handlers to their *own* loggers (not root), especially
    # stpipe. Remove their console handlers so records propagate to root and get
    # forwarded into hst123 instead of printing in the stpipe format.
    def _strip_console_handlers(logger: logging.Logger) -> None:
        for h in list(logger.handlers):
            # stpipe config uses StreamHandler(sys.stderr/sys.stdout) for console output.
            if isinstance(h, logging.StreamHandler):
                try:
                    logger.removeHandler(h)
                except Exception:
                    pass

    # Remove handlers from any already-instantiated stpipe logger(s).
    try:
        for name, obj in list(logging.Logger.manager.loggerDict.items()):
            if isinstance(obj, logging.Logger) and str(name).startswith("stpipe"):
                _strip_console_handlers(obj)
                obj.propagate = True
    except Exception:
        pass

    # Prevent stpipe from re-attaching its own console StreamHandler later.
    global _STPIPE_ADDHANDLER_ORIG
    try:
        stp = logging.getLogger("stpipe")
        if block_stream_handlers and _STPIPE_ADDHANDLER_ORIG is None:
            _STPIPE_ADDHANDLER_ORIG = stp.addHandler

            def _stpipe_add_handler_blocking_stream(h):
                if isinstance(h, logging.StreamHandler):
                    return
                return _STPIPE_ADDHANDLER_ORIG(h)

            stp.addHandler = _stpipe_add_handler_blocking_stream  # type: ignore[assignment]
    except Exception:
        pass

    # stpipe's import-time configuration (stpipe.log.load_configuration) may attach
    # StreamHandlers to *many* stpipe.* loggers, not just "stpipe". Block that at
    # the Logger class level so late-imported stpipe can't leak to console.
    global _LOGGER_ADDHANDLER_ORIG
    if block_stream_handlers and _LOGGER_ADDHANDLER_ORIG is None:
        _LOGGER_ADDHANDLER_ORIG = logging.Logger.addHandler

        def _logger_add_handler_block_stpipe_stream(self: logging.Logger, h: logging.Handler) -> None:
            try:
                if str(getattr(self, "name", "")).startswith("stpipe") and isinstance(
                    h, logging.StreamHandler
                ):
                    return
            except Exception:
                pass
            return _LOGGER_ADDHANDLER_ORIG(self, h)  # type: ignore[misc]

        logging.Logger.addHandler = _logger_add_handler_block_stpipe_stream  # type: ignore[assignment]


def _get_level(level):
    """
    Resolve a logging level name or number to an integer level.

    Parameters
    ----------
    level : int or str
        Level (e.g. 20, "INFO"). Invalid names fall back to INFO.

    Returns
    -------
    int
        logging level constant (e.g. logging.INFO).
    """
    if isinstance(level, int):
        return level
    result = logging.getLevelName(str(level).upper())
    return result if isinstance(result, int) else logging.INFO


def compress_logger_name(name: str) -> str:
    """
    Shorten logger *name* for console and session logs.

    - ``hst123`` package: drop the ``hst123.primitives.`` prefix, drop redundant
      ``*_primitive`` leafs (e.g. ``run_dolphot.run_dolphot_primitive`` →
      ``run_dolphot``), shorten ``utils.`` → ``u.``, strip a leading underscore
      from the first segment (``_pipeline`` → ``pipeline``).
    - Other packages: keep only the last dotted segment (e.g.
      ``stwcs.updatewcs.makewcs`` → ``makewcs``, ``stwcs.wcsutil.altwcs`` → ``altwcs``)
      to save space in session logs.

    Set ``HST123_LOG_FULL_NAMES=1`` to return *name* unchanged.
    """
    if not name:
        return ""
    if os.environ.get("HST123_LOG_FULL_NAMES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return name
    if name.startswith(ROOT_LOGGER):
        rest = name[len(ROOT_LOGGER) :].lstrip(".")
        if not rest:
            return ROOT_LOGGER
        if rest.startswith("primitives."):
            rest = rest[len("primitives.") :]
        parts = rest.split(".")
        if (
            len(parts) >= 2
            and parts[-1].endswith("_primitive")
            and parts[-1][: -len("_primitive")] == parts[-2]
        ):
            parts = parts[:-1]
        # utils.foo -> u.foo
        compact: list[str] = []
        i = 0
        while i < len(parts):
            if parts[i] == "utils" and i + 1 < len(parts):
                compact.append("u")
                compact.append(parts[i + 1])
                i += 2
            else:
                compact.append(parts[i])
                i += 1
        if compact and compact[0].startswith("_"):
            compact[0] = compact[0].lstrip("_")
        # Deep stwcs delegation chains (hst123.utils.stwcs.*) stay long; use last leaf only.
        if (
            len(compact) >= 3
            and compact[0] == "u"
            and compact[1] == "stwcs"
        ):
            return compact[-1]
        return ".".join(compact) if compact else ROOT_LOGGER
    parts = name.split(".")
    if len(parts) <= 2:
        return name
    return parts[-1]


class Hst123CompactFormatter(logging.Formatter):
    """Format records with a short ``compactname`` instead of full ``name``."""

    def format(self, record: logging.LogRecord) -> str:
        record.compactname = compress_logger_name(record.name)
        return super().format(record)


def _make_formatter():
    """
    Create the default log formatter ([timestamp][compactname][level]: message).

    Returns
    -------
    Hst123CompactFormatter
        Formatter with datefmt %Y-%m-%dT%H:%M:%S.
    """
    return Hst123CompactFormatter(
        "[%(asctime)s]"
        "[%(compactname)s]"
        "[%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


class RunIdFilter(logging.Filter):
    """Add run_id attribute to log records (current RUN_ID)."""

    def filter(self, record):
        """
        Set record.run_id and allow the record.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to modify.

        Returns
        -------
        bool
            True (record is always allowed).
        """
        record.run_id = RUN_ID
        return True


class PackageCaptureFilter(logging.Filter):
    """Allow records from ROOT_LOGGER or LOG_CAPTURE_PACKAGES; drop others when capturing."""

    def filter(self, record):
        """
        Allow record if name starts with ROOT_LOGGER or any LOG_CAPTURE_PACKAGES prefix.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to test.

        Returns
        -------
        bool
            True if record should be logged.
        """
        for pkg in LOG_CAPTURE_PACKAGES:
            if record.name.startswith(pkg):
                return True
        return record.name.startswith(ROOT_LOGGER)


class LogConfig:
    """
    Logging configuration: level, stdout/file handlers, rotation, and filters.

    Uses env vars HST123_LOG_LEVEL, HST123_LOG_ENABLE_STDOUT, HST123_LOG_ENABLE_FILE,
    HST123_LOG_DIR when options are None. Apply with apply() or context().
    """

    def __init__(
        self,
        *,
        level=None,
        enable_stdout=None,
        enable_file=None,
        log_dir=None,
        filename_prefix=DEFAULT_FILENAME_PREFIX,
        rotate=False,
        max_bytes=50_000_000,
        backup_count=5,
        log_stream=None,
    ):
        """
        Build LogConfig from arguments and environment.

        Parameters
        ----------
        level : int or str, optional
            Logging level (default from HST123_LOG_LEVEL or INFO).
        enable_stdout : bool, optional
            Add StreamHandler to stdout (default from HST123_LOG_ENABLE_STDOUT or True).
        enable_file : bool, optional
            Add file handler (default from HST123_LOG_ENABLE_FILE or False).
        log_dir : str, optional
            Directory for log files (default ~/hst123_logs when enable_file True).
        filename_prefix : str, optional
            Prefix for log filename. Default "hst123_log".
        rotate : bool, optional
            Use RotatingFileHandler. Default False.
        max_bytes : int, optional
            Max bytes per file when rotating. Default 50_000_000.
        backup_count : int, optional
            Number of backup files when rotating. Default 5.
        log_stream : file-like, optional
            Stream for ``StreamHandler`` when ``enable_stdout`` is True.
            Default ``sys.stdout``. CLI entry points use ``sys.stderr`` via
            ``ensure_cli_logging_configured``.
        """
        if level is None:
            level = os.getenv("HST123_LOG_LEVEL", "INFO")
        self.level = _get_level(level)

        if enable_stdout is None:
            enable_stdout = os.getenv(
                "HST123_LOG_ENABLE_STDOUT", "true"
            ).lower() in {"1", "true", "yes", "on"}
        self.enable_stdout = enable_stdout

        if enable_file is None:
            enable_file = os.getenv(
                "HST123_LOG_ENABLE_FILE", "false"
            ).lower() in {"1", "true", "yes", "on"}
        self.enable_file = enable_file

        if log_dir is None and self.enable_file:
            log_dir = os.getenv("HST123_LOG_DIR")
            if log_dir is None:
                log_dir = os.path.join(
                    os.path.expanduser("~"),
                    "hst123_logs",
                ) + os.sep

        self.filename_prefix = filename_prefix
        self.log_dir = log_dir
        self.rotate = rotate
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_stream = sys.stdout if log_stream is None else log_stream

        self.handlers = []
        self._previous_levels = {}

        self.formatter = _make_formatter()
        self.runid_filter = RunIdFilter()
        self.capture_filter = PackageCaptureFilter()

    def _make_file_handler(self, path):
        """
        Create a file handler for the given path (rotating or plain).

        Parameters
        ----------
        path : str
            Log file path.

        Returns
        -------
        logging.FileHandler or logging.handlers.RotatingFileHandler
        """
        if self.rotate:
            return RotatingFileHandler(
                path,
                mode="a",
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )
        return logging.FileHandler(path, mode="a")

    def _setup_handlers(self):
        """Build stdout and/or file handlers with formatter and filters."""
        handlers = []

        if self.enable_stdout:
            sh = logging.StreamHandler(self.log_stream)
            sh.setFormatter(self.formatter)
            sh.addFilter(self.runid_filter)
            sh.addFilter(self.capture_filter)
            handlers.append(sh)

        if self.enable_file and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(
                self.log_dir,
                f"{self.filename_prefix}_{RUN_ID}_{os.getpid()}.txt",
            )
            fh = self._make_file_handler(path)
            fh.setFormatter(self.formatter)
            fh.addFilter(self.runid_filter)
            fh.addFilter(self.capture_filter)
            handlers.append(fh)

        self.handlers = handlers

    def apply(self, log_names=None):
        """
        Attach handlers to loggers and set level.

        Parameters
        ----------
        log_names : list of str, optional
            Logger names to configure; default [ROOT_LOGGER]. Also applies to LOG_CAPTURE_PACKAGES.
        """
        if not self.handlers:
            self._setup_handlers()

        if log_names is None:
            log_names = [ROOT_LOGGER]

        for name in log_names:
            logger = logging.getLogger(name)
            self._previous_levels[name] = logger.level
            logger.setLevel(self.level)

            for h in self.handlers:
                if h not in logger.handlers:
                    logger.addHandler(h)

        if ROOT_LOGGER in log_names:
            _apply_hst123_propagate_policy()

        for lname in logging.root.manager.loggerDict:
            if any(lname.startswith(p) for p in LOG_CAPTURE_PACKAGES):
                logger = logging.getLogger(lname)
                for h in self.handlers:
                    if h not in logger.handlers:
                        logger.addHandler(h)

    def undo(self, log_names=None, close_handlers=True):
        """
        Remove handlers from loggers and restore previous levels.

        Parameters
        ----------
        log_names : list of str, optional
            Logger names; default [ROOT_LOGGER].
        close_handlers : bool, optional
            Close handlers after removal. Default True.
        """
        if log_names is None:
            log_names = [ROOT_LOGGER]

        for name in log_names:
            logger = logging.getLogger(name)
            for h in self.handlers:
                logger.removeHandler(h)

            if name in self._previous_levels:
                logger.setLevel(self._previous_levels[name])

        for lname in logging.root.manager.loggerDict:
            if any(lname.startswith(p) for p in LOG_CAPTURE_PACKAGES):
                logger = logging.getLogger(lname)
                for h in self.handlers:
                    logger.removeHandler(h)

        for h in self.handlers:
            h.flush()
            if close_handlers:
                h.close()

        self.handlers = []

    @contextmanager
    def context(self, log_names=None):
        """
        Context manager: apply config on enter, undo on exit.

        Parameters
        ----------
        log_names : list of str, optional
            Passed to apply() and undo().
        """
        self.apply(log_names)
        try:
            yield
        finally:
            self.undo(log_names)


def _start_listener(queue, config: LogConfig):
    """
    Start the QueueListener with config handlers (for multiprocessing logging).

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue for log records.
    config : LogConfig
        Config whose handlers are used by the listener.
    """
    global _LISTENER

    if not config.handlers:
        config._setup_handlers()

    _LISTENER = QueueListener(queue, *config.handlers)
    _LISTENER.start()


def _stop_listener():
    """Stop the global QueueListener if it was started."""
    global _LISTENER
    if _LISTENER is not None:
        _LISTENER.stop()
        _LISTENER = None


def _configure_process(queue, level):
    """
    Configure root logger for a worker process to send records to queue.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue to send log records to.
    level : int or str
        Root logger level.
    """
    root = logging.getLogger()
    root.setLevel(_get_level(level))
    root.handlers.clear()
    root.addHandler(QueueHandler(queue))


@contextmanager
def logging_context(config_dict=None, queue=None):
    """
    Context manager to configure logging (optionally with a multiprocessing queue).

    Parameters
    ----------
    config_dict : dict, optional
        Config dict; "logging" key used for LogConfig (level, enable_stdout, etc.).
    queue : multiprocessing.Queue, optional
        If provided, used instead of creating a new queue (e.g. for multiprocessing).

    Yields
    ------
    None
    """
    global _QUEUE, _CONFIGURED

    if not _CONFIGURED:
        if queue is None:
            _QUEUE = mp.Queue(-1)
        else:
            _QUEUE = queue

        log_cfg = (config_dict or {}).get("logging", {})

        cfg = LogConfig(
            level=log_cfg.get("level", "INFO"),
            enable_stdout=log_cfg.get("enable_stdout"),
            enable_file=log_cfg.get("enable_file"),
            log_dir=log_cfg.get("log_dir"),
            filename_prefix=log_cfg.get(
                "filename_prefix", DEFAULT_FILENAME_PREFIX
            ),
            rotate=log_cfg.get("rotate", False),
            max_bytes=log_cfg.get("max_bytes", 50_000_000),
            backup_count=log_cfg.get("backup_count", 5),
        )

        _configure_process(_QUEUE, cfg.level)

        if queue is None:
            _start_listener(_QUEUE, cfg)

        _CONFIGURED = True

    try:
        yield
    finally:
        _stop_listener()
        _CONFIGURED = False


def get_queue():
    """
    Return the global logging queue (set by logging_context).

    Returns
    -------
    multiprocessing.Queue or None
        The queue used by the listener, or None if not configured.
    """
    return _QUEUE


def get_logger(name=None):
    """
    Return a logger for the given name (default ROOT_LOGGER).

    Parameters
    ----------
    name : str, optional
        Logger name; default ROOT_LOGGER ("hst123").

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name if name is not None else ROOT_LOGGER)


def ensure_cli_logging_configured(*, level=None):
    """
    Install a default ``hst123`` handler on first use (stderr, formatted).

    Safe to call multiple times (idempotent for handlers). Use at CLI entry
    (``python -m hst123``, ``hst123-install-dolphot``) so package output is
    mediated through this module.

    Parameters
    ----------
    level : int, str, optional
        Logger level (e.g. ``logging.DEBUG`` for ``--verbose``). When handlers
        already exist, only the level is updated.
    """
    global _CLI_LOGGING_INSTALLED
    root = logging.getLogger(ROOT_LOGGER)
    env_level = os.environ.get("HST123_LOG_LEVEL", "INFO")
    eff = level if level is not None else env_level
    root.setLevel(_get_level(eff))
    if root.handlers:
        _apply_hst123_propagate_policy()
        _CLI_LOGGING_INSTALLED = True
        return
    cfg = LogConfig(
        level=eff,
        enable_stdout=True,
        enable_file=False,
        log_stream=sys.stderr,
    )
    cfg.apply(log_names=[ROOT_LOGGER])
    # apply() already calls _apply_hst123_propagate_policy when configuring hst123
    _CLI_LOGGING_INSTALLED = True


def attach_work_dir_log_file(
    work_dir: str | os.PathLike[str] | None,
    *,
    process_name: str = "pipeline",
    level=None,
) -> str | None:
    """
    Mirror ``hst123`` logger output to ``<work_dir>/logs/hst123_<process>_<time>_<pid>.log``.

    Creates ``logs`` under the work directory. Safe to call once per process;
    a second call returns the path of the existing handler without adding another.

    Parameters
    ----------
    work_dir
        Absolute or relative work directory (typically ``opt.work_dir``).
    process_name
        Short label for the filename (e.g. ``pipeline``, ``install_dolphot``).
    level
        Handler level; default matches ``hst123`` logger effective level.

    Returns
    -------
    str or None
        Path to the log file, or None if *work_dir* is missing/invalid.
    """
    global _WORK_DIR_LOG_HANDLER, _WORK_DIR_LOG_ATEXIT_REGISTERED
    if not work_dir:
        return None
    wd = os.path.abspath(os.path.expanduser(os.fspath(work_dir)))
    if not os.path.isdir(wd):
        try:
            os.makedirs(wd, exist_ok=True)
        except OSError:
            return None

    root = logging.getLogger(ROOT_LOGGER)
    if _WORK_DIR_LOG_HANDLER is not None:
        if _WORK_DIR_LOG_HANDLER in root.handlers:
            return getattr(_WORK_DIR_LOG_HANDLER, "_hst123_log_path", None)
        _WORK_DIR_LOG_HANDLER = None

    log_subdir = os.path.join(wd, "logs")
    os.makedirs(log_subdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in process_name)
    basename = f"hst123_{safe}_{stamp}_{os.getpid()}.log"
    path = os.path.join(log_subdir, basename)

    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(_make_formatter())
    fh.addFilter(RunIdFilter())
    fh.addFilter(PackageCaptureFilter())
    eff = level if level is not None else root.level
    fh.setLevel(eff)
    root.addHandler(fh)
    fh._hst123_log_path = path  # type: ignore[attr-defined]
    _WORK_DIR_LOG_HANDLER = fh
    if not _WORK_DIR_LOG_ATEXIT_REGISTERED:
        atexit.register(_flush_work_dir_session_log)
        _WORK_DIR_LOG_ATEXIT_REGISTERED = True
    root.info("Session log: %s", os.path.basename(path))
    fh.flush()
    _install_session_log_stream_mirrors(path)
    return path


EXTERNAL_LOGGER = "hst123.external"
ASTRODRIZZLE_DETAIL_LOGGER = "hst123.astrodrizzle"
PHOTEQ_DETAIL_LOGGER = "hst123.photeq"
# Env HST123_REPLAY_SUBLOGS in these values → one-line summary only (default is full replay).
_REPLAY_SUBLOGS_SUMMARY_VALUES = frozenset(
    ("0", "false", "no", "off", "summary", "compact")
)


def _ingest_compact_line(text: str) -> str:
    """Collapse internal whitespace to single spaces for denser log lines."""
    return re.sub(r"\s+", " ", text.strip())


def ephemeral_pipeline_runfile(work_dir: str, stem: str) -> str:
    """
    Path under ``<work_dir>/.hst123_runfiles/`` for drizzlepac/photeq runfiles.

    Intended flow: pass path to C extensions, :func:`ingest_text_file_to_logger`,
    then delete the file so nothing is left in the work directory root.
    """
    base = os.path.abspath(os.path.expanduser(work_dir or "."))
    d = os.path.join(base, ".hst123_runfiles")
    os.makedirs(d, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return os.path.join(d, f"{safe}_{stamp}_{os.getpid()}.log")


def ingest_text_file_to_logger(
    path: str | os.PathLike[str],
    logger: logging.Logger,
    *,
    level: int | str | None = None,
    missing_ok: bool = True,
    encoding: str = "utf-8",
    errors: str = "replace",
    begin_end_markers: bool = True,
    log_tag: str = "astrodrizzle log",
    replay_full: bool | None = None,
    compact_ws: bool = False,
    delete_after: bool = False,
) -> int:
    """
    Record a text file in *logger*: by default **full line-by-line** replay.

    Used after AstroDrizzle (``astrodrizzle.log``) and photeq (``photeq.log``).
    Set environment ``HST123_REPLAY_SUBLOGS=0``, ``false``, or ``summary`` (or pass
    ``replay_full=False``) for a one-line summary only; full runfiles stay on disk.

    Parameters
    ----------
    path
        File to read.
    logger
        Target logger (e.g. ``get_logger(ASTRODRIZZLE_DETAIL_LOGGER)``).
    level
        Level for summary (default mode) or each line (replay mode); default INFO.
    missing_ok
        If True, no record when *path* is missing; if False, log a warning.
    encoding, errors
        Passed to :func:`open`.
    begin_end_markers
        If True and *replay_full*, log delimiter lines before and after the body.
    log_tag
        Short label (e.g. ``"astrodrizzle log"``, ``"photeq"``).
    replay_full
        If True, log every non-empty line. If None, default is full replay unless
        ``HST123_REPLAY_SUBLOGS`` is ``0`` / ``false`` / ``summary`` / ``compact`` / ….
    compact_ws
        If True (replay mode only), collapse runs of whitespace in each logged line.
    delete_after
        If True, remove *path* from disk after a successful read (replay or summary).

    Returns
    -------
    int
        Number of non-empty lines in the file.
    """
    lv = _get_level(level) if level is not None else logging.INFO
    p = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(p):
        if not missing_ok:
            logger.warning("Expected log file not found: %s", p)
        return 0
    tag = log_tag.strip() or "file"
    if replay_full is None:
        v = os.getenv("HST123_REPLAY_SUBLOGS", "1").strip().lower()
        replay_full = v not in _REPLAY_SUBLOGS_SUMMARY_VALUES

    if not replay_full:
        n_lines = 0
        try:
            with open(p, encoding=encoding, errors=errors) as fh:
                for line in fh:
                    if line.strip():
                        n_lines += 1
        except OSError as exc:
            logger.warning("Could not read %s: %s", p, exc)
            return 0
        logger.log(
            lv,
            "[%s] %d lines %s (full log on disk; HST123_REPLAY_SUBLOGS=0/summary for compact)",
            tag,
            n_lines,
            p,
        )
        if delete_after:
            try:
                os.unlink(p)
            except OSError:
                pass
        return n_lines

    n_lines = 0
    if begin_end_markers:
        logger.log(lv, "[%s] --- begin %s ---", tag, p)
    try:
        with open(p, encoding=encoding, errors=errors) as fh:
            for line in fh:
                text = line.rstrip("\n\r")
                if text.strip():
                    out = _ingest_compact_line(text) if compact_ws else text
                    logger.log(lv, "[%s] %s", tag, out)
                    n_lines += 1
    except OSError as exc:
        logger.warning("Could not read %s: %s", p, exc)
        return n_lines
    if begin_end_markers:
        logger.log(
            lv,
            "[%s] --- end %s (%d line(s)) ---",
            tag,
            p,
            n_lines,
        )
    if delete_after:
        try:
            os.unlink(p)
        except OSError:
            pass
    return n_lines


def run_external_command(
    cmd: str | list[str] | tuple[str, ...],
    *,
    log: logging.Logger | None = None,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    shell: bool = False,
    tee_path: str | os.PathLike[str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess and send its combined stdout/stderr through *log*.

    stdout and stderr are merged so interleaved progress lines stay ordered.
    Each non-empty output line is logged at INFO under the ``hst123.external``
    namespace (or the logger you pass). If *tee_path* is set, the same bytes are
    written to that file (typical for DOLPHOT console logs).

    Parameters
    ----------
    cmd
        Argument list (preferred), or a string when *shell* is True.
    log
        Logger for command start/finish and each output line; default
        ``get_logger("hst123.external")``.
    cwd, env
        Passed to :class:`subprocess.Popen`.
    shell
        If True, *cmd* must be a string and is run through the shell.
    tee_path
        If set, open for writing (truncates like shell ``>``) and duplicate
        stdout there.
    check
        If True (default), raise :exc:`subprocess.CalledProcessError` when the
        exit code is non-zero.

    Returns
    -------
    subprocess.CompletedProcess
        ``stdout`` holds the captured combined output; ``stderr`` is ``""``.

    Raises
    ------
    subprocess.CalledProcessError
        If *check* is True and the process exits non-zero.
    """
    lg = log or get_logger(EXTERNAL_LOGGER)

    if shell:
        if not isinstance(cmd, str):
            raise TypeError("run_external_command: shell=True requires cmd str")
        argv_display = cmd
        popen_args: str | list[str] = cmd
        exe_tag = "shell"
        toks = cmd.strip().split()
        if toks:
            exe_tag = os.path.basename(toks[0])
    else:
        if isinstance(cmd, str):
            argv = shlex.split(cmd)
        else:
            argv = list(cmd)
        if not argv:
            raise ValueError("run_external_command: empty command")
        popen_args = argv
        argv_display = subprocess.list2cmdline(argv)
        exe_tag = os.path.basename(argv[0])

    lg.info("[%s] %s", exe_tag, argv_display)

    tee_file = None
    if tee_path is not None:
        tee_file = open(
            os.path.expanduser(tee_path),
            "w",
            encoding="utf-8",
            errors="replace",
        )

    proc = subprocess.Popen(
        popen_args,
        shell=shell,
        cwd=os.fspath(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    chunks: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            if tee_file is not None:
                tee_file.write(line)
            # Split embedded newlines; collapse runs of whitespace per log line
            text = line.replace("\r\n", "\n").replace("\r", "\n")
            for piece in text.split("\n"):
                s = " ".join(piece.split())
                if s:
                    lg.info("%s", s)
    finally:
        if tee_file is not None:
            tee_file.close()

    ret = proc.wait()
    out = "".join(chunks)

    if check and ret != 0:
        tail = ""
        if out:
            lines = out.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            tail = "\n".join(lines[-25:]).strip()
        if tail:
            lg.error(
                "External command failed (exit %s): %s\n--- last output lines ---\n%s",
                ret,
                argv_display,
                tail,
            )
        else:
            lg.error(
                "External command failed (exit %s): %s",
                ret,
                argv_display,
            )
        raise subprocess.CalledProcessError(ret, popen_args, output=out)

    lg.debug("Finished external [%s] (exit 0)", exe_tag)
    return subprocess.CompletedProcess(popen_args, ret, stdout=out, stderr="")


def log_calls(
    _fn=None,
    *,
    level=logging.DEBUG,
    logger_name: str | None = None,
    log_arguments: bool = False,
):
    """
    Decorator: log callable entry/exit at *level* (default DEBUG).

    Visible when ``HST123_LOG_LEVEL=DEBUG`` (or equivalent). Use on pipeline
    methods for call tracing without flooding INFO-level CLI output.

    Parameters
    ----------
    log_arguments : bool, optional
        If True, log ``repr`` of positional and keyword arguments (verbose).
    """

    def decorator(fn):
        lg_name = logger_name or getattr(fn, "__module__", ROOT_LOGGER)
        lg = logging.getLogger(lg_name)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            qual = getattr(fn, "__qualname__", fn.__name__)
            if log_arguments:
                lg.log(level, "→ %s(%r, %r)", qual, args, kwargs)
            else:
                lg.log(level, "→ %s", qual)
            try:
                out = fn(*args, **kwargs)
                lg.log(level, "← %s", qual)
                return out
            except Exception:
                lg.log(level, "← %s (raising)", qual, exc_info=True)
                raise

        return wrapper

    if _fn is not None and callable(_fn):
        return decorator(_fn)
    return decorator


# ANSI colors and banner for pipeline status

green = "\033[1;32;40m"
red = "\033[1;31;40m"
end = "\033[0;0m"


def make_banner(message):
    """
    Log a compact section marker (no multi-line hash banners).

    Parameters
    ----------
    message : str
        Banner text.
    """
    _banner_logger = logging.getLogger(ROOT_LOGGER)
    msg = " ".join(str(message).split())
    _banner_logger.info("— %s —", msg)


def format_hdu_list_summary(hdulist, *, max_show=10):
    """
    One-line summary of an Astropy ``HDUList`` (replaces noisy ``HDUList.info()``).

    Parameters
    ----------
    hdulist
        Open ``astropy.io.fits.HDUList``.
    max_show : int
        Max extensions to list before ``+N more``.

    Returns
    -------
    str
        e.g. ``19 ext [PRIMARY:1, SCI:1, …]``.
    """
    try:
        n = len(hdulist)
    except Exception:
        return "HDU (?)"
    parts = []
    for j, h in enumerate(hdulist):
        if j >= max_show:
            parts.append(f"+{n - max_show} more")
            break
        nm = (getattr(h, "name", "") or "PRIMARY").strip() or "PRIMARY"
        try:
            ver = int(getattr(h, "ver", 1))
        except (TypeError, ValueError):
            ver = 1
        parts.append(f"{nm}:{ver}")
    return f"{n} ext [{', '.join(parts)}]"


def log_pipeline_phase_summary(
    logger,
    phases: list[tuple[str, float]],
    *,
    wall_seconds: float,
) -> None:
    """
    Log one INFO line with per-phase durations (seconds) and wall time.

    *phases* is a list of (label, elapsed_seconds) from consecutive perf_counter
    segments (see :func:`main` in ``hst123._pipeline``).
    """
    if not phases:
        logger.info("Pipeline timing: (no phases recorded) | wall=%.1fs", wall_seconds)
        return
    parts = [f"{name}={dt:.1f}s" for name, dt in phases]
    summed = sum(dt for _, dt in phases)
    logger.info(
        "Pipeline timing: %s | segments_sum=%.1fs wall=%.1fs",
        " | ".join(parts),
        summed,
        wall_seconds,
    )


def log_pipeline_configuration(logger, opt, *, version, coord_hmsdms, cwd=None):
    """
    Log resolved paths, coordinate, and effective CLI options (grouped).

    Parameters
    ----------
    logger : logging.Logger
        Typically ``hst123`` pipeline logger.
    opt : argparse.Namespace
        Parsed CLI options after ``handle_args``.
    version : str
        Package version string.
    coord_hmsdms : str
        Target position, e.g. from ``SkyCoord.to_string('hmsdms')``.
    cwd : str, optional
        Process working directory; default ``os.getcwd()``.
    """
    if cwd is None:
        cwd = os.getcwd()
    work_abs = (
        os.path.abspath(os.path.expanduser(opt.work_dir))
        if getattr(opt, "work_dir", None)
        else cwd
    )
    raw_abs = (
        os.path.abspath(os.path.expanduser(opt.raw_dir))
        if getattr(opt, "raw_dir", None)
        else os.path.join(work_abs, "raw")
    )

    sel = []
    if opt.before:
        sel.append(f"before={opt.before}")
    if opt.after:
        sel.append(f"after={opt.after}")
    if opt.only_filter:
        sel.append(f"only_filter={opt.only_filter}")
    if opt.only_wide:
        sel.append("only_wide")
    if opt.keep_short:
        sel.append("keep_short")
    sel_s = ",".join(sel) if sel else "—"

    logger.info(
        "hst123 %s | %s | work=%s | raw=%s | %s",
        version,
        coord_hmsdms,
        work_abs,
        raw_abs,
        sys.executable,
    )
    max_cores = getattr(opt, "drizzle_num_cores", None)
    if max_cores is None:
        max_cores = 1
    logger.info(
        "MAST dl=%s clob=%s arch=%s [%s] | align=%s skip_tr=%s hier=%s | "
        "drizzle=%s redriz=%s dim=%s by_vis=%s max_cores=%s | dp run=%s scrape=%s hdf5=%s %s lim=%s "
        "clean=%s fake=%s | keep_driz_art=%s keep_obj=%s | redo=%s redo_dp=%s redo_a=%s redo_d=%s",
        opt.download,
        opt.clobber,
        getattr(opt, "archive", None) or "—",
        sel_s,
        opt.align_with,
        opt.skip_tweakreg,
        opt.hierarchical,
        opt.drizzle_all,
        opt.redrizzle,
        opt.drizzle_dim,
        opt.by_visit,
        max_cores,
        opt.run_dolphot,
        opt.scrape_dolphot,
        getattr(opt, "write_dolphot_hdf5", True),
        opt.dolphot,
        opt.dolphot_lim,
        opt.cleanup,
        opt.do_fake,
        getattr(opt, "keep_drizzle_artifacts", False),
        getattr(opt, "keep_objfile", False),
        getattr(opt, "redo", False),
        getattr(opt, "redo_dolphot", False),
        getattr(opt, "redo_astrometry", False),
        getattr(opt, "redo_astrodrizzle", False),
    )


def format_success(prefix):
    """
    Return a colored success line for status messages (e.g. download).

    Parameters
    ----------
    prefix : str
        The status line shown before the result (e.g. "Downloading file: ...").

    Returns
    -------
    str
        Full line with \\r prefix and green [SUCCESS] suffix, ready to write to stdout.
    """
    return "\r" + prefix + green + " [SUCCESS]" + end + "\n"


def format_failure(prefix):
    """
    Return a colored failure line for status messages (e.g. download).

    Parameters
    ----------
    prefix : str
        The status line shown before the result (e.g. "Downloading file: ...").

    Returns
    -------
    str
        Full line with \\r prefix and red [FAILURE] suffix, ready to write to stdout.
    """
    return "\r" + prefix + red + " [FAILURE]" + end + "\n"
