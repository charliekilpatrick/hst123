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
calls ``attach_work_dir_log_file()`` to create ``<work-dir>/logs/`` and append a
``FileHandler`` so the same formatted records as stderr go to a unique file
(``hst123_<process>_<timestamp>_<pid>.log``).

**External programs** (``dolphot``, ``calcsky``, ``make``, …): use
``run_external_command()`` so their stdout/stderr is merged, streamed to the
logger (INFO), and optionally copied to a file (e.g. DOLPHOT ``.output``).

Third-party libraries (e.g. drizzlepac, astroquery) may still write to stdout/stderr
directly; some call sites use ``suppress_stdout`` where needed. AstroDrizzle and photeq runfiles live under ``<work-dir>/.hst123_runfiles/``, are replayed into
loggers ``hst123.astrodrizzle`` / ``hst123.photeq``, then removed. The pipeline **always** replays those runfiles into the session log with compact whitespace (not controlled by ``HST123_REPLAY_SUBLOGS``). For other uses of :func:`ingest_text_file_to_logger`, default (**``HST123_REPLAY_SUBLOGS``** unset or ``1``) is full replay; set ``HST123_REPLAY_SUBLOGS=0`` or ``summary`` for a one-line summary only.

Environment variables: ``HST123_LOG_LEVEL`` (use ``DEBUG`` to show ``@log_calls``
entry/exit lines), ``HST123_LOG_ENABLE_STDOUT``, ``HST123_LOG_ENABLE_FILE``,
``HST123_LOG_DIR``, ``HST123_REPLAY_SUBLOGS``.
"""
from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

_QUEUE = None
_LISTENER = None
_CONFIGURED = False
_CLI_LOGGING_INSTALLED = False
_WORK_DIR_LOG_HANDLER = None

RUN_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
ROOT_LOGGER = "hst123"
LOG_CAPTURE_PACKAGES = []
DEFAULT_FILENAME_PREFIX = "hst123_log"


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


def _make_formatter():
    """
    Create the default log formatter ([timestamp][name][level]: message).

    Returns
    -------
    logging.Formatter
        Formatter with datefmt %Y-%m-%dT%H:%M:%S.
    """
    return logging.Formatter(
        "[%(asctime)s]"
        "[%(name)s]"
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
        _CLI_LOGGING_INSTALLED = True
        return
    cfg = LogConfig(
        level=eff,
        enable_stdout=True,
        enable_file=False,
        log_stream=sys.stderr,
    )
    cfg.apply(log_names=[ROOT_LOGGER])
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
    global _WORK_DIR_LOG_HANDLER
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
    root.info("Session log (copy of console): %s", path)
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
    logger.info(
        "MAST dl=%s clob=%s arch=%s [%s] | align=%s skip_tr=%s hier=%s | "
        "drizzle=%s redriz=%s dim=%s by_vis=%s | dp run=%s scrape=%s %s lim=%s "
        "clean=%s fake=%s | keep_driz_art=%s keep_obj=%s",
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
        opt.run_dolphot,
        opt.scrape_dolphot,
        opt.dolphot,
        opt.dolphot_lim,
        opt.cleanup,
        opt.do_fake,
        getattr(opt, "keep_drizzle_artifacts", False),
        getattr(opt, "keep_objfile", False),
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
