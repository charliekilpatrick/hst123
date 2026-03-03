"""Logging setup (LogConfig, get_logger, make_banner) and status formatting."""
import logging
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

_QUEUE = None
_LISTENER = None
_CONFIGURED = False

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
            sh = logging.StreamHandler(sys.stdout)
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


# ANSI colors and banner for pipeline status

green = "\033[1;32;40m"
red = "\033[1;31;40m"
end = "\033[0;0m"


def make_banner(message):
    """
    Log a banner message (section header with # lines).

    Parameters
    ----------
    message : str
        Banner text; logged with 80-char # lines above and below.
    """
    _banner_logger = logging.getLogger(ROOT_LOGGER)
    _banner_logger.info("\n\n%s\n%s\n%s\n\n", message, "#" * 80, "#" * 80)


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
