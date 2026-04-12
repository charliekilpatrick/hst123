"""
Shared utilities: logging, CLI options, display, visits, WCS, progress, and STScI WCS shims.

The package re-exports commonly used helpers so callers can use
``from hst123.utils import get_logger, …`` without importing each submodule.

Exports
-------
See ``__all__`` for the stable names exposed at package level. Submodules such as
:mod:`hst123.utils.astrodrizzle_helpers` and :mod:`hst123.utils.options` contain
additional functions used by the pipeline but not re-exported here.

Notes
-----
Bundled STScI WCS code lives under :mod:`hst123.utils.stwcs` and is accessed via
:mod:`hst123.utils.stsci_wcs` (not the PyPI ``stwcs`` package).
"""
from hst123.utils.logging import (
    LogConfig,
    attach_work_dir_log_file,
    ensure_cli_logging_configured,
    format_failure,
    format_success,
    get_logger,
    get_queue,
    log_calls,
    logging_context,
    make_banner,
    run_external_command,
)
from hst123.utils import options
from hst123.utils.stdio import suppress_stdout, suppress_stdout_fd
from hst123.utils.display import (
    format_instrument_display_name,
    show_photometry_data,
    write_snana_photometry,
    show_photometry,
)
from hst123.utils.visit import add_visit_info
from hst123.utils.wcs_utils import make_meta_wcs_header
from hst123.utils.progress_log import (
    LoggedProgress,
    calcsky_progress_enabled,
    null_progress,
    progress_log_enabled,
)

__all__ = [
    "format_success",
    "format_failure",
    "make_banner",
    "get_logger",
    "get_queue",
    "attach_work_dir_log_file",
    "ensure_cli_logging_configured",
    "run_external_command",
    "log_calls",
    "LogConfig",
    "logging_context",
    "options",
    "suppress_stdout",
    "suppress_stdout_fd",
    "format_instrument_display_name",
    "show_photometry_data",
    "write_snana_photometry",
    "show_photometry",
    "add_visit_info",
    "make_meta_wcs_header",
    "LoggedProgress",
    "calcsky_progress_enabled",
    "null_progress",
    "progress_log_enabled",
]
