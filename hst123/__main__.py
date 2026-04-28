"""
CLI entry when running ``python -m hst123``.

Configures logging before importing the pipeline so early messages use the same
handlers as the rest of the run.
"""
from hst123.utils.logging import (
    ensure_cli_logging_configured,
    get_logger,
    capture_root_logging_to_hst123,
)

# Configure logging before importing the heavy stack so early messages use the
# same handlers as the rest of the run (stderr, ``hst123`` logger).
if __name__ == "__main__":
    ensure_cli_logging_configured()
    # Capture any third-party root logger output (stpipe/drizzlepac) into hst123 logs
    # and prevent them from attaching their own console StreamHandlers.
    capture_root_logging_to_hst123()
    get_logger("hst123.cli").info(
        "Loading dependencies (drizzlepac, astroquery, …) — please wait…"
    )

from hst123._pipeline import main

if __name__ == "__main__":
    # Re-assert policy after heavy imports (some stacks attach handlers late).
    capture_root_logging_to_hst123()
    main()
