"""
Throttled progress reporting through the standard logging stack.

Use :class:`LoggedProgress` for long-running steps so log files and console
handlers receive periodic lines with a text bar, fraction complete, elapsed
time, and ETA (linear extrapolation from progress so far).

Typical pattern::

    from hst123.utils.logging import get_logger
    from hst123.utils.progress_log import LoggedProgress

    log = get_logger(__name__)
    with LoggedProgress(log, "my-step", total=len(items), unit="items") as prog:
        for i, item in enumerate(items):
            process(item)
            prog.update(i + 1)

Disable with environment variable ``HST123_PROGRESS_LOG=0`` (also respected by
calcsky via the same check inside :func:`calcsky_progress_enabled`).
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any


def progress_log_enabled() -> bool:
    """False if ``HST123_PROGRESS_LOG`` is ``0``/``false``/``no``/``off``."""
    raw = os.environ.get("HST123_PROGRESS_LOG", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def calcsky_progress_enabled() -> bool:
    """Calcsky may use ``HST123_CALCSKY_PROGRESS=0`` to disable; else ``progress_log_enabled()``."""
    raw = os.environ.get("HST123_CALCSKY_PROGRESS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return progress_log_enabled()


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "…"
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    if seconds >= 10:
        return f"{seconds:.0f}s"
    return f"{seconds:.1f}s"


def _bar(fraction: float, width: int) -> str:
    frac = max(0.0, min(1.0, fraction))
    filled = int(round(frac * width))
    # ASCII-safe for all consoles / log viewers
    return "#" * filled + "-" * (width - filled)


class LoggedProgress:
    """
    Emit throttled ``logger.info`` lines with bar, %, counts, elapsed, ETA.

    Parameters
    ----------
    logger
        Logger used for output (same formatting as rest of hst123).
    label
        Short name prepended to each line (e.g. ``"calcsky stage1"``).
    total
        Total work units (must be >= 1).
    min_interval
        Minimum seconds between emissions (unless ``force`` or ``complete``).
    min_fraction_delta
        Minimum increase in completion fraction to emit (0–1).
    unit
        Noun for counts in ``(current/total unit)``.
    bar_width
        Character width of the ``#---`` bar.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        total: int,
        *,
        min_interval: float = 0.85,
        min_fraction_delta: float = 0.015,
        unit: str = "units",
        bar_width: int = 28,
    ) -> None:
        self._log = logger
        self._label = label
        self._total = max(1, int(total))
        self._current = 0
        self._min_interval = float(min_interval)
        self._min_fraction_delta = float(min_fraction_delta)
        self._unit = unit
        self._bar_width = int(bar_width)
        self._t0 = time.monotonic()
        self._last_emit = self._t0
        self._last_frac = -1.0
        self._started = False

    def __enter__(self) -> LoggedProgress:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.complete()
        else:
            self._emit(force=True, suffix=" (interrupted)")

    def start(self) -> None:
        """Log a start line with expected total (0% baseline)."""
        if self._started:
            return
        self._started = True
        self._log.info(
            "%s |%s| 0.0%% (0/%d %s) — started, total work units=%d",
            self._label,
            _bar(0.0, self._bar_width),
            self._total,
            self._unit,
            self._total,
        )
        self._last_emit = time.monotonic()
        self._last_frac = 0.0

    def update(self, completed: int, *, force: bool = False) -> None:
        """Set absolute progress to *completed* (clamped to ``[0, total]``)."""
        self._current = max(0, min(int(completed), self._total))
        self._emit(force=force)

    def tick(self, n: int = 1, *, force: bool = False) -> None:
        """Increment progress by *n*."""
        self.update(self._current + n, force=force)

    def complete(self) -> None:
        """Mark 100%% and force a final log line (skipped if already reported ~100%%)."""
        self._current = self._total
        if self._last_frac >= 1.0 - 1e-9:
            return
        self._emit(force=True)

    def _emit(self, *, force: bool = False, suffix: str = "") -> None:
        now = time.monotonic()
        frac = self._current / self._total
        if not force and frac < 1.0 - 1e-12:
            dt = now - self._last_emit
            dfrac = frac - self._last_frac
            if dt < self._min_interval and dfrac < self._min_fraction_delta:
                return

        elapsed = now - self._t0
        if frac > 1e-9:
            est_total_dur = elapsed / frac
            remaining = max(0.0, est_total_dur - elapsed)
        else:
            est_total_dur = float("nan")
            remaining = float("nan")

        bar = _bar(frac, self._bar_width)
        rem_s = _format_duration(remaining) if math.isfinite(remaining) else "…"
        tot_s = _format_duration(est_total_dur) if math.isfinite(est_total_dur) else "…"
        el_s = _format_duration(elapsed)

        self._log.info(
            "%s |%s| %.1f%% (%d/%d %s) elapsed=%s est_total≈%s remaining≈%s%s",
            self._label,
            bar,
            100.0 * frac,
            self._current,
            self._total,
            self._unit,
            el_s,
            tot_s,
            rem_s,
            suffix,
        )
        self._last_emit = now
        self._last_frac = frac


def null_progress() -> Any:
    """No-op object with ``start/update/tick/complete`` for optional progress."""
    class _NP:
        def start(self) -> None:
            pass

        def update(self, *_a, **_k) -> None:
            pass

        def tick(self, *_a, **_k) -> None:
            pass

        def complete(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    return _NP()
