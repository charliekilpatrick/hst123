"""
Post-step cleanup and output validation for pipeline primitives.

Each primitive should call :func:`run_primitive_cleanup` (or ``BasePrimitive._primitive_cleanup``)
after mutating the work directory or producing FITS/text products, so interstitial files are
removed and outputs are checked, with actions logged for transparency.
"""
from __future__ import annotations

import glob
import logging
import os
import statistics
from typing import Any, Callable, Iterable, Sequence


def _safe_keep_artifacts(pipeline: Any) -> bool:
    try:
        args = pipeline.options.get("args")
        if args is None:
            return False
        return bool(getattr(args, "keep_drizzle_artifacts", False))
    except Exception:
        return False


def remove_interstitial_files(
    work_dir: str | None,
    globs: Sequence[str],
    explicit_paths: Sequence[str],
    logger: logging.Logger,
    *,
    step_name: str,
    skip_removal: bool = False,
) -> tuple[int, list[str]]:
    """
    Remove files matching *globs* (under *work_dir*) and *explicit_paths*.

    Returns
    -------
    tuple
        (count_removed, list of basenames removed).
    """
    if skip_removal or not work_dir:
        return 0, []
    wd = os.path.abspath(os.path.expanduser(work_dir))
    if not os.path.isdir(wd):
        return 0, []
    removed: list[str] = []
    for pattern in globs:
        for path in glob.glob(os.path.join(wd, pattern)):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed.append(os.path.basename(path))
            except OSError as exc:
                logger.debug(
                    "[primitive cleanup] %s: could not remove %s: %s",
                    step_name,
                    path,
                    exc,
                )
    for path in explicit_paths:
        ap = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(ap):
            continue
        try:
            os.remove(ap)
            removed.append(os.path.basename(ap))
        except OSError as exc:
            logger.debug(
                "[primitive cleanup] %s: could not remove %s: %s",
                step_name,
                ap,
                exc,
            )
    if removed:
        logger.info(
            "[primitive cleanup] %s: removed %d interstitial file(s): %s",
            step_name,
            len(removed),
            ", ".join(sorted(set(removed))[:25])
            + (" …" if len(set(removed)) > 25 else ""),
        )
    else:
        logger.info(
            "[primitive cleanup] %s: interstitial sweep (no extra files to remove)",
            step_name,
        )
    return len(removed), removed


def validate_fits_outputs(
    paths: Iterable[str],
    logger: logging.Logger,
    *,
    step_name: str,
    require_basic_wcs: bool = False,
) -> tuple[int, int]:
    """
    Open each path and ensure there is at least one 2-D data HDU.

    Returns
    -------
    tuple
        (n_ok, n_checked).
    """
    n_ok = 0
    paths = [os.fspath(p) for p in paths if p]
    n_checked = len(paths)
    for path in paths:
        if not os.path.isfile(path):
            logger.warning(
                "[primitive cleanup] %s: validation skipped (missing file) %s",
                step_name,
                path,
            )
            continue
        try:
            from astropy.io import fits

            with fits.open(path, mode="readonly", memmap=False) as hdul:
                if len(hdul) < 1:
                    logger.warning(
                        "[primitive cleanup] %s: invalid FITS (empty HDUList) %s",
                        step_name,
                        path,
                    )
                    continue
                found_2d = False
                wcs_ok = not require_basic_wcs
                for h in hdul:
                    naxis = int(h.header.get("NAXIS", 0) or 0)
                    if naxis >= 2 and h.data is not None:
                        found_2d = True
                        if require_basic_wcs:
                            wcs_ok = "CRVAL1" in h.header and "CRVAL2" in h.header
                        break
                if not found_2d:
                    logger.warning(
                        "[primitive cleanup] %s: no 2-D data HDU in %s",
                        step_name,
                        path,
                    )
                    continue
                if require_basic_wcs and not wcs_ok:
                    logger.warning(
                        "[primitive cleanup] %s: missing basic WCS (CRVAL*) in %s",
                        step_name,
                        path,
                    )
                    continue
                logger.debug(
                    "[primitive cleanup] %s: validated FITS %s | HDUs=%d",
                    step_name,
                    os.path.basename(path),
                    len(hdul),
                )
                n_ok += 1
        except Exception as exc:
            logger.warning(
                "[primitive cleanup] %s: could not read FITS %s: %s",
                step_name,
                path,
                exc,
            )
    if n_checked == 0:
        logger.info(
            "[primitive cleanup] %s: no FITS paths supplied for validation",
            step_name,
        )
    else:
        logger.info(
            "[primitive cleanup] %s: FITS validation %d/%d OK",
            step_name,
            n_ok,
            n_checked,
        )
    return n_ok, n_checked


def validate_text_outputs(
    paths: Iterable[str],
    logger: logging.Logger,
    *,
    step_name: str,
    min_size: int = 1,
) -> tuple[int, int]:
    """Ensure text files exist and have size >= *min_size* bytes."""
    n_ok = 0
    paths = [os.fspath(p) for p in paths if p]
    n_checked = len(paths)
    for path in paths:
        if not os.path.isfile(path):
            logger.warning(
                "[primitive cleanup] %s: missing text output %s",
                step_name,
                path,
            )
            continue
        sz = os.path.getsize(path)
        if sz < min_size:
            logger.warning(
                "[primitive cleanup] %s: text output too small (%d B) %s",
                step_name,
                sz,
                path,
            )
            continue
        logger.info(
            "[primitive cleanup] %s: validated text output %s (%d B)",
            step_name,
            os.path.basename(path),
            sz,
        )
        n_ok += 1
    if n_checked:
        logger.info(
            "[primitive cleanup] %s: text validation %d/%d OK",
            step_name,
            n_ok,
            n_checked,
        )
    return n_ok, n_checked


def validate_astropy_tables(
    tables: Sequence[Any],
    logger: logging.Logger,
    *,
    step_name: str,
) -> None:
    """Log one summary line for photometry table shapes (may be empty)."""
    try:
        n = len(tables)
    except TypeError:
        logger.info(
            "[primitive cleanup] %s: table validation skipped (not a sequence)",
            step_name,
        )
        return
    if n == 0:
        logger.info(
            "[primitive cleanup] %s: no photometry tables returned",
            step_name,
        )
        return

    nrows: list[int | None] = []
    ncols: list[int | None] = []
    for t in tables:
        try:
            nrows.append(len(t))
        except Exception:
            nrows.append(None)
        try:
            ncols.append(len(t.colnames) if hasattr(t, "colnames") else None)
        except Exception:
            ncols.append(None)

    valid_rows = [r for r in nrows if r is not None]
    valid_cols = [c for c in ncols if c is not None]
    if not valid_rows:
        logger.info(
            "[primitive cleanup] %s: photometry catalog: %d table(s) "
            "(could not read row counts)",
            step_name,
            n,
        )
        return

    rmin, rmax = min(valid_rows), max(valid_rows)
    rmed = int(statistics.median(valid_rows))
    col_set = set(valid_cols)
    if len(col_set) == 1:
        cols_summary = str(next(iter(col_set)))
    elif col_set:
        lo, hi = min(col_set), max(col_set)
        cols_summary = f"{lo}" if lo == hi else f"{lo}–{hi}"
    else:
        cols_summary = "?"

    if rmin == rmax and len(col_set) <= 1:
        logger.info(
            "[primitive cleanup] %s: photometry catalog: %d table(s), "
            "each %d rows × %s cols",
            step_name,
            n,
            rmin,
            cols_summary,
        )
    else:
        logger.info(
            "[primitive cleanup] %s: photometry catalog: %d table(s); "
            "rows min=%d max=%d median=%d; cols %s",
            step_name,
            n,
            rmin,
            rmax,
            rmed,
            cols_summary,
        )

    if logger.isEnabledFor(logging.DEBUG):
        for i, t in enumerate(tables):
            try:
                nrow = len(t)
            except Exception:
                nrow = "?"
            try:
                ncol = len(t.colnames) if hasattr(t, "colnames") else "?"
            except Exception:
                ncol = "?"
            logger.debug(
                "[primitive cleanup] %s: photometry table[%d] rows=%s cols=%s",
                step_name,
                i,
                nrow,
                ncol,
            )


def run_primitive_cleanup(
    logger: logging.Logger,
    step_name: str,
    *,
    work_dir: str | None = None,
    remove_globs: Sequence[str] = (),
    remove_paths: Sequence[str] = (),
    validate_fits_paths: Sequence[str] = (),
    validate_text_paths: Sequence[str] = (),
    require_basic_wcs: bool = False,
    text_min_size: int = 1,
    skip_interstitial_removal: bool = False,
    validation_notes: dict[str, Any] | None = None,
    custom_validators: Sequence[Callable[[], None]] = (),
    validate_tables: Sequence[Any] | None = None,
    pipeline: Any | None = None,
    respect_keep_artifacts: bool = True,
) -> None:
    """
    Logged cleanup: optional interstitial removal, FITS/text/table validation, custom hooks.

    Parameters
    ----------
    skip_interstitial_removal
        If True, only run validation (no file removal).
    respect_keep_artifacts
        If True and *pipeline* has ``keep_drizzle_artifacts``, skip interstitial removal.
    """
    logger.debug("[primitive cleanup] %s: starting post-step cleanup", step_name)

    skip_remove = skip_interstitial_removal
    if (
        respect_keep_artifacts
        and pipeline is not None
        and _safe_keep_artifacts(pipeline)
    ):
        skip_remove = True
        logger.info(
            "[primitive cleanup] %s: skipping interstitial removal (keep_drizzle_artifacts)",
            step_name,
        )

    if remove_globs or remove_paths:
        remove_interstitial_files(
            work_dir,
            tuple(remove_globs),
            tuple(remove_paths),
            logger,
            step_name=step_name,
            skip_removal=skip_remove,
        )

    if validate_fits_paths:
        validate_fits_outputs(
            validate_fits_paths,
            logger,
            step_name=step_name,
            require_basic_wcs=require_basic_wcs,
        )

    if validate_text_paths:
        validate_text_outputs(
            validate_text_paths,
            logger,
            step_name=step_name,
            min_size=text_min_size,
        )

    if validate_tables is not None:
        validate_astropy_tables(validate_tables, logger, step_name=step_name)

    if validation_notes:
        for key, val in validation_notes.items():
            logger.info(
                "[primitive cleanup] %s: %s = %s",
                step_name,
                key,
                val,
            )

    for fn in custom_validators:
        try:
            fn()
        except Exception as exc:
            logger.warning(
                "[primitive cleanup] %s: custom validator failed: %s",
                step_name,
                exc,
            )

    logger.debug("[primitive cleanup] %s: cleanup complete", step_name)
