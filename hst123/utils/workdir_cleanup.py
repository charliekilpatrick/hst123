"""
Relocate or remove drizzlepac / TweakReg scratch files in the work directory.

Default pipeline behavior keeps the work tree tidy: logs go under ``logs/`` and
common intermediate FITS/text files from AstroDrizzle are deleted after a
successful step. Use ``--keep-drizzle-artifacts`` to skip this.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
from datetime import datetime

# Exact basenames often left in the work directory (cwd during drizzle).
_INTERSTITIAL_EXACT = frozenset(
    {
        "staticMask.fits",
        "drz_med.fits",
        "drz.mask.fits",
        "drz.weight.fits",
        "catalog.coo",
        "crclean.fits",
        "crmask.fits",
        "final_mask.fits",
        "single_mask.fits",
        "blt.fits",
        "sci1.fits",
        "sci2.fits",
        "shifts_wcs.fits",
        "skymask_cat",
    }
)

# Glob patterns relative to work_dir (headerlets; drizzlepac temp FITS).
_INTERSTITIAL_GLOBS = (
    "*_hlet.fits",
    "*drztmp*.fits",
    "*_skymatch_mask_*.fits",
    "*staticMask.fits",
    "*StaticMask.fits",
    "*drz_med.fits",
    "*.drz.mask.fits",
    "*.drz.weight.fits",
)


def _logs_dir(work_dir: str) -> str:
    d = os.path.join(work_dir, "logs")
    os.makedirs(d, exist_ok=True)
    return d


def _archive_file(work_dir: str, basename: str, log: logging.Logger) -> None:
    """Move ``work_dir/basename`` into ``work_dir/logs/`` with a timestamp suffix."""
    src = os.path.join(work_dir, basename)
    if not os.path.isfile(src):
        return
    logs = _logs_dir(work_dir)
    root, ext = os.path.splitext(basename)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{root}_{stamp}{ext}" if ext else f"{root}_{stamp}"
    dst = os.path.join(logs, name)
    shutil.move(src, dst)
    log.info("Archived pipeline sidecar file: %s -> %s", src, dst)


def remove_superseded_instrument_mask_reference_drizzle(
    logical_drc_path: str | os.PathLike[str],
    *,
    log: logging.Logger,
    keep_artifacts: bool = False,
) -> None:
    """
    Remove the instrument-only reference drizzle produced when ``n_input < 3``.

    :meth:`hst123.hst123.pick_reference` first drizzles to ``{inst}.ref.drc.fits``
    (e.g. ``acs_wfc_full.ref.drc.fits``) to build static masks, then drizzles again
    to the filter-named product (e.g. ``acs.f814w.ref_0001.drc.fits``). The first
    file is interstitial and should be deleted after the final reference succeeds.
    """
    if keep_artifacts or not logical_drc_path:
        return
    drc = os.path.abspath(os.path.expanduser(os.fspath(logical_drc_path)))
    if not drc.lower().endswith(".drc.fits"):
        log.debug("Skip interstitial ref removal (not .drc.fits): %s", drc)
        return

    from hst123.utils.astrodrizzle_helpers import (
        drizzle_canonical_weight_mask_paths,
        drizzle_sidecar_paths,
    )
    from hst123.utils.astrodrizzle_paths import logical_driz_to_internal_astrodrizzle

    internal = os.path.abspath(logical_driz_to_internal_astrodrizzle(drc))
    candidates = [drc, internal]
    sci, wht, ctx = drizzle_sidecar_paths(internal)
    candidates.extend([sci, wht, ctx])
    candidates.extend(drizzle_canonical_weight_mask_paths(internal))
    # drizzlepac median product: ``{root}_med.fits`` with root ``*.drz``
    if internal.lower().endswith(".fits"):
        candidates.append(internal[:-5] + "_med.fits")

    removed: list[str] = []
    seen: set[str] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if not os.path.isfile(p):
            continue
        try:
            os.remove(p)
            removed.append(os.path.basename(p))
        except OSError as exc:
            log.debug("Could not remove interstitial ref drizzle %s: %s", p, exc)

    if removed:
        log.info(
            "Removed superseded instrument-mask reference drizzle (%d file(s)): %s",
            len(removed),
            ", ".join(sorted(removed)),
        )


def cleanup_after_astrodrizzle(
    work_dir: str,
    *,
    log: logging.Logger,
    keep_artifacts: bool = False,
) -> None:
    """
    After a successful AstroDrizzle run: remove well-known scratch products.

    ``astrodrizzle.log`` is normally written under ``.hst123_runfiles/`` and
    ingested into the session log (nothing to archive in the work root).
    """
    if keep_artifacts or not work_dir:
        return
    wd = os.path.abspath(os.path.expanduser(work_dir))
    if not os.path.isdir(wd):
        return

    _archive_file(wd, "astrodrizzle.log", log)

    removed = []
    for name in _INTERSTITIAL_EXACT:
        path = os.path.join(wd, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(name)
            except OSError as e:
                log.debug("Could not remove %s: %s", path, e)

    for pattern in _INTERSTITIAL_GLOBS:
        for path in glob.glob(os.path.join(wd, pattern)):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed.append(os.path.basename(path))
            except OSError as e:
                log.debug("Could not remove %s: %s", path, e)

    if removed:
        log.info(
            "Removed %d drizzle/tweakreg scratch file(s): %s",
            len(removed),
            ", ".join(sorted(set(removed))[:20])
            + (" …" if len(set(removed)) > 20 else ""),
        )


def cleanup_after_tweakreg(
    work_dir: str,
    *,
    log: logging.Logger,
    keep_artifacts: bool = False,
) -> None:
    """
    After TweakReg: replay shift / headerlet text into the session log, then
    remove those files (no separate log copies under ``logs/`` unless
    *keep_artifacts*). Also remove shifts WCS helper and extracted headerlet FITS.
    """
    if keep_artifacts or not work_dir:
        return
    wd = os.path.abspath(os.path.expanduser(work_dir))
    if not os.path.isdir(wd):
        return

    from hst123.utils.logging import get_logger, ingest_text_file_to_logger

    tr_detail = get_logger("hst123.tweakreg")

    def _ingest_and_remove(path: str, tag: str) -> None:
        if not os.path.isfile(path):
            return
        ingest_text_file_to_logger(
            path,
            tr_detail,
            log_tag=tag,
            replay_full=True,
            begin_end_markers=False,
            compact_ws=True,
            delete_after=True,
        )

    _ingest_and_remove(os.path.join(wd, "drizzle_shifts.txt"), "tweakreg shifts")
    for path in sorted(glob.glob(os.path.join(wd, "drizzle_shifts_*.txt"))):
        _ingest_and_remove(path, "tweakreg shifts")
    _ingest_and_remove(os.path.join(wd, "headerlet.log"), "headerlet")

    for pattern in _INTERSTITIAL_GLOBS:
        for path in glob.glob(os.path.join(wd, pattern)):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                log.debug("Removed headerlet scratch: %s", path)
            except OSError as e:
                log.debug("Could not remove %s: %s", path, e)

    sw = os.path.join(wd, "shifts_wcs.fits")
    if os.path.isfile(sw):
        try:
            os.remove(sw)
            log.debug("Removed %s", sw)
        except OSError as e:
            log.debug("Could not remove %s: %s", sw, e)
