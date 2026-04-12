"""
Relocate or remove drizzlepac / TweakReg scratch files in the work directory.

Default pipeline behavior keeps the work tree tidy: logs go under ``logs/`` and
common intermediate FITS/text files from AstroDrizzle are deleted after a
successful step. Use ``--keep-drizzle-artifacts`` to skip this.

Intermediate names such as ``single_sci.fits``, ``single_wht.fits``,
``staticMask.fits``, and ``drz_med.fits`` are produced by DrizzlePac during
``run_astrodrizzle``. WFPC2 pipeline scratch files matching
``*hst123drz*_c0m.fits`` / ``*hst123drz*_c1m.fits`` come from
``wfpc2_astrodrizzle_scratch_paths`` in ``hst123.utils.astrodrizzle_helpers``.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
from collections.abc import Sequence
from datetime import datetime


def remove_files_matching_globs(
    work_dir: str | os.PathLike[str],
    patterns: Sequence[str],
    *,
    log: logging.Logger | None = None,
    log_each_path: bool = False,
) -> int:
    """
    Remove regular files under ``work_dir`` matching each glob in ``patterns``.

    Patterns are joined with ``work_dir`` (not the process CWD), so cleanup
    finds products under ``--work-dir`` (e.g. ``test_data/*.drc.noise.fits``)
    regardless of where the CLI was launched.

    The same path is only removed once even if multiple patterns match.

    Parameters
    ----------
    work_dir
        Working directory containing pipeline outputs.
    patterns
        Glob patterns relative to ``work_dir`` (e.g. ``*drc.noise.fits``).
    log
        If set and ``log_each_path`` is True, ``log.info`` each removed file.
    log_each_path
        Log every removal (default False for silent bulk deletes).

    Returns
    -------
    int
        Number of files removed.
    """
    wd = os.path.abspath(os.path.expanduser(os.fspath(work_dir)))
    removed = 0
    seen: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(wd, pattern)):
            ap = os.path.abspath(path)
            if ap in seen:
                continue
            seen.add(ap)
            if not os.path.isfile(ap):
                continue
            try:
                os.remove(ap)
                removed += 1
                if log is not None and log_each_path:
                    log.info("Removing file: %s", ap)
            except OSError as exc:
                if log is not None:
                    log.warning("Could not remove %s: %s", ap, exc)
    return removed


# Exact basenames often left in the work directory (cwd during drizzle) or next
# to the drizzle output (e.g. ``work_dir/drizzle/`` when ``--drizzle-all``).
# ``single_sci.fits`` / ``single_wht.fits``: separation-step intermediates from
# DrizzlePac during :func:`drizzlepac.astrodrizzle.AstroDrizzle`.
_INTERSTITIAL_EXACT = frozenset(
    {
        "staticMask.fits",
        "drz_med.fits",
        "drz.mask.fits",
        "drz.weight.fits",
        "catalog.coo",
        "crclean.fits",
        "crmask.fits",
        "dqmask.fits",
        "final_mask.fits",
        "single_mask.fits",
        "single_sci.fits",
        "single_wht.fits",
        "blt.fits",
        "sci1.fits",
        "sci2.fits",
        "shifts_wcs.fits",
        "skymask_cat",
    }
)

# Glob patterns relative to work_dir (headerlets; drizzlepac temp FITS).
# DrizzlePac often writes root-prefixed names (e.g. ``wfpc2..._1_staticMask.fits``),
# not only the legacy bare basenames in ``_INTERSTITIAL_EXACT``.
_INTERSTITIAL_GLOBS = (
    "*_hlet.fits",
    "*drztmp*.fits",
    "*_skymatch_mask_*.fits",
    "*staticMask.fits",
    "*StaticMask.fits",
    "*drz_med.fits",
    "*.drz.mask.fits",
    "*.drz.weight.fits",
    "*crmask.fits",
    "*dqmask.fits",
    "*final_mask.fits",
    "*single_mask.fits",
    "*single_sci.fits",
    "*single_wht.fits",
    "*blt.fits",
    "*crclean.fits",
    # WFPC2 scratch copies beside inputs: ``wfpc2_astrodrizzle_scratch_paths``
    # in ``hst123.utils.astrodrizzle_helpers``.
    "*hst123drz*_c0m.fits",
    "*hst123drz*_c1m.fits",
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


def _remove_interstitial_drizzle_scratch_files(
    directory: str,
    *,
    log: logging.Logger,
) -> list[str]:
    """
    Delete DrizzlePac / pipeline scratch FITS and sidecar text under *directory*.

    *directory* must exist; callers pass ``work_dir`` and optionally
    ``work_dir/drizzle`` (``--drizzle-all`` output root) because AstroDrizzle
    writes intermediates relative to CWD and beside the output path.
    """
    removed: list[str] = []
    for name in _INTERSTITIAL_EXACT:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(name)
            except OSError as e:
                log.debug("Could not remove %s: %s", path, e)

    for pattern in _INTERSTITIAL_GLOBS:
        for path in glob.glob(os.path.join(directory, pattern)):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed.append(os.path.basename(path))
            except OSError as e:
                log.debug("Could not remove %s: %s", path, e)
    return removed


def cleanup_after_astrodrizzle(
    work_dir: str,
    *,
    log: logging.Logger,
    keep_artifacts: bool = False,
    base_work_dir: str | os.PathLike[str] | None = None,
) -> None:
    """
    After a successful AstroDrizzle run: remove well-known scratch products.

    ``astrodrizzle.log`` is normally written under ``.hst123_runfiles/`` and
    ingested into the session log (nothing to archive in the work root).

    Matching uses both fixed basenames (legacy drizzle cwd) and globs because
    DrizzlePac often emits **root-prefixed** mask names (e.g. ``*_crmask.fits``)
    rather than bare ``crmask.fits`` in ``work_dir``.

    When ``--drizzle-all`` is used, drizzle products live under
    ``<base_work_dir>/drizzle/``. *work_dir* is typically ``<base>/workspace`` (the
    AstroDrizzle scratch/output tree); DrizzlePac may also drop ``staticMask.fits``
    and similar files in the **base** work directory, so pass *base_work_dir* to
    scrub both trees.
    """
    if keep_artifacts or not work_dir:
        return
    wd = os.path.abspath(os.path.expanduser(work_dir))
    if not os.path.isdir(wd):
        return

    _archive_file(wd, "astrodrizzle.log", log)

    scan_dirs: list[str] = [wd]
    d_sub = os.path.join(wd, "drizzle")
    if os.path.isdir(d_sub):
        scan_dirs.append(d_sub)

    if base_work_dir:
        bw = os.path.abspath(os.path.expanduser(os.fspath(base_work_dir)))
        if bw != wd:
            scan_dirs.append(bw)
            bd = os.path.join(bw, "drizzle")
            if os.path.isdir(bd):
                scan_dirs.append(bd)

    # De-duplicate (workspace/drizzle should not appear when drizzle/ is only under base)
    seen_dir: set[str] = set()
    uniq_dirs: list[str] = []
    for d in scan_dirs:
        ad = os.path.abspath(d)
        if ad not in seen_dir and os.path.isdir(ad):
            seen_dir.add(ad)
            uniq_dirs.append(ad)

    removed: list[str] = []
    for d in uniq_dirs:
        removed.extend(_remove_interstitial_drizzle_scratch_files(d, log=log))

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
