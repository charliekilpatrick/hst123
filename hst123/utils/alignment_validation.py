"""
Post-TweakReg checks: shift-table statistics and sub-pixel alignment reporting.

DrizzlePac ``drizzle_shifts.txt`` lists pixel-plane offsets (``xoffset``, ``yoffset``)
applied per image relative to the reference. These metrics demonstrate whether the
solution is in the **sub-pixel regime** (typically <|dx|,|dy| < 1 px on the detector).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

_LOG = logging.getLogger(__name__)

# Max |dx| or |dy| (pixels) to call "sub-pixel" for logging / PASS label
DEFAULT_SUBPIXEL_PIXEL_MAX = 1.0
# Warn when any shift magnitude exceeds this (likely mis-match or bad ref)
WARN_SHIFT_PIXEL_MAG = 5.0


def plate_scale_arcsec_per_pixel(
    fits_path: str,
    *,
    ext: int = 0,
) -> float | None:
    """
    Mean plate scale in arcsec/pixel at the reference image WCS tangent plane.

    Parameters
    ----------
    fits_path : str
        Path to a FITS file (reference image).
    ext : int
        HDU index for WCS (default primary).

    Returns
    -------
    float or None
        Mean of ``proj_plane_pixel_scales()`` in arcsec/pixel, or None if WCS fails.
    """
    if not fits_path or not os.path.isfile(fits_path):
        return None
    try:
        with fits.open(fits_path) as hdul:
            w = WCS(hdul[ext].header, hdul)
            scales = w.proj_plane_pixel_scales()
            return float(np.mean(scales.to(u.arcsec).value))
    except Exception as exc:
        _LOG.debug("plate_scale_arcsec_per_pixel: %s: %s", fits_path, exc)
        return None


def log_tweakreg_shift_metrics(
    shifts: Table,
    *,
    ref_path: str,
    log: logging.Logger,
    tolerance_arcsec: float | None = None,
    batch_index: int | None = None,
    summary_prefix: str | None = None,
    subpixel_pixel_max: float = DEFAULT_SUBPIXEL_PIXEL_MAX,
) -> dict[str, Any]:
    """
    Log statistics on TweakReg shift columns and sub-pixel / tolerance checks.

    Parameters
    ----------
    shifts : `astropy.table.Table`
        Must contain ``xoffset`` and ``yoffset`` (pixels).
    ref_path : str
        Reference image path (for plate scale); may be ``dummy.fits``.
    log : logging.Logger
        Logger (e.g. pipeline ``hst123.astrometry``).
    tolerance_arcsec : float, optional
        TweakReg ``tolerance`` (arcsec) used for source matching — reported alongside
        max offset in arcsec for context.
    batch_index : int, optional
        If set, log lines are prefixed with ``batch N:`` (ignored if *summary_prefix* is set).
    summary_prefix : str, optional
        If set (e.g. ``"TweakReg aggregate"``), used as the log line prefix instead of batch index.
    subpixel_pixel_max : float
        Threshold (pixels) for labeling alignment "sub-pixel" in the log.

    Returns
    -------
    dict
        Summary keys: ``n``, ``rms_dx``, ``rms_dy``, ``max_abs_dx``, ``max_abs_dy``,
        ``max_hypot_px``, ``max_hypot_arcsec``, ``subpixel_pass`` (bool).
    """
    empty: dict[str, Any] = {
        "n": 0,
        "subpixel_pass": False,
        "max_hypot_px": float("nan"),
    }
    if shifts is None or len(shifts) == 0:
        log.warning("TweakReg validation: empty shift table")
        return empty

    try:
        dx = np.asarray(shifts["xoffset"], dtype=float)
        dy = np.asarray(shifts["yoffset"], dtype=float)
    except KeyError as exc:
        log.warning("TweakReg validation: missing columns: %s", exc)
        return empty

    mask = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(mask):
        log.warning("TweakReg validation: no finite xoffset/yoffset rows")
        return empty

    dx = dx[mask]
    dy = dy[mask]
    n = int(dx.size)
    rms_dx = float(np.sqrt(np.mean(dx**2)))
    rms_dy = float(np.sqrt(np.mean(dy**2)))
    max_adx = float(np.max(np.abs(dx)))
    max_ady = float(np.max(np.abs(dy)))
    hyp = np.hypot(dx, dy)
    max_hyp = float(np.max(hyp))
    mean_hyp = float(np.mean(hyp))

    px_scale = plate_scale_arcsec_per_pixel(ref_path) if ref_path else None
    max_arcsec = max_hyp * px_scale if px_scale is not None else float("nan")
    rms_arcsec = (
        float(np.sqrt(np.mean((hyp * px_scale) ** 2))) if px_scale is not None else float("nan")
    )

    subpixel_pass = max_adx < subpixel_pixel_max and max_ady < subpixel_pixel_max

    bp = ""
    if summary_prefix:
        bp = f"{summary_prefix.strip()}: "
    elif batch_index is not None:
        bp = f"batch {batch_index}: "

    log.info(
        "%sTweakReg shift stats (%d image(s)): RMS dx=%.4f dy=%.4f px | "
        "max |dx|=%.4f |dy|=%.4f px | mean|dr|=%.4f max|dr|=%.4f px",
        bp,
        n,
        rms_dx,
        rms_dy,
        max_adx,
        max_ady,
        mean_hyp,
        max_hyp,
    )

    if px_scale is not None:
        log.info(
            "%sPlate scale (~ref) ≈ %.4f arcsec/pix → max |Δ| ≈ %.4f arcsec "
            "(RMS offset magnitude ≈ %.4f arcsec)",
            bp,
            px_scale,
            max_hyp * px_scale,
            rms_arcsec,
        )
        if tolerance_arcsec is not None:
            log.info(
                "%sTweakReg matching tolerance was %.4f arcsec (sources must agree within "
                "this radius; solution offsets above are applied shifts, not residuals).",
                bp,
                tolerance_arcsec,
            )

    if subpixel_pass:
        log.info(
            "%sAlignment quality: PASS — all shifts sub-pixel (|dx|,|dy| < %.2f px).",
            bp,
            subpixel_pixel_max,
        )
    else:
        log.warning(
            "%sAlignment quality: CHECK — at least one axis shift ≥ %.2f px "
            "(may still be astrometrically fine for wide dithers).",
            bp,
            subpixel_pixel_max,
        )

    if max_hyp > WARN_SHIFT_PIXEL_MAG:
        log.warning(
            "%sLarge shift magnitude max|Δ|=%.2f px — inspect ref/image pairing and input WCS.",
            bp,
            max_hyp,
        )

    return {
        "n": n,
        "rms_dx": rms_dx,
        "rms_dy": rms_dy,
        "max_abs_dx": max_adx,
        "max_abs_dy": max_ady,
        "mean_hypot_px": mean_hyp,
        "max_hypot_px": max_hyp,
        "max_hypot_arcsec": max_arcsec,
        "plate_scale_arcsec_per_pix": px_scale,
        "subpixel_pass": subpixel_pass,
    }
