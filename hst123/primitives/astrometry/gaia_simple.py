"""
Simple Gaia alignment (translation-only) without JHAT.

This module implements a lightweight, dependency-minimal Gaia anchoring step
intended to replace the default "JHAT Gaia anchor" in drizzle-first mode.

Algorithm (two-pass):
- Prefetch Gaia DR3 cone catalog (reuse existing hst123 cache naming).
- Propagate catalog positions to the HST observation epoch with proper motion
  and (when available) parallax.
- Prefer **strict** Gaia quality cuts and a small set of high-S/N HST
  calibrators (Gaussian centroids). If fewer than three good matches remain,
  automatically **widen** quality cuts / apertures as a fallback for sparse
  fields.
- Compute (dx, dy) = measured - predicted offsets in pixels.
- Sigma-clip outliers on |offset| and take an S/N²-weighted mean offset.
- Apply offset as a CRPIX shift (translation-only) to the image WCS, then repeat
  with a smaller radius (fine pass).
- Compute absolute dispersion metrics (RMS*ABS) from the final matches.

HST centroids default to a 2D Gaussian fit (``centroid_2dg``) with a tight fine
aperture, isolation and S/N cuts, and S/N²-weighted mean offsets; center-of-mass
remains available as a fallback / coarse-pass method.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from hst123.utils.gaia_prefetch import (
    gaia_prefetch_cache_path,
    icrs_field_center_from_fits,
    prefetch_gaia_catalog,
)

log = logging.getLogger("pipeline")

# Gaia DR3 reference epoch (barycentric).
GAIA_DR3_EPOCH = Time(2016.0, format="jyear", scale="tcb")

# Prefer a small set of clean calibrators for the fine-pass fit. Widen only
# when matching fails to produce at least GAIA_MIN_MATCHES_FOR_FIT stars.
GAIA_PREFERRED_N_CALIBRATORS = 5
GAIA_MIN_MATCHES_FOR_FIT = 3

# Preferred (default) quality cuts: reject saturated / poorly measured Gaia
# solutions while still leaving enough isolated HST calibrators on typical fields.
GAIA_STRICT_RUWE_MAX = 1.4
GAIA_STRICT_AEN_MAX_MAS = 1.0
GAIA_STRICT_GMAG_MIN = 14.5
GAIA_STRICT_GMAG_MAX = 20.5
GAIA_STRICT_ISOLATION_ARCSEC = 0.5
GAIA_STRICT_FINE_RADIUS_ARCSEC = 0.2
GAIA_STRICT_MIN_CENTROID_SNR = 4.0

# Relaxed fallback for sparse / hard fields (widen the net once).
# Keep the fine aperture nearly as tight — large apertures dominate scatter.
GAIA_RELAXED_RUWE_MAX = 1.6
GAIA_RELAXED_AEN_MAX_MAS = 2.0
GAIA_RELAXED_GMAG_MIN = 13.5
GAIA_RELAXED_GMAG_MAX = 21.0
GAIA_RELAXED_ISOLATION_ARCSEC = 0.35
GAIA_RELAXED_FINE_RADIUS_ARCSEC = 0.3
GAIA_RELAXED_MIN_CENTROID_SNR = 2.0

# Back-compat aliases used by older tests / callers.
GAIA_QUALITY_CUT_MIN_STARS = 6  # >5 to apply catalog-level cuts
GAIA_RUWE_MAX = GAIA_STRICT_RUWE_MAX
GAIA_AEN_MAX_MAS = GAIA_STRICT_AEN_MAX_MAS
GAIA_GMAG_MIN = GAIA_STRICT_GMAG_MIN
GAIA_GMAG_MAX = GAIA_STRICT_GMAG_MAX

# Default fine search radius for Gaussian centroids (arcsec). COM historically used 0.5".
GAIA_FINE_RADIUS_GAUSS_ARCSEC = GAIA_STRICT_FINE_RADIUS_ARCSEC
GAIA_FINE_RADIUS_COM_ARCSEC = 0.5
# COM historically had no S/N gate; Gaussian centroids benefit from a mild cut.
GAIA_MIN_CENTROID_SNR = 0.0
GAIA_MIN_CENTROID_SNR_GAUSSIAN = GAIA_STRICT_MIN_CENTROID_SNR
GAIA_ISOLATION_ARCSEC = GAIA_STRICT_ISOLATION_ARCSEC


@dataclass(frozen=True)
class GaiaSimpleMatch:
    ra: float
    dec: float
    x_pred: float
    y_pred: float
    x_meas: float
    y_meas: float
    dx: float
    dy: float
    flux: float
    bkg: float
    n_good: int
    snr: float = 0.0


def _find_sci_hdu_index(hdul: fits.HDUList) -> int:
    for i, h in enumerate(hdul):
        name = str(h.header.get("EXTNAME", "") or "").strip().upper()
        if name == "SCI" and getattr(h, "data", None) is not None and int(h.header.get("NAXIS", 0) or 0) >= 2:
            return i
    for i, h in enumerate(hdul):
        if getattr(h, "data", None) is None:
            continue
        if int(h.header.get("NAXIS", 0) or 0) < 2:
            continue
        if "CRVAL1" in h.header and "CRPIX1" in h.header:
            return i
    return 0


def _dq_or_mask_array(hdul: fits.HDUList, sci_shape: tuple[int, int]) -> np.ndarray:
    ny, nx = int(sci_shape[0]), int(sci_shape[1])
    for name in ("DQ", "WHT", "CTX"):
        for h in hdul:
            ext = str(h.header.get("EXTNAME", "") or "").strip().upper()
            if ext != name:
                continue
            if getattr(h, "data", None) is None:
                continue
            arr = np.asarray(h.data)
            if arr.shape[-2:] != (ny, nx):
                continue
            if name == "DQ":
                return np.asarray(arr != 0)
            if name == "WHT":
                return np.asarray(~np.isfinite(arr) | (arr <= 0))
            if name == "CTX":
                return np.asarray(arr == 0)
    return np.zeros((ny, nx), dtype=bool)


def _pixel_scale_arcsec(w: WCS) -> float:
    sc = proj_plane_pixel_scales(w)  # deg/pix
    v = float(np.nanmedian(sc)) * 3600.0
    if not np.isfinite(v) or v <= 0:
        raise RuntimeError("gaia_simple: could not derive pixel scale from WCS.")
    return v


def _circle_mask(ny: int, nx: int, cx: float, cy: float, r: float) -> np.ndarray:
    y, x = np.mgrid[0:ny, 0:nx]
    return (x - cx) ** 2 + (y - cy) ** 2 <= float(r) ** 2


def _header_float(header: fits.Header, key: str) -> float | None:
    if key not in header:
        return None
    try:
        v = float(header[key])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def hst_obstime_from_hdul(hdul: fits.HDUList) -> Time | None:
    """
    Best-effort HST observation epoch from primary / SCI headers.

    Preference: ``MJD-AVG``, mean of ``EXPSTART``/``EXPEND``, ``MJD-OBS``,
    then ``DATE-OBS``.
    """
    headers = [hdul[0].header]
    for h in hdul:
        if str(getattr(h, "name", "") or "").strip().upper() == "SCI":
            headers.append(h.header)

    for hdr in headers:
        mjd_avg = _header_float(hdr, "MJD-AVG")
        if mjd_avg is not None:
            return Time(mjd_avg, format="mjd", scale="utc")
        exp0 = _header_float(hdr, "EXPSTART")
        exp1 = _header_float(hdr, "EXPEND")
        if exp0 is not None and exp1 is not None:
            return Time(0.5 * (exp0 + exp1), format="mjd", scale="utc")
        mjd_obs = _header_float(hdr, "MJD-OBS")
        if mjd_obs is not None:
            return Time(mjd_obs, format="mjd", scale="utc")
        if "DATE-OBS" in hdr:
            try:
                return Time(str(hdr["DATE-OBS"]), format="isot", scale="utc")
            except Exception:
                pass
    return None


def apply_gaia_quality_cuts(
    gaia: Table,
    *,
    min_stars: int = GAIA_QUALITY_CUT_MIN_STARS,
    ruwe_max: float = GAIA_STRICT_RUWE_MAX,
    aen_max_mas: float = GAIA_STRICT_AEN_MAX_MAS,
    gmag_min: float = GAIA_STRICT_GMAG_MIN,
    gmag_max: float = GAIA_STRICT_GMAG_MAX,
    min_keep: int | None = None,
    force: bool = False,
) -> tuple[Table, dict[str, int | bool | float]]:
    """
    Reject poor Gaia astrometric solutions.

    Default thresholds are the **strict** set (prefer fewer, cleaner stars).
    Cuts are applied when ``len(gaia) >= min_stars`` (or *force* is True) and
    enough stars remain after filtering (``>= min_keep``, default *min_stars*).

    - ``ruwe < ruwe_max`` (NaN RUWE allowed)
    - ``astrometric_excess_noise < aen_max_mas`` (NaN allowed)
    - ``gmag_min < mag < gmag_max``
    """
    n0 = int(len(gaia))
    keep_floor = int(min_stars if min_keep is None else min_keep)
    meta: dict[str, int | bool | float] = {
        "applied": False,
        "n_in": n0,
        "n_out": n0,
        "ruwe_max": float(ruwe_max),
        "aen_max_mas": float(aen_max_mas),
        "gmag_min": float(gmag_min),
        "gmag_max": float(gmag_max),
    }
    if (not force) and n0 < int(min_stars):
        return gaia, meta

    keep = np.ones(n0, dtype=bool)
    cols = {c.lower(): c for c in gaia.colnames}

    if "ruwe" in cols:
        ruwe = np.asarray(gaia[cols["ruwe"]], dtype=float)
        # Missing RUWE is common in some Vizier extracts — do not reject NaNs.
        keep &= (~np.isfinite(ruwe)) | (ruwe < float(ruwe_max))
    if "astrometric_excess_noise" in cols:
        aen = np.asarray(gaia[cols["astrometric_excess_noise"]], dtype=float)
        keep &= (~np.isfinite(aen)) | (aen < float(aen_max_mas))
    mag_c = cols.get("mag")
    if mag_c is not None:
        mag = np.asarray(gaia[mag_c], dtype=float)
        keep &= np.isfinite(mag) & (mag > float(gmag_min)) & (mag < float(gmag_max))

    n_keep = int(np.count_nonzero(keep))
    meta["n_out"] = n_keep
    if n_keep < keep_floor:
        # Too few stars after cuts — keep the unfiltered catalog.
        return gaia, meta

    meta["applied"] = True
    return gaia[keep], meta


def select_best_calibrators(
    matches: list[GaiaSimpleMatch],
    *,
    max_n: int = GAIA_PREFERRED_N_CALIBRATORS,
) -> list[GaiaSimpleMatch]:
    """
    Keep the highest-S/N matches (prefer fewer clean calibrators).

    Falls back to flux ranking when S/N is unavailable. Returns *matches*
    unchanged when ``len(matches) <= max_n`` or ``max_n <= 0``.
    """
    if max_n <= 0 or len(matches) <= int(max_n):
        return list(matches)
    ranked = sorted(
        matches,
        key=lambda m: (float(m.snr) if np.isfinite(m.snr) else -1.0, float(m.flux)),
        reverse=True,
    )
    return ranked[: int(max_n)]


def propagate_gaia_to_obstime(gaia: Table, obstime: Time | None) -> tuple[Table, dict[str, int | bool]]:
    """
    Propagate Gaia DR3 sky positions to *obstime* with PM (± parallax).

    Updates ``ra`` / ``dec`` in a copy. Stars without usable PM are left at the
    catalog epoch. Parallax is used only when ``parallax > 0``.
    """
    out = gaia.copy()
    meta: dict[str, int | bool] = {
        "applied": False,
        "n_pm": 0,
        "n_parallax": 0,
    }
    if obstime is None:
        return out, meta
    if "pmra" not in out.colnames or "pmdec" not in out.colnames:
        return out, meta

    ra = np.asarray(out["ra"], dtype=float)
    dec = np.asarray(out["dec"], dtype=float)
    pmra = np.asarray(out["pmra"], dtype=float)
    pmdec = np.asarray(out["pmdec"], dtype=float)
    plx = (
        np.asarray(out["parallax"], dtype=float)
        if "parallax" in out.colnames
        else np.full(len(out), np.nan, dtype=float)
    )

    n_pm = 0
    n_plx = 0
    ra_new = ra.copy()
    dec_new = dec.copy()
    for i in range(len(out)):
        if not (np.isfinite(ra[i]) and np.isfinite(dec[i])):
            continue
        if not (np.isfinite(pmra[i]) and np.isfinite(pmdec[i])):
            continue
        try:
            kwargs: dict[str, object] = dict(
                ra=ra[i] * u.deg,
                dec=dec[i] * u.deg,
                pm_ra_cosdec=pmra[i] * u.mas / u.yr,
                pm_dec=pmdec[i] * u.mas / u.yr,
                obstime=GAIA_DR3_EPOCH,
                frame="icrs",
            )
            if np.isfinite(plx[i]) and float(plx[i]) > 0.0:
                kwargs["distance"] = Distance(parallax=float(plx[i]) * u.mas)
                use_plx = True
            else:
                use_plx = False
            c0 = SkyCoord(**kwargs)
            c1 = c0.apply_space_motion(new_obstime=obstime)
            ra_new[i] = float(c1.ra.deg)
            dec_new[i] = float(c1.dec.deg)
            n_pm += 1
            if use_plx:
                n_plx += 1
        except Exception:
            continue

    out["ra"] = ra_new
    out["dec"] = dec_new
    meta["applied"] = n_pm > 0
    meta["n_pm"] = n_pm
    meta["n_parallax"] = n_plx
    return out, meta


def _centroid_in_radius(
    data: np.ndarray,
    bad: np.ndarray,
    *,
    x0: float,
    y0: float,
    r_pix: float,
    method: str = "gaussian",
    min_snr: float = GAIA_MIN_CENTROID_SNR,
) -> tuple[float, float, float, float, int, float]:
    """
    Centroid and flux around (x0, y0) within r_pix.

    Parameters
    ----------
    method
        ``\"gaussian\"`` uses ``photutils.centroids.centroid_2dg`` (fallback to
        center-of-mass on failure). ``\"com\"`` uses ``centroid_com`` only.
    min_snr
        Reject cutouts whose aperture S/N is below this threshold.

    Returns
    -------
    x_centroid, y_centroid, flux, bkg, n_good, snr
        Positions in full-image pixels.
    """
    try:
        from photutils.centroids import centroid_2dg, centroid_com  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("gaia_simple requires photutils (pip install photutils).") from exc

    ny, nx = data.shape
    r = float(r_pix)
    if r <= 0:
        raise ValueError("r_pix must be positive")
    method_l = str(method or "gaussian").strip().lower()
    if method_l not in ("gaussian", "com", "centroid_com"):
        raise ValueError(f"unknown centroid method {method!r}")

    x0f = float(x0)
    y0f = float(y0)
    x_min = max(0, int(math.floor(x0f - r - 1)))
    x_max = min(nx - 1, int(math.ceil(x0f + r + 1)))
    y_min = max(0, int(math.floor(y0f - r - 1)))
    y_max = min(ny - 1, int(math.ceil(y0f + r + 1)))

    sub = np.asarray(data[y_min : y_max + 1, x_min : x_max + 1], dtype=float)
    msub = np.asarray(bad[y_min : y_max + 1, x_min : x_max + 1], dtype=bool)
    if sub.size < 9:
        raise RuntimeError("cutout too small")

    cx = x0f - float(x_min)
    cy = y0f - float(y_min)
    circ = _circle_mask(sub.shape[0], sub.shape[1], cx, cy, r)
    good = circ & (~msub) & np.isfinite(sub)
    n_good = int(np.count_nonzero(good))
    if n_good < 5:
        raise RuntimeError("too few unmasked pixels in search radius")

    ann = _circle_mask(sub.shape[0], sub.shape[1], cx, cy, 2.0 * r) & (~circ)
    ann_good = ann & (~msub) & np.isfinite(sub)
    if int(np.count_nonzero(ann_good)) >= 10:
        bkg = float(np.nanmedian(sub[ann_good]))
        bkg_rms = float(np.nanstd(sub[ann_good]))
    else:
        bkg = float(np.nanmedian(sub[good]))
        bkg_rms = float(np.nanstd(sub[good]))
    if not np.isfinite(bkg_rms) or bkg_rms <= 0:
        bkg_rms = 1.0

    sub2 = sub - bkg
    sub2[~good] = 0.0
    sub_pos = np.where(sub2 > 0, sub2, 0.0)
    if not np.any(sub_pos > 0):
        raise RuntimeError("no positive flux after background subtraction")

    flux = float(np.nansum(sub2[good]))
    snr = float(flux / (bkg_rms * math.sqrt(max(n_good, 1))))
    if float(min_snr) > 0 and (not np.isfinite(snr) or snr < float(min_snr)):
        raise RuntimeError(f"centroid S/N too low ({snr:.2f})")

    # Soft saturation / cosmic-ray rejection: extreme spikes only.
    peak = float(np.nanmax(sub_pos))
    if np.isfinite(peak) and peak > 300.0 * bkg_rms and peak > 50.0 * abs(flux) / max(n_good, 1):
        raise RuntimeError("centroid peak looks like CR/saturation spike")

    cx2 = cy2 = float("nan")
    if method_l == "gaussian":
        try:
            # Mask non-positive / bad pixels for the Gaussian fitter.
            fit_img = np.array(sub_pos, dtype=float, copy=True)
            fit_img[~good] = 0.0
            cx2, cy2 = centroid_2dg(fit_img)
        except Exception:
            cx2, cy2 = float("nan"), float("nan")
    if not (np.isfinite(cx2) and np.isfinite(cy2)):
        cx2, cy2 = centroid_com(sub_pos)
    if not (np.isfinite(cx2) and np.isfinite(cy2)):
        raise RuntimeError("centroid failed")

    # Reject large pulls from the Gaia prediction (blend / wrong peak).
    pull_lim = 0.85 * r if method_l == "gaussian" else 1.05 * r
    if math.hypot(cx2 - cx, cy2 - cy) > pull_lim:
        raise RuntimeError("centroid pulled too far from prediction")

    x_full = float(x_min) + float(cx2)
    y_full = float(y_min) + float(cy2)
    return x_full, y_full, flux, bkg, n_good, snr


def _gaia_sky_isolation_mask(
    gaia: Table,
    *,
    isolation_arcsec: float,
) -> np.ndarray:
    """True for rows whose nearest other Gaia neighbor is beyond *isolation_arcsec*."""
    n = len(gaia)
    if n == 0 or float(isolation_arcsec) <= 0:
        return np.ones(n, dtype=bool)
    ra = np.asarray(gaia["ra"], dtype=float)
    dec = np.asarray(gaia["dec"], dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    keep = np.ones(n, dtype=bool)
    if int(np.count_nonzero(ok)) < 2:
        return keep
    coords = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg, frame="icrs")
    # For each star, separation to nearest other.
    idx_all = np.flatnonzero(ok)
    for j, i in enumerate(idx_all):
        seps = coords[j].separation(coords).to(u.arcsec).value
        seps[j] = np.inf
        if float(np.min(seps)) < float(isolation_arcsec):
            keep[i] = False
    return keep


def load_gaia_prefetch_table(*, image_path: str, outdir: str, radius: u.Quantity) -> Table:
    """Download/cache Gaia and return the table (columns ra, dec, mag, dmag)."""
    cen = icrs_field_center_from_fits(image_path)
    cache = gaia_prefetch_cache_path(outdir, cen, radius)
    if not (os.path.isfile(cache) and os.path.getsize(cache) > 80):
        prefetch_gaia_catalog(center=cen, radius=radius, out_path=cache)
    tab = Table.read(cache, format="ascii")
    cols = {c.lower(): c for c in tab.colnames}
    if "ra" not in cols or "dec" not in cols:
        raise RuntimeError(f"Gaia prefetch table missing ra/dec columns: {tab.colnames}")
    return tab


def measure_gaia_centroid_offsets(
    w: WCS,
    data: np.ndarray,
    bad: np.ndarray,
    gaia: Table,
    *,
    search_radius_arcsec: float,
    max_sources: int | None = None,
    centroid_method: str = "gaussian",
    min_snr: float = GAIA_MIN_CENTROID_SNR,
    isolation_arcsec: float = 0.0,
) -> list[GaiaSimpleMatch]:
    scale_as = _pixel_scale_arcsec(w)
    r_pix = float(search_radius_arcsec) / float(scale_as)
    if not np.isfinite(r_pix) or r_pix <= 0:
        raise RuntimeError("invalid radius->pixel conversion")

    rows = gaia
    if float(isolation_arcsec) > 0:
        iso = _gaia_sky_isolation_mask(rows, isolation_arcsec=float(isolation_arcsec))
        rows = rows[iso]
    if "mag" in rows.colnames:
        order = np.argsort(np.asarray(rows["mag"], dtype=float))
        rows = rows[order]
    if max_sources is not None and int(max_sources) > 0:
        rows = rows[: int(max_sources)]

    ny, nx = data.shape
    out: list[GaiaSimpleMatch] = []
    for r in rows:
        ra = float(r["ra"])
        dec = float(r["dec"])
        try:
            x_pred, y_pred = w.world_to_pixel_values(ra, dec)
        except Exception:
            continue
        if not (np.isfinite(x_pred) and np.isfinite(y_pred)):
            continue
        if (x_pred < -r_pix) or (x_pred > (nx - 1) + r_pix) or (y_pred < -r_pix) or (y_pred > (ny - 1) + r_pix):
            continue
        try:
            x_meas, y_meas, flux, bkg, n_good, snr = _centroid_in_radius(
                data,
                bad,
                x0=float(x_pred),
                y0=float(y_pred),
                r_pix=r_pix,
                method=centroid_method,
                min_snr=float(min_snr),
            )
        except Exception:
            continue
        if not (np.isfinite(x_meas) and np.isfinite(y_meas) and np.isfinite(flux)):
            continue
        out.append(
            GaiaSimpleMatch(
                ra=ra,
                dec=dec,
                x_pred=float(x_pred),
                y_pred=float(y_pred),
                x_meas=float(x_meas),
                y_meas=float(y_meas),
                dx=float(x_meas - x_pred),
                dy=float(y_meas - y_pred),
                flux=float(flux),
                bkg=float(bkg),
                n_good=int(n_good),
                snr=float(snr),
            )
        )
    return out


def clip_and_weighted_mean_offset(
    matches: list[GaiaSimpleMatch],
    *,
    sigma: float = 3.0,
    weight_by_snr: bool = True,
) -> tuple[float, float, list[GaiaSimpleMatch]]:
    if not matches:
        return 0.0, 0.0, []
    dx = np.asarray([m.dx for m in matches], dtype=float)
    dy = np.asarray([m.dy for m in matches], dtype=float)
    r = np.hypot(dx, dy)
    rr = sigma_clip(r, sigma=float(sigma), maxiters=5, masked=True)
    keep = ~np.asarray(getattr(rr, "mask", np.zeros_like(r, dtype=bool)))
    kept = [m for (m, ok) in zip(matches, keep) if bool(ok)]
    if not kept:
        return 0.0, 0.0, []
    if weight_by_snr:
        w = np.asarray([max(0.0, float(m.snr)) ** 2 for m in kept], dtype=float)
        if not np.any(w > 0):
            w = np.asarray([max(0.0, float(m.flux)) for m in kept], dtype=float)
    else:
        w = np.asarray([max(0.0, float(m.flux)) for m in kept], dtype=float)
    if not np.any(w > 0):
        w = np.ones(len(kept), dtype=float)
    dxm = float(np.average([m.dx for m in kept], weights=w))
    dym = float(np.average([m.dy for m in kept], weights=w))
    return dxm, dym, kept


def apply_crpix_shift_to_hdul(hdul: fits.HDUList, dx: float, dy: float) -> int:
    """Apply a translation-only correction by shifting CRPIX on WCS-carrying HDUs."""
    n = 0
    for h in hdul:
        if "CRPIX1" in h.header and "CRPIX2" in h.header and ("CRVAL1" in h.header or "CTYPE1" in h.header):
            try:
                h.header["CRPIX1"] = (float(h.header["CRPIX1"]) + float(dx), "gaia_simple dx (px)")
                h.header["CRPIX2"] = (float(h.header["CRPIX2"]) + float(dy), "gaia_simple dy (px)")
                n += 1
            except Exception:
                continue
    return n


def rms_abs_keywords_from_matches(w: WCS, matches: list[GaiaSimpleMatch]) -> dict[str, float | int]:
    if len(matches) < 2:
        return {}
    dx = np.asarray([m.dx for m in matches], dtype=float)
    dy = np.asarray([m.dy for m in matches], dtype=float)
    rmsx = float(np.std(dx, ddof=1))
    rmsy = float(np.std(dy, ddof=1))

    ra_ref = np.asarray([m.ra for m in matches], dtype=float)
    dec_ref = np.asarray([m.dec for m in matches], dtype=float)
    x = np.asarray([m.x_meas for m in matches], dtype=float)
    y = np.asarray([m.y_meas for m in matches], dtype=float)
    ra_img, dec_img = w.pixel_to_world_values(x, y)
    img = SkyCoord(ra_img * u.deg, dec_img * u.deg, frame="icrs")
    ref = SkyCoord(ra_ref * u.deg, dec_ref * u.deg, frame="icrs")
    sep_as = img.separation(ref).to(u.arcsec).value
    rms_sky = float(np.sqrt(np.mean(sep_as**2)))

    dra_deg = (np.asarray(ra_img, dtype=float) - ra_ref + 180.0) % 360.0 - 180.0
    ddec_deg = np.asarray(dec_img, dtype=float) - dec_ref
    dra_as = dra_deg * np.cos(np.radians(dec_ref)) * 3600.0
    ddec_as = ddec_deg * 3600.0
    rms_ra = float(np.std(dra_as, ddof=1))
    rms_dec = float(np.std(ddec_as, ddof=1))

    return {
        "RMSXABS": rmsx,
        "RMSYABS": rmsy,
        "RMSRAABS": rms_ra,
        "RMSDEABS": rms_dec,
        "RMSSKYAB": rms_sky,  # extra (arcsec)
        "NGAIAABS": int(len(matches)),
    }


def align_to_gaia_simple_inplace(
    fits_path: str | os.PathLike[str],
    *,
    outdir: str | os.PathLike[str],
    gaia_prefetch_radius: u.Quantity = 22 * u.arcmin,
    coarse_radius_arcsec: float = 5.0,
    fine_radius_arcsec: float | None = None,
    max_gaia_sources: int = 4000,
    clip_sigma: float = 3.0,
    write_diagnostics: bool = False,
    apply_pm: bool = True,
    apply_quality_cuts: bool = True,
    centroid_method: str = "gaussian",
    min_centroid_snr: float | None = None,
    isolation_arcsec: float | None = None,
    weight_by_snr: bool = True,
    preferred_n_calibrators: int = GAIA_PREFERRED_N_CALIBRATORS,
    min_matches: int = GAIA_MIN_MATCHES_FOR_FIT,
    allow_relaxed_fallback: bool = True,
) -> dict[str, object] | None:
    """
    Align *fits_path* to Gaia by updating its WCS in-place (CRPIX shifts).

    Default strategy (preferred):
    - Epoch-correct Gaia with PM ± parallax.
    - Apply **strict** RUWE / excess-noise / G-mag cuts.
    - Gaussian HST centroids in a tight fine aperture.
    - Keep only the top ``preferred_n_calibrators`` matches by S/N for the fit
      (fewer clean stars beat many contaminated ones).

    Fallback (sparse / hard fields): if fewer than ``min_matches`` survive,
    automatically widen quality cuts and apertures once, then retry matching.

    Returns a stats dict (n_match, rms_*_as/pix, etc.) when possible.
    """
    src = os.path.abspath(os.path.expanduser(os.fspath(fits_path)))
    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    os.makedirs(od, exist_ok=True)

    method_l = str(centroid_method or "gaussian").strip().lower()

    gaia = load_gaia_prefetch_table(image_path=src, outdir=od, radius=gaia_prefetch_radius)

    with fits.open(src, mode="update", memmap=False) as hdul:
        obstime = hst_obstime_from_hdul(hdul)
        if apply_pm:
            gaia_ep, pm_meta = propagate_gaia_to_obstime(gaia, obstime)
        else:
            gaia_ep, pm_meta = gaia, {"applied": False, "n_pm": 0, "n_parallax": 0}

        if pm_meta.get("applied"):
            log.info(
                "Gaia simple: epoch-corrected %d/%d stars to MJD=%.3f "
                "(%d with parallax)",
                int(pm_meta.get("n_pm", 0) or 0),
                len(gaia),
                float(obstime.mjd) if obstime is not None else float("nan"),
                int(pm_meta.get("n_parallax", 0) or 0),
            )
        elif apply_pm and obstime is None:
            log.warning(
                "Gaia simple: no HST epoch in headers; using Gaia DR3 positions "
                "without proper-motion correction"
            )

        sci_i = _find_sci_hdu_index(hdul)
        data = np.asarray(hdul[sci_i].data, dtype=float)
        bad = _dq_or_mask_array(hdul, data.shape[-2:])

        # Tier definitions: strict first, then optional relaxed fallback.
        tiers: list[dict[str, object]] = [
            {
                "name": "strict",
                "ruwe_max": GAIA_STRICT_RUWE_MAX,
                "aen_max_mas": GAIA_STRICT_AEN_MAX_MAS,
                "gmag_min": GAIA_STRICT_GMAG_MIN,
                "gmag_max": GAIA_STRICT_GMAG_MAX,
                "isolation_arcsec": (
                    float(isolation_arcsec)
                    if isolation_arcsec is not None
                    else GAIA_STRICT_ISOLATION_ARCSEC
                ),
                "fine_radius_arcsec": (
                    float(fine_radius_arcsec)
                    if fine_radius_arcsec is not None
                    else (
                        GAIA_STRICT_FINE_RADIUS_ARCSEC
                        if method_l == "gaussian"
                        else GAIA_FINE_RADIUS_COM_ARCSEC
                    )
                ),
                "min_snr": (
                    float(min_centroid_snr)
                    if min_centroid_snr is not None
                    else (
                        GAIA_STRICT_MIN_CENTROID_SNR
                        if method_l == "gaussian"
                        else 0.0
                    )
                ),
                # Catalog-level revert floor stays at QUALITY_CUT_MIN_STARS so
                # we do not silently drop back to the unfiltered Gaia catalog
                # when only a handful of clean stars remain.
                "min_keep": GAIA_QUALITY_CUT_MIN_STARS,
            }
        ]
        if allow_relaxed_fallback:
            tiers.append(
                {
                    "name": "relaxed",
                    "ruwe_max": GAIA_RELAXED_RUWE_MAX,
                    "aen_max_mas": GAIA_RELAXED_AEN_MAX_MAS,
                    "gmag_min": GAIA_RELAXED_GMAG_MIN,
                    "gmag_max": GAIA_RELAXED_GMAG_MAX,
                    "isolation_arcsec": (
                        float(isolation_arcsec)
                        if isolation_arcsec is not None
                        else GAIA_RELAXED_ISOLATION_ARCSEC
                    ),
                    "fine_radius_arcsec": (
                        float(fine_radius_arcsec)
                        if fine_radius_arcsec is not None
                        else GAIA_RELAXED_FINE_RADIUS_ARCSEC
                    ),
                    "min_snr": (
                        float(min_centroid_snr)
                        if min_centroid_snr is not None
                        else (
                            GAIA_RELAXED_MIN_CENTROID_SNR
                            if method_l == "gaussian"
                            else 0.0
                        )
                    ),
                    "min_keep": GAIA_MIN_MATCHES_FOR_FIT,
                }
            )

        dx_c = dy_c = dx_f = dy_f = 0.0
        coarse_kept: list[GaiaSimpleMatch] = []
        fine_kept: list[GaiaSimpleMatch] = []
        final_kept: list[GaiaSimpleMatch] = []
        q_meta: dict[str, object] = {"applied": False, "n_in": len(gaia_ep), "n_out": len(gaia_ep)}
        tier_used = "strict"
        fine_radius_used = float(tiers[0]["fine_radius_arcsec"])  # type: ignore[arg-type]
        # Snapshot WCS before any tier so fallback can restart cleanly.
        wcs_backup_headers = [
            (i, h.header.copy())
            for i, h in enumerate(hdul)
            if "CRPIX1" in h.header and "CRPIX2" in h.header
        ]

        def _restore_wcs() -> None:
            for i, hdr in wcs_backup_headers:
                hdul[i].header = hdr

        for tier in tiers:
            tier_used = str(tier["name"])
            fine_radius_used = float(tier["fine_radius_arcsec"])  # type: ignore[arg-type]
            iso = float(tier["isolation_arcsec"])  # type: ignore[arg-type]
            fine_snr = float(tier["min_snr"])  # type: ignore[arg-type]

            if apply_quality_cuts:
                gaia_use, q_meta = apply_gaia_quality_cuts(
                    gaia_ep,
                    ruwe_max=float(tier["ruwe_max"]),  # type: ignore[arg-type]
                    aen_max_mas=float(tier["aen_max_mas"]),  # type: ignore[arg-type]
                    gmag_min=float(tier["gmag_min"]),  # type: ignore[arg-type]
                    gmag_max=float(tier["gmag_max"]),  # type: ignore[arg-type]
                    min_keep=int(tier["min_keep"]),  # type: ignore[arg-type]
                    force=(tier_used == "relaxed"),
                )
            else:
                gaia_use = gaia_ep
                q_meta = {
                    "applied": False,
                    "n_in": len(gaia_ep),
                    "n_out": len(gaia_ep),
                    "ruwe_max": float(tier["ruwe_max"]),  # type: ignore[arg-type]
                    "aen_max_mas": float(tier["aen_max_mas"]),  # type: ignore[arg-type]
                    "gmag_min": float(tier["gmag_min"]),  # type: ignore[arg-type]
                    "gmag_max": float(tier["gmag_max"]),  # type: ignore[arg-type]
                }

            if q_meta.get("applied"):
                log.info(
                    "Gaia simple [%s]: quality cuts %d → %d stars "
                    "(RUWE<%.2f, AEN<%.2f mas, %.1f<G<%.1f)",
                    tier_used,
                    int(q_meta.get("n_in", 0) or 0),
                    int(q_meta.get("n_out", 0) or 0),
                    float(q_meta.get("ruwe_max", 0) or 0),
                    float(q_meta.get("aen_max_mas", 0) or 0),
                    float(q_meta.get("gmag_min", 0) or 0),
                    float(q_meta.get("gmag_max", 0) or 0),
                )

            _restore_wcs()
            w0 = WCS(hdul[sci_i].header, hdul, naxis=2)

            # Coarse pass: larger aperture; COM for robustness when far off.
            coarse_method = "com" if method_l == "gaussian" else method_l
            # Coarse: keep all usable matches for a robust rough shift; prefer
            # few high-S/N calibrators only on the fine / final passes.
            coarse_all = measure_gaia_centroid_offsets(
                w0,
                data,
                bad,
                gaia_use,
                search_radius_arcsec=float(coarse_radius_arcsec),
                max_sources=int(max_gaia_sources) if max_gaia_sources and max_gaia_sources > 0 else None,
                centroid_method=coarse_method,
                min_snr=0.0 if coarse_method == "com" else fine_snr,
                isolation_arcsec=0.0,
            )
            dx_c, dy_c, coarse_kept = clip_and_weighted_mean_offset(
                coarse_all, sigma=float(clip_sigma), weight_by_snr=bool(weight_by_snr)
            )
            apply_crpix_shift_to_hdul(hdul, dx_c, dy_c)

            w1 = WCS(hdul[sci_i].header, hdul, naxis=2)
            fine_all = measure_gaia_centroid_offsets(
                w1,
                data,
                bad,
                gaia_use,
                search_radius_arcsec=fine_radius_used,
                max_sources=int(max_gaia_sources) if max_gaia_sources and max_gaia_sources > 0 else None,
                centroid_method=method_l,
                min_snr=0.0 if method_l == "com" else fine_snr,
                isolation_arcsec=iso if method_l == "gaussian" else 0.0,
            )
            fine_all = select_best_calibrators(
                fine_all, max_n=int(preferred_n_calibrators)
            )
            dx_f, dy_f, fine_kept = clip_and_weighted_mean_offset(
                fine_all, sigma=float(clip_sigma), weight_by_snr=bool(weight_by_snr)
            )
            apply_crpix_shift_to_hdul(hdul, dx_f, dy_f)

            w_final = WCS(hdul[sci_i].header, hdul, naxis=2)
            final_matches = measure_gaia_centroid_offsets(
                w_final,
                data,
                bad,
                gaia_use,
                search_radius_arcsec=fine_radius_used,
                max_sources=int(max_gaia_sources) if max_gaia_sources and max_gaia_sources > 0 else None,
                centroid_method=method_l,
                min_snr=0.0 if method_l == "com" else fine_snr,
                isolation_arcsec=iso if method_l == "gaussian" else 0.0,
            )
            final_matches = select_best_calibrators(
                final_matches, max_n=int(preferred_n_calibrators)
            )
            _, _, final_kept = clip_and_weighted_mean_offset(
                final_matches, sigma=float(clip_sigma), weight_by_snr=bool(weight_by_snr)
            )

            n_fit = len(final_kept) if final_kept else len(fine_kept)
            log.info(
                "Gaia simple [%s]: fine matches kept=%d (prefer ≤%d); "
                "coarse dx,dy=(%+.3f,%+.3f) fine dx,dy=(%+.3f,%+.3f) px",
                tier_used,
                n_fit,
                int(preferred_n_calibrators),
                dx_c,
                dy_c,
                dx_f,
                dy_f,
            )
            if n_fit >= int(min_matches):
                break
            if tier_used == "strict" and allow_relaxed_fallback:
                log.warning(
                    "Gaia simple: only %d calibrator(s) after strict cuts "
                    "(need ≥%d); widening quality cuts / apertures as fallback",
                    n_fit,
                    int(min_matches),
                )

        rms_matches = final_kept if len(final_kept) >= 2 else fine_kept
        w_final = WCS(hdul[sci_i].header, hdul, naxis=2)
        kw = rms_abs_keywords_from_matches(w_final, rms_matches)

        # Record summary on primary header (non-destructive; doesn't set TWEAKSUC).
        ph = hdul[0].header
        for k in ("RMSXABS", "RMSYABS", "RMSRAABS", "RMSDEABS", "RMSSKYAB", "NGAIAABS"):
            if k in ph and not kw:
                del ph[k]
        ph["GAIASIMP"] = (True, "hst123: simple Gaia anchor (no JHAT)")
        ph["GSCODX"] = (float(dx_c), "coarse dx applied to CRPIX (px)")
        ph["GSCODY"] = (float(dy_c), "coarse dy applied to CRPIX (px)")
        ph["GSFIDX"] = (float(dx_f), "fine dx applied to CRPIX (px)")
        ph["GSFIDY"] = (float(dy_f), "fine dy applied to CRPIX (px)")
        ph["GSCORAD"] = (float(coarse_radius_arcsec), "coarse search radius (arcsec)")
        ph["GSFIRAD"] = (float(fine_radius_used), "fine search radius (arcsec)")
        ph["GSCLIPS"] = (float(clip_sigma), "sigma clip on |offset|")
        ph["GSCENTM"] = (str(method_l), "gaia_simple HST centroid method")
        ph["GSTIER"] = (str(tier_used), "gaia_simple calibrator tier (strict/relaxed)")
        ph["GSNMAX"] = (int(preferred_n_calibrators), "gaia_simple max calibrators used")
        ph["GSEPPMAP"] = (bool(pm_meta.get("applied")), "gaia_simple: PM/parallax epoch corr")
        ph["GSEPQCUT"] = (bool(q_meta.get("applied")), "gaia_simple: RUWE/AEN/G quality cuts")
        if obstime is not None:
            ph["GSEPMJD"] = (float(obstime.mjd), "gaia_simple: HST epoch used for PM (MJD)")
        for k, v in kw.items():
            if k == "NGAIAABS":
                ph[k] = (int(v), "gaia_simple dispersion (pix or arcsec)")
            else:
                ph[k] = (float(v), "gaia_simple dispersion (pix or arcsec)")

        if write_diagnostics:
            def _write(path: str, rows: list[GaiaSimpleMatch]) -> None:
                t = Table()
                t["ra"] = [m.ra for m in rows]
                t["dec"] = [m.dec for m in rows]
                t["x_pred"] = [m.x_pred for m in rows]
                t["y_pred"] = [m.y_pred for m in rows]
                t["x_meas"] = [m.x_meas for m in rows]
                t["y_meas"] = [m.y_meas for m in rows]
                t["dx"] = [m.dx for m in rows]
                t["dy"] = [m.dy for m in rows]
                t["flux"] = [m.flux for m in rows]
                t["snr"] = [m.snr for m in rows]
                t["n_good"] = [m.n_good for m in rows]
                t.write(path, format="ascii.basic", overwrite=True)

            base = os.path.join(od, os.path.basename(src))
            _write(base + ".gaia_simple_coarse_kept.txt", coarse_kept)
            _write(base + ".gaia_simple_fine_kept.txt", fine_kept)

        hdul.flush()

    if not kw:
        return None
    return {
        "n_match": int(kw.get("NGAIAABS", 0) or 0),
        "rms_ra_as": float(kw.get("RMSRAABS", float("nan"))),
        "rms_dec_as": float(kw.get("RMSDEABS", float("nan"))),
        "rms_sky_as": float(kw.get("RMSSKYAB", float("nan"))),
        "rms_x_pix": float(kw.get("RMSXABS", float("nan"))),
        "rms_y_pix": float(kw.get("RMSYABS", float("nan"))),
        "pm_applied": bool(pm_meta.get("applied")),
        "quality_cuts_applied": bool(q_meta.get("applied")),
        "centroid_method": method_l,
        "fine_radius_arcsec": float(fine_radius_used),
        "tier": str(tier_used),
        "preferred_n_calibrators": int(preferred_n_calibrators),
    }
