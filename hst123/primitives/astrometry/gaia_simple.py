"""
Simple Gaia alignment (translation-only) without JHAT.

This module implements a lightweight, dependency-minimal Gaia anchoring step
intended to replace the default "JHAT Gaia anchor" in drizzle-first mode.

Algorithm (two-pass):
- Prefetch Gaia DR3 cone catalog (reuse existing hst123 cache naming).
- For Gaia sources that project near/on the frame and have unmasked pixels in a
  search radius, measure a flux centroid around the predicted position.
- Compute (dx, dy) = measured - predicted offsets in pixels.
- Sigma-clip outliers on |offset| and take a flux-weighted mean offset.
- Apply offset as a CRPIX shift (translation-only) to the image WCS, then repeat
  with a smaller radius (fine pass).
- Compute absolute dispersion metrics (RMS*ABS) from the final matches.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from hst123.utils.gaia_prefetch import (
    gaia_prefetch_cache_path,
    icrs_field_center_from_fits,
    prefetch_gaia_catalog,
)


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


def _centroid_in_radius(
    data: np.ndarray,
    bad: np.ndarray,
    *,
    x0: float,
    y0: float,
    r_pix: float,
) -> tuple[float, float, float, float, int]:
    """
    Centroid and flux around (x0, y0) within r_pix.

    Returns (x_centroid, y_centroid, flux, bkg, n_good) in full-image pixels.
    """
    try:
        from photutils.centroids import centroid_com  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("gaia_simple requires photutils (pip install photutils).") from exc

    ny, nx = data.shape
    r = float(r_pix)
    if r <= 0:
        raise ValueError("r_pix must be positive")

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
    else:
        bkg = float(np.nanmedian(sub[good]))

    sub2 = sub - bkg
    sub2[~good] = 0.0
    sub_pos = np.where(sub2 > 0, sub2, 0.0)
    if not np.any(sub_pos > 0):
        raise RuntimeError("no positive flux after background subtraction")

    cx2, cy2 = centroid_com(sub_pos)
    if not (np.isfinite(cx2) and np.isfinite(cy2)):
        raise RuntimeError("centroid failed")

    flux = float(np.nansum(sub2[good]))
    x_full = float(x_min) + float(cx2)
    y_full = float(y_min) + float(cy2)
    return x_full, y_full, flux, bkg, n_good


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
) -> list[GaiaSimpleMatch]:
    scale_as = _pixel_scale_arcsec(w)
    r_pix = float(search_radius_arcsec) / float(scale_as)
    if not np.isfinite(r_pix) or r_pix <= 0:
        raise RuntimeError("invalid radius->pixel conversion")

    rows = gaia
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
            x_meas, y_meas, flux, bkg, n_good = _centroid_in_radius(
                data, bad, x0=float(x_pred), y0=float(y_pred), r_pix=r_pix
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
            )
        )
    return out


def clip_and_weighted_mean_offset(
    matches: list[GaiaSimpleMatch],
    *,
    sigma: float = 3.0,
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
    fine_radius_arcsec: float = 0.5,
    max_gaia_sources: int = 4000,
    clip_sigma: float = 3.0,
    write_diagnostics: bool = False,
) -> dict[str, object] | None:
    """
    Align *fits_path* to Gaia by updating its WCS in-place (CRPIX shifts).

    Returns a stats dict (n_match, rms_*_as/pix, etc.) when possible.
    """
    src = os.path.abspath(os.path.expanduser(os.fspath(fits_path)))
    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    os.makedirs(od, exist_ok=True)

    gaia = load_gaia_prefetch_table(image_path=src, outdir=od, radius=gaia_prefetch_radius)

    with fits.open(src, mode="update", memmap=False) as hdul:
        sci_i = _find_sci_hdu_index(hdul)
        data = np.asarray(hdul[sci_i].data, dtype=float)
        w0 = WCS(hdul[sci_i].header, hdul, naxis=2)
        bad = _dq_or_mask_array(hdul, data.shape[-2:])

        coarse_all = measure_gaia_centroid_offsets(
            w0,
            data,
            bad,
            gaia,
            search_radius_arcsec=float(coarse_radius_arcsec),
            max_sources=int(max_gaia_sources) if max_gaia_sources and max_gaia_sources > 0 else None,
        )
        dx_c, dy_c, coarse_kept = clip_and_weighted_mean_offset(coarse_all, sigma=float(clip_sigma))
        apply_crpix_shift_to_hdul(hdul, dx_c, dy_c)

        sci_i2 = _find_sci_hdu_index(hdul)
        w1 = WCS(hdul[sci_i2].header, hdul, naxis=2)
        fine_all = measure_gaia_centroid_offsets(
            w1,
            data,
            bad,
            gaia,
            search_radius_arcsec=float(fine_radius_arcsec),
            max_sources=int(max_gaia_sources) if max_gaia_sources and max_gaia_sources > 0 else None,
        )
        dx_f, dy_f, fine_kept = clip_and_weighted_mean_offset(fine_all, sigma=float(clip_sigma))
        apply_crpix_shift_to_hdul(hdul, dx_f, dy_f)

        sci_i3 = _find_sci_hdu_index(hdul)
        w_final = WCS(hdul[sci_i3].header, hdul, naxis=2)
        kw = rms_abs_keywords_from_matches(w_final, fine_kept)

        # Record summary on primary header (non-destructive; doesn't set TWEAKSUC).
        ph = hdul[0].header
        ph["GAIASIMP"] = (True, "hst123: simple Gaia anchor (no JHAT)")
        ph["GSCODX"] = (float(dx_c), "coarse dx applied to CRPIX (px)")
        ph["GSCODY"] = (float(dy_c), "coarse dy applied to CRPIX (px)")
        ph["GSFIDX"] = (float(dx_f), "fine dx applied to CRPIX (px)")
        ph["GSFIDY"] = (float(dy_f), "fine dy applied to CRPIX (px)")
        ph["GSCORAD"] = (float(coarse_radius_arcsec), "coarse search radius (arcsec)")
        ph["GSFIRAD"] = (float(fine_radius_arcsec), "fine search radius (arcsec)")
        ph["GSCLIPS"] = (float(clip_sigma), "sigma clip on |offset|")
        for k, v in kw.items():
            if k == "NGAIAABS":
                ph[k] = (int(v), "gaia_simple dispersion (pix or arcsec)")
            else:
                ph[k] = (float(v), "gaia_simple dispersion (pix or arcsec)")

        if write_diagnostics:
            # Minimal diagnostics: write coarse/fine kept matches with offsets.
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
                t["n_good"] = [m.n_good for m in rows]
                t.write(path, format="ascii.basic", overwrite=True)

            base = os.path.join(od, os.path.basename(src))
            _write(base + ".gaia_simple_coarse_kept.txt", coarse_kept)
            _write(base + ".gaia_simple_fine_kept.txt", fine_kept)

        hdul.flush()

    if not kw:
        return None
    # Return a jhat-like stats dict for logging compatibility (arcsec components).
    return {
        "n_match": int(kw.get("NGAIAABS", 0) or 0),
        "rms_ra_as": float(kw.get("RMSRAABS", float("nan"))),
        "rms_dec_as": float(kw.get("RMSDEABS", float("nan"))),
        "rms_sky_as": float(kw.get("RMSSKYAB", float("nan"))),
        "rms_x_pix": float(kw.get("RMSXABS", float("nan"))),
        "rms_y_pix": float(kw.get("RMSYABS", float("nan"))),
    }

