"""Run JHAT to align HST/JWST images to Gaia or a user catalog. Requires optional `jhat` package."""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import re
import shutil
import types
import time

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

log = logging.getLogger(__name__)


def _jhat_filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    """Keep only keyword names accepted by *fn* (supports older JHAT builds)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters.values()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return dict(kwargs)
    allowed = set(inspect.signature(fn).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def install_jhat_pandas_read_table_compat() -> None:
    """
    JHAT calls ``pandas.read_table(..., delim_whitespace=...)`` via pdastro.

    pandas 2.2+ removed ``delim_whitespace``; patch ``pd.read_table`` once per process.
    """
    try:
        import inspect

        import pandas as pd  # type: ignore

        sig = inspect.signature(pd.read_table)
        if "delim_whitespace" not in sig.parameters and not hasattr(pd, "_hst123_read_table_compat"):
            _orig_read_table = pd.read_table

            def _read_table_compat(*args, delim_whitespace=None, **kwargs):
                kwargs.pop("delim_whitespace", None)
                if delim_whitespace:
                    kwargs.setdefault("sep", r"\s+")
                    return pd.read_csv(*args, **kwargs)
                return _orig_read_table(*args, **kwargs)

            pd.read_table = _read_table_compat  # type: ignore[assignment]
            pd._hst123_read_table_compat = True  # type: ignore[attr-defined]
    except Exception:
        pass


@contextlib.contextmanager
def jhat_hst_skip_multichip_astrodrizzle():
    """
    Context manager: skip AstroDrizzle inside JHAT for multi-extension HST FLCs.

    For ACS/WFC ``*_flc.fits``, JHAT's ``hst_photclass.load_image`` sets
    ``do_driz=True`` when ``SCI,2`` exists and runs DrizzlePac to build a
    single-plane image before photometry. That is unnecessary for relative
    translation alignment after an FLC CRPIX grid guess.

    Patches ``hst_photclass.prepare_image`` so DrizzlePac is never invoked and
    forces ``do_driz=False`` after load so ``run_align2refcat`` uses the native
    FLC. Photometry then follows JHAT's non-drizzled path (SCI,1–centric).

    Restores class methods on exit.
    """
    try:
        from jhat.simple_jwst_phot import hst_photclass as _hpc
    except ImportError:
        yield
        return

    _orig_prep = _hpc.prepare_image
    _orig_load = _hpc.load_image

    def prepare_image(self, data_original, imhdr, do_driz=False, area=None, dq=None, dq_ignore_bits=2 + 4):
        return _orig_prep(self, data_original, imhdr, False, area=area, dq=dq, dq_ignore_bits=dq_ignore_bits)

    def load_image(self, imagename, imagetype=None, DNunits=False, use_dq=False, skip_preparing=False):
        _orig_load(self, imagename, imagetype=imagetype, DNunits=DNunits, use_dq=use_dq, skip_preparing=skip_preparing)
        self.do_driz = False

    _hpc.prepare_image = prepare_image  # type: ignore[method-assign]
    _hpc.load_image = load_image  # type: ignore[method-assign]
    try:
        yield
    finally:
        _hpc.prepare_image = _orig_prep  # type: ignore[method-assign]
        _hpc.load_image = _orig_load  # type: ignore[method-assign]


def _ensure_hst_sci_extension_for_jhat(align_image: str, *, outdir: str) -> tuple[str, str | None]:
    """
    JHAT's HST photometry loader expects a ``SCI`` extension.

    hst123 sometimes uses single-HDU drizzle products (PRIMARY image only). Those
    are valid FITS images with WCS, but JHAT raises ``Extension 'SCI' not found.``
    for them. Create a temporary 2-HDU wrapper with ``SCI,1`` holding the PRIMARY
    image so JHAT can proceed.

    Returns (path_to_use, temp_path_or_None).
    """
    from astropy.io import fits

    try:
        with fits.open(align_image, mode="readonly") as hdul:
            if "SCI" in hdul:
                return align_image, None
            prim = hdul[0]
            naxis = int(prim.header.get("NAXIS", 0) or 0)
            if naxis < 2 or prim.data is None:
                return align_image, None
            data = np.asarray(prim.data)
            hdr = prim.header.copy()
    except Exception:
        return align_image, None

    # Write a temp wrapper with the *same basename* so JHAT outputs are named
    # consistently (it keys output table names off the input basename).
    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    tmpdir = os.path.join(od, ".hst123_runfiles")
    try:
        os.makedirs(tmpdir, exist_ok=True)
    except Exception:
        tmpdir = od
    tmp_path = os.path.join(tmpdir, os.path.basename(align_image))

    try:
        phdu = fits.PrimaryHDU(header=hdr)
        shdr = hdr.copy()
        shdr["EXTNAME"] = "SCI"
        shdr["EXTVER"] = 1
        sci = fits.ImageHDU(data=data, header=shdr, name="SCI")
        fits.HDUList([phdu, sci]).writeto(tmp_path, overwrite=True, output_verify="silentfix")
        return tmp_path, tmp_path
    except Exception:
        return align_image, None


def _infer_jhat_telescope(align_image: str) -> str:
    """
    JHAT ``st_wcs_align.run_all`` requires ``telescope='hst'`` or ``'jwst'`` (see JHAT HST examples).
    """
    try:
        from astropy.io import fits

        tel = str(fits.getval(align_image, "TELESCOP", ext=0)).strip().upper()
    except Exception:
        tel = ""
    if tel == "JWST":
        return "jwst"
    # Default HST (TELESCOP missing, 'HST', etc.)
    return "hst"


def jhat_gaia_good_phot_path(align_image: str | os.PathLike[str], outdir: str | os.PathLike[str]) -> str:
    """
    Path to JHAT's post-fit matched photometry table (``*_jhat.good.phot.txt``).

    Basename logic matches ``jhat.st_wcs_align.set_outbasename`` / ``update_phottable_final_wcs``.
    """
    base = os.path.basename(os.fspath(align_image))
    inputbasename = re.sub(r"_([a-zA-Z0-9]+)\.fits$", "", base)
    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    return os.path.join(od, f"{inputbasename}_jhat.good.phot.txt")


def jhat_image_phot_path(
    align_image: str | os.PathLike[str], outdir: str | os.PathLike[str]
) -> str:
    """
    Path to JHAT's per-image photometry output (``*.phot.txt``) written by ``run_phot``.

    This file includes at least ``x,y`` and (after JHAT computes WCS for those
    detections) ``ra,dec`` columns, which makes it usable as a relative reference
    catalog for aligning other images.
    """
    base = os.path.basename(os.fspath(align_image))
    inputbasename = re.sub(r"_([a-zA-Z0-9]+)\.fits$", "", base)
    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    # JHAT writes per-image photometry as "<outbasename>.phot.txt" where outbasename
    # is "<outdir>/<inputbasename>" (no "_jhat" infix).
    return os.path.join(od, f"{inputbasename}.phot.txt")


def write_flc_anchor_refcat_for_jhat(
    anchor_image: str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    *,
    max_sources: int = 4000,
    fwhm_px: float = 3.5,
) -> str:
    """
    Write a whitespace-delimited ``ra dec mag dmag`` catalog from sources on the
    anchor FLC using its existing WCS (no Gaia).

    Used for FLC–FLC relative JHAT when Gaia anchoring must be deferred to the
    drizzled reference only: each non-anchor exposure aligns to this catalog via
    ``run_jhat(..., gaia=False, photfilename=...)``.
    """
    from astropy.io import fits as afits
    from astropy.wcs import WCS
    from scipy.ndimage import gaussian_filter, maximum_filter

    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    os.makedirs(od, exist_ok=True)
    base = os.path.basename(os.fspath(anchor_image))
    stem = re.sub(r"\.fits$", "", base, flags=re.I)
    out_path = os.path.join(od, f"{stem}_hst123_anchor_refcat.txt")

    with afits.open(
        os.path.abspath(os.path.expanduser(os.fspath(anchor_image))),
        mode="readonly",
        memmap=False,
    ) as hdul:
        use_hdu = None
        for h in hdul:
            if getattr(h, "data", None) is None:
                continue
            arr = np.asarray(h.data)
            if arr.ndim < 2 or arr.size < 64:
                continue
            if "CRPIX1" not in h.header or "CRVAL1" not in h.header:
                continue
            use_hdu = h
            break
        if use_hdu is None:
            raise RuntimeError(
                f"No science image HDU with WCS found in {anchor_image!r} for anchor refcat."
            )

        data = np.asarray(use_hdu.data, dtype=float)
        hdr = use_hdu.header
        # ACS/WFC3 FLCs use SIP / -TAB distortions backed by NPOL, D2IMARR, etc. in
        # other HDUs; WCS(header) alone raises "HDUList is required for Lookup table distortion".
        w = WCS(hdr, fobj=hdul)
        med = float(np.nanmedian(data))
        sig = float(np.nanstd(data))
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
        resid = data - med
        sm = gaussian_filter(resid, sigma=float(fwhm_px) / 2.35482)
        k = max(3, int(round(float(fwhm_px) * 2)))
        mx = maximum_filter(sm, size=k)
        sel = (sm == mx) & (sm > 5.0 * sig)
        yy, xx = np.nonzero(sel)
        if yy.size == 0:
            raise RuntimeError(
                "Anchor refcat: no peaks detected (try different data or headers)."
            )
        flux = sm[yy, xx]
        order = np.argsort(flux)[::-1][: int(max_sources)]
        xx = xx[order].astype(float) + 1.0
        yy = yy[order].astype(float) + 1.0

        sky = w.pixel_to_world(xx, yy)
        ra = sky.ra.deg
        dec = sky.dec.deg
        zp = None
        if "PHOTZPT" in hdr:
            try:
                zp = float(hdr["PHOTZPT"])
            except (TypeError, ValueError):
                zp = None
        if zp is None:
            zp = 25.0
        mag = zp - 2.5 * np.log10(np.maximum(flux, 1e-12))
        dmag = np.full_like(mag, 0.05, dtype=float)

        n_src = int(len(ra))
        # JHAT loads custom refcats with pd.read_table (header row required); without
        # a header, the first data row is misread as column names.
        with open(out_path, "w", encoding="ascii") as fh:
            fh.write("ra dec mag dmag\n")
            for i in range(n_src):
                fh.write(
                    f"{float(ra[i]):.8f} {float(dec[i]):.8f} "
                    f"{float(mag[i]):.5f} {float(dmag[i]):.5f}\n"
                )

    log.info(
        "JHAT anchor refcat (no Gaia): wrote %d sources → %s",
        n_src,
        os.path.basename(out_path),
    )
    return out_path


def _flc_first_sci_hdu(hdul):
    """First HDU with 2D data and CRPIX/CRVAL (same convention as anchor refcat)."""
    for h in hdul:
        if getattr(h, "data", None) is None:
            continue
        arr = np.asarray(h.data)
        if arr.ndim < 2 or arr.size < 64:
            continue
        if "CRPIX1" not in h.header or "CRVAL1" not in h.header:
            continue
        return h
    return None


def iter_sci_imaging_hdus(hdul):
    """Yield SCI image HDUs that carry a usable astrometric WCS."""
    for h in hdul:
        if getattr(h, "name", "").upper() != "SCI":
            continue
        if getattr(h, "data", None) is None:
            continue
        if "CRPIX1" not in h.header or "CRVAL1" not in h.header:
            continue
        yield h


def _predict_sky_for_jhat_shift_only(hdul, sci_hdus, x, y, ra_ref, dec_ref):
    """
    Predict ICRS sky positions from detector (x, y) for JHAT shift-only alignment.

    Drizzled products typically have a single SCI HDU: one vectorized ``pixel_to_world``
    call. Multi-chip FLCs build each ``WCS`` once; for each source we keep the chip
    whose predicted sky is closest to the reference catalog position (same rule as
    JHAT's original nested loop, without reconstructing ``WCS`` per star per chip).
    """
    from astropy.wcs import WCS

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    ra_ref = np.asarray(ra_ref, dtype=float).reshape(-1)
    dec_ref = np.asarray(dec_ref, dtype=float).reshape(-1)

    wcs_list = [WCS(h.header, hdul) for h in sci_hdus]

    if len(wcs_list) == 1:
        sk = wcs_list[0].pixel_to_world(x, y)
        return np.asarray(sk.ra.deg, dtype=float), np.asarray(sk.dec.deg, dtype=float)

    cref = SkyCoord(ra_ref * u.deg, dec_ref * u.deg, frame="icrs")
    best_sep = np.full(x.shape[0], np.inf, dtype=float)
    best_ra = np.full(x.shape[0], np.nan, dtype=float)
    best_dec = np.full(x.shape[0], np.nan, dtype=float)

    for wj in wcs_list:
        sk = wj.pixel_to_world(x, y)
        ra_j = np.asarray(sk.ra.deg, dtype=float).reshape(-1)
        dec_j = np.asarray(sk.dec.deg, dtype=float).reshape(-1)
        sk_coord = SkyCoord(ra=ra_j * u.deg, dec=dec_j * u.deg, frame="icrs")
        sep = cref.separation(sk_coord).to(u.arcsec).value
        mask = sep < best_sep
        best_sep = np.where(mask, sep, best_sep)
        best_ra = np.where(mask, ra_j, best_ra)
        best_dec = np.where(mask, dec_j, best_dec)

    return best_ra, best_dec


def world_to_pixel_sci_covering_sky(hdul, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """
    Project ICRS ``(ra_deg, dec_deg)`` to FITS pixels using the **SCI** extension
    whose pixel bounding box contains that projection.

    Multi-chip FLCs (ACS/WFC, …): using only SCI,1 for every sky position mis-projects
    positions that lie on another chip and can shift diagnostic comparisons by many pixels.
    """
    from astropy.wcs import WCS

    sci_hdus = list(iter_sci_imaging_hdus(hdul))
    if not sci_hdus:
        raise ValueError("world_to_pixel_sci_covering_sky: no SCI image with WCS")
    for h in sci_hdus:
        w = WCS(h.header, hdul)
        px, py = w.world_to_pixel_values(float(ra_deg), float(dec_deg))
        data = np.asarray(h.data)
        ny, nx = int(data.shape[-2]), int(data.shape[-1])
        if 0.5 < float(px) <= nx + 0.5 and 0.5 < float(py) <= ny + 0.5:
            return float(px), float(py)
    w0 = WCS(sci_hdus[0].header, hdul)
    px, py = w0.world_to_pixel_values(float(ra_deg), float(dec_deg))
    return float(px), float(py)


def extract_peak_pixels_flc(
    align_image: str | os.PathLike[str],
    *,
    max_sources: int = 800,
    fwhm_px: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Local-maximum pixels on the first SCI extension (FITS 1-based coordinates),
    same detection recipe as :func:`write_flc_anchor_refcat_for_jhat`.
    """
    from astropy.io import fits as afits
    from scipy.ndimage import gaussian_filter, maximum_filter

    path = os.path.abspath(os.path.expanduser(os.fspath(align_image)))
    with afits.open(path, mode="readonly", memmap=False) as hdul:
        use_hdu = _flc_first_sci_hdu(hdul)
        if use_hdu is None:
            raise RuntimeError(f"No science image HDU with WCS found in {align_image!r}.")
        data = np.asarray(use_hdu.data, dtype=float)
        med = float(np.nanmedian(data))
        sig = float(np.nanstd(data))
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
        resid = data - med
        sm = gaussian_filter(resid, sigma=float(fwhm_px) / 2.35482)
        k = max(3, int(round(float(fwhm_px) * 2)))
        mx = maximum_filter(sm, size=k)
        sel = (sm == mx) & (sm > 5.0 * sig)
        yy, xx = np.nonzero(sel)
        if yy.size == 0:
            raise RuntimeError("Guess shift: no peaks detected on align image.")
        flux = sm[yy, xx]
        order = np.argsort(flux)[::-1][: int(max_sources)]
        xx = xx[order].astype(np.float64) + 1.0
        yy = yy[order].astype(np.float64) + 1.0
    return xx, yy


def _dispersion_matched_median_arcsec(
    ref_ra_deg: np.ndarray,
    ref_dec_deg: np.ndarray,
    img_ra_deg: np.ndarray,
    img_dec_deg: np.ndarray,
    *,
    dist_limit_arcsec: float,
    sigma_clip: float,
) -> tuple[float, int]:
    """
    Nearest-neighbor match (image → ref); cost = sqrt(median of clipped d2d²), arcsec.

    Returns ``(cost, n_within_limit)`` where *n_within_limit* is the count of
    image peaks whose nearest refcat neighbor lies within *dist_limit_arcsec*.
    Cost is ``+inf`` if there are fewer than three such peaks.
    """
    from astropy.stats import sigma_clipped_stats

    n = int(np.asarray(img_ra_deg).size)
    if n < 3:
        return float("inf"), 0
    cref = SkyCoord(ref_ra_deg * u.deg, ref_dec_deg * u.deg, frame="icrs")
    cimg = SkyCoord(img_ra_deg * u.deg, img_dec_deg * u.deg, frame="icrs")
    _, sep2d, _ = cimg.match_to_catalog_sky(cref)
    sep_as = sep2d.arcsec
    ok = np.isfinite(sep_as) & (sep_as <= float(dist_limit_arcsec))
    sep_ok = sep_as[np.nonzero(ok)[0]]
    n_ok = int(sep_ok.size)
    if sep_ok.size < 3:
        return float("inf"), n_ok
    clip_mean, clip_med, _ = sigma_clipped_stats(
        sep_ok**2,
        sigma_lower=None,
        sigma_upper=float(sigma_clip),
    )
    # sqrt(median variance of separations) ~ robust RMS scale
    med = float(clip_med) if np.isfinite(clip_med) else float(np.median(sep_ok**2))
    return float(np.sqrt(max(med, 0.0))), n_ok


def _refine_crpix_delta_median_pixels(
    w_win,
    xx: np.ndarray,
    yy: np.ndarray,
    ref_ra_deg: np.ndarray,
    ref_dec_deg: np.ndarray,
    *,
    dist_limit_arcsec: float,
) -> tuple[float, float]:
    """
    Sub-pixel Δ(CRPIX) from median peak − predicted-pixel offsets at *w_win*.

    Uses the same image→ref nearest-neighbor pairing and *dist_limit_arcsec* gate
    as :func:`_dispersion_matched_median_arcsec`. For each accepted pair, the
    reference RA/Dec is projected to pixels with *w_win*; the median of
    (peak_x − pred_x, peak_y − pred_y) estimates a small translation to add to the
    grid winner (robust to outliers).
    """
    cref = SkyCoord(ref_ra_deg * u.deg, ref_dec_deg * u.deg, frame="icrs")
    sky = w_win.pixel_to_world(xx, yy)
    cimg = SkyCoord(sky)
    idx, sep2d, _ = cimg.match_to_catalog_sky(cref)
    sep_as = sep2d.arcsec
    ok = np.isfinite(sep_as) & (sep_as <= float(dist_limit_arcsec))
    if np.count_nonzero(ok) < 3:
        return 0.0, 0.0
    rdx: list[float] = []
    rdy: list[float] = []
    for k in np.flatnonzero(ok):
        rf = cref[int(idx[int(k)])]
        px, py = w_win.world_to_pixel(rf)
        rdx.append(float(xx[int(k)] - px))
        rdy.append(float(yy[int(k)] - py))
    return float(np.median(rdx)), float(np.median(rdy))


def guess_shift_hst_flc(
    align_image: str | os.PathLike[str],
    refcat_path: str | os.PathLike[str],
    *,
    radius_px: float | None = None,
    step_px: float = 5.0,
    dist_limit_arcsec: float = 5.0,
    sigma_clip: float = 2.0,
    max_peak_sources: int = 800,
    fwhm_px: float = 3.5,
    refine_crossmatch: bool = True,
) -> tuple[float, float, float, int]:
    """
    Grid search over CRPIX offsets on the first SCI WCS (jwst123-style): for each
    trial shift, project peak pixels to sky and score dispersion vs the anchor
    refcat. Returns ``(dx_px, dy_px, min_cost_arcsec, n_match)`` to **add** to each
    SCI ``CRPIX1``/``CRPIX2`` (same shift on all chips, consistent with
    :func:`apply_jhat_shift_to_science_image`).

    If *refine_crossmatch* is True and the grid finds a finite-cost winner with
    enough pairs, *dx_px* and *dy_px* include an extra sub-pixel correction from
    the median peak−catalog pixel offset at the winning WCS (same NN gate as the
    scorer). *min_cost_arcsec* is still the score at the **discrete** grid winner.

    *n_match* is the number of moving-image peaks with a nearest-neighbor refcat
    match within *dist_limit_arcsec* at the winning CRPIX trial (0 if no finite-cost
    trial).

    Trial CRPIX shifts run from ``-radius_px`` to ``+radius_px`` in steps of
    ``step_px`` on each axis. **radius_px must be strictly less than step_px**
    (enforced by clamping); the default when ``radius_px`` is ``None`` is
    ``step_px / 2``. ``dist_limit_arcsec`` is the maximum NN peak↔refcat separation
    in arcseconds used when scoring each trial (local on-sky window).
    """
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    tab = Table.read(os.path.abspath(os.path.expanduser(os.fspath(refcat_path))), format="ascii")
    if "ra" not in tab.colnames or "dec" not in tab.colnames:
        raise RuntimeError("Anchor refcat must have columns ra, dec.")
    ref_ra = np.asarray(tab["ra"], dtype=float)
    ref_dec = np.asarray(tab["dec"], dtype=float)

    xx, yy = extract_peak_pixels_flc(
        align_image,
        max_sources=max_peak_sources,
        fwhm_px=fwhm_px,
    )

    res = float(step_px)
    if res <= 0:
        raise ValueError("step_px must be positive.")
    if radius_px is None:
        rad = 0.5 * res
    else:
        rad = float(radius_px)
    if rad < 0.0:
        raise ValueError("radius_px must be non-negative.")
    max_rad = float(np.nextafter(res, 0.0))
    if rad >= res:
        log.warning(
            "guess_shift_hst_flc: radius_px (%.6g px) must be < step_px (%.6g px); "
            "clamping to largest value below step.",
            rad,
            res,
        )
        rad = max_rad
    rad = float(min(rad, max_rad))
    xsh = np.arange(-rad, rad + res * 0.5, res, dtype=float)
    ysh = np.arange(-rad, rad + res * 0.5, res, dtype=float)
    if not np.any(np.isclose(xsh, 0.0)):
        xsh = np.sort(np.unique(np.append(xsh, 0.0)))
    if not np.any(np.isclose(ysh, 0.0)):
        ysh = np.sort(np.unique(np.append(ysh, 0.0)))

    path = os.path.abspath(os.path.expanduser(os.fspath(align_image)))
    costs = []
    n_matches = []
    dxs = []
    dys = []
    with afits.open(path, mode="readonly", memmap=False) as hdul:
        use_hdu = _flc_first_sci_hdu(hdul)
        if use_hdu is None:
            raise RuntimeError("Guess shift: no SCI WCS in align image.")
        hdr0 = use_hdu.header
        crpix1 = float(hdr0.get("CRPIX1", np.nan))
        crpix2 = float(hdr0.get("CRPIX2", np.nan))
        if not (np.isfinite(crpix1) and np.isfinite(crpix2)):
            raise RuntimeError("Guess shift: invalid CRPIX in first SCI header.")

        for xs in xsh:
            for ys in ysh:
                hdr_try = hdr0.copy()
                hdr_try["CRPIX1"] = (crpix1 + float(xs), "trial guess shift")
                hdr_try["CRPIX2"] = (crpix2 + float(ys), "trial guess shift")
                w_try = WCS(hdr_try, fobj=hdul)
                sky = w_try.pixel_to_world(xx, yy)
                img_ra = np.asarray(sky.ra.deg, dtype=float)
                img_dec = np.asarray(sky.dec.deg, dtype=float)
                cost, n_ok = _dispersion_matched_median_arcsec(
                    ref_ra,
                    ref_dec,
                    img_ra,
                    img_dec,
                    dist_limit_arcsec=dist_limit_arcsec,
                    sigma_clip=sigma_clip,
                )
                costs.append(cost)
                n_matches.append(int(n_ok))
                dxs.append(float(xs))
                dys.append(float(ys))

        costs_arr = np.asarray(costs, dtype=float)
        if not np.any(np.isfinite(costs_arr)):
            log.warning(
                "Guess shift: no finite score at any CRPIX trial (all inf). "
                "Increase flc_guess_shift_dist_limit_arcsec or radius_px/step_px."
            )
            return 0.0, 0.0, float("inf"), 0
        j = int(np.nanargmin(costs_arr))
        dx_win = dxs[j]
        dy_win = dys[j]
        min_cost = float(costs_arr[j])
        n_win = int(n_matches[j]) if j < len(n_matches) else 0

        dx_out = float(dx_win)
        dy_out = float(dy_win)
        if (
            bool(refine_crossmatch)
            and np.isfinite(min_cost)
            and n_win >= 3
        ):
            hdr_win = hdr0.copy()
            hdr_win["CRPIX1"] = (crpix1 + dx_win, "guess shift grid winner")
            hdr_win["CRPIX2"] = (crpix2 + dy_win, "guess shift grid winner")
            w_win = WCS(hdr_win, fobj=hdul)
            dx_ref, dy_ref = _refine_crpix_delta_median_pixels(
                w_win,
                xx,
                yy,
                ref_ra,
                ref_dec,
                dist_limit_arcsec=dist_limit_arcsec,
            )
            dx_out += float(dx_ref)
            dy_out += float(dy_ref)
            log.debug(
                "guess_shift_hst_flc: grid Δ(CRPIX)=(%.5f, %.5f) px; "
                "cross-match refine=(%.5f, %.5f) px; total=(%.5f, %.5f) px",
                dx_win,
                dy_win,
                dx_ref,
                dy_ref,
                dx_out,
                dy_out,
            )

    return dx_out, dy_out, min_cost, n_win


def apply_crpix_guess_shift_to_flc(
    align_image: str | os.PathLike[str],
    dx_px: float,
    dy_px: float,
    *,
    min_cost_arcsec: float | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """
    Add ``(dx_px, dy_px)`` to ``CRPIX1``/``CRPIX2`` on every SCI extension, and
    record ``HST123GSX``, ``HST123GSY`` on PRIMARY (pixels, same convention as
    grid search).
    """
    from astropy.io import fits as afits

    _log = logger or log
    path = os.path.abspath(os.path.expanduser(os.fspath(align_image)))
    n = 0
    with afits.open(path, mode="update") as hdul:
        for hdu in hdul:
            if getattr(hdu, "name", "").upper() != "SCI":
                continue
            if "CRPIX1" not in hdu.header or "CRPIX2" not in hdu.header:
                continue
            hdu.header["CRPIX1"] = (
                float(hdu.header["CRPIX1"]) + float(dx_px),
                "hst123 FLC guess-shift grid (pre-JHAT)",
            )
            hdu.header["CRPIX2"] = (
                float(hdu.header["CRPIX2"]) + float(dy_px),
                "hst123 FLC guess-shift grid (pre-JHAT)",
            )
            n += 1
        if n:
            hdul[0].header["HST123GSX"] = (float(dx_px), "FLC-FLC CRPIX guess dx (px)")
            hdul[0].header["HST123GSY"] = (float(dy_px), "FLC-FLC CRPIX guess dy (px)")
            hdul.flush()
            cost_s = (
                f"{float(min_cost_arcsec):.4f}"
                if min_cost_arcsec is not None and np.isfinite(min_cost_arcsec)
                else "n/a"
            )
            _log.info(
                "FLC guess shift: applied Δ(CRPIX)=(%.4f, %.4f) px to %d SCI HDU(s) on %s; "
                "grid min cost ≈ %s″",
                float(dx_px),
                float(dy_px),
                n,
                os.path.basename(path),
                cost_s,
            )
    return n


def jhat_corrected_fits_path(align_image: str | os.PathLike[str], outdir: str | os.PathLike[str]) -> str:
    """
    Path to JHAT's corrected FITS (``<root>_jhat.fits``), matching JHAT's basename rule.
    """
    base = os.path.basename(os.fspath(align_image))
    inputbasename = re.sub(r"_([a-zA-Z0-9]+)\.fits$", "", base)
    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    return os.path.join(od, f"{inputbasename}_jhat.fits")


def apply_jhat_shift_to_science_image(
    align_image: str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """
    Copy the sky translation implied by JHAT's ``*_jhat.fits`` onto *align_image*.

    JHAT (including hst123's shift-only shim) writes the updated astrometry to a
    sidecar ``<root>_jhat.fits``. DrizzlePac and :meth:`input_list` use the
    original calibrated file (e.g. ``*_c0m.fits``, ``*_flc.fits``); without this
    step those inputs still carry pre-JHAT WCS and stacks can sit ~1 arcsec off
    Gaia-anchored ACS data. The shift measured on the first JHAT SCI HDU is
    applied to every SCI extension (all WFPC2 chips / ACS chips).
    """
    _log = logger or log
    jhat_path = jhat_corrected_fits_path(align_image, outdir)
    src = os.path.abspath(os.path.expanduser(os.fspath(align_image)))
    if not os.path.isfile(jhat_path) or not os.path.isfile(src):
        _log.debug(
            "JHAT propagate: skip (jhat=%s sci=%s)",
            os.path.isfile(jhat_path),
            os.path.isfile(src),
        )
        return False

    from astropy.io import fits

    def _first_wcs_hdu(hdul):
        for h in hdul:
            if h.data is None and "CRVAL1" not in h.header:
                continue
            if getattr(h, "name", "").upper() == "SCI" and "CRVAL1" in h.header:
                return h
        if hdul[0].data is not None and "CRVAL1" in hdul[0].header:
            return hdul[0]
        return None

    with fits.open(jhat_path, mode="readonly") as jh:
        j_h = _first_wcs_hdu(jh)
        if j_h is None:
            return False
        j_ver = int(j_h.header.get("EXTVER", 1) or 1)
        jh_cr1 = float(j_h.header.get("CRVAL1", float("nan")))
        jh_cr2 = float(j_h.header.get("CRVAL2", float("nan")))
        if not (np.isfinite(jh_cr1) and np.isfinite(jh_cr2)):
            return False

    with fits.open(src, mode="update") as hdul:
        o_h = None
        for h in hdul:
            if getattr(h, "name", "").upper() != "SCI":
                continue
            if "CRVAL1" not in h.header or "CRVAL2" not in h.header:
                continue
            if int(h.header.get("EXTVER", 1) or 1) == j_ver:
                o_h = h
                break
        if o_h is None:
            o_h = _first_wcs_hdu(hdul)
        if o_h is None:
            return False
        o_cr1 = float(o_h.header.get("CRVAL1", float("nan")))
        o_cr2 = float(o_h.header.get("CRVAL2", float("nan")))
        if not (np.isfinite(o_cr1) and np.isfinite(o_cr2)):
            return False
        d1 = jh_cr1 - o_cr1
        d2 = jh_cr2 - o_cr2
        if not (np.isfinite(d1) and np.isfinite(d2)) or (abs(d1) < 1e-12 and abs(d2) < 1e-12):
            return False

        n = 0
        for h in hdul:
            if getattr(h, "name", "").upper() == "SCI" and "CRVAL1" in h.header and "CRVAL2" in h.header:
                h.header["CRVAL1"] = (float(h.header["CRVAL1"]) + d1, "hst123 JHAT dCRVAL1")
                h.header["CRVAL2"] = (float(h.header["CRVAL2"]) + d2, "hst123 JHAT dCRVAL2")
                n += 1
        if n == 0 and hdul[0].data is not None and "CRVAL1" in hdul[0].header:
            hdul[0].header["CRVAL1"] = (
                float(hdul[0].header["CRVAL1"]) + d1,
                "hst123 JHAT dCRVAL1",
            )
            hdul[0].header["CRVAL2"] = (
                float(hdul[0].header["CRVAL2"]) + d2,
                "hst123 JHAT dCRVAL2",
            )
            n = 1
        if n:
            hdul.flush()
            _log.info(
                "JHAT: propagated Δ(CRVAL)=(%.7f°, %.7f°) from %s into %d WCS HDU(s) of %s",
                d1,
                d2,
                os.path.basename(jhat_path),
                n,
                os.path.basename(src),
            )
        return n > 0


def flc_grid_quality_sep_cap_from_min_cost(
    grid_min_cost_arcsec: float,
    *,
    enabled: bool,
    floor_arcsec: float,
    multiplier: float,
    abs_max_arcsec: float,
) -> float | None:
    """
    Derive a separation cap (arcsec) for filtering JHAT phot rows when scoring
    relative-alignment quality, from the FLC CRPIX grid's ``min_cost`` metric.

    ``cap = min(abs_max, max(floor, multiplier * min_cost))`` sets a prior width
    from the grid: disabled or non-finite cost returns ``None`` (no extra filtering).
    Defaults in :mod:`~hst123.settings` use a **wide** floor/abs_max so the cap is
    permissive unless tightened per-project.
    """
    if not enabled:
        return None
    m = float(grid_min_cost_arcsec)
    if not np.isfinite(m) or m <= 0.0:
        return None
    cap = min(float(abs_max_arcsec), max(float(floor_arcsec), float(multiplier) * m))
    return float(cap)


def read_jhat_gaia_residual_stats(
    align_image: str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    *,
    sep_max_arcsec: float | None = None,
):
    """
    RMS residuals of the JHAT solution vs Gaia (ICRS) from the final matched table.

    Rows with non-finite image or reference RA/Dec are excluded so RMS is not
    polluted by NaNs (JHAT often leaves unmatched rows with NaN reference cols).

    Parameters
    ----------
    sep_max_arcsec : float, optional
        If set, only pairs with on-sky separation ≤ this value (arcsec) are used
        for ``n_match`` and RMS statistics (FLC grid–tightened quality gate).

    Returns
    -------
    dict or None
        ``n_match``, ``rms_ra_as``, ``rms_dec_as``, ``rms_sky_as`` (great-circle RMS
        of separations), ``rms_ra_deg``, ``rms_dec_deg`` for FITS ``CRDER*`` (deg),
        and ``rms_sky_deg``. When ``sep_max_arcsec`` is used, also
        ``n_pairs_before_sep_cap`` and ``sep_cap_arcsec``.
    """
    path = jhat_gaia_good_phot_path(align_image, outdir)
    if not os.path.isfile(path):
        log.debug("JHAT Gaia residual table not found: %s", path)
        return None
    try:
        tab = Table.read(path, format="ascii", guess=True)
    except Exception as exc:
        log.warning("Could not read JHAT Gaia phot table %s: %s", path, exc)
        return None
    cols = [str(c) for c in tab.colnames]
    lower = {c.lower(): c for c in cols}
    if "ra" not in lower or "dec" not in lower:
        log.warning("JHAT phot table %s missing ra/dec columns", path)
        return None
    ra_img = lower["ra"]
    dec_img = lower["dec"]

    ref_ra = ref_dec = None
    for key in ("gaia_ra", "gaiadr3_ra", "gaiadr2_ra"):
        if key in lower:
            ref_ra = lower[key]
            break
    if ref_ra is None:
        for c in cols:
            cl = c.lower()
            if cl.endswith("_ra") and cl not in ("ra", "ora", "era"):
                ref_ra = c
                break
    for key in ("gaia_dec", "gaiadr3_dec", "gaiadr2_dec"):
        if key in lower:
            ref_dec = lower[key]
            break
    if ref_dec is None:
        prefix = ref_ra.rsplit("_", 1)[0] if ref_ra else ""
        cand_dec = f"{prefix}_dec" if prefix else None
        if cand_dec and cand_dec in cols:
            ref_dec = cand_dec
        else:
            for c in cols:
                cl = c.lower()
                if cl.endswith("_dec") and cl != "dec":
                    ref_dec = c
                    break
    if ref_ra is None or ref_dec is None:
        log.warning("JHAT phot table %s: could not identify Gaia/ref RA/Dec columns", path)
        return None

    ra_arr = np.asarray(tab[ra_img], dtype=float)
    dec_arr = np.asarray(tab[dec_img], dtype=float)
    rr_arr = np.asarray(tab[ref_ra], dtype=float)
    rd_arr = np.asarray(tab[ref_dec], dtype=float)
    finite = (
        np.isfinite(ra_arr)
        & np.isfinite(dec_arr)
        & np.isfinite(rr_arr)
        & np.isfinite(rd_arr)
    )
    if int(np.count_nonzero(finite)) < 1:
        log.debug("JHAT phot table %s: no finite image/ref coordinate pairs", path)
        return None

    ra_arr = ra_arr[finite]
    dec_arr = dec_arr[finite]
    rr_arr = rr_arr[finite]
    rd_arr = rd_arr[finite]

    try:
        img = SkyCoord(ra_arr * u.deg, dec_arr * u.deg, frame="icrs")
        ref = SkyCoord(rr_arr * u.deg, rd_arr * u.deg, frame="icrs")
    except Exception as exc:
        log.warning("JHAT phot table %s: invalid coordinates: %s", path, exc)
        return None

    sep_as = img.separation(ref).to(u.arcsec).value
    n_before_cap = int(np.size(sep_as))
    if n_before_cap < 1:
        return None

    if sep_max_arcsec is not None and np.isfinite(float(sep_max_arcsec)):
        cap = float(sep_max_arcsec)
        take = sep_as <= cap
        if not np.any(take):
            log.debug(
                "JHAT phot table %s: sep_max_arcsec=%.4f excludes all %d pair(s)",
                os.path.basename(path),
                cap,
                n_before_cap,
            )
            return None
        ra_arr = ra_arr[take]
        dec_arr = dec_arr[take]
        rr_arr = rr_arr[take]
        rd_arr = rd_arr[take]
        sep_as = sep_as[take]

    n = int(np.size(sep_as))
    if n < 1:
        return None

    dec_rad = np.radians(dec_arr)
    dra_deg = ra_arr - rr_arr
    ddec_deg = dec_arr - rd_arr
    dra_as = dra_deg * np.cos(dec_rad) * 3600.0
    ddec_as = ddec_deg * 3600.0

    if n < 2:
        log.debug("JHAT Gaia phot table %s: need ≥2 finite matches for RMS dispersion", path)
        return None

    rms_ra_as = float(np.std(dra_as, ddof=1))
    rms_dec_as = float(np.std(ddec_as, ddof=1))
    rms_sky_as = float(np.sqrt(np.mean(sep_as**2)))

    out: dict = {
        "n_match": n,
        "rms_ra_as": rms_ra_as,
        "rms_dec_as": rms_dec_as,
        "rms_sky_as": rms_sky_as,
        "rms_ra_deg": rms_ra_as / 3600.0,
        "rms_dec_deg": rms_dec_as / 3600.0,
        "rms_sky_deg": rms_sky_as / 3600.0,
        "phot_path": path,
    }
    if sep_max_arcsec is not None and np.isfinite(float(sep_max_arcsec)):
        out["n_pairs_before_sep_cap"] = n_before_cap
        out["sep_cap_arcsec"] = float(sep_max_arcsec)
    return out


def jhat_alignment_acceptable(
    stats: dict | None,
    *,
    max_rms_arcsec: float,
    min_matches: int,
) -> bool:
    """True if JHAT residual stats meet RMS and match-count thresholds."""
    if stats is None:
        return False
    try:
        n = int(stats.get("n_match", 0))
    except (TypeError, ValueError):
        return False
    if n < min_matches:
        return False
    rms = stats.get("rms_sky_as")
    if rms is None:
        return False
    rf = float(rms)
    if not np.isfinite(rf):
        return False
    return rf <= float(max_rms_arcsec)


_MATCH_ONLY_GAIA_RETRY_KEYS = frozenset(
    {"d2d_max", "iterate_with_xyshifts", "Nbright", "Nbright4match"}
)


def jhat_quality_retry_overlay_match_only(attempt_index: int) -> dict[str, object]:
    """
    Subset of :func:`jhat_quality_retry_overlay` for Gaia anchoring: broaden on-sky
    matching and source counts only, keeping sharpness, roundness, SNR, dmag, and
    magnitude limits at the attempt-0 values (same quality cuts as the first pass).
    """
    if attempt_index <= 0:
        return {}
    full = jhat_quality_retry_overlay(attempt_index)
    return {k: v for k, v in full.items() if k in _MATCH_ONLY_GAIA_RETRY_KEYS}


def jhat_quality_retry_overlay(attempt_index: int) -> dict[str, object]:
    """
    Extra kwargs merged into JHAT Gaia-alignment parameters for successive quality retries.

    *attempt_index* 0 is the primary run (empty overlay). Higher indices apply
    progressively relaxed **catalog matching** on the **same** prefetched Gaia
    cone: larger *d2d_max* (arcseconds), broader magnitudes, lower S/N cuts,
    *iterate_with_xyshifts*, and larger *Nbright* / *Nbright4match*. Retries do
    **not** widen the Gaia prefetch radius (that is fixed for the run).
    """
    if attempt_index <= 0:
        return {}
    tiers: tuple[dict[str, object], ...] = (
        {
            "d2d_max": 1.5,
            "dmag_max": 2.0,
            "objmag_lim": (12, 26),
            "SNR_min": 2,
            "sharpness_lim": (0.2, 1.0),
            "roundness1_lim": (-0.9, 0.9),
        },
        {
            "d2d_max": 3.0,
            "dmag_max": 3.0,
            "objmag_lim": (10, 28),
            "SNR_min": 1.5,
            "iterate_with_xyshifts": True,
            "Nbright": 2000,
            "Nbright4match": 2000,
        },
        {
            "d2d_max": 6.0,
            "dmag_max": 4.0,
            "objmag_lim": (8, 29),
            "SNR_min": 1.0,
            "iterate_with_xyshifts": True,
            "Nbright": 4000,
            "Nbright4match": 4000,
        },
    )
    idx = min(attempt_index - 1, len(tiers) - 1)
    return dict(tiers[idx])


_JHAT_GAIA_RUN_ALL_DEFAULTS: dict[str, object] = {
    "overwrite": True,
    "use_dq": False,
    "d2d_max": 0.5,
    "showplots": 0,
    "histocut_order": "dxdy",
    "sharpness_lim": (0.3, 0.9),
    "roundness1_lim": (-0.7, 0.7),
    "SNR_min": 3,
    "dmag_max": 1.0,
    "objmag_lim": (14, 24),
}

# Explicit keyword names from JHAT ``st_wcs_align.run_all`` (remaining keys may patch ``wcs_align``).
_GAIA_RUN_ALL_KNOWN_KEYS = frozenset(
    {
        "telescope",
        "outrootdir",
        "outsubdir",
        "overwrite",
        "use_dq",
        "distortion_file",
        "coron_info_file",
        "skip_if_exists",
        "photometry_method",
        "find_stars_threshold",
        "sci_xy_catalog",
        "refcatname",
        "refcat_racol",
        "refcat_deccol",
        "refcat_magcol",
        "refcat_magerrcol",
        "refcat_colorcol",
        "pmflag",
        "pm_median",
        "photfilename",
        "load_photcat_if_exists",
        "rematch_refcat",
        "SNR_min",
        "d2d_max",
        "dmag_max",
        "sharpness_lim",
        "roundness1_lim",
        "delta_mag_lim",
        "objmag_lim",
        "refmag_lim",
        "Nbright4match",
        "Nbright",
        "histocut_order",
        "slope_min",
        "slope_Nsteps",
        "Nfwhm",
        "xshift",
        "yshift",
        "iterate_with_xyshifts",
        "showplots",
        "saveplots",
        "savephottable",
        "psf_model",
        "ee_radius",
        "use_sextractor",
        "sexpath",
        "sexworkdir",
    }
)


def _merged_gaia_run_all_kw(params_base: dict, attempt_index: int) -> dict[str, object]:
    kw = dict(_JHAT_GAIA_RUN_ALL_DEFAULTS)
    base = dict(params_base)
    match_only_retries = bool(base.pop("_hst123_gaia_anchor_match_only_retries", False))
    kw.update(base)
    if match_only_retries:
        kw.update(jhat_quality_retry_overlay_match_only(attempt_index))
    else:
        kw.update(jhat_quality_retry_overlay(attempt_index))
    return kw


def _loosest_run_phot_kw_for_gaia_retries(
    params_base: dict, *, quality_retry: bool, max_attempts: int
) -> dict[str, object]:
    """
    When quality retries are enabled, ``run_phot`` runs once using the lowest
    ``SNR_min`` and largest ``Nbright4match`` across all planned attempts (same
    net effect as invoking JHAT ``run_all`` repeatedly with relaxed detection).
    """
    if not quality_retry:
        return {}
    snr_mins: list[float] = []
    nb4: list[int] = []
    for a in range(max(1, max_attempts)):
        m = _merged_gaia_run_all_kw(params_base, a)
        snr = m.get("SNR_min")
        if snr is not None:
            try:
                snr_mins.append(float(snr))
            except (TypeError, ValueError):
                pass
        n4 = m.get("Nbright4match")
        if n4 is not None:
            try:
                nb4.append(int(n4))
            except (TypeError, ValueError):
                pass
    out: dict[str, object] = {}
    if snr_mins:
        out["SNR_min"] = min(snr_mins)
    if nb4:
        out["Nbright4match"] = max(nb4)
    return out


def _pop_jhat_gaia_sidecar_keys(extra: dict) -> dict:
    """Prefetch / hst123-only keys not forwarded to JHAT."""
    meta: dict = {}
    for k in (
        "gaia_refcat_path",
        "skip_gaia_prefetch",
        "gaia_prefetch_radius",
        "gaia_prefetch_center",
    ):
        if k in extra:
            meta[k] = extra.pop(k)
    return meta


def _jhat_apply_unknown_run_all_kwargs(wcs_align, merged: dict) -> None:
    """Mirror JHAT ``run_all`` trailing ``**kwargs`` assignment into ``wcs_align``."""
    for k, v in merged.items():
        if k not in _GAIA_RUN_ALL_KNOWN_KEYS and k in wcs_align.__dict__:
            wcs_align.__dict__[k] = v


def _jhat_phot_drop_refcat_match_columns(phot) -> None:
    """Drop columns from a prior refcat match before calling ``match_refcat`` again."""
    if getattr(phot, "refcat", None) is None:
        return
    drop_list: list[str] = ["dx", "dy", "delta_mag"]
    short = getattr(phot.refcat, "short", "") or ""
    if short:
        pat = re.compile(rf"^{re.escape(str(short))}_")
        for col in list(phot.t.columns):
            if pat.match(str(col)):
                drop_list.append(str(col))
    seen: set[str] = set()
    cols = [c for c in drop_list if c in phot.t.columns and not (c in seen or seen.add(c))]
    if cols:
        phot.t = phot.t.drop(columns=cols)


def _jhat_gaia_resolve_refcat_name_pmflag(align_image: str, outdir: str, meta: dict) -> tuple[str, bool]:
    """
    Return ``(refcatname, pmflag)`` where *refcatname* is a file path or the
    literal ``\"Gaia\"`` for the Gaia TAP catalog.
    """
    gaia_refcat_path = meta.get("gaia_refcat_path")
    skip_gaia_prefetch = bool(meta.get("skip_gaia_prefetch", False))
    gaia_prefetch_radius_raw = meta.get("gaia_prefetch_radius")
    gaia_prefetch_center = meta.get("gaia_prefetch_center")

    if gaia_refcat_path is not None:
        return str(gaia_refcat_path), False
    if not skip_gaia_prefetch:
        from hst123 import settings as _hst123_settings
        from hst123.utils.gaia_prefetch import (
            gaia_prefetch_cache_path,
            icrs_field_center_from_fits,
            prefetch_gaia_catalog,
        )

        radius_q = getattr(_hst123_settings, "jhat_gaia_prefetch_radius", 22 * u.arcmin)
        if gaia_prefetch_radius_raw is not None:
            if hasattr(gaia_prefetch_radius_raw, "to"):
                radius_q = u.Quantity(gaia_prefetch_radius_raw).to(u.deg)
            else:
                radius_q = float(gaia_prefetch_radius_raw) * u.deg

        if gaia_prefetch_center is not None:
            if not isinstance(gaia_prefetch_center, SkyCoord):
                raise TypeError("jhat_params['gaia_prefetch_center'] must be a SkyCoord")
            center = gaia_prefetch_center.transform_to("icrs")
        else:
            center = icrs_field_center_from_fits(align_image)

        cache_path = gaia_prefetch_cache_path(outdir, center, radius_q)
        min_bytes = 80
        if os.path.isfile(cache_path) and os.path.getsize(cache_path) >= min_bytes:
            log.info(
                "JHAT: using existing Gaia prefetch catalog %s",
                os.path.basename(cache_path),
            )
            return cache_path, False
        log.info(
            "JHAT: pre-downloading Gaia DR3 cone (r=%s) → %s",
            radius_q,
            os.path.basename(cache_path),
        )
        res = prefetch_gaia_catalog(
            center=center,
            radius=radius_q,
            out_path=cache_path,
        )
        log.info("JHAT: Gaia prefetch complete (%s, n_row=%d).", res.source, res.n_rows)
        return res.path, False

    return "Gaia", True


def _jhat_gaia_vizier_cache_path(align_image: str, outdir: str) -> str:
    from astroquery.vizier import Vizier  # type: ignore

    from hst123 import settings as _hst123_settings
    from hst123.utils.gaia_prefetch import icrs_field_center_from_fits

    _cen = icrs_field_center_from_fits(align_image)
    ra0 = float(_cen.ra.deg)
    dec0 = float(_cen.dec.deg)
    radius_deg = float(
        getattr(_hst123_settings, "jhat_gaia_prefetch_radius", 22 * u.arcmin).to(u.deg).value
    )
    cache_path = os.path.join(outdir, f"gaia_vizier_cache_{ra0:.5f}_{dec0:.5f}_{radius_deg:.5f}.txt")
    if not os.path.isfile(cache_path):
        v = Vizier(columns=["RA_ICRS", "DE_ICRS", "Gmag", "e_Gmag"])
        v.ROW_LIMIT = -1
        tabs = v.query_region(
            SkyCoord(ra0, dec0, unit="deg", frame="icrs"),
            radius=radius_deg * u.deg,
            catalog="I/355/gaiadr3",
        )
        if not tabs:
            raise RuntimeError("Vizier Gaia cone search returned no tables.")
        t = tabs[0]
        out = Table()
        out["ra"] = np.asarray(t["RA_ICRS"], dtype=float)
        out["dec"] = np.asarray(t["DE_ICRS"], dtype=float)
        out["mag"] = np.asarray(t["Gmag"], dtype=float)
        if "e_Gmag" in t.colnames:
            out["dmag"] = np.asarray(t["e_Gmag"], dtype=float)
        else:
            out["dmag"] = np.full(len(out), 0.02, dtype=float)
        out.write(cache_path, format="ascii.basic", overwrite=True)
    return cache_path


def _jhat_gaia_maybe_distortion_swap(
    wcs_align, input_image: str, distortion_file
) -> tuple[str, str | None]:
    _runflag, assignwcs_filename = wcs_align.apply_distortion_coefficients(
        input_image, distortion_file, outdir=os.path.dirname(wcs_align.outbasename)
    )
    return assignwcs_filename, assignwcs_filename


def _jhat_gaia_coron_ixs(wcs_align, coron_info_file):
    phot = wcs_align.phot
    ixs = phot.getindices()
    if phot.instrument == "NIRCAM" and (re.search(r"^MASK", phot.pupil) is not None):
        if coron_info_file is None:
            raise RuntimeError(
                f"pupil={phot.pupil} means coronography, but no coronography info file "
                "was passed (expected CoronInfo.txt)."
            )
        from jhat.pdastro import pdastroclass

        coroninfo = pdastroclass()
        coroninfo.load(coron_info_file)
        ixs_coroninfo = coroninfo.ix_equal("apername", phot.aperture.lower())
        ixs_coroninfo = coroninfo.ix_equal("filter", phot.filtername.lower(), indices=ixs_coroninfo)
        ixs_coroninfo = coroninfo.ix_equal("pupil", phot.pupil.lower(), indices=ixs_coroninfo)
        if len(ixs_coroninfo) > 0:
            if len(ixs_coroninfo) > 1:
                coroninfo.write(indices=ixs_coroninfo)
                raise RuntimeError(
                    f"More than one entry for {phot.aperture.lower()} {phot.filtername.lower()} "
                    f"{phot.pupil.lower()} in {coron_info_file}!"
                )
            ix_coroninfo = ixs_coroninfo[0]
            ixs = phot.ix_inrange("x", coroninfo.t.loc[ix_coroninfo, "xmin1"], coroninfo.t.loc[ix_coroninfo, "xmax1"])
            ixs = phot.ix_inrange(
                "y", coroninfo.t.loc[ix_coroninfo, "ymin1"], coroninfo.t.loc[ix_coroninfo, "ymax1"], indices=ixs
            )
            if coroninfo.t.loc[ix_coroninfo, "xmin2"] != np.nan:
                ixs_tmp = phot.ix_inrange(
                    "x", coroninfo.t.loc[ix_coroninfo, "xmin2"], coroninfo.t.loc[ix_coroninfo, "xmax2"]
                )
                ixs_tmp = phot.ix_inrange(
                    "y", coroninfo.t.loc[ix_coroninfo, "ymin2"], coroninfo.t.loc[ix_coroninfo, "ymax2"], indices=ixs_tmp
                )
                ixs = np.concatenate((ixs, ixs_tmp), axis=0)
    return ixs


def _jhat_gaia_call_load_and_match(
    wcs_align,
    *,
    ixs_use,
    refcatname: str,
    pmflag: bool,
    pm_median: bool,
    refcat_racol,
    refcat_deccol,
    refcat_magcol,
    refcat_magerrcol,
    refcat_colorcol,
    refmag_lim,
    initialize_only: bool,
) -> None:
    wcs_align.phot.load_and_match_refcat(
        ixs_obj=ixs_use,
        refcatname=refcatname,
        refcat_racol=refcat_racol,
        refcat_deccol=refcat_deccol,
        refcat_magcol=refcat_magcol,
        refcat_magerrcol=refcat_magerrcol,
        refcat_colorcol=refcat_colorcol,
        refmag_lim=refmag_lim,
        refmagerr_lim=(None, None),
        refcolor_lim=(None, None),
        pmflag=pmflag,
        pm_median=pm_median,
        initialize_only=initialize_only,
    )


def _jhat_gaia_first_load_or_match(
    wcs_align,
    align_image: str,
    outdir: str,
    *,
    ixs_use,
    refcatname: str,
    pmflag: bool,
    pm_median: bool,
    refcat_racol,
    refcat_deccol,
    refcat_magcol,
    refcat_magerrcol,
    refcat_colorcol,
    refmag_lim,
    rematch_refcat: bool,
    photcat_loaded: bool,
) -> tuple[str, bool]:
    """
    First in-memory reference association for this run.

    Returns ``(refcatname_effective, pmflag_effective)`` after optional Vizier
    fallback when the Gaia TAP name is used.
    """
    initialize_only = (not rematch_refcat) and photcat_loaded
    if refcatname != "Gaia":
        _jhat_gaia_call_load_and_match(
            wcs_align,
            ixs_use=ixs_use,
            refcatname=refcatname,
            pmflag=pmflag,
            pm_median=pm_median,
            refcat_racol=refcat_racol,
            refcat_deccol=refcat_deccol,
            refcat_magcol=refcat_magcol,
            refcat_magerrcol=refcat_magerrcol,
            refcat_colorcol=refcat_colorcol,
            refmag_lim=refmag_lim,
            initialize_only=initialize_only,
        )
        return refcatname, pmflag

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            _jhat_gaia_call_load_and_match(
                wcs_align,
                ixs_use=ixs_use,
                refcatname="Gaia",
                pmflag=True,
                pm_median=pm_median,
                refcat_racol=refcat_racol,
                refcat_deccol=refcat_deccol,
                refcat_magcol=refcat_magcol,
                refcat_magerrcol=refcat_magerrcol,
                refcat_colorcol=refcat_colorcol,
                refmag_lim=refmag_lim,
                initialize_only=initialize_only,
            )
            return "Gaia", True
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            retryable = (
                "Error 500" in msg
                or "Cannot find result 'result'" in msg
                or "RemoteServiceError" in msg
                or "requests.exceptions.HTTPError" in msg
            )
            if not retryable or attempt >= 3:
                break
            log.warning("JHAT Gaia query failed (attempt %d/3): %s", attempt, exc)
            time.sleep(2.0 * attempt)

    if last_exc is None:
        return "Gaia", True

    try:
        cache_path = _jhat_gaia_vizier_cache_path(align_image, outdir)
        log.warning(
            "JHAT Gaia TAP failed; falling back to cached Vizier Gaia DR3 catalog: %s",
            os.path.basename(cache_path),
        )
        _jhat_gaia_call_load_and_match(
            wcs_align,
            ixs_use=ixs_use,
            refcatname=cache_path,
            pmflag=False,
            pm_median=pm_median,
            refcat_racol="ra",
            refcat_deccol="dec",
            refcat_magcol="mag",
            refcat_magerrcol="dmag",
            refcat_colorcol=refcat_colorcol,
            refmag_lim=refmag_lim,
            initialize_only=initialize_only,
        )
        return cache_path, False
    except Exception:
        raise last_exc


def _jhat_gaia_run_phot_once(
    wcs_align,
    input_image: str,
    outdir: str,
    *,
    verbose: bool,
    xshift: float,
    yshift: float,
    merged_loose: dict,
) -> tuple[bool, str, str | None]:
    """Run ``run_phot`` once.

    Returns ``(photcat_loaded, work_image_path, assignwcs_temp_or_none)`` where
    *work_image_path* is the FITS path passed to ``run_align2refcat`` (distortion
    product when ``distortion_file`` is set). *assignwcs_temp_or_none* is removed
    after all quality attempts (JHAT temporary assign-WCS file).
    """
    _jhat_apply_unknown_run_all_kwargs(wcs_align, dict(merged_loose))

    m = merged_loose
    telescope = m.get("telescope") or _infer_jhat_telescope(input_image)
    overwrite = bool(m.get("overwrite", False))
    use_dq = bool(m.get("use_dq", False))
    load_photcat_if_exists = bool(m.get("load_photcat_if_exists", False))
    photometry_method = str(m.get("photometry_method", "aperture"))
    find_stars_threshold = float(m.get("find_stars_threshold", 3.0))
    sci_xy_catalog = m.get("sci_xy_catalog")
    psf_model = m.get("psf_model")
    ee_radius = m.get("ee_radius", 70)
    use_sextractor = bool(m.get("use_sextractor", False))
    sexpath = m.get("sexpath", "sex")
    sexworkdir = m.get("sexworkdir")
    snr_min = m.get("SNR_min")
    nbright4match = m.get("Nbright4match")
    distortion_file = m.get("distortion_file")

    wcs_align.verbose = bool(verbose)
    wcs_align.set_outbasename(outrootdir=outdir, outsubdir=None, inputname=input_image)
    wcs_align.set_telescope(telescope=telescope, imname=input_image)

    work_image = input_image
    assign_tmp: str | None = None
    if distortion_file is not None:
        work_image, assign_tmp = _jhat_gaia_maybe_distortion_swap(wcs_align, input_image, distortion_file)

    wcs_align.phot.verbose = bool(verbose)
    photfilename = f"{wcs_align.outbasename}.phot.txt"
    _phot_kw = dict(
        use_dq=use_dq,
        photfilename=photfilename,
        load_photcat_if_exists=load_photcat_if_exists,
        overwrite=overwrite,
        Nbright4match=nbright4match,
        SNR_min=snr_min,
        xshift=xshift,
        yshift=yshift,
        ee_radius=ee_radius,
        sci_xy_catalog=sci_xy_catalog,
        psf_model=psf_model,
        photometry_method=photometry_method,
        find_stars_threshold=find_stars_threshold,
        use_sextractor=use_sextractor,
        sexpath=sexpath,
        sexworkdir=sexworkdir,
    )
    _phot_kw = _jhat_filter_kwargs_for_callable(wcs_align.phot.run_phot, _phot_kw)
    photfilename_4check, photcat_loaded = wcs_align.phot.run_phot(work_image, **_phot_kw)
    if photfilename != photfilename_4check:
        raise RuntimeError(f"JHAT phot filename mismatch: {photfilename} != {photfilename_4check}")
    # *work_image* is the file path photometry used (distortion-swapped FITS when applicable).
    return photcat_loaded, work_image, assign_tmp


def _jhat_gaia_match_align_one_attempt(
    wcs_align,
    input_image: str,
    align_image_for_tap: str,
    outdir: str,
    *,
    verbose: bool,
    attempt_index: int,
    params_base: dict,
    refcatname0: str,
    pmflag0: bool,
    photcat_loaded: bool,
    refcat_in_memory: bool,
) -> tuple[int, bool, str, bool]:
    """
    Per quality attempt: ``initial_cut_photcat`` → ``load_and_match_refcat`` or
    ``match_refcat`` → ``find_good_refcat_matches`` → ``run_align2refcat`` →
    ``update_phottable_final_wcs`` (no ``run_all``).
    """
    merged = _merged_gaia_run_all_kw(params_base, attempt_index)
    _jhat_apply_unknown_run_all_kwargs(wcs_align, dict(merged))
    wcs_align.verbose = bool(verbose)
    wcs_align.phot.verbose = bool(verbose)

    savephottable = int(merged.get("savephottable", 1))
    overwrite = bool(merged.get("overwrite", False))
    skip_if_exists = bool(merged.get("skip_if_exists", False))
    rematch_refcat = bool(merged.get("rematch_refcat", False))
    photometry_method = str(merged.get("photometry_method", "aperture"))
    coron_info_file = merged.get("coron_info_file")
    sci_xy_catalog = merged.get("sci_xy_catalog")

    refcatname = refcatname0
    pmflag = pmflag0

    if os.path.isfile(str(refcatname)):
        refcat_racol = merged.get("refcat_racol", "ra")
        refcat_deccol = merged.get("refcat_deccol", "dec")
        refcat_magcol = merged.get("refcat_magcol", "mag")
        refcat_magerrcol = merged.get("refcat_magerrcol", "dmag")
    else:
        refcat_racol = merged.get("refcat_racol", "auto")
        refcat_deccol = merged.get("refcat_deccol", "auto")
        refcat_magcol = merged.get("refcat_magcol")
        refcat_magerrcol = merged.get("refcat_magerrcol")
    refcat_colorcol = merged.get("refcat_colorcol")
    pm_median = bool(merged.get("pm_median", False))

    dmag_max = merged.get("dmag_max")
    sharpness_lim = merged.get("sharpness_lim", (None, None))
    roundness1_lim = merged.get("roundness1_lim", (None, None))
    objmag_lim = merged.get("objmag_lim", (None, None))
    nbright = merged.get("Nbright")
    refmag_lim = merged.get("refmag_lim", (None, None))

    d2d_max = merged.get("d2d_max")
    delta_mag_lim = merged.get("delta_mag_lim", (None, None))
    histocut_order = merged.get("histocut_order", "dxdy")
    slope_min = float(merged.get("slope_min", -10 / 2048.0))
    slope_nsteps = int(merged.get("slope_Nsteps", 200))
    nfwhm = float(merged.get("Nfwhm", 2.5))
    showplots = merged.get("showplots", 0)
    iterate_xy = bool(merged.get("iterate_with_xyshifts", False))
    saveplots = int(merged.get("saveplots", 0))

    already_matched = sci_xy_catalog is not None

    phot = wcs_align.phot
    ixs = _jhat_gaia_coron_ixs(wcs_align, coron_info_file)
    # Some JHAT builds optionally support pixel-bound cuts (xmin/xmax/ymin/ymax) in
    # initial_cut_photcat; keep these purely optional and signature-gated.
    _cut_kw = dict(
        dmag_max=dmag_max,
        sharpness_lim=sharpness_lim,
        roundness1_lim=roundness1_lim,
        objmag_lim=objmag_lim,
        Nbright=nbright,
        ixs=ixs,
        xmin=merged.get("xmin"),
        xmax=merged.get("xmax"),
        ymin=merged.get("ymin"),
        ymax=merged.get("ymax"),
    )
    ixs_use = phot.initial_cut_photcat(**_jhat_filter_kwargs_for_callable(phot.initial_cut_photcat, _cut_kw))

    # Optional diagnostics: dump the post-cut photometry catalog so users can inspect
    # which detections are eligible for Gaia matching.
    if bool(merged.get("hst123_dump_initial_cut_photcat", False)):
        try:
            dump_path = str(merged.get("hst123_dump_initial_cut_photcat_path") or "")
            if not dump_path:
                dump_path = f"{wcs_align.outbasename}.initial_cut.phot.txt"
            tab = Table.from_pandas(phot.t.loc[ixs_use].copy())
            tab.write(dump_path, format="ascii.basic", overwrite=True)
            log.info("JHAT Gaia: wrote initial_cut_photcat catalog → %s", os.path.basename(dump_path))
        except Exception as exc:
            log.warning("JHAT Gaia: could not dump initial_cut_photcat catalog: %s", exc)

    if photometry_method == "1pass":
        wcs_align.psfphot_1pass_jwst(input_image, ixs=ixs_use)

    first_load = not refcat_in_memory
    if first_load:
        refcatname, pmflag = _jhat_gaia_first_load_or_match(
            wcs_align,
            align_image_for_tap,
            outdir,
            ixs_use=ixs_use,
            refcatname=refcatname,
            pmflag=pmflag,
            pm_median=pm_median,
            refcat_racol=refcat_racol,
            refcat_deccol=refcat_deccol,
            refcat_magcol=refcat_magcol,
            refcat_magerrcol=refcat_magerrcol,
            refcat_colorcol=refcat_colorcol,
            refmag_lim=refmag_lim,
            rematch_refcat=rematch_refcat,
            photcat_loaded=photcat_loaded,
        )
        refcat_in_memory = True
    else:
        _jhat_phot_drop_refcat_match_columns(phot)
        phot.match_refcat(ixs_obj=ixs_use, ixs_refcat=phot.ixs_use_refcat)

    # Optional diagnostics: dump the loaded/eligible reference catalog rows.
    if bool(merged.get("hst123_dump_refcat_for_match", False)):
        try:
            ixs_ref = getattr(phot, "ixs_use_refcat", None)
            if ixs_ref is None and getattr(phot, "refcat", None) is not None:
                ixs_ref = phot.refcat.getindices()
            dump_path = str(merged.get("hst123_dump_refcat_for_match_path") or "")
            if not dump_path:
                dump_path = f"{wcs_align.outbasename}.refcat_for_match.txt"
            if getattr(phot, "refcat", None) is not None and ixs_ref is not None:
                tab = Table.from_pandas(phot.refcat.t.loc[ixs_ref].copy())
                tab.write(dump_path, format="ascii.basic", overwrite=True)
                log.info("JHAT Gaia: wrote refcat-for-match catalog → %s", os.path.basename(dump_path))
        except Exception as exc:
            log.warning("JHAT Gaia: could not dump refcat-for-match catalog: %s", exc)

    refcatfilename = f"{wcs_align.outbasename}.refcat.txt"
    log.info("Saving refcat file into %s", refcatfilename)
    phot.refcat.write(refcatfilename, overwrite=True)

    _fgm_kw = dict(
        ixs=ixs_use,
        d2d_max=d2d_max,
        delta_mag_lim=delta_mag_lim,
        refmag_lim=refmag_lim,
        histocut_order=histocut_order,
        slope_min=slope_min,
        slope_Nsteps=slope_nsteps,
        Nfwhm=nfwhm,
        show_initial_plot=showplots,
        show_histofit_plots=showplots,
        savephottable=savephottable,
        outbasename=wcs_align.outbasename,
        binsize_px=wcs_align.binsize_px,
        already_matched=already_matched,
        Nbright=nbright,
    )
    ixs_best = wcs_align.find_good_refcat_matches(
        **_jhat_filter_kwargs_for_callable(wcs_align.find_good_refcat_matches, _fgm_kw)
    )

    if iterate_xy:
        dx_median = phot.t.loc[ixs_best, "dx"].median()
        dy_median = phot.t.loc[ixs_best, "dy"].median()
        if wcs_align.verbose:
            print(
                f"dx median of best matched objects of 1st iteration: {dx_median}",
                f"dy median of best matched objects of 1st iteration: {dy_median}",
            )
        refcatcols = ["dx", "dy", "delta_mag"]
        short = getattr(phot.refcat, "short", None) if getattr(phot, "refcat", None) is not None else None
        if short:
            pat = re.compile(rf"^{re.escape(str(short))}_")
            for col in phot.t.columns:
                if pat.match(str(col)):
                    refcatcols.append(str(col))
        phot.t = phot.t.drop(columns=[c for c in refcatcols if c in phot.t.columns])
        phot.xy_to_radec(indices=ixs_use, xshift=float(dx_median), yshift=float(dy_median))
        phot.match_refcat(ixs_obj=ixs_use, ixs_refcat=phot.ixs_use_refcat)
        _fgm_kw2 = dict(
            ixs=ixs_use,
            d2d_max=d2d_max,
            delta_mag_lim=delta_mag_lim,
            refmag_lim=refmag_lim,
            histocut_order=histocut_order,
            slope_min=slope_min,
            slope_Nsteps=slope_nsteps,
            Nfwhm=nfwhm,
            show_initial_plot=0,
            show_histofit_plots=showplots,
            savephottable=savephottable,
            outbasename=wcs_align.outbasename,
            binsize_px=wcs_align.binsize_px,
            already_matched=already_matched,
            Nbright=nbright,
        )
        ixs_best = wcs_align.find_good_refcat_matches(
            **_jhat_filter_kwargs_for_callable(wcs_align.find_good_refcat_matches, _fgm_kw2)
        )

    jhatfits = f"{wcs_align.outbasename}_jhat.fits"
    _runflag, jhat_out = wcs_align.run_align2refcat(
        input_image,
        outputfits=jhatfits,
        ixs=ixs_best,
        overwrite=overwrite,
        skip_if_exists=skip_if_exists,
    )

    wcs_align.update_phottable_final_wcs(
        jhat_out,
        ixs_bestmatch=ixs_best,
        showplots=showplots,
        saveplots=saveplots,
        savephottable=savephottable,
        overwrite=overwrite,
    )

    return savephottable, refcat_in_memory, refcatname, pmflag


def _jhat_relative_run_one_attempt(
    wcs_align,
    align_image: str,
    outdir: str,
    photfilename: str,
    verbose: bool,
    xshift: float,
    yshift: float,
    Nbright_arg: int,
    extra_src: dict,
) -> int:
    """Relative alignment against *photfilename*; returns *savephottable* flag."""
    extra = dict(extra_src)
    tel = extra.pop("telescope", None) or _infer_jhat_telescope(align_image)
    rel_kw = {
        "telescope": tel,
        "overwrite": True,
        "showplots": 0,
        "refcat_racol": "ra",
        "refcat_deccol": "dec",
        "refcat_magcol": "mag",
        "refcat_magerrcol": "dmag",
        **extra,
    }
    rel_kw.pop("outrootdir", None)
    rel_kw.pop("outsubdir", None)
    savephottable = int(rel_kw.pop("savephottable", 1))
    nbright = int(rel_kw.pop("Nbright", Nbright_arg))
    wcs_align.run_all(
        align_image,
        outrootdir=outdir,
        outsubdir=None,
        refcatname=photfilename,
        use_dq=False,
        verbose=verbose,
        xshift=xshift,
        yshift=yshift,
        savephottable=savephottable,
        Nbright=nbright,
        **rel_kw,
    )
    return savephottable



def run_jhat(
    align_image,
    outdir,
    params,
    gaia=False,
    photfilename=None,
    xshift=0,
    yshift=0,
    Nbright=800,
    verbose=False,
    *,
    quality_sep_cap_arcsec: float | None = None,
):
    """
    Run JHAT to align an HST or JWST image to Gaia or a photometric reference catalog.

    Parameters
    ----------
    align_image : str
        Image to align.
    outdir : str
        Output directory.
    params : dict
        Parameters for JHAT (e.g. strict_gaia_params, strict_jwst_params).
        For *gaia* = True, these drive a custom pipeline: ``run_phot`` once, then
        per-attempt ``load_and_match_refcat`` / ``match_refcat``,
        ``find_good_refcat_matches``, and alignment (no ``run_all``). For
        *gaia* = False, keyword arguments are passed through to
        ``st_wcs_align().run_all()``.

        Gaia-related keys (consumed by hst123, not forwarded to JHAT):

        ``gaia_refcat_path`` (str), optional
            Use this local refcat (``ra``, ``dec``, ``mag``, ``dmag``) directly; the
            automatic prefetch step is skipped.

        ``skip_gaia_prefetch`` (bool), optional
            If True, do not pre-download Gaia; JHAT uses its built-in ``refcatname="Gaia"``
            TAP path (legacy behavior).

        ``gaia_prefetch_radius``
            Cone radius as an ``astropy.units.Quantity`` or a float interpreted as **degrees**.

        ``gaia_prefetch_center`` (:class:`~astropy.coordinates.SkyCoord`), optional
            Override field center for the prefetch cone (ICRS). Default: image center
            from the input FITS WCS.
    gaia : bool, optional
        If True, align to Gaia. Default is False.
    photfilename : str, optional
        Photometry file name (required when gaia is False).
    xshift, yshift : float, optional
        x and y shift in pixels. Default 0.
    Nbright : int, optional
        Number of bright stars to use. Default 800.
    verbose : bool, optional
        Verbose output. Default False.
    quality_sep_cap_arcsec : float, optional
        For *gaia* = False only: if set, :func:`read_jhat_gaia_residual_stats` only
        uses photometry rows with image–reference separation ≤ this (arcsec) when
        computing RMS for the quality-retry loop (e.g. derived from the FLC grid
        ``min_cost``). Default ``None`` uses all finite pairs.
    Notes
    -----
    When :mod:`~hst123.settings` ``jhat_quality_retry_enabled`` is true (default),
    each alignment run may repeat with progressively relaxed parameters until
    residuals in the matched photometry table fall below
    ``jhat_max_acceptable_rms_arcsec`` or attempts are exhausted. For *gaia* =
    True (Gaia anchor), magnitude limits on both catalogs are fixed by
    ``jhat_gaia_anchor_mag_lim`` (default 15–22 mag), and retries only relax
    on-sky matching and source-count settings (*d2d_max*, *Nbright*,
    *Nbright4match*, *iterate_with_xyshifts*), leaving sharpness, roundness,
    *SNR_min*, *dmag_max*, and magnitudes at the first-pass values. Relative
    (*gaia* = False) runs still use the full retry overlay (including relaxed
    magnitude and detection cuts).

    Returns
    -------
    dict or None
        When *gaia* is True and ``savephottable`` leaves a ``*_jhat.good.phot.txt``
        file, returns :func:`read_jhat_gaia_residual_stats`; otherwise ``None``.
        When *gaia* is False, returns ``None``.

    Raises
    ------
    ImportError
        If the `jhat` package is not installed.
    ValueError
        If gaia is False and photfilename is None.
    """
    try:
        from jhat import st_wcs_align
    except ImportError as e:
        raise ImportError(
            "run_jhat requires the jhat package. Install with: pip install jhat"
        ) from e

    from hst123.primitives.astrometry.jhat_wfpc2_patch import (
        ensure_jhat_hst_phot_wfpc2_patch,
        pop_jhat_ee_calibration_dir,
        push_jhat_ee_calibration_dir,
    )

    ensure_jhat_hst_phot_wfpc2_patch()

    wcs_align = st_wcs_align()
    align_image = os.path.abspath(os.path.expanduser(os.fspath(align_image)))
    # JHAT appends outsubdir to outrootdir (default '.'). Passing an absolute path
    # as outsubdir yields './/abs/...' and breaks photometry / shift file locations.
    outdir = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    params_base = dict(params or {})
    extra = dict(params_base)

    # If this is an HST PRIMARY-only FITS, wrap to include SCI for JHAT.
    tmp_to_cleanup = None
    try:
        tel0 = (extra.get("telescope") or _infer_jhat_telescope(align_image)).strip().lower()
        if tel0 == "hst":
            align_image_use, tmp_to_cleanup = _ensure_hst_sci_extension_for_jhat(
                align_image, outdir=outdir
            )
            align_image = align_image_use
    except Exception:
        tmp_to_cleanup = None

    install_jhat_pandas_read_table_compat()

    # -------------------------------------------------------------------------
    # Safety shim: avoid known segfault in tweakwcs.imalign.align_wcs() invoked
    # by JHAT via tweakreg_hack on some HST inputs (MultiSlitModel from *_flc.fits).
    # Instead, apply a robust translation-only update to CRVAL based on the
    # JHAT-matched objects, keeping the original distortion (SIP) terms intact.
    # -------------------------------------------------------------------------
    tel_infer = (extra.get("telescope") or _infer_jhat_telescope(align_image)).strip().lower()
    if tel_infer == "hst":

        def _run_align2refcat_shift_only(
            self,
            imfilename,
            *,
            outputfits=None,
            phot=None,
            ixs=None,
            refcat_racol=None,
            refcat_deccol=None,
            xcol="x",
            ycol="y",
            outdir=None,
            overwrite=False,
            skip_if_exists=False,
            savephot=True,
        ):
            from astropy.io import fits
            from astropy.wcs import WCS

            if phot is None:
                phot = self.phot
            if ixs is None or len(ixs) < 1:
                raise RuntimeError("JHAT safe align: no matched objects (ixs empty).")

            if refcat_racol is None:
                refcat_racol = phot.refcat_racol
            if refcat_deccol is None:
                refcat_deccol = phot.refcat_deccol

            if outputfits is None:
                base = os.path.basename(str(imfilename))
                outbase = re.sub(r"_([a-zA-Z0-9]+)\.fits$", "", base) + "_jhat.fits"
                if outdir is None:
                    outdir = getattr(self, "outdir", None) or os.path.dirname(str(imfilename))
                outputfits = os.path.join(str(outdir), outbase)

            if os.path.exists(outputfits):
                if not overwrite:
                    if skip_if_exists:
                        return (False, outputfits)
                    raise RuntimeError(f"Output exists: {outputfits}")
                try:
                    os.remove(outputfits)
                except Exception:
                    pass

            # Copy input → output, then update WCS in-place.
            shutil.copyfile(str(imfilename), str(outputfits))

            with fits.open(str(outputfits), mode="update") as hdul:
                sci_hdus = list(iter_sci_imaging_hdus(hdul))
                if not sci_hdus:
                    if hdul[0].data is not None and "CRVAL1" in hdul[0].header:
                        sci_hdus = [hdul[0]]
                    else:
                        raise RuntimeError("JHAT safe align: no SCI image HDU with WCS found.")

                t = phot.t.loc[ixs, [xcol, ycol, refcat_racol, refcat_deccol]]
                x = np.asarray(t[xcol], dtype=float)
                y = np.asarray(t[ycol], dtype=float)
                ra_ref = np.asarray(t[refcat_racol], dtype=float)
                dec_ref = np.asarray(t[refcat_deccol], dtype=float)

                log.info(
                    "JHAT shift-only: predicting sky for %d matched sources (%d SCI HDUs).",
                    int(x.size),
                    len(sci_hdus),
                )
                ra_pred, dec_pred = _predict_sky_for_jhat_shift_only(
                    hdul, sci_hdus, x, y, ra_ref, dec_ref
                )

                good = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(ra_ref)
                    & np.isfinite(dec_ref)
                    & np.isfinite(ra_pred)
                    & np.isfinite(dec_pred)
                )
                if not np.any(good):
                    raise RuntimeError("JHAT safe align: all matched points are non-finite.")

                dra_deg = (ra_ref[good] - ra_pred[good] + 180.0) % 360.0 - 180.0
                ddec_deg = dec_ref[good] - dec_pred[good]

                # Median sky residual (degrees) for a rigid CRVAL translation — small-angle.
                dra_med = float(np.nanmedian(dra_deg))
                ddec_med = float(np.nanmedian(ddec_deg))
                if not (np.isfinite(dra_med) and np.isfinite(ddec_med)):
                    raise RuntimeError("JHAT safe align: computed non-finite median WCS shift.")

                for hdu in sci_hdus:
                    hdr = hdu.header
                    wloc = WCS(hdr, hdul)
                    crval1 = float(hdr.get("CRVAL1", wloc.wcs.crval[0]))
                    crval2 = float(hdr.get("CRVAL2", wloc.wcs.crval[1]))
                    if not np.isfinite(crval1) or not np.isfinite(crval2):
                        crval1, crval2 = map(float, wloc.wcs.crval)
                    hdr["CRVAL1"] = (
                        crval1 + dra_med,
                        "Updated by hst123 (JHAT safe shift-only)",
                    )
                    hdr["CRVAL2"] = (
                        crval2 + ddec_med,
                        "Updated by hst123 (JHAT safe shift-only)",
                    )
                    hdu.header = hdr
                hdul.flush()

            return (True, outputfits)

        # Monkeypatch the instance method that triggers the segfault.
        try:
            wcs_align.run_align2refcat = types.MethodType(_run_align2refcat_shift_only, wcs_align)
            log.warning(
                "JHAT: applying hst123 safety shim (shift-only WCS update) to avoid a known "
                "segfault in tweakwcs alignment on some HST inputs."
            )
        except Exception:
            pass

    jhat_assign_cleanup: str | None = None
    push_jhat_ee_calibration_dir(outdir)
    try:
        if gaia:
            from hst123 import settings as _hst123_settings_quality

            _q_retry = getattr(_hst123_settings_quality, "jhat_quality_retry_enabled", True)
            _max_rms = float(getattr(_hst123_settings_quality, "jhat_max_acceptable_rms_arcsec", 2.0))
            _min_n = int(getattr(_hst123_settings_quality, "jhat_quality_min_matches", 5))
            _max_q_attempts = int(getattr(_hst123_settings_quality, "jhat_quality_retry_max_attempts", 4))
            _eff_q_attempts = max(1, _max_q_attempts if _q_retry else 1)

            extra_work = dict(params_base)
            extra_work.pop("outrootdir", None)
            extra_work.pop("outsubdir", None)
            meta = _pop_jhat_gaia_sidecar_keys(extra_work)
            refcat_name, pmflag = _jhat_gaia_resolve_refcat_name_pmflag(align_image, outdir, meta)
            if os.path.isfile(str(refcat_name)):
                extra_work.setdefault("refcat_racol", "ra")
                extra_work.setdefault("refcat_deccol", "dec")
                extra_work.setdefault("refcat_magcol", "mag")
                extra_work.setdefault("refcat_magerrcol", "dmag")
            if not extra_work.get("telescope"):
                extra_work["telescope"] = _infer_jhat_telescope(align_image)

            _mag_lim = getattr(_hst123_settings_quality, "jhat_gaia_anchor_mag_lim", (15.0, 22.0))
            if _mag_lim is not None:
                if isinstance(_mag_lim, (list, tuple)) and len(_mag_lim) == 2:
                    _m0, _m1 = _mag_lim[0], _mag_lim[1]
                    if _m0 is not None and _m1 is not None:
                        extra_work["refmag_lim"] = (float(_m0), float(_m1))
                        extra_work["objmag_lim"] = (float(_m0), float(_m1))
            extra_work["_hst123_gaia_anchor_match_only_retries"] = True

            phot_kw = _merged_gaia_run_all_kw(extra_work, 0)
            phot_kw.update(
                _loosest_run_phot_kw_for_gaia_retries(
                    extra_work, quality_retry=_q_retry, max_attempts=_eff_q_attempts
                )
            )
            photcat_loaded, align_image_jhat, jhat_assign_cleanup = _jhat_gaia_run_phot_once(
                wcs_align,
                align_image,
                outdir,
                verbose=verbose,
                xshift=xshift,
                yshift=yshift,
                merged_loose=phot_kw,
            )
            log.info(
                "JHAT Gaia: photometry complete; quality attempts use refcat match + shift only "
                "(no additional run_all / run_phot).",
            )

            refcat_mem = False
            refn, pmf = refcat_name, pmflag
            _last_stats = None
            for _q_attempt in range(_eff_q_attempts):
                _saveph, refcat_mem, refn, pmf = _jhat_gaia_match_align_one_attempt(
                    wcs_align,
                    align_image_jhat,
                    align_image,
                    outdir,
                    verbose=verbose,
                    attempt_index=_q_attempt,
                    params_base=extra_work,
                    refcatname0=refn,
                    pmflag0=pmf,
                    photcat_loaded=photcat_loaded,
                    refcat_in_memory=refcat_mem,
                )
                if _saveph:
                    _last_stats = read_jhat_gaia_residual_stats(
                        align_image,
                        outdir,
                        sep_max_arcsec=quality_sep_cap_arcsec,
                    )
                else:
                    _last_stats = None
                if not _q_retry or not _saveph:
                    return _last_stats
                if jhat_alignment_acceptable(
                    _last_stats,
                    max_rms_arcsec=_max_rms,
                    min_matches=_min_n,
                ):
                    if _q_attempt > 0:
                        log.info(
                            "JHAT Gaia: residuals acceptable after quality retry %d",
                            _q_attempt,
                        )
                    return _last_stats
                if _q_attempt + 1 >= _eff_q_attempts:
                    log.error(
                        "JHAT Gaia: RMS vs reference still exceeds %.3f″ or n_match<%d after %d attempt(s): "
                        "rms_sky≈%s″ n=%s",
                        _max_rms,
                        _min_n,
                        _eff_q_attempts,
                        None if _last_stats is None else _last_stats.get("rms_sky_as"),
                        None if _last_stats is None else _last_stats.get("n_match"),
                    )
                    return _last_stats
                log.warning(
                    "JHAT Gaia: poor residuals (rms_sky≈%s″ n=%s); retry %d/%d with broader matching",
                    None if _last_stats is None else _last_stats.get("rms_sky_as"),
                    None if _last_stats is None else _last_stats.get("n_match"),
                    _q_attempt + 2,
                    _eff_q_attempts,
                )
        else:
            if photfilename is None:
                raise ValueError("Input photometric catalog is required when gaia=False")
            from hst123 import settings as _hst123_settings_quality

            _q_retry = getattr(_hst123_settings_quality, "jhat_quality_retry_enabled", True)
            _max_rms = float(getattr(_hst123_settings_quality, "jhat_max_acceptable_rms_arcsec", 2.0))
            _min_n = int(getattr(_hst123_settings_quality, "jhat_quality_min_matches", 5))
            _max_q_attempts = int(getattr(_hst123_settings_quality, "jhat_quality_retry_max_attempts", 4))
            _eff_q_attempts = max(1, _max_q_attempts if _q_retry else 1)

            for _q_attempt in range(_eff_q_attempts):
                _extra_try = dict(params_base)
                _extra_try.update(jhat_quality_retry_overlay(_q_attempt))
                _saveph = _jhat_relative_run_one_attempt(
                    wcs_align,
                    align_image,
                    outdir,
                    photfilename,
                    verbose,
                    xshift,
                    yshift,
                    Nbright,
                    _extra_try,
                )
                if not _q_retry or not _saveph:
                    return None
                _last_stats = read_jhat_gaia_residual_stats(
                    align_image,
                    outdir,
                    sep_max_arcsec=quality_sep_cap_arcsec,
                )
                if jhat_alignment_acceptable(
                    _last_stats,
                    max_rms_arcsec=_max_rms,
                    min_matches=_min_n,
                ):
                    if _q_attempt > 0:
                        log.info(
                            "JHAT relative: residuals acceptable after quality retry %d",
                            _q_attempt,
                        )
                    return None
                if _q_attempt + 1 >= _eff_q_attempts:
                    log.error(
                        "JHAT relative: RMS vs reference still exceeds %.3f″ or n_match<%d after %d attempt(s): "
                        "rms_sky≈%s″ n=%s",
                        _max_rms,
                        _min_n,
                        _eff_q_attempts,
                        None if _last_stats is None else _last_stats.get("rms_sky_as"),
                        None if _last_stats is None else _last_stats.get("n_match"),
                    )
                    return None
                log.warning(
                    "JHAT relative: poor residuals (rms_sky≈%s″ n=%s); retry %d/%d with broader matching",
                    None if _last_stats is None else _last_stats.get("rms_sky_as"),
                    None if _last_stats is None else _last_stats.get("n_match"),
                    _q_attempt + 2,
                    _eff_q_attempts,
                )
            return None
    finally:
        pop_jhat_ee_calibration_dir()
        if jhat_assign_cleanup:
            try:
                from jhat.pdastro import rmfile

                rmfile(jhat_assign_cleanup)
            except Exception:
                try:
                    os.remove(jhat_assign_cleanup)
                except OSError:
                    pass
        if tmp_to_cleanup:
            try:
                os.remove(tmp_to_cleanup)
            except Exception:
                pass
