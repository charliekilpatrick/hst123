"""Run JHAT to align HST/JWST images to Gaia or a user catalog. Requires optional `jhat` package."""

from __future__ import annotations

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
    Path to JHAT's per-image photometry output (``*.phot.txt``) written by run_all().

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


def read_jhat_gaia_residual_stats(align_image: str | os.PathLike[str], outdir: str | os.PathLike[str]):
    """
    RMS residuals of the JHAT solution vs Gaia (ICRS) from the final matched table.

    Returns
    -------
    dict or None
        ``n_match``, ``rms_ra_as``, ``rms_dec_as``, ``rms_sky_as`` (great-circle RMS
        of separations), ``rms_ra_deg``, ``rms_dec_deg`` for FITS ``CRDER*`` (deg),
        and ``rms_sky_deg``.
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

    try:
        img = SkyCoord(tab[ra_img], tab[dec_img], unit=u.deg, frame="icrs")
        ref = SkyCoord(tab[ref_ra], tab[ref_dec], unit=u.deg, frame="icrs")
    except Exception as exc:
        log.warning("JHAT phot table %s: invalid coordinates: %s", path, exc)
        return None

    sep_as = img.separation(ref).to(u.arcsec).value
    n = int(np.size(sep_as))
    if n < 1:
        return None

    dec_rad = np.radians(np.asarray(tab[dec_img], dtype=float))
    dra_deg = np.asarray(tab[ra_img], dtype=float) - np.asarray(tab[ref_ra], dtype=float)
    ddec_deg = np.asarray(tab[dec_img], dtype=float) - np.asarray(tab[ref_dec], dtype=float)
    dra_as = dra_deg * np.cos(dec_rad) * 3600.0
    ddec_as = ddec_deg * 3600.0

    if n < 2:
        log.debug("JHAT Gaia phot table %s: need ≥2 matches for RMS dispersion", path)
        return None

    rms_ra_as = float(np.std(dra_as, ddof=1))
    rms_dec_as = float(np.std(ddec_as, ddof=1))
    rms_sky_as = float(np.sqrt(np.mean(sep_as**2)))

    return {
        "n_match": n,
        "rms_ra_as": rms_ra_as,
        "rms_dec_as": rms_dec_as,
        "rms_sky_as": rms_sky_as,
        "rms_ra_deg": rms_ra_as / 3600.0,
        "rms_dec_deg": rms_dec_as / 3600.0,
        "rms_sky_deg": rms_sky_as / 3600.0,
        "phot_path": path,
    }


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
        Passed as keyword arguments to ``st_wcs_align().run_all()``.

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
    extra = dict(params or {})

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

    # JHAT currently calls pandas.read_table(..., delim_whitespace=...) via pdastro.
    # pandas 2.2+ removed/changed this kw, raising TypeError. Patch in a small
    # compatibility shim so JHAT can read whitespace-delimited catalogs.
    try:
        import inspect
        import pandas as pd  # type: ignore

        sig = inspect.signature(pd.read_table)
        if "delim_whitespace" not in sig.parameters and not hasattr(pd, "_hst123_read_table_compat"):
            _orig_read_table = pd.read_table

            def _read_table_compat(*args, delim_whitespace=None, **kwargs):
                kwargs.pop("delim_whitespace", None)
                if delim_whitespace:
                    # Match old behavior: any whitespace is a delimiter.
                    kwargs.setdefault("sep", r"\s+")
                    return pd.read_csv(*args, **kwargs)
                return _orig_read_table(*args, **kwargs)

            pd.read_table = _read_table_compat  # type: ignore[assignment]
            pd._hst123_read_table_compat = True  # type: ignore[attr-defined]
    except Exception:
        pass

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
                # Prefer SCI,1 WCS when present; else fall back to primary.
                if "SCI" in hdul:
                    hdu = hdul["SCI", 1] if ("SCI", 1) in hdul else hdul["SCI"]
                else:
                    hdu = hdul[0]
                hdr = hdu.header
                w = WCS(hdr, hdul)

                t = phot.t.loc[ixs, [xcol, ycol, refcat_racol, refcat_deccol]]
                x = np.asarray(t[xcol], dtype=float)
                y = np.asarray(t[ycol], dtype=float)
                ra_ref = np.asarray(t[refcat_racol], dtype=float)
                dec_ref = np.asarray(t[refcat_deccol], dtype=float)

                pred = w.pixel_to_world(x, y)
                ra_pred = np.asarray(pred.ra.deg, dtype=float)
                dec_pred = np.asarray(pred.dec.deg, dtype=float)

                # Small-angle offsets in tangent plane (arcsec), then convert to CRVAL deltas.
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
                dec_rad = np.deg2rad(dec_ref[good])
                dra_cos_deg = dra_deg * np.cos(dec_rad)
                ddec_deg = (dec_ref[good] - dec_pred[good])

                dra_cos_med = float(np.nanmedian(dra_cos_deg))
                ddec_med = float(np.nanmedian(ddec_deg))
                if not (np.isfinite(dra_cos_med) and np.isfinite(ddec_med)):
                    raise RuntimeError("JHAT safe align: computed non-finite median WCS shift.")

                crval1 = float(hdr.get("CRVAL1", w.wcs.crval[0]))
                crval2 = float(hdr.get("CRVAL2", w.wcs.crval[1]))
                if not np.isfinite(crval1) or not np.isfinite(crval2):
                    crval1, crval2 = map(float, w.wcs.crval)

                cos_crval2 = float(np.cos(np.deg2rad(crval2)))
                if (not np.isfinite(cos_crval2)) or cos_crval2 == 0.0:
                    cos_crval2 = 1.0

                hdr["CRVAL1"] = (crval1 + dra_cos_med / cos_crval2, "Updated by hst123 (JHAT safe shift-only)")
                hdr["CRVAL2"] = (crval2 + ddec_med, "Updated by hst123 (JHAT safe shift-only)")
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

    push_jhat_ee_calibration_dir(outdir)
    try:
        if gaia:
            # Defaults from JHAT HST "Align to Gaia" example; user/settings may override via *params*.
            tel = extra.pop("telescope", None) or _infer_jhat_telescope(align_image)
            gaia_refcat_path = extra.pop("gaia_refcat_path", None)
            skip_gaia_prefetch = bool(extra.pop("skip_gaia_prefetch", False))
            gaia_prefetch_radius_raw = extra.pop("gaia_prefetch_radius", None)
            gaia_prefetch_center = extra.pop("gaia_prefetch_center", None)

            if gaia_refcat_path is None and not skip_gaia_prefetch:
                from hst123 import settings as _hst123_settings
                from hst123.utils.gaia_prefetch import (
                    gaia_prefetch_cache_path,
                    icrs_field_center_from_fits,
                    prefetch_gaia_catalog,
                )

                radius_q = getattr(
                    _hst123_settings, "jhat_gaia_prefetch_radius", 22 * u.arcmin
                )
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
                    gaia_refcat_path = cache_path
                else:
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
                    log.info(
                        "JHAT: Gaia prefetch complete (%s, n_row=%d).",
                        res.source,
                        res.n_rows,
                    )
                    gaia_refcat_path = res.path

            gaia_kw: dict = {
                "telescope": tel,
                "overwrite": True,
                "d2d_max": 0.5,
                "showplots": 0,
                "histocut_order": "dxdy",
                "sharpness_lim": (0.3, 0.9),
                "roundness1_lim": (-0.7, 0.7),
                "SNR_min": 3,
                "dmag_max": 1.0,
                "objmag_lim": (14, 24),
            }
            gaia_kw.update(extra)
            gaia_kw.pop("outrootdir", None)
            gaia_kw.pop("outsubdir", None)
            # Final matched table (needed for CRDER* on reference drizzle); user may set 0 in jhat_params.
            savephottable = int(gaia_kw.pop("savephottable", 1))
            # If using a custom (prefetched) catalog file, JHAT needs explicit column names.
            if gaia_refcat_path:
                gaia_kw.setdefault("refcat_racol", "ra")
                gaia_kw.setdefault("refcat_deccol", "dec")
                gaia_kw.setdefault("refcat_magcol", "mag")
                gaia_kw.setdefault("refcat_magerrcol", "dmag")
            def _run_all_with_gaia() -> None:
                wcs_align.run_all(
                    align_image,
                    outrootdir=outdir,
                    outsubdir=None,
                    refcatname=gaia_refcat_path or "Gaia",
                    pmflag=False if gaia_refcat_path else True,
                    use_dq=False,
                    verbose=verbose,
                    xshift=xshift,
                    yshift=yshift,
                    savephottable=savephottable,
                    **gaia_kw,
                )

            # If we were given a pre-fetched catalog file, use it directly without
            # exercising TAP retries/fallback logic here.
            if gaia_refcat_path:
                _run_all_with_gaia()
            else:
                # Gaia TAP can intermittently return HTTP 500 for async-job result retrieval.
                # Retry a few times; if it still fails, fall back to a cached local cone-search
                # catalog so alignment can proceed.
                last_exc: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        _run_all_with_gaia()
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        msg = str(exc)
                        # Only retry for known network/service failures.
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

                if last_exc is not None:
                    # Fallback: use a cached local catalog (from Vizier) instead of TAP "Gaia".
                    try:
                        from astroquery.vizier import Vizier  # type: ignore

                        from hst123 import settings as _hst123_settings
                        from hst123.utils.gaia_prefetch import icrs_field_center_from_fits

                        _cen = icrs_field_center_from_fits(align_image)
                        ra0 = float(_cen.ra.deg)
                        dec0 = float(_cen.dec.deg)
                        radius_deg = float(
                            getattr(
                                _hst123_settings, "jhat_gaia_prefetch_radius", 22 * u.arcmin
                            )
                            .to(u.deg)
                            .value
                        )
                        cache_path = os.path.join(
                            outdir,
                            f"gaia_vizier_cache_{ra0:.5f}_{dec0:.5f}_{radius_deg:.5f}.txt",
                        )
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
                            # Write a minimal JHAT-compatible refcat: ra dec mag dmag
                            out = Table()
                            out["ra"] = np.asarray(t["RA_ICRS"], dtype=float)
                            out["dec"] = np.asarray(t["DE_ICRS"], dtype=float)
                            out["mag"] = np.asarray(t["Gmag"], dtype=float)
                            if "e_Gmag" in t.colnames:
                                out["dmag"] = np.asarray(t["e_Gmag"], dtype=float)
                            else:
                                out["dmag"] = np.full(len(out), 0.02, dtype=float)
                            out.write(cache_path, format="ascii.basic", overwrite=True)

                        log.warning(
                            "JHAT Gaia TAP failed; falling back to cached Vizier Gaia DR3 catalog: %s",
                            os.path.basename(cache_path),
                        )
                        # Re-run as a custom-catalog alignment (no proper motion correction).
                        rel_kw = dict(gaia_kw)
                        rel_kw.update(
                            {
                                "refcat_racol": "ra",
                                "refcat_deccol": "dec",
                                "refcat_magcol": "mag",
                                "refcat_magerrcol": "dmag",
                            }
                        )
                        wcs_align.run_all(
                            align_image,
                            outrootdir=outdir,
                            outsubdir=None,
                            refcatname=cache_path,
                            pmflag=False,
                            use_dq=False,
                            verbose=verbose,
                            xshift=xshift,
                            yshift=yshift,
                            savephottable=savephottable,
                            **rel_kw,
                        )
                    except Exception:
                        # If fallback also fails, raise the original Gaia exception.
                        raise last_exc
            if savephottable:
                return read_jhat_gaia_residual_stats(align_image, outdir)
            return None
        else:
            if photfilename is None:
                raise ValueError("Input photometric catalog is required when gaia=False")
        tel = extra.pop("telescope", None) or _infer_jhat_telescope(align_image)
        # JHAT's custom-catalog path sometimes leaves refcat.racol/refcat.deccol set to
        # the literal string "auto", and then treats it as a real column name,
        # failing even when columns "ra"/"dec" exist. For hst123's use (JHAT phot.txt
        # catalogs), the columns are always named "ra"/"dec".
        rel_kw = {
            "telescope": tel,
            "overwrite": True,
            "showplots": 0,
            "refcat_racol": "ra",
            "refcat_deccol": "dec",
            # JHAT requires a "mainfilter" magnitude column name for custom catalogs.
            # Our JHAT phot.txt tables always include "mag" and "dmag".
            "refcat_magcol": "mag",
            "refcat_magerrcol": "dmag",
            **extra,
        }
        rel_kw.pop("outrootdir", None)
        rel_kw.pop("outsubdir", None)
        # Save the matched photometry table by default so hst123 can diagnose/QA
        # relative alignment quality (users may override via jhat_params).
        savephottable = int(rel_kw.pop("savephottable", 1))
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
            Nbright=Nbright,
            **rel_kw,
        )
        return None
    finally:
        pop_jhat_ee_calibration_dir()
        if tmp_to_cleanup:
            try:
                os.remove(tmp_to_cleanup)
            except Exception:
                pass
