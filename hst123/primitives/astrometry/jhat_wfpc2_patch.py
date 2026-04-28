"""
Monkey-patch JHAT's ``hst_photclass.load_image`` so WFPC2 ``*_c0m.fits`` (and other
WFPC2 products) work with ``st_wcs_align``.

Upstream ``jhat.simple_jwst_phot.hst_photclass`` assumes ACS/WFC3-style headers:
``FILTER1`` is a string (``'CLEAR' not in ...`` breaks on numeric headers), and
filenames must match flt/flc/drz/drc only (``*_c0m.fits`` raises "Unknown image
type"). WFPC2 calibration uses ``FILTNAM1``/``FILTNAM2`` and ``*_c0m.fits``.

This module applies a one-time class-method replacement after ``import jhat``.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


def _ensure_jhat_scihdr_mjd_avg(primaryhdr: Any, scihdr: Any) -> None:
    """
    JHAT expects ``MJD-AVG`` in the SCI header when ``pmflag=True`` (Gaia proper motion).

    hst123 writes ``MJD-AVG`` to PRIMARY; many calibrated HST products do not copy it
    into SCI. Inject it into the in-memory SCI header as a compatibility shim.
    """
    try:
        if scihdr is None:
            return
        if "MJD-AVG" in scihdr:
            return
    except Exception:
        return
    mjd = None
    try:
        from hst123.utils.mjd_header import mjd_avg_from_primary_header

        mjd = mjd_avg_from_primary_header(primaryhdr)
    except Exception:
        mjd = None
    if mjd is None:
        try:
            mjd = float(primaryhdr.get("MJD-AVG"))
        except Exception:
            mjd = None
    if mjd is None:
        return
    try:
        scihdr["MJD-AVG"] = (float(mjd), "Representative MJD (mid-exposure, hst123)")
    except Exception:
        return


def wfpc2_filter_key_and_name(primaryhdr: Any) -> tuple[str, str]:
    """
    Return (header_key, filter_string) for WFPC2 primary headers.

    Avoids ``'CLEAR' in FILTER1`` when ``FILTER1`` is not a string (JHAT bug).
    """
    for key in ("FILTNAM1", "FILTNAM2", "FILTER"):
        if key not in primaryhdr:
            continue
        raw = primaryhdr[key]
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.upper() in ("N/A", "NONE", ""):
            continue
        if s.upper() == "CLEAR" and key.startswith("FILTNAM"):
            continue
        return key, s
    return "FILTNAM1", "CLEAR"


def _wfpc2_hst_photclass_load_image(
    self,
    imagename: str,
    imagetype: str | None = None,
    DNunits: bool = False,
    use_dq: bool = False,
    skip_preparing: bool = False,
) -> None:
    """Replacement for ``hst_photclass.load_image`` when ``INSTRUME`` is WFPC2."""
    from astropy.io import fits as fits_mod
    from astropy import wcs as wcs_mod

    self.imagename = imagename
    self.im = fits_mod.open(imagename)
    self.primaryhdr = self.im["PRIMARY"].header
    try:
        self.scihdr = self.im["SCI"].header
    except KeyError as exc:
        self.im.close()
        raise RuntimeError(
            f"JHAT WFPC2 patch: no SCI extension in {imagename!r} "
            "(expected MEF with at least one SCI HDU)."
        ) from exc

    self.NAXIS1 = self.scihdr["NAXIS1"]
    self.NAXIS2 = self.scihdr["NAXIS2"]
    self.instrument = str(self.primaryhdr.get("INSTRUME", "WFPC2")).strip()
    _ensure_jhat_scihdr_mjd_avg(self.primaryhdr, self.scihdr)

    fk, fn = wfpc2_filter_key_and_name(self.primaryhdr)
    self.filterkey = fk
    self.filtername = fn

    ap_raw = self.primaryhdr.get("APERTURE", "LARGE")
    ap = str(ap_raw).replace("-", "")
    if "ACS" in self.instrument.upper():
        self.aperture = "J" + ap
    else:
        self.aperture = "I" + ap

    psf_scalar = self.psf_fwhm
    self.filters = {self.instrument: [self.filtername]}
    self.psf_fwhm = {self.instrument: [psf_scalar]}
    self.dict_utils = {}
    for instrument in self.filters:
        self.dict_utils[instrument.upper()] = {
            self.filters[instrument.upper()][i]: {"psf fwhm": self.psf_fwhm[instrument.upper()][i]}
            for i in range(len(self.filters[instrument]))
        }

    self.sci_wcs = wcs_mod.WCS(self.scihdr, self.im)
    try:
        self.err = self.im["ERR"].data
    except Exception:
        self.err = None
    self.pixel_scale = (
        wcs_mod.utils.proj_plane_pixel_scales(self.sci_wcs)[0]
        * self.sci_wcs.wcs.cunit[0].to("arcsec")
    )

    if self.verbose:
        log.info(
            "JHAT WFPC2 patch: instrument=%s filter=%s aperture=%s",
            self.instrument,
            self.filtername,
            self.aperture,
        )

    if imagetype is None:
        if re.search(
            r"flt\.fits$|flc\.fits$|tweakregstep\.fits$|assignwcsstep\.fits$",
            imagename,
            re.I,
        ):
            self.imagetype = "flc"
        elif re.search(r"drz\.fits$|drc\.fits$", imagename, re.I):
            self.imagetype = "drz"
        elif re.search(r"c0m\.fits$", imagename, re.I):
            self.imagetype = "wfpc2_c0m"
        else:
            self.im.close()
            raise RuntimeError(
                f"JHAT WFPC2 patch: unknown image type for file {imagename!r}"
            )
        # Skip ACS/WFC3 PAM path and internal AstroDrizzle shortcut: use direct SCI
        # data like the drizzle-product branch in upstream JHAT.
        self.pipeline_level = 3
        self.do_driz = False
    else:
        self.imagetype = imagetype
        self.pipeline_level = 3
        self.do_driz = False

    if not skip_preparing:
        # pipeline_level=3 / do_driz=False => prepare_image uses raw SCI counts
        # (no PAM path; avoids drizzlepac on WFPC2 *_c0m.fits).
        (self.data, self.mask) = self.prepare_image(
            self.im["SCI"].data,
            self.im["SCI"].header,
            self.do_driz,
        )


def ensure_jhat_hst_phot_wfpc2_patch() -> None:
    """
    Replace ``jhat.simple_jwst_phot.hst_photclass.load_image`` with a WFPC2-safe
    wrapper (idempotent; safe if ``jhat`` is re-imported).
    """
    import jhat.simple_jwst_phot as sjp

    cur = sjp.hst_photclass.load_image
    if getattr(cur, "__name__", "") == "load_image_hst123_wfpc2":
        # Still allow safe patches below to be applied idempotently.
        pass

    _orig_load_image = cur

    # Patch JHAT's HST EE correction helper to avoid SciPy failures when the
    # wavelength grid is not pre-sorted in upstream tables (RectBivariateSpline
    # requires strictly increasing x/y).
    _orig_hst_get_ee_corr = getattr(sjp, "hst_get_ee_corr", None)

    def _hst_get_ee_corr_sorted(ap, pxscale, filt, inst):
        try:
            import numpy as np
            import scipy
            import os
            import urllib
            from astropy.table import Table

            # Re-implement upstream logic, but sort waves/apps before spline.
            if str(inst).lower() == "ir":
                if not os.path.exists("ir_ee_corrections.csv"):
                    urllib.request.urlretrieve(
                        "https://www.stsci.edu/files/live/sites/www/files/home/hst/"
                        "instrumentation/wfc3/data-analysis/photometric-calibration/"
                        "ir-encircled-energy/_documents/ir_ee_corrections.csv",
                        "ir_ee_corrections.csv",
                    )
                ee = Table.read("ir_ee_corrections.csv", format="ascii")
                ee.rename_column("PIVOT", "WAVELENGTH")
            else:
                if not os.path.exists("wfc3uvis2_aper_007_syn.csv"):
                    urllib.request.urlretrieve(
                        "https://www.stsci.edu/files/live/sites/www/files/home/hst/"
                        "instrumentation/wfc3/data-analysis/photometric-calibration/"
                        "uvis-encircled-energy/_documents/wfc3uvis2_aper_007_syn.csv",
                        "wfc3uvis2_aper_007_syn.csv",
                    )
                ee = Table.read("wfc3uvis2_aper_007_syn.csv", format="ascii")
                if str(filt).upper() not in [str(x).upper() for x in ee["FILTER"]]:
                    if not os.path.exists("bohlin2016_wfc_ee-1.txt"):
                        urllib.request.urlretrieve(
                            "https://www.stsci.edu/files/live/sites/www/files/home/hst/"
                            "instrumentation/acs/data-analysis/aperture-corrections/_documents/"
                            "bohlin2016_wfc_ee-1.txt",
                            "bohlin2016_wfc_ee-1.txt",
                        )
                    ee = Table.read("bohlin2016_wfc_ee-1.txt", format="ascii", data_start=1)
                    ee.rename_column("col1", "FILTER")
                    ee["WAVELENGTH"] = [
                        float(x[1:-1]) * 10 if len(x) == 5 else float(x[1:-2]) * 10
                        for x in ee["FILTER"]
                    ]
                    px_cols = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 40.0]
                    n = 0
                    for col in list(ee.colnames):
                        if col in ["FILTER", "WAVELENGTH"]:
                            continue
                        ee.rename_column(col, "#" + str(pxscale * px_cols[n]))
                        n += 1

            filts = np.asarray(ee["FILTER"])
            ee.remove_column("FILTER")
            waves = np.asarray(ee["WAVELENGTH"], dtype=float)
            ee.remove_column("WAVELENGTH")

            # Build grid arrays
            colnames = list(ee.colnames)
            apps = np.asarray([float(x.split("#")[1]) for x in colnames], dtype=float)
            ee_arr = np.asarray([np.asarray(ee[col], dtype=float) for col in colnames], dtype=float)  # (napps, nwaves)

            # Sort apps and waves to satisfy RectBivariateSpline contract.
            aidx = np.argsort(apps)
            widx = np.argsort(waves)
            apps_s = apps[aidx]
            waves_s = waves[widx]
            ee_arr_s = ee_arr[aidx][:, widx]
            filts_s = filts[widx]

            # Guard: strictly increasing required.
            if np.any(np.diff(waves_s) <= 0) or np.any(np.diff(apps_s) <= 0):
                return np.asarray([1.0], dtype=float)

            interp = scipy.interpolate.RectBivariateSpline(waves_s, apps_s, ee_arr_s.T)
            # Choose the row corresponding to this filter (exact match).
            m = np.where(np.char.upper(filts_s.astype(str)) == str(filt).upper())[0]
            if m.size == 0:
                return np.asarray([1.0], dtype=float)
            filt_wave = float(waves_s[m[0]])
            return interp(filt_wave, float(ap) * float(pxscale)).flatten()
        except Exception:
            # Fall back to upstream if available; otherwise no correction.
            try:
                if _orig_hst_get_ee_corr is not None:
                    return _orig_hst_get_ee_corr(ap, pxscale, filt, inst)
            except Exception:
                pass
            import numpy as np

            return np.asarray([1.0], dtype=float)

    # Apply the EE correction patch once (idempotent).
    try:
        if getattr(sjp.hst_get_ee_corr, "__name__", "") != "_hst_get_ee_corr_sorted":
            _hst_get_ee_corr_sorted.__name__ = "_hst_get_ee_corr_sorted"
            sjp.hst_get_ee_corr = _hst_get_ee_corr_sorted  # type: ignore[attr-defined]
    except Exception:
        pass

    # Patch refcat matching to be more tolerant of poor initial WCS on some HST inputs
    # (e.g. WFPC2). Upstream drops all catalog sources if their WCS-projected x/y fall
    # outside the image bounds; that prevents any matching from happening at all. If
    # that occurs, retry with an expanded "in-bounds" window and a larger match radius.
    _orig_match_refcat = getattr(sjp.hst_photclass, "match_refcat", None)

    def _match_refcat_hst123(self, *args, **kwargs):
        out = None
        if callable(_orig_match_refcat):
            out = _orig_match_refcat(self, *args, **kwargs)
        try:
            tel = str(getattr(self, "primaryhdr", {}).get("TELESCOP", "")).strip().upper()
        except Exception:
            tel = ""
        if tel != "HST":
            return out
        if out not in (0, None):
            return out
        try:
            kw = dict(kwargs)
            kw["borderpadding"] = -10000
            kw.setdefault("max_sep", 5.0)
            return _orig_match_refcat(self, *args, **kw)
        except Exception:
            return out

    try:
        if callable(_orig_match_refcat) and getattr(sjp.hst_photclass.match_refcat, "__name__", "") != "_match_refcat_hst123":
            _match_refcat_hst123.__name__ = "_match_refcat_hst123"
            sjp.hst_photclass.match_refcat = _match_refcat_hst123  # type: ignore[assignment]
    except Exception:
        pass

    def load_image_hst123_wfpc2(
        self,
        imagename: str,
        imagetype: str | None = None,
        DNunits: bool = False,
        use_dq: bool = False,
        skip_preparing: bool = False,
    ) -> None:
        from astropy.io import fits as fits_mod

        try:
            ph = fits_mod.getheader(imagename, 0)
        except Exception:
            return _orig_load_image(
                self,
                imagename,
                imagetype,
                DNunits,
                use_dq,
                skip_preparing,
            )
        inst = str(ph.get("INSTRUME", "")).strip().upper()
        if inst != "WFPC2":
            # Upstream JHAT sets do_driz=True for multi-chip HST FLC/FLT and then
            # runs an internal AstroDrizzle to make a single-plane drizzled image
            # for fitting. That path has been observed to segfault later inside
            # tweakwcs alignment in our environment. To keep the pipeline stable,
            # we prevent internal drizzling by calling the original loader with
            # skip_preparing=True, then prepare the SCI data ourselves with
            # do_driz forced off.
            out = _orig_load_image(
                self,
                imagename,
                imagetype,
                DNunits,
                use_dq,
                True,  # skip_preparing=True (we will prepare below)
            )
            _ensure_jhat_scihdr_mjd_avg(
                getattr(self, "primaryhdr", ph), getattr(self, "scihdr", None)
            )

            tel = str(getattr(self, "primaryhdr", ph).get("TELESCOP", "")).strip().upper()
            if tel == "HST" and hasattr(self, "do_driz"):
                self.do_driz = False

            if not skip_preparing:
                # Mirror upstream behavior for pipeline_level==2 and do_driz==False.
                # This keeps photometry/matching working while avoiding internal AstroDrizzle.
                from stsci.skypac import pamutils  # type: ignore

                dq = None
                if use_dq:
                    try:
                        dq = self.im["DQ"].data  # type: ignore[attr-defined]
                    except Exception:
                        dq = None
                area = None
                try:
                    area = pamutils.pam_from_file(
                        self.imagename, ("sci", 1), self.imagename + "_pam.fits"
                    )
                except Exception:
                    area = None

                data_original = self.im["SCI"].data  # type: ignore[attr-defined]
                imhdr = self.im["SCI"].header  # type: ignore[attr-defined]
                (self.data, self.mask) = self.prepare_image(  # type: ignore[attr-defined]
                    data_original,
                    imhdr,
                    area=area,
                    dq=dq,
                )
            return out
        return _wfpc2_hst_photclass_load_image(
            self,
            imagename,
            imagetype,
            DNunits,
            use_dq,
            skip_preparing,
        )

    load_image_hst123_wfpc2.__name__ = "load_image_hst123_wfpc2"
    sjp.hst_photclass.load_image = load_image_hst123_wfpc2
    log.debug("Applied JHAT hst_photclass.load_image WFPC2 monkey-patch")
