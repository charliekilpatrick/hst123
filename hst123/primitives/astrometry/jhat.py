"""Run JHAT to align HST/JWST images to Gaia or a user catalog. Requires optional `jhat` package."""

from __future__ import annotations

import logging
import os
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

log = logging.getLogger(__name__)


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

    wcs_align = st_wcs_align()
    align_image = os.path.abspath(os.path.expanduser(os.fspath(align_image)))
    # JHAT appends outsubdir to outrootdir (default '.'). Passing an absolute path
    # as outsubdir yields './/abs/...' and breaks photometry / shift file locations.
    outdir = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    extra = dict(params or {})

    if gaia:
        # Defaults from JHAT HST "Align to Gaia" example; user/settings may override via *params*.
        tel = extra.pop("telescope", None) or _infer_jhat_telescope(align_image)
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
        wcs_align.run_all(
            align_image,
            outrootdir=outdir,
            outsubdir=None,
            refcatname="Gaia",
            pmflag=True,
            use_dq=False,
            verbose=verbose,
            xshift=xshift,
            yshift=yshift,
            savephottable=savephottable,
            **gaia_kw,
        )
        if savephottable:
            return read_jhat_gaia_residual_stats(align_image, outdir)
        return None
    else:
        if photfilename is None:
            raise ValueError("Input photometric catalog is required when gaia=False")
        tel = extra.pop("telescope", None) or _infer_jhat_telescope(align_image)
        rel_kw = {"telescope": tel, "overwrite": True, "showplots": 0, **extra}
        rel_kw.pop("outrootdir", None)
        rel_kw.pop("outsubdir", None)
        wcs_align.run_all(
            align_image,
            outrootdir=outdir,
            outsubdir=None,
            refcatname=photfilename,
            use_dq=False,
            verbose=verbose,
            xshift=xshift,
            yshift=yshift,
            Nbright=Nbright,
            **rel_kw,
        )
        return None
