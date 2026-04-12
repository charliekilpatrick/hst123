"""FITS metadata and image discovery (zpt, chip, filter, instrument, input/split images, DQ)."""
import glob
import os

import numpy as np
from astropy.io import fits

from hst123.primitives.base import BasePrimitive
from hst123.utils.paths import normalize_fits_path, pipeline_workspace_dir
from hst123.utils.logging import log_calls


def _instrument_from_header(header):
    """
    Derive instrument/detector/subarray string from primary FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Primary (or science) FITS header with INSTRUME and DETECTOR.

    Returns
    -------
    str
        String like ``acs_wfc_full`` or ``wfpc2_wfpc2_full``.
    """
    inst = header["INSTRUME"].lower()
    if inst.upper() == "WFPC2":
        return f"{inst}_wfpc2_full"
    det = header["DETECTOR"].lower()
    sub = "sub" if str(header.get("SUBARRAY", False)) in ("T", "True") else "full"
    return f"{inst}_{det}_{sub}"


def _phot_zero_point_ab(photflam, photplam):
    """
    Compute AB magnitude zero point from PHOTFLAM and PHOTPLAM.

    Parameters
    ----------
    photflam : float
        Flux of a 0-mag source in erg/s/cm^2/A (PHOTFLAM).
    photplam : float
        Pivot wavelength in Angstroms (PHOTPLAM).

    Returns
    -------
    float
        AB zero point: ZP_AB = -2.5*log10(PHOTFLAM) - 5*log10(PHOTPLAM) - 2.408.
    """
    return -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408


class FitsHelper(BasePrimitive):
    """
    FITS metadata and image discovery for the hst123 pipeline.

    Provides get_zpt, get_chip, get_filter, get_instrument, get_input_images,
    get_split_images, and get_dq_image. Used by the main pipeline for
    zero points, chip/filter/instrument lookup, and file discovery.
    """

    def get_zpt(self, image, ccdchip=1, zptype="abmag"):
        """
        Get photometric zero point for an image (AB or STmag).

        Parameters
        ----------
        image : str
            Path to FITS file.
        ccdchip : int, optional
            Chip number for multi-chip instruments. Default 1.
        zptype : str, optional
            Zero point type: "abmag" or "stmag". Default "abmag".

        Returns
        -------
        float or None
            Zero point in magnitudes, or None if not found.
        """
        hdu = fits.open(image, mode="readonly")
        inst = self.get_instrument(image).lower()
        use_hdu = None
        zpt = None
        sci = []
        for i, h in enumerate(hdu):
            keys = list(h.header.keys())
            if "PHOTPLAM" in keys and "PHOTFLAM" in keys:
                sci.append(h)
        if len(sci) == 1:
            use_hdu = sci[0]
        elif len(sci) > 1:
            chips = []
            for h in sci:
                if "acs" in inst or "wfc3" in inst:
                    if "CCDCHIP" in h.header.keys() and h.header["CCDCHIP"] == ccdchip:
                        chips.append(h)
                else:
                    if "DETECTOR" in h.header.keys() and h.header["DETECTOR"] == ccdchip:
                        chips.append(h)
            if len(chips) > 0:
                use_hdu = chips[0]
        if use_hdu:
            photplam = float(use_hdu.header["PHOTPLAM"])
            photflam = float(use_hdu.header["PHOTFLAM"])
            if "ab" in zptype:
                zpt = _phot_zero_point_ab(photflam, photplam)
            elif "st" in zptype:
                zpt = -2.5 * np.log10(photflam) - 21.1
        return zpt

    def get_chip(self, image):
        """
        Get chip/detector identifier for an image.

        Parameters
        ----------
        image : str
            Path to FITS file.

        Returns
        -------
        int or str
            CCDCHIP or DETECTOR value; 1 if not found.
        """
        hdu = fits.open(image)
        chip = None
        for h in hdu:
            if "CCDCHIP" in h.header.keys():
                chip = h.header["CCDCHIP"] if chip is None else 1
            elif "DETECTOR" in h.header.keys():
                chip = h.header["DETECTOR"] if chip is None else 1
        return chip if chip is not None else 1

    def get_filter(self, image):
        """
        Get filter name from image header.

        Parameters
        ----------
        image : str
            Path to FITS file.

        Returns
        -------
        str
            Filter name (lowercase); uses FILTER, FILTER1/FILTER2, or FILTNAM1/FILTNAM2 for WFPC2.
        """
        if "wfpc2" in str(fits.getval(image, "INSTRUME")).lower():
            f = str(fits.getval(image, "FILTNAM1"))
            if len(f.strip()) == 0:
                f = str(fits.getval(image, "FILTNAM2"))
            if len(f.strip()) == 0:
                try:
                    f = str(fits.getval(image, "FILTER"))
                except Exception:
                    f = ""
        else:
            try:
                f = str(fits.getval(image, "FILTER"))
            except Exception:
                f = str(fits.getval(image, "FILTER1"))
                if "clear" in f.lower():
                    f = str(fits.getval(image, "FILTER2"))
        return f.lower().strip()

    def get_instrument(self, image):
        """
        Get instrument/detector/subarray string for an image.

        Parameters
        ----------
        image : str
            Path to FITS file.

        Returns
        -------
        str
            String like ``acs_wfc_full`` from primary header.
        """
        hdu = fits.open(image, mode="readonly")
        return _instrument_from_header(hdu[0].header)

    @log_calls
    def get_input_images(self, pattern=None, workdir=None):
        """
        Discover input science images (c1m, c0m, flc, flt) in workdir.

        Parameters
        ----------
        pattern : list of str, optional
            Glob patterns; default ``['*c1m.fits', '*c0m.fits', '*flc.fits', '*flt.fits']``.
        workdir : str, optional
            Directory to search; default ".".

        Returns
        -------
        list of str
            Paths to matching FITS files.
        """
        workdir = workdir or "."
        pattern = pattern or ["*c1m.fits", "*c0m.fits", "*flc.fits", "*flt.fits"]
        base = os.path.abspath(os.path.expanduser(workdir))
        ws = pipeline_workspace_dir(base)
        search_roots = []
        if ws and os.path.isdir(ws):
            search_roots.append(ws)
        search_roots.append(base)
        seen: set[str] = set()
        out: list[str] = []
        for root in search_roots:
            for p in pattern:
                for s in glob.glob(os.path.join(root, p)):
                    if not os.path.isfile(s):
                        continue
                    n = normalize_fits_path(s)
                    if n not in seen:
                        seen.add(n)
                        out.append(n)
        self._primitive_cleanup(
            "get_input_images",
            work_dir=workdir,
            validate_fits_paths=out,
        )
        return out

    def get_split_images(self, pattern=None, workdir=None):
        """
        Discover split (per-chip) images in workdir.

        Parameters
        ----------
        pattern : list of str, optional
            Glob patterns; default chip patterns for c0m/flc/flt.
        workdir : str, optional
            Directory to search; default ".".

        Returns
        -------
        list of str
            Paths to matching split FITS files.
        """
        workdir = workdir or "."
        pattern = pattern or ["*c0m.chip?.fits", "*flc.chip?.fits", "*flt.chip?.fits"]
        base = os.path.abspath(os.path.expanduser(workdir))
        out = [
            normalize_fits_path(s)
            for p in pattern
            for s in glob.glob(os.path.join(base, p))
            if os.path.isfile(s)
        ]
        self._primitive_cleanup(
            "get_split_images",
            work_dir=workdir,
            validate_fits_paths=out,
        )
        return out

    def get_dq_image(self, image):
        """
        Get path to DQ (data quality) image for dolphot masking.

        Parameters
        ----------
        image : str
            Path to science FITS file.

        Returns
        -------
        str
            Path to external DQ file (e.g. c1m for WFPC2). For ACS/WFC3 the DQ
            lives in the science MEF, so this returns ``""`` and *mask tools are
            invoked with the science file only.
        """
        if self.get_instrument(image).split("_")[0].upper() == "WFPC2":
            return image.replace("c0m.fits", "c1m.fits")
        return ""
