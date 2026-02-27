"""
FITS metadata and image discovery primitives.

FitsHelper is used by the main hst123 pipeline for get_zpt, get_chip,
get_filter, get_instrument, get_input_images, get_split_images, get_dq_image.
"""
import glob
import os

import numpy as np
from astropy.io import fits

from hst123.primitives.base import BasePrimitive


def _instrument_from_header(header):
    """Primitive: derive instrument/detector/subarray string from primary header."""
    inst = header["INSTRUME"].lower()
    if inst.upper() == "WFPC2":
        return f"{inst}_wfpc2_full"
    det = header["DETECTOR"].lower()
    sub = "sub" if str(header.get("SUBARRAY", False)) in ("T", "True") else "full"
    return f"{inst}_{det}_{sub}"


def _phot_zero_point_ab(photflam, photplam):
    """Primitive: AB magnitude zero point from PHOTFLAM and PHOTPLAM."""
    return -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408


class FitsHelper(BasePrimitive):
    """FITS metadata and image discovery (get_zpt, get_chip, get_filter, get_instrument, etc.)."""

    def __init__(self, pipeline):
        super().__init__(pipeline)

    def get_zpt(self, image, ccdchip=1, zptype="abmag"):
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
        hdu = fits.open(image)
        chip = None
        for h in hdu:
            if "CCDCHIP" in h.header.keys():
                chip = h.header["CCDCHIP"] if chip is None else 1
            elif "DETECTOR" in h.header.keys():
                chip = h.header["DETECTOR"] if chip is None else 1
        return chip if chip is not None else 1

    def get_filter(self, image):
        if "wfpc2" in str(fits.getval(image, "INSTRUME")).lower():
            f = str(fits.getval(image, "FILTNAM1"))
            if len(f.strip()) == 0:
                f = str(fits.getval(image, "FILTNAM2"))
        else:
            try:
                f = str(fits.getval(image, "FILTER"))
            except Exception:
                f = str(fits.getval(image, "FILTER1"))
                if "clear" in f.lower():
                    f = str(fits.getval(image, "FILTER2"))
        return f.lower()

    def get_instrument(self, image):
        hdu = fits.open(image, mode="readonly")
        return _instrument_from_header(hdu[0].header)

    def get_input_images(self, pattern=None, workdir=None):
        workdir = workdir or "."
        pattern = pattern or ["*c1m.fits", "*c0m.fits", "*flc.fits", "*flt.fits"]
        return [s for p in pattern for s in glob.glob(os.path.join(workdir, p))]

    def get_split_images(self, pattern=None, workdir=None):
        workdir = workdir or "."
        pattern = pattern or ["*c0m.chip?.fits", "*flc.chip?.fits", "*flt.chip?.fits"]
        return [s for p in pattern for s in glob.glob(os.path.join(workdir, p))]

    def get_dq_image(self, image):
        if self.get_instrument(image).split("_")[0].upper() == "WFPC2":
            return image.replace("c0m.fits", "c1m.fits")
        return ""
