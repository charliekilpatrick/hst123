"""WCS header helpers: meta headers, SIP/CTYPE consistency for stwcs/updatewcs."""

from __future__ import annotations

import logging
import os

from astropy.io import fits

_LOG = logging.getLogger(__name__)


def _header_has_sip_polynomials(header) -> bool:
    """True if FITS header carries SIP distortion orders (HST-style A_ORDER / B_ORDER)."""
    try:
        ao = int(header.get("A_ORDER", 0) or 0)
        bo = int(header.get("B_ORDER", 0) or 0)
    except (TypeError, ValueError):
        return False
    return ao > 0 or bo > 0


def _tan_ctypes_need_sip_suffix(header) -> bool:
    if not _header_has_sip_polynomials(header):
        return False
    ct1 = str(header.get("CTYPE1", "") or "").strip().upper()
    ct2 = str(header.get("CTYPE2", "") or "").strip().upper()
    if not ct1.startswith("RA---TAN") or not ct2.startswith("DEC--TAN"):
        return False
    # Require both CTYPEs to carry -SIP when SIP polynomials are present
    return not (ct1.endswith("-SIP") and ct2.endswith("-SIP"))


def fix_sip_ctype_headers_fits(
    image_path: str | os.PathLike[str],
    *,
    logger: logging.Logger | None = None,
) -> int:
    """
    If SIP polynomials are present but CTYPE* lack the ``-SIP`` suffix, fix in place.

    MAST ACS/WFC3 FLT/FLC files often trigger multi-line astropy ``INFO`` spam and
    inconsistent-WCS behavior when CTYPE is ``RA---TAN`` while SIP keys exist.
    """
    log = logger or _LOG
    path = os.fspath(image_path)
    n_fix = 0
    with fits.open(path, mode="update") as hdul:
        for i, hdu in enumerate(hdul):
            try:
                if int(hdu.header.get("NAXIS", 0) or 0) < 2:
                    continue
            except (TypeError, ValueError):
                continue
            if not _tan_ctypes_need_sip_suffix(hdu.header):
                continue
            hdu.header["CTYPE1"] = "RA---TAN-SIP"
            hdu.header["CTYPE2"] = "DEC--TAN-SIP"
            n_fix += 1
            log.debug("SIP CTYPE fix %s HDU %s", os.path.basename(path), i)
        if n_fix:
            hdul.flush()
    return n_fix


def make_meta_wcs_header(header):
    """Return a minimal WCS header dict; CTYPE* SIP variants normalized to TAN."""
    meta_header = {}
    for key in [
        "NAXIS", "NAXIS1", "NAXIS2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
        "CTYPE1", "CTYPE2",
    ]:
        meta_header[key] = header[key]

    if meta_header["CTYPE1"] == "RA---TAN-SIP":
        meta_header["CTYPE1"] = "RA---TAN"
    if meta_header["CTYPE2"] == "DEC--TAN-SIP":
        meta_header["CTYPE2"] = "DEC--TAN"

    return meta_header
