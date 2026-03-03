"""Minimal WCS header builder (strip SIP to avoid WCS errors)."""


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
