"""JHAT ΔCRVAL propagation from *_jhat.fits onto drizzle inputs (all SCI HDUs)."""

import numpy as np
from astropy.io import fits

from hst123.primitives.astrometry import jhat as jhat_mod


def _minimal_sci_wcs_header(extver: int) -> fits.Header:
    h = fits.Header()
    h["EXTNAME"] = "SCI"
    h["EXTVER"] = extver
    h["NAXIS"] = 2
    h["NAXIS1"] = 64
    h["NAXIS2"] = 64
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRPIX1"] = 32.0
    h["CRPIX2"] = 32.0
    h["CRVAL1"] = 150.0
    h["CRVAL2"] = 2.5
    h["CD1_1"] = -5.0e-5
    h["CD1_2"] = 0.0
    h["CD2_1"] = 0.0
    h["CD2_2"] = 5.0e-5
    return h


def test_apply_jhat_shift_all_sci_extensions(tmp_path):
    wd = tmp_path / "ws"
    wd.mkdir()
    sci_path = wd / "iecf02ysq_c0m.fits"
    jhat_path = wd / "iecf02ysq_jhat.fits"

    d1, d2 = 0.0015, -0.0008
    sci_h = [
        fits.ImageHDU(
            data=np.zeros((8, 8), dtype=np.float32),
            header=_minimal_sci_wcs_header(1),
        ),
        fits.ImageHDU(
            data=np.zeros((8, 8), dtype=np.float32),
            header=_minimal_sci_wcs_header(2),
        ),
    ]
    fits.HDUList([fits.PrimaryHDU()] + sci_h).writeto(str(sci_path), overwrite=True)

    jh_h = []
    for ver in (1, 2):
        h = _minimal_sci_wcs_header(ver)
        if ver == 1:
            h["CRVAL1"] = float(h["CRVAL1"]) + d1
            h["CRVAL2"] = float(h["CRVAL2"]) + d2
        jh_h.append(
            fits.ImageHDU(
                data=np.zeros((8, 8), dtype=np.float32),
                header=h,
            )
        )
    fits.HDUList([fits.PrimaryHDU()] + jh_h).writeto(str(jhat_path), overwrite=True)

    ok = jhat_mod.apply_jhat_shift_to_science_image(str(sci_path), str(wd))
    assert ok

    with fits.open(str(sci_path)) as hdul:
        for hdu in hdul:
            if getattr(hdu, "name", "").upper() != "SCI":
                continue
            assert np.isclose(float(hdu.header["CRVAL1"]), 150.0 + d1)
            assert np.isclose(float(hdu.header["CRVAL2"]), 2.5 + d2)


def test_apply_jhat_shift_skips_when_no_jhat_file(tmp_path):
    wd = tmp_path / "ws"
    wd.mkdir()
    sci_path = wd / "orphan_c0m.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.zeros((4, 4), dtype=np.float32),
                header=_minimal_sci_wcs_header(1),
            ),
        ]
    ).writeto(str(sci_path), overwrite=True)
    assert not jhat_mod.apply_jhat_shift_to_science_image(str(sci_path), str(wd))
