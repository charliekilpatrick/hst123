"""Unit tests for utils.wcs_utils (make_meta_wcs_header, SIP CTYPE fix)."""
import numpy as np
import pytest
from astropy.io import fits

from hst123.utils.wcs_utils import (
    fix_sip_ctype_headers_fits,
    make_meta_wcs_header,
    remove_conflicting_alt_wcs_duplicate_names,
    wcs_from_fits_hdu,
)

class TestMakeMetaWcsHeader:
    def test_returns_dict_with_required_keys(self):
        class Header:
            def __getitem__(self, key):
                return {
                    "NAXIS": 2,
                    "NAXIS1": 100,
                    "NAXIS2": 100,
                    "CD1_1": 1e-5,
                    "CD1_2": 0,
                    "CD2_1": 0,
                    "CD2_2": 1e-5,
                    "CRVAL1": 10.0,
                    "CRVAL2": 20.0,
                    "CRPIX1": 50.0,
                    "CRPIX2": 50.0,
                    "CTYPE1": "RA---TAN",
                    "CTYPE2": "DEC--TAN",
                }[key]
        h = Header()
        meta = make_meta_wcs_header(h)
        assert meta["NAXIS"] == 2
        assert meta["NAXIS1"] == 100
        assert meta["CRVAL1"] == 10.0
        assert meta["CTYPE1"] == "RA---TAN"
        assert meta["CTYPE2"] == "DEC--TAN"

    def test_normalizes_sip_ctype(self):
        class Header:
            def __getitem__(self, key):
                d = {
                    "NAXIS": 2, "NAXIS1": 100, "NAXIS2": 100,
                    "CD1_1": 1e-5, "CD1_2": 0, "CD2_1": 0, "CD2_2": 1e-5,
                    "CRVAL1": 10.0, "CRVAL2": 20.0,
                    "CRPIX1": 50.0, "CRPIX2": 50.0,
                    "CTYPE1": "RA---TAN-SIP",
                    "CTYPE2": "DEC--TAN-SIP",
                }
                return d[key]
        meta = make_meta_wcs_header(Header())
        assert meta["CTYPE1"] == "RA---TAN"
        assert meta["CTYPE2"] == "DEC--TAN"


class TestFixSipCtypeHeaders:
    def test_appends_sip_when_polynomials_present(self, tmp_path):
        p = tmp_path / "sip.fits"
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 32
        hdr["NAXIS2"] = 32
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 16.0
        hdr["CRPIX2"] = 16.0
        hdr["CRVAL1"] = 10.0
        hdr["CRVAL2"] = 20.0
        hdr["CD1_1"] = -1e-5
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 1e-5
        hdr["A_ORDER"] = 2
        hdr["B_ORDER"] = 2
        hdr["A_2_0"] = 1e-10
        hdr["B_2_0"] = 1e-10
        fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float32), header=hdr).writeto(
            str(p), overwrite=True
        )
        assert fix_sip_ctype_headers_fits(str(p)) == 1
        with fits.open(p) as hdul:
            assert hdul[0].header["CTYPE1"] == "RA---TAN-SIP"
            assert hdul[0].header["CTYPE2"] == "DEC--TAN-SIP"

    def test_idempotent_and_skips_without_sip(self, tmp_path):
        p = tmp_path / "plain.fits"
        fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32)).writeto(str(p), overwrite=True)
        assert fix_sip_ctype_headers_fits(str(p)) == 0

    def test_second_pass_noop_after_fix(self, tmp_path):
        p = tmp_path / "sip2.fits"
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 16
        hdr["NAXIS2"] = 16
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 8.0
        hdr["CRPIX2"] = 8.0
        hdr["CRVAL1"] = 10.0
        hdr["CRVAL2"] = 20.0
        hdr["CD1_1"] = -1e-5
        hdr["CD2_2"] = 1e-5
        hdr["A_ORDER"] = 1
        hdr["B_ORDER"] = 1
        hdr["A_1_0"] = 1e-12
        hdr["B_0_1"] = 1e-12
        fits.PrimaryHDU(np.zeros((16, 16), dtype=np.float32), header=hdr).writeto(
            str(p), overwrite=True
        )
        assert fix_sip_ctype_headers_fits(str(p)) == 1
        assert fix_sip_ctype_headers_fits(str(p)) == 0


class TestWcsFromFitsHdu:
    """wcs_from_fits_hdu must pass fobj for HST-style distortion tables."""

    def test_linear_wcs_from_primary(self, tmp_path):
        p = tmp_path / "linear.fits"
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 32
        hdr["NAXIS2"] = 32
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 16.0
        hdr["CRPIX2"] = 16.0
        hdr["CRVAL1"] = 45.0
        hdr["CRVAL2"] = 60.0
        hdr["CD1_1"] = -1e-5
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 1e-5
        fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float32), header=hdr).writeto(
            str(p), overwrite=True
        )
        with fits.open(p) as hdul:
            w = wcs_from_fits_hdu(hdul, 0)
            assert w.naxis == 2
            px = w.world_to_pixel_values(45.0, 60.0)
            assert abs(px[0] - 15.0) < 0.1  # 0-based ~ CRPIX-1
            assert abs(px[1] - 15.0) < 0.1

    def test_fallback_when_d2imarr_extension_missing(self, tmp_path):
        """Split-chip FITS can keep D2IM* keys but omit D2IMARR HDUs (Astropy KeyError)."""
        p = tmp_path / "d2im_missing.fits"
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 32
        hdr["NAXIS2"] = 32
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 16.0
        hdr["CRPIX2"] = 16.0
        hdr["CRVAL1"] = 45.0
        hdr["CRVAL2"] = 60.0
        hdr["CD1_1"] = -1e-5
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 1e-5
        hdr["D2IMDIS1"] = "LOOKUP"
        hdr["D2IMERR1"] = 1.0
        hdr["D2IM1.EXTVER"] = 3
        hdr["D2IM1.AXIS.1"] = 1
        fits.PrimaryHDU(np.zeros((32, 32), dtype=np.float32), header=hdr).writeto(
            str(p), overwrite=True
        )
        with fits.open(p) as hdul:
            w = wcs_from_fits_hdu(hdul, 0)
            assert w.naxis == 2
            px = w.world_to_pixel_values(45.0, 60.0)
            assert abs(px[0] - 15.0) < 0.1
            assert abs(px[1] - 15.0) < 0.1


class TestRemoveConflictingAltWcs:
    def test_noop_when_no_duplicate_alternates(self, tmp_path):
        p = tmp_path / "plain.fits"
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 16
        hdr["NAXIS2"] = 16
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRPIX1"] = 8.0
        hdr["CRPIX2"] = 8.0
        hdr["CRVAL1"] = 10.0
        hdr["CRVAL2"] = 20.0
        hdr["CD1_1"] = -1e-5
        hdr["CD2_2"] = 1e-5
        hdr["WCSNAME"] = "IDC_TEST"
        fits.PrimaryHDU(np.zeros((16, 16), dtype=np.float32), header=hdr).writeto(
            str(p), overwrite=True
        )
        assert remove_conflicting_alt_wcs_duplicate_names(p) == 0
