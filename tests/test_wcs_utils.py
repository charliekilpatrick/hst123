"""Unit tests for utils.wcs_utils (make_meta_wcs_header, SIP CTYPE fix)."""
import numpy as np
import pytest
from astropy.io import fits

from hst123.utils.wcs_utils import fix_sip_ctype_headers_fits, make_meta_wcs_header


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
