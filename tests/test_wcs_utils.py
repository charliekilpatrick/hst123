"""Unit tests for utils.wcs_utils (make_meta_wcs_header)."""
import pytest

from hst123.utils.wcs_utils import make_meta_wcs_header


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
