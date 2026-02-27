"""Tests for primitives package (BasePrimitive, FitsHelper, PhotometryHelper and their primitives)."""
import numpy as np
import pytest

from hst123.primitives import BasePrimitive, FitsHelper, PhotometryHelper
from hst123.primitives.photometry import weighted_avg_flux_to_mag, estimate_limit_from_snr_bins
from hst123.primitives.fits import _instrument_from_header, _phot_zero_point_ab


class TestBasePrimitive:
    def test_requires_pipeline(self):
        with pytest.raises(TypeError, match="pipeline"):
            BasePrimitive(None)

    def test_stores_pipeline_and_exposes_property(self):
        class P:
            pass
        p = P()
        h = FitsHelper(p)
        assert h.pipeline is p
        assert h._p is p

    def test_fits_and_photometry_inherit_from_base(self):
        assert issubclass(FitsHelper, BasePrimitive)
        assert issubclass(PhotometryHelper, BasePrimitive)


class TestFitsPrimitives:
    def test_phot_zero_point_ab(self):
        z = _phot_zero_point_ab(1.0e-19, 5000.0)
        assert np.isfinite(z)
        assert z > 0

    def test_instrument_from_header_wfc3(self):
        class H:
            header = {"INSTRUME": "WFC3", "DETECTOR": "UVIS", "SUBARRAY": False}
        assert "wfc3" in _instrument_from_header(H.header)
        assert "uvis" in _instrument_from_header(H.header)


class TestFitsHelper:
    def test_helper_needs_pipeline_mock(self):
        class P:
            pass
        p = P()
        h = FitsHelper(p)
        assert h._p is p


class TestPhotometryPrimitives:
    def test_weighted_avg_flux_to_mag(self):
        mag, magerr = weighted_avg_flux_to_mag(
            np.array([100.0, 200.0]), np.array([10.0, 20.0])
        )
        assert np.isfinite(mag)
        assert np.isfinite(magerr)
        assert magerr > 0

    def test_estimate_limit_from_snr_bins(self):
        mags = np.linspace(20, 26, 50)
        errs = np.ones(50) * 0.1
        limit = estimate_limit_from_snr_bins(mags, errs, snr_target=3.0)
        # Can be finite, nan, or ±inf depending on extrapolation
        assert isinstance(limit, (float, np.floating))


class TestPhotometryHelper:
    def test_avg_magnitudes_empty(self):
        class P:
            pass
        h = PhotometryHelper(P())
        mag, err = h.avg_magnitudes([], [], [], [])
        assert np.isnan(mag)
        assert np.isnan(err)

    def test_avg_magnitudes_single_good(self):
        class P:
            pass
        h = PhotometryHelper(P())
        mag, err = h.avg_magnitudes([0.1], [1000.0], [100.0], [25.0])
        assert np.isfinite(mag)
        assert np.isfinite(err)
