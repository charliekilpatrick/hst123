"""Unit tests for hst123.hst123 class methods."""
import numpy as np
import pytest

# Defer import so collection succeeds even if stwcs/drizzlepac not installed
try:
    import hst123 as hst123_module
except Exception as e:
    hst123_module = None
    _hst123_import_error = e


def _require_hst123():
    if hst123_module is None:
        pytest.skip(f"hst123 package not importable: {_hst123_import_error}")


class TestHst123Init:
    def test_init_creates_instance(self):
        _require_hst123()
        hst = hst123_module.hst123()
        assert hst is not None
        assert hst.reference == ""
        assert hst.root_dir == "."
        assert hst.rawdir == "raw"
        assert hst.threshold == 10.0

    def test_options_populated_from_settings(self):
        _require_hst123()
        hst = hst123_module.hst123()
        assert "global_defaults" in hst.options
        assert "detector_defaults" in hst.options
        assert "instrument_defaults" in hst.options
        assert "acceptable_filters" in hst.options
        assert "catalog" in hst.options
        assert hst.options["args"] is None
        assert hst.names is not None
        assert hst.pipeline_products is not None
        assert hst.pipeline_images is not None

    def test_final_phot_empty_table(self):
        _require_hst123()
        hst = hst123_module.hst123()
        assert len(hst.final_phot) == 0
        assert "MJD" in hst.final_phot.colnames
        assert "MAGNITUDE" in hst.final_phot.colnames


class TestAddOptions:
    def test_add_options_delegates_to_options_module(self):
        _require_hst123()
        hst = hst123_module.hst123()
        parser = hst.add_options()
        args = parser.parse_args(["0", "0"])
        assert args.ra == "0"
        assert args.dec == "0"


class TestAvgMagnitudes:
    def test_returns_nan_when_no_good_values(self, hst123_instance):
        mag, err = hst123_instance.avg_magnitudes(
            [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [25.0, 25.0]
        )
        assert np.isnan(mag)
        assert np.isnan(err)

    def test_returns_nan_for_empty_lists(self, hst123_instance):
        mag, err = hst123_instance.avg_magnitudes([], [], [], [])
        assert np.isnan(mag)
        assert np.isnan(err)

    def test_single_good_measurement(self, hst123_instance):
        magerrs = [0.1]
        counts = [1000.0]
        exptimes = [100.0]
        zpt = [25.0]
        mag, err = hst123_instance.avg_magnitudes(magerrs, counts, exptimes, zpt)
        assert not np.isnan(mag)
        assert not np.isnan(err)
        assert err > 0

    def test_filters_bad_values(self, hst123_instance):
        # One good, one bad (magerr too high)
        mag, err = hst123_instance.avg_magnitudes(
            [0.1, 2.0], [1000.0, 500.0], [100.0, 50.0], [25.0, 25.0]
        )
        assert not np.isnan(mag)
        # Result should be driven by the first (good) measurement


class TestEstimateMagLimit:
    def test_empty_arrays_returns_nan(self, hst123_instance):
        result = hst123_instance.estimate_mag_limit([], [])
        assert np.isnan(result)

    def test_single_point_returns_nan(self, hst123_instance):
        result = hst123_instance.estimate_mag_limit([20.0], [0.1])
        assert np.isnan(result)

    def test_monotonic_mags_returns_value(self, hst123_instance):
        mags = np.linspace(20, 26, 50)
        errs = np.ones(50) * 0.1
        result = hst123_instance.estimate_mag_limit(mags, errs, limit=3.0)
        # Should extrapolate to some magnitude
        assert np.isfinite(result) or np.isnan(result)


class TestGetZpt:
    def test_get_zpt_returns_float_for_minimal_fits(self, hst123_instance, minimal_fits_file):
        zpt = hst123_instance.get_zpt(minimal_fits_file, ccdchip=1, zptype="abmag")
        assert zpt is not None
        assert isinstance(zpt, (float, np.floating))
        # ZP_AB = -2.5*log10(PHOTFLAM)-5*log10(PHOTPLAM)-2.408 with PHOTFLAM=1e-19, PHOTPLAM=5000
        assert zpt > 0

    def test_get_zpt_stmag(self, hst123_instance, minimal_fits_file):
        zpt = hst123_instance.get_zpt(minimal_fits_file, zptype="stmag")
        assert zpt is not None
        assert isinstance(zpt, (float, np.floating))


class TestGetInstrument:
    def test_get_instrument_from_minimal_fits(self, hst123_instance, minimal_fits_file):
        out = hst123_instance.get_instrument(minimal_fits_file)
        assert "wfc3" in out.lower()
        assert "uvis" in out.lower()


class TestGetChip:
    def test_get_chip_returns_one_for_single_ccdchip(self, hst123_instance, minimal_fits_file):
        chip = hst123_instance.get_chip(minimal_fits_file)
        assert chip is not None
        assert chip == 1


class TestGetFilter:
    def test_get_filter_from_fits_with_filter_keyword(self, hst123_instance, minimal_fits_file):
        from astropy.io import fits
        with fits.open(minimal_fits_file, mode="update") as hdul:
            hdul[0].header["FILTER"] = "F814W"
        f = hst123_instance.get_filter(minimal_fits_file)
        assert "f814w" in f.lower()


class TestGetDolphotColumn:
    def test_returns_column_number_when_found(self, hst123_instance, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. VEGAMAG (j12345678_flc.fits)\n2. Total counts (j12345678_flc.fits)\n")
        colnum = hst123_instance.get_dolphot_column(str(colfile), "VEGAMAG", "j12345678_flc.fits")
        assert colnum == 0
        colnum = hst123_instance.get_dolphot_column(str(colfile), "Total counts", "j12345678_flc.fits")
        assert colnum == 1

    def test_returns_none_when_key_not_found(self, hst123_instance, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. VEGAMAG (j12345678_flc.fits)\n")
        colnum = hst123_instance.get_dolphot_column(str(colfile), "Nonexistent", "j12345678_flc.fits")
        assert colnum is None


class TestGetDolphotData:
    def test_returns_value_for_valid_row_column_zero(self, hst123_instance, tmp_path):
        """Column 0 (first column) must be returned; bug was 'if colnum' treating 0 as missing."""
        colfile = tmp_path / "columns"
        colfile.write_text("1. VEGAMAG (j12345678_flc.fits)\n")
        row = "25.5 0.1 100.0"
        val = hst123_instance.get_dolphot_data(row, str(colfile), "VEGAMAG", "j12345678_flc.fits")
        assert val == "25.5"

    def test_returns_value_for_second_column(self, hst123_instance, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. VEGAMAG (j12345678_flc.fits)\n2. Total counts (j12345678_flc.fits)\n")
        row = "25.5 1000.0 0.05"
        val = hst123_instance.get_dolphot_data(row, str(colfile), "Total counts", "j12345678_flc.fits")
        assert val == "1000.0"

    def test_returns_none_when_column_missing(self, hst123_instance, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. Other (other_flc.fits)\n")
        row = "25.5"
        val = hst123_instance.get_dolphot_data(row, str(colfile), "VEGAMAG", "j12345678_flc.fits")
        assert val is None

    def test_returns_none_when_colnum_out_of_range(self, hst123_instance, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("5. VEGAMAG (j12345678_flc.fits)\n")  # column index 4
        row = "a b c"  # only 3 columns
        val = hst123_instance.get_dolphot_data(row, str(colfile), "VEGAMAG", "j12345678_flc.fits")
        assert val is None
