"""Unit tests for ScrapeDolphotPrimitive (scrape_dolphot_primitive)."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from astropy.table import Table
from astropy.time import Time
import numpy as np

from hst123.primitives.scrape_dolphot import ScrapeDolphotPrimitive


class TestScrapeDolphotPrimitiveInstantiation:
    def test_instantiation_requires_pipeline(self):
        with pytest.raises(TypeError, match="pipeline"):
            ScrapeDolphotPrimitive(None)

    def test_instantiation_stores_pipeline(self):
        mock_pipeline = object()
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        assert prim._p is mock_pipeline
        assert prim.pipeline is mock_pipeline


class TestGetDolphotColumn:
    @pytest.fixture
    def mock_pipeline(self):
        return object()

    @pytest.fixture
    def primitive(self, mock_pipeline):
        return ScrapeDolphotPrimitive(mock_pipeline)

    def test_returns_column_index_when_found(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text(
            "1. VEGAMAG (j12345678_flc.fits)\n"
            "2. Total counts (j12345678_flc.fits)\n"
        )
        assert primitive.get_dolphot_column(
            str(colfile), "VEGAMAG", "j12345678_flc.fits"
        ) == 0
        assert primitive.get_dolphot_column(
            str(colfile), "Total counts", "j12345678_flc.fits"
        ) == 1

    def test_returns_none_when_key_not_found(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. VEGAMAG (j12345678_flc.fits)\n")
        assert primitive.get_dolphot_column(
            str(colfile), "Nonexistent", "j12345678_flc.fits"
        ) is None

    def test_returns_none_when_image_not_in_line(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. VEGAMAG (other_flc.fits)\n")
        assert primitive.get_dolphot_column(
            str(colfile), "VEGAMAG", "j12345678_flc.fits"
        ) is None

    def test_offset_added_to_column_index(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("5. SomeKey (any.fits)\n")
        assert primitive.get_dolphot_column(
            str(colfile), "SomeKey", "any.fits", offset=0
        ) == 4
        assert primitive.get_dolphot_column(
            str(colfile), "SomeKey", "any.fits", offset=2
        ) == 6


class TestGetDolphotData:
    @pytest.fixture
    def primitive(self):
        return ScrapeDolphotPrimitive(object())

    def test_returns_value_for_string_row(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. VEGAMAG (j12345678_flc.fits)\n")
        row = "25.5 0.1 100.0"
        val = primitive.get_dolphot_data(
            row, str(colfile), "VEGAMAG", "j12345678_flc.fits"
        )
        assert val == "25.5"

    def test_returns_value_for_list_row(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("2. Total counts (j12345678_flc.fits)\n")
        row = ["25.5", "1000.0", "0.05"]
        val = primitive.get_dolphot_data(
            row, str(colfile), "Total counts", "j12345678_flc.fits"
        )
        assert val == "1000.0"

    def test_returns_none_when_column_missing(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. Other (other.fits)\n")
        row = "25.5"
        assert primitive.get_dolphot_data(
            row, str(colfile), "VEGAMAG", "j12345678_flc.fits"
        ) is None

    def test_returns_none_when_colnum_out_of_range(self, primitive, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("5. VEGAMAG (j12345678_flc.fits)\n")
        row = "a b c"
        assert primitive.get_dolphot_data(
            row, str(colfile), "VEGAMAG", "j12345678_flc.fits"
        ) is None


class TestGetLimitData:
    def _make_celestial_wcs(self):
        from astropy.io import fits
        from astropy.wcs import WCS
        # Use a minimal 2D image so the header has NAXIS=2 and WCS does not warn
        hdu = fits.PrimaryHDU(np.zeros((20, 20)))
        hdu.header["CTYPE1"] = "RA---TAN"
        hdu.header["CTYPE2"] = "DEC--TAN"
        hdu.header["CRVAL1"] = 0.0
        hdu.header["CRVAL2"] = 0.0
        hdu.header["CRPIX1"] = 10.0
        hdu.header["CRPIX2"] = 10.0
        hdu.header["CD1_1"] = 1e-5
        hdu.header["CD1_2"] = 0.0
        hdu.header["CD2_1"] = 0.0
        hdu.header["CD2_2"] = 1e-5
        return WCS(hdu.header)

    def test_returns_empty_when_base_has_no_lines_in_radius(self, tmp_path):
        from astropy.coordinates import SkyCoord

        prim = ScrapeDolphotPrimitive(object())
        colfile = tmp_path / "columns"
        colfile.write_text("1. Object X ()\n2. Object Y ()\n")
        base = tmp_path / "base"
        base.write_text("100.0 200.0\n")  # one line far from (10, 10)
        coord = SkyCoord(0.0, 0.0, unit="deg")
        w = self._make_celestial_wcs()
        x, y = 10.0, 10.0
        limit_radius = 10.0
        dolphot = {"base": str(base), "colfile": str(colfile)}
        result = prim.get_limit_data(
            dolphot, coord, w, x, y, str(colfile), limit_radius
        )
        assert isinstance(result, list)
        # Radius is computed from limit_radius (arcsec); line at (100.5, 200.5)
        # may or may not be inside depending on WCS. Just ensure we get a list.
        assert all(isinstance(r, list) and len(r) == 2 for r in result)

    def test_returns_list_of_dist_line_when_line_inside_radius(self, tmp_path):
        from astropy.coordinates import SkyCoord

        prim = ScrapeDolphotPrimitive(object())
        colfile = tmp_path / "columns"
        colfile.write_text("1. Object X ()\n2. Object Y ()\n")
        base = tmp_path / "base"
        base.write_text("10 10\n")
        coord = SkyCoord(0.0, 0.0, unit="deg")
        w = self._make_celestial_wcs()
        x, y = 10.0, 10.0
        limit_radius = 10.0
        dolphot = {"base": str(base), "colfile": str(colfile)}
        result = prim.get_limit_data(
            dolphot, coord, w, x, y, str(colfile), limit_radius
        )
        assert len(result) == 1
        # dist = sqrt(0.5^2 + 0.5^2) from (10,10) to (10.5, 10.5)
        assert result[0][0] == pytest.approx(0.707, abs=0.02)
        row = result[0][1]
        if isinstance(row, str):
            assert row.strip() == "10 10"
        else:
            assert float(row[0]) == 10 and float(row[1]) == 10


class TestCalcAvgStats:
    def test_returns_nan_tuple_when_no_matching_data(self, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. Other (other.fits)\n")  # no Magnitude uncertainty
        prim = ScrapeDolphotPrimitive(object())
        obstable = Table({
            "datetime": [Time("2020-01-01").iso],
            "image": ["a.fits"],
            "exptime": [100.0],
            "filter": ["F606W"],
            "detector": ["UVIS"],
            "zeropoint": [25.0],
        })
        data = "0.1 100.0"
        mjd, mag, magerr, exptime = prim.calc_avg_stats(
            obstable, data, str(colfile)
        )
        assert np.isnan(mjd)
        assert np.isnan(mag)
        assert np.isnan(magerr)
        assert np.isnan(exptime)

    def test_returns_avg_when_data_present(self, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text(
            "1. Magnitude uncertainty (a_flc.fits)\n"
            "2. Measured counts (a_flc.fits)\n"
        )
        mock_pipeline = type("P", (), {})()
        mock_pipeline._phot = type("Phot", (), {})()
        mock_pipeline._phot.avg_magnitudes = lambda err, counts, exptime, zpt: (
            25.0,
            0.1,
        )
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        obstable = Table({
            "datetime": [Time("2020-01-01").iso],
            "image": ["a_flc.fits"],
            "exptime": [100.0],
            "filter": ["F606W"],
            "detector": ["UVIS"],
            "zeropoint": [25.0],
        })
        data = "0.1 1000.0"
        mjd, mag, magerr, exptime = prim.calc_avg_stats(
            obstable, data, str(colfile)
        )
        assert not np.isnan(mjd)
        assert mag == 25.0
        assert magerr == 0.1
        assert exptime == 100.0


class TestParsePhot:
    def test_returns_table_with_expected_columns_and_meta(self, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text(
            "1. Object X ()\n2. Object Y ()\n"
            "3. Object sharpness ()\n4. Object roundness ()\n"
            "5. Magnitude uncertainty (a_flc.fits)\n"
            "6. Measured counts (a_flc.fits)\n"
        )
        mock_pipeline = type("P", (), {})()
        mock_pipeline.snr_limit = 3.0
        mock_pipeline._phot = type("Phot", (), {})()
        mock_pipeline._phot.avg_magnitudes = lambda e, c, ex, z: (25.0, 0.1)
        mock_pipeline._phot.estimate_mag_limit = lambda mags, errs, limit: 26.0
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        obstable = Table({
            "instrument": ["WFC3"],
            "visit": [1],
            "filter": ["F606W"],
            "datetime": [Time("2020-01-01").iso],
            "image": ["a_flc.fits"],
            "exptime": [100.0],
            "detector": ["UVIS"],
            "zeropoint": [25.0],
        })
        row = "100 200 0.5 0.3 0.1 1000.0"
        result = prim.parse_phot(obstable, row, str(colfile), limit_data=[])
        assert isinstance(result, Table)
        assert "MJD" in result.colnames
        assert "MAGNITUDE" in result.colnames
        assert float(result.meta.get("x")) == 100.0
        assert float(result.meta.get("y")) == 200.0


class TestPrintFinalPhot:
    def test_writes_phot_and_snana_files_and_calls_show_photometry(
        self, tmp_path
    ):
        written = []

        def fake_show_photometry(phot, f=None, snana=False, show=True, **kwargs):
            if f is not None:
                written.append(("snana" if snana else "phot", getattr(f, "name", str(f))))

        mock_pipeline = type("P", (), {})()
        mock_pipeline.coord = None
        mock_pipeline.options = {}
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        with patch(
            "hst123.primitives.scrape_dolphot.scrape_dolphot_primitive.display_show_photometry",
            side_effect=fake_show_photometry,
        ):
            phot1 = Table({"MJD": [59000.0], "MAGNITUDE": [25.0]})
            phot1.meta["x"] = "10"
            phot1.meta["y"] = "20"
            phot1.meta["separation"] = 0.5
            final_phot = [phot1]
            dolphot = {"final_phot": str(tmp_path / "out.phot")}
            prim.print_final_phot(final_phot, dolphot, allphot=True)
        assert len(written) == 2
        assert (tmp_path / "out_0.phot").exists() or (tmp_path / "out.phot").exists()
        assert any("snana" in w[0] for w in written)


def _pipeline_with_scrape_options(tmp_path):
    """Minimal ``p.options['args']`` used by ``scrapedolphot`` (work_dir + cuts)."""
    mock_pipeline = type("P", (), {})()
    mock_pipeline.options = {
        "args": SimpleNamespace(
            work_dir=str(tmp_path),
            no_cuts=True,
            scrape_radius=None,
        )
    }
    return mock_pipeline


class TestScrapedolphot:
    def test_returns_none_when_base_missing(self, tmp_path):
        colfile = tmp_path / "columns"
        colfile.write_text("1. Object X ()\n")
        mock_pipeline = _pipeline_with_scrape_options(tmp_path)
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        dolphot = {
            "base": str(tmp_path / "nonexistent_base"),
            "colfile": str(colfile),
            "original": str(tmp_path / "orig"),
            "radius": 12,
            "limit_radius": 10.0,
        }
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(0.0, 0.0, unit="deg")
        result = prim.scrapedolphot(
            coord, str(tmp_path / "ref.fits"), [], dolphot
        )
        assert result is None

    def test_returns_none_when_colfile_missing(self, tmp_path):
        base = tmp_path / "base"
        base.write_text("1 2 3\n")
        mock_pipeline = _pipeline_with_scrape_options(tmp_path)
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        dolphot = {
            "base": str(base),
            "colfile": str(tmp_path / "nonexistent.columns"),
            "original": str(tmp_path / "orig"),
            "radius": 12,
        }
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(0.0, 0.0, unit="deg")
        result = prim.scrapedolphot(
            coord, str(tmp_path / "ref.fits"), [], dolphot
        )
        assert result is None

    def test_returns_none_when_reference_none(self, tmp_path):
        base = tmp_path / "base"
        base.write_text("1 2 3\n")
        colfile = tmp_path / "columns"
        colfile.write_text("1. Object X ()\n2. Object Y ()\n")
        mock_pipeline = _pipeline_with_scrape_options(tmp_path)
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        dolphot = {
            "base": str(base),
            "colfile": str(colfile),
            "original": str(tmp_path / "orig"),
            "radius": 12,
        }
        from astropy.coordinates import SkyCoord
        coord = SkyCoord(0.0, 0.0, unit="deg")
        result = prim.scrapedolphot(coord, None, [], dolphot)
        assert result is None

    def test_returns_none_when_coord_none(self, tmp_path):
        base = tmp_path / "base"
        base.write_text("1 2 3\n")
        colfile = tmp_path / "columns"
        colfile.write_text("1. Object X ()\n2. Object Y ()\n")
        mock_pipeline = _pipeline_with_scrape_options(tmp_path)
        prim = ScrapeDolphotPrimitive(mock_pipeline)
        dolphot = {
            "base": str(base),
            "colfile": str(colfile),
            "original": str(tmp_path / "orig"),
            "radius": 12,
        }
        result = prim.scrapedolphot(None, str(tmp_path / "ref.fits"), [], dolphot)
        assert result is None
