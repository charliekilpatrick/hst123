"""Unit tests for the astrometry primitive (parse_coord, AstrometryPrimitive)."""
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

import os

from hst123.primitives.astrometry import AstrometryPrimitive, parse_coord
from hst123.primitives.astrometry.astrometry_primitive import (
    _resolve_work_dir_chdir,
)


def test_resolve_work_dir_chdir_no_double_test_data(tmp_path, monkeypatch):
    """After chdir(work_dir), shiftfile path must not be work_dir/shift under CWD."""
    root = tmp_path / "repo"
    root.mkdir()
    wd = root / "test_data"
    wd.mkdir()
    monkeypatch.chdir(root)
    resolved = _resolve_work_dir_chdir("test_data")
    assert resolved == str(wd.resolve())
    assert os.getcwd() == str(wd.resolve())
    shift = os.path.join(resolved, "drizzle_shifts.txt")
    assert shift == str(wd / "drizzle_shifts.txt")
    assert "test_data" + os.sep + "test_data" not in shift


class TestParseCoord:
    def test_degree_input(self):
        coord = parse_coord(180.0, -45.0)
        assert coord is not None
        assert isinstance(coord, SkyCoord)
        assert coord.ra.deg == pytest.approx(180.0)
        assert coord.dec.deg == pytest.approx(-45.0)

    def test_sexagesimal_input(self):
        coord = parse_coord("12:00:00", "+00:00:00")
        assert coord is not None
        assert isinstance(coord, SkyCoord)
        assert coord.ra.hour == pytest.approx(12.0)
        assert coord.dec.deg == pytest.approx(0.0)

    def test_invalid_input_returns_none(self):
        result = parse_coord("not", "valid")
        assert result is None

    def test_string_degrees(self):
        coord = parse_coord("0", "0")
        assert coord is not None
        assert coord.ra.deg == pytest.approx(0.0)
        assert coord.dec.deg == pytest.approx(0.0)


class TestAstrometryPrimitive:
    """Lightweight tests for AstrometryPrimitive (no drizzlepac/tweakreg runs)."""

    @pytest.fixture
    def mock_pipeline(self):
        """Minimal pipeline-like object for primitive instantiation."""
        return type("MockPipeline", (), {})()

    def test_instantiation_requires_pipeline(self):
        with pytest.raises(TypeError, match="pipeline instance"):
            AstrometryPrimitive(None)

    def test_instantiation_with_mock_pipeline(self, mock_pipeline):
        astrom = AstrometryPrimitive(mock_pipeline)
        assert astrom._p is mock_pipeline
        assert astrom.pipeline is mock_pipeline

    def test_copy_wcs_keys_copies_header_keys(self, mock_pipeline):
        astrom = AstrometryPrimitive(mock_pipeline)
        from_hdu = fits.PrimaryHDU()
        from_hdu.header["CRPIX1"] = 100.0
        from_hdu.header["CRPIX2"] = 200.0
        from_hdu.header["CRVAL1"] = 180.0
        from_hdu.header["CRVAL2"] = -45.0
        to_hdu = fits.PrimaryHDU()
        astrom.copy_wcs_keys(from_hdu, to_hdu)
        assert to_hdu.header["CRPIX1"] == 100.0
        assert to_hdu.header["CRPIX2"] == 200.0
        assert to_hdu.header["CRVAL1"] == 180.0
        assert to_hdu.header["CRVAL2"] == -45.0

    def test_tweakreg_error_does_not_raise(self, mock_pipeline, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        astrom = AstrometryPrimitive(mock_pipeline)
        astrom.tweakreg_error(ValueError("test"))
        assert "tweakreg failed" in caplog.text
        assert "ValueError" in caplog.text
