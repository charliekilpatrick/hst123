"""Unit tests for the astrometry primitive (parse_coord, AstrometryPrimitive)."""
import os
from unittest.mock import MagicMock

import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from hst123.primitives.astrometry import AstrometryPrimitive, parse_coord
from hst123.primitives.astrometry.astrometry_primitive import (
    _resolve_work_dir_chdir,
)


def test_resolve_work_dir_chdir_no_double_test_data(tmp_path, monkeypatch):
    """After chdir, shiftfile lives under workspace/; paths must not double work_dir."""
    root = tmp_path / "repo"
    root.mkdir()
    wd = root / "test_data"
    wd.mkdir()
    ws = wd / "workspace"
    monkeypatch.chdir(root)
    resolved = _resolve_work_dir_chdir("test_data")
    assert resolved == str(ws.resolve())
    assert os.getcwd() == str(ws.resolve())
    shift = os.path.join(resolved, "drizzle_shifts.txt")
    assert shift == str(ws / "drizzle_shifts.txt")
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

    def test_ensure_workspace_rawtmps_rebuilds_missing(self, tmp_path, mock_pipeline):
        """Missing *.rawtmp.fits is recreated from the workspace science *.fits."""
        sci = tmp_path / "sci_flc.fits"
        raw = tmp_path / "sci_flc.rawtmp.fits"
        fits.PrimaryHDU().writeto(str(sci), overwrite=True)
        assert not raw.is_file()
        mock_run_cosmic = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "acs_wfc"
        mock_pipeline.options = {"instrument_defaults": {"acs": {"crpars": {}}}}
        mock_pipeline.run_cosmic = mock_run_cosmic
        astrom = AstrometryPrimitive(mock_pipeline)
        astrom._ensure_workspace_rawtmps([str(raw)], do_cosmic=True)
        assert raw.is_file()
        mock_run_cosmic.assert_called_once()

    def test_ensure_workspace_rawtmps_noop_when_do_cosmic_false(
        self, tmp_path, mock_pipeline
    ):
        raw = tmp_path / "sci_flc.rawtmp.fits"
        mock_pipeline.run_cosmic = MagicMock()
        astrom = AstrometryPrimitive(mock_pipeline)
        astrom._ensure_workspace_rawtmps([str(raw)], do_cosmic=False)
        assert not raw.is_file()
        mock_pipeline.run_cosmic.assert_not_called()

    def test_build_tweakreg_batches_bridges_filters_to_pipeline_reference(
        self, tmp_path, mock_pipeline
    ):
        """
        Hierarchical alignment: each filter aligns internally to its deepest image,
        and each filter anchor is additionally aligned to the pipeline reference.
        """
        # Create original images (used for filter lookup) and rawtmp working copies.
        imgs = []
        for name, filt, exptime in [
            ("a_flc", "F300W", 100.0),
            ("b_flc", "F300W", 200.0),
            ("c_flc", "F814W", 300.0),
            ("d_flc", "F814W", 400.0),
        ]:
            orig = tmp_path / f"{name}.fits"
            rawtmp = tmp_path / f"{name}.rawtmp.fits"
            h = fits.PrimaryHDU()
            h.header["EXPTIME"] = exptime
            h.writeto(str(orig), overwrite=True)
            h.writeto(str(rawtmp), overwrite=True)
            imgs.append(str(rawtmp))

        # Pipeline reference drizzle.
        ref = tmp_path / "ref.drc.fits"
        h = fits.PrimaryHDU()
        h.header["EXPTIME"] = 999.0
        h.writeto(str(ref), overwrite=True)

        def _get_filter(path):
            base = os.path.basename(path)
            if base.startswith("a_") or base.startswith("b_"):
                return "F300W"
            return "F814W"

        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_filter.side_effect = _get_filter
        astrom = AstrometryPrimitive(mock_pipeline)

        batches = astrom._build_tweakreg_batches(imgs, str(ref))

        # Expect two bridge batches (one per filter anchor), plus two main batches.
        # Bridge batches align deepest-of-filter to the pipeline reference.
        assert (str(ref), [str(tmp_path / "b_flc.rawtmp.fits")]) in batches
        assert (str(ref), [str(tmp_path / "d_flc.rawtmp.fits")]) in batches

        # Main batches align all images in each filter to their deepest anchor.
        assert (str(tmp_path / "b_flc.rawtmp.fits"), [str(tmp_path / "a_flc.rawtmp.fits"), str(tmp_path / "b_flc.rawtmp.fits")]) in batches
        assert (str(tmp_path / "d_flc.rawtmp.fits"), [str(tmp_path / "c_flc.rawtmp.fits"), str(tmp_path / "d_flc.rawtmp.fits")]) in batches

    def test_run_alignment_jhat_does_not_call_tweakreg(self, tmp_path, monkeypatch):
        """When align_with=jhat, run_alignment should not invoke run_tweakreg."""
        # Minimal pipeline/options
        class Args:
            align_with = "jhat"
            work_dir = str(tmp_path)
            clobber = False

        class P:
            options = {"args": Args()}
            _jhat_gaia_ref_stats = []

        p = P()
        astrom = AstrometryPrimitive(p)

        # Make two dummy images and a reference drizzle.
        im1 = tmp_path / "a.fits"
        im2 = tmp_path / "b.fits"
        ref = tmp_path / "ref.drc.fits"
        fits.PrimaryHDU().writeto(im1, overwrite=True)
        fits.PrimaryHDU().writeto(im2, overwrite=True)
        fits.PrimaryHDU().writeto(ref, overwrite=True)

        # Stub run_tweakreg to explode if called.
        def _boom(*args, **kwargs):
            raise AssertionError("run_tweakreg should not be called for jhat")

        monkeypatch.setattr(astrom, "run_tweakreg", _boom)

        # Stub JHAT helpers so we don't require jhat in unit tests.
        import hst123.primitives.astrometry.jhat as jh

        def fake_run_jhat(*args, **kwargs):
            return None

        monkeypatch.setattr(jh, "run_jhat", fake_run_jhat)
        # Also pretend the reference catalogs exist (avoid warning path raising).
        (tmp_path / "ref_jhat.good.phot.txt").write_text("ra dec\n0 0\n")

        obstable = {"image": [str(im1), str(im2)]}
        out = astrom.run_alignment(obstable, str(ref), do_cosmic=False, skip_wcs=True)
        assert out[0] == "jhat success"
