"""Unit tests for the astrometry primitive (parse_coord, AstrometryPrimitive)."""
import os
import shutil
from unittest.mock import MagicMock

import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from hst123.primitives.astrometry import AstrometryPrimitive, parse_coord
from hst123.primitives.astrometry.astrometry_primitive import (
    _fits_header_merge_value,
    _fits_safe_primary_header_string,
    _resolve_work_dir_chdir,
)


def test_fits_safe_primary_header_string_ascii():
    """Bibliographic HISTORY/COMMENT text must not break prepare_reference_tweakreg."""
    raw = (
        "\n  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"
    )
    out = _fits_safe_primary_header_string(raw)
    assert "\n" not in out
    assert all(32 <= ord(c) <= 126 for c in out)


def test_write_flc_anchor_refcat_for_jhat_smoke(tmp_path):
    """Synthetic SCI image + WCS yields a non-empty anchor refcat."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from hst123.primitives.astrometry.jhat import write_flc_anchor_refcat_for_jhat

    ny, nx = 128, 128
    data = np.random.RandomState(0).normal(100.0, 10.0, (ny, nx)).astype(np.float32)
    data[60:62, 60:62] += 5000.0

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.crval = [337.1, 30.29]
    wcs.wcs.cdelt = [-5e-5, 5e-5]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    hdr = wcs.to_header()
    hdr["EXPTIME"] = 100.0
    hdr["PHOTZPT"] = 21.0
    im = tmp_path / "x_flc.fits"
    fits.writeto(im, data, hdr, overwrite=True)

    out = write_flc_anchor_refcat_for_jhat(im, tmp_path)
    assert os.path.isfile(out)
    txt = open(out, encoding="ascii").read().strip().splitlines()
    assert len(txt) >= 2
    assert txt[0].split() == ["ra", "dec", "mag", "dmag"]


def test_fits_header_merge_value_numpy_str():
    import numpy as np

    s = np.str_("line1\nline2 and \u00c5&A")
    out = _fits_header_merge_value(s)
    assert "\n" not in out
    assert all(32 <= ord(c) <= 126 for c in out)


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

    def _astrom_with_args(self, **args):
        from types import SimpleNamespace

        pipeline = type("MockPipeline", (), {})()
        pipeline.options = {"args": SimpleNamespace(**args)}
        return AstrometryPrimitive(pipeline)

    def test_resolve_fitgeometry_defaults_to_rscale(self):
        astrom = self._astrom_with_args(tweakreg_fitgeometry=None)
        assert astrom._resolve_tweakreg_fitgeometry() == "rscale"

    def test_resolve_fitgeometry_cli_override(self):
        astrom = self._astrom_with_args(tweakreg_fitgeometry="general")
        assert astrom._resolve_tweakreg_fitgeometry() == "general"

    def test_resolve_fitgeometry_invalid_falls_back(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        astrom = self._astrom_with_args(tweakreg_fitgeometry="bogus")
        assert astrom._resolve_tweakreg_fitgeometry() == "rscale"
        assert "Unknown tweakreg fitgeometry" in caplog.text

    def test_resolve_fitgeometry_missing_attr_uses_default(self, mock_pipeline):
        mock_pipeline.options = {"args": object()}
        astrom = AstrometryPrimitive(mock_pipeline)
        assert astrom._resolve_tweakreg_fitgeometry() == "rscale"

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

    def test_build_tweakreg_batches_reference_in_band_is_one_batch(
        self, tmp_path, mock_pipeline
    ):
        """
        When the TweakReg reference file is one of the inputs (e.g. first FLC in
        a pair), use a single batch—no spurious "need ≥2" from a one-file list, and
        no second batch that re-anchors the reference to the deeper exposure.
        """
        a_orig = tmp_path / "a_flc.fits"
        b_raw = tmp_path / "b_flc.rawtmp.fits"
        b_orig = tmp_path / "b_flc.fits"
        for path, exptime in [(a_orig, 100.0), (b_orig, 300.0)]:
            h = fits.PrimaryHDU()
            h.header["EXPTIME"] = exptime
            h.writeto(path, overwrite=True)
        shutil.copyfile(b_orig, b_raw)

        def _get_filter(path):
            return "F814W"

        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_filter.side_effect = _get_filter
        astrom = AstrometryPrimitive(mock_pipeline)

        imgs = [str(a_orig), str(b_raw)]
        batches = astrom._build_tweakreg_batches(imgs, str(a_orig))
        assert batches == [(str(a_orig), imgs)]

    def test_run_alignment_jhat_with_reference_skips_tweakreg(self, tmp_path, monkeypatch):
        """With a pipeline drizzle reference, hierarchical JHAT does not use TweakReg."""
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

    def test_run_alignment_jhat_internal_uses_jhat_not_tweakreg(
        self, tmp_path, monkeypatch
    ):
        """No pipeline reference: FLC–FLC uses JHAT relative only (no TweakReg)."""
        import hst123.primitives.astrometry.jhat as jh

        class Args:
            align_with = "jhat"
            work_dir = str(tmp_path)
            clobber = False

        class P:
            options = {"args": Args()}
            _jhat_gaia_ref_stats = []

        astrom = AstrometryPrimitive(P())

        im1 = tmp_path / "a.fits"
        im2 = tmp_path / "b.fits"
        for path, exptime in [(im1, 100.0), (im2, 200.0)]:
            h = fits.PrimaryHDU()
            h.header["EXPTIME"] = exptime
            h.writeto(path, overwrite=True)

        def boom_tweakreg(*args, **kwargs):
            raise AssertionError("TweakReg must not run when align_with=jhat")

        monkeypatch.setattr(astrom, "run_tweakreg", boom_tweakreg)

        jhat_calls = []

        def fake_run_jhat(*args, **kwargs):
            jhat_calls.append(kwargs)

        monkeypatch.setattr(jh, "run_jhat", fake_run_jhat)
        monkeypatch.setattr(
            jh,
            "write_flc_anchor_refcat_for_jhat",
            lambda a, o: str(tmp_path / "anchor_cat.txt"),
        )
        monkeypatch.setattr(
            jh,
            "apply_jhat_shift_to_science_image",
            lambda *a, **k: True,
        )

        obstable = {"image": [str(im1), str(im2)]}
        out = astrom.run_alignment(obstable, "", do_cosmic=False, skip_wcs=True)
        assert out[0] == "jhat success"
        assert len(jhat_calls) == 1
        assert jhat_calls[0]["gaia"] is False
        assert jhat_calls[0]["photfilename"] == str(tmp_path / "anchor_cat.txt")


def test_dispersion_matched_median_arcsec_identical():
    """Perfect overlap yields ~zero dispersion cost."""
    import numpy as np

    from hst123.primitives.astrometry.jhat import _dispersion_matched_median_arcsec

    ra = np.array([10.0, 10.001, 10.002])
    dec = np.array([20.0, 20.001, 20.002])
    cost, n_ok = _dispersion_matched_median_arcsec(
        ra,
        dec,
        ra,
        dec,
        dist_limit_arcsec=10.0,
        sigma_clip=3.0,
    )
    assert cost == pytest.approx(0.0, abs=1e-9)
    assert n_ok == 3


def test_guess_shift_hst_flc_zero_radius_returns_no_pixel_shift(tmp_path):
    """radius_px=0 scans only the intrinsic CRPIX; winning shift must be (0, 0)."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from hst123.primitives.astrometry.jhat import (
        guess_shift_hst_flc,
        write_flc_anchor_refcat_for_jhat,
    )

    ny, nx = 128, 128
    data = np.random.RandomState(0).normal(100.0, 10.0, (ny, nx)).astype(np.float32)
    # Several compact boosts so peak detection yields >=3 sources (dispersion needs >=3 matches).
    for y0, x0 in ((60, 60), (22, 92), (101, 41)):
        data[y0 : y0 + 2, x0 : x0 + 2] += 5000.0

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2, ny / 2]
    wcs.wcs.crval = [337.1, 30.29]
    wcs.wcs.cdelt = [-5e-5, 5e-5]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    hdr = wcs.to_header()
    hdr["EXPTIME"] = 100.0
    hdr["PHOTZPT"] = 21.0
    im = tmp_path / "align_flc.fits"
    fits.writeto(im, data, hdr, overwrite=True)

    refcat = write_flc_anchor_refcat_for_jhat(im, tmp_path)
    dx, dy, cost, n_match = guess_shift_hst_flc(
        im,
        refcat,
        radius_px=0.0,
        step_px=5.0,
    )
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.0)
    assert np.isfinite(cost)
    assert n_match >= 3


def test_apply_crpix_guess_shift_to_flc_updates_sci_headers(tmp_path):
    """CRPIX and PRIMARY guess keywords are updated on all SCI HDUs."""
    import logging

    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from hst123.primitives.astrometry.jhat import apply_crpix_guess_shift_to_flc

    ny, nx = 32, 32
    data = np.zeros((ny, nx), dtype=np.float32)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [16.0, 16.0]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.cdelt = [-1e-4, 1e-4]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    ihdu = fits.ImageHDU(data, header=hdr, name="SCI")
    phdu = fits.PrimaryHDU()
    path = tmp_path / "two_sci.fits"
    fits.HDUList([phdu, ihdu]).writeto(path, overwrite=True)

    c1 = float(fits.getval(path, "CRPIX1", extname="SCI"))
    c2 = float(fits.getval(path, "CRPIX2", extname="SCI"))
    log = logging.getLogger("test_apply_crpix")
    n = apply_crpix_guess_shift_to_flc(
        path,
        1.25,
        -0.5,
        min_cost_arcsec=0.042,
        logger=log,
    )
    assert n == 1
    with fits.open(path) as hdul:
        assert hdul["SCI"].header["CRPIX1"] == pytest.approx(c1 + 1.25)
        assert hdul["SCI"].header["CRPIX2"] == pytest.approx(c2 - 0.5)
        assert hdul[0].header["HST123GSX"] == pytest.approx(1.25)
        assert hdul[0].header["HST123GSY"] == pytest.approx(-0.5)
