"""Unit tests for DolphotPrimitive (run_dolphot_primitive)."""
import os
import shutil
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest
from astropy.io import fits
import numpy as np

from hst123.primitives.run_dolphot import (
    DOLPHOT_REQUIRED_SCRIPTS,
    DolphotPrimitive,
)
from hst123.primitives.run_dolphot.run_dolphot_primitive import (
    calcsky_subprocess_env,
    dolphot_subprocess_env,
)


class TestDolphotPrimitiveInstantiation:
    def test_instantiation_requires_pipeline(self):
        with pytest.raises(TypeError, match="pipeline"):
            DolphotPrimitive(None)

    def test_instantiation_stores_pipeline(self):
        mock_pipeline = object()
        prim = DolphotPrimitive(mock_pipeline)
        assert prim._p is mock_pipeline
        assert prim.pipeline is mock_pipeline


class TestDolphotSubprocessEnv:
    """macOS OpenMP: avoid SIGTRAP by defaulting OMP_NUM_THREADS unless user set it."""

    def test_respects_existing_omp_num_threads(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.delenv("HST123_DOLPHOT_OMP_THREADS", raising=False)
        e = dolphot_subprocess_env()
        assert e["OMP_NUM_THREADS"] == "4"

    def test_hst123_override_when_omp_unset(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.setenv("HST123_DOLPHOT_OMP_THREADS", "2")
        e = dolphot_subprocess_env()
        assert e["OMP_NUM_THREADS"] == "2"

    def test_darwin_defaults_omp_one(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("HST123_DOLPHOT_OMP_THREADS", raising=False)
        monkeypatch.setattr(sys, "platform", "darwin")
        e = dolphot_subprocess_env()
        assert e.get("OMP_NUM_THREADS") == "1"

    def test_non_darwin_does_not_set_omp_when_unset(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("HST123_DOLPHOT_OMP_THREADS", raising=False)
        monkeypatch.setattr(sys, "platform", "linux")
        e = dolphot_subprocess_env()
        assert "OMP_NUM_THREADS" not in e


class TestCalcskySubprocessEnv:
    """calcsky ignores inherited OMP on Darwin (avoids SIGTRAP); see calcsky_subprocess_env."""

    def test_darwin_forces_omp_one_even_if_parent_set_omp(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        monkeypatch.delenv("HST123_CALCSKY_OMP_THREADS", raising=False)
        monkeypatch.setattr(sys, "platform", "darwin")
        e = calcsky_subprocess_env()
        assert e["OMP_NUM_THREADS"] == "1"

    def test_darwin_respects_hst123_calcsky_omp_threads(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        monkeypatch.setenv("HST123_CALCSKY_OMP_THREADS", "2")
        monkeypatch.setattr(sys, "platform", "darwin")
        e = calcsky_subprocess_env()
        assert e["OMP_NUM_THREADS"] == "2"

    def test_linux_delegates_to_dolphot_env(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv("HST123_DOLPHOT_OMP_THREADS", raising=False)
        monkeypatch.setattr(sys, "platform", "linux")
        e = calcsky_subprocess_env()
        assert "OMP_NUM_THREADS" not in e


class TestDolphotRequiredScripts:
    """DOLPHOT_REQUIRED_SCRIPTS is the single source of truth for pipeline and tests."""

    def test_list_contains_all_expected_executables(self):
        expected = [
            "dolphot",
            "calcsky",
        ]
        assert DOLPHOT_REQUIRED_SCRIPTS == expected

    @pytest.mark.dolphot
    def test_all_scripts_on_path_when_dolphot_installed(self, require_dolphot):
        """When DOLPHOT is installed, every required script is found on PATH."""
        for name in DOLPHOT_REQUIRED_SCRIPTS:
            path = shutil.which(name)
            assert path, f"{name} not on PATH"
        prim = DolphotPrimitive(object())
        assert prim.check_for_dolphot() is True


class TestCheckForDolphot:
    def test_returns_true_when_all_scripts_found(self):
        prim = DolphotPrimitive(object())
        with patch("shutil.which", return_value="/usr/bin/dolphot"):
            assert prim.check_for_dolphot() is True

    def test_returns_false_when_any_script_missing(self):
        prim = DolphotPrimitive(object())
        call_count = [0]

        def which(s):
            call_count[0] += 1
            return None if s == "calcsky" else f"/usr/bin/{s}"

        with patch("shutil.which", side_effect=which):
            assert prim.check_for_dolphot() is False
        # Stops at first missing (calcsky is second in list)
        assert call_count[0] >= 1


class TestMakeDolphotDict:
    def test_returns_dict_with_expected_keys(self):
        prim = DolphotPrimitive(object())
        out = prim.make_dolphot_dict("mydolphot")
        assert "base" in out
        assert "param" in out
        assert "colfile" in out
        assert "log" in out
        assert "final_phot" in out
        assert "radius" in out
        assert "limit_radius" in out

    def test_base_and_param_suffixes(self):
        prim = DolphotPrimitive(object())
        out = prim.make_dolphot_dict("mydolphot")
        assert out["base"] == "mydolphot"
        assert out["param"] == "mydolphot.param"
        assert out["colfile"] == "mydolphot.columns"
        assert out["final_phot"] == "mydolphot.phot"
        assert out["radius"] == 12
        assert out["limit_radius"] == 10.0

    def test_outputs_go_under_workdir_dolphot(self, tmp_path):
        prim = DolphotPrimitive(object())
        out = prim.make_dolphot_dict("dp0000", work_dir=str(tmp_path))
        expected_dir = tmp_path / "dolphot"
        assert expected_dir.is_dir()
        assert out["base"] == str(expected_dir / "dp0000")
        assert out["param"] == str(expected_dir / "dp0000.param")
        assert out["log"] == str(expected_dir / "dp0000.output")
        assert out["final_phot"] == str(expected_dir / "dp0000.phot")


class TestNeedsToCalcSky:
    def test_returns_true_when_no_sky_file_exists(self, tmp_path):
        prim = DolphotPrimitive(object())
        image = str(tmp_path / "img.fits")
        fits.PrimaryHDU().writeto(image, overwrite=True)
        assert prim.needs_to_calc_sky(image) is True

    def test_sky_path_uses_trailing_fits_only(self, tmp_path):
        """Regression: dirname must not break .fits -> .sky.fits replacement."""
        sub = tmp_path / "foo.fits"
        sub.mkdir()
        image = sub / "bar.drz.fits"
        fits.PrimaryHDU().writeto(str(image), overwrite=True)
        prim = DolphotPrimitive(object())
        assert prim.needs_to_calc_sky(str(image)) is True
        sky = sub / "bar.drz.sky.fits"
        fits.PrimaryHDU().writeto(str(sky), overwrite=True)
        assert prim.needs_to_calc_sky(str(image), check_wcs=False) is False

    def test_returns_false_when_sky_file_exists_and_check_wcs_false(
        self, tmp_path
    ):
        prim = DolphotPrimitive(object())
        image = tmp_path / "img.fits"
        fits.PrimaryHDU().writeto(str(image), overwrite=True)
        sky = tmp_path / "img.sky.fits"
        fits.PrimaryHDU().writeto(str(sky), overwrite=True)
        assert prim.needs_to_calc_sky(str(image), check_wcs=False) is False

    def test_returns_true_when_sky_exists_but_wcs_differs(self, tmp_path):
        prim = DolphotPrimitive(object())
        image = tmp_path / "img.fits"
        hdu = fits.PrimaryHDU(np.zeros((100, 100)))
        hdu.writeto(str(image), overwrite=True)
        sky = tmp_path / "img.sky.fits"
        hdu2 = fits.PrimaryHDU(np.zeros((50, 50)))
        hdu2.writeto(str(sky), overwrite=True)
        # Different NAXIS1/NAXIS2 so check_wcs finds a difference -> returns False
        assert prim.needs_to_calc_sky(str(image), check_wcs=True) is False


class TestCalcSkyPythonFallback:
    def test_uses_fallback_when_calcsky_returns_negative_exit(self, tmp_path):
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "acs_wfc_full"
        prim = DolphotPrimitive(mock_pipeline)
        img = tmp_path / "acs.f555w.test.drz.fits"
        fits.PrimaryHDU(np.zeros((32, 24), dtype=np.float32)).writeto(
            str(img), overwrite=True
        )
        options = {
            "acs_wfc": {
                "dolphot_sky": {
                    "r_in": 15,
                    "r_out": 35,
                    "step": 4,
                    "sigma_low": 2.25,
                    "sigma_high": 2.0,
                }
            }
        }
        fake_cp = MagicMock()
        fake_cp.returncode = -5
        with patch(
            "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command",
            return_value=fake_cp,
        ):
            prim.calc_sky(str(img), options)
        sky = tmp_path / "acs.f555w.test.drz.sky.fits"
        assert sky.is_file()


class TestNeedsToBeMasked:
    def test_returns_true_when_no_dol_header(self, minimal_fits_file):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "WFC3_UVIS"
        prim = DolphotPrimitive(mock_pipeline)
        assert prim.needs_to_be_masked(minimal_fits_file) is True

    def test_returns_false_when_dol_wfc3_zero(self, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "WFC3_UVIS"
        path = tmp_path / "img.fits"
        hdu = fits.PrimaryHDU()
        hdu.header["DOL_WFC3"] = 0
        hdu.writeto(str(path), overwrite=True)
        prim = DolphotPrimitive(mock_pipeline)
        assert prim.needs_to_be_masked(str(path)) is False

    def test_returns_true_when_dol_acs_nonzero(self, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "ACS_WFC"
        path = tmp_path / "img.fits"
        hdu = fits.PrimaryHDU()
        hdu.header["DOL_ACS"] = 1
        hdu.writeto(str(path), overwrite=True)
        prim = DolphotPrimitive(mock_pipeline)
        assert prim.needs_to_be_masked(str(path)) is True

    def test_dolphot_mask_markers_done_in_primary_matches_wfc3(self, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "WFC3_UVIS"
        path = tmp_path / "img.fits"
        hdu = fits.PrimaryHDU()
        hdu.header["DOL_WFC3"] = 0
        hdu.writeto(str(path), overwrite=True)
        prim = DolphotPrimitive(mock_pipeline)
        assert prim.dolphot_mask_markers_done_in_primary(str(path)) is True


class TestGenerateBaseParamFile:
    def test_writes_nimg_and_dolphot_params(self):
        mock_pipeline = object()
        prim = DolphotPrimitive(mock_pipeline)
        buf = StringIO()
        options = {"dolphot": {"SigFind": 2.5, "SigFinal": 3.0}}
        prim.generate_base_param_file(buf, options, 3)
        content = buf.getvalue()
        assert "Nimg = 3" in content
        assert "SigFind = 2.5" in content
        assert "SigFinal = 3.0" in content


class TestMakeDolphotFileNimg:
    """Nimg counts only science images; img0000 reference is separate."""

    def test_nimg_is_reference_plus_chip_count(self, tmp_path):
        ref = tmp_path / "ref.drc.fits"
        ref.write_bytes(b"SIMPLE")
        chip1 = tmp_path / "a.chip1.fits"
        chip2 = tmp_path / "a.chip2.fits"
        chip1.write_bytes(b"SIMPLE")
        chip2.write_bytes(b"SIMPLE")
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "ACS_WFC"
        mock_pipeline.options = {
            "args": MagicMock(work_dir=str(tmp_path)),
            "detector_defaults": {
                "acs_wfc": {"dolphot": {"RAper": 2, "RPSF": 10}},
            },
            "global_defaults": {"dolphot": {"SigFind": 2.5, "SigFinal": 3.0}},
        }
        mock_pipeline.dolphot = {
            "param": "dp0000.param",
            "base": "dp0000",
            "log": "dp0000.output",
            "original": "dp0000.orig",
        }
        prim = DolphotPrimitive(mock_pipeline)
        prim.make_dolphot_file([str(chip1), str(chip2)], str(ref))
        text = (tmp_path / "dp0000.param").read_text(encoding="utf-8")
        assert "Nimg = 2" in text
        assert "img0000_file = " in text
        assert "img0001_file = " in text
        assert "img0002_file = " in text


class TestGetDolphotInstrumentParameters:
    def test_returns_detector_dolphot_options(self):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "wfc3_uvis"
        options = {
            "wfc3_uvis": {"dolphot": {"RAper": 3, "RPSF": 15}},
            "acs_wfc": {"dolphot": {"RAper": 4}},
        }
        prim = DolphotPrimitive(mock_pipeline)
        out = prim.get_dolphot_instrument_parameters("any.fits", options)
        assert out == {"RAper": 3, "RPSF": 15}


class TestAddImageToParamFile:
    def test_writes_image_file_and_params(self, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "wfc3_uvis"
        options = {"wfc3_uvis": {"dolphot": {"RAper": 3}}}
        prim = DolphotPrimitive(mock_pipeline)
        path = tmp_path / "j9abc01dq_flc.fits"
        path.write_bytes(b"SIMPLE")
        buf = StringIO()
        prim.add_image_to_param_file(buf, str(path), 1, options)
        content = buf.getvalue()
        stem = os.path.splitext(os.path.abspath(str(path)))[0]
        assert f"img0001_file = {stem}" in content
        assert "img0001_RAper = 3" in content


class TestRunDolphot:
    def test_logs_error_when_param_file_missing(self, caplog):
        import logging
        mock_pipeline = MagicMock()
        mock_pipeline.dolphot = {
            "param": "/nonexistent/dolphot.param",
            "base": "/nonexistent/dolphot",
            "log": "/nonexistent/dolphot.output",
        }
        prim = DolphotPrimitive(mock_pipeline)
        caplog.set_level(logging.ERROR)
        prim.run_dolphot()
        assert "parameter file" in caplog.text.lower() or "does not exist" in caplog.text

    def test_does_not_remove_base_when_param_missing(self):
        mock_pipeline = MagicMock()
        mock_pipeline.dolphot = {
            "param": "/nonexistent/dolphot.param",
            "base": "/nonexistent/dolphot",
            "log": "/nonexistent/dolphot.output",
        }
        prim = DolphotPrimitive(mock_pipeline)
        with patch("os.path.isfile", return_value=False):
            with patch("os.remove") as rm:
                prim.run_dolphot()
                rm.assert_not_called()

    def test_run_dolphot_cleanup_removes_drc_noise_fits(self, tmp_path):
        """Ephemeral *.drc.noise.fits (sky sidecar) is swept after dolphot."""
        param = tmp_path / "dolphot.param"
        param.write_text("Nimg = 1\n", encoding="utf-8")
        base = tmp_path / "phot"
        base.write_text("", encoding="utf-8")
        logf = tmp_path / "phot.output"
        logf.write_text("", encoding="utf-8")
        noise = tmp_path / "stack.drc.noise.fits"
        noise.write_bytes(b"x")
        mock_pipeline = MagicMock()
        mock_pipeline.dolphot = {
            "param": str(param),
            "base": str(base),
            "log": str(logf),
            "original": str(tmp_path / "phot.orig"),
        }
        args = MagicMock()
        args.work_dir = str(tmp_path)
        mock_pipeline.options = {"args": args}
        prim = DolphotPrimitive(mock_pipeline)
        with patch(
            "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command",
        ), patch("hst123.primitives.run_dolphot.run_dolphot_primitive.time.sleep"):
            prim.run_dolphot()
        assert not noise.is_file()


class TestMaskImage:
    """acsmask / *mask invocation list (DQ path + ACS PAM preflight)."""

    def test_acs_passes_image_only_when_no_external_dq(self, minimal_fits_file):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_dq_image.return_value = ""
        prim = DolphotPrimitive(mock_pipeline)
        with patch(
            "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command"
        ) as run_ext:
            with patch(
                "hst123.dolphot_install.verify_acs_wfc_pam_files",
                return_value=(True, []),
            ):
                with patch(
                    "hst123.utils.dolphot_mask.apply_dolphot_mask_instrument",
                ) as py_mask:
                    prim.mask_image(str(minimal_fits_file), "acs")
        py_mask.assert_called_once()
        run_ext.assert_not_called()

    def test_acs_falls_back_to_external_when_python_mask_fails(
        self, minimal_fits_file, monkeypatch
    ):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_dq_image.return_value = ""
        prim = DolphotPrimitive(mock_pipeline)
        monkeypatch.delenv("HST123_DOLPHOT_MASK_EXTERNAL", raising=False)

        def _boom(*_a, **_k):
            raise RuntimeError("no tree")

        with patch(
            "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command"
        ) as run_ext:
            with patch(
                "hst123.dolphot_install.verify_acs_wfc_pam_files",
                return_value=(True, []),
            ):
                with patch(
                    "hst123.utils.dolphot_mask.apply_dolphot_mask_instrument",
                    side_effect=_boom,
                ):
                    prim.mask_image(str(minimal_fits_file), "acs")
        run_ext.assert_called_once()
        (cmd,), _ = run_ext.call_args
        assert cmd == ["acsmask", str(minimal_fits_file)]

    def test_acs_raises_before_acsmask_when_pam_missing(self, minimal_fits_file):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        prim = DolphotPrimitive(mock_pipeline)
        with patch(
            "hst123.dolphot_install.verify_acs_wfc_pam_files",
            return_value=(False, ["missing /tmp/wfc2_pam.fits"]),
        ):
            with pytest.raises(RuntimeError, match="WFC PAM"):
                prim.mask_image(str(minimal_fits_file), "acs")

    def test_wfpc2_appends_c1m_when_get_dq_image_returns_existing_file(
        self, minimal_fits_file, tmp_path, monkeypatch
    ):
        dq = tmp_path / "dq.fits"
        dq.write_bytes(b"simple")
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_dq_image.return_value = str(dq)
        prim = DolphotPrimitive(mock_pipeline)
        monkeypatch.setenv("HST123_DOLPHOT_MASK_EXTERNAL", "1")
        with patch(
            "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command"
        ) as run_ext:
            prim.mask_image(str(minimal_fits_file), "wfpc2")
        (cmd,), _ = run_ext.call_args
        assert cmd[0] == "wfpc2mask"
        assert cmd[1] == str(minimal_fits_file)
        assert cmd[2] == str(dq)


class TestPrepareDolphot:
    def test_returns_empty_list_when_no_chip_files(self, minimal_fits_file):
        mock_pipeline = MagicMock()
        mock_pipeline._fits = MagicMock()
        mock_pipeline._fits.get_instrument.return_value = "WFC3_UVIS"
        mock_pipeline.options = {"args": MagicMock(include_all_splits=False)}
        mock_pipeline.coord = None
        mock_pipeline.split_image_contains.return_value = False
        prim = DolphotPrimitive(mock_pipeline)
        with patch.object(prim, "needs_to_be_masked", return_value=False), patch.object(
            prim, "needs_to_split_groups", return_value=False
        ), patch("glob.glob", return_value=[]):
            out = prim.prepare_dolphot(minimal_fits_file)
        assert out == []


class TestGetDolphotPhotometry:
    def test_logs_warning_when_base_missing(self, caplog):
        from astropy.coordinates import SkyCoord
        import logging
        mock_pipeline = MagicMock()
        mock_pipeline.coord = SkyCoord(0.0, 0.0, unit="deg")
        mock_pipeline.dolphot = {
            "base": "/nonexistent/base",
            "colfile": "/nonexistent/columns",
        }
        mock_pipeline.options = {"args": MagicMock(scrape_all=False)}
        prim = DolphotPrimitive(mock_pipeline)
        caplog.set_level(logging.WARNING)
        with patch("os.path.exists", return_value=False):
            prim.get_dolphot_photometry([], "/ref.fits")
        assert "dolphot" in caplog.text.lower() or "run" in caplog.text.lower()


class TestDoFake:
    def test_returns_none_when_base_missing(self):
        mock_pipeline = MagicMock()
        mock_pipeline.dolphot = {"base": "/nonexistent/base"}
        prim = DolphotPrimitive(mock_pipeline)
        with patch("os.path.exists", return_value=False):
            result = prim.do_fake(
                [], "/ref.fits"
            )
        assert result is None

    def test_returns_none_when_base_empty(self, tmp_path):
        base = tmp_path / "base"
        base.write_bytes(b"")
        mock_pipeline = MagicMock()
        mock_pipeline.dolphot = {"base": str(base)}
        mock_pipeline.coord = None
        prim = DolphotPrimitive(mock_pipeline)
        with patch("os.path.exists", return_value=True):
            result = prim.do_fake([], str(tmp_path / "ref.fits"))
        assert result is None
