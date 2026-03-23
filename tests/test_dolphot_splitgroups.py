"""Tests for hst123.utils.dolphot_splitgroups."""
import glob
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from unittest.mock import patch

from hst123.utils.dolphot_splitgroups import (
    apply_splitgroups,
    count_expected_split_outputs,
    SplitgroupsError,
)


def _two_sci_mef(path: Path) -> None:
    p = fits.PrimaryHDU()
    p.header["INSTRUME"] = "ACS"
    p.header["PHOTFLAM"] = 1.0
    d1 = np.ones((10, 12), dtype=np.float32) * 1.0
    d2 = np.ones((10, 12), dtype=np.float32) * 2.0
    h1 = fits.ImageHDU(data=d1, name="SCI")
    h1.header["CRVAL1"] = 10.0
    h1.header["CCDCHIP"] = 1
    h2 = fits.ImageHDU(data=d2, name="SCI")
    h2.header["CRVAL1"] = 20.0
    h2.header["CCDCHIP"] = 2
    fits.HDUList([p, h1, h2]).writeto(path, overwrite=True)


def test_count_expected_mef_and_3d(tmp_path):
    mef = tmp_path / "m.fits"
    _two_sci_mef(mef)
    assert count_expected_split_outputs(mef) == 2

    cube = tmp_path / "w.fits"
    data = np.zeros((4, 8, 8), dtype=np.float32)
    ph = fits.PrimaryHDU(data=data)
    ph.header["INSTRUME"] = "WFPC2"
    fits.HDUList([ph]).writeto(cube, overwrite=True)
    assert count_expected_split_outputs(cube) == 4


def test_apply_splitgroups_two_sci(tmp_path):
    mef = tmp_path / "raw.fits"
    _two_sci_mef(mef)
    out = apply_splitgroups(mef)
    assert len(out) == 2
    assert (tmp_path / "raw.chip1.fits").is_file()
    assert (tmp_path / "raw.chip2.fits").is_file()
    for p in out:
        with fits.open(p) as hdul:
            assert hdul[0].name == "SCI"
            assert hdul[0].data.shape == (10, 12)
    with fits.open(out[0]) as h:
        assert float(h[0].header["CRVAL1"]) == 10.0
        assert h[0].header["INSTRUME"] == "ACS"
    with fits.open(out[1]) as h:
        assert float(h[0].header["CRVAL1"]) == 20.0


def test_apply_splitgroups_3d_primary(tmp_path):
    cube = tmp_path / "w.fits"
    data = np.arange(4 * 5 * 5, dtype=np.float32).reshape(4, 5, 5)
    ph = fits.PrimaryHDU(data=data)
    ph.header["INSTRUME"] = "WFPC2"
    fits.HDUList([ph]).writeto(cube, overwrite=True)
    out = apply_splitgroups(cube)
    assert len(out) == 4
    with fits.open(out[0]) as h:
        assert h[0].name == "SCI"
        assert h[0].data.shape == (5, 5)


def test_apply_splitgroups_raises_when_nothing_to_split(tmp_path):
    p = tmp_path / "empty.fits"
    fits.PrimaryHDU(data=np.zeros((3, 3), dtype=np.float32)).writeto(p, overwrite=True)
    with pytest.raises(SplitgroupsError, match="No SCI"):
        apply_splitgroups(p)


def test_needs_to_split_groups_uses_expected_count(minimal_fits_file):
    from hst123.primitives.run_dolphot import DolphotPrimitive

    prim = DolphotPrimitive(object())
    assert prim.needs_to_split_groups(minimal_fits_file) is True
    apply_splitgroups(minimal_fits_file)
    assert prim.needs_to_split_groups(minimal_fits_file) is False


def test_split_groups_python_path_no_external(tmp_path):
    from hst123.primitives.run_dolphot import DolphotPrimitive

    mef = tmp_path / "x.fits"
    _two_sci_mef(mef)
    prim = DolphotPrimitive(object())
    with patch(
        "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command"
    ) as run_ext:
        prim.split_groups(str(mef), delete_non_science=False)
    run_ext.assert_not_called()
    assert len(glob.glob(str(tmp_path / "x.chip?.fits"))) == 2


def test_split_groups_falls_back_on_python_failure(tmp_path, monkeypatch):
    from hst123.primitives.run_dolphot import DolphotPrimitive

    mef = tmp_path / "x.fits"
    _two_sci_mef(mef)
    monkeypatch.delenv("HST123_DOLPHOT_SPLITGROUPS_EXTERNAL", raising=False)
    prim = DolphotPrimitive(object())

    def _boom(*_a, **_k):
        raise RuntimeError("fail")

    with patch(
        "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command"
    ) as run_ext:
        with patch(
            "hst123.utils.dolphot_splitgroups.apply_splitgroups",
            side_effect=_boom,
        ):
            prim.split_groups(str(mef), delete_non_science=False)
    run_ext.assert_called_once()
    assert run_ext.call_args[0][0] == ["splitgroups", str(mef)]
