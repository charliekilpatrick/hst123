"""JHAT encircled-energy download directory (stack + work-dir behavior)."""

import os

import pytest

from hst123.primitives.astrometry import jhat_wfpc2_patch as jwp


def test_ee_dir_stack_nested(tmp_path, tmp_path_factory):
    # conftest autouse already pushed tmp_path
    outer = os.path.abspath(str(tmp_path))
    assert jwp.get_jhat_ee_calibration_dir() == outer
    inner = tmp_path_factory.mktemp("inner_ws")
    inner_abs = os.path.abspath(str(inner))
    jwp.push_jhat_ee_calibration_dir(str(inner))
    assert jwp.get_jhat_ee_calibration_dir() == inner_abs
    jwp.pop_jhat_ee_calibration_dir()
    assert jwp.get_jhat_ee_calibration_dir() == outer


def test_urlretrieve_writes_under_pushed_dir(tmp_path, monkeypatch):
    """Encircled-energy download targets the stacked calibration dir, not cwd."""
    import urllib.request

    dests: list[str] = []

    def spy(_url, dest, *_a, **_k):
        dests.append(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Minimal grid: two wavelengths, two apertures, filter match for spline path.
        open(dest, "w", encoding="ascii").write(
            "FILTER WAVELENGTH #0.05 #0.1\n"
            "F814W 8000.0 1.0 1.0\n"
            "F814W 8100.0 1.0 1.0\n"
        )

    monkeypatch.setattr(urllib.request, "urlretrieve", spy)

    ee_dir = str(tmp_path / "ee_sub")
    os.makedirs(ee_dir, exist_ok=True)
    jwp.push_jhat_ee_calibration_dir(ee_dir)
    jwp.ensure_jhat_hst_phot_wfpc2_patch()

    import jhat.simple_jwst_phot as sjp

    uvis = os.path.join(ee_dir, "wfc3uvis2_aper_007_syn.csv")
    if os.path.isfile(uvis):
        os.remove(uvis)

    _ = sjp.hst_get_ee_corr(3.0, 0.05, "F814W", "uvis")

    assert dests, "expected urlretrieve to run for missing UVIS table"
    assert os.path.dirname(os.path.abspath(dests[0])) == os.path.abspath(ee_dir)

    jwp.pop_jhat_ee_calibration_dir()
