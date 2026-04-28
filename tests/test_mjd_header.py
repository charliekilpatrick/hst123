"""Tests for :mod:`hst123.utils.mjd_header`."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from hst123.utils.mjd_header import (
    ensure_mjd_avg_on_pipeline_science_image,
    mjd_avg_from_primary_header,
)


def test_mjd_avg_from_expstart_expend():
    h = fits.Header()
    h["EXPSTART"] = 60000.0
    h["EXPEND"] = 60000.5
    assert mjd_avg_from_primary_header(h) == pytest.approx(60000.25)


def test_mjd_avg_expstart_only():
    h = fits.Header()
    h["EXPSTART"] = 59000.0
    assert mjd_avg_from_primary_header(h) == pytest.approx(59000.0)


def test_ensure_mjd_avg_writes_flc(tmp_path: Path):
    p = tmp_path / "j9test01_flc.fits"
    h = fits.Header()
    h["EXPSTART"] = 58000.0
    h["EXPEND"] = 58000.2
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32), header=h).writeto(
        p, overwrite=True
    )
    assert ensure_mjd_avg_on_pipeline_science_image(str(p))
    with fits.open(p) as hdul:
        assert hdul[0].header["MJD-AVG"] == pytest.approx(58000.1)


def test_skips_chip_file(tmp_path: Path):
    p = tmp_path / "j9test01_flc.chip1.fits"
    h = fits.Header()
    h["EXPSTART"] = 58000.0
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32), header=h).writeto(
        p, overwrite=True
    )
    assert not ensure_mjd_avg_on_pipeline_science_image(str(p))
    with fits.open(p) as hdul:
        assert "MJD-AVG" not in hdul[0].header
