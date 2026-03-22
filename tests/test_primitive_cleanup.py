"""Tests for hst123.primitives.primitive_cleanup helpers."""
import logging

import numpy as np
from astropy.io import fits

from hst123.primitives.primitive_cleanup import (
    remove_interstitial_files,
    validate_fits_outputs,
    validate_text_outputs,
)


def test_remove_interstitial_files_respects_skip(tmp_path):
    log = logging.getLogger("t_rm")
    f = tmp_path / "a.txt"
    f.write_text("x")
    n, _ = remove_interstitial_files(
        str(tmp_path),
        (),
        (str(f),),
        log,
        step_name="test",
        skip_removal=True,
    )
    assert n == 0
    assert f.is_file()


def test_remove_interstitial_files_explicit(tmp_path):
    log = logging.getLogger("t_rm2")
    f = tmp_path / "b.txt"
    f.write_text("y")
    n, names = remove_interstitial_files(
        str(tmp_path),
        (),
        (str(f),),
        log,
        step_name="test",
        skip_removal=False,
    )
    assert n == 1
    assert not f.is_file()


def test_validate_fits_outputs_ok(tmp_path):
    log = logging.getLogger("t_vf")
    img = tmp_path / "x.fits"
    fits.PrimaryHDU(np.zeros((4, 5), dtype=np.float32)).writeto(str(img))
    ok, n = validate_fits_outputs([str(img)], log, step_name="t")
    assert n == 1 and ok == 1


def test_validate_text_outputs(tmp_path):
    log = logging.getLogger("t_vt")
    p = tmp_path / "t.dat"
    p.write_text("hello")
    ok, n = validate_text_outputs([str(p)], log, step_name="t", min_size=1)
    assert ok == 1 and n == 1
