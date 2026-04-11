"""Tests for TweakReg shift validation helpers."""

import logging

import numpy as np
import pytest
from astropy.table import Table

from hst123.utils import alignment_validation as av


def test_log_tweakreg_shift_metrics_subpixel_pass(caplog):
    caplog.set_level(logging.INFO)
    t = Table(
        {
            "xoffset": [0.1, -0.2, 0.05],
            "yoffset": [0.15, 0.1, -0.08],
        }
    )
    log = logging.getLogger("test_alignment")
    out = av.log_tweakreg_shift_metrics(
        t,
        ref_path="/nonexistent/ref.fits",
        log=log,
        tolerance_arcsec=0.25,
        batch_index=0,
    )
    assert out["n"] == 3
    assert out["subpixel_pass"] is True
    assert out["max_hypot_px"] == pytest.approx(np.max(np.hypot([0.1, -0.2, 0.05], [0.15, 0.1, -0.08])))
    assert "PASS" in caplog.text


def test_log_tweakreg_shift_metrics_warns_large_shift(caplog):
    caplog.set_level(logging.WARNING)
    t = Table({"xoffset": [10.0], "yoffset": [0.0]})
    log = logging.getLogger("test_alignment2")
    out = av.log_tweakreg_shift_metrics(
        t,
        ref_path="/nonexistent/ref.fits",
        log=log,
        batch_index=1,
    )
    assert out["subpixel_pass"] is False
    assert "CHECK" in caplog.text
    assert "Large shift magnitude" in caplog.text


def test_log_tweakreg_shift_metrics_empty():
    log = logging.getLogger("test_alignment3")
    out = av.log_tweakreg_shift_metrics(
        Table({"xoffset": [], "yoffset": []}),
        ref_path="x.fits",
        log=log,
    )
    assert out["n"] == 0


def test_summary_prefix_overrides_batch_index(caplog):
    caplog.set_level(logging.INFO)
    t = Table({"xoffset": [0.01], "yoffset": [0.02]})
    log = logging.getLogger("test_alignment4")
    av.log_tweakreg_shift_metrics(
        t,
        ref_path="/nonexistent/ref.fits",
        log=log,
        batch_index=99,
        summary_prefix="TweakReg aggregate",
    )
    assert "TweakReg aggregate:" in caplog.text
    assert "batch 99" not in caplog.text
