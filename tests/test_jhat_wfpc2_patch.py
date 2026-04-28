"""Tests for :mod:`hst123.primitives.astrometry.jhat_wfpc2_patch`."""

from astropy.io import fits

from hst123.primitives.astrometry.jhat_wfpc2_patch import wfpc2_filter_key_and_name


def test_wfpc2_filter_prefers_filtnam_over_numeric_filter1():
    h = fits.Header()
    h["INSTRUME"] = "WFPC2"
    h["FILTER1"] = 0  # triggers TypeError in unpatched JHAT: 'CLEAR' not in int
    h["FILTNAM1"] = "F300W"
    k, name = wfpc2_filter_key_and_name(h)
    assert k == "FILTNAM1"
    assert name == "F300W"


def test_wfpc2_filter_filtnam2_when_filtnam1_clear():
    h = fits.Header()
    h["FILTNAM1"] = "CLEAR"
    h["FILTNAM2"] = "F814W"
    k, name = wfpc2_filter_key_and_name(h)
    assert k == "FILTNAM2"
    assert name == "F814W"


def test_wfpc2_filter_fallback_clear():
    h = fits.Header()
    k, name = wfpc2_filter_key_and_name(h)
    assert k == "FILTNAM1"
    assert name == "CLEAR"
