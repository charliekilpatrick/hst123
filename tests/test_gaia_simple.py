"""Unit tests for gaia_simple epoch correction and quality cuts."""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from hst123.primitives.astrometry import gaia_simple as gs


def test_apply_gaia_quality_cuts_skipped_when_few_stars():
    t = Table()
    t["ra"] = [1.0, 2.0, 3.0]
    t["dec"] = [0.0, 0.0, 0.0]
    t["mag"] = [12.0, 18.0, 25.0]  # would cut mag extremes if applied
    t["ruwe"] = [2.0, 1.0, 1.0]
    t["astrometric_excess_noise"] = [0.1, 0.1, 0.1]
    out, meta = gs.apply_gaia_quality_cuts(t)
    assert meta["applied"] is False
    assert len(out) == 3


def test_apply_gaia_quality_cuts_when_many_stars():
    n = 12
    t = Table()
    t["ra"] = np.linspace(1.0, 2.0, n)
    t["dec"] = np.zeros(n)
    t["mag"] = np.full(n, 18.0)
    t["ruwe"] = np.full(n, 1.0)
    t["astrometric_excess_noise"] = np.full(n, 0.2)
    # Contaminate a few rows
    t["ruwe"][0] = 2.5
    t["astrometric_excess_noise"][1] = 5.0
    t["mag"][2] = 12.0  # too bright
    t["mag"][3] = 22.0  # too faint
    t["ruwe"][4] = np.nan  # missing RUWE should still pass
    out, meta = gs.apply_gaia_quality_cuts(t)
    assert meta["applied"] is True
    assert meta["n_in"] == 12
    assert meta["n_out"] == 8
    assert len(out) == 8


def test_apply_gaia_quality_cuts_strict_rejects_borderline_ruwe():
    t = Table()
    t["ra"] = np.arange(10, dtype=float)
    t["dec"] = np.zeros(10)
    t["mag"] = np.full(10, 17.0)
    t["ruwe"] = np.full(10, 1.0)
    t["astrometric_excess_noise"] = np.full(10, 0.2)
    t["ruwe"][0:3] = 1.5  # passes relaxed 1.6, fails preferred 1.4
    out, meta = gs.apply_gaia_quality_cuts(t)  # preferred / strict defaults
    assert meta["applied"] is True
    assert meta["n_out"] == 7


def test_select_best_calibrators_keeps_top_snr():
    matches = [
        gs.GaiaSimpleMatch(0, 0, 0, 0, 0, 0, 0, 0, flux=1.0, bkg=0, n_good=5, snr=s)
        for s in (2.0, 10.0, 5.0, 8.0, 1.0)
    ]
    best = gs.select_best_calibrators(matches, max_n=3)
    assert len(best) == 3
    assert [m.snr for m in best] == [10.0, 8.0, 5.0]


def test_apply_gaia_quality_cuts_reverts_if_too_few_remain():
    t = Table()
    t["ra"] = np.arange(8, dtype=float)
    t["dec"] = np.zeros(8)
    t["mag"] = np.full(8, 18.0)
    t["ruwe"] = np.full(8, 3.0)  # all fail RUWE
    t["astrometric_excess_noise"] = np.full(8, 0.1)
    out, meta = gs.apply_gaia_quality_cuts(t)
    assert meta["applied"] is False
    assert len(out) == 8


def test_propagate_gaia_to_obstime_moves_high_pm_star():
    t = Table()
    t["ra"] = [180.0]
    t["dec"] = [0.0]
    t["pmra"] = [100.0]  # mas/yr
    t["pmdec"] = [0.0]
    t["parallax"] = [10.0]  # mas
    # 5 years after Gaia DR3 epoch
    obstime = Time(2021.0, format="jyear", scale="tcb")
    out, meta = gs.propagate_gaia_to_obstime(t, obstime)
    assert meta["applied"] is True
    assert meta["n_pm"] == 1
    assert meta["n_parallax"] == 1
    # ~100 mas/yr * 5 yr = 0.5 arcsec = 0.5/3600 deg along RA*cos(dec)
    dra_as = (float(out["ra"][0]) - 180.0) * 3600.0
    assert abs(dra_as - 0.5) < 0.05


def test_propagate_gaia_skips_without_pm_or_epoch():
    t = Table()
    t["ra"] = [10.0]
    t["dec"] = [20.0]
    out, meta = gs.propagate_gaia_to_obstime(t, Time(2020.0, format="jyear"))
    assert meta["applied"] is False
    assert float(out["ra"][0]) == 10.0

    t2 = Table()
    t2["ra"] = [10.0]
    t2["dec"] = [20.0]
    t2["pmra"] = [1.0]
    t2["pmdec"] = [1.0]
    out2, meta2 = gs.propagate_gaia_to_obstime(t2, None)
    assert meta2["applied"] is False


def test_hst_obstime_from_hdul_prefers_mjd_avg():
    phdu = fits.PrimaryHDU()
    phdu.header["MJD-AVG"] = 59315.5
    phdu.header["EXPSTART"] = 59315.0
    phdu.header["EXPEND"] = 59316.0
    hdul = fits.HDUList([phdu])
    t = gs.hst_obstime_from_hdul(hdul)
    assert t is not None
    assert abs(float(t.mjd) - 59315.5) < 1e-6
