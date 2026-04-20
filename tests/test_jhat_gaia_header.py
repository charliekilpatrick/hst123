"""JHAT Gaia residual stats and CRDER* keywords on reference drizzle."""
import logging
import os

import numpy as np
from astropy.io import fits

from hst123.primitives.astrometry import jhat as jhat_mod
from hst123.utils.alignment_validation import (
    aggregate_jhat_gaia_stats,
    write_gaia_alignment_crderr_to_reference_driz,
)


def test_aggregate_jhat_gaia_stats_weighted(tmp_path):
    s = [
        {"n_match": 4, "rms_ra_as": 0.04, "rms_dec_as": 0.03, "rms_sky_as": 0.05},
        {"n_match": 6, "rms_ra_as": 0.02, "rms_dec_as": 0.02, "rms_sky_as": 0.028},
    ]
    agg = aggregate_jhat_gaia_stats(s)
    assert agg is not None
    assert agg["n_gaia"] == 10
    expect_ra = np.sqrt((4 * 0.04**2 + 6 * 0.02**2) / 10.0)
    assert abs(agg["rms_ra_deg"] * 3600.0 - expect_ra) < 1e-9


def test_read_jhat_gaia_residual_stats_from_ascii(tmp_path):
    wd = tmp_path / "ws"
    wd.mkdir()
    # Basename iecf02ysq_flt.fits -> iecf02ysq_jhat.good.phot.txt
    phot = wd / "iecf02ysq_jhat.good.phot.txt"
    phot.write_text(
        "ra dec gaia_ra gaia_dec\n"
        "10.0 20.0 10.0000277778 20.0000277778\n"
        "10.0 20.0 10.0000555556 20.0000555556\n",
        encoding="utf-8",
    )
    img = tmp_path / "iecf02ysq_flt.fits"
    img.write_text("", encoding="utf-8")
    st = jhat_mod.read_jhat_gaia_residual_stats(str(img), str(wd))
    assert st is not None
    assert st["n_match"] == 2
    assert st["rms_ra_as"] > 0
    assert st["rms_dec_as"] > 0


def test_write_gaia_alignment_crderr_to_reference_driz(tmp_path):
    path = tmp_path / "acs_wfc.f555w.ref_0001.drc.fits"
    pri = fits.PrimaryHDU()
    pri.header["NINPUT"] = (1, "test")
    pri.header["INPUT"] = ("dummy.fits", "test")
    fits.HDUList([pri]).writeto(str(path))
    stats = [
        {
            "n_match": 8,
            "rms_ra_as": 0.05,
            "rms_dec_as": 0.06,
            "rms_sky_as": 0.078,
        }
    ]
    ok = write_gaia_alignment_crderr_to_reference_driz(
        str(path), stats, log=logging.getLogger("test_jhat_hdr")
    )
    assert ok
    with fits.open(str(path)) as hdul:
        h = hdul[0].header
        assert "CRDER1" in h
        assert "CRDER2" in h
        assert h["CRDER1"] > 0
        assert h["CRDER2"] > 0
        assert h["HIERARCH HST123 NJHATGAIA"] == 8


def test_jhat_good_phot_path():
    p = jhat_mod.jhat_gaia_good_phot_path("/data/iecf02ysq_flt.fits", "/tmp/w")
    assert p.endswith(os.path.join("w", "iecf02ysq_jhat.good.phot.txt"))
