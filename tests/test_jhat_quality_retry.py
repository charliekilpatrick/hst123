"""JHAT quality gate and relaxed-parameter retry overlays."""

from hst123.primitives.astrometry.jhat import (
    _merged_gaia_run_all_kw,
    jhat_alignment_acceptable,
    jhat_quality_retry_overlay,
    jhat_quality_retry_overlay_match_only,
)


def test_jhat_alignment_acceptable():
    assert not jhat_alignment_acceptable(None, max_rms_arcsec=2.0, min_matches=5)
    assert not jhat_alignment_acceptable(
        {"n_match": 4, "rms_sky_as": 0.5},
        max_rms_arcsec=2.0,
        min_matches=5,
    )
    assert jhat_alignment_acceptable(
        {"n_match": 10, "rms_sky_as": 1.5},
        max_rms_arcsec=2.0,
        min_matches=5,
    )
    assert not jhat_alignment_acceptable(
        {"n_match": 10, "rms_sky_as": 2.5},
        max_rms_arcsec=2.0,
        min_matches=5,
    )


def test_jhat_quality_retry_overlay_progression():
    assert jhat_quality_retry_overlay(0) == {}
    o1 = jhat_quality_retry_overlay(1)
    assert o1["d2d_max"] == 1.5
    assert "iterate_with_xyshifts" not in o1
    o2 = jhat_quality_retry_overlay(2)
    assert o2["d2d_max"] == 3.0
    assert o2.get("iterate_with_xyshifts") is True
    o3 = jhat_quality_retry_overlay(3)
    assert o3["d2d_max"] == 6.0
    # Beyond defined tiers, reuse strongest relaxation
    o9 = jhat_quality_retry_overlay(9)
    assert o9["d2d_max"] == o3["d2d_max"]


def test_jhat_quality_retry_overlay_match_only_strips_detection_relaxed_keys():
    assert jhat_quality_retry_overlay_match_only(0) == {}
    m1 = jhat_quality_retry_overlay_match_only(1)
    assert m1["d2d_max"] == 1.5
    assert "objmag_lim" not in m1
    assert "sharpness_lim" not in m1
    assert "dmag_max" not in m1
    m2 = jhat_quality_retry_overlay_match_only(2)
    assert m2["d2d_max"] == 3.0
    assert m2.get("iterate_with_xyshifts") is True
    assert "objmag_lim" not in m2


def test_merged_gaia_run_all_kw_anchor_keeps_mag_and_quality_across_retries():
    base = {
        "_hst123_gaia_anchor_match_only_retries": True,
        "refmag_lim": (15.0, 22.0),
        "objmag_lim": (15.0, 22.0),
    }
    m2 = _merged_gaia_run_all_kw(base, 2)
    assert m2["objmag_lim"] == (15.0, 22.0)
    assert m2["refmag_lim"] == (15.0, 22.0)
    assert m2["d2d_max"] == 3.0
    assert m2.get("iterate_with_xyshifts") is True
    # Without anchor flag, retry overlay relaxes objmag_lim etc.
    m2_loose = _merged_gaia_run_all_kw({"objmag_lim": (14, 24)}, 2)
    assert m2_loose["objmag_lim"] == (10, 28)
