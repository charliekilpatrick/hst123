"""Unit tests for hst123.utils.astrodrizzle_helpers."""
import logging
import os

import numpy as np
import pytest
from astropy.io import fits

from hst123.utils.astrodrizzle_helpers import (
    astrodrizzle_exc_is_missing_dq_extension,
    astrodrizzle_exc_is_restore_wcs_distortion_failure,
    astrodrizzle_wcskey_for_run,
    canonical_drizzle_input_stem,
    canonicalize_wfpc2_astrodrizzle_input_path,
    combine_type_and_nhigh,
    drizzle_canonical_weight_mask_paths,
    drizzle_reference_inputs_match,
    drizzle_sidecar_paths,
    ensure_flc_dq_err_extensions_inplace,
    flc_sci_extensions_missing_dq_err,
    is_hst123_wfpc2_astrodrizzle_scratch,
    resolve_drizzle_clean_flag,
    rename_astrodrizzle_sidecars,
    suppress_drizzlepac_interactive_dgeo_prompt,
    wfpc2_astrodrizzle_scratch_paths,
    wcs_image_hdu_index,
    write_drc_multis_extension_if_requested,
)


def test_suppress_drizzlepac_interactive_dgeo_prompt_restores_userstop():
    pytest.importorskip("drizzlepac")
    import drizzlepac.processInput as pi

    orig = pi.userStop
    with suppress_drizzlepac_interactive_dgeo_prompt():
        assert pi.userStop is not orig
        assert pi.userStop("any prompt") is False
    assert pi.userStop is orig


def test_astrodrizzle_wcskey_for_run():
    assert astrodrizzle_wcskey_for_run(skip_tweakreg=True) == " "
    assert astrodrizzle_wcskey_for_run(skip_tweakreg=False) == "TWEAK"


def test_astrodrizzle_exc_is_restore_wcs_distortion_failure():
    e = MemoryError(
        "NAXES was not set (or bad) for Lookup   distortion on axis 2"
    )
    assert astrodrizzle_exc_is_restore_wcs_distortion_failure(e)
    assert astrodrizzle_exc_is_restore_wcs_distortion_failure(
        KeyError("Keyword 'D2IM1.AXIS.1' not found.")
    )
    assert astrodrizzle_exc_is_restore_wcs_distortion_failure(
        KeyError("Extension ('D2IMARR', 1.0) not found.")
    )
    assert astrodrizzle_exc_is_restore_wcs_distortion_failure(
        KeyError("Keyword 'CD1_2' not found.")
    )
    assert not astrodrizzle_exc_is_restore_wcs_distortion_failure(ValueError("other"))
    assert not astrodrizzle_exc_is_restore_wcs_distortion_failure(MemoryError("out of memory"))


def test_astrodrizzle_exc_is_missing_dq_extension():
    assert astrodrizzle_exc_is_missing_dq_extension(ValueError("no extension number found"))
    assert not astrodrizzle_exc_is_missing_dq_extension(ValueError("other"))


def test_ensure_flc_dq_err_extensions_inplace(tmp_path):
    sci = np.ones((4, 5), dtype=np.float32)
    phdu = fits.PrimaryHDU()
    phdu.header["NEXTEND"] = 2
    hdul = fits.HDUList([phdu])
    for ev in (1, 2):
        h = fits.ImageHDU(data=sci.copy())
        h.header["EXTNAME"] = "SCI"
        h.header["EXTVER"] = ev
        hdul.append(h)
    path = tmp_path / "trunc_flc.fits"
    hdul.writeto(path)
    with fits.open(path) as ro:
        assert flc_sci_extensions_missing_dq_err(ro)
    assert ensure_flc_dq_err_extensions_inplace(path)
    with fits.open(path) as out:
        assert not flc_sci_extensions_missing_dq_err(out)
        assert len(out) == 7  # PRIMARY + 2×(SCI,ERR,DQ)
        names = [out[i].name for i in range(1, len(out))]
        assert names == ["SCI", "ERR", "DQ", "SCI", "ERR", "DQ"]


def test_canonical_drizzle_input_stem_wfpc2_scratch():
    s = "/w/u2460107t_hst123drz8297683a44664ac2_c0m.fits"
    assert canonical_drizzle_input_stem(s) == "u2460107t_c0m"
    assert canonical_drizzle_input_stem("u2460107t_c0m.fits") == "u2460107t_c0m"


def test_is_hst123_wfpc2_astrodrizzle_scratch():
    assert is_hst123_wfpc2_astrodrizzle_scratch("/x/u1_hst123drzab_c0m.fits")
    assert is_hst123_wfpc2_astrodrizzle_scratch("/x/u1_hst123drzab_c1m.fits")
    assert not is_hst123_wfpc2_astrodrizzle_scratch("/x/u1_c0m.fits")


def test_canonicalize_wfpc2_astrodrizzle_input_path(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    c0 = d / "u2460107t_c0m.fits"
    c0.write_text("x", encoding="ascii")
    scratch = d / "u2460107t_hst123drz111_c0m.fits"
    scratch.write_text("y", encoding="ascii")
    assert canonicalize_wfpc2_astrodrizzle_input_path(str(scratch)) == str(c0)
    assert canonicalize_wfpc2_astrodrizzle_input_path(str(c0)) == str(c0)


def test_drizzle_reference_inputs_match_scratch_vs_workspace():
    hdr = fits.Header()
    hdr["NINPUT"] = 2
    hdr["INPUT"] = (
        "/x/u2460107t_hst123drz111_c0m,/y/u2460108t_hst123drz222_c0m"
    )
    paths = ["/data/u2460108t_c0m.fits", "/data/u2460107t_c0m.fits"]
    assert drizzle_reference_inputs_match(paths, hdr)


def test_wfpc2_astrodrizzle_scratch_paths_pairs_c1m(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    c0 = d / "u2460107t_c0m.fits"
    c1 = d / "u2460107t_c1m.fits"
    c0.write_text("x")
    c1.write_text("y")
    t0, t1 = wfpc2_astrodrizzle_scratch_paths(str(c0), 12345)
    assert t0.endswith("u2460107t_hst123drz12345_c0m.fits")
    assert t1 is not None and t1.endswith("u2460107t_hst123drz12345_c1m.fits")


def test_combine_type_and_nhigh_small_stack():
    ct, nh = combine_type_and_nhigh(2, None)
    assert ct == "minmed" and nh == 0


def test_combine_type_and_nhigh_large_stack():
    ct, nh = combine_type_and_nhigh(10, None)
    assert ct == "median" and nh == 3


def test_combine_type_override():
    ct, nh = combine_type_and_nhigh(2, "median")
    assert ct == "median" and nh == 0


def test_resolve_drizzle_clean_flag():
    assert resolve_drizzle_clean_flag(None, True) is True
    assert resolve_drizzle_clean_flag(None, False) is False
    assert resolve_drizzle_clean_flag(True, False) is True
    assert resolve_drizzle_clean_flag(False, True) is False


def test_drizzle_sidecar_paths():
    sci, wht, ctx = drizzle_sidecar_paths("/tmp/out.drz.fits")
    assert sci.endswith("_sci.fits") and "out.drz" in sci
    assert wht.endswith("_wht.fits")
    assert ctx.endswith("_ctx.fits")


def test_rename_astrodrizzle_sidecars(tmp_path):
    log = logging.getLogger("t_rename")
    root = tmp_path / "x.drz.fits"
    root_str = str(root)
    sci = tmp_path / "x.drz_sci.fits"
    wht = tmp_path / "x.drz_wht.fits"
    ctx = tmp_path / "x.drz_ctx.fits"
    sci.write_bytes(b"a")
    wht.write_bytes(b"b")
    ctx.write_bytes(b"c")
    wf, mf = rename_astrodrizzle_sidecars(root_str, log)
    assert root.is_file()
    wdest, mdest = drizzle_canonical_weight_mask_paths(root_str)
    assert wf == wdest and mf == mdest
    assert os.path.isfile(wdest) and os.path.isfile(mdest)


def test_write_drc_multis_extension_writes_logical_path(tmp_path):
    log = logging.getLogger("t_wdrc")
    drz = tmp_path / "x.drz.fits"
    wht = tmp_path / "x.drz.weight.fits"
    ctx = tmp_path / "x.drz.mask.fits"
    drc_out = tmp_path / "logical.drc.fits"
    fits.PrimaryHDU(np.ones((4, 5), dtype=np.float32)).writeto(str(drz))
    fits.PrimaryHDU(np.ones((4, 5), dtype=np.float32)).writeto(str(wht))
    fits.PrimaryHDU(np.zeros((4, 5), dtype=np.int32)).writeto(str(ctx))

    def _fmt(hdul):
        return "test"

    path = write_drc_multis_extension_if_requested(
        str(drz),
        str(wht),
        str(ctx),
        True,
        log,
        format_hdu_list_summary=_fmt,
        logical_drc_path=str(drc_out),
    )
    assert path == str(drc_out)
    assert drc_out.is_file()
    with fits.open(drc_out) as hdul:
        assert wcs_image_hdu_index(hdul) == 1
        assert str(hdul[1].name).upper() == "SCI"
        assert hdul[1].header["NAXIS2"] == 4
