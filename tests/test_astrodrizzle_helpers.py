"""Unit tests for hst123.utils.astrodrizzle_helpers."""
import logging
import os

import numpy as np
from astropy.io import fits

from hst123.utils.astrodrizzle_helpers import (
    combine_type_and_nhigh,
    drizzle_canonical_weight_mask_paths,
    drizzle_sidecar_paths,
    resolve_drizzle_clean_flag,
    rename_astrodrizzle_sidecars,
    wcs_image_hdu_index,
    write_drc_multis_extension_if_requested,
)


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
