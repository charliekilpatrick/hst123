"""Alignment provenance (HST123 HIERARCH keywords) and redundant-skip logic."""
import numpy as np
import pytest
from astropy.io import fits
from types import SimpleNamespace

from hst123.primitives.astrometry.alignment_meta import (
    alignment_is_redundant,
    normalize_alignment_ref_id,
    read_alignment_provenance,
    write_alignment_provenance,
)
from hst123.primitives.astrometry import AstrometryPrimitive


def test_normalize_ref_gaia_not_realpath(tmp_path):
    assert normalize_alignment_ref_id("GAIA") == "GAIA"
    # Avoid treating GAIA as a relative path
    (tmp_path / "GAIA").mkdir()
    assert normalize_alignment_ref_id("GAIA") == "GAIA"


def test_write_read_roundtrip_primary_header():
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 4
    hdr["NAXIS2"] = 4
    write_alignment_provenance(
        hdr, method="tweakreg", ref_id="/tmp/ref.fits", success=True
    )
    prov = read_alignment_provenance(hdr)
    assert prov is not None
    assert prov["ok"] is True
    assert prov["method"] == "tweakreg"
    assert prov["ref"] == normalize_alignment_ref_id("/tmp/ref.fits")


def test_alignment_is_redundant_requires_all_three():
    hdr = fits.Header()
    write_alignment_provenance(
        hdr, method="tweakreg", ref_id="/x/ref.fits", success=True
    )
    assert alignment_is_redundant(
        hdr, method="tweakreg", ref_id="/x/ref.fits", require_success=True
    )
    assert not alignment_is_redundant(
        hdr, method="jhat", ref_id="/x/ref.fits", require_success=True
    )
    assert not alignment_is_redundant(
        hdr, method="tweakreg", ref_id="/other/ref.fits", require_success=True
    )


def test_check_images_skips_redundant_tweakreg(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
    ref_path = tmp_path / "ref.fits"
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(ref_path)

    img_hdr = fits.Header()
    img_hdr["NAXIS"] = 2
    img_hdr["NAXIS1"] = 8
    img_hdr["NAXIS2"] = 8
    write_alignment_provenance(
        img_hdr,
        method="tweakreg",
        ref_id=str(ref_path),
        success=True,
    )
    img_path = tmp_path / "sci.fits"
    fits.PrimaryHDU(
        np.zeros((8, 8), dtype=np.float32), header=img_hdr
    ).writeto(img_path)

    mock = SimpleNamespace(
        options={
            "args": SimpleNamespace(align_with="tweakreg", clobber=False),
        }
    )
    astrom = AstrometryPrimitive(mock)
    out = astrom.check_images_for_tweakreg(
        [str(img_path)],
        alignment_method="tweakreg",
        alignment_ref_id=str(ref_path),
        force_realign=False,
    )
    assert out is None
    assert "Skipping alignment" in caplog.text


def test_check_images_runs_when_reference_differs(tmp_path):
    ref_a = tmp_path / "ref_a.fits"
    ref_b = tmp_path / "ref_b.fits"
    for p in (ref_a, ref_b):
        fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(p)

    img_hdr = fits.Header()
    img_hdr["NAXIS"] = 2
    img_hdr["NAXIS1"] = 8
    img_hdr["NAXIS2"] = 8
    write_alignment_provenance(
        img_hdr,
        method="tweakreg",
        ref_id=str(ref_a),
        success=True,
    )
    img_path = tmp_path / "sci.fits"
    fits.PrimaryHDU(
        np.zeros((8, 8), dtype=np.float32), header=img_hdr
    ).writeto(img_path)

    mock = SimpleNamespace(
        options={
            "args": SimpleNamespace(align_with="tweakreg", clobber=False),
        }
    )
    astrom = AstrometryPrimitive(mock)
    out = astrom.check_images_for_tweakreg(
        [str(img_path)],
        alignment_method="tweakreg",
        alignment_ref_id=str(ref_b),
        force_realign=False,
    )
    assert out is not None
    assert str(img_path) in out


def test_check_images_forced_realign_with_clobber(tmp_path):
    ref_path = tmp_path / "ref.fits"
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(ref_path)

    img_hdr = fits.Header()
    img_hdr["NAXIS"] = 2
    img_hdr["NAXIS1"] = 8
    img_hdr["NAXIS2"] = 8
    write_alignment_provenance(
        img_hdr,
        method="tweakreg",
        ref_id=str(ref_path),
        success=True,
    )
    img_path = tmp_path / "sci.fits"
    fits.PrimaryHDU(
        np.zeros((8, 8), dtype=np.float32), header=img_hdr
    ).writeto(img_path)

    mock = SimpleNamespace(
        options={
            "args": SimpleNamespace(align_with="tweakreg", clobber=True),
        }
    )
    astrom = AstrometryPrimitive(mock)
    out = astrom.check_images_for_tweakreg(
        [str(img_path)],
        alignment_method="tweakreg",
        alignment_ref_id=str(ref_path),
        force_realign=True,
    )
    assert out is not None
