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
from hst123.utils.paths import normalize_fits_path


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


def test_check_images_skips_when_batch_ref_is_deepest_rawtmp_not_drizzle(
    tmp_path, caplog
):
    """Pipeline reference may be a drizzle product; TweakReg uses deepest *.rawtmp."""
    import logging

    caplog.set_level(logging.INFO)

    class _MockFits:
        def get_filter(self, path):
            s = str(path).lower()
            if "drizzle" in s:
                return "DRIZZLE"
            return "F555W"

    # Path must contain "drizzle" so the mock treats this as a drizzle product
    # (different band / role than science *.fits), like real *_ref_*.drc.fits paths.
    drizzle_dir = tmp_path / "drizzle"
    drizzle_dir.mkdir()
    drizzle = drizzle_dir / "stack.drc.fits"
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(drizzle)

    def _hdr(exptime):
        h = fits.Header()
        h["NAXIS"] = 2
        h["NAXIS1"] = 8
        h["NAXIS2"] = 8
        h["EXPTIME"] = float(exptime)
        return h

    sci_shallow = tmp_path / "sci_shallow.fits"
    sci_deep = tmp_path / "sci_deep.fits"
    fits.PrimaryHDU(
        np.zeros((8, 8), dtype=np.float32), header=_hdr(10.0)
    ).writeto(sci_shallow)
    fits.PrimaryHDU(
        np.zeros((8, 8), dtype=np.float32), header=_hdr(100.0)
    ).writeto(sci_deep)

    rt_deep = tmp_path / "sci_deep.rawtmp.fits"
    fits.PrimaryHDU(
        np.zeros((8, 8), dtype=np.float32), header=_hdr(100.0)
    ).writeto(rt_deep)
    hdr_shallow_rt = _hdr(10.0)
    # Aligned to deepest exposure reference (same ref_use as _build_tweakreg_batches).
    write_alignment_provenance(
        hdr_shallow_rt,
        method="tweakreg",
        ref_id=normalize_fits_path(str(rt_deep)),
        success=True,
    )
    fits.PrimaryHDU(
        np.zeros((8, 8), dtype=np.float32), header=hdr_shallow_rt
    ).writeto(tmp_path / "sci_shallow.rawtmp.fits")

    mock = SimpleNamespace(
        options={
            "args": SimpleNamespace(align_with="tweakreg", clobber=False),
        },
        _fits=_MockFits(),
    )
    astrom = AstrometryPrimitive(mock)
    out = astrom.check_images_for_tweakreg(
        [str(sci_shallow)],
        alignment_method="tweakreg",
        alignment_ref_id=str(drizzle),
        force_realign=False,
        tweakreg_reference_images=[str(sci_shallow), str(sci_deep)],
        tweakreg_pipeline_reference=str(drizzle),
    )
    assert out is None
    assert "Skipping alignment" in caplog.text


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
