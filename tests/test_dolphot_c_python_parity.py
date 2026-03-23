"""
Parity and difference notes: DOLPHOT C tools vs hst123 Python ports
====================================================================

These tests **document** whether the Python implementations match the upstream
C executables on disk. When the C binary and DOLPHOT data tree are missing,
relevant cases ``pytest.skip`` so default CI still passes.

Summary (see individual test docstrings for procedure and tolerances)
---------------------------------------------------------------------

**calcsky** (:mod:`hst123.utils.dolphot_sky`)
    Goal: same algorithm as ``calcsky.c`` ``getsky`` (stage 1) + box mean (stage 2).
    **Parity:** On a small sanitized FITS, Python ``compute_sky_map_dolphot`` vs
    external ``calcsky`` should match within **~1e-4 relative** (float32 sky FITS
    I/O). Stage-1 sky is **quantized to float32** before the box smooth, like the
    C tool’s float image grid; σ-clip uses the same sequential double sum as
    ``calcsky.c``. The small-image test disables Numba for a stable baseline; a
    **1000×1000** test uses Numba with a fresh ``NUMBA_CACHE_DIR``. **Not exact**
    bit-identical vs every C build.

**splitgroups** (:mod:`hst123.utils.dolphot_splitgroups`)
    Goal: one file per SCI plane (or WFPC2-style 3-D primary), merged headers.
    **Parity:** For a minimal 2×SCI MEF, C ``splitgroups`` and
    :func:`~hst123.utils.dolphot_splitgroups.apply_splitgroups` should produce
    the **same pixel data** and aligned WCS-style keywords on the primary.
    **May differ:** FITS card order, comments, HISTORY, ``DATE``/checksums.

**acsmask** (:mod:`hst123.utils.dolphot_mask`)
    Goal: port of ``acs/acsmask.c`` for supported layouts; ``_classify_acs`` is
    **simplified** vs full C (docstring in source). **Parity:** Drizzled WFC
    (single SCI+WHT+CTX-style triplet, ``tp=2``) is compared in-process vs C
    ``acsmask`` on a copy. **May differ:** header ordering; rare edge layouts
    only handled in C; ``_either_string`` vs C HDRTAB/primary keyword lookup.

**wfc3mask** (:mod:`hst123.utils.dolphot_mask`)
    Same pattern as ACS for a minimal drizzled UVIS-style MEF (``tp=2``).
    **May differ:** as for ACS; IR/UVIS read-noise branches must match headers.

**wfpc2mask** (:mod:`hst123.utils.dolphot_mask`)
    **Native** ``(4, 800, 800)`` primary cube: Python raises
    :class:`~hst123.utils.dolphot_mask.UnsupportedMaskFormat` — use C binary or
    ``splitgroups`` first. **Drizzled** WFPC2 (``tp=-1``, SCI+WHT): parity test
    vs C when ``wfpc2mask`` is on PATH and DOLPHOT tree exists where needed.

Environment
-----------
* ``HST123_DOLPHOT_ROOT`` — DOLPHOT source tree (``acs/data``, ``wfc3/data`` PAMs).
* C tools on ``PATH``: ``calcsky``, ``splitgroups``, ``acsmask``, ``wfc3mask``,
  ``wfpc2mask`` (each test skips if its binary is missing).

Marker: ``dolphot_parity`` — ``pytest -m dolphot_parity`` runs only these tests.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

pytestmark = pytest.mark.dolphot_parity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True)


def _primary_data(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        d = hdul[0].data
        assert d is not None
        return np.asarray(d, dtype=np.float64)


def _sci1_data(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        d = hdul[1].data
        assert d is not None
        return np.asarray(d, dtype=np.float64)


def _two_sci_mef(path: Path) -> None:
    """Same layout as tests/test_dolphot_splitgroups.py."""
    p = fits.PrimaryHDU()
    p.header["INSTRUME"] = "ACS"
    p.header["PHOTFLAM"] = 1.0
    d1 = np.ones((10, 12), dtype=np.float32) * 1.0
    d2 = np.ones((10, 12), dtype=np.float32) * 2.0
    h1 = fits.ImageHDU(data=d1, name="SCI")
    h1.header["CRVAL1"] = 10.0
    h1.header["CCDCHIP"] = 1
    h2 = fits.ImageHDU(data=d2, name="SCI")
    h2.header["CRVAL1"] = 20.0
    h2.header["CCDCHIP"] = 2
    fits.HDUList([p, h1, h2]).writeto(path, overwrite=True)


def _acs_wfc_driz_mef(path: Path, *, seed: int = 0) -> None:
    """Minimal ACS/WFC drizzled MEF matching Python ``tp=2`` (SCI+WHT+CTX)."""
    rng = np.random.default_rng(seed)
    ny, nx = 20, 22
    prim = fits.PrimaryHDU()
    prim.header["TELESCOP"] = "HST"
    prim.header["INSTRUME"] = "ACS"
    prim.header["DETECTOR"] = "WFC"
    prim.header["FILETYPE"] = "SCI"
    prim.header["EXPTIME"] = 100.0
    prim.header["EXPSTART"] = 52000.0
    prim.header["EXPEND"] = 52000.02
    prim.header["CCDAMP"] = "ABCD"
    for k, v in (
        ("READNSEA", 3.1),
        ("READNSEB", 3.2),
        ("READNSEC", 3.3),
        ("READNSED", 3.4),
    ):
        prim.header[k] = v
    prim.header["NRPTEXP"] = 1
    prim.header["CRSPLIT"] = 1
    sci = (rng.random((ny, nx)) * 50 + 5.0).astype(np.float32)
    wht = (rng.random((ny, nx)) * 0.02 + 1e-4).astype(np.float32)
    ctx = np.ones((ny, nx), dtype=np.float32)
    h1 = fits.ImageHDU(data=sci, name="SCI")
    h2 = fits.ImageHDU(data=wht, name="WHT")
    h3 = fits.ImageHDU(data=ctx, name="CTX")
    fits.HDUList([prim, h1, h2, h3]).writeto(path, overwrite=True)


def _wfc3_uvis_driz_mef(path: Path, *, seed: int = 1) -> None:
    """Minimal WFC3/UVIS drizzled MEF (``tp=2``)."""
    rng = np.random.default_rng(seed)
    ny, nx = 18, 20
    prim = fits.PrimaryHDU()
    prim.header["TELESCOP"] = "HST"
    prim.header["INSTRUME"] = "WFC3"
    prim.header["DETECTOR"] = "UVIS"
    prim.header["FILETYPE"] = "SCI"
    prim.header["EXPTIME"] = 80.0
    prim.header["EXPSTART"] = 53000.0
    prim.header["EXPEND"] = 53000.01
    prim.header["CCDAMP"] = "ABCD"
    for k, v in (
        ("READNSEA", 3.0),
        ("READNSEB", 3.0),
        ("READNSEC", 3.0),
        ("READNSED", 3.0),
    ):
        prim.header[k] = v
    prim.header["NRPTEXP"] = 1
    prim.header["CRSPLIT"] = 1
    sci = (rng.random((ny, nx)) * 40 + 4.0).astype(np.float32)
    wht = (rng.random((ny, nx)) * 0.03 + 1e-4).astype(np.float32)
    ctx = np.ones((ny, nx), dtype=np.float32)
    h1 = fits.ImageHDU(data=sci, name="SCI")
    h2 = fits.ImageHDU(data=wht, name="WHT")
    h3 = fits.ImageHDU(data=ctx, name="CTX")
    fits.HDUList([prim, h1, h2, h3]).writeto(path, overwrite=True)


def _wfpc2_driz_mef(path: Path, *, seed: int = 2) -> None:
    """WFPC2 drizzled stack (``tp=-1``): PRIMARY + SCI + WHT."""
    rng = np.random.default_rng(seed)
    ny, nx = 14, 16
    prim = fits.PrimaryHDU()
    prim.header["TELESCOP"] = "HST"
    prim.header["INSTRUME"] = "WFPC2"
    prim.header["FILETYPE"] = "SCI"
    prim.header["ATODGAIN"] = 7
    prim.header["EXPTIME"] = 200.0
    prim.header["EXPSTART"] = 48000.0
    prim.header["EXPEND"] = 48000.01
    prim.header["RSDPFILL"] = -1.0
    prim.header["SATURATE"] = 27000.0
    sci = (rng.random((ny, nx)) * 100 + 1.0).astype(np.float32)
    wht = (rng.random((ny, nx)) * 0.05 + 1e-3).astype(np.float32)
    h1 = fits.ImageHDU(data=sci, name="SCI")
    h2 = fits.ImageHDU(data=wht, name="WHT")
    fits.HDUList([prim, h1, h2]).writeto(path, overwrite=True)


@pytest.fixture
def dolphot_mask_tree():
    """DOLPHOT install root with ``acs/data`` (and usually ``wfc3/data``)."""
    from hst123.utils.dolphot_mask import _resolve_dolphot_tree

    try:
        return _resolve_dolphot_tree(logging.getLogger(__name__))
    except FileNotFoundError as e:
        pytest.skip(f"DOLPHOT tree / PAM data not found: {e}")


# ---------------------------------------------------------------------------
# calcsky
# ---------------------------------------------------------------------------


def test_calcsky_python_matches_c_on_small_image(tmp_path, monkeypatch):
    """
    **Parity (approximate):** ``compute_sky_map_dolphot`` vs ``calcsky`` binary.

    Uses a small single-HDU float image and identical annulus parameters.
    Expect **near-exact** agreement (rtol ~1e-4) after float32 sky FITS round-trip;
    not bit-identical due to compiler math vs NumPy/LLVM.
    """
    exe = _which("calcsky")
    if not exe:
        pytest.skip("calcsky not on PATH")

    monkeypatch.setenv("HST123_CALCSKY_NUMBA", "0")

    from hst123.utils.dolphot_sky import compute_sky_map_dolphot, write_calcsky_sanitized_input

    raw = tmp_path / "skyin.fits"
    rng = np.random.default_rng(42)
    data = (rng.random((36, 38)) * 500 + 10.0).astype(np.float32)
    fits.PrimaryHDU(data=data).writeto(raw, overwrite=True)
    san = tmp_path / "san.fits"
    write_calcsky_sanitized_input(str(raw), str(san))

    rin, rout, step, sl, sh = 4, 10, 2, 2.25, 2.0
    with fits.open(san, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=np.float64)
    py_sky = compute_sky_map_dolphot(
        arr, r_in=rin, r_out=rout, step=step, sigma_low=sl, sigma_high=sh
    )

    work = tmp_path / "c_run"
    work.mkdir()
    shutil.copy(san, work / "san.fits")
    _run(
        ["calcsky", "san", str(rin), str(rout), str(step), str(sl), str(sh)],
        cwd=work,
    )
    c_sky_path = work / "san.sky.fits"
    assert c_sky_path.is_file()
    c_sky = _primary_data(c_sky_path).astype(np.float32)

    assert py_sky.shape == c_sky.shape
    assert np.allclose(py_sky, c_sky, rtol=1.2e-4, atol=1e-5), (
        "calcsky C vs Python sky map mismatch (see module docstring: float pipeline)"
    )


def test_calcsky_python_matches_c_on_1000_image(tmp_path, monkeypatch):
    """
    **Parity (approximate):** Same as the small-image test, on **1000×1000** pixels.

    DOLPHOT ``calcsky`` vs :func:`~hst123.utils.dolphot_sky.compute_sky_map_dolphot`
    with typical annulus parameters (``15 / 35 / 4 / 2.25 / 2.0``). **Numba is
    required** here: the pure-Python double loop is too slow at this size; Numba
    matches the Python port (see ``test_dolphot_sky_port_numba_matches_python``).
    """
    exe = _which("calcsky")
    if not exe:
        pytest.skip("calcsky not on PATH")

    try:
        import numba  # noqa: F401
    except ImportError:
        pytest.skip("numba required for 1000×1000 calcsky parity (runtime)")

    # Fresh Numba cache: stale JIT from older _clip_iter can otherwise match C poorly
    # while the current source would pass (user-visible symptom: unchanged max_abs).
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba_cache_calcsky_parity"))

    monkeypatch.delenv("HST123_CALCSKY_NUMBA", raising=False)
    import hst123.utils.dolphot_sky as dolphot_sky_module

    dolphot_sky_module._NUMBA_SKY_STAGES = None

    from hst123.utils.dolphot_sky import (
        compute_sky_map_dolphot,
        parse_calcsky_dataminmax,
        write_calcsky_sanitized_input,
    )

    ny, nx = 1000, 1000
    raw = tmp_path / "sky1k.fits"
    rng = np.random.default_rng(20260226)
    data = (rng.random((ny, nx)) * 500 + 10.0).astype(np.float32)
    fits.PrimaryHDU(data=data).writeto(raw, overwrite=True)
    san = tmp_path / "san1k.fits"
    write_calcsky_sanitized_input(str(raw), str(san))

    rin, rout, step, sl, sh = 15, 35, 4, 2.25, 2.0
    with fits.open(san, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=np.float64)
        dmin, dmax = parse_calcsky_dataminmax(hdul[0].header)
    py_sky = compute_sky_map_dolphot(
        arr,
        r_in=rin,
        r_out=rout,
        step=step,
        sigma_low=sl,
        sigma_high=sh,
        dmin=dmin,
        dmax=dmax,
    )

    work = tmp_path / "c_run_1k"
    work.mkdir()
    shutil.copy(san, work / "san1k.fits")
    _run(
        ["calcsky", "san1k", str(rin), str(rout), str(step), str(sl), str(sh)],
        cwd=work,
    )
    c_sky_path = work / "san1k.sky.fits"
    assert c_sky_path.is_file()
    c_sky = _primary_data(c_sky_path).astype(np.float32)

    assert py_sky.shape == c_sky.shape == (ny, nx)
    diff = np.abs(py_sky.astype(np.float64) - c_sky.astype(np.float64))
    max_abs = float(np.max(diff))
    rtol, atol = 1.2e-4, 1e-5
    ok = np.allclose(py_sky, c_sky, rtol=rtol, atol=atol)
    if not ok:
        rel = diff / (np.abs(c_sky.astype(np.float64)) + 1e-12)
        max_rel = float(np.max(rel))
        pcts = np.percentile(diff, [50.0, 90.0, 99.0, 99.9, 100.0])
        pytest.fail(
            f"calcsky C vs Python (1000×1000) mismatch: max_abs={max_abs:.6g} "
            f"max_rel={max_rel:.6g} (rtol={rtol}, atol={atol}); "
            f"|Δ| pctiles 50/90/99/99.9/100% = {pcts.tolist()}. "
            f"If max_abs is unchanged across edits, clear Numba cache (see NUMBA_CACHE_DIR "
            f"in this test) or remove ~/.cache/numba."
        )


# ---------------------------------------------------------------------------
# splitgroups
# ---------------------------------------------------------------------------


def test_splitgroups_python_matches_c_two_sci_mef(tmp_path):
    """
    **Parity (data + key WCS):** C ``splitgroups`` vs :func:`apply_splitgroups`.

    **Exact parity:** science array values for each chip.
    **Not guaranteed:** full header byte-identity (DATE, checksums, card order).
    """
    exe = _which("splitgroups")
    if not exe:
        pytest.skip("splitgroups not on PATH")

    from hst123.utils.dolphot_splitgroups import apply_splitgroups

    base = tmp_path / "raw.fits"
    _two_sci_mef(base)

    py_dir = tmp_path / "py"
    py_dir.mkdir()
    shutil.copy(base, py_dir / "raw.fits")
    py_out = apply_splitgroups(py_dir / "raw.fits")
    assert len(py_out) == 2

    c_dir = tmp_path / "c"
    c_dir.mkdir()
    shutil.copy(base, c_dir / "raw.fits")
    _run(["splitgroups", "raw.fits"], cwd=c_dir)

    c_candidates = [p for p in c_dir.glob("*.fits") if p.name != "raw.fits"]
    assert len(c_candidates) == 2, f"unexpected splitgroups outputs: {list(c_dir.iterdir())}"

    def _mean_chip(p: Path) -> float:
        return float(np.mean(_primary_data(p)))

    py_sorted = sorted((_mean_chip(Path(x)), Path(x)) for x in py_out)
    c_sorted = sorted((_mean_chip(p), p) for p in c_candidates)
    for (_, p_py), (_, p_c) in zip(py_sorted, c_sorted):
        d_py = _primary_data(p_py)
        d_c = _primary_data(p_c)
        assert np.allclose(d_py, d_c, rtol=0, atol=0), (p_py, p_c)
        with fits.open(p_py) as a, fits.open(p_c) as b:
            assert float(a[0].header["CRVAL1"]) == float(b[0].header["CRVAL1"])


# ---------------------------------------------------------------------------
# acsmask
# ---------------------------------------------------------------------------


def test_acsmask_python_matches_c_driz_wfc(tmp_path, dolphot_mask_tree):
    """
    **Parity (approximate):** drizzled ACS/WFC (single output chip) SCI pixels.

    C and Python both modify FITS in place; we compare copies. **May differ**
    slightly on headers and on pixels at ~1e-5 if exposure/PAM paths differ;
    here PAM is unused (drizzle branch). Uses ``allclose`` rtol 1e-5.
    """
    exe = _which("acsmask")
    if not exe:
        pytest.skip("acsmask not on PATH")

    from hst123.utils.dolphot_mask import apply_acsmask

    src = tmp_path / "acs_drz.fits"
    _acs_wfc_driz_mef(src)

    c_copy = tmp_path / "acs_c.fits"
    py_copy = tmp_path / "acs_py.fits"
    shutil.copy(src, c_copy)
    shutil.copy(src, py_copy)

    _run(["acsmask", str(c_copy.name)], cwd=tmp_path)
    apply_acsmask(py_copy, dolphot_mask_tree)

    d_c = _sci1_data(c_copy)
    d_py = _sci1_data(py_copy)
    assert d_c.shape == d_py.shape
    assert np.allclose(d_c, d_py, rtol=1e-5, atol=1e-4), "acsmask C vs Python SCI mismatch"


# ---------------------------------------------------------------------------
# wfc3mask
# ---------------------------------------------------------------------------


def test_wfc3mask_python_matches_c_driz_uvis(tmp_path, dolphot_mask_tree):
    """
    **Parity (approximate):** drizzled WFC3/UVIS single-chip output.

    Same caveats as :func:`test_acsmask_python_matches_c_driz_wfc`.
    """
    exe = _which("wfc3mask")
    if not exe:
        pytest.skip("wfc3mask not on PATH")

    from hst123.utils.dolphot_mask import apply_wfc3mask

    src = tmp_path / "w3_drz.fits"
    _wfc3_uvis_driz_mef(src)

    c_copy = tmp_path / "w3_c.fits"
    py_copy = tmp_path / "w3_py.fits"
    shutil.copy(src, c_copy)
    shutil.copy(src, py_copy)

    _run(["wfc3mask", str(c_copy.name)], cwd=tmp_path)
    apply_wfc3mask(py_copy, dolphot_mask_tree)

    d_c = _sci1_data(c_copy)
    d_py = _sci1_data(py_copy)
    assert d_c.shape == d_py.shape
    assert np.allclose(d_c, d_py, rtol=1e-5, atol=1e-4), "wfc3mask C vs Python SCI mismatch"


# ---------------------------------------------------------------------------
# wfpc2mask
# ---------------------------------------------------------------------------


def test_wfpc2mask_python_matches_c_driz(tmp_path):
    """
    **Parity (approximate):** drizzled WFPC2 (SCI+WHT) in-place mask.

    Native 4×800×800 layouts are **not** ported in Python (see
    :func:`test_wfpc2mask_native_cube_not_supported_in_python`).
    """
    exe = _which("wfpc2mask")
    if not exe:
        pytest.skip("wfpc2mask not on PATH")

    from hst123.utils.dolphot_mask import apply_wfpc2mask

    src = tmp_path / "wf2_drz.fits"
    _wfpc2_driz_mef(src)

    c_copy = tmp_path / "wf2_c.fits"
    py_copy = tmp_path / "wf2_py.fits"
    shutil.copy(src, c_copy)
    shutil.copy(src, py_copy)

    _run(["wfpc2mask", str(c_copy.name)], cwd=tmp_path)
    apply_wfpc2mask(py_copy, None)

    d_c = _sci1_data(c_copy)
    d_py = _sci1_data(py_copy)
    assert d_c.shape == d_py.shape
    assert np.allclose(d_c, d_py, rtol=1e-5, atol=1e-4), "wfpc2mask C vs Python SCI mismatch"


def test_wfpc2mask_native_cube_not_supported_in_python(tmp_path):
    """
    **No Python parity:** WFPC2 ``(4, 800, 800)`` primary is explicitly rejected.

    The implementation still requires a *dq_path* file to exist whenever ``tp>0``
    (native layouts), before the unsupported message — stricter than C, which only
    needs DQ for the multi-extension path. Use the C ``wfpc2mask`` + DQ file, or
    ``splitgroups`` then per-chip masks.
    """
    from hst123.utils.dolphot_mask import UnsupportedMaskFormat, apply_wfpc2mask

    p = tmp_path / "cube.fits"
    # Exact shape C code classifies as native WFPC2 cube (Python port skips it).
    data = np.zeros((4, 800, 800), dtype=np.float32)
    ph = fits.PrimaryHDU(data=data)
    ph.header["TELESCOP"] = "HST"
    ph.header["INSTRUME"] = "WFPC2"
    ph.header["FILETYPE"] = "SCI"
    ph.header["ATODGAIN"] = 7
    ph.header["EXPTIME"] = 100.0
    ph.header["EXPSTART"] = 48000.0
    ph.header["EXPEND"] = 48000.01
    ph.header["RSDPFILL"] = -1.0
    ph.header["SATURATE"] = 27000.0
    fits.HDUList([ph]).writeto(p, overwrite=True)

    dq = tmp_path / "dq_placeholder.fits"
    fits.PrimaryHDU().writeto(dq, overwrite=True)

    with pytest.raises(UnsupportedMaskFormat, match="800×800×4 primary not supported"):
        apply_wfpc2mask(p, dq)


def test_either_string_differs_from_c_lookup_documentation():
    """
    **Documented difference (not a runtime binary test):** Python
    :func:`hst123.utils.dolphot_mask._either_string` scans HDRTAB-like
    extensions 2 and 3 only, as an approximation of C ``eitherstring``.

    Full parity with every MAST HDRTAB variant is **not** asserted here.
    """
    from hst123.utils import dolphot_mask as dm

    assert "Approximate" in (dm._either_string.__doc__ or "")


def test_acs_classify_simplified_documentation():
    """
    **Documented difference:** :func:`hst123.utils.dolphot_mask._classify_acs`
    states it is simplified vs full C ``ACStype`` handling.
    """
    from hst123.utils import dolphot_mask as dm

    assert "Simplified" in (dm._classify_acs.__doc__ or "")
