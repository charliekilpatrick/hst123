"""Tests for hst123.utils.dolphot_sky (sky path + Python calcsky fallback)."""
import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from hst123.utils.dolphot_sky import (
    _calcsky_annulus_offsets,
    _calcsky_list_capacity,
    _dolphot_stage2_box_mean,
    calcsky_max_pixels_external,
    noise_fits_path,
    primary_array_pixel_count,
    sky_fits_path,
    write_calcsky_sanitized_input,
    write_sky_fits_fallback,
)


def test_primary_array_pixel_count(tmp_path):
    img = tmp_path / "x.fits"
    fits.PrimaryHDU(np.zeros((10, 20), dtype=np.float32)).writeto(str(img))
    n1, n2, n = primary_array_pixel_count(str(img))
    assert n1 * n2 == n == 200


def test_primary_array_pixel_count_drc_style_mef(tmp_path):
    """Empty PRIMARY + SCI: pixel count uses SCI (like pipeline .drc.fits)."""
    img = tmp_path / "stack.drc.fits"
    prim = fits.PrimaryHDU()
    prim.header["NAXIS"] = 0
    sci = fits.ImageHDU(
        np.zeros((30, 40), dtype=np.float32), name="SCI", header=fits.Header()
    )
    sci.header["NAXIS"] = 2
    sci.header["NAXIS1"] = 40
    sci.header["NAXIS2"] = 30
    fits.HDUList([prim, sci]).writeto(str(img), overwrite=True)
    n1, n2, n = primary_array_pixel_count(str(img))
    assert (n1, n2, n) == (40, 30, 1200)


def test_write_calcsky_sanitized_from_sci_extension(tmp_path):
    src = tmp_path / "mef.fits"
    dst = tmp_path / "out.fits"
    prim = fits.PrimaryHDU()
    prim.header["NAXIS"] = 0
    sci = fits.ImageHDU(
        np.ones((12, 10), dtype=np.float32) * 5.0, name="SCI", header=fits.Header()
    )
    sci.header["NAXIS"] = 2
    sci.header["NAXIS1"] = 10
    sci.header["NAXIS2"] = 12
    fits.HDUList([prim, sci]).writeto(str(src), overwrite=True)
    write_calcsky_sanitized_input(str(src), str(dst))
    with fits.open(dst) as hdul:
        assert len(hdul) == 1
        assert hdul[0].data.shape == (12, 10)


def test_calcsky_max_pixels_from_env(monkeypatch):
    monkeypatch.setenv("HST123_CALCSKY_MAX_PIXELS", "12345")
    assert calcsky_max_pixels_external() == 12345
    monkeypatch.delenv("HST123_CALCSKY_MAX_PIXELS", raising=False)
    assert calcsky_max_pixels_external() >= 1_000_000


def test_noise_fits_path_trailing_fits_only():
    assert noise_fits_path("/a/b/stack.drc.fits") == "/a/b/stack.drc.noise.fits"


def test_sky_fits_path_trailing_fits_only():
    assert sky_fits_path("/a/b/c/img.fits") == "/a/b/c/img.sky.fits"
    p = "/data/foo.fits/bar.drz.fits"
    assert sky_fits_path(p) == "/data/foo.fits/bar.drz.sky.fits"


def test_sky_fits_path_fit_suffix():
    assert sky_fits_path("/x/y.z.fit") == "/x/y.z.sky.fits"


def test_calcsky_annulus_offsets_bounded_by_list_capacity():
    """Offset count must fit in calcsky list buffer (same as C calloc size)."""
    rin2, rout2, rsky, step = 0, 35 * 35, 35, 4
    oy, ox = _calcsky_annulus_offsets(rin2, rout2, rsky, step)
    cap = _calcsky_list_capacity(rsky, step)
    assert oy.shape[0] == ox.shape[0] <= cap


def test_dolphot_stage2_matches_naive_loop():
    """Summed-area stage 2 matches explicit C-order loops (float reorder only)."""
    rng = np.random.default_rng(3)
    tsky = rng.normal(80.0, 3.0, (24, 32)).astype(np.float64)
    step = 4
    ny, nx = tsky.shape
    naive = np.empty_like(tsky)
    for y in range(ny):
        for x in range(nx):
            n, acc = 0, 0.0
            for yy in range(y - step + 1, y + step + 1):
                for xx in range(x - step + 1, x + step + 1):
                    if 0 <= xx < nx and 0 <= yy < ny:
                        n += 1
                        acc += tsky[yy, xx]
            naive[y, x] = acc / n if n else tsky[y, x]
    fast = _dolphot_stage2_box_mean(tsky, step)
    np.testing.assert_allclose(naive, fast, rtol=0, atol=1e-9)


def test_write_sky_fits_fallback_shape_and_file(tmp_path):
    img = tmp_path / "sci.fits"
    sky = tmp_path / "sci.sky.fits"
    data = np.random.default_rng(0).normal(100.0, 5.0, (64, 48)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(str(img), overwrite=True)
    write_sky_fits_fallback(
        str(img),
        str(sky),
        r_in=15,
        r_out=35,
        step=4,
        sigma_low=2.25,
        sigma_high=2.0,
    )
    assert sky.is_file()
    with fits.open(sky) as hdul:
        assert hdul[0].data.shape == (64, 48)


def test_write_calcsky_sanitized_input_nan_to_median(tmp_path):
    src = tmp_path / "in.fits"
    dst = tmp_path / "san.fits"
    d = np.ones((20, 16), dtype=np.float32) * 3.0
    d[5:8, 5:8] = np.nan
    fits.PrimaryHDU(d).writeto(str(src), overwrite=True)
    write_calcsky_sanitized_input(str(src), str(dst))
    with fits.open(dst) as hdul:
        assert len(hdul) == 1
        arr = hdul[0].data
        assert np.all(np.isfinite(arr))
        assert arr.shape == (20, 16)
        assert np.allclose(arr[5:8, 5:8], 3.0)


def test_write_calcsky_sanitized_input_drops_second_hdu(tmp_path):
    """Mimic drizzle PRIMARY+HDRTAB: output must be single HDU for calcsky."""
    src = tmp_path / "mef.fits"
    dst = tmp_path / "out.fits"
    prim = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    hdr = fits.BinTableHDU.from_columns(
        [fits.Column(name="x", array=np.array([1]), format="K")]
    )
    hdr.name = "HDRTAB"
    fits.HDUList([prim, hdr]).writeto(str(src), overwrite=True)
    write_calcsky_sanitized_input(str(src), str(dst))
    with fits.open(dst) as hdul:
        assert len(hdul) == 1


@pytest.mark.dolphot
def test_calcsky_succeeds_on_sanitized_image(tmp_path):
    """If DOLPHOT calcsky is installed, it must run on a NaN-free single-HDU copy."""
    if not shutil.which("calcsky"):
        pytest.skip("calcsky not on PATH")
    src = tmp_path / "sci.fits"
    rng = np.random.default_rng(0)
    d = rng.normal(100.0, 5.0, (128, 96)).astype(np.float32)
    d[40:45, 40:45] = np.nan
    fits.PrimaryHDU(d).writeto(str(src), overwrite=True)
    san = tmp_path / "san.fits"
    write_calcsky_sanitized_input(str(src), str(san))
    root = str(san.with_suffix(""))  # strip .fits
    sky_expect = tmp_path / "san.sky.fits"
    r = subprocess.run(
        ["calcsky", root, "5", "15", "2", "2.25", "2.0"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert sky_expect.is_file()
    with fits.open(sky_expect) as hdul:
        assert hdul[0].data.shape == (128, 96)


def test_calc_sky_skips_external_when_over_pixel_cap(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from hst123.primitives.run_dolphot.run_dolphot_primitive import DolphotPrimitive

    monkeypatch.setenv("HST123_CALCSKY_MAX_PIXELS", "500")
    mock_pipeline = MagicMock()
    mock_pipeline._fits = MagicMock()
    mock_pipeline._fits.get_instrument.return_value = "acs_wfc_full"
    prim = DolphotPrimitive(mock_pipeline)
    img = tmp_path / "big.fits"
    fits.PrimaryHDU(np.zeros((25, 25), dtype=np.float32)).writeto(str(img))
    options = {
        "acs_wfc": {
            "dolphot_sky": {
                "r_in": 15,
                "r_out": 35,
                "step": 4,
                "sigma_low": 2.25,
                "sigma_high": 2.0,
            }
        }
    }
    with patch(
        "hst123.primitives.run_dolphot.run_dolphot_primitive.run_external_command",
    ) as mock_run:
        prim.calc_sky(str(img), options)
    mock_run.assert_not_called()
    assert (tmp_path / "big.sky.fits").is_file()


@pytest.mark.dolphot
def test_calcsky_cli_vs_python_dolphot_port_correlation(tmp_path, monkeypatch):
    """
    CLI calcsky vs Python DOLPHOT getsky port should agree closely (same algorithm).

    See DOLPHOT User's Guide §4.1 and ``calcsky.c`` ``getsky``.
    """
    import hst123.utils.dolphot_sky as ds

    monkeypatch.delenv("HST123_CALCSKY_LEGACY", raising=False)
    if not shutil.which("calcsky"):
        pytest.skip("calcsky not on PATH")
    rng = np.random.default_rng(42)
    data = rng.normal(100.0, 6.0, (256, 256)).astype(np.float32)
    img = tmp_path / "sci.fits"
    fits.PrimaryHDU(data).writeto(str(img), overwrite=True)
    san = tmp_path / "san.fits"
    write_calcsky_sanitized_input(str(img), str(san))
    root = str(san.with_suffix(""))
    r = subprocess.run(
        ["calcsky", root, "15", "35", "4", "2.25", "2.0"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    sky_cli = tmp_path / "san.sky.fits"
    assert sky_cli.is_file()
    sky_py = tmp_path / "san.py.sky.fits"
    ds.write_sky_fits_fallback(
        str(san),
        str(sky_py),
        r_in=15,
        r_out=35,
        step=4,
        sigma_low=2.25,
        sigma_high=2.0,
    )
    a = np.asarray(fits.getdata(str(sky_cli)), dtype=np.float64)
    b = np.asarray(fits.getdata(str(sky_py)), dtype=np.float64)
    assert a.shape == b.shape
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert corr > 0.999
    mad = float(np.max(np.abs(a - b)))
    assert mad < 0.15


def test_dolphot_sky_port_numba_matches_python(monkeypatch):
    """Pure-Python and Numba DOLPHOT sky paths should match on a small grid."""
    import hst123.utils.dolphot_sky as ds

    try:
        import numba  # noqa: F401
    except ImportError:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(7)
    data = rng.normal(50.0, 4.0, (48, 40)).astype(np.float64)
    monkeypatch.setenv("HST123_CALCSKY_NUMBA", "0")
    a = ds.compute_sky_map_dolphot(
        data, r_in=3, r_out=12, step=2, sigma_low=2.25, sigma_high=2.0
    )
    monkeypatch.setenv("HST123_CALCSKY_NUMBA", "1")
    ds._NUMBA_SKY_STAGES = None
    b = ds.compute_sky_map_dolphot(
        data, r_in=3, r_out=12, step=2, sigma_low=2.25, sigma_high=2.0
    )
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_write_sky_fits_fallback_rejects_non_2d(tmp_path):
    img = tmp_path / "bad.fits"
    fits.PrimaryHDU(np.zeros(10)).writeto(str(img), overwrite=True)
    with pytest.raises(ValueError, match="2-D"):
        write_sky_fits_fallback(
            str(img),
            str(tmp_path / "out.sky.fits"),
            r_in=15,
            r_out=35,
            step=4,
            sigma_low=2.25,
            sigma_high=2.0,
        )
