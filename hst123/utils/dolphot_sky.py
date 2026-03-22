"""
Sky maps for DOLPHOT: ``calcsky`` preprocessing and a Python fallback.

Per the DOLPHOT User's Guide (§4.1 *calcsky*), the sky is estimated with an
iterative mean/rejection loop.  Drizzled HST images often contain **NaN** (and
sometimes **Inf**) in the PRIMARY array; mean/std in that loop become NaN and
older ``calcsky`` builds can abort with **SIGTRAP** on macOS.  AstroDrizzle
products are also commonly **PRIMARY + HDRTAB**, which some CFITSIO paths handle
poorly.

:func:`write_calcsky_sanitized_input` writes a **single-HDU**, float image with
non-finite pixels replaced by the finite-data median so the external ``calcsky``
binary can run reliably.  If it still fails, :func:`write_sky_fits_fallback`
produces ``*.sky.fits`` in-process.
"""
from __future__ import annotations

import logging
import os

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip

log = logging.getLogger(__name__)

# External DOLPHOT calcsky often SIGTRAP/crashes on very large drizzled mosaics (macOS).
_DEFAULT_CALCSKY_MAX_PIXELS = 6_000_000


def calcsky_max_pixels_external() -> int:
    raw = os.environ.get("HST123_CALCSKY_MAX_PIXELS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _DEFAULT_CALCSKY_MAX_PIXELS


def _science_hdu_index_for_calcsky(hdul: fits.HDUList) -> int:
    """PRIMARY if it holds a 2-D image; else first ``SCI`` extension with data."""
    prim = hdul[0]
    na = int(prim.header.get("NAXIS", 0) or 0)
    if na >= 2 and prim.data is not None:
        return 0
    if (
        len(hdul) > 1
        and str(hdul[1].name).strip().upper() == "SCI"
        and hdul[1].data is not None
    ):
        return 1
    return 0


def primary_array_pixel_count(fits_path: str) -> tuple[int, int, int]:
    """
    Return (naxis1, naxis2, n_pixels) for the science array from header only.

    MEF drizzle products (``.drc.fits``) often have an empty PRIMARY; then the
    science grid is taken from ``SCI`` (extension 1).
    """
    with fits.open(fits_path, mode="readonly", memmap=False) as hdul:
        idx = _science_hdu_index_for_calcsky(hdul)
        h = hdul[idx].header
        if int(h.get("NAXIS", 0) or 0) < 2:
            return 0, 0, 0
        n1 = int(h.get("NAXIS1", 0) or 0)
        n2 = int(h.get("NAXIS2", 0) or 0)
        return n1, n2, n1 * n2


def summarize_primary_for_calcsky(fits_path: str) -> str:
    """One-line summary of the science HDU for logs (shape, HDUs; bad-pixel stats if not huge)."""
    with fits.open(fits_path, mode="readonly", memmap=True) as hdul:
        n_ext = len(hdul)
        idx = _science_hdu_index_for_calcsky(hdul)
        h = hdul[idx].header
        na = int(h.get("NAXIS", 0) or 0)
        if na < 2:
            return f"no 2-D science array (ext {idx}) | HDUs={n_ext}"
        n1 = int(h.get("NAXIS1", 0) or 0)
        n2 = int(h.get("NAXIS2", 0) or 0)
        npx = n1 * n2
        d = hdul[idx].data
        dtype = getattr(d, "dtype", None)
        if d is None:
            return f"{n1}x{n2}={npx}px nodata ext={idx} | HDUs={n_ext}"
        if npx > 4_000_000:
            return (
                f"{n1}x{n2}={npx}px dtype={dtype} ext={idx} | HDUs={n_ext} "
                f"(bad-pixel stats skipped)"
            )
        arr = np.asarray(d, dtype=np.float64)
        finite = np.isfinite(arr)
        nbad = int(arr.size - np.count_nonzero(finite))
        frac = (nbad / arr.size) if arr.size else 0.0
        return (
            f"{n1}x{n2}={npx}px dtype={dtype} ext={idx} nan/inf={nbad}/{arr.size}({frac:.4f}) "
            f"| HDUs={n_ext}"
        )

# Above this size, Photutils Background2D is too slow for interactive pipelines;
# use the scipy median-filter path only.
_FALLBACK_FAST_PIXELS = 8_000_000


def _median_filter_sky_large(work: np.ndarray, box: int) -> np.ndarray:
    """
    Large-image sky smoothing (scipy ``median_filter`` by default).

    Set ``HST123_SKY_NUMBA=1`` to use a parallel Numba implementation (faster on
    big mosaics when ``numba`` is installed).
    """
    from scipy.ndimage import median_filter

    ny, nx = int(work.shape[0]), int(work.shape[1])
    m = max(3, min(ny, nx) // 4)
    if m % 2 == 0:
        m += 1
    k = min(int(box), m)
    if k % 2 == 0:
        k += 1
    env = os.environ.get("HST123_SKY_NUMBA", "").strip().lower()
    if env in ("1", "true", "yes"):
        try:
            return _median_filter_numba_parallel(work.astype(np.float64), k).astype(
                np.float32
            )
        except Exception as exc:
            log.debug("HST123_SKY_NUMBA median failed (%s); using scipy", exc)
    return median_filter(work, size=k).astype(np.float32)


def _median_filter_numba_parallel(work: np.ndarray, k: int) -> np.ndarray:
    """Odd kernel size *k*; median of k×k patch per pixel (parallel over rows)."""
    from numba import njit, prange

    ny, nx = work.shape
    r = k // 2
    out = np.empty_like(work)

    @njit(parallel=True)
    def run(inp, o):
        for iy in prange(ny):
            for ix in range(nx):
                ymin = max(0, iy - r)
                ymax = min(ny, iy + r + 1)
                xmin = max(0, ix - r)
                xmax = min(nx, ix + r + 1)
                patch = inp[ymin:ymax, xmin:xmax].ravel()
                o[iy, ix] = np.median(patch)

    run(work, out)
    return out


def sky_fits_path(image_fits: str) -> str:
    """
    Return the path to the DOLPHOT sky companion for *image_fits*.

    Only the trailing ``.fits`` / ``.fit`` suffix is replaced (paths may
    contain ``.fits`` more than once).
    """
    img = os.fspath(image_fits)
    low = img.lower()
    if low.endswith(".fits"):
        return img[:-5] + ".sky.fits"
    if low.endswith(".fit"):
        return img[:-4] + ".sky.fits"
    return img + ".sky.fits"


def write_calcsky_sanitized_input(src_fits: str, dst_fits: str) -> None:
    """
    Copy the science image to *dst_fits* in a form suited to DOLPHOT ``calcsky``.

    - Single PRIMARY HDU (no HDRTAB / extra extensions).
    - 2-D ``float32`` data; NaN and Inf replaced by the median of finite pixels
      (DOLPHOT §4.1 mean/rejection is not NaN-safe).
    - Header trimmed of extension-table keywords that confuse single-HDU files.
    """
    with fits.open(src_fits, memmap=True) as hdul:
        idx = _science_hdu_index_for_calcsky(hdul)
        src_hdu = hdul[idx]
        data = src_hdu.data
        if data is None:
            raise ValueError(f"No 2-D science data in {src_fits!r} (tried ext {idx})")
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError(
                f"calcsky needs a 2-D image in {src_fits!r}, got shape {arr.shape}"
            )
        hdr = src_hdu.header.copy()
        finite = np.isfinite(arr)
        if np.any(finite):
            med = float(np.median(arr[finite]))
        else:
            med = 0.0
        out = np.where(finite, arr, med).astype(np.float32, copy=False)
        # Strip MEF / scaling cards that confuse CFITSIO or DOLPHOT calcsky
        for key in (
            "EXTEND",
            "NEXTEND",
            "BZERO",
            "BSCALE",
            "BLANK",
            "DATAMIN",
            "DATAMAX",
        ):
            try:
                del hdr[key]
            except KeyError:
                pass
        hdr["EXTEND"] = False
        hdu_out = fits.PrimaryHDU(data=out, header=hdr)
        hdu_out.writeto(dst_fits, overwrite=True, output_verify="silentfix")


def write_sky_fits_fallback(
    image_fits: str,
    sky_fits: str,
    *,
    r_in: int,
    r_out: int,
    step: int,
    sigma_low: float,
    sigma_high: float,
) -> None:
    """
    Build a smooth sky image and write *sky_fits*.

    Parameters
    ----------
    image_fits, sky_fits
        Paths to the science FITS and output sky FITS.
    r_in, r_out, step, sigma_low, sigma_high
        Same meaning as DOLPHOT ``calcsky`` arguments; used to choose box size
        and sigma-clipping for the Photutils path.
    """
    with fits.open(image_fits, memmap=False) as hdul:
        idx = _science_hdu_index_for_calcsky(hdul)
        sh = hdul[idx]
        data = np.asarray(sh.data, dtype=np.float64)
        header = sh.header.copy()
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2-D science image in {image_fits!r}, got shape {data.shape}"
        )

    med = float(np.nanmedian(data))
    if not np.isfinite(med):
        med = 0.0
    work = np.where(np.isfinite(data), data, med)

    box = max(int(step) * 4, int(r_out), 32)
    if box % 2 == 0:
        box += 1
    filt = max(3, min(box // 8, 32))

    sky: np.ndarray
    npx = int(work.size)
    use_photutils = npx <= _FALLBACK_FAST_PIXELS
    if use_photutils:
        try:
            from photutils import Background2D, MedianBackground

            sigma_clip = SigmaClip(
                sigma_lower=float(sigma_low),
                sigma_upper=float(sigma_high),
                maxiters=10,
            )
            bkg = Background2D(
                work,
                box_size=(box, box),
                filter_size=(filt, filt),
                bkg_estimator=MedianBackground(),
                sigma_clip=sigma_clip,
                exclude_percentile=10.0,
            )
            sky = np.asarray(bkg.background, dtype=np.float32)
        except Exception as exc:
            log.debug(
                "Photutils sky fallback failed (%s); using scipy median_filter",
                exc,
            )
            use_photutils = False
    if not use_photutils:
        sky = _median_filter_sky_large(work, box)

    hdu_out = fits.PrimaryHDU(data=sky, header=header)
    hdu_out.writeto(sky_fits, overwrite=True, output_verify="silentfix")
