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
produces ``*.sky.fits`` in-process using a Python port of DOLPHOT's ``getsky``
from ``calcsky.c`` (same annulus sampling, σ-rejection, and box smooth as the
C tool; see also ``param/fits.param`` defaults for pixel inclusion). Stage-1 sky
values are rounded to **float32** before stage 2, matching the C program’s
float image buffer (avoids a systematic offset vs the binary on large grids).

**Environment**

* ``HST123_CALCSKY_LEGACY`` — if ``1``/``true``, use the older Photutils /
  ``scipy.ndimage.median_filter`` fallback instead of the DOLPHOT port.
* ``HST123_CALCSKY_NUMBA`` — if ``0``/``false``, disable Numba for stage 1 only
  (stage 2 uses a vectorized summed-area table in both paths). Without Numba,
  stage 1 remains a pure Python double loop; **install ``numba``** for large
  images (often **orders of magnitude** faster). Progress lines during stage 1
  use a **single** update after the parallel pass when Numba is on (batched
  row-by-row updates would serialize work and ruin performance).
* ``HST123_CALCSKY_PROGRESS`` / ``HST123_PROGRESS_LOG`` — set to ``0`` to disable
  throttled progress lines during the Python/Numba sky map (see
  :mod:`hst123.utils.progress_log`).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip

from hst123.utils.progress_log import LoggedProgress, calcsky_progress_enabled

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


def noise_fits_path(image_fits: str) -> str:
    """
    Return the path to the pipeline ``.noise.fits`` sidecar (copy of sky for
    drizzled products), matching :meth:`hst123._pipeline.hst123.drizzle_all`.
    """
    img = os.fspath(image_fits)
    low = img.lower()
    if low.endswith(".fits"):
        return img[:-5] + ".noise.fits"
    if low.endswith(".fit"):
        return img[:-4] + ".noise.fits"
    return img + ".noise.fits"


# -----------------------------------------------------------------------------
# DOLPHOT calcsky.c ``getsky()`` port (DOLPHOT 3.1; User's Guide §4.1).
# Pixel inclusion limits match ``readfits`` + ``parsecards`` after ``fits.param``
# (BADPIX/SATURATE keywords; defaults minval_val=-1, maxval_val=65535).
# -----------------------------------------------------------------------------

_DEFAULT_CALCSKY_DMIN = -1.0
_DEFAULT_CALCSKY_DMAX = 65535.0


def parse_calcsky_dataminmax(header) -> tuple[float, float]:
    """
    Return (DMIN, DMAX) for calcsky-style pixel gating: DMIN < pix < DMAX.

    Tries BADPIX then MINVAL, SATURATE then MAXVAL (DOLPHOT ``param/fits.param``).
    """
    dmin = _DEFAULT_CALCSKY_DMIN
    dmax = _DEFAULT_CALCSKY_DMAX
    if "BADPIX" in header:
        try:
            dmin = float(header["BADPIX"])
        except (TypeError, ValueError):
            pass
    elif "MINVAL" in header:
        try:
            dmin = float(header["MINVAL"])
        except (TypeError, ValueError):
            pass
    if "SATURATE" in header:
        try:
            dmax = float(header["SATURATE"])
        except (TypeError, ValueError):
            pass
    elif "MAXVAL" in header:
        try:
            dmax = float(header["MAXVAL"])
        except (TypeError, ValueError):
            pass
    return dmin, dmax


def _calcsky_adjust_r_out_step(r_out: int, step: int) -> tuple[int, int]:
    """
    Match calcsky.c: ``rsky`` is argv outer radius (≥1), then rounded up to a
    multiple of ``skip`` for the sampling grid span only.

    **Important:** ``rout2`` used in the annulus test ``rin2 <= r^2 <= rout2`` is
    ``argv_r_out**2`` **before** this bump (see ``calcsky.c``: ``rout2=rsky*rsky``
    then ``if (rsky%skip) rsky=…``). Using ``rsky**2`` after the bump wrongly
    enlarges the outer annulus when ``r_out`` is not divisible by ``step``
    (e.g. 35 / 4 → grid to 36 px but ``rout2`` stays 35²).
    """
    s = int(step)
    if s == 0:
        raise ValueError("calcsky step must be nonzero")
    rsky = int(r_out)
    if rsky < 1:
        rsky = 1
    if rsky % s != 0:
        rsky = (rsky + s - 1) // s * s
    return rsky, s


def _calcsky_list_capacity(rsky: int, step: int) -> int:
    """``(2*rsky/skip+1)^2`` with C-style integer division (calcsky.c line 16)."""
    g = 2 * (rsky // step) + 1
    return max(1, g * g)


def _calcsky_annulus_offsets(rin2: int, rout2: int, rsky: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Integer (dy, dx) offsets on the calcsky sampling grid, in the same order as
    calcsky.c (``yy`` outer, ``xx`` inner, both stepped by *step*).
    """
    off_y: list[int] = []
    off_x: list[int] = []
    yy = -int(rsky)
    st = int(step)
    while yy <= rsky:
        xx = -int(rsky)
        while xx <= rsky:
            r = xx * xx + yy * yy
            if r >= int(rin2) and r <= int(rout2):
                off_y.append(yy)
                off_x.append(xx)
            xx += st
        yy += st
    return (
        np.asarray(off_y, dtype=np.int32),
        np.asarray(off_x, dtype=np.int32),
    )


def _dolphot_stage2_box_mean(tsky: np.ndarray, step: int) -> np.ndarray:
    """
    Second pass of calcsky.c ``getsky``: local mean of ``tsky`` over in-bounds
    pixels with ``yy in [y-step+1, y+step]``, ``xx in [x-step+1, x+step]``.

    Fully vectorized (summed-area table); numerically matches the nested loops.
    """
    data = np.asarray(tsky, dtype=np.float64)
    ny, nx = data.shape
    s = int(step)
    yy = np.arange(ny, dtype=np.int32)[:, None]
    xx = np.arange(nx, dtype=np.int32)[None, :]
    y0 = np.maximum(0, yy - s + 1)
    y1 = np.minimum(ny - 1, yy + s)
    x0 = np.maximum(0, xx - s + 1)
    x1 = np.minimum(nx - 1, xx + s)
    S = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    S[1:, 1:] = np.cumsum(np.cumsum(data, axis=0), axis=1)
    sum_w = (
        S[y1 + 1, x1 + 1] - S[y0, x1 + 1] - S[y1 + 1, x0] + S[y0, x0]
    )
    cnt = (y1 - y0 + 1).astype(np.float64) * (x1 - x0 + 1).astype(np.float64)
    out = np.where(cnt > 0.0, sum_w / cnt, data)
    return out


def _quantize_stage1_tsky_like_calcsky_c(tsky: np.ndarray) -> np.ndarray:
    """
    Align stage 1 → stage 2 with DOLPHOT ``calcsky.c``.

    The C code keeps the first-pass sky map in the same **float** image type as
    the input (typically IEEE float32 / ``BITPIX=-32``). The box-mean pass then
    sums those **already-rounded** values. Keeping ``tsky`` in full ``float64``
    through stage 2 biases the local mean vs the binary and shows up as a
    **systematic** ~0.2–1 ADU median offset on large grids—not a σ-clip edge case.
    """
    return np.asarray(tsky, dtype=np.float32).astype(np.float64)


def _dolphot_sigma_clip_iterate(
    list_buf: np.ndarray, n_in: int, siglo: float, sighi: float
) -> tuple[float, int]:
    """
    Iterative mean/σ rejection from calcsky.c ``getsky`` inner loop.

    σ uses ``sqrt(1 + sum((x-mean)^2)/(n-1))`` for n>1, else ``sqrt(1+sum)``.
    """
    n = int(n_in)
    xx = 1
    sky = 0.0
    while xx:
        xx = 0
        if n == 0:
            sky = 0.0
        else:
            # Match calcsky.c: sequential double accumulation (not np.sum), so σ-clip
            # thresholds match the C binary; np.sum can differ at rejection edges.
            sky = 0.0
            for i in range(n):
                sky += float(list_buf[i])
            sky /= n
            sig = 0.0
            for i in range(n):
                d = float(list_buf[i]) - sky
                sig += d * d
            if n > 1:
                sig = float(np.sqrt(1.0 + sig / (n - 1)))
            else:
                sig = float(np.sqrt(1.0 + sig))
            i = 0
            while i < n:
                v = float(list_buf[i])
                if v < sky - siglo * sig or v > sky + sighi * sig:
                    n -= 1
                    list_buf[i] = list_buf[n]
                    xx = 1
                else:
                    i += 1
    return sky, n


def _compute_sky_dolphot_getsky_py(
    data: np.ndarray,
    rin2: int,
    rsky: int,
    rout2: int,
    step: int,
    siglo: float,
    sighi: float,
    dmin: float,
    dmax: float,
    progress: LoggedProgress | None = None,
) -> np.ndarray:
    """Pure NumPy/Python implementation of calcsky.c ``getsky`` (skip>0 path)."""
    ny, nx = int(data.shape[0]), int(data.shape[1])
    tsky = np.zeros((ny, nx), dtype=np.float64)
    cap = _calcsky_list_capacity(rsky, step)
    list_buf = np.empty(cap, dtype=np.float64)
    arr = np.asarray(data, dtype=np.float64)
    off_y, off_x = _calcsky_annulus_offsets(rin2, rout2, rsky, int(step))
    k_n = int(off_y.shape[0])
    # Log often enough for UX; throttling is inside LoggedProgress
    row_tick = max(1, ny // 200)

    for y in range(ny):
        for x in range(nx):
            n = 0
            for k in range(k_n):
                yy = y + int(off_y[k])
                xx = x + int(off_x[k])
                if 0 <= xx < nx and 0 <= yy < ny:
                    v = arr[yy, xx]
                    if v > dmin and v < dmax:
                        list_buf[n] = v
                        n += 1
            sky, _ = _dolphot_sigma_clip_iterate(list_buf, n, siglo, sighi)
            tsky[y, x] = sky
        if progress is not None and ((y + 1) % row_tick == 0 or y + 1 == ny):
            progress.update(y + 1)

    if progress is not None:
        progress.complete()
    t_s2 = time.monotonic()
    tsky = _quantize_stage1_tsky_like_calcsky_c(tsky)
    out = _dolphot_stage2_box_mean(tsky, int(step))
    if progress is not None:
        log.info(
            "calcsky stage2 (box mean) finished in %.2fs",
            time.monotonic() - t_s2,
        )
    return out.astype(np.float32)


def _use_numba_calcsky() -> bool:
    raw = os.environ.get("HST123_CALCSKY_NUMBA", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def _register_numba_calcsky():
    """Build Numba JIT kernels mirroring calcsky.c (optional dependency)."""
    import math

    from numba import njit, prange

    @njit(cache=True)
    def _clip_iter(list_buf, n_in, siglo, sighi):
        # Mean: same sequential sum as calcsky.c and _dolphot_sigma_clip_iterate.
        n = n_in
        xx = 1
        sky = 0.0
        while xx != 0:
            xx = 0
            if n == 0:
                sky = 0.0
            else:
                sky = 0.0
                for j in range(n):
                    sky += list_buf[j]
                sky /= n
                sig = 0.0
                for i in range(n):
                    d = list_buf[i] - sky
                    sig += d * d
                if n > 1:
                    sig = math.sqrt(1.0 + sig / (n - 1))
                else:
                    sig = math.sqrt(1.0 + sig)
                i = 0
                while i < n:
                    v = list_buf[i]
                    if v < sky - siglo * sig or v > sky + sighi * sig:
                        n -= 1
                        list_buf[i] = list_buf[n]
                        xx = 1
                    else:
                        i += 1
        return sky

    @njit(parallel=True, cache=True)
    def _stage1(data, off_y, off_x, siglo, sighi, dmin, dmax, scratch):
        ny, nx = data.shape
        k_n = off_y.shape[0]
        tsky = np.empty((ny, nx), dtype=np.float64)
        for y in prange(ny):
            list_buf = scratch[y]
            for x in range(nx):
                n = 0
                for k in range(k_n):
                    yy = y + off_y[k]
                    xx = x + off_x[k]
                    if 0 <= xx < nx and 0 <= yy < ny:
                        v = data[yy, xx]
                        if v > dmin and v < dmax:
                            list_buf[n] = v
                            n += 1
                sky = _clip_iter(list_buf, n, siglo, sighi)
                tsky[y, x] = sky
        return tsky

    return _stage1


_NUMBA_SKY_STAGES = None


def _get_numba_sky_stages():
    global _NUMBA_SKY_STAGES
    if _NUMBA_SKY_STAGES is None:
        _NUMBA_SKY_STAGES = _register_numba_calcsky()
    return _NUMBA_SKY_STAGES  # single JIT kernel: parallel stage-1 getsky


def compute_sky_map_dolphot(
    data: np.ndarray,
    *,
    r_in: int,
    r_out: int,
    step: int,
    sigma_low: float,
    sigma_high: float,
    dmin: float | None = None,
    dmax: float | None = None,
    progress: LoggedProgress | None = None,
) -> np.ndarray:
    """
    Sky map using the same algorithm as DOLPHOT ``calcsky`` (``getsky``, step>0).

    Parameters match the CLI: inner radius, outer radius, step, then σ multipliers
    as passed to the binary (``sigma_low`` → C ``argv[5]``/``sighi`` for the
    upper rejection ``> sky + sighi*σ``, ``sigma_high`` → ``siglo`` for the
    lower rejection ``< sky - siglo*σ``), each clamped to ≥ 1 like ``calcsky.c``.

    progress
        Optional :class:`~hst123.utils.progress_log.LoggedProgress` for stage 1.
        Caller should call ``start()`` before this function. With Numba, stage 1
        logs once at completion (full parallel pass). The pure-Python path still
        reports row progress. Stage 2 (box mean) logs its own timing when
        *progress* is set.
    """
    if dmin is None:
        dmin = _DEFAULT_CALCSKY_DMIN
    if dmax is None:
        dmax = _DEFAULT_CALCSKY_DMAX
    rin_i = max(0, int(r_in))
    rin2 = rin_i * rin_i
    r_out_arg = max(1, int(r_out))
    # C: rout2 from argv before rsky is aligned to a multiple of skip (calcsky.c).
    rout2 = r_out_arg * r_out_arg
    rsky, step_i = _calcsky_adjust_r_out_step(int(r_out), int(step))
    sighi = float(sigma_low)
    if sighi < 1.0:
        sighi = 1.0
    siglo = float(sigma_high)
    if siglo < 1.0:
        siglo = 1.0

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")

    if _use_numba_calcsky():
        try:
            cap = _calcsky_list_capacity(rsky, step_i)
            off_y, off_x = _calcsky_annulus_offsets(rin2, rout2, rsky, step_i)
            s1_full = _get_numba_sky_stages()
            ny, nx = arr.shape
            scratch = np.empty((ny, cap), dtype=np.float64)
            # Always one full parallel pass. Batching rows for per-batch progress
            # would run many sequential Numba kernels and destroy throughput.
            tsky = s1_full(arr, off_y, off_x, siglo, sighi, dmin, dmax, scratch)
            if progress is not None:
                progress.update(ny)
                progress.complete()
            t_stage2 = time.monotonic()
            tsky = _quantize_stage1_tsky_like_calcsky_c(tsky)
            out = _dolphot_stage2_box_mean(tsky, step_i)
            if progress is not None:
                log.info(
                    "calcsky stage2 (box mean) finished in %.2fs",
                    time.monotonic() - t_stage2,
                )
            return out.astype(np.float32)
        except Exception as exc:
            log.debug("Numba DOLPHOT sky failed (%s); using Python port", exc)

    return _compute_sky_dolphot_getsky_py(
        arr,
        rin2,
        rsky,
        rout2,
        step_i,
        siglo,
        sighi,
        dmin,
        dmax,
        progress,
    )


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

    By default uses a Python port of DOLPHOT ``calcsky.c`` ``getsky`` (step>0),
    matching the CLI algorithm and ``fits.param`` pixel limits. Set
    ``HST123_CALCSKY_LEGACY=1`` to restore the Photutils / ``median_filter``
    approximation.

    Parameters
    ----------
    image_fits, sky_fits
        Paths to the science FITS and output sky FITS.
    r_in, r_out, step, sigma_low, sigma_high
        Same arguments as the ``calcsky`` executable (see DOLPHOT User's Guide §4.1).
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

    legacy = os.environ.get("HST123_CALCSKY_LEGACY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    step_i = int(step)
    if step_i <= 0:
        log.warning(
            "calcsky step=%s (DOLPHOT quick/interpolation mode); "
            "using legacy sky fallback (getsky_q not ported)",
            step,
        )
        legacy = True

    if legacy:
        med = float(np.nanmedian(data))
        if not np.isfinite(med):
            med = 0.0
        work = np.where(np.isfinite(data), data, med)
        box = max(abs(step_i) * 4, int(r_out), 32)
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
    else:
        dmin, dmax = parse_calcsky_dataminmax(header)
        ny = int(data.shape[0])
        prog: LoggedProgress | None = None
        if calcsky_progress_enabled():
            prog = LoggedProgress(
                log,
                f"calcsky {os.path.basename(image_fits)}",
                ny,
                unit="rows",
            )
            prog.start()
        sky = compute_sky_map_dolphot(
            data,
            r_in=int(r_in),
            r_out=int(r_out),
            step=step_i,
            sigma_low=float(sigma_low),
            sigma_high=float(sigma_high),
            dmin=dmin,
            dmax=dmax,
            progress=prog,
        )

    hdu_out = fits.PrimaryHDU(data=sky, header=header)
    hdu_out.writeto(sky_fits, overwrite=True, output_verify="silentfix")
