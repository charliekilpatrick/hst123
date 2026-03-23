"""
Pure-Python DOLPHOT ``acsmask`` / ``wfc3mask`` / ``wfpc2mask`` (DOLPHOT 3.1).

Ports the science masking, PAM (or exposure) correction, and header updates from
the upstream C sources so hst123 does not require those binaries on ``PATH``.
PAM and PSF support files must still exist under the DOLPHOT source tree
(``.../acs/data``, ``.../wfc3/data``) as installed by ``hst123-install-dolphot``.

Set ``HST123_DOLPHOT_MASK_EXTERNAL=1`` to force the legacy C executables.

References: ``acs/acsmask.c``, ``wfc3/wfc3mask.c``, ``wfpc2/wfpc2mask.c``,
``wfpc2/wfpc2distort.c``, ``fits_lib.c`` (``safedown`` / ``safeup`` / ``insertcards``).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

# acspsfdata.h / wfc3psfdata.h
ACS_CTMULT = np.array([1.124583, 0.951763, 0.951763], dtype=np.float64)  # HRC, WFC1, WFC2
WFC3_CTMULT = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # IR, UVIS1, UVIS2


def safe_down(x: float) -> float:
    """DOLPHOT ``safedown`` (fits_lib.c)."""
    return float(x - 1.0e-5 * abs(x) - 1.0)


def safe_up(x: float) -> float:
    """DOLPHOT ``safeup`` (fits_lib.c)."""
    return float(x + 1.0e-5 * abs(x) + 1.0)


class UnsupportedMaskFormat(Exception):
    """File layout not recognized by the Python mask port."""


def _resolve_dolphot_tree(log: logging.Logger | None = None) -> Path:
    env = os.environ.get("HST123_DOLPHOT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "acs" / "data").is_dir() or (p / "Makefile").is_file():
            return p
    prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if prefix:
        for sub in (
            Path(prefix) / "opt" / "hst123-dolphot" / "dolphot3.1",
            Path(prefix) / "opt" / "hst123-dolphot",
        ):
            s = sub.resolve()
            if (s / "acs" / "data").is_dir():
                return s
            # Upstream ACS_WFC_PAM tarball used only dolphot2.0/acs/data (no acs/data yet).
            if (s / "dolphot2.0" / "acs" / "data").is_dir():
                return s
    raise FileNotFoundError(
        "DOLPHOT source tree not found (need acs/data for PAMs). "
        "Set HST123_DOLPHOT_ROOT or install via hst123-install-dolphot."
    )


def _hdr_float(h: fits.Header, key: str, default: float = 0.0) -> float:
    v = h.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _check_hst_sci_primary(prim: fits.Header, hdul: fits.HDUList, instrume: str) -> None:
    """Require HST + INSTRUME; FILETYPE must be SCI if present (MAST cal files may omit it)."""
    ft = _hdr_str(prim, "FILETYPE")
    if ft and ft.strip().upper() != "SCI":
        raise UnsupportedMaskFormat(f"Expected FILETYPE SCI, got {ft!r}")
    if _hdr_str(prim, "TELESCOP") != "HST":
        raise UnsupportedMaskFormat("Expected TELESCOP HST")
    if _either_string(hdul, "INSTRUME") != instrume:
        raise UnsupportedMaskFormat(f"Expected INSTRUME {instrume}")


def _hdr_str(h: fits.Header, key: str, default: str = "") -> str:
    v = h.get(key)
    if v is None:
        return default
    s = str(v).strip()
    if s.upper() in ("NONE", "N/A", ""):
        return default
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1].strip()
    return s


def _either_string(hdul: fits.HDUList, key: str) -> str:
    """Approximate acsmask ``eitherstring`` (HDRTAB then primary)."""
    for idx in (3, 2):
        if len(hdul) > idx:
            hdu = hdul[idx]
            if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None:
                names = hdu.columns.names
                col = None
                for cand in (key, key.upper(), key.lower()):
                    if cand in names:
                        col = cand
                        break
                if col is not None and len(hdu.data) > 0:
                    v = hdu.data[col][0]
                    if v is not None and str(v).strip():
                        return str(v).strip()
    return _hdr_str(hdul[0].header, key, "")


def _either_double(hdul: fits.HDUList, key: str) -> float:
    s = _either_string(hdul, key)
    if s:
        try:
            return float(s)
        except ValueError:
            pass
    return _hdr_float(hdul[0].header, key, 0.0)


def _ext_shape(hdu: fits.ImageHDU | fits.CompImageHDU | Any) -> tuple[int, int, int]:
    d = getattr(hdu, "data", None)
    if d is None:
        return 0, 0, 0
    if d.ndim == 2:
        return int(d.shape[1]), int(d.shape[0]), 1
    if d.ndim == 3:
        return int(d.shape[2]), int(d.shape[1]), int(d.shape[0])
    return 0, 0, 0


def _insert_dolphot_cards(
    header: fits.Header,
    *,
    gain: float,
    rn: float,
    exp: float,
    dmin: float,
    dmax: float,
    epoch: float,
    exp0: float,
    air: float = 0.0,
) -> None:
    """Match ``insertcards`` keyword names from default DOLPHOT ``fits.param``."""
    def _set(kw: str, val: float, comment: str) -> None:
        if val > -0.9e30:
            header[kw] = (val, comment)

    _set("GAIN", gain, "Gain (electrons/ADU)")
    _set("EXPTIME", exp, "Exposure time (s)")
    _set("RNOISE", rn, "Read noise (electrons)")
    _set("BADPIX", dmin, "Low bad data value (DN)")
    _set("SATURATE", dmax, "High bad data value (DN)")
    _set("MJD-OBS", epoch, "Epoch (MJD)")
    _set("AIRMASS", air, "Air mass")
    header["EXPTIME0"] = (exp0, "Single image exposure time (s)")


def _acs_mask_sci(
    sci: np.ndarray,
    dq: np.ndarray,
    dmin: float,
    dmax: float,
    *,
    mask_cr: bool,
    mask_sat: bool,
) -> tuple[np.ndarray, float, float]:
    """ACSmask drz=0 branch."""
    dmax1 = dmax * (2.0 if dmax > 0 else 0.5)
    dmin1 = dmin * (2.0 if dmin < 0 else 0.5)
    out = np.asarray(sci, dtype=np.float32).copy()
    dq_i = np.rint(np.asarray(dq, dtype=np.float64)).astype(np.int32)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            q = int(dq_i[y, x])
            if q & 2048:
                out[y, x] = safe_up(dmax1)
            elif (q & 256) and mask_sat:
                out[y, x] = safe_up(dmax1)
            elif q & 1727:
                out[y, x] = safe_down(dmin1)
            elif (q & 12288) and mask_cr:
                out[y, x] = safe_down(dmin1)
            elif out[y, x] >= dmax and mask_sat:
                out[y, x] = safe_up(dmax1)
            elif out[y, x] <= dmin:
                out[y, x] = safe_down(dmin1)
    return out, float(dmin1), float(dmax1)


def _acs_mask_drz(
    sci: np.ndarray,
    wht_or_ctx: np.ndarray,
    dmin: float,
    *,
    use_wht: bool,
) -> np.ndarray:
    out = np.asarray(sci, dtype=np.float32).copy()
    if use_wht:
        w = np.asarray(wht_or_ctx, dtype=np.float64)
        bad = w <= 0.0
    else:
        ctx = np.asarray(wht_or_ctx, dtype=np.float64)
        if ctx.ndim == 2:
            bad = np.rint(ctx) == 0
        else:
            bad = np.all(np.rint(ctx) == 0, axis=0)
    out[bad] = safe_down(dmin)
    return out


def _pam_mult(
    sci: np.ndarray,
    err: np.ndarray,
    pam: np.ndarray,
    mult: float,
    dmin: float,
    dmax: float,
    *,
    mask_sat: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """ACSpixcorr / WFC3_PAM_mult style."""
    dmax1 = 1.5 * dmax if dmax > 0 else 0.0
    dmin1 = 1.5 * dmin if dmin < 0 else 0.0
    s = np.asarray(sci, dtype=np.float32).copy()
    e = np.asarray(err, dtype=np.float32).copy()
    p = np.asarray(pam, dtype=np.float64)
    factor = p / mult
    for y in range(s.shape[0]):
        for x in range(s.shape[1]):
            v = float(s[y, x])
            if v > dmin and (v < dmax or not mask_sat):
                f = float(factor[y, x])
                s[y, x] = v * f
                e[y, x] = float(e[y, x]) * f
            elif v >= dmax:
                s[y, x] = safe_up(dmax1)
            else:
                s[y, x] = safe_down(dmin1)
    return s, e, float(dmin1), float(dmax1)


def _acs_get_rn(hdul: fits.HDUList, mode: int) -> float:
    """ACSgetcards readnoise from CCDAMP (MODE: 0=HRC, 1=WFC1 amps AB, 2=WFC2 amps CD, else quad)."""
    ccd = _either_string(hdul, "CCDAMP").upper()
    if ccd == "ABCD":
        if mode == 1:
            return 0.5 * (_either_double(hdul, "READNSEA") + _either_double(hdul, "READNSEB"))
        if mode == 2:
            return 0.5 * (_either_double(hdul, "READNSEC") + _either_double(hdul, "READNSED"))
        return 0.25 * (
            _either_double(hdul, "READNSEA")
            + _either_double(hdul, "READNSEB")
            + _either_double(hdul, "READNSEC")
            + _either_double(hdul, "READNSED")
        )
    if ccd == "A":
        return _either_double(hdul, "READNSEA")
    if ccd == "B":
        return _either_double(hdul, "READNSEB")
    if ccd == "C":
        return _either_double(hdul, "READNSEC")
    if ccd == "D":
        return _either_double(hdul, "READNSED")
    raise UnsupportedMaskFormat(f"Unknown ACS CCDAMP={ccd!r}")


def _acs_ncombine(hdul: fits.HDUList) -> int:
    prim = hdul[0].header
    s = str(prim.get("NRPTEXP", "") or "").strip()
    nc = int(float(s)) if s else 1
    if not s:
        log.warning("NRPTEXP not found; assuming 1")
    s2 = str(prim.get("CRSPLIT", "") or "").strip()
    if not s2:
        log.warning("CRSPLIT not found; assuming 1")
    else:
        nc *= int(float(s2))
    return max(1, nc)


def _acs_dmin_dmax_from_sci(hdu: fits.ImageHDU) -> tuple[float, float]:
    d = np.asarray(hdu.data, dtype=np.float64)
    return float(np.min(d)), float(np.max(d))


def _acs_dmin_dmax_good(hdu: fits.ImageHDU) -> tuple[float, float]:
    h = hdu.header
    return safe_down(_hdr_float(h, "GOODMIN", -1.0)), safe_up(_hdr_float(h, "GOODMAX", 1.0))


def _load_pam_slice(tree: Path, rel: str, ny: int, nx: int, oy: int, ox: int) -> np.ndarray:
    path = tree / rel.replace("/", os.sep)
    # Upstream ACS_WFC_PAM.tar.gz still uses a dolphot2.0/... path inside the tarball.
    if not path.is_file():
        legacy = tree / "dolphot2.0" / rel.replace("/", os.sep)
        if legacy.is_file():
            path = legacy
    if not path.is_file():
        raise FileNotFoundError(path)
    with fits.open(path, memmap=False) as pam_hdul:
        pdata = np.asarray(pam_hdul[0].data, dtype=np.float64)
        if pdata.ndim == 2:
            sl = pdata[oy : oy + ny, ox : ox + nx]
        else:
            sl = pdata[0, oy : oy + ny, ox : ox + nx]
    if sl.shape != (ny, nx):
        raise ValueError(f"PAM slice shape {sl.shape} != {(ny, nx)} for {path}")
    return sl


def _acs_process_chip_triplet(
    hdul: fits.HDUList,
    sci_idx: int,
    mode: int,
    drz: bool,
    pam_rel: str,
    ct_idx: int,
    tree: Path,
    offsety: int,
    offsetx: int,
    *,
    fixed_et: float | None,
    fixed_nc: int | None,
    use_wht: bool,
    mask_cr: bool,
    mask_sat: bool,
) -> fits.ImageHDU:
    """Return updated SCI HDU; caller removes ERR/DQ from hdul."""
    prim = hdul[0].header
    exp = float(fixed_et) if fixed_et and fixed_et > 0 else _hdr_float(prim, "EXPTIME", 1.0)
    nc = int(fixed_nc) if fixed_nc and fixed_nc > 0 else _acs_ncombine(hdul)
    rn = _acs_get_rn(hdul, mode)
    epoch = 0.5 * (_hdr_float(prim, "EXPSTART", 0.0) + _hdr_float(prim, "EXPEND", 0.0))
    sci_h = hdul[sci_idx]
    err_h = hdul[sci_idx + 1]
    dq_h = hdul[sci_idx + 2]
    if drz:
        dmin, dmax = _acs_dmin_dmax_from_sci(sci_h)
    else:
        dmin, dmax = _acs_dmin_dmax_good(sci_h)

    sci = sci_h.data
    err = err_h.data
    dq = dq_h.data
    ny, nx = sci.shape
    if drz:
        wht = hdul[sci_idx + 1].data
        ctx = hdul[sci_idx + 2].data
        arr = wht if use_wht else ctx
        sci_m = _acs_mask_drz(sci, arr, dmin, use_wht=use_wht)
        mult = float(ACS_CTMULT[ct_idx])
        sci_m = sci_m * (exp / mult)
        dmin = dmin * exp / mult
        dmax = dmax * exp / mult
        new_h = fits.ImageHDU(data=sci_m.astype(np.float32), header=sci_h.header.copy())
        exp0 = exp / nc
        _insert_dolphot_cards(
            new_h.header,
            gain=1.0,
            rn=rn * np.sqrt(nc),
            exp=exp,
            dmin=dmin,
            dmax=dmax,
            epoch=epoch,
            exp0=exp0,
        )
        return new_h

    sci_m, dmin, dmax = _acs_mask_sci(sci, dq, dmin, dmax, mask_cr=mask_cr, mask_sat=mask_sat)
    pam = _load_pam_slice(tree, os.path.join("acs", "data", pam_rel), ny, nx, offsety, offsetx)
    sci_f, err_f, dmin, dmax = _pam_mult(
        sci_m, err, pam, float(ACS_CTMULT[ct_idx]), dmin, dmax, mask_sat=mask_sat
    )
    exp0 = exp / nc
    new_h = fits.ImageHDU(data=sci_f, header=sci_h.header.copy())
    _insert_dolphot_cards(
        new_h.header,
        gain=1.0,
        rn=rn * np.sqrt(nc),
        exp=exp,
        dmin=dmin,
        dmax=dmax,
        epoch=epoch,
        exp0=exp0,
    )
    return new_h


def _classify_acs(hdul: fits.HDUList) -> tuple[int, int, int]:
    """
    Return (tp, offsetx, offsety) per ``ACStype``. tp=0 unknown.
    Simplified vs full C: covers standard WFC/HRC full-frame and drizzle cases.
    """
    n = len(hdul) - 1
    ex = [_ext_shape(hdul[i]) for i in range(1, len(hdul))]
    det = _either_string(hdul, "DETECTOR")
    apt = _either_string(hdul, "APERTURE")

    if det == "WFC":
        if n >= 6 and all(ex[i] == (4096, 2048, 1) for i in range(6)):
            return 1, 0, 0
        if n in (3, 4, 5) and ex[0][2] == 1 and ex[1][:2] == ex[0][:2]:
            if ex[0][0] > 0 and (n == 3 or ex[2][:2] == ex[0][:2] or ex[2][0] == ex[0][0]):
                log.info("Irregular size; assuming drizzled (ACS)")
                return 2, 0, 0
    if det == "HRC":
        if n >= 3 and all(ex[i] == (1024, 1024, 1) for i in range(3)):
            return 3, 0, 0
        if n in (3, 4) and ex[0][2] == 1 and ex[1][:2] == ex[0][:2]:
            log.info("Irregular size; assuming drizzled (ACS HRC)")
            return 4, 0, 0
    return 0, 0, 0


def apply_acsmask(
    path: str | Path,
    tree: Path | None = None,
    *,
    mask_cr: bool = True,
    mask_sat: bool = True,
    fixed_et: float | None = None,
    fixed_nc: int | None = None,
    use_wht: bool = False,
) -> None:
    path = Path(path).resolve()
    t = tree or _resolve_dolphot_tree(log)
    with fits.open(path, mode="readonly") as hdul:
        if _hdr_str(hdul[0].header, "DOL_ACS"):
            log.info("%s already has DOL_ACS; skipping acsmask", path.name)
            return
        tp, ox, oy = _classify_acs(hdul)
        if tp == 0:
            raise UnsupportedMaskFormat("Unrecognized ACS layout for Python acsmask")
        hdul_r = fits.HDUList([h.copy() for h in hdul])

    prim = hdul_r[0]
    _check_hst_sci_primary(prim.header, hdul_r, "ACS")

    new_hdus: list[Any] = [prim]

    if tp == 1:
        # WFC2 chip first triplet, then WFC1
        h1 = _acs_process_chip_triplet(
            hdul_r, 1, 2, False, "wfc2_pam.fits", 2, t, oy, ox,
            fixed_et=fixed_et, fixed_nc=fixed_nc, use_wht=use_wht,
            mask_cr=mask_cr, mask_sat=mask_sat,
        )
        h1.header["DOL_ACS"] = (2, "DOLPHOT ACS tag")
        new_hdus.append(h1)
        # Drop processed ERR,DQ; shift indices: old 4,5,6 -> 1,2,3 in hdul_r
        hdul_r = fits.HDUList([prim, hdul_r[4], hdul_r[5], hdul_r[6]])
        h2 = _acs_process_chip_triplet(
            hdul_r, 1, 1, False, "wfc1_pam.fits", 1, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, use_wht=use_wht,
            mask_cr=mask_cr, mask_sat=mask_sat,
        )
        h2.header["DOL_ACS"] = (1, "DOLPHOT ACS tag")
        new_hdus.append(h2)
        out = fits.HDUList(new_hdus)
        out.writeto(path, overwrite=True, output_verify="warn")
        return

    if tp == 2:
        h = _acs_process_chip_triplet(
            hdul_r, 1, -1, True, "", 1, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, use_wht=use_wht,
            mask_cr=mask_cr, mask_sat=mask_sat,
        )
        h.header["DOL_ACS"] = (-2, "DOLPHOT ACS tag")
        fits.HDUList([prim, h]).writeto(path, overwrite=True, output_verify="warn")
        return

    if tp == 3:
        h = _acs_process_chip_triplet(
            hdul_r, 1, 0, False, "hrc_pam.fits", 0, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, use_wht=use_wht,
            mask_cr=mask_cr, mask_sat=mask_sat,
        )
        h.header["DOL_ACS"] = (0, "DOLPHOT ACS tag")
        fits.HDUList([prim, h]).writeto(path, overwrite=True, output_verify="warn")
        return

    if tp == 4:
        h = _acs_process_chip_triplet(
            hdul_r, 1, -2, True, "", 0, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, use_wht=use_wht,
            mask_cr=mask_cr, mask_sat=mask_sat,
        )
        h.header["DOL_ACS"] = (-1, "DOLPHOT ACS tag")
        fits.HDUList([prim, h]).writeto(path, overwrite=True, output_verify="warn")
        return

    raise UnsupportedMaskFormat(f"ACS tp={tp} not implemented in Python port")


# --- WFC3 -----------------------------------------------------------------

def _wfc3_get_rn_uvis(hdul: fits.HDUList, mode: int) -> float:
    return _acs_get_rn(hdul, mode)


def _wfc3_get_rn_ir(hdul: fits.HDUList) -> float:
    ns = int(_hdr_float(hdul[0].header, "NSAMP", 1))
    rn = 14.6 * np.sqrt((12.0 * (ns - 1)) / (ns * (ns + 1)))
    log.info("WFC3 IR read noise %f from NSAMP=%d", rn, ns)
    return float(rn)


def _wfc3_dq_mask(
    sci: np.ndarray,
    dq: np.ndarray,
    dmin: float,
    dmax: float,
    *,
    drz: bool,
    ir: bool,
    mask_cr: bool,
    mask_sat: bool,
    use_flat_flag: bool,
    use_wht: bool,
) -> tuple[np.ndarray, float, float] | np.ndarray:
    if not drz:
        dmax1 = dmax * (2.0 if dmax > 0 else 0.5)
        dmin1 = dmin * (2.0 if dmin < 0 else 0.5)
        out = np.asarray(sci, dtype=np.float32).copy()
        dq_i = np.rint(np.asarray(dq, dtype=np.float64)).astype(np.int32)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                q = int(dq_i[y, x])
                if q & 2048 or ((q & 256) and mask_sat):
                    out[y, x] = safe_up(dmax1)
                elif q & 191 or ((q & 512) and (use_flat_flag or not ir)) or ((q & 1024) and not ir):
                    out[y, x] = safe_down(dmin1)
                elif (q & 12288) and mask_cr:
                    out[y, x] = safe_down(dmin1)
                elif out[y, x] >= dmax and mask_sat:
                    out[y, x] = safe_up(dmax1)
                elif out[y, x] <= dmin:
                    out[y, x] = safe_down(dmin1)
        return out, float(dmin1), float(dmax1)

    out = np.asarray(sci, dtype=np.float32).copy()
    if use_wht:
        bad = np.asarray(dq, dtype=np.float64) <= 0.0
    else:
        ctx = np.asarray(dq, dtype=np.float64)
        if ctx.ndim == 2:
            bad = np.rint(ctx) == 0
        else:
            bad = np.all(np.rint(ctx) == 0, axis=0)
    out[bad] = safe_down(dmin)
    return out


def _classify_wfc3(hdul: fits.HDUList) -> tuple[int, int, int]:
    n = len(hdul) - 1
    ex = [_ext_shape(hdul[i]) for i in range(1, len(hdul))]
    det = _either_string(hdul, "DETECTOR")
    if det == "UVIS":
        if n >= 6 and all(ex[i] == (4096, 2051, 1) for i in range(6)):
            return 1, 0, 0
        if n in (3, 4) and ex[0][2] == 1 and ex[1][:2] == ex[0][:2]:
            log.info("Irregular size; assuming drizzled (WFC3 UVIS)")
            return 2, 0, 0
    if det == "IR":
        if n >= 5 and all(ex[i] == (1014, 1014, 1) for i in range(5)):
            return 3, 0, 0
        if n in (3, 4) and ex[0][2] == 1 and ex[1][:2] == ex[0][:2]:
            log.info("Irregular size; assuming drizzled (WFC3 IR)")
            return 4, 0, 0
    return 0, 0, 0


def _wfc3_chip_pipeline(
    hdul: fits.HDUList,
    sci_idx: int,
    uvis_mode: int,
    drz: bool,
    ir: bool,
    pam_name: str,
    ct_idx: int,
    tree: Path,
    oy: int,
    ox: int,
    *,
    fixed_et: float | None,
    fixed_nc: int | None,
    mask_cr: bool,
    mask_sat: bool,
    use_flat: bool,
    use_wht: bool,
) -> fits.ImageHDU:
    prim = hdul[0].header
    exp = float(fixed_et) if fixed_et and fixed_et > 0 else _hdr_float(prim, "EXPTIME", 1.0)
    if ir:
        rn = _wfc3_get_rn_ir(hdul)
    else:
        rn = _wfc3_get_rn_uvis(hdul, uvis_mode)
    epoch = 0.5 * (_hdr_float(prim, "EXPSTART", 0.0) + _hdr_float(prim, "EXPEND", 0.0))
    sci_h = hdul[sci_idx]
    err_h = hdul[sci_idx + 1]
    dq_h = hdul[sci_idx + 2]
    if drz:
        dmin, dmax = _acs_dmin_dmax_from_sci(sci_h)
    else:
        dmin, dmax = _acs_dmin_dmax_good(sci_h)

    ncomb_str = str(prim.get("NCOMBINE", "") or "").strip()
    if fixed_nc and fixed_nc > 0:
        nc = int(fixed_nc)
    elif ncomb_str:
        nc = max(1, int(float(ncomb_str)))
    else:
        nc = 1
        log.warning("NCOMBINE not found; assuming 1")
    exp0 = exp / nc

    sci = sci_h.data
    err = err_h.data
    dq = dq_h.data
    ny, nx = sci.shape

    if drz:
        wht = err_h.data
        ctx = dq_h.data
        arr = wht if use_wht else ctx
        sci_m = _wfc3_dq_mask(
            sci, arr, dmin, dmax, drz=True, ir=ir,
            mask_cr=mask_cr, mask_sat=mask_sat, use_flat_flag=use_flat, use_wht=use_wht,
        )
        mult = float(WFC3_CTMULT[ct_idx])
        sci_m = sci_m * (exp / mult)
        dmin *= exp / mult
        dmax *= exp / mult
        new_h = fits.ImageHDU(data=np.asarray(sci_m, dtype=np.float32), header=sci_h.header.copy())
        _insert_dolphot_cards(
            new_h.header, 1.0, rn * np.sqrt(nc), exp, dmin, dmax, epoch, exp0
        )
        return new_h

    r = _wfc3_dq_mask(
        sci, dq, dmin, dmax, drz=False, ir=ir,
        mask_cr=mask_cr, mask_sat=mask_sat, use_flat_flag=use_flat, use_wht=False,
    )
    assert isinstance(r, tuple)
    sci_m, dmin, dmax = r
    pam = _load_pam_slice(tree, os.path.join("wfc3", "data", pam_name), ny, nx, oy, ox)
    sci_f, err_f, dmin, dmax = _pam_mult(
        sci_m, err, pam, float(WFC3_CTMULT[ct_idx]), dmin, dmax, mask_sat=mask_sat
    )
    if ir:
        mult = 1.0
        sci_f = sci_f * (exp / mult)
        err_f = err_f * (exp / mult)
        dmin *= exp / mult
        dmax *= exp / mult
    new_h = fits.ImageHDU(data=sci_f, header=sci_h.header.copy())
    _insert_dolphot_cards(
        new_h.header, 1.0, rn * np.sqrt(nc), exp, dmin, dmax, epoch, exp0
    )
    return new_h


def apply_wfc3mask(
    path: str | Path,
    tree: Path | None = None,
    *,
    mask_cr: bool = True,
    mask_sat: bool = True,
    fixed_et: float | None = None,
    fixed_nc: int | None = None,
    use_wht: bool = False,
    use_flat: bool = False,
) -> None:
    path = Path(path).resolve()
    t = tree or _resolve_dolphot_tree(log)
    with fits.open(path, mode="readonly") as hdul:
        if _hdr_str(hdul[0].header, "DOL_WFC3"):
            log.info("%s already has DOL_WFC3; skipping wfc3mask", path.name)
            return
        tp, _, _ = _classify_wfc3(hdul)
        if tp == 0:
            raise UnsupportedMaskFormat("Unrecognized WFC3 layout for Python wfc3mask")
        hdul_r = fits.HDUList([h.copy() for h in hdul])

    prim = hdul_r[0]
    _check_hst_sci_primary(prim.header, hdul_r, "WFC3")

    if tp == 1:
        h2 = _wfc3_chip_pipeline(
            hdul_r, 1, 2, False, False, "UVIS2wfc3_map.fits", 2, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, mask_cr=mask_cr, mask_sat=mask_sat,
            use_flat=use_flat, use_wht=use_wht,
        )
        h2.header["DOL_WFC3"] = (2, "DOLPHOT WFC3 tag")
        hdul_r = fits.HDUList([prim, hdul_r[4], hdul_r[5], hdul_r[6]])
        h1 = _wfc3_chip_pipeline(
            hdul_r, 1, 1, False, False, "UVIS1wfc3_map.fits", 1, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, mask_cr=mask_cr, mask_sat=mask_sat,
            use_flat=use_flat, use_wht=use_wht,
        )
        h1.header["DOL_WFC3"] = (1, "DOLPHOT WFC3 tag")
        fits.HDUList([prim, h1, h2]).writeto(path, overwrite=True, output_verify="warn")
        return

    if tp == 2:
        h = _wfc3_chip_pipeline(
            hdul_r, 1, -1, True, False, "", 1, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, mask_cr=mask_cr, mask_sat=mask_sat,
            use_flat=use_flat, use_wht=use_wht,
        )
        h.header["DOL_WFC3"] = (-2, "DOLPHOT WFC3 tag")
        fits.HDUList([prim, h]).writeto(path, overwrite=True, output_verify="warn")
        return

    if tp == 3:
        h = _wfc3_chip_pipeline(
            hdul_r, 1, 0, False, True, "ir_wfc3_map.fits", 0, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, mask_cr=mask_cr, mask_sat=mask_sat,
            use_flat=use_flat, use_wht=use_wht,
        )
        h.header["DOL_WFC3"] = (0, "DOLPHOT WFC3 tag")
        # IR: drop extensions 2–5 (ERR,DQ + extra) — mimic WFC3setcards removing 4 HDUs
        fits.HDUList([prim, h]).writeto(path, overwrite=True, output_verify="warn")
        return

    if tp == 4:
        h = _wfc3_chip_pipeline(
            hdul_r, 1, -2, True, True, "", 0, t, 0, 0,
            fixed_et=fixed_et, fixed_nc=fixed_nc, mask_cr=mask_cr, mask_sat=mask_sat,
            use_flat=use_flat, use_wht=use_wht,
        )
        h.header["DOL_WFC3"] = (-1, "DOLPHOT WFC3 tag")
        fits.HDUList([prim, h]).writeto(path, overwrite=True, output_verify="warn")
        return

    raise UnsupportedMaskFormat(f"WFC3 tp={tp} not implemented")


# --- WFPC2 geometry (wfpc2distort.c) --------------------------------------

CC = np.array(
    [
        [3.54356e02, -8.12003e02, -8.07068e02, 7.72904e02],
        [1.00021e00, 1.95172e-02, -2.18757e00, 1.37619e-02],
        [9.79758e-04, -2.18805e00, -8.51284e-03, 2.18899e00],
        [9.84222e-08, -1.51015e-06, -3.21458e-07, 1.33326e-06],
        [-6.31327e-09, 5.06693e-06, 2.77459e-06, -4.00828e-06],
        [-7.19983e-07, 1.04063e-06, -2.06874e-06, -5.71282e-07],
        [-3.73922e-08, -4.55094e-10, 7.39223e-08, -2.00520e-09],
        [6.65101e-10, 7.50249e-08, -1.18819e-09, -7.86673e-08],
        [-3.51470e-08, -1.51652e-09, 7.73056e-08, -3.37491e-09],
        [-2.55003e-09, 7.34568e-08, -3.72971e-10, -7.81490e-08],
    ],
    dtype=np.float64,
)
DD = np.array(
    [
        [3.43646e02, 7.66592e02, -7.71489e02, -7.74638e02],
        [1.00544e-03, 2.18607e00, 7.40319e-03, -2.18688e00],
        [9.99786e-01, 2.02006e-02, -2.18554e00, 1.29831e-02],
        [-5.80870e-07, -3.79469e-06, -1.49869e-06, 1.30814e-06],
        [-5.07221e-07, -2.99518e-06, 2.96650e-06, 2.01760e-06],
        [3.41574e-07, 1.05789e-07, -2.02756e-07, -8.47656e-07],
        [-1.08091e-09, -7.47857e-08, -1.45720e-09, 7.70408e-08],
        [-3.42070e-08, -1.67666e-09, 7.72421e-08, -1.68514e-09],
        [1.17819e-09, -7.55302e-08, 8.52633e-10, 7.75257e-08],
        [-4.48966e-08, 3.02579e-10, 7.68023e-08, -3.69212e-10],
    ],
    dtype=np.float64,
)


def _wfpc2_fwddistort(c: int, x: float, y: float) -> tuple[float, float]:
    """Chip pixel (x,y) -> undistorted frame; ``wfpc2distort.c`` ``WFPC2fwddistort``."""
    x0 = x - 399.5
    y0 = y - 399.5
    xv = (
        CC[0, c]
        + CC[1, c] * x0
        + CC[2, c] * y0
        + CC[3, c] * x0 * x0
        + CC[4, c] * x0 * y0
        + CC[5, c] * y0 * y0
        + CC[6, c] * x0**3
        + CC[7, c] * x0 * x0 * y0
        + CC[8, c] * x0 * y0 * y0
        + CC[9, c] * y0**3
    )
    yv = (
        DD[0, c]
        + DD[1, c] * x0
        + DD[2, c] * y0
        + DD[3, c] * x0 * x0
        + DD[4, c] * x0 * y0
        + DD[5, c] * y0 * y0
        + DD[6, c] * x0**3
        + DD[7, c] * x0 * x0 * y0
        + DD[8, c] * x0 * y0 * y0
        + DD[9, c] * y0**3
    )
    return float(xv), float(yv)


def _wfpc2_xsize(c: int, x: float, y: float) -> float:
    x1, y1 = x - 0.5, y
    x2, y2 = x + 0.5, y
    x1, y1 = _wfpc2_fwddistort(c, x1, y1)
    x2, y2 = _wfpc2_fwddistort(c, x2, y2)
    dist = np.hypot(x1 - x2, y1 - y2)
    return float(dist / np.hypot(CC[1, c], DD[1, c]))


def _wfpc2_ysize(c: int, x: float, y: float) -> float:
    x1, y1 = x, y - 0.5
    x2, y2 = x, y + 0.5
    x1, y1 = _wfpc2_fwddistort(c, x1, y1)
    x2, y2 = _wfpc2_fwddistort(c, x2, y2)
    dist = np.hypot(x1 - x2, y1 - y2)
    return float(dist / np.hypot(CC[2, c], DD[2, c]))


def _wfpc2_pixcorr_mult(hdul: fits.HDUList, dmin: float, dmax: float, mask_sat: bool) -> None:
    dmax1 = 1.5 * dmax if dmax > 0 else 0.0
    dmin1 = 1.5 * dmin if dmin < 0 else 0.0
    for ext_i in range(1, len(hdul)):
        h = hdul[ext_i]
        if not isinstance(h, (fits.ImageHDU, fits.CompImageHDU)) or h.data is None:
            continue
        d = np.asarray(h.data, dtype=np.float32)
        c = ext_i - 1
        ny, nx = d.shape[-2], d.shape[-1]
        if d.ndim == 2:
            for y in range(ny):
                for x in range(nx):
                    v = float(d[y, x])
                    if v > dmin and (v < dmax or not mask_sat):
                        d[y, x] = v * _wfpc2_xsize(c, x, y) * _wfpc2_ysize(c, x, y)
                    elif v >= dmax:
                        d[y, x] = safe_up(dmax1)
                    else:
                        d[y, x] = safe_down(dmin1)
            h.data = d
        else:
            for z in range(d.shape[0]):
                for y in range(ny):
                    for x in range(nx):
                        v = float(d[z, y, x])
                        if v > dmin and (v < dmax or not mask_sat):
                            d[z, y, x] = v * _wfpc2_xsize(z, x, y) * _wfpc2_ysize(z, x, y)
                        elif v >= dmax:
                            d[z, y, x] = safe_up(dmax1)
                        else:
                            d[z, y, x] = safe_down(dmin1)
            h.data = d


def _wfpc2_insert_cards(header: fits.Header, gain: float, rn: float, exp: float, dmin: float, dmax: float, epoch: float, exp0: float) -> None:
    header["GAIN"] = (gain, "Gain (e-/DN)")
    header["RNOISE"] = (rn, "Read noise (e-)")
    header["EXPTIME"] = (exp, "Exposure time (s)")
    header["BADPIX"] = (dmin, "Low bad data value")
    header["SATURATE"] = (dmax, "High bad data value")
    header["MJD-OBS"] = (epoch, "Epoch (MJD)")
    header["EXPTIME0"] = (exp0, "Single exposure (s)")


def apply_wfpc2mask(
    sci_path: str | Path,
    dq_path: str | Path | None,
) -> None:
    """WFPC2 mask: needs science + data-quality FITS for multi-extension stacks."""
    sci_path = Path(sci_path).resolve()
    with fits.open(sci_path, mode="readonly") as hdul:
        if _hdr_str(hdul[0].header, "DOLWFPC2"):
            log.info("%s already has DOLWFPC2; skipping wfpc2mask", sci_path.name)
            return
        prim = hdul[0]
        _check_hst_sci_primary(prim.header, hdul, "WFPC2")

        n_ext = len(hdul) - 1
        tp = 0
        if prim.data is not None and prim.data.shape == (4, 800, 800):
            tp = 1
        elif n_ext >= 4:
            ex = [_ext_shape(hdul[i]) for i in range(1, 5)]
            if all(e == (800, 800, 1) for e in ex):
                tp = 2

        if tp == 0 and n_ext >= 3:
            e0 = _ext_shape(hdul[1])
            if e0[2] == 1 and n_ext >= 3:
                log.info("Irregular size; assuming drizzled WFPC2")
                tp = -1

        if tp == 0:
            raise UnsupportedMaskFormat("Unrecognized WFPC2 layout")

        hdul_w = fits.HDUList([h.copy() for h in hdul])

    if tp > 0:
        if not dq_path or not Path(dq_path).is_file():
            raise FileNotFoundError("WFPC2 native format requires a data quality FITS")
        with fits.open(dq_path) as dq_h:
            dq_w = fits.HDUList([h.copy() for h in dq_h])

    prim = hdul_w[0]
    gain = _hdr_float(prim.header, "ATODGAIN", 0.0)
    if gain == 7:
        rn = 5.0
    elif gain in (14, 15):
        rn = 7.5
        gain = 14.0
    else:
        raise UnsupportedMaskFormat(f"Unsupported WFPC2 ATODGAIN={gain}")
    exp = _hdr_float(prim.header, "EXPTIME", 1.0)
    dmin = safe_down(_hdr_float(prim.header, "RSDPFILL", -1.0))
    dmax = safe_up(_hdr_float(prim.header, "SATURATE", 1.0e30))
    epoch = 0.5 * (_hdr_float(prim.header, "EXPSTART", 0.0) + _hdr_float(prim.header, "EXPEND", 0.0))

    if tp == 1:
        raise UnsupportedMaskFormat(
            "WFPC2 800×800×4 primary not supported in Python port; use splitgroups or external wfpc2mask"
        )

    if tp == 2:
        for z in range(4):
            sci = np.asarray(hdul_w[z + 1].data, dtype=np.float32)
            dq = np.asarray(dq_w[z + 1].data, dtype=np.float64)
            out = np.empty_like(sci)
            ny, nx = sci.shape
            dq_i = np.rint(dq).astype(np.int32)
            for y in range(ny):
                for x in range(nx):
                    idq = int(dq_i[y, x])
                    v = float(sci[y, x])
                    if idq & 8 or v > 3500:
                        out[y, x] = safe_up(dmax)
                    elif idq & 5047 or y == ny - 1 or x == nx - 1:
                        out[y, x] = safe_down(dmin)
                    else:
                        out[y, x] = v
            hdul_w[z + 1].data = out
        _wfpc2_pixcorr_mult(hdul_w, dmin, dmax, mask_sat=True)
        for i in range(4):
            _wfpc2_insert_cards(
                hdul_w[i + 1].header, gain, rn, exp, dmin, dmax, epoch, exp
            )
            hdul_w[i + 1].header["DOLWFPC2"] = (i, "DOLPHOT WFPC2 tag")
        hdul_w.writeto(sci_path, overwrite=True, output_verify="warn")
        return

    if tp == -1:
        # Drizzled single SCI + WHT
        sci_h = hdul_w[1]
        wht_h = hdul_w[2] if len(hdul_w) > 2 else None
        s = np.asarray(sci_h.data, dtype=np.float32)
        if wht_h is not None and wht_h.data is not None:
            w = np.asarray(wht_h.data, dtype=np.float64)
            s = s.copy()
            s[w == 0.0] = safe_down(dmin)
        mult = 1.0
        dmax_n = dmax * exp / mult
        dmin_n = dmin * exp / mult
        s = s * (exp / mult)
        sci_h.data = s.astype(np.float32)
        _wfpc2_insert_cards(sci_h.header, gain, rn, exp, dmin_n, dmax_n, epoch, exp)
        sci_h.header["DOLWFPC2"] = (-1, "DOLPHOT WFPC2 tag")
        # drop extra extensions
        fits.HDUList([prim, sci_h]).writeto(sci_path, overwrite=True, output_verify="warn")
        return

    raise UnsupportedMaskFormat(f"WFPC2 tp={tp}")


def apply_dolphot_mask_instrument(
    instrument: str,
    image: str | Path,
    dq_image: str | Path | None,
    *,
    log_: logging.Logger | None = None,
) -> None:
    """Run Python mask for ``acs`` | ``wfc3`` | ``wfpc2``."""
    lg = log_ or log
    inst = instrument.lower().strip()
    tree = _resolve_dolphot_tree(lg)
    if inst == "acs":
        apply_acsmask(image, tree)
    elif inst == "wfc3":
        apply_wfc3mask(image, tree)
    elif inst == "wfpc2":
        apply_wfpc2mask(image, dq_image)
    else:
        raise ValueError(f"Unknown instrument for mask: {instrument!r}")
