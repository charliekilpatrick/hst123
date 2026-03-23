"""
Pure-Python DOLPHOT ``splitgroups`` (DOLPHOT 3.x style).

Writes one FITS per science plane: ``<root>.chipN.fits`` with ``EXTNAME=SCI`` on
the primary HDU, merged primary + extension metadata (WCS, photometry), matching
what :func:`hst123.primitives.run_dolphot.run_dolphot_primitive.DolphotPrimitive.split_groups`
expects.

References: DOLPHOT ``splitgroups`` utility (instrument Makefiles); behavior
aligned with ``needs_to_split_groups`` / ``prepare_dolphot`` (``.chip?.fits``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

__all__ = ["apply_splitgroups", "count_expected_split_outputs", "SplitgroupsError"]

log = logging.getLogger(__name__)


class SplitgroupsError(RuntimeError):
    """Input FITS cannot be split with the Python implementation."""


_SKIP_SCI_KEYS = frozenset(
    {
        "XTENSION",
        "PCOUNT",
        "GCOUNT",
        "EXTNAME",
        "EXTVER",
        "INHERIT",
        "IMAGEID",
    }
)


def _is_sci_hdu(hdu: Any) -> bool:
    name = getattr(hdu, "name", None) or ""
    return name.upper() == "SCI"


def _primary_3d_chip_planes(phdu: fits.PrimaryHDU) -> list[np.ndarray]:
    """
    WFPC2-style data cube on PRIMARY: (Ny, Nx, 4) or (4, Ny, Nx).
    Returns list of 2D float arrays (views/copies as appropriate).
    """
    d = phdu.data
    if d is None:
        return []
    arr = np.asarray(d)
    if arr.ndim != 3:
        return []
    if arr.shape[2] == 4:
        return [np.asarray(arr[:, :, k]) for k in range(4)]
    if arr.shape[0] == 4 and arr.shape[1] == arr.shape[2]:
        return [np.asarray(arr[k]) for k in range(4)]
    return []


def _n_primary_3d_planes_from_header(hdr: fits.Header) -> int:
    """Infer 4 WFPC2-style planes from PRIMARY header without loading the array."""
    if hdr.get("NAXIS") != 3:
        return 0
    n1 = int(hdr.get("NAXIS1", 0) or 0)
    n2 = int(hdr.get("NAXIS2", 0) or 0)
    n3 = int(hdr.get("NAXIS3", 0) or 0)
    if n3 == 4:
        return 4
    if n1 == 4 and n2 > 0 and n2 == n3:
        return 4
    return 0


def count_expected_split_outputs(path: str | Path) -> int:
    """
    Number of per-chip files Python splitgroups would write (SCI count or 3-D primary).
    """
    p = Path(path)
    with fits.open(p, mode="readonly", memmap=False) as hdul:
        n_sci = sum(1 for h in hdul if _is_sci_hdu(h))
        if n_sci > 0:
            return n_sci
        if len(hdul) == 0:
            return 0
        return _n_primary_3d_planes_from_header(hdul[0].header)


def _merge_headers(primary_hdr: fits.Header, sci_hdr: fits.Header | None) -> fits.Header:
    """Build a primary header: global keywords + science WCS/photometry (SCI wins on conflict)."""
    h = primary_hdr.copy()
    if sci_hdr is not None:
        for card in sci_hdr.cards:
            kw = card.keyword
            if kw in ("", "COMMENT") or kw is None:
                continue
            if str(kw).startswith("HISTORY"):
                continue
            if kw in _SKIP_SCI_KEYS:
                continue
            if kw in ("BITPIX", "NAXIS") or (isinstance(kw, str) and kw.startswith("NAXIS")):
                continue
            h[kw] = (card.value, card.comment)
    for kw in ("XTENSION", "PCOUNT", "GCOUNT", "EXTVER"):
        try:
            del h[kw]
        except KeyError:
            pass
    h["EXTNAME"] = ("SCI", "science data")
    return h


def _write_chip_file(
    out_path: Path,
    data: np.ndarray,
    primary_hdr: fits.Header,
    sci_hdr: fits.Header | None,
    *,
    log_: logging.Logger | None,
) -> None:
    hdr = _merge_headers(primary_hdr, sci_hdr)
    # Let Astropy set BITPIX / NAXIS* from the array
    hdu = fits.PrimaryHDU(data=np.asarray(data), header=hdr)
    hdu.writeto(out_path, overwrite=True, output_verify="warn")
    if log_ is not None:
        log_.info(
            "splitgroups (Python): wrote %s shape=%s",
            out_path.name,
            getattr(data, "shape", None),
        )


def apply_splitgroups(
    image: str | Path,
    *,
    log_: logging.Logger | None = None,
) -> list[str]:
    """
    Split a multi-extension HST science FITS (or WFPC2-style 3-D primary) into
    ``root.chipN.fits`` files.

    Parameters
    ----------
    image
        Path to input FITS.
    log_
        Optional logger (defaults to module logger).

    Returns
    -------
    list of str
        Paths of files written, sorted by chip number.

    Raises
    ------
    SplitgroupsError
        If there is nothing to split or data are invalid.
    """
    lg = log_ or log
    p = Path(image).resolve()
    if not p.is_file():
        raise SplitgroupsError(f"Not a file: {p}")

    out_paths: list[tuple[int, Path]] = []
    stem = p.with_suffix("")  # strip .fits

    with fits.open(p, mode="readonly", memmap=True) as hdul:
        primary_hdr = hdul[0].header
        sci_hdus = [(i, h) for i, h in enumerate(hdul) if _is_sci_hdu(h)]

        if sci_hdus:
            for j, (_i, sh) in enumerate(sci_hdus, start=1):
                if sh.data is None:
                    raise SplitgroupsError(f"SCI HDU {_i} has no data")
                outp = stem.parent / f"{stem.name}.chip{j}.fits"
                _write_chip_file(outp, sh.data, primary_hdr, sh.header, log_=lg)
                out_paths.append((j, outp))
        else:
            planes = _primary_3d_chip_planes(hdul[0])
            if not planes:
                raise SplitgroupsError(
                    f"No SCI extensions and no 4-plane 3-D primary in {p.name}"
                )
            for k, plane in enumerate(planes, start=1):
                outp = stem.parent / f"{stem.name}.chip{k}.fits"
                _write_chip_file(outp, plane, primary_hdr, None, log_=lg)
                out_paths.append((k, outp))

    # Deterministic order; if duplicate chip numbers, keep first-write order
    out_paths.sort(key=lambda t: t[0])
    written = [str(x[1]) for x in out_paths]
    lg.info(
        "splitgroups (Python): %d chip file(s) from %s",
        len(written),
        p.name,
    )
    return written
