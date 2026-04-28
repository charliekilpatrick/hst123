"""Set ``MJD-AVG`` on HST FITS primary headers (JHAT and downstream tooling)."""
from __future__ import annotations

import logging
import os
import re

from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.time import Time

__all__ = [
    "mjd_avg_from_primary_header",
    "set_mjd_avg_primary",
    "ensure_mjd_avg_on_pipeline_science_image",
]

log = logging.getLogger(__name__)

_CHIP_RE = re.compile(r"^.+\.(chip\d+)\.fits$", re.IGNORECASE)


def mjd_avg_from_primary_header(header: fits.Header) -> float | None:
    """
    Representative MJD for an exposure: mid-time from ``EXPSTART``/``EXPEND`` when
    both exist, else ``EXPSTART``, else ``DATE-OBS`` + ``TIME-OBS``.
    """
    try:
        if "EXPSTART" in header and "EXPEND" in header:
            es = float(header["EXPSTART"])
            ee = float(header["EXPEND"])
            return 0.5 * (es + ee)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        if "EXPSTART" in header:
            return float(Time(header["EXPSTART"], format="mjd").mjd)
    except Exception:
        pass
    try:
        if "DATE-OBS" in header and "TIME-OBS" in header:
            ts = f"{str(header['DATE-OBS']).strip()}T{str(header['TIME-OBS']).strip()}"
            return float(Time(ts).mjd)
    except Exception:
        pass
    return None


def set_mjd_avg_primary(
    path: str,
    mjd: float,
    *,
    comment: str = "Representative MJD (mid-exposure, hst123)",
) -> None:
    """Write ``MJD-AVG`` on the PRIMARY header of *path* (in place)."""
    _write_mjd_avg_primary(path, mjd, comment=comment)


def _write_mjd_avg_primary(path: str, mjd: float, *, comment: str) -> None:
    """In-place PRIMARY update; use relaxed FITS verify (HST MEFs often fail strict)."""
    for verify in ("silentfix", "ignore"):
        try:
            with fits.open(path, mode="update", output_verify=verify) as hdul:
                hdul[0].header["MJD-AVG"] = (mjd, comment)
                hdul.flush(output_verify=verify)
            # Return only after context exit so close() uses the same output_verify.
            return
        except VerifyError:
            if verify == "ignore":
                raise
            log.debug(
                "MJD-AVG write: silentfix verify failed for %s, retrying with ignore",
                os.path.basename(path),
            )


def _is_pipeline_parent_science_fits(basename: str) -> bool:
    """True for ``*_flc.fits``, ``*_flt.fits``, ``*_c0m.fits`` (not chip splits)."""
    b = basename.lower()
    if _CHIP_RE.match(b):
        return False
    return b.endswith("_flc.fits") or b.endswith("_flt.fits") or b.endswith("_c0m.fits")


def ensure_mjd_avg_on_pipeline_science_image(path: str) -> bool:
    """
    Set ``MJD-AVG`` from the file's own headers if this is a pipeline science
    image (FLC/FLT/c0m parent, not ``*.chipN.fits``).

    Returns
    -------
    bool
        True if the file was recognized and updated (or already had a consistent
        value), False if skipped or MJD could not be derived.
    """
    if not os.path.isfile(path):
        return False
    if not _is_pipeline_parent_science_fits(os.path.basename(path)):
        return False
    try:
        with fits.open(path, mode="readonly") as hdul:
            mjd = mjd_avg_from_primary_header(hdul[0].header)
        if mjd is None:
            log.debug("Could not derive MJD-AVG for %s", os.path.basename(path))
            return False
        for verify in ("silentfix", "ignore"):
            try:
                with fits.open(path, mode="update", output_verify=verify) as hdul:
                    cur = hdul[0].header.get("MJD-AVG")
                    if cur is not None:
                        try:
                            cval = cur[0] if isinstance(cur, tuple) else cur
                            if float(cval) == float(mjd):
                                # No write; exit context before returning.
                                already = True
                            else:
                                already = False
                        except (TypeError, ValueError):
                            already = False
                    else:
                        already = False
                    if not already:
                        hdul[0].header["MJD-AVG"] = (
                            mjd,
                            "Representative MJD (mid-exposure, hst123)",
                        )
                    hdul.flush(output_verify=verify)
                return True
            except VerifyError:
                if verify == "ignore":
                    raise
                log.debug(
                    "MJD-AVG ensure: silentfix verify failed for %s, retrying with ignore",
                    os.path.basename(path),
                )
        return False
    except OSError as exc:
        log.warning("Could not set MJD-AVG on %s: %s", path, exc)
        return False
    except VerifyError as exc:
        log.warning("Could not set MJD-AVG on %s (FITS verify): %s", path, exc)
        return False
