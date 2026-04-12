"""WCS header helpers: meta headers, SIP/CTYPE consistency for updatewcs."""

from __future__ import annotations

import logging
import os

from astropy.io import fits

_LOG = logging.getLogger(__name__)


def _header_has_sip_polynomials(header) -> bool:
    """True if FITS header carries SIP distortion orders (HST-style A_ORDER / B_ORDER)."""
    try:
        ao = int(header.get("A_ORDER", 0) or 0)
        bo = int(header.get("B_ORDER", 0) or 0)
    except (TypeError, ValueError):
        return False
    return ao > 0 or bo > 0


def _tan_ctypes_need_sip_suffix(header) -> bool:
    if not _header_has_sip_polynomials(header):
        return False
    ct1 = str(header.get("CTYPE1", "") or "").strip().upper()
    ct2 = str(header.get("CTYPE2", "") or "").strip().upper()
    if not ct1.startswith("RA---TAN") or not ct2.startswith("DEC--TAN"):
        return False
    # Require both CTYPEs to carry -SIP when SIP polynomials are present
    return not (ct1.endswith("-SIP") and ct2.endswith("-SIP"))


def fix_sip_ctype_headers_fits(
    image_path: str | os.PathLike[str],
    *,
    logger: logging.Logger | None = None,
) -> int:
    """
    If SIP polynomials are present but CTYPE* lack the ``-SIP`` suffix, fix in place.

    MAST ACS/WFC3 FLT/FLC files often trigger multi-line astropy ``INFO`` spam and
    inconsistent-WCS behavior when CTYPE is ``RA---TAN`` while SIP keys exist.
    """
    log = logger or _LOG
    path = os.fspath(image_path)
    n_fix = 0
    with fits.open(path, mode="update") as hdul:
        for i, hdu in enumerate(hdul):
            try:
                if int(hdu.header.get("NAXIS", 0) or 0) < 2:
                    continue
            except (TypeError, ValueError):
                continue
            if not _tan_ctypes_need_sip_suffix(hdu.header):
                continue
            hdu.header["CTYPE1"] = "RA---TAN-SIP"
            hdu.header["CTYPE2"] = "DEC--TAN-SIP"
            n_fix += 1
            log.debug("SIP CTYPE fix %s HDU %s", os.path.basename(path), i)
        if n_fix:
            hdul.flush()
    return n_fix


def make_meta_wcs_header(header):
    """Return a minimal WCS header dict; CTYPE* SIP variants normalized to TAN."""
    meta_header = {}
    for key in [
        "NAXIS", "NAXIS1", "NAXIS2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
        "CTYPE1", "CTYPE2",
    ]:
        meta_header[key] = header[key]

    if meta_header["CTYPE1"] == "RA---TAN-SIP":
        meta_header["CTYPE1"] = "RA---TAN"
    if meta_header["CTYPE2"] == "DEC--TAN-SIP":
        meta_header["CTYPE2"] = "DEC--TAN"

    return meta_header


def wcs_from_fits_hdu(hdul: fits.HDUList, hdu_index: int = 0, *, relax: bool = True):
    """
    Build ``astropy.wcs.WCS`` from a FITS image HDU inside *hdul*.

    Calibrated HST images (and Python ``splitgroups`` products) often reference
    **lookup-table** distortions (``CPDIS*``, ``D2IMARR*``, ``DISTNAME``, …).
    Astropy must read those tables from other extensions, so constructing
    ``WCS(header)`` without the file context raises::

        ValueError: an astropy.io.fits.HDUList is required for Lookup table distortion.

    Passing ``fobj=hdul`` fixes that when all referenced extensions exist.

    **Split-chip / trimmed FITS:** The primary can still list ``D2IMARR`` /
    ``CPDIS`` tables that exist only in the original MEF. Then Astropy raises
    ``KeyError`` (e.g. ``Extension ('D2IMARR', 3.0) not found``). We fall back to
    a linear WCS from :func:`make_meta_wcs_header`, which is enough for bounds
    checks (e.g. pipeline ``split_image_contains`` on split-chip FITS).

    On ``MemoryError``, ``ValueError``, or ``KeyError``, falls back to that
    linear WCS.
    """
    from astropy.wcs import WCS

    hdu = hdul[hdu_index]
    try:
        return WCS(hdu.header, fobj=hdul, relax=relax)
    except MemoryError:
        return WCS(make_meta_wcs_header(hdu.header))
    except ValueError:
        return WCS(make_meta_wcs_header(hdu.header))
    except KeyError:
        # Missing distortion extension in this HDUList (common for splitgroups output)
        return WCS(make_meta_wcs_header(hdu.header))


def _delete_alt_wcs_key_silent(hdul: fits.HDUList, ext_index: int, wkey: str) -> bool:
    """Remove alternate WCS *wkey* from one extension (no prints; mirrors STScI ``deleteWCS``)."""
    from hst123.utils.stsci_wcs import altwcs_module

    altwcs = altwcs_module()
    if wkey == "O":
        return False
    hdr = hdul[ext_index].header
    if wkey not in altwcs.wcskeys(hdr):
        return False
    hwcs = altwcs.wcs_from_key(hdul, ext_index, from_key=wkey, exclude_special=False)
    if not hwcs:
        return False
    for k in hwcs:
        if k in hdr:
            del hdr[k]
    return True


def remove_conflicting_alt_wcs_duplicate_names(
    image_path: str | os.PathLike[str],
    *,
    logger: logging.Logger | None = None,
) -> int:
    """
    Drop **stale** alternate WCS solutions that reuse the primary ``WCSNAME`` but
    differ from the primary WCS.

    ``updatehdr.update_wcs`` (used by ``updatewcs`` / AstrometryDB) calls
    ``altwcs.archive_wcs(..., wcsname=<primary name>, mode=QUIET_ABORT)``. If an
    alternate WCS already exists under that same name with **different** FITS
    keywords, ``archive_wcs`` cannot archive the primary and logs::

        'wcsname' must be unique in image header...

    That is a real header inconsistency (e.g. old ``TWEAK`` alternate vs a new
    primary with the same name). Removing the non-matching alternates (except
    key ``O``, the pipeline OPUS archive) restores a state ``updatewcs`` can
    handle without warnings.
    """
    from hst123.utils.stsci_wcs import altwcs_module

    altwcs = altwcs_module()
    log = logger or _LOG
    path = os.fspath(image_path)
    n_removed = 0
    with fits.open(path, mode="update") as hdul:
        for ext_index, hdu in enumerate(hdul):
            if getattr(hdu, "name", None) != "SCI":
                continue
            hdr = hdu.header
            if "WCSNAME" not in hdr:
                continue
            pri = str(hdr["WCSNAME"]).strip()
            if not pri:
                continue
            pri_u = pri.upper()

            names = altwcs.wcsnames(hdul, ext=ext_index)
            stale_keys: list[str] = []
            for wkey, wname in names.items():
                if wkey in (" ", "O"):
                    continue
                if str(wname).strip().upper() != pri_u:
                    continue
                try:
                    same = altwcs._test_wcs_equal(hdul, ext_index, " ", wkey)
                except Exception:
                    # Unreadable/broken alt WCS: treat as stale so updatewcs can proceed
                    same = False
                if same:
                    continue
                stale_keys.append(wkey)

            for wkey in stale_keys:
                if _delete_alt_wcs_key_silent(hdul, ext_index, wkey):
                    n_removed += 1
                    log.debug(
                        "Removed alternate WCS key %r on %s SCI ext %s "
                        "(duplicate WCSNAME %r inconsistent with primary)",
                        wkey,
                        os.path.basename(path),
                        ext_index,
                        pri,
                    )
        if n_removed:
            hdul.flush()
    return n_removed
