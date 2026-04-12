"""FITS header provenance for hst123 alignment (TweakReg / JHAT): skip redundant re-runs."""

from __future__ import annotations

import hashlib
import os

# HIERARCH keywords (Astropy / FITS); short values for portability
H_ALIGNOK = "HIERARCH HST123 ALIGNOK"
H_ALIGNMT = "HIERARCH HST123 ALIGNMT"
H_ALIGNRF = "HIERARCH HST123 ALIGNRF"

# Maximum single-string value without relying on CONTINUE cards
_MAX_REF_LEN = 64


def normalize_alignment_ref_id(ref: str | None) -> str:
    """
    Stable identifier for the alignment reference (path, catalog name, or sentinel).

    Long paths are hashed so they fit in a single FITS string value.
    """
    if ref is None:
        return ""
    r = str(ref).strip()
    if not r:
        return ""
    if r.upper() == "DUMMY.FITS" or r == "DUMMY":
        return "DUMMY"
    # Non-path reference labels (JHAT / catalogs), not os.path.realpath
    if r.upper() == "GAIA":
        return "GAIA"
    expanded = os.path.normcase(os.path.abspath(os.path.realpath(r)))
    if len(expanded) <= _MAX_REF_LEN:
        return expanded
    digest = hashlib.sha256(expanded.encode("utf-8", errors="replace")).hexdigest()[:40]
    return f"sha256:{digest}"


def alignment_method_token(align_with: str) -> str:
    """Normalized method string stored in FITS (lowercase)."""
    return str(align_with or "tweakreg").strip().lower()


def read_alignment_provenance(primary_header) -> dict | None:
    """
    Read hst123 alignment provenance from the primary header.

    Returns
    -------
    dict or None
        Keys: ``ok`` (bool), ``method`` (str), ``ref`` (str).
        None if no provenance block is present.
    """
    if H_ALIGNOK not in primary_header:
        return None
    try:
        raw_ok = primary_header[H_ALIGNOK]
        try:
            ok = bool(raw_ok)
        except Exception:
            ok = str(raw_ok).upper().strip() in ("T", "TRUE", "1", "YES")
        method = str(primary_header.get(H_ALIGNMT, "")).strip().lower()
        ref = str(primary_header.get(H_ALIGNRF, "")).strip()
        if not method:
            return None
        return {"ok": ok, "method": method, "ref": ref}
    except Exception:
        return None


def alignment_is_redundant(
    primary_header,
    *,
    method: str,
    ref_id: str | None,
    require_success: bool = True,
) -> bool:
    """
    True if header records a successful alignment with the same method and reference.

    If *ref_id* is None or empty, reference is not compared (cannot prove redundancy).
    """
    prov = read_alignment_provenance(primary_header)
    if prov is None:
        return False
    if require_success and not prov["ok"]:
        return False
    if prov["method"] != alignment_method_token(method):
        return False
    if ref_id is None or str(ref_id).strip() == "":
        return False
    want = normalize_alignment_ref_id(ref_id)
    return prov["ref"] == want


def write_alignment_provenance(
    primary_header,
    *,
    method: str,
    ref_id: str | None,
    success: bool,
) -> None:
    """Write or update HST123 alignment keywords on the primary header."""
    # Short comments only: HIERARCH lines must stay within FITS card rules.
    primary_header[H_ALIGNOK] = success
    primary_header[H_ALIGNMT] = alignment_method_token(method)[:16]
    rid = normalize_alignment_ref_id(ref_id) if ref_id is not None else ""
    primary_header[H_ALIGNRF] = rid


def clear_alignment_provenance(primary_header) -> None:
    """Remove hst123 alignment keywords (e.g. before forced re-align)."""
    for k in (H_ALIGNOK, H_ALIGNMT, H_ALIGNRF):
        if k in primary_header:
            del primary_header[k]


def alignment_done_on_primary_header(primary_header) -> bool:
    """
    True if this HDU header indicates TweakReg (or hierarchical) alignment completed.

    Uses only ``TWEAKSUC`` and ``HIERARCH`` — not generic ALIGN* provenance alone.
    Skipping ``updatewcs`` based on ALIGN without a reference match would disagree
    with :func:`alignment_is_redundant` and could suppress ``updatewcs`` while
    :meth:`check_images_for_tweakreg` still schedules TweakReg.
    """
    try:
        ts = float(primary_header.get("TWEAKSUC", 0))
    except (TypeError, ValueError):
        ts = 0.0
    if ts == 1.0:
        return True
    if "HIERARCH" in primary_header and primary_header.get("HIERARCH") == 1:
        return True
    return False
