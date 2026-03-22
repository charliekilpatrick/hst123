"""
AstroDrizzle ``output`` path handling.

drizzlepac uses different filename patterns when ``output`` has no ``.fits``
suffix (linear root), producing ``{root}_drz_sci.fits`` instead of
``{root}_sci.fits`` for a ``{root}.fits`` stem.
"""
from __future__ import annotations

import logging
import os


def logical_driz_to_internal_astrodrizzle(path: str) -> str:
    """
    Map obstable logical product (``.drc.fits``) to the internal AstroDrizzle stem
    (``.drz.fits``) used while drizzlepac runs; sidecars are ``*_sci.fits``, etc.
    """
    on = os.fspath(path)
    if on.lower().endswith(".drc.fits"):
        return on[:-9] + ".drz.fits"
    return on


def logical_drc_path_from_internal_drz(internal_drz: str) -> str:
    """``foo.drz.fits`` → ``foo.drc.fits`` (canonical multi-extension product)."""
    on = os.fspath(internal_drz)
    if not on.lower().endswith(".drz.fits"):
        raise ValueError(f"expected .drz.fits internal drizzle path, got {internal_drz!r}")
    return on[:-9] + ".drc.fits"


def normalize_astrodrizzle_output_path(output_name, logger: logging.Logger):
    """
    Ensure ``output`` ends with .fits so AstroDrizzle emits *_sci.fits / *_wht.fits.

    If the path has no extension, drizzlepac writes ``{root}_drz_sci.fits``,
    which breaks rename logic and causes :exc:`FileNotFoundError` on downstream
    steps. Truncated ``drizname`` values (legacy fixed-width table columns) often
    dropped the ``.drz.fits`` suffix.

    Paths ending in ``.drc.fits`` (logical pipeline product) are returned unchanged;
    callers should map them to ``.drz.fits`` with :func:`logical_driz_to_internal_astrodrizzle`
    before calling AstroDrizzle.
    """
    on = os.fspath(output_name)
    if on.lower().endswith(".drc.fits"):
        return output_name
    if on.lower().endswith((".fits", ".fit")):
        return output_name
    logger.warning(
        "Drizzle output path %r has no .fits suffix; appending .drz.fits so "
        "AstroDrizzle writes *_sci.fits / *_wht.fits names this pipeline expects. "
        "(Truncated drizname from older fixed-width table columns produced paths "
        "like .../acs.f555w.ut without .drz.fits.)",
        output_name,
    )
    return on + ".drz.fits"


def recover_drizzlepac_linear_output(output_name, logger: logging.Logger):
    """
    If drizzlepac wrote ``{root}_drz_sci.fits`` (root without .fits), rename to
    the ``{root}.drz_sci.fits`` pattern for a canonical ``{root}.drz.fits``.

    Returns the path to the final drizzle science file stem (``*.drz.fits``).
    """
    if os.path.isfile(output_name):
        return output_name
    op = os.fspath(output_name)
    if op.lower().endswith((".fits", ".fit")):
        return output_name
    sci = op + "_drz_sci.fits"
    if not os.path.isfile(sci):
        return output_name
    canonical = op + ".drz.fits"
    sci_dst = canonical.replace(".fits", "_sci.fits")
    wht_src, wht_dst = op + "_drz_wht.fits", canonical.replace(".fits", "_wht.fits")
    ctx_src, ctx_dst = op + "_drz_ctx.fits", canonical.replace(".fits", "_ctx.fits")
    try:
        os.rename(sci, sci_dst)
        if os.path.isfile(wht_src):
            os.rename(wht_src, wht_dst)
        if os.path.isfile(ctx_src):
            os.rename(ctx_src, ctx_dst)
    except OSError as exc:
        logger.error("Could not recover drizzlepac _drz_* outputs: %s", exc)
        return output_name
    logger.warning(
        "Recovered AstroDrizzle products from *_drz_*.fits names; "
        "canonical drizzle file is now %s",
        canonical,
    )
    return canonical


def astrodrizzle_output_exists(output_path) -> bool:
    """
    Return True if the drizzle product is on disk.

    With ``build=False``, drizzlepac typically writes ``{output}_sci.fits``,
    ``_wht.fits``, and ``_ctx.fits`` first; the pipeline then renames the
    science extension to the canonical ``output_path``. Before that rename,
    only the ``*_sci.fits`` file exists — treat that as success.

    For logical ``*.drc.fits`` paths, success is the multi-extension file itself.
    While a run is in progress, the internal ``*.drz.fits`` stem may still exist
    without ``.drc`` yet — treat that as success too.
    """
    op = os.path.expanduser(os.fspath(output_path))
    if os.path.isfile(op):
        return True
    if op.lower().endswith(".drc.fits"):
        internal = logical_driz_to_internal_astrodrizzle(op)
        if internal != op and os.path.isfile(internal):
            return True
        sci = internal[:-5] + "_sci.fits"
        return os.path.isfile(sci)
    if not op.lower().endswith(".fits"):
        return False
    sci = op[:-5] + "_sci.fits"
    return os.path.isfile(sci)
