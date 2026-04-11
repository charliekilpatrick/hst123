"""
Helpers for :meth:`hst123.hst123.run_astrodrizzle` — combine rules, sidecar renames, WCS tweaks.

Keeps drizzle orchestration readable and avoids duplicated path logic.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import copy


def combine_type_and_nhigh(n_frames: int, combine_type_override: str | None) -> tuple[str, int]:
    """
    drizzlepac ``combine_type`` / ``combine_nhigh`` from frame count and CLI override.

    Parameters
    ----------
    n_frames
        Number of input exposures in the drizzle stack.
    combine_type_override
        If set (non-empty), returned as ``combine_type``; ``combine_nhigh`` still computed.

    Returns
    -------
    tuple
        ``(combine_type, combine_nhigh)``.
    """
    if n_frames < 7:
        combine_type = "minmed"
        combine_nhigh = 0
    else:
        combine_type = "median"
        combine_nhigh = max(int((n_frames - 4) / 2), 0)
    if combine_type_override:
        combine_type = combine_type_override
    return combine_type, combine_nhigh


def resolve_drizzle_clean_flag(clean: bool | None, args_cleanup: bool) -> bool:
    """
    ``clean`` argument to AstroDrizzle: explicit *clean* wins, else CLI ``--cleanup``.
    """
    return args_cleanup if clean is None else clean


def drizzle_sidecar_paths(output_fits: str) -> tuple[str, str, str]:
    """Return (science, weight, context) sidecar paths for a canonical drizzle root."""
    root = output_fits
    if not root.lower().endswith(".fits"):
        root = root + ".fits"
    sci = root[:-5] + "_sci.fits"
    wht = root[:-5] + "_wht.fits"
    ctx = root[:-5] + "_ctx.fits"
    return sci, wht, ctx


def drizzle_canonical_weight_mask_paths(output_fits: str) -> tuple[str, str]:
    """Paths after renaming ``*_wht.fits`` / ``*_ctx.fits`` to pipeline names."""
    root = output_fits
    if not root.lower().endswith(".fits"):
        root = root + ".fits"
    return root[:-5] + ".weight.fits", root[:-5] + ".mask.fits"


def wcs_image_hdu_index(hdul) -> int:
    """
    Index of the HDU that carries the 2-D image and WCS for a drizzled product.

    Single-extension ``*.drz.fits`` uses PRIMARY; MEF ``*.drc.fits`` often has an
    empty PRIMARY and science in ``SCI`` (extension 1).
    """
    prim = hdul[0]
    na = int(prim.header.get("NAXIS", 0) or 0)
    if na >= 2 and getattr(prim, "data", None) is not None:
        return 0
    if len(hdul) > 1 and str(hdul[1].name).strip().upper() == "SCI":
        return 1
    return 0


def remove_internal_linear_drizzle_products(
    internal_drz_fits: str,
    logger: logging.Logger,
) -> None:
    """Delete internal ``*.drz.fits`` plus ``.weight.fits`` / ``.mask.fits`` sidecars."""
    root = os.fspath(internal_drz_fits)
    paths = [root, *drizzle_canonical_weight_mask_paths(root)]
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            os.remove(p)
            logger.debug("Removed internal drizzle linear product: %s", p)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", p, exc)


def drizzle_product_catalog_header(hdul):
    """Header containing NINPUT/INPUT for a drizzled reference (PRIMARY or SCI)."""
    h0 = hdul[0].header
    if "NINPUT" in h0 and "INPUT" in h0:
        return h0
    if len(hdul) > 1 and str(hdul[1].name).strip().upper() == "SCI":
        h1 = hdul[1].header
        if "NINPUT" in h1 and "INPUT" in h1:
            return h1
    return h0


def rename_astrodrizzle_sidecars(
    output_name: str,
    logger: logging.Logger,
) -> tuple[str | None, str | None]:
    """
    Rename drizzlepac ``_sci`` / ``_wht`` / ``_ctx`` products to canonical layout.

    Returns
    -------
    tuple
        ``(weight_path, mask_path)`` after rename if those files exist, else ``(None, None)``.
    """
    sci, wht, ctx = drizzle_sidecar_paths(output_name)
    weight_dest, mask_dest = drizzle_canonical_weight_mask_paths(output_name)

    weight_final: str | None = None
    mask_final: str | None = None

    if os.path.isfile(wht):
        os.rename(wht, weight_dest)
        weight_final = weight_dest
        logger.debug("Renamed drizzle weight: %s -> %s", wht, weight_dest)
    if os.path.isfile(ctx):
        os.rename(ctx, mask_dest)
        mask_final = mask_dest
        logger.debug("Renamed drizzle context: %s -> %s", ctx, mask_dest)
    if os.path.isfile(sci):
        os.rename(sci, output_name)
        logger.debug("Renamed drizzle science: %s -> %s", sci, output_name)

    return weight_final, mask_final


def header_has_tweak_wcsname(header) -> bool:
    """True if any header key containing WCSNAME has string value ``TWEAK``."""
    for key in header.keys():
        if "WCSNAME" not in key:
            continue
        try:
            val = str(header[key]).strip()
        except Exception:
            continue
        if val == "TWEAK":
            return True
    return False


def ensure_wcsname_tweak_on_image(image_path: str, logger: logging.Logger) -> None:
    """
    Set ``WCSNAME=TWEAK`` on **SCI** extensions only (in place).

    Setting ``WCSNAME`` on every HDU (PRIMARY, ERR, DQ, …) duplicates the same
    name across alternates and triggers ``altwcs`` warnings during
    ``updatewcs`` / headerlet steps.
    """
    from astropy.io import fits

    with fits.open(image_path, mode="update") as imhdu:
        modified = False
        for i, h in enumerate(imhdu):
            if h.name != "SCI":
                continue
            if header_has_tweak_wcsname(h.header):
                continue
            logger.debug("WCSNAME -> TWEAK %s ext %s", image_path, i)
            imhdu[i].header["WCSNAME"] = "TWEAK"
            modified = True
        if modified:
            imhdu.flush()


def wfpc2_astrodrizzle_scratch_paths(c0m_path: str, pid: int) -> tuple[str, str | None]:
    """
    Scratch paths for WFPC2 ``*_c0m.fits`` (and paired ``*_c1m.fits``) passed to AstroDrizzle.

    DrizzlePac :meth:`drizzlepac.wfpc2Data.WFPC2InputImage.find_DQ_extension` infers the
    DQ filename from the four characters immediately before ``.fits``. Those must be
    ``_c0m`` so that string replacement yields the matching ``*_c1m.fits``. Names such
    as ``*_c0m.drztmp.fits`` break this logic and produce invalid paths like
    ``*_c0m.dr_c1p.fits`` (see drizzlepac ``wfpc2Data.py``).

    Parameters
    ----------
    c0m_path
        Absolute path to the science ``*_c0m.fits`` file.
    pid
        Process id (or other disambiguator) so concurrent runs do not collide.

    Returns
    -------
    tuple
        ``(temp_c0m, temp_c1m)`` where ``temp_c1m`` is ``None`` if no ``*_c1m.fits`` exists
        beside the original exposure root.
    """
    d = os.path.dirname(os.path.abspath(os.path.expanduser(c0m_path)))
    base = os.path.basename(c0m_path)
    if len(base) < 9 or not base.lower().endswith("_c0m.fits"):
        raise ValueError(f"expected WFPC2 *_c0m.fits path, got {c0m_path!r}")
    root = base[:-9]
    tmp_c0m = os.path.join(d, f"{root}_hst123drz{pid}_c0m.fits")
    c1m_orig = os.path.join(d, f"{root}_c1m.fits")
    tmp_c1m = os.path.join(d, f"{root}_hst123drz{pid}_c1m.fits")
    if os.path.isfile(c1m_orig):
        return tmp_c0m, tmp_c1m
    return tmp_c0m, None


def build_astrodrizzle_keyword_args(
    *,
    output_name: str,
    logfile_name: str,
    wcskey: str,
    options: dict[str, Any],
    dd: dict[str, Any],
    ra: float | None,
    dec: float | None,
    rotation: float | None,
    combine_type: str,
    combine_nhigh: int,
    skysub: bool,
    skymask_cat: str | None,
    wht_type: str,
    pixscale: float,
    clean: bool,
) -> dict[str, Any]:
    """Keyword arguments for :func:`drizzlepac.astrodrizzle.AstroDrizzle` (second call style)."""
    return dict(
        output=output_name,
        runfile=logfile_name,
        wcskey=wcskey,
        context=True,
        group="",
        build=False,
        num_cores=dd["num_cores"],
        preserve=False,
        clean=clean,
        skysub=skysub,
        skymethod="globalmin+match",
        skymask_cat=skymask_cat,
        skystat="mode",
        skylower=0.0,
        skyupper=None,
        updatewcs=False,
        driz_sep_fillval=None,
        driz_sep_bits=options["driz_bits"],
        driz_sep_wcs=True,
        driz_sep_rot=rotation,
        driz_sep_scale=options["driz_sep_scale"],
        driz_sep_outnx=options["nx"],
        driz_sep_outny=options["ny"],
        driz_sep_ra=ra,
        driz_sep_dec=dec,
        driz_sep_pixfrac=dd["driz_sep_pixfrac"],
        combine_maskpt=dd["combine_maskpt"],
        combine_type=combine_type,
        combine_nlow=0,
        combine_nhigh=combine_nhigh,
        combine_lthresh=-10000,
        combine_hthresh=None,
        combine_nsigma=dd["combine_nsigma"],
        driz_cr_corr=True,
        driz_cr=True,
        driz_cr_snr=dd["driz_cr_snr"],
        driz_cr_grow=dd["driz_cr_grow"],
        driz_cr_ctegrow=dd["driz_cr_ctegrow"],
        driz_cr_scale=dd["driz_cr_scale"],
        final_pixfrac=dd["final_pixfrac"],
        final_fillval=None,
        final_bits=options["driz_bits"],
        final_units="counts",
        final_wcs=True,
        final_refimage=None,
        final_wht_type=wht_type,
        final_rot=rotation,
        final_scale=pixscale,
        final_outnx=options["nx"],
        final_outny=options["ny"],
        final_ra=ra,
        final_dec=dec,
    )


def build_wfpc2_skymask_catalog(
    tmp_input: list[str],
    outdir: str,
    logger: logging.Logger,
) -> str | None:
    """
    Build ``skymask_cat`` for WFPC2 c0m inputs; return absolute path to the
    catalog file, or None if not applicable.
    """
    from astropy.io import fits

    skymask_lines: list[str] = []
    for file in tmp_input:
        if "c0m" not in file:
            continue
        maskfile = file.split("_")[0] + "_c1m.fits"
        if not os.path.exists(maskfile):
            continue
        with fits.open(maskfile) as mhdu:
            for i, h in enumerate(mhdu):
                if h.name == "SCI" and mhdu[i].data is not None:
                    d = mhdu[i].data
                    d[d == 258] = 256
            mhdu.writeto(maskfile, overwrite=True, output_verify="silentfix")

        with fits.open(file) as imhdu:
            exts = [str(i) for i, h in enumerate(imhdu) if h.name == "SCI"]
            if not exts:
                continue
            line = file + "{" + ",".join(exts) + "},"
            line += ",".join([maskfile + "[" + ext + "]" for ext in exts])
            skymask_lines.append(line + " \n")

    if not skymask_lines:
        return None
    skymask_path = os.path.abspath(os.path.join(outdir, "skymask_cat"))
    with open(skymask_path, "w", encoding="utf-8") as fh:
        fh.writelines(skymask_lines)
    logger.debug("Wrote skymask_cat with %d line(s) for WFPC2 c0m inputs", len(skymask_lines))
    # Absolute path: AstroDrizzle is invoked before run_alignment's chdir(work_dir),
    # so a basename-only skymask_cat would be resolved from the process CWD and missed.
    return skymask_path


def write_drc_multis_extension_if_requested(
    output_name: str,
    weight_path: str | None,
    mask_path: str | None,
    save_fullfile: bool,
    logger: logging.Logger,
    *,
    format_hdu_list_summary,
    logical_drc_path: str | None = None,
) -> str | None:
    """
    When weight and context/mask sidecars exist, build multi-extension ``.drc.fits``.

    The pipeline always prefers ``.drc.fits`` as the on-disk drizzle product when
    sidecars are present; *save_fullfile* is kept for API compatibility only.

    Parameters
    ----------
    output_name
        Internal linear ``*.drz.fits`` path (science data) after sidecar renames.
    logical_drc_path
        Optional explicit ``*.drc.fits`` path (obstable logical name). Default:
        ``output_name`` with ``.drz.fits`` → ``.drc.fits``.

    Returns
    -------
    str or None
        Path to written ``.drc.fits``, or None if skipped.
    """
    from astropy.io import fits

    w_ok = bool(weight_path and os.path.isfile(weight_path))
    m_ok = bool(mask_path and os.path.isfile(mask_path))
    if not (w_ok and m_ok):
        logger.debug(
            "Skipping .drc.fits (weight_ok=%s mask_ok=%s save_fullfile=%s)",
            w_ok,
            m_ok,
            save_fullfile,
        )
        return None
    if not str(output_name).lower().endswith(".drz.fits"):
        logger.warning(
            "Skipping multi-extension .drc.fits build: output %r does not end "
            "with .drz.fits (internal drizzle stem).",
            output_name,
        )
        return None

    drc_path = logical_drc_path or output_name.replace(".drz.fits", ".drc.fits")

    hdu = fits.open(output_name)
    try:
        logger.info("Building MEF .drc.fits | %s", format_hdu_list_summary(hdu))
        newhdu = fits.HDUList()
        newhdu.append(copy.copy(hdu[0]))
        newhdu[0].data = None
        newhdu[0].header["EXTNAME"] = "PRIMARY"
        newhdu.append(copy.copy(hdu[0]))
        newhdu[1].header["EXTNAME"] = "SCI"

        with fits.open(weight_path) as wh:
            newhdu.append(copy.copy(wh[0]))
        newhdu[2].header["EXTNAME"] = "WHT"
        newhdu[2].header["BUNIT"] = "UNITLESS"

        with fits.open(mask_path) as mh:
            newhdu.append(copy.copy(mh[0]))
        newhdu[3].header["EXTNAME"] = "CTX"
        newhdu[3].header["BUNIT"] = "UNITLESS"

        if "HDRTAB" in [h.name for h in hdu]:
            newhdu.append(copy.copy(hdu["HDRTAB"]))

        newhdu.writeto(drc_path, overwrite=True, output_verify="silentfix")
        logger.info("Wrote MEF %s", drc_path)
        return drc_path
    finally:
        hdu.close()
