"""Per-image DOLPHOT run summary from sidecar files and the main catalog."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hst123.primitives.photometry import estimate_limit_from_snr_bins
from hst123.utils.dolphot_catalog_hdf5 import (
    find_column_index_0based,
    group_param_by_image,
    load_dolphot_catalog_array,
    parse_dolphot_columns_file,
    parse_dolphot_data_file,
    parse_dolphot_info_file,
    parse_dolphot_param_file,
)

PathLike = str | Path

_SUMMARY_FIELDS = (
    "image_num",
    "image_name",
    "filter",
    "exptime",
    "align_stars_used",
    "align_sig",
    "psf_central_adj",
    "apcor_stars_used",
    "limit_mag_3sig",
)


def _sidecar_paths(base: Path) -> dict[str, Path]:
    stem = base.name if base.suffix == "" else base.stem
    parent = base.parent
    return {
        "param": parent / f"{stem}.param",
        "info": parent / f"{stem}.info",
        "data": parent / f"{stem}.data",
        "columns": parent / f"{stem}.columns",
        "catalog": base if base.is_file() else parent / stem,
    }


def _index_maps(data: dict[str, Any]) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    align = {
        int(row["image_index"]): row
        for row in data.get("align_images") or []
    }
    psf = {
        int(row["image_index"]): row
        for row in data.get("psf") or []
    }
    apcor = {
        int(row["image_index"]): row
        for row in data.get("apcor") or []
    }
    return align, psf, apcor


def _info_filter_exptime(info: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in info.get("image_filter_exptime") or []:
        idx = row.get("image_index")
        if idx is not None:
            out[int(idx)] = row
    return out


def _image_indices_from_param(param: dict[str, str]) -> list[int]:
    indices: list[int] = []
    for key in param:
        m = re.match(r"^img(\d+)_file$", key)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def _per_image_limit_mag(
    catalog: np.ndarray,
    columns,
    image_name: str,
    *,
    snr_limit: float,
    object_type_idx: int | None,
) -> float:
    """Median-binned 3-sigma limiting magnitude for one DOLPHOT input image."""
    mag_idx = find_column_index_0based(
        columns, "Instrumental VEGAMAG magnitude", image_name,
    )
    err_idx = find_column_index_0based(
        columns, "Magnitude uncertainty", image_name,
    )
    if mag_idx is None or err_idx is None:
        return float("nan")
    ncols = catalog.shape[1]
    if mag_idx >= ncols or err_idx >= ncols:
        return float("nan")

    mags = catalog[:, mag_idx]
    errs = catalog[:, err_idx]
    ok = np.isfinite(mags) & np.isfinite(errs) & (errs > 0.0) & (errs < 0.5)
    if object_type_idx is not None and 0 <= object_type_idx < catalog.shape[1]:
        obj_type = catalog[:, object_type_idx]
        ok &= np.isfinite(obj_type) & (obj_type == 1.0)
    if not np.any(ok):
        return float("nan")
    return float(
        estimate_limit_from_snr_bins(
            mags[ok], errs[ok], snr_target=snr_limit,
        )
    )


def build_dolphot_image_summary_rows(
    base: PathLike,
    *,
    snr_limit: float = 3.0,
) -> list[dict[str, Any]]:
    """
    Build one summary dict per DOLPHOT input image (``img0000`` .. ``imgNNNN``).

    Uses ``*.param``, ``*.info``, ``*.data``, ``*.columns``, and the main catalog.
    """
    base_path = Path(base)
    paths = _sidecar_paths(base_path)
    if not paths["param"].is_file():
        return []

    param = parse_dolphot_param_file(paths["param"])
    _, by_image = group_param_by_image(param)
    info = (
        parse_dolphot_info_file(paths["info"])
        if paths["info"].is_file()
        else {}
    )
    data = (
        parse_dolphot_data_file(paths["data"])
        if paths["data"].is_file()
        else {}
    )
    align_by_idx, psf_by_idx, apcor_by_idx = _index_maps(data)
    filter_by_idx = _info_filter_exptime(info)

    columns = []
    catalog = None
    object_type_idx = None
    if paths["columns"].is_file():
        columns = parse_dolphot_columns_file(paths["columns"])
        object_type_idx = find_column_index_0based(columns, "Object type", "")
    if paths["catalog"].is_file() and paths["catalog"].stat().st_size > 0:
        catalog = load_dolphot_catalog_array(paths["catalog"])

    rows: list[dict[str, Any]] = []
    for img_key_idx in _image_indices_from_param(param):
        img_key = f"img{img_key_idx:04d}"
        image_name = str(by_image.get(img_key, {}).get("file", "")).strip()
        if not image_name:
            continue

        meta = filter_by_idx.get(img_key_idx, {})
        filt = str(meta.get("filter", ""))
        exptime = meta.get("exptime")

        align = align_by_idx.get(img_key_idx, {})
        align_vals = align.get("values") or []
        align_sig = float(align_vals[-1]) if align_vals else float("nan")
        align_used = align.get("n2")

        psf = psf_by_idx.get(img_key_idx, {})
        psf_vals = psf.get("values") or []
        psf_adj = float(psf_vals[1]) if len(psf_vals) > 1 else float("nan")

        apcor = apcor_by_idx.get(img_key_idx, {})
        apcor_vals = apcor.get("values") or []
        apcor_used = int(apcor_vals[1]) if len(apcor_vals) > 1 else 0

        limit_mag = float("nan")
        if catalog is not None and columns:
            limit_mag = _per_image_limit_mag(
                catalog,
                columns,
                image_name,
                snr_limit=snr_limit,
                object_type_idx=object_type_idx,
            )

        rows.append(
            {
                "image_num": f"{img_key_idx:03d}",
                "image_name": image_name,
                "filter": filt,
                "exptime": exptime,
                "align_stars_used": align_used,
                "align_sig": align_sig,
                "psf_central_adj": psf_adj,
                "apcor_stars_used": apcor_used,
                "limit_mag_3sig": limit_mag,
            }
        )
    return rows


def format_dolphot_image_summary_line(row: Mapping[str, Any]) -> str:
    """Format one DOLPHOT per-image summary row for logging."""
    exptime = row.get("exptime")
    exptime_s = (
        f"{float(exptime):.1f}"
        if exptime is not None and np.isfinite(float(exptime))
        else "—"
    )
    align_used = row.get("align_stars_used")
    align_used_s = str(align_used) if align_used is not None else "—"
    align_sig = row.get("align_sig")
    align_sig_s = (
        f"{float(align_sig):.3f}"
        if align_sig is not None and np.isfinite(float(align_sig))
        else "—"
    )
    psf_adj = row.get("psf_central_adj")
    psf_adj_s = (
        f"{float(psf_adj):+.6f}"
        if psf_adj is not None and np.isfinite(float(psf_adj))
        else "—"
    )
    limit_mag = row.get("limit_mag_3sig")
    limit_mag_s = (
        f"{float(limit_mag):.3f}"
        if limit_mag is not None and np.isfinite(float(limit_mag))
        else "—"
    )
    filt = row.get("filter") or "—"
    return (
        f"img={row.get('image_num', '—')} "
        f"name={row.get('image_name', '—')} "
        f"filter={filt} "
        f"exptime={exptime_s}s "
        f"align_used={align_used_s} "
        f"align_sig={align_sig_s} "
        f"psf_adj={psf_adj_s} "
        f"apcor_used={row.get('apcor_stars_used', 0)} "
        f"limit_3sig={limit_mag_s}"
    )


def log_dolphot_image_summary(
    base: PathLike,
    log: logging.Logger,
    *,
    snr_limit: float = 3.0,
) -> list[dict[str, Any]]:
    """Log a line-by-line DOLPHOT per-image summary at INFO level."""
    rows = build_dolphot_image_summary_rows(base, snr_limit=snr_limit)
    if not rows:
        log.info(
            "DOLPHOT image summary: no per-image rows "
            "(missing or empty %s.param sidecar).",
            Path(base).name,
        )
        return rows
    log.info(
        "DOLPHOT image summary (%i image(s); 3-sigma limit from catalog type=1 sources):",
        len(rows),
    )
    for row in rows:
        log.info("  %s", format_dolphot_image_summary_line(row))
    return rows
