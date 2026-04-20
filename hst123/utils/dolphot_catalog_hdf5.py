"""
Parse DOLPHOT multi-extension output (catalog, ``.columns``, ``.param``, ``.data``,
``.info``, ``.warnings``) and export a single HDF5 file with a labeled catalog plus metadata.

Designed against DOLPHOT 3.x text products: whitespace-separated numeric ``dpXXXX`` catalog,
one descriptive line per column in ``dpXXXX.columns``, and sidecar metadata files.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

PathLike = Union[str, Path]

# HDF5 column names: alphanumeric + underscore; must not start with digit in some tools
_MAX_HDF5_NAME_LEN = 63


@dataclass(frozen=True)
class DolphotColumn:
    """One row from a DOLPHOT ``*.columns`` file."""

    index_1based: int
    index_0based: int
    description: str
    raw_line: str


def parse_column_index_and_description(line: str) -> Optional[tuple[int, str]]:
    """
    Parse a single ``*.columns`` line of the form ``N. description``.

    Returns
    -------
    (1-based index, description) or None if the line does not match.
    """
    m = re.match(r"^\s*(\d+)\.\s*(.*)$", line.rstrip("\n"))
    if not m:
        return None
    n = int(m.group(1))
    desc = m.group(2).strip()
    return (n, desc)


def parse_dolphot_columns_file(path: PathLike) -> list[DolphotColumn]:
    """
    Read a DOLPHOT ``*.columns`` file into structured column definitions.

    Parameters
    ----------
    path : path-like
        Path to ``dpXXXX.columns``.

    Returns
    -------
    list of DolphotColumn
        One entry per line, in file order (column 1 .. N).
    """
    path = Path(path)
    out: list[DolphotColumn] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_column_index_and_description(line)
            if not parsed:
                continue
            n1, desc = parsed
            out.append(
                DolphotColumn(
                    index_1based=n1,
                    index_0based=n1 - 1,
                    description=desc,
                    raw_line=line.rstrip("\n"),
                )
            )
    return out


def _sanitize_hdf5_column_name(description: str, index_1based: int) -> str:
    """Build a unique, HDF5-safe ASCII name from a full column description."""
    # Drop path noise for readability but keep filter/instrument hints
    s = description.strip()
    s = re.sub(r"/[^\s]+", "", s)  # remove absolute paths
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = f"col_{index_1based}"
    if s[0].isdigit():
        s = "c_" + s
    if len(s) > _MAX_HDF5_NAME_LEN:
        s = s[: _MAX_HDF5_NAME_LEN]
    return s


def unique_hdf5_column_names(columns: list[DolphotColumn]) -> list[str]:
    """Return safe column names, de-duplicated with numeric suffixes if needed."""
    names: list[str] = []
    used: set[str] = set()
    for c in columns:
        base = _sanitize_hdf5_column_name(c.description, c.index_1based)
        cand = base
        n_dup = 0
        while cand in used:
            n_dup += 1
            suffix = f"_{n_dup}"
            cand = (base[: _MAX_HDF5_NAME_LEN - len(suffix)] + suffix).rstrip("_")
        used.add(cand)
        names.append(cand)
    return names


def find_column_index_0based(
    columns: list[DolphotColumn],
    key: str,
    image: str = "",
) -> Optional[int]:
    """
    Match DOLPHOT ``*.columns`` semantics used by
    :meth:`hst123.primitives.scrape_dolphot.ScrapeDolphotPrimitive.get_dolphot_column`.

    *key* must appear in the column description. If *image* is non-empty, it must
    appear in the description (after stripping a ``.fits`` suffix). If *image* is
    empty, any column whose description contains *key* may match—the first such
    column is returned (same as substring ``"" in line`` being true for every line).
    """
    img = image.replace(".fits", "")
    for col in columns:
        if key not in col.description:
            continue
        if not img:
            return col.index_0based
        if img in col.description:
            return col.index_0based
    return None


def load_dolphot_catalog_array(path: PathLike) -> np.ndarray:
    """
    Load the main DOLPHOT catalog (numeric rows, whitespace-separated).

    Parameters
    ----------
    path : path-like
        Path to ``dpXXXX`` (no extension), same as ``dolphot['base']`` in the pipeline.

    Returns
    -------
    numpy.ndarray
        2-D float array of shape (n_sources, n_columns).

    Notes
    -----
    Uses :func:`numpy.loadtxt`. Scraping uses a single load plus vectorized
    filtering (see :mod:`hst123.primitives.scrape_dolphot`) so the catalog is
    read once per scrape.
    """
    path = Path(path)
    return np.loadtxt(path, dtype=np.float64)


def dolphot_columns_to_astropy_table(
    catalog: np.ndarray,
    columns: list[DolphotColumn],
    names: Optional[list[str]] = None,
):
    """
    Build an :class:`astropy.table.Table` with descriptive column names.

    Parameters
    ----------
    catalog : ndarray
        Output of :func:`load_dolphot_catalog_array`.
    columns : list of DolphotColumn
        From :func:`parse_dolphot_columns_file`.
    names : list of str, optional
        HDF5-safe names; defaults to :func:`unique_hdf5_column_names`.
    """
    from astropy.table import Table

    if catalog.ndim != 2:
        raise ValueError("catalog must be a 2-D array")
    ncol = catalog.shape[1]
    if len(columns) != ncol:
        raise ValueError(
            f"column definition count ({len(columns)}) != catalog columns ({ncol})"
        )
    if names is None:
        names = unique_hdf5_column_names(columns)
    if len(names) != ncol:
        raise ValueError("names length must match catalog columns")
    return Table(data=[catalog[:, i] for i in range(ncol)], names=names)


def parse_dolphot_param_file(path: PathLike) -> dict[str, str]:
    """
    Parse ``dpXXXX.param`` (``key = value`` lines, DOLPHOT style).
    """
    path = Path(path)
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def group_param_by_image(param: Mapping[str, str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """
    Split flat ``imgNNNN_*`` keys into per-image dicts (``img0001``, ...).
    """
    global_params: dict[str, str] = {}
    by_image: dict[str, dict[str, str]] = defaultdict(dict)
    for k, v in param.items():
        m = re.match(r"^img(\d+)_(.+)$", k)
        if m:
            img_key = f"img{m.group(1)}"
            sub_key = m.group(2)
            by_image[img_key][sub_key] = v
        else:
            global_params[k] = v
    return global_params, dict(by_image)


def parse_dolphot_info_file(path: PathLike) -> dict[str, Any]:
    """
    Parse ``dpXXXX.info``: MJD per input image, limits, filter/exptime lines, alignment, apcor.

    Structure varies slightly by instrument; unknown lines are kept in ``extra_lines``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: dict[str, Any] = {"raw_text": text, "paths_mjd": []}

    i = 0
    if lines:
        msets = re.match(r"^(\d+)\s+sets\s+of\s+output", lines[0], re.I)
        if msets:
            out["n_output_sets"] = int(msets.group(1))
            i = 1

    # Alternating: absolute path, then indented MJD
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if ln.startswith("EXTENSION") or ln.startswith("Limits"):
            break
        if ln.startswith("/") or ln.startswith("~"):
            path_line = ln
            mjd: Optional[float] = None
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                try:
                    mjd = float(nxt)
                    i += 2
                except ValueError:
                    i += 1
            else:
                i += 1
            out["paths_mjd"].append({"path": path_line, "mjd": mjd})
            continue
        i += 1

    # Limits line after "Limits"
    for j in range(len(lines)):
        if lines[j].strip() == "Limits" and j + 1 < len(lines):
            parts = lines[j + 1].split()
            if len(parts) >= 4:
                try:
                    out["limits_xy"] = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]
                except ValueError:
                    out["limits_xy_raw"] = lines[j + 1]
            break

    image_meta: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^\*\s+image\s+(\d+):\s+(\S+)\s+(\S+)\s+(\S+)", ln)
        if m:
            image_meta.append(
                {
                    "image_index": int(m.group(1)),
                    "filter": m.group(2),
                    "chip": m.group(3),
                    "exptime": float(m.group(4)),
                }
            )
    if image_meta:
        out["image_filter_exptime"] = image_meta

    # Alignment block: after "Alignment" keyword, next non-empty lines until blank or keyword
    align_rows: list[list[float]] = []
    for j, ln in enumerate(lines):
        if ln.strip() == "Alignment":
            k = j + 1
            while k < len(lines):
                row = lines[k].strip()
                if not row or row.startswith("*") or row.startswith("Aperture"):
                    break
                nums = [float(x) for x in row.split()]
                if len(nums) >= 5:
                    align_rows.append(nums)
                k += 1
            break
    if align_rows:
        out["alignment"] = align_rows

    apcor: list[float] = []
    for j, ln in enumerate(lines):
        if ln.strip() == "Aperture corrections":
            k = j + 1
            while k < len(lines):
                row = lines[k].strip()
                if not row:
                    break
                try:
                    apcor.append(float(row.split()[0]))
                except ValueError:
                    pass
                k += 1
            break
    if apcor:
        out["aperture_corrections"] = apcor

    return out


def parse_dolphot_data_file(path: PathLike) -> dict[str, Any]:
    """
    Parse ``dpXXXX.data``: WCS, alignment iteration counts, PSF quality, aperture-correction stats.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: dict[str, Any] = {"raw_text": text}

    wcs_lines: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^WCS image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts]
            wcs_lines.append({"image_index": int(m.group(1)), "values": nums})
    if wcs_lines:
        out["wcs"] = wcs_lines

    align_header = None
    for ln in lines:
        m = re.match(r"^Align:\s+(\d+)", ln.strip())
        if m:
            align_header = int(m.group(1))
            break
    if align_header is not None:
        out["align_n_stars"] = align_header

    align_img: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^Align image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts[2:]] if len(parts) > 2 else []
            align_img.append(
                {
                    "image_index": int(m.group(1)),
                    "n1": int(parts[0]) if len(parts) > 0 else None,
                    "n2": int(parts[1]) if len(parts) > 1 else None,
                    "values": nums,
                }
            )
    if align_img:
        out["align_images"] = align_img

    psf_lines: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^PSF image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts]
            psf_lines.append({"image_index": int(m.group(1)), "values": nums})
    if psf_lines:
        out["psf"] = psf_lines

    apcor_lines: list[dict[str, Any]] = []
    for ln in lines:
        m = re.match(r"^Apcor image\s+(\d+):\s+(.*)$", ln.strip())
        if m:
            parts = m.group(2).split()
            nums = [float(x) for x in parts]
            apcor_lines.append({"image_index": int(m.group(1)), "values": nums})
    if apcor_lines:
        out["apcor"] = apcor_lines

    return out


def parse_dolphot_warnings_file(path: PathLike) -> dict[str, Any]:
    """
    Parse ``dpXXXX.warnings`` into all lines and optional buckets by image path.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    by_path: dict[str, list[str]] = defaultdict(list)
    global_lines: list[str] = []
    # Match full paths ending in .chipN or .fits
    path_re = re.compile(r"(/[^\s,]+(?:\.fits|\.chip\d+))")
    for ln in lines:
        found = path_re.findall(ln)
        if found:
            for pth in found:
                by_path[pth].append(ln)
        else:
            global_lines.append(ln)
    return {
        "lines": lines,
        "by_image_path": dict(by_path),
        "global_lines": global_lines,
    }


def merge_image_metadata(
    param: Mapping[str, str],
    info: Mapping[str, Any],
    data: Mapping[str, Any],
    warnings: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Combine parsed sidecars into one JSON-serializable tree keyed by ``img0000`` .. ``imgNNNN``.
    """
    _, by_img = group_param_by_image(param)
    merged: dict[str, Any] = {"images": {}}

    path_to_img: dict[str, str] = {}
    for ik, kv in by_img.items():
        fp = kv.get("file")
        if fp:
            path_to_img[str(Path(fp).resolve())] = ik
            path_to_img[fp] = ik

    pmjd = info.get("paths_mjd") or []
    for entry in pmjd:
        p = entry.get("path")
        if not p:
            continue
        key = None
        try:
            rp = str(Path(p).resolve())
            key = path_to_img.get(rp) or path_to_img.get(p)
        except OSError:
            key = path_to_img.get(p)
        if key and key not in merged["images"]:
            merged["images"][key] = {"param": by_img.get(key, {})}
        if key:
            merged["images"][key].setdefault("mjd", entry.get("mjd"))

    for ik in by_img:
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["param"] = by_img[ik]

    ife = info.get("image_filter_exptime") or []
    for row in ife:
        idx = row.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["info_filter_exptime"] = row

    wcs = data.get("wcs") or []
    for w in wcs:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_wcs"] = w

    for w in data.get("align_images") or []:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_align"] = w

    for w in data.get("psf") or []:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_psf"] = w

    for w in data.get("apcor") or []:
        idx = w.get("image_index")
        if idx is None:
            continue
        ik = f"img{idx:04d}"
        merged["images"].setdefault(ik, {})
        merged["images"][ik]["data_apcor"] = w

    warn_by = warnings.get("by_image_path") or {}
    for pth, wlines in warn_by.items():
        key = path_to_img.get(pth)
        if not key:
            try:
                key = path_to_img.get(str(Path(pth).resolve()))
            except OSError:
                key = None
        if key:
            merged["images"].setdefault(key, {})
            merged["images"][key]["warnings"] = wlines

    merged["global_param"], _ = group_param_by_image(param)
    merged["info_parsed"] = {k: v for k, v in info.items() if k != "raw_text"}
    merged["warnings_global_lines"] = warnings.get("global_lines", [])
    return merged


def _embed_dolphot_directory_metadata(
    hf: Any,
    dolphot_dir: Path,
    *,
    embed_text: bool = False,
    max_text_bytes: int = 8 * 1024 * 1024,
) -> None:
    """
    Under ``metadata/dolphot_directory/``, store a JSON manifest of every file
    under *dolphot_dir* (path, size, mtime). Optionally embed UTF-8 text for
    non-FITS files up to *max_text_bytes* (skips binary-looking content).
    """
    import h5py

    root = dolphot_dir.resolve()
    manifest: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        manifest.append(
            {"relpath": rel, "size": st.st_size, "mtime": st.st_mtime}
        )

    meta = hf.require_group("metadata")
    gpath = "metadata/dolphot_directory"
    if gpath in hf:
        del hf[gpath]
    dg = meta.create_group("dolphot_directory")
    dt = h5py.string_dtype(encoding="utf-8")
    mj = dg.create_dataset("manifest_json", (1,), dtype=dt)
    mj[0] = json.dumps(manifest, indent=2)
    dg.attrs["root"] = str(root)
    dg.attrs["n_files"] = len(manifest)
    dg.attrs["text_embed"] = bool(embed_text)

    if not embed_text:
        return

    tg = dg.create_group("text_embed")
    i = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if ".fits" in p.name.lower():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size == 0 or st.st_size > max_text_bytes:
            continue
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            txt = raw.decode("utf-8", errors="replace")
        ds = tg.create_dataset(f"file_{i:06d}", (1,), dtype=dt)
        ds[0] = txt
        ds.attrs["relpath"] = rel
        ds.attrs["size_bytes"] = st.st_size
        i += 1


def write_dolphot_catalog_hdf5(
    out_path: PathLike,
    base: PathLike,
    *,
    photometry_path: str = "photometry",
    include_raw_sidecars: bool = True,
    compression: bool = True,
    dolphot_dir: Optional[PathLike] = None,
    embed_dolphot_directory_text: bool = False,
    catalog_array: Optional[np.ndarray] = None,
    include_directory_manifest: bool = True,
    serialize_meta: bool = True,
) -> Path:
    """
    Write one HDF5 file with the full DOLPHOT catalog and metadata.

    Parameters
    ----------
    out_path : path-like
        Output ``.h5`` file.
    base : path-like
        DOLPHOT output base path **without extension** (same as pipeline ``dolphot['base']``).

    photometry_path : str
        HDF5 path for the :class:`~astropy.table.Table` (Astropy default metadata).

    include_raw_sidecars : bool
        If True, store UTF-8 text of ``.param``, ``.info``, ``.data``, ``.warnings``,
        and ``.columns`` as datasets under ``metadata/raw/``.

    compression : bool
        If True, use gzip compression on the table (requires Astropy + h5py).

    dolphot_dir : path-like, optional
        Directory scanned for a full file manifest (default: ``base.parent``, i.e.
        the DOLPHOT output folder such as ``<work>/dolphot/``). Every file path,
        size, and mtime are stored. Optional text embedding under
        ``metadata/dolphot_directory/text_embed/`` when *embed_dolphot_directory_text*
        is True.
    embed_dolphot_directory_text : bool, optional
        If True, embed small UTF-8 text files from *dolphot_dir* (can be slow for
        large trees). Default False (manifest only).
    catalog_array : ndarray, optional
        If provided, skips re-reading the DOLPHOT ``base`` catalog from disk
        (avoids a second full ``numpy.loadtxt`` when the pipeline already holds
        the array in memory after scraping).
    include_directory_manifest : bool, optional
        If True, walk *dolphot_dir* and store a JSON file manifest (can be slow
        for trees with very many files). Set False for faster HDF5 when the
        manifest is not needed.
    serialize_meta : bool, optional
        If False, skip rich HDF5 table metadata (faster writes, smaller files).

    Returns
    -------
    pathlib.Path
        Path to the written file.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(
            "write_dolphot_catalog_hdf5 requires h5py; pip install h5py"
        ) from exc

    base = Path(base)
    out_path = Path(out_path)
    if dolphot_dir is None:
        dolphot_dir = base.parent
    else:
        dolphot_dir = Path(dolphot_dir)

    col_path = Path(str(base) + ".columns")
    if not col_path.is_file():
        raise FileNotFoundError(f"missing columns file: {col_path}")

    columns = parse_dolphot_columns_file(col_path)
    names = unique_hdf5_column_names(columns)
    if catalog_array is not None:
        catalog = np.asarray(catalog_array, dtype=np.float64)
        if catalog.ndim != 2:
            raise ValueError("catalog_array must be 2-D")
    else:
        catalog = load_dolphot_catalog_array(base)
    table = dolphot_columns_to_astropy_table(catalog, columns, names=names)

    # Column descriptions as table meta (serialized by Astropy HDF5)
    for i, col in enumerate(columns):
        table.meta[f"dolphot_desc_{names[i]}"] = col.description
        table.meta[f"dolphot_index_{names[i]}"] = col.index_1based

    write_kw: dict[str, Any] = {
        "format": "hdf5",
        "path": photometry_path,
        "overwrite": True,
        "serialize_meta": serialize_meta,
    }
    if compression:
        write_kw["compression"] = "gzip"
    table.write(str(out_path), **write_kw)

    param_p = Path(str(base) + ".param")
    info_p = Path(str(base) + ".info")
    data_p = Path(str(base) + ".data")
    warn_p = Path(str(base) + ".warnings")

    param_d: dict[str, str] = parse_dolphot_param_file(param_p) if param_p.is_file() else {}
    info_d = parse_dolphot_info_file(info_p) if info_p.is_file() else {}
    data_d = parse_dolphot_data_file(data_p) if data_p.is_file() else {}
    warn_d = parse_dolphot_warnings_file(warn_p) if warn_p.is_file() else {}

    merged = merge_image_metadata(param_d, info_d, data_d, warn_d)

    def _write_str_dataset(g: h5py.Group, key: str, text: str) -> None:
        dt = h5py.string_dtype(encoding="utf-8")
        ds = g.create_dataset(key, (1,), dtype=dt)
        ds[0] = text

    with h5py.File(out_path, "a") as hf:
        root = hf[photometry_path]
        root.attrs["hst123_dolphot_hdf5_format"] = 1
        root.attrs["dolphot_base"] = str(base)
        root.attrs["dolphot_column_json"] = json.dumps(
            [
                {
                    "index_1based": c.index_1based,
                    "description": c.description,
                    "hdf5_name": names[i],
                }
                for i, c in enumerate(columns)
            ],
            indent=2,
        )
        root.attrs["dolphot_merged_metadata_json"] = json.dumps(merged, indent=2)

        meta = hf.require_group("metadata")
        meta.attrs["global_param_json"] = json.dumps(merged.get("global_param", {}), indent=2)
        meta.attrs["merged_images_json"] = json.dumps(merged.get("images", {}), indent=2)

        if include_raw_sidecars:
            raw = meta.require_group("raw")
            if col_path.is_file():
                _write_str_dataset(raw, "columns", col_path.read_text(encoding="utf-8", errors="replace"))
            if param_p.is_file():
                _write_str_dataset(raw, "param", param_p.read_text(encoding="utf-8", errors="replace"))
            if info_p.is_file():
                _write_str_dataset(raw, "info", info_p.read_text(encoding="utf-8", errors="replace"))
            if data_p.is_file():
                _write_str_dataset(raw, "data", data_p.read_text(encoding="utf-8", errors="replace"))
            if warn_p.is_file():
                _write_str_dataset(raw, "warnings", warn_p.read_text(encoding="utf-8", errors="replace"))

        if include_directory_manifest and dolphot_dir.is_dir():
            _embed_dolphot_directory_metadata(
                hf,
                dolphot_dir,
                embed_text=embed_dolphot_directory_text,
            )

    return out_path


def append_scraped_final_phot_hdf5(
    out_path: PathLike,
    final_phot: list,
    *,
    group_path: str = "scraped_photometry",
    compression: bool = False,
    serialize_meta: bool = False,
) -> None:
    """
    Append all scraped-source photometry tables as one stacked dataset in an existing HDF5 file.

    Adds column ``scrape_source_index`` (0-based) so rows from different sources
    can be split after readback. Uses a single HDF5 write (no per-source files).

    Parameters
    ----------
    out_path : path-like
        Existing file produced by :func:`write_dolphot_catalog_hdf5`.
    final_phot : list of astropy.table.Table
        One table per source from the scrape pipeline (e.g. output of ``scrapedolphot``).
    group_path : str, optional
        HDF5 path for the stacked table. Default ``scraped_photometry``.
    compression : bool, optional
        If True, gzip the appended table.
    serialize_meta : bool, optional
        If False, smaller/faster writes (recommended for large ``--scrape-all`` runs).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(
            "append_scraped_final_phot_hdf5 requires h5py; pip install h5py"
        ) from exc

    from astropy.table import vstack

    if not final_phot:
        return
    out_path = Path(out_path)
    if not out_path.is_file():
        raise FileNotFoundError(f"expected existing HDF5: {out_path}")

    pieces = []
    for i, t in enumerate(final_phot):
        if t is None or len(t) == 0:
            continue
        if "scrape_source_index" in t.colnames:
            raise ValueError("table already has column scrape_source_index")
        ti = t.copy()
        ti["scrape_source_index"] = np.full(len(ti), i, dtype=np.int32)
        pieces.append(ti)
    if not pieces:
        return

    stacked = vstack(pieces, metadata_conflicts="silent")
    write_kw: dict[str, Any] = {
        "format": "hdf5",
        "path": group_path,
        "append": True,
        "overwrite": False,
        "serialize_meta": serialize_meta,
    }
    if compression:
        write_kw["compression"] = "gzip"
    stacked.write(str(out_path), **write_kw)

    with h5py.File(out_path, "a") as hf:
        if group_path in hf:
            grp = hf[group_path]
            grp.attrs["hst123_scraped_photometry_format"] = 1
            grp.attrs["n_scrape_sources"] = int(len(final_phot))


def read_dolphot_catalog_hdf5(path: PathLike, photometry_path: str = "photometry"):
    """
    Read back a table written by :func:`write_dolphot_catalog_hdf5`.

    Returns
    -------
    astropy.table.Table
    """
    from astropy.table import Table

    return Table.read(str(path), format="hdf5", path=photometry_path)
