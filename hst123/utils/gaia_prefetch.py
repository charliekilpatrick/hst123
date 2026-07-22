"""
Prefetch Gaia DR3 catalogs for JHAT and related aligners.

`run_jhat(..., gaia=True)` pre-downloads a cone catalog under the workspace
`outdir` by default and passes it to JHAT so JHAT does not query Gaia TAP
internally. Callers may still set ``gaia_refcat_path`` in ``jhat_params`` or
``skip_gaia_prefetch`` to override.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


@dataclass(frozen=True)
class GaiaPrefetchResult:
    path: str
    source: str  # "tap" or "vizier"
    n_rows: int


def icrs_field_center_from_fits(fits_path: str | os.PathLike[str]) -> SkyCoord:
    """
    ICRS sky position near the geometric center of the first HDU that has a
    usable 2D WCS and image data (SCI extensions are tried before others).
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    path = os.path.abspath(os.path.expanduser(os.fspath(fits_path)))
    last_exc: Exception | None = None
    with fits.open(path, mode="readonly") as hdul:
        indexed = list(enumerate(hdul))

        def sort_key(item: tuple[int, object]) -> tuple[int, int]:
            _i, h = item
            extname = str(h.header.get("EXTNAME", "") or "").strip().upper()
            name_attr = str(getattr(h, "name", "") or "").strip().upper()
            pri = 0 if extname == "SCI" or name_attr == "SCI" else 1
            return (pri, item[0])

        indexed.sort(key=sort_key)
        for _i, hdu in indexed:
            if getattr(hdu, "data", None) is None:
                continue
            if int(hdu.header.get("NAXIS", 0) or 0) < 2:
                continue
            try:
                w = WCS(hdu.header, hdul, naxis=2)
                shp = np.asarray(hdu.data.shape)
                ny, nx = int(shp[-2]), int(shp[-1])
                xc = 0.5 * float(max(0, nx - 1))
                yc = 0.5 * float(max(0, ny - 1))
                pt = w.pixel_to_world(xc, yc)
                if hasattr(pt, "transform_to"):
                    return pt.transform_to("icrs")
                return SkyCoord(ra=pt.ra, dec=pt.dec, frame="icrs")
            except Exception as exc:
                last_exc = exc
                continue
    msg = f"Could not derive ICRS field center from {path!r} for Gaia prefetch."
    if last_exc:
        raise RuntimeError(msg) from last_exc
    raise RuntimeError(msg)


# Prefetch file schema version. Bump when columns required by gaia_simple change
# (PM/parallax epoch correction and RUWE / excess-noise quality cuts).
GAIA_PREFETCH_CACHE_VERSION = "v2"


def gaia_prefetch_cache_path(
    outdir: str | os.PathLike[str],
    center: SkyCoord,
    radius: u.Quantity,
) -> str:
    """
    Stable filename under *outdir* so the same pointing and cone reuse one file.

    Coordinates are rounded so nearby exposures from one visit map to one path.
    """
    od = os.path.abspath(os.path.expanduser(os.fspath(outdir)))
    ra = round(float(center.icrs.ra.deg), 4)
    dec = round(float(center.icrs.dec.deg), 4)
    rd = round(float(radius.to(u.deg).value), 5)
    return os.path.join(
        od,
        f"hst123_gaia_dr3_prefetch_{GAIA_PREFETCH_CACHE_VERSION}"
        f"_ra{ra}_dec{dec}_r{rd}deg.txt",
    )


def _write_jhat_refcat(path: str, *, tab: Table) -> None:
    """
    Write a minimal JHAT-compatible custom refcat.

    Required for hst123/JHAT usage:
    - ra, dec (deg)
    - mag, dmag (Gaia G and uncertainty)
    Optional:
    - ID (source_id)
    - pmra, pmdec (mas/yr)
    - parallax (mas)
    - ruwe
    - astrometric_excess_noise (mas)
    """
    out = Table()
    cols = {c.lower(): c for c in tab.colnames}

    def _col(*names: str) -> str | None:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    ra_c = _col("ra", "ra_icrs", "ra_deg")
    dec_c = _col("dec", "de_icrs", "dec_deg")
    if ra_c is None or dec_c is None:
        raise ValueError("Gaia prefetch: could not identify RA/Dec columns.")

    g_c = _col("phot_g_mean_mag", "gmag", "g")
    if g_c is None:
        raise ValueError("Gaia prefetch: could not identify G magnitude column.")
    eg_c = _col("phot_g_mean_mag_error", "e_gmag", "e_g")

    out["ra"] = np.asarray(tab[ra_c], dtype=float)
    out["dec"] = np.asarray(tab[dec_c], dtype=float)
    out["mag"] = np.asarray(tab[g_c], dtype=float)
    if eg_c is not None:
        out["dmag"] = np.asarray(tab[eg_c], dtype=float)
    else:
        out["dmag"] = np.full(len(out), 0.02, dtype=float)

    sid_c = _col("source_id", "id")
    if sid_c is not None:
        # Keep as string to avoid int overflow on some platforms
        out["ID"] = np.asarray(tab[sid_c]).astype(str)

    pmra_c = _col("pmra", "pmra_masyr")
    pmdec_c = _col("pmdec", "pmdec_masyr")
    if pmra_c is not None and pmdec_c is not None:
        out["pmra"] = np.asarray(tab[pmra_c], dtype=float)
        out["pmdec"] = np.asarray(tab[pmdec_c], dtype=float)

    plx_c = _col("parallax", "plx", "parallax_mas")
    if plx_c is not None:
        out["parallax"] = np.asarray(tab[plx_c], dtype=float)

    ruwe_c = _col("ruwe")
    if ruwe_c is not None:
        out["ruwe"] = np.asarray(tab[ruwe_c], dtype=float)

    aen_c = _col("astrometric_excess_noise", "aen", "excess_noise")
    if aen_c is not None:
        out["astrometric_excess_noise"] = np.asarray(tab[aen_c], dtype=float)

    out.write(path, format="ascii.basic", overwrite=True)


def prefetch_gaia_catalog(
    *,
    center: SkyCoord,
    radius: u.Quantity,
    out_path: str,
    tap_attempts: int = 3,
    vizier_fallback: bool = True,
) -> GaiaPrefetchResult:
    """
    Prefetch a Gaia DR3 cone-search catalog to *out_path*.

    Strategy:
    - Try Gaia TAP (astroquery.gaia) with retries for transient HTTP 500 issues.
    - If TAP keeps failing and *vizier_fallback* is True, use Vizier Gaia DR3.

    The output file is a whitespace-delimited table with columns compatible with
    hst123's JHAT wrapper: ra, dec, mag, dmag (and optional ID, pmra, pmdec).
    """
    out_path = os.path.abspath(os.path.expanduser(os.fspath(out_path)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # 1) TAP attempt(s)
    last_exc: Exception | None = None
    try:
        from astroquery.gaia import Gaia  # type: ignore

        ra = float(center.icrs.ra.deg)
        dec = float(center.icrs.dec.deg)
        rad_deg = float(radius.to(u.deg).value)

        # Columns for JHAT + gaia_simple epoch correction / quality cuts.
        query = f"""
        SELECT
          source_id,
          ra, dec,
          pmra, pmdec,
          parallax,
          ruwe,
          astrometric_excess_noise,
          phot_g_mean_mag,
          phot_g_mean_mag_error
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
          POINT('ICRS', ra, dec),
          CIRCLE('ICRS', {ra}, {dec}, {rad_deg})
        )
        """

        for attempt in range(1, max(1, int(tap_attempts)) + 1):
            try:
                job = Gaia.launch_job_async(query)
                tab = job.get_results()
                _write_jhat_refcat(out_path, tab=tab)
                return GaiaPrefetchResult(path=out_path, source="tap", n_rows=len(tab))
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                retryable = "Error 500" in msg or "Cannot find result 'result'" in msg
                if (not retryable) or attempt >= tap_attempts:
                    break
                time.sleep(2.0 * attempt)
    except Exception as exc:
        last_exc = exc

    # 2) Vizier fallback
    if not vizier_fallback:
        raise last_exc or RuntimeError("Gaia prefetch: TAP failed (unknown error).")

    from astroquery.vizier import Vizier  # type: ignore

    v = Vizier(
        columns=[
            "RA_ICRS",
            "DE_ICRS",
            "Source",
            "pmRA",
            "pmDE",
            "Plx",
            "RUWE",
            "epsi",
            "Gmag",
            "e_Gmag",
        ]
    )
    v.ROW_LIMIT = -1
    tabs = v.query_region(center.icrs, radius=radius, catalog="I/355/gaiadr3")
    if not tabs:
        raise last_exc or RuntimeError("Gaia prefetch: Vizier returned no tables.")
    tab = tabs[0]

    # Normalize Vizier column names to the same writer.
    norm = Table()
    norm["source_id"] = np.asarray(tab["Source"]).astype(str)
    norm["ra"] = np.asarray(tab["RA_ICRS"], dtype=float)
    norm["dec"] = np.asarray(tab["DE_ICRS"], dtype=float)
    if "pmRA" in tab.colnames and "pmDE" in tab.colnames:
        norm["pmra"] = np.asarray(tab["pmRA"], dtype=float)
        norm["pmdec"] = np.asarray(tab["pmDE"], dtype=float)
    if "Plx" in tab.colnames:
        norm["parallax"] = np.asarray(tab["Plx"], dtype=float)
    if "RUWE" in tab.colnames:
        norm["ruwe"] = np.asarray(tab["RUWE"], dtype=float)
    # Vizier "epsi" is Gaia astrometric excess noise (mas).
    if "epsi" in tab.colnames:
        norm["astrometric_excess_noise"] = np.asarray(tab["epsi"], dtype=float)
    norm["phot_g_mean_mag"] = np.asarray(tab["Gmag"], dtype=float)
    if "e_Gmag" in tab.colnames:
        norm["phot_g_mean_mag_error"] = np.asarray(tab["e_Gmag"], dtype=float)

    _write_jhat_refcat(out_path, tab=norm)
    return GaiaPrefetchResult(path=out_path, source="vizier", n_rows=len(norm))

