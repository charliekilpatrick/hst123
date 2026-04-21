"""Read DOLPHOT catalogs, parse photometry, compute limits, print results. Used by pipeline scrapedolphot."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord

from hst123.primitives.base import BasePrimitive
from hst123.utils.display import show_photometry as display_show_photometry
from hst123.utils.dolphot_catalog_hdf5 import (
    find_column_index_0based,
    load_dolphot_catalog_array,
    parse_column_index_and_description,
    parse_dolphot_columns_file,
)
from hst123.utils.wcs_utils import wcs_from_fits_hdu

log = logging.getLogger(__name__)

# Limiting-magnitude statistics do not need every neighbor in the aperture; a
# modest random sample keeps scrape time bounded for large catalogs.
_LIMIT_SAMPLE_MAX = 400

# Show parse_phot progress (bar or periodic log lines) when this many sources.
_SCRAPE_PARSE_PROGRESS_MIN = 8


def _eta_hms(seconds: float) -> str:
    """Human-readable ETA from seconds (or ``?`` if unknown)."""
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def _meta_strings_from_array(rd: np.ndarray, ix_x, ix_y, ix_sh, ix_rn):
    """Format Object X/Y/sharpness/roundness for ``final_phot.meta`` (array row path)."""
    n = len(rd)

    def _one(ix):
        if ix is None or ix < 0 or ix >= n:
            return None
        v = float(rd[ix])
        if not np.isfinite(v):
            return None
        return str(v)

    return _one(ix_x), _one(ix_y), _one(ix_sh), _one(ix_rn)


def _iter_ftable_inst_visit_filter(obstable):
    """Yield subtables for each (instrument, visit, filter) combination."""
    for inst in np.unique(obstable["instrument"]):
        m0 = obstable["instrument"] == inst
        sub0 = obstable[m0]
        for visit in np.unique(sub0["visit"]):
            m1 = sub0["visit"] == visit
            sub1 = sub0[m1]
            for filt in np.unique(sub1["filter"]):
                yield sub1[sub1["filter"] == filt]


def _iter_ftable_inst_filter(obstable):
    """Yield subtables for each (instrument, filter) (all visits combined)."""
    for inst in np.unique(obstable["instrument"]):
        sub0 = obstable[obstable["instrument"] == inst]
        for filt in np.unique(sub0["filter"]):
            yield sub0[sub0["filter"] == filt]


def _subsample_limit_rows(limit_data, max_rows, rng):
    """Randomly subsample ``[dist, row]`` limit rows for fast limit estimation."""
    if len(limit_data) <= max_rows:
        return limit_data
    pick = rng.choice(len(limit_data), size=max_rows, replace=False)
    pick.sort()
    return [limit_data[i] for i in pick]


class ScrapeDolphotPrimitive(BasePrimitive):
    """
    Read dolphot catalogs, parse photometry, and print final photometry.

    Provides get_dolphot_column, get_dolphot_data, get_limit_data, calc_avg_stats,
    parse_phot, print_final_phot, and scrapedolphot for pipeline coord.
    """

    def __init__(self, pipeline):
        super().__init__(pipeline)
        self._col_cache: dict[tuple[str, float], list] = {}

    def _columns_for(self, colfile):
        """Parse and cache ``*.columns`` by (abspath, mtime)."""
        if not isinstance(colfile, str):
            colfile = str(colfile)
        p = os.path.abspath(os.path.expanduser(colfile))
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            return None
        key = (p, mtime)
        if key not in self._col_cache:
            self._col_cache[key] = parse_dolphot_columns_file(p)
        return self._col_cache[key]

    @staticmethod
    def _row_tokens(row):
        if isinstance(row, str):
            return row.split()
        if isinstance(row, np.ndarray):
            return row
        return row

    def get_dolphot_column(self, colfile, key, image, offset=0):
        """
        Get column index for a key from dolphot columns file (optionally with offset).

        Parameters
        ----------
        colfile : str or file-like
            Path to or open dolphot .columns file.
        key : str
            Column label to find (e.g. "Object X", "Magnitude uncertainty").
        image : str
            Image base name (e.g. "" or filename without .fits) to match line.
        offset : int, optional
            Added to 0-based column index. Default 0.

        Returns
        -------
        int or None
            Column index (0-based) + offset, or None if not found.
        """
        columns = self._columns_for(colfile)
        if columns is not None:
            idx = find_column_index_0based(columns, key, image)
            if idx is not None:
                return idx + offset
            return None
        coldata = ""
        with open(colfile) as colfile_data:
            for line in colfile_data:
                if image.replace(".fits", "") in line and key in line:
                    coldata = line.strip().strip("\n")
                    break
        if not coldata:
            return None
        parsed = parse_column_index_and_description(coldata)
        if not parsed:
            return None
        n1, _desc = parsed
        try:
            return int(n1) - 1 + offset
        except Exception:
            return None

    def get_dolphot_data(self, row, colfile, key, image):
        """
        Get value for a column key from a dolphot output row.

        Parameters
        ----------
        row : str or array-like
            One line or row from dolphot output (whitespace-separated).
        colfile : str or file-like
            Dolphot .columns file.
        key : str
            Column label.
        image : str
            Image base name for column lookup.

        Returns
        -------
        str or None
            Cell value, or None if column missing or invalid.
        """
        colnum = self.get_dolphot_column(colfile, key, image)
        rdata = self._row_tokens(row)
        if colnum is not None:
            if colnum < 0 or colnum > len(rdata) - 1:
                log.error(
                    "ERROR: tried to use bad column %s in dolphot output",
                    colnum,
                )
                return None
            v = rdata[colnum]
            if isinstance(v, (np.floating, float, np.integer, int)):
                return str(float(v))
            return str(v)
        return None

    def print_final_phot(self, final_phot, dolphot, allphot=True):
        """
        Write final photometry tables and snana files for each source.

        Parameters
        ----------
        final_phot : list of astropy.table.Table
            One table per source with MJD, INSTRUMENT, FILTER, EXPTIME, MAGNITUDE, etc.
        dolphot : dict
            Must contain "final_phot" key (base output path, e.g. dp.phot).
        allphot : bool, optional
            If True, write dp_001.phot, dp_002.phot, etc. Default True.
        """
        p = self._p
        written = []
        for i, phot in enumerate(final_phot):
            outfile = dolphot["final_phot"]
            if not allphot:
                out = outfile
            else:
                num = str(i).zfill(len(str(len(final_phot))))
                out = outfile.replace(".phot", "_" + num + ".phot")
            snana = out.replace(".phot", ".snana")
            with open(out, "w") as f:
                display_show_photometry(
                    phot,
                    f=f,
                    coord=p.coord,
                    options=p.options,
                    log=log,
                    log_rows=False,
                )
            written.append(out)
            with open(snana, "w") as f:
                display_show_photometry(
                    phot,
                    f=f,
                    snana=True,
                    show=False,
                    coord=p.coord,
                    options=p.options,
                    log=log,
                    log_rows=False,
                )
            written.append(snana)
        log.info(
            "Wrote photometry for %d source(s) to %s and matching .snana files.",
            len(final_phot),
            dolphot["final_phot"],
        )
        self._primitive_cleanup(
            "print_final_phot",
            validate_text_paths=written,
            text_min_size=0,
            validation_notes={"n_sources": len(final_phot)},
        )

    def log_scrape_summary(self, phot_tables: list) -> None:
        """Log aggregate scrape statistics: source count, magnitude range, median limiting mags."""
        if not phot_tables:
            return
        p = self._p
        try:
            snr = float(getattr(p, "snr_limit", 3.0))
        except Exception:
            snr = 3.0
        n_src = len(phot_tables)
        limits_by_band: dict[tuple[str, str], list[float]] = {}
        mags_by_band: dict[tuple[str, str], list[float]] = {}
        has_limit_col = False
        for tbl in phot_tables:
            if tbl is None or len(tbl) == 0:
                continue
            if "IS_AVG" not in tbl.colnames:
                continue
            sub = tbl[tbl["IS_AVG"] == 1]
            if len(sub) == 0:
                continue
            if "LIMIT" in sub.colnames:
                has_limit_col = True
            for row in sub:
                inst = str(row["INSTRUMENT"]).strip()
                filt = str(row["FILTER"]).strip()
                key = (inst, filt)
                m = row["MAGNITUDE"]
                if m is not None and np.isfinite(m):
                    mags_by_band.setdefault(key, []).append(float(m))
                if "LIMIT" in sub.colnames:
                    lim = row["LIMIT"]
                    if lim is not None and np.isfinite(lim) and float(lim) < 99:
                        limits_by_band.setdefault(key, []).append(float(lim))

        log.info(
            "Scrape catalog summary: %d source(s) with instrument/filter-averaged rows (IS_AVG=1).",
            n_src,
        )
        if mags_by_band:
            bits = []
            for (inst, filt) in sorted(mags_by_band.keys()):
                vals = np.array(mags_by_band[(inst, filt)], dtype=float)
                bits.append(
                    "%s/%s: mag median=%.2f min=%.2f max=%.2f"
                    % (
                        inst,
                        filt,
                        float(np.nanmedian(vals)),
                        float(np.nanmin(vals)),
                        float(np.nanmax(vals)),
                    )
                )
            log.info("Scrape photometry (per band, over sources): %s.", "; ".join(bits))
        if has_limit_col and limits_by_band:
            lbits = []
            for (inst, filt) in sorted(limits_by_band.keys()):
                vals = np.array(limits_by_band[(inst, filt)], dtype=float)
                lbits.append(
                    "%s/%s: median m_lim=%.2f (%.0fσ over %d sources)"
                    % (
                        inst,
                        filt,
                        float(np.nanmedian(vals)),
                        snr,
                        len(vals),
                    )
                )
            log.info(
                "Scrape limiting magnitude (neighbor-based median per source, then median over sources): %s.",
                "; ".join(lbits),
            )
        elif has_limit_col:
            log.info(
                "Scrape limiting magnitude: no finite LIMIT values (per band); "
                "try a larger scrape radius or more neighbors for limits.",
            )

    def get_limit_data(
        self,
        dolphot,
        coord,
        w,
        x,
        y,
        colfile,
        limit_radius,
        catalog=None,
        xcol=None,
        ycol=None,
    ):
        """
        Get dolphot rows within limit_radius (arcsec) of (x, y) for limit estimation.

        Parameters
        ----------
        dolphot : dict
            Must have "base" key (path to dolphot output).
        coord : astropy.coordinates.SkyCoord
            Target coordinate.
        w : astropy.wcs.WCS
            WCS for pixel/sky conversion.
        x, y : float
            Reference pixel position.
        colfile : str
            Dolphot .columns file path.
        limit_radius : float
            Radius in pixels (derived from arcsec) for inclusion.
        catalog : ndarray, optional
            Pre-loaded 2-D catalog (n_sources, n_columns). When given with *xcol*
            and *ycol*, uses a vectorized pass (fast).
        xcol, ycol : int, optional
            0-based column indices for Object X / Y on the reference image.

        Returns
        -------
        list of list
            Each element [dist, line] where *line* is a catalog row (ndarray) or
            original text line (str).
        """
        if coord.dec.degree < 89:
            dec1 = coord.dec.degree + limit_radius / 3600.0
        else:
            dec1 = coord.dec.degree - limit_radius / 3600.0
        coord1 = SkyCoord(coord.ra.degree, dec1, unit="deg")
        x1, y1 = wcs.utils.skycoord_to_pixel(coord1, w, origin=1)
        radius = float(np.sqrt((x - x1) ** 2 + (y - y1) ** 2))

        if (
            catalog is not None
            and xcol is not None
            and ycol is not None
            and catalog.ndim == 2
        ):
            xline = catalog[:, xcol].astype(np.float64, copy=False) + 0.5
            yline = catalog[:, ycol].astype(np.float64, copy=False) + 0.5
            dist = np.hypot(xline - x, yline - y)
            m = dist < radius
            ii = np.flatnonzero(m)
            return [[float(dist[i]), catalog[i]] for i in ii]

        limit_data = []
        xi = xcol
        yi = ycol
        if xi is None:
            xi = self.get_dolphot_column(colfile, "Object X", "")
        if yi is None:
            yi = self.get_dolphot_column(colfile, "Object Y", "")
        if xi is None or yi is None:
            return limit_data
        with open(dolphot["base"]) as dp:
            for line in dp:
                parts = line.split()
                xline = float(parts[xi]) + 0.5
                yline = float(parts[yi]) + 0.5
                dist = np.sqrt((xline - x) ** 2 + (yline - y) ** 2)
                if dist < radius:
                    limit_data.append([dist, line])
        return limit_data

    def _calc_avg_stats_array(self, obstable, rd, measurement_idx, _colfile):
        """Vector-friendly path when the DOLPHOT row is a 1-D float array."""
        p = self._p
        n = len(obstable)
        if n == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        rd = np.asarray(rd, dtype=np.float64, order="C")
        if rd.ndim != 1:
            return (np.nan, np.nan, np.nan, np.nan)
        nrd = rd.shape[0]
        imgs = obstable["image"]
        ei = np.zeros(n, dtype=np.intp)
        ci = np.zeros(n, dtype=np.intp)
        ok_idx = np.zeros(n, dtype=bool)
        for i in range(n):
            pair = measurement_idx.get(imgs[i]) if measurement_idx else None
            if (
                pair
                and pair[0] is not None
                and pair[1] is not None
                and 0 <= pair[0] < nrd
                and 0 <= pair[1] < nrd
            ):
                ei[i] = pair[0]
                ci[i] = pair[1]
                ok_idx[i] = True
        if not np.any(ok_idx):
            return (np.nan, np.nan, np.nan, np.nan)
        t_mjd = Time(obstable["datetime"]).mjd
        ext = obstable["exptime"].data.astype(np.float64, copy=False)
        zp = obstable["zeropoint"].data.astype(np.float64, copy=False)
        me = np.zeros(n, dtype=np.float64)
        mc = np.zeros(n, dtype=np.float64)
        me[ok_idx] = rd[ei[ok_idx]]
        mc[ok_idx] = rd[ci[ok_idx]]
        phot_ok = (
            ok_idx
            & (me < 0.5)
            & (mc > 0.0)
            & (ext > 0.0)
            & (zp > 0.0)
            & np.isfinite(me)
            & np.isfinite(mc)
        )
        if not np.any(phot_ok):
            return (np.nan, np.nan, np.nan, np.nan)
        avg_mjd = float(np.mean(t_mjd[phot_ok]))
        total_exptime = float(np.sum(ext[phot_ok]))
        mag, magerr = p._phot.avg_magnitudes(
            me[phot_ok], mc[phot_ok], ext[phot_ok], zp[phot_ok]
        )
        return (avg_mjd, mag, magerr, total_exptime)

    def calc_avg_stats(self, obstable, data, colfile, measurement_idx=None):
        """
        Compute average MJD, magnitude, error, and total exptime for one source/filter.

        Parameters
        ----------
        obstable : astropy.table.Table
            Rows with image, datetime, exptime, filter, detector, zeropoint.
        data : str or array-like
            Dolphot row(s) or line for this source.
        colfile : str
            Dolphot .columns file.
        measurement_idx : dict, optional
            Maps each ``obstable['image']`` to ``(magnitude_uncertainty_col,
            measured_counts_col)`` 0-based indices. Built automatically when omitted.

        Returns
        -------
        tuple
            (avg_mjd, mag, magerr, total_exptime) or (nan, nan, nan, nan) if no data.
        """
        p = self._p
        estr = "Magnitude uncertainty"
        cstr = "Measured counts"
        if measurement_idx is None:
            columns = self._columns_for(colfile)
            if columns is None:
                columns = parse_dolphot_columns_file(
                    os.path.abspath(str(colfile))
                )
            imgs = list(dict.fromkeys(obstable["image"]))
            measurement_idx = {
                img: (
                    find_column_index_0based(columns, estr, img),
                    find_column_index_0based(columns, cstr, img),
                )
                for img in imgs
            }
        rd = self._row_tokens(data)
        if isinstance(rd, np.ndarray) and rd.ndim == 1 and measurement_idx:
            return self._calc_avg_stats_array(
                obstable, rd, measurement_idx, colfile
            )
        mjds, err, counts, exptime, filts, det, zpt = [], [], [], [], [], [], []
        for row in obstable:
            img = row["image"]
            pair = measurement_idx.get(img) if measurement_idx else None
            if pair and pair[0] is not None and pair[1] is not None:
                ei, ci = pair
                if ei >= len(rd) or ci >= len(rd):
                    error = count = None
                else:
                    error = str(float(rd[ei]))
                    count = str(float(rd[ci]))
            else:
                error = self.get_dolphot_data(data, colfile, estr, img)
                count = self.get_dolphot_data(data, colfile, cstr, img)
            if error and count:
                mjds.append(Time(row["datetime"]).mjd)
                err.append(error)
                counts.append(count)
                exptime.append(row["exptime"])
                filts.append(row["filter"])
                det.append(row["detector"])
                zpt.append(row["zeropoint"])
        if len(mjds) > 0:
            avg_mjd = np.mean(mjds)
            total_exptime = np.sum(exptime)
            mag, magerr = p._phot.avg_magnitudes(err, counts, exptime, zpt)
            return (avg_mjd, mag, magerr, total_exptime)
        return (np.nan, np.nan, np.nan, np.nan)

    def _precompute_limit_maglimits_for_aperture(
        self,
        obstable,
        limit_data,
        cfile,
        measurement_idx,
    ):
        """
        Limiting magnitudes from neighbor rows depend only on obstable and the
        aperture (not on which catalog source is being parsed). Precompute once
        per :meth:`parse_phot` call and reuse for every source in ``--scrape-all``
        to avoid millions of redundant ``calc_avg_stats`` calls.
        """
        visit_m: dict[tuple[str, str, str], float] = {}
        inst_m: dict[tuple[str, str], float] = {}
        if not limit_data:
            return visit_m, inst_m
        for ftable in _iter_ftable_inst_visit_filter(obstable):
            key = (
                str(ftable["instrument"][0]),
                str(ftable["visit"][0]),
                str(ftable["filter"][0]),
            )
            visit_m[key] = self._limit_mag_from_neighbors(
                ftable,
                limit_data,
                cfile,
                measurement_idx,
                reject_limmag_ge_99=True,
            )
        for ftable in _iter_ftable_inst_filter(obstable):
            key = (str(ftable["instrument"][0]), str(ftable["filter"][0]))
            inst_m[key] = self._limit_mag_from_neighbors(
                ftable,
                limit_data,
                cfile,
                measurement_idx,
                reject_limmag_ge_99=False,
            )
        return visit_m, inst_m

    def _limit_mag_from_neighbors(
        self,
        ftable,
        limit_data,
        cfile,
        measurement_idx,
        *,
        reject_limmag_ge_99: bool,
    ):
        """
        Median-based limiting magnitude from neighbor rows; returns NaN if too few points.
        """
        if not limit_data:
            return np.nan
        mags, errs = [], []
        ld = limit_data
        cas = self.calc_avg_stats
        for data in ld:
            _mjd, limmag, limerr, _exp = cas(
                ftable,
                data[1],
                cfile,
                measurement_idx=measurement_idx,
            )
            if np.isnan(limmag) or np.isnan(limerr):
                continue
            if reject_limmag_ge_99 and limmag >= 99:
                continue
            mags.append(limmag)
            errs.append(limerr)
        if len(mags) <= 30:
            return np.nan
        return self._p._phot.estimate_mag_limit(
            mags, errs, limit=self._p.snr_limit
        )

    def _add_parse_phot_block(
        self,
        ftable,
        row,
        cfile,
        measurement_idx,
        limit_data,
        final_phot,
        is_avg: int,
        reject_limmag_ge_99: bool,
        maglimit_precomputed: float | None = None,
    ):
        """One instrument[/visit]/filter table → one output row (and limit column if needed)."""
        p = self._p
        mjd, mag, err, exptime = self.calc_avg_stats(
            ftable, row, cfile, measurement_idx=measurement_idx
        )
        inst = str(ftable["instrument"][0])
        filt = str(ftable["filter"][0])
        if limit_data:
            if maglimit_precomputed is not None:
                maglimit = float(maglimit_precomputed)
            else:
                maglimit = self._limit_mag_from_neighbors(
                    ftable,
                    limit_data,
                    cfile,
                    measurement_idx,
                    reject_limmag_ge_99=reject_limmag_ge_99,
                )
            final_phot.add_row(
                (mjd, inst, filt, exptime, mag, err, is_avg, maglimit)
            )
        else:
            final_phot.add_row((mjd, inst, filt, exptime, mag, err, is_avg))

    def parse_phot(
        self,
        obstable,
        row,
        cfile,
        limit_data=None,
        measurement_idx=None,
        columns=None,
        limit_maglimit_caches=None,
    ):
        """
        Build photometry table for one dolphot source (per instrument/filter, with optional limits).

        Parameters
        ----------
        obstable : astropy.table.Table
            Image list with instrument, visit, filter, datetime, exptime, zeropoint.
        row : str or array-like
            Dolphot output row for this source.
        cfile : str
            Dolphot .columns file.
        limit_data : list, optional
            From get_limit_data for limit magnitude estimation. Default None.
        measurement_idx : dict, optional
            Precomputed per-image column indices for ``calc_avg_stats``.
        columns : list, optional
            Parsed ``*.columns`` definitions (avoids re-parsing when given).
        limit_maglimit_caches : tuple of two dicts, optional
            ``(visit_maglimits, inst_maglimits)`` from
            :meth:`_precompute_limit_maglimits_for_aperture`. When set, LIMIT
            values are taken from these dicts (same for all sources in an aperture).

        Returns
        -------
        astropy.table.Table
            Table with MJD, INSTRUMENT, FILTER, EXPTIME, MAGNITUDE, MAGNITUDE_ERROR, IS_AVG, [LIMIT].
        """
        p = self._p
        limit_data = limit_data or []
        if columns is None:
            columns = self._columns_for(cfile)
        if columns is None:
            columns = parse_dolphot_columns_file(os.path.abspath(str(cfile)))
        if measurement_idx is None:
            estr = "Magnitude uncertainty"
            cstr = "Measured counts"
            imgs = list(dict.fromkeys(obstable["image"]))
            measurement_idx = {
                img: (
                    find_column_index_0based(columns, estr, img),
                    find_column_index_0based(columns, cstr, img),
                )
                for img in imgs
            }
        fnames = [
            "MJD",
            "INSTRUMENT",
            "FILTER",
            "EXPTIME",
            "MAGNITUDE",
            "MAGNITUDE_ERROR",
            "IS_AVG",
        ]
        init_row = [[0.0], ["X" * 24], ["X" * 12], [0.0], [0.0], [0.0], [0]]
        if limit_data:
            fnames += ["LIMIT"]
            init_row += [[0.0]]
        final_phot = Table(init_row, names=fnames)
        final_phot = final_phot[:0].copy()

        rd = self._row_tokens(row)
        if isinstance(rd, np.ndarray) and rd.ndim == 1:
            ix_x = find_column_index_0based(columns, "Object X", "")
            ix_y = find_column_index_0based(columns, "Object Y", "")
            ix_sh = find_column_index_0based(columns, "Object sharpness", "")
            ix_rn = find_column_index_0based(columns, "Object roundness", "")
            x, y, sharpness, roundness = _meta_strings_from_array(
                rd, ix_x, ix_y, ix_sh, ix_rn
            )
        else:
            x = self.get_dolphot_data(row, cfile, "Object X", "")
            y = self.get_dolphot_data(row, cfile, "Object Y", "")
            sharpness = self.get_dolphot_data(row, cfile, "Object sharpness", "")
            roundness = self.get_dolphot_data(row, cfile, "Object roundness", "")
        final_phot.meta["x"] = x
        final_phot.meta["y"] = y
        final_phot.meta["sharpness"] = sharpness
        final_phot.meta["roundness"] = roundness
        if limit_data:
            final_phot.meta["limit"] = "{0}-sigma estimate".format(
                int(p.snr_limit)
            )

        visit_m, inst_m = limit_maglimit_caches or (None, None)
        for ftable in _iter_ftable_inst_visit_filter(obstable):
            key = (
                str(ftable["instrument"][0]),
                str(ftable["visit"][0]),
                str(ftable["filter"][0]),
            )
            pre = visit_m.get(key) if visit_m is not None else None
            self._add_parse_phot_block(
                ftable,
                row,
                cfile,
                measurement_idx,
                limit_data,
                final_phot,
                0,
                reject_limmag_ge_99=True,
                maglimit_precomputed=pre,
            )
        for ftable in _iter_ftable_inst_filter(obstable):
            key = (str(ftable["instrument"][0]), str(ftable["filter"][0]))
            pre = inst_m.get(key) if inst_m is not None else None
            self._add_parse_phot_block(
                ftable,
                row,
                cfile,
                measurement_idx,
                limit_data,
                final_phot,
                1,
                reject_limmag_ge_99=False,
                maglimit_precomputed=pre,
            )
        return final_phot

    def scrapedolphot(
        self,
        coord,
        reference,
        images,
        dolphot,
        scrapeall=False,
        get_limits=False,
        brightest=False,
        visit_obstable=None,
    ):
        """
        Scrape dolphot catalog for sources near coord; return list of photometry tables.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Target coordinate.
        reference : str
            Reference image path for WCS.
        images : list of str
            Split image paths for obstable.
        dolphot : dict
            Keys: base, colfile, original, radius, limit_radius (optional).
        scrapeall : bool, optional
            If True, return all candidates within radius; else closest. Default False.
        get_limits : bool, optional
            If True, compute limit magnitudes. Default False.
        brightest : bool, optional
            If True, sort by brightness instead of separation. Default False.
        visit_obstable : astropy.table.Table, optional
            Per-visit observation table. When provided, chip-level metadata is derived
            from this table instead of calling :meth:`input_list` on every split FITS
            (avoids repeated FITS I/O).

        Returns
        -------
        list of astropy.table.Table or None
            One table per source with meta (x, y, separation, etc.); None if no sources or error.
        """
        p = self._p
        p.__dict__.pop("_last_dolphot_catalog_array", None)
        wd = os.path.abspath(
            os.path.expanduser(getattr(p.options["args"], "work_dir", None) or ".")
        )

        def _scrapedolphot_cleanup(
            phot_tables,
            *,
            note: str = "",
        ) -> None:
            tmp_path = os.path.join(wd, "tmp")
            rpaths = [tmp_path] if os.path.isfile(tmp_path) else []
            vfit = []
            if reference and os.path.isfile(str(reference)):
                vfit.append(str(reference))
            vtxt = []
            b = dolphot.get("base")
            c = dolphot.get("colfile")
            if b and os.path.isfile(b):
                vtxt.append(b)
            if c and os.path.isfile(c):
                vtxt.append(c)
            self._primitive_cleanup(
                "scrapedolphot",
                work_dir=wd,
                remove_paths=rpaths,
                validate_fits_paths=vfit,
                validate_text_paths=vtxt,
                text_min_size=1,
                validate_tables=phot_tables if phot_tables is not None else [],
                validation_notes={"exit": note} if note else {},
            )

        base = dolphot["base"]
        colfile = dolphot["colfile"]
        if not os.path.isfile(base) or not os.path.isfile(colfile):
            log.error(
                "ERROR: dolphot output %s does not exist. Use --run-dolphot "
                "or check your dolphot output for errors",
                dolphot["base"],
            )
            _scrapedolphot_cleanup(None, note="missing_dolphot_io")
            return None
        if not reference or not coord:
            log.error(
                "ERROR: Need a reference image and coordinate to "
                "scrape data from the dolphot catalog. Exiting..."
            )
            return None

        columns = self._columns_for(colfile)
        if columns is None:
            columns = parse_dolphot_columns_file(os.path.abspath(colfile))
        obstable = None
        if visit_obstable is not None:
            obstable = p.expand_obstable_for_split_images(visit_obstable, images)
            if obstable is not None:
                p.refresh_obstable_zeropoints_from_fits(obstable)
        if obstable is None:
            obstable = p.input_list(images, show=False, save=False)
        if not obstable:
            _scrapedolphot_cleanup(None, note="empty_obstable")
            return None
        estr = "Magnitude uncertainty"
        cstr = "Measured counts"
        imgs = list(dict.fromkeys(obstable["image"]))
        measurement_idx = {
            img: (
                find_column_index_0based(columns, estr, img),
                find_column_index_0based(columns, cstr, img),
            )
            for img in imgs
        }

        if not os.path.exists(dolphot["original"]):
            shutil.copyfile(dolphot["base"], dolphot["original"])
        typecol = find_column_index_0based(columns, "Object type", "")
        if not p.options["args"].no_cuts:
            catalog = load_dolphot_catalog_array(base)
            if typecol is None:
                log.warning(
                    "Object type column not found; skipping cuts to dolphot output."
                )
            else:
                n0 = len(catalog)
                good = catalog[:, typecol] == 1
                n1 = int(np.count_nonzero(good))
                log.info(
                    "Dolphot catalog %s: %d sources, %d after type==1 cut.",
                    os.path.basename(base),
                    n0,
                    n1,
                )
                if n1 < n0:
                    catalog = catalog[good]
                    np.savetxt(base, catalog, fmt="%.16g")
        else:
            if os.path.exists(dolphot["original"]):
                shutil.copyfile(dolphot["original"], dolphot["base"])
            catalog = load_dolphot_catalog_array(base)

        ix = find_column_index_0based(columns, "Object X", "")
        iy = find_column_index_0based(columns, "Object Y", "")
        isn = find_column_index_0based(columns, "Signal-to-noise", "")
        icnt = find_column_index_0based(columns, "Normalized count rate", "")
        if ix is None or iy is None:
            log.error("Dolphot columns file missing Object X / Object Y.")
            _scrapedolphot_cleanup(None, note="missing_xy_columns")
            return None

        with fits.open(reference) as hdu:
            w = wcs_from_fits_hdu(hdu, 0)
            x, y = wcs.utils.skycoord_to_pixel(coord, w, origin=1)
            ra = coord.ra.degree
            dec = coord.dec.degree
            radius = dolphot["radius"]
            if p.options["args"].scrape_radius:
                angradius = p.options["args"].scrape_radius / 3600.0
                dec1 = (
                    coord.dec.degree + angradius
                    if dec < 89
                    else coord.dec.degree - angradius
                )
                coord1 = SkyCoord(ra, dec1, unit="deg")
                x1, y1 = wcs.utils.skycoord_to_pixel(coord1, w, origin=1)
                radius = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

        xline = catalog[:, ix].astype(np.float64, copy=False) + 0.5
        yline = catalog[:, iy].astype(np.float64, copy=False) + 0.5
        dist = np.hypot(xline - x, yline - y)
        mask = dist < float(radius)
        idx = np.flatnonzero(mask)
        data = []
        for i in idx:
            row = catalog[i]
            sn = (
                float(row[isn])
                if isn is not None and isn < row.shape[0]
                else float("nan")
            )
            cnt = (
                float(row[icnt])
                if icnt is not None and icnt < row.shape[0]
                else float("nan")
            )
            data.append(
                {
                    "sep": float(dist[i]),
                    "data": row,
                    "sn": sn,
                    "counts": cnt,
                }
            )

        limit_data = []
        if get_limits:
            limit_radius = dolphot["limit_radius"]
            limit_data = self.get_limit_data(
                dolphot,
                coord,
                w,
                x,
                y,
                colfile,
                limit_radius,
                catalog=catalog,
                xcol=ix,
                ycol=iy,
            )
            if limit_data:
                n_lim = len(limit_data)
                limit_data = _subsample_limit_rows(
                    limit_data,
                    _LIMIT_SAMPLE_MAX,
                    np.random.default_rng(
                        (hash(os.path.basename(base)) & 0xFFFFFFFF)
                        ^ (int(float(x) * 1000) & 0xFFFF)
                        ^ (int(float(y) * 1000) & 0xFFFF)
                    ),
                )
                if len(limit_data) < n_lim:
                    log.info(
                        "Limit sample: using %d of %d random aperture sources "
                        "for fast limit statistics.",
                        len(limit_data),
                        n_lim,
                    )
        log.info(
            "Scrape: x=%s y=%s r=%s px | %d source(s) in %s near %.6f %.6f",
            "%7.2f" % float(x),
            "%7.2f" % float(y),
            "%7.4f" % float(radius),
            len(data),
            os.path.basename(dolphot["base"]),
            ra,
            dec,
        )
        if len(data) == 0:
            _scrapedolphot_cleanup(None, note="no_sources_in_radius")
            return None
        if len(data) > 1:
            if brightest:
                data = sorted(data, key=lambda obj: obj["counts"], reverse=True)
                if not scrapeall:
                    log.info(
                        "Multiple sources in radius: choosing brightest "
                        "(SN/cnts ordering)."
                    )
            else:
                data = sorted(data, key=lambda obj: obj["sep"])
                if not scrapeall:
                    log.info(
                        "Multiple sources in radius: choosing closest to "
                        "target (%.6f, %.6f).",
                        ra,
                        dec,
                    )
        if not scrapeall:
            data = [data[0]]
            log.info(
                "Selected source: sep=%.4f px, S/N=%s",
                data[0]["sep"],
                data[0]["sn"],
            )

        limit_mag_caches_outer = None
        if limit_data:
            limit_mag_caches_outer = self._precompute_limit_maglimits_for_aperture(
                obstable,
                limit_data,
                colfile,
                measurement_idx,
            )

        def _parse_one(dat):
            finished = False
            limit_radius = dolphot.get("limit_radius", 10.0)
            limit_data_cur = limit_data
            while not finished:
                cur_caches = None
                if limit_data_cur:
                    if (
                        limit_data_cur is limit_data
                        and limit_mag_caches_outer is not None
                    ):
                        cur_caches = limit_mag_caches_outer
                    else:
                        cur_caches = self._precompute_limit_maglimits_for_aperture(
                            obstable,
                            limit_data_cur,
                            colfile,
                            measurement_idx,
                        )
                source_phot = self.parse_phot(
                    obstable,
                    dat["data"],
                    colfile,
                    limit_data=limit_data_cur,
                    measurement_idx=measurement_idx,
                    columns=columns,
                    limit_maglimit_caches=cur_caches,
                )
                finished = True
                if "LIMIT" in source_phot.keys():
                    if any(np.isnan(row["LIMIT"]) for row in source_phot):
                        if limit_radius < 80:
                            limit_radius = 2 * limit_radius
                            finished = False
                            limit_data_cur = self.get_limit_data(
                                dolphot,
                                coord,
                                w,
                                x,
                                y,
                                colfile,
                                limit_radius,
                                catalog=catalog,
                                xcol=ix,
                                ycol=iy,
                            )
                            if len(limit_data_cur) > _LIMIT_SAMPLE_MAX:
                                limit_data_cur = _subsample_limit_rows(
                                    limit_data_cur,
                                    _LIMIT_SAMPLE_MAX,
                                    np.random.default_rng(
                                        int(limit_radius * 1000) & 0xFFFFFFFF
                                    ),
                                )
            source_phot.meta["separation"] = dat["sep"]
            source_phot.meta["magsystem"] = p.magsystem
            if "x" in source_phot.meta.keys() and "y" in source_phot.meta.keys():
                xm = [float(source_phot.meta["x"])]
                ym = [float(source_phot.meta["y"])]
                coord_out = wcs.utils.pixel_to_skycoord(
                    xm, ym, w, origin=1
                )[0]
                source_phot.meta["ra"] = coord_out.ra.degree
                source_phot.meta["dec"] = coord_out.dec.degree
            source_phot.sort(["MJD", "FILTER"])
            return source_phot

        mc = getattr(p.options["args"], "max_cores", None)
        if mc is None:
            mc = min(8, (os.cpu_count() or 4))
        else:
            mc = max(1, int(mc))
        cpu = os.cpu_count() or 8
        # Large --scrape-all batches: allow more threads than max_cores default cap.
        nd = len(data)
        n_workers = (
            min(max(mc, min(cpu * 2, 48)), nd) if nd > 1 else 1
        )
        show_progress = (
            nd >= _SCRAPE_PARSE_PROGRESS_MIN
            and os.environ.get("HST123_NO_SCRAPE_PROGRESS", "").strip().lower()
            not in ("1", "true", "yes", "on")
        )
        prefer_log_lines = (
            show_progress
            and (
                not sys.stderr.isatty()
                or os.environ.get("HST123_SCRAPE_PROGRESS_LOG_ONLY", "").strip().lower()
                in ("1", "true", "yes", "on")
            )
        )

        def _indexed(src_idx: int, item) -> tuple[int, Table]:
            # Must not name this ``ix`` — outer scope uses ``ix``/``iy`` for DOLPHOT
            # Object X/Y column indices; reusing ``ix`` here clobbers them while worker
            # threads still run (late-binding closures), causing IndexError in get_limit_data.
            return src_idx, _parse_one(item)

        if n_workers > 1 and nd > 1:
            if show_progress:
                log.info(
                    "Scrape parse_phot: starting %d source(s) with %d worker thread(s)…",
                    nd,
                    n_workers,
                )
            results: list = [None] * nd
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_indexed, i, data[i]) for i in range(nd)]
                pb = None
                if show_progress and not prefer_log_lines:
                    try:
                        import progressbar

                        widgets: list = [
                            "Scrape parse_phot ",
                            progressbar.Bar(marker="#", left="[", right="]"),
                            " ",
                            progressbar.Percentage(),
                            " ",
                            progressbar.SimpleProgress(),
                            " ",
                            progressbar.Timer(format="elapsed %s "),
                            progressbar.ETA(format="ETA %s"),
                        ]
                        try:
                            pb = progressbar.ProgressBar(
                                max_value=nd,
                                widgets=widgets,
                                fd=sys.stderr,
                            )
                        except TypeError:
                            pb = progressbar.ProgressBar(
                                maxval=nd,
                                widgets=widgets,
                                fd=sys.stderr,
                            )
                        try:
                            pb.start()
                        except Exception:
                            pb = None
                    except Exception:
                        pb = None
                # Log lines when user asked, or bar unavailable (TTY), or bar init failed.
                use_log_steps = bool(
                    show_progress and (prefer_log_lines or pb is None)
                )
                done = 0
                step = max(1, nd // 20)
                last_emit = t0
                for fut in as_completed(futs):
                    src_idx, phot = fut.result()
                    results[src_idx] = phot
                    done += 1
                    if pb is not None:
                        try:
                            pb.update(done)
                        except Exception:
                            pass
                    if use_log_steps:
                        now = time.perf_counter()
                        if (
                            done == nd
                            or done % step == 0
                            or (now - last_emit) >= 15.0
                        ):
                            elapsed = now - t0
                            rate = done / elapsed if elapsed > 0 else 0.0
                            eta_s = (nd - done) / rate if rate > 0 and done < nd else 0.0
                            log.info(
                                "Scrape parse_phot: %d / %d (%.1f%%) | elapsed %s | "
                                "ETA ~%s | %.2f sources/s",
                                done,
                                nd,
                                100.0 * done / nd,
                                _eta_hms(elapsed),
                                _eta_hms(eta_s) if done < nd else "0s",
                                rate,
                            )
                            last_emit = now
                if pb is not None:
                    try:
                        pb.finish()
                    except Exception:
                        pass
            if show_progress:
                log.info(
                    "Scrape parse_phot: finished %d source(s) in %s.",
                    nd,
                    _eta_hms(time.perf_counter() - t0),
                )
            final_phot = results
        elif nd >= _SCRAPE_PARSE_PROGRESS_MIN and show_progress:
            log.info(
                "Scrape parse_phot: starting %d source(s) sequentially…",
                nd,
            )
            t0 = time.perf_counter()
            final_phot = []
            step = max(1, nd // 20)
            last_emit = t0
            for i, dat in enumerate(data):
                final_phot.append(_parse_one(dat))
                done = i + 1
                now = time.perf_counter()
                if (
                    done == nd
                    or done % step == 0
                    or (now - last_emit) >= 15.0
                ):
                    elapsed = now - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta_s = (nd - done) / rate if rate > 0 and done < nd else 0.0
                    log.info(
                        "Scrape parse_phot: %d / %d (%.1f%%) | elapsed %s | ETA ~%s | %.2f sources/s",
                        done,
                        nd,
                        100.0 * done / nd,
                        _eta_hms(elapsed),
                        _eta_hms(eta_s) if done < nd else "0s",
                        rate,
                    )
                    last_emit = now
            log.info(
                "Scrape parse_phot: finished %d source(s) in %s.",
                nd,
                _eta_hms(time.perf_counter() - t0),
            )
        else:
            final_phot = [_parse_one(dat) for dat in data]
        p._last_dolphot_catalog_array = catalog
        _scrapedolphot_cleanup(final_phot, note="ok")
        return final_phot
