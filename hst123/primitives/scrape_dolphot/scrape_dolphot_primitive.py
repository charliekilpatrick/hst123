"""Read DOLPHOT catalogs, parse photometry, compute limits, print results. Used by pipeline scrapedolphot."""
from __future__ import annotations

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

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

    def _calc_avg_stats_array(self, obstable, rd, measurement_idx, colfile):
        """Vector-friendly path when the DOLPHOT row is a 1-D float array."""
        p = self._p
        estr = "Magnitude uncertainty"
        cstr = "Measured counts"
        mjds, err, counts, exptime, zpt = [], [], [], [], []
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
                error = self.get_dolphot_data(rd, colfile, estr, img)
                count = self.get_dolphot_data(rd, colfile, cstr, img)
            if error and count:
                mjds.append(Time(row["datetime"]).mjd)
                err.append(error)
                counts.append(count)
                exptime.append(row["exptime"])
                zpt.append(row["zeropoint"])
        if len(mjds) > 0:
            avg_mjd = np.mean(mjds)
            total_exptime = np.sum(exptime)
            mag, magerr = p._phot.avg_magnitudes(err, counts, exptime, zpt)
            return (avg_mjd, mag, magerr, total_exptime)
        return (np.nan, np.nan, np.nan, np.nan)

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

    def parse_phot(
        self,
        obstable,
        row,
        cfile,
        limit_data=None,
        measurement_idx=None,
        columns=None,
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
        for inst in list(set(obstable["instrument"])):
            insttable = obstable[obstable["instrument"] == inst]
            for visit in list(set(insttable["visit"])):
                visittable = insttable[insttable["visit"] == visit]
                for filt in list(set(visittable["filter"])):
                    ftable = visittable[visittable["filter"] == filt]
                    mjd, mag, err, exptime = self.calc_avg_stats(
                        ftable, row, cfile, measurement_idx=measurement_idx
                    )
                    new_row = (mjd, inst, filt, exptime, mag, err, 0)
                    if limit_data:
                        mags, errs = [], []
                        for data in limit_data:
                            mjd, limmag, limerr, exp = self.calc_avg_stats(
                                ftable,
                                data[1],
                                cfile,
                                measurement_idx=measurement_idx,
                            )
                            if (
                                not np.isnan(limmag)
                                and not np.isnan(limerr)
                                and limmag < 99
                            ):
                                mags.append(limmag)
                                errs.append(limerr)
                        if len(mags) > 30:
                            maglimit = p._phot.estimate_mag_limit(
                                mags, errs, limit=p.snr_limit
                            )
                        else:
                            maglimit = np.nan
                        new_row = (
                            mjd,
                            inst,
                            filt,
                            exptime,
                            mag,
                            err,
                            0,
                            maglimit,
                        )
                    final_phot.add_row(new_row)
        for inst in list(set(obstable["instrument"])):
            insttable = obstable[obstable["instrument"] == inst]
            for filt in list(set(insttable["filter"])):
                ftable = insttable[insttable["filter"] == filt]
                mjd, mag, err, exptime = self.calc_avg_stats(
                    ftable, row, cfile, measurement_idx=measurement_idx
                )
                new_row = (mjd, inst, filt, exptime, mag, err, 1)
                if limit_data:
                    mags, errs = [], []
                    for data in limit_data:
                        mjd, limmag, limerr, exp = self.calc_avg_stats(
                            ftable,
                            data[1],
                            cfile,
                            measurement_idx=measurement_idx,
                        )
                        if not np.isnan(limmag) and not np.isnan(limerr):
                            mags.append(limmag)
                            errs.append(limerr)
                    if len(mags) > 30:
                        maglimit = p._phot.estimate_mag_limit(
                            mags, errs, limit=p.snr_limit
                        )
                    else:
                        maglimit = np.nan
                    new_row = (
                        mjd,
                        inst,
                        filt,
                        exptime,
                        mag,
                        err,
                        1,
                        maglimit,
                    )
                final_phot.add_row(new_row)
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

        def _parse_one(dat):
            finished = False
            limit_radius = dolphot.get("limit_radius", 10.0)
            limit_data_cur = limit_data
            while not finished:
                source_phot = self.parse_phot(
                    obstable,
                    dat["data"],
                    colfile,
                    limit_data=limit_data_cur,
                    measurement_idx=measurement_idx,
                    columns=columns,
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
        n_workers = min(mc, len(data)) if len(data) > 1 else 1
        if n_workers > 1 and len(data) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                final_phot = list(ex.map(_parse_one, data))
        else:
            final_phot = [_parse_one(dat) for dat in data]
        _scrapedolphot_cleanup(final_phot, note="ok")
        return final_phot
