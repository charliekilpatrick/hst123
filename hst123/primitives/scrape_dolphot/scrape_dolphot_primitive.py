"""Read DOLPHOT catalogs, parse photometry, compute limits, print results. Used by pipeline scrapedolphot."""
import logging
import os
import shutil

import numpy as np
import progressbar
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord

from hst123.primitives.base import BasePrimitive
from hst123.utils.display import show_photometry as display_show_photometry

log = logging.getLogger(__name__)


class ScrapeDolphotPrimitive(BasePrimitive):
    """
    Read dolphot catalogs, parse photometry, and print final photometry.

    Provides get_dolphot_column, get_dolphot_data, get_limit_data, calc_avg_stats,
    parse_phot, print_final_phot, and scrapedolphot for pipeline coord.
    """

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
        coldata = ""
        with open(colfile) as colfile_data:
            for line in colfile_data:
                if image.replace(".fits", "") in line and key in line:
                    coldata = line.strip().strip("\n")
                    break
        if not coldata:
            return None
        try:
            colnum = int(coldata.split(".")[0].strip()) - 1 + offset
            return colnum
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
        rdata = row.split() if isinstance(row, str) else row
        if colnum is not None:
            if colnum < 0 or colnum > len(rdata) - 1:
                log.error(
                    "ERROR: tried to use bad column %s in dolphot output",
                    colnum,
                )
                return None
            return rdata[colnum]
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
        for i, phot in enumerate(final_phot):
            outfile = dolphot["final_phot"]
            if not allphot:
                out = outfile
            else:
                num = str(i).zfill(len(str(len(final_phot))))
                out = outfile.replace(".phot", "_" + num + ".phot")
            snana = out.replace(".phot", ".snana")
            message = "Photometry for source {n} ".format(n=i)
            keys = phot.meta.keys()
            if "x" in keys and "y" in keys and "separation" in keys:
                message += "at x,y={x},{y}.\nSeparated from input coordinate by {sep} pix."
                message = message.format(
                    x=phot.meta["x"],
                    y=phot.meta["y"],
                    sep=phot.meta["separation"],
                )
            log.info(message)
            with open(out, "w") as f:
                display_show_photometry(phot, f=f, coord=p.coord, options=p.options, log=log)
            with open(snana, "w") as f:
                display_show_photometry(phot, f=f, snana=True, show=False, coord=p.coord, options=p.options, log=log)
            log.info("")

    def get_limit_data(self, dolphot, coord, w, x, y, colfile, limit_radius):
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

        Returns
        -------
        list of list
            Each element [dist, line] for sources within radius.
        """
        limit_data = []
        if coord.dec.degree < 89:
            dec1 = coord.dec.degree + limit_radius / 3600.0
        else:
            dec1 = coord.dec.degree - limit_radius / 3600.0
        coord1 = SkyCoord(coord.ra.degree, dec1, unit="deg")
        x1, y1 = wcs.utils.skycoord_to_pixel(coord1, w, origin=1)
        radius = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        with open(dolphot["base"]) as dp:
            for line in dp:
                xcol = self.get_dolphot_column(colfile, "Object X", "")
                ycol = self.get_dolphot_column(colfile, "Object Y", "")
                xline = float(line.split()[xcol]) + 0.5
                yline = float(line.split()[ycol]) + 0.5
                dist = np.sqrt((xline - x) ** 2 + (yline - y) ** 2)
                if dist < radius:
                    limit_data.append([dist, line])
        return limit_data

    def calc_avg_stats(self, obstable, data, colfile):
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

        Returns
        -------
        tuple
            (avg_mjd, mag, magerr, total_exptime) or (nan, nan, nan, nan) if no data.
        """
        p = self._p
        estr = "Magnitude uncertainty"
        cstr = "Measured counts"
        mjds, err, counts, exptime, filts, det, zpt = [], [], [], [], [], [], []
        for row in obstable:
            error = self.get_dolphot_data(data, colfile, estr, row["image"])
            count = self.get_dolphot_data(data, colfile, cstr, row["image"])
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

    def parse_phot(self, obstable, row, cfile, limit_data=None):
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

        Returns
        -------
        astropy.table.Table
            Table with MJD, INSTRUMENT, FILTER, EXPTIME, MAGNITUDE, MAGNITUDE_ERROR, IS_AVG, [LIMIT].
        """
        p = self._p
        limit_data = limit_data or []
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
                        ftable, row, cfile
                    )
                    new_row = (mjd, inst, filt, exptime, mag, err, 0)
                    if limit_data:
                        mags, errs = [], []
                        for data in limit_data:
                            mjd, limmag, limerr, exp = self.calc_avg_stats(
                                ftable, data[1], cfile
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
                    ftable, row, cfile
                )
                new_row = (mjd, inst, filt, exptime, mag, err, 1)
                if limit_data:
                    mags, errs = [], []
                    for data in limit_data:
                        mjd, limmag, limerr, exp = self.calc_avg_stats(
                            ftable, data[1], cfile
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
        self, coord, reference, images, dolphot, scrapeall=False,
        get_limits=False, brightest=False
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

        Returns
        -------
        list of astropy.table.Table or None
            One table per source with meta (x, y, separation, etc.); None if no sources or error.
        """
        import filecmp

        p = self._p
        base = dolphot["base"]
        colfile = dolphot["colfile"]
        if not os.path.isfile(base) or not os.path.isfile(colfile):
            log.error(
                "ERROR: dolphot output %s does not exist. Use --run-dolphot "
                "or check your dolphot output for errors",
                dolphot["base"],
            )
            return None
        if not reference or not coord:
            log.error(
                "ERROR: Need a reference image and coordinate to "
                "scrape data from the dolphot catalog. Exiting..."
            )
            return None
        if not os.path.exists(dolphot["original"]):
            shutil.copyfile(dolphot["base"], dolphot["original"])
        if not p.options["args"].no_cuts:
            log.info("Cutting bad sources from dolphot catalog.")
            f = open("tmp", "w")
            numlines = sum(1 for _ in open(base))
            log.info(
                "There are %s sources in dolphot file %s. Cutting bad sources...",
                numlines,
                dolphot["base"],
            )
            bar = progressbar.ProgressBar(maxval=numlines).start()
            typecol = self.get_dolphot_column(colfile, "Object type", "")
            with open(dolphot["base"]) as dolphot_file:
                for i, line in enumerate(dolphot_file):
                    bar.update(i)
                    if int(line.split()[typecol]) == 1:
                        f.write(line)
            bar.finish()
            f.close()
            log.info("Done cutting bad sources")
            if filecmp.cmp(dolphot["base"], "tmp"):
                log.info("No changes to dolphot file %s.", dolphot["base"])
                os.remove("tmp")
            else:
                log.info("Updating dolphot file %s.", dolphot["base"])
                shutil.move("tmp", dolphot["base"])
        else:
            if os.path.exists(dolphot["original"]):
                shutil.copyfile(dolphot["original"], dolphot["base"])
        hdu = fits.open(reference)
        w = wcs.WCS(hdu[0].header)
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
        log.info(
            "Looking for a source around x=%s, y=%s in %s with a radius of %s",
            "%7.2f" % float(x),
            "%7.2f" % float(y),
            reference,
            "%7.4f" % float(radius),
        )
        data = []
        with open(dolphot["base"]) as dp:
            for line in dp:
                xcol = self.get_dolphot_column(colfile, "Object X", "")
                ycol = self.get_dolphot_column(colfile, "Object Y", "")
                xline = float(line.split()[xcol]) + 0.5
                yline = float(line.split()[ycol]) + 0.5
                dist = np.sqrt((xline - x) ** 2 + (yline - y) ** 2)
                sn = self.get_dolphot_data(line, colfile, "Signal-to-noise", "")
                counts = self.get_dolphot_data(
                    line, colfile, "Normalized count rate", ""
                )
                if dist < radius:
                    data.append(
                        {
                            "sep": dist,
                            "data": line,
                            "sn": float(sn),
                            "counts": float(counts),
                        }
                    )
        limit_data = []
        if get_limits:
            limit_radius = dolphot["limit_radius"]
            limit_data = self.get_limit_data(
                dolphot, coord, w, x, y, colfile, limit_radius
            )
        log.info(
            "Done looking for sources in dolphot file %s. "
            "hst123 found %s sources around: %s %s",
            dolphot["base"],
            len(data),
            ra,
            dec,
        )
        if len(data) == 0:
            return None
        if brightest:
            data = sorted(data, key=lambda obj: obj["counts"], reverse=True)
            log.warning(
                "WARNING: found more than one source. Picking brightest object"
            )
        else:
            data = sorted(data, key=lambda obj: obj["sep"])
            log.warning(
                "WARNING: found more than one source. Picking closest to %s %s",
                ra,
                dec,
            )
        if not scrapeall:
            data = [data[0]]
            log.info(
                "Separation=%s, Signal-to-noise=%s",
                data[0]["sep"],
                data[0]["sn"],
            )
        obstable = p.input_list(images, show=False, save=False)
        if not obstable:
            return None
        final_phot = []
        for dat in data:
            finished = False
            limit_radius = dolphot.get("limit_radius", 10.0)
            limit_data_cur = limit_data
            while not finished:
                source_phot = self.parse_phot(
                    obstable, dat["data"], colfile, limit_data=limit_data_cur
                )
                finished = True
                if "LIMIT" in source_phot.keys():
                    if any(
                        np.isnan(row["LIMIT"]) for row in source_phot
                    ):
                        if limit_radius < 80:
                            limit_radius = 2 * limit_radius
                            finished = False
                            limit_data_cur = self.get_limit_data(
                                dolphot, coord, w, x, y, colfile, limit_radius
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
            final_phot.append(source_phot)
        return final_phot
