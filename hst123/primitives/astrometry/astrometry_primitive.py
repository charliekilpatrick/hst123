"""
Astrometry primitive: TweakReg-based image alignment and WCS registration.

Holds all logic for preparing references for tweakreg, checking/setting
TWEAKSUC, threshold estimation, running TweakReg, and copying WCS keys.
Also provides parse_coord helper for RA/Dec parsing.
Depends on BasePrimitive and the pipeline for get_instrument, get_filter,
update_image_wcs, run_cosmic, pick_deepest_images, input_list, sanitize_reference.
"""

import copy
import glob
import logging
import os
import random
import shutil
import time

import numpy as np

log = logging.getLogger(__name__)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, Column
from scipy.interpolate import interp1d

from hst123 import settings
from hst123.primitives.base import BasePrimitive


def _is_number(num):
    """Return True if num can be interpreted as a number."""
    try:
        float(num)
        return True
    except (ValueError, TypeError):
        return False


def parse_coord(ra, dec):
    """
    Parse RA and Dec (degrees or sexagesimal) into an ICRS SkyCoord.

    Parameters
    ----------
    ra, dec : str or float
        Right ascension and declination. Use sexagesimal (e.g. "12:00:00", "+00:00:00")
        or numeric degrees.

    Returns
    -------
    SkyCoord or None
        ICRS coordinate, or None if parsing fails.
    """
    if not (_is_number(ra) and _is_number(dec)) and (
        ":" not in str(ra) and ":" not in str(dec)
    ):
        log.error("cannot interpret: %s %s", ra, dec)
        return None

    if ":" in str(ra) and ":" in str(dec):
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        return SkyCoord(ra, dec, frame="icrs", unit=unit)
    except ValueError:
        log.error("Cannot parse coordinates: %s %s", ra, dec)
        return None

# Optional heavy imports (same as pipeline)
try:
    from drizzlepac import tweakreg, catalogs
    import stwcs
except ImportError:
    tweakreg = None
    catalogs = None
    stwcs = None


class AstrometryPrimitive(BasePrimitive):
    """Image alignment and astrometry (TweakReg, WCS)."""

    def prepare_reference_tweakreg(self, reference):
        """Prepare reference image for tweakreg. Requires specific HDU layout."""
        if not os.path.exists(reference):
            log.error("tried to sanitize non-existence ref %s", reference)
            return False

        hdu = fits.open(reference)
        data = [h._summary() for h in hdu]

        if len(data) == 1:
            newhdu = fits.HDUList()
            newhdu.append(hdu[0])
            newhdu.append(hdu[0])
            newhdu[0].data = None
            newhdu[0].header["EXTVER"] = 1
            newhdu[1].header["EXTVER"] = 1
            newhdu[0].header["EXTNAME"] = "PRIMARY"
            newhdu[1].header["EXTNAME"] = "SCI"
            newhdu.writeto(reference, output_verify="silentfix", overwrite=True)
            return True

        idxIm = [i for i, d in enumerate(data) if d[2].strip() == "ImageHDU"]
        idxPr = [i for i, d in enumerate(data) if d[0].strip().upper() == "PRIMARY"]

        if len(idxIm) == 0:
            return False

        newhdu = fits.HDUList()
        newhdu.append(hdu[np.min(idxIm)])
        newhdu.append(hdu[np.min(idxIm)])
        newhdu[0].data = None

        if len(idxPr) > 0:
            primary = hdu[np.min(idxPr)]
            for n, key in enumerate(primary.header.keys()):
                if not key.strip():
                    continue
                if isinstance(hdu[0].header[key], str) and "\n" in hdu[0].header[key]:
                    continue
                if key == "FILETYPE":
                    newhdu[0].header[key] = "SCI"
                elif key == "FILENAME":
                    newhdu[0].header[key] = reference
                elif key == "EXTEND":
                    newhdu[0].header[key] = True
                else:
                    val = hdu[0].header[key]
                    if isinstance(val, str):
                        val = val.strip().replace("\n", " ")
                    newhdu[0].header[key] = val

        newhdu[0].header["FILENAME"] = reference
        newhdu[1].header["FILENAME"] = reference
        newhdu[0].header["EXTVER"] = 1
        newhdu[1].header["EXTVER"] = 1
        newhdu[0].header["EXTNAME"] = "PRIMARY"
        newhdu[1].header["EXTNAME"] = "SCI"

        for j in (0, 1):
            if "SANITIZE" in newhdu[j].header.keys():
                del newhdu[j].header["SANITIZE"]

        if os.path.exists(reference.replace(".fits", ".sky.fits")):
            os.remove(reference.replace(".fits", ".sky.fits"))

        newhdu.writeto(reference, output_verify="silentfix", overwrite=True)
        return True

    def check_images_for_tweakreg(self, run_images):
        """Return list of images that do not yet have TWEAKSUC=1."""
        if not run_images:
            return None

        images = copy.copy(run_images)
        for file in list(images):
            log.info("Checking %s for TWEAKSUC=1", file)
            hdu = fits.open(file, mode="readonly")
            remove_image = (
                "TWEAKSUC" in hdu[0].header.keys() and hdu[0].header["TWEAKSUC"] == 1
            )
            if remove_image:
                images.remove(file)

        if len(images) == 0:
            return None
        return images

    def get_nsources(self, image, thresh):
        """Return number of sources detected in image at given threshold."""
        imghdu = fits.open(image)
        nsources = 0
        log.info("Getting number of sources in %s at threshold=%s", image, thresh)
        for i, h in enumerate(imghdu):
            if h.name == "SCI" or (len(imghdu) == 1 and h.name == "PRIMARY"):
                filename = "{:s}[{:d}]".format(image, i)
                wcs = stwcs.wcsutil.HSTWCS(filename)
                catalog_mode = "automatic"
                catalog = catalogs.generateCatalog(
                    wcs,
                    mode=catalog_mode,
                    catalog=filename,
                    threshold=thresh,
                    **self._p.options["catalog"],
                )
                try:
                    catalog.buildCatalogs()
                    nsources += catalog.num_objects
                except Exception:
                    pass

        log.info("Got %s total sources", nsources)
        return nsources

    def count_nsources(self, images):
        """Count catalog sources from coo files (tagged with threshold)."""
        cat_str = "_sci*_xy_catalog.coo"
        n = 0
        for im in images:
            for catalog in glob.glob(im.replace(".fits", cat_str)):
                with open(catalog, "r+") as f:
                    for line in f:
                        if "threshold" not in line:
                            n += 1
        return n

    def get_tweakreg_thresholds(self, image, target):
        """Estimate threshold vs. source count for image to reach target nobj."""
        log.info("Getting tweakreg threshold for %s.  Target nobj=%s", image, target)
        inp_data = []
        for t in np.flip([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 80.0]):
            nobj = self.get_nsources(image, t)
            if len(inp_data) < 3:
                inp_data.append((float(nobj), float(t)))
            elif nobj < inp_data[-1][0]:
                break
            else:
                inp_data.append((float(nobj), float(t)))
                if nobj > target:
                    break
        return inp_data

    def add_thresh_data(self, thresh_data, image, inp_data):
        """Append threshold/source-count row to thresh_data table."""
        if not thresh_data:
            keys = []
            data = []
            for val in inp_data:
                keys.append("%2.1f" % float(val[1]))
                data.append([val[0]])
            keys.insert(0, "file")
            data.insert(0, [image])
            thresh_data = Table(data, names=keys)
            return thresh_data

        keys = []
        data = []
        for val in inp_data:
            key = "%2.1f" % float(val[1])
            keys.append(key)
            data.append(float(val[0]))
            if key not in thresh_data.keys():
                thresh_data.add_column(Column([np.nan] * len(thresh_data), name=key))
        keys.insert(0, "file")
        data.insert(0, image)
        thresh_data = Table(thresh_data)
        for key in thresh_data.keys():
            if key not in keys:
                data.append(np.nan)
        thresh_data.add_row(data)
        return thresh_data

    def get_best_tweakreg_threshold(self, thresh_data, target):
        """Interpolate threshold for target source count; clamp to settings."""
        thresh = []
        nsources = []
        thresh_data = Table(thresh_data)
        for key in thresh_data.keys():
            if key == "file":
                continue
            thresh.append(float(key))
            nsources.append(float(thresh_data[key]))

        thresh = np.array(thresh)
        nsources = np.array(nsources)
        mask = (~np.isnan(thresh)) & (~np.isnan(nsources))
        thresh = thresh[mask]
        nsources = nsources[mask]

        thresh_func = interp1d(
            nsources, thresh, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        threshold = thresh_func(target)

        trd = settings.tweakreg_defaults
        if threshold < trd["threshold_min"]:
            threshold = trd["threshold_min"]
        if threshold > trd["threshold_max"]:
            threshold = trd["threshold_max"]

        log.info("Using threshold: %s", threshold)
        return threshold

    def get_shallow_param(self, image):
        """Return (filter, pivot_wavelength, exptime) for shallow-image checks."""
        filt = self._p.get_filter(image)
        hdu = fits.open(image)
        pivot = 0.0
        for h in hdu:
            if "PHOTPLAM" in h.header.keys():
                pivot = float(h.header["PHOTPLAM"])
                break
        exptime = 0.0
        for h in hdu:
            if "EXPTIME" in h.header.keys():
                exptime = float(h.header["EXPTIME"])
                break
        return (filt, pivot, exptime)

    def tweakreg_error(self, exception):
        """Log tweakreg failure banner."""
        log.warning(
            "tweakreg failed: %s\n%s\nAdjusting thresholds and images...",
            exception.__class__.__name__,
            exception,
        )

    def apply_tweakreg_success(self, shifts):
        """Set TWEAKSUC=1 in header for files with valid shifts."""
        for row in shifts:
            if ~np.isnan(row["xoffset"]) and ~np.isnan(row["yoffset"]):
                file = row["file"]
                if not os.path.exists(file):
                    log.warning("%s does not exist", file)
                    continue
                hdu = fits.open(file, mode="update")
                hdu[0].header["TWEAKSUC"] = 1
                hdu.close()

    def copy_wcs_keys(self, from_hdu, to_hdu):
        """Copy WCS header keys from one HDU to another."""
        for key in [
            "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "CTYPE1", "CTYPE2",
        ]:
            if key in from_hdu.header.keys():
                to_hdu.header[key] = from_hdu.header[key]

    def run_tweakreg(
        self,
        obstable,
        reference,
        do_cosmic=True,
        skip_wcs=False,
        search_radius=None,
        update_hdr=True,
    ):
        """Run TweakReg on images in obstable; return (success, shift_table)."""
        p = self._p
        if p.options["args"].work_dir:
            outdir = p.options["args"].work_dir
        else:
            outdir = "."

        os.chdir(outdir)
        options = p.options["global_defaults"]

        run_images = self.check_images_for_tweakreg(list(obstable["image"]))
        if not run_images:
            return ("tweakreg success", None)
        if reference in run_images:
            run_images.remove(reference)

        shift_table = Table(
            [run_images, [np.nan] * len(run_images), [np.nan] * len(run_images)],
            names=("file", "xoffset", "yoffset"),
        )

        if not run_images:
            log.warning("All images have been run through tweakreg.")
            return (True, shift_table)

        log.info("Need to run tweakreg for images:")
        p.input_list(obstable["image"], show=True, save=False)

        tmp_images = []
        for image in run_images:
            if p.updatewcs and not skip_wcs:
                det = "_".join(p.get_instrument(image).split("_")[:2])
                wcsoptions = p.options["detector_defaults"][det]
                p.update_image_wcs(image, wcsoptions)

            if not do_cosmic:
                tmp_images.append(image)
                continue

            if image == reference or "wfc3_ir" in p.get_instrument(image):
                log.info("Skipping adjustments for %s as WFC3/IR or reference", image)
                tmp_images.append(image)
                continue

            rawtmp = image.replace(".fits", ".rawtmp.fits")
            tmp_images.append(rawtmp)
            if os.path.exists(rawtmp):
                log.info("%s exists. Skipping...", rawtmp)
                continue

            shutil.copyfile(image, rawtmp)
            inst = p.get_instrument(image).split("_")[0]
            crpars = p.options["instrument_defaults"][inst]["crpars"]
            p.run_cosmic(rawtmp, crpars)

        modified = False
        ref_images = p.pick_deepest_images(tmp_images)
        deepest = sorted(ref_images, key=lambda im: fits.getval(im, "EXPTIME"))[-1]

        if not reference or reference == "dummy.fits":
            reference = "dummy.fits"
            log.info("Copying %s to reference dummy.fits", deepest)
            shutil.copyfile(deepest, reference)
        elif not self.prepare_reference_tweakreg(reference):
            reference = "dummy.fits"
            log.info("Copying %s to reference dummy.fits", deepest)
            shutil.copyfile(deepest, reference)
        else:
            modified = True

        log.info("Tweakreg is executing...")
        start_tweak = time.time()

        tweakreg_success = False
        tweak_img = copy.copy(tmp_images)
        ithresh = p.threshold
        rthresh = p.threshold
        shallow_img = []
        thresh_data = None
        tries = 0

        while not tweakreg_success and tries < 10:
            tweak_img = self.check_images_for_tweakreg(tweak_img)
            if not tweak_img:
                break

            if shallow_img:
                for img in shallow_img:
                    if img in tweak_img:
                        tweak_img.remove(img)

            if len(tweak_img) == 0:
                log.error("removed all images as shallow")
                tweak_img = copy.copy(tmp_images)
                tweak_img = self.check_images_for_tweakreg(tweak_img)

            success = list(set(tmp_images) ^ set(tweak_img))
            if tries > 1 and reference == "dummy.fits" and len(success) > 0:
                n = len(success) - 1
                shutil.copyfile(success[random.randint(0, n)], "dummy.fits")

            log.info("Reference image: %s  Images: %s", reference, ",".join(tweak_img))

            deepest = sorted(tweak_img, key=lambda im: fits.getval(im, "EXPTIME"))[-1]

            if not thresh_data or deepest not in thresh_data["file"]:
                inp_data = self.get_tweakreg_thresholds(deepest, options["nbright"] * 4)
                thresh_data = self.add_thresh_data(thresh_data, deepest, inp_data)
            mask = thresh_data["file"] == deepest
            inp_thresh = thresh_data[mask][0]
            log.info("Getting image threshold...")
            new_ithresh = self.get_best_tweakreg_threshold(inp_thresh, options["nbright"] * 4)

            if not thresh_data or reference not in thresh_data["file"]:
                inp_data = self.get_tweakreg_thresholds(reference, options["nbright"] * 4)
                thresh_data = self.add_thresh_data(thresh_data, reference, inp_data)
            mask = thresh_data["file"] == reference
            inp_thresh = thresh_data[mask][0]
            log.info("Getting reference threshold...")
            new_rthresh = self.get_best_tweakreg_threshold(inp_thresh, options["nbright"] * 4)

            if not rthresh:
                rthresh = p.threshold
            if not ithresh:
                ithresh = p.threshold

            nbright = options["nbright"]
            minobj = options["minobj"]
            search_rad = int(np.round(options["search_rad"]))
            if search_radius:
                search_rad = search_radius

            trd = settings.tweakreg_defaults
            rconv = trd["conv_width"]
            iconv = trd["conv_width"]
            tol = trd["tolerance"]
            for detkey, overrides in trd["detector_overrides"].items():
                if detkey in p.get_instrument(reference):
                    rconv = overrides["conv_width"]
                    break
            for detkey, overrides in trd["detector_overrides"].items():
                if all(detkey in p.get_instrument(i) for i in tweak_img):
                    iconv = overrides["conv_width"]
                    tol = overrides["tolerance"]
                    break

            if (new_ithresh >= ithresh or new_rthresh >= rthresh) and tries > 1:
                log.info("Decreasing threshold and increasing tolerance...")
                ithresh = np.max([new_ithresh * (0.95 ** tries), trd["threshold_min"]])
                rthresh = np.max([new_rthresh * (0.95 ** tries), trd["threshold_min"]])
                tol = tol * 1.3 ** tries
                search_rad = search_rad * 1.2 ** tries
            else:
                ithresh = new_ithresh
                rthresh = new_rthresh

            if tries > 7:
                minobj = trd["minobj_fallback"]

            log.info(
                "Adjusting thresholds: Reference threshold=%s Image threshold=%s Tolerance=%s Search radius=%s",
                "%2.4f" % rthresh,
                "%2.4f" % ithresh,
                "%2.4f" % tol,
                "%2.4f" % search_rad,
            )

            outshifts = os.path.join(outdir, "drizzle_shifts.txt")

            try:
                tweakreg.TweakReg(
                    files=tweak_img,
                    refimage=reference,
                    verbose=False,
                    interactive=False,
                    clean=True,
                    writecat=True,
                    updatehdr=update_hdr,
                    reusename=True,
                    rfluxunits="counts",
                    minobj=minobj,
                    wcsname="TWEAK",
                    searchrad=search_rad,
                    searchunits="arcseconds",
                    runfile="",
                    tolerance=tol,
                    refnbright=nbright,
                    nbright=nbright,
                    separation=trd["separation"],
                    residplot="No plot",
                    see2dplot=False,
                    fitgeometry="shift",
                    imagefindcfg={"threshold": ithresh, "conv_width": iconv, "use_sharp_round": True},
                    refimagefindcfg={"threshold": rthresh, "conv_width": rconv, "use_sharp_round": True},
                    shiftfile=True,
                    outshifts=outshifts,
                )
                shallow_img = []

            except AssertionError as e:
                self.tweakreg_error(e)
                log.info("Re-running tweakreg with shallow images removed:")
                for img in tweak_img:
                    nsources = self.get_nsources(img, ithresh)
                    if nsources < 1000:
                        shallow_img.append(img)

            except TypeError as e:
                self.tweakreg_error(e)

            log.info("Reading in shift file: %s", outshifts)
            shifts = Table.read(
                outshifts,
                format="ascii",
                names=("file", "xoffset", "yoffset", "rotation1", "rotation2", "scale1", "scale2"),
            )

            self.apply_tweakreg_success(shifts)

            for row in shifts:
                filename = os.path.basename(row["file"])
                filename = filename.replace(".rawtmp.fits", "").replace(".fits", "")
                idx = [i for i, r in enumerate(shift_table) if filename in r["file"]]
                if len(idx) == 1:
                    shift_table[idx[0]]["xoffset"] = row["xoffset"]
                    shift_table[idx[0]]["yoffset"] = row["yoffset"]

            if not self.check_images_for_tweakreg(tmp_images):
                tweakreg_success = True

            tries += 1

        log.info("Tweakreg took %s seconds to execute.", time.time() - start_tweak)
        log.info("Shift table: %s", shift_table)

        # Fix CRVAL/CRPIX indexing after tweakreg
        for image in tmp_images:
            rawtmp = image
            rawhdu = fits.open(rawtmp, mode="readonly")
            tweaksuc = (
                "TWEAKSUC" in rawhdu[0].header.keys() and rawhdu[0].header["TWEAKSUC"] == 1
            )
            if "wfc3_ir" in p.get_instrument(image):
                continue
            for i, h in enumerate(rawhdu):
                if (
                    tweaksuc
                    and "CRVAL1" in h.header.keys()
                    and "CRVAL2" in h.header.keys()
                ):
                    rawhdu[i].header["CRPIX1"] = rawhdu[i].header["CRPIX1"] - 0.5
                    rawhdu[i].header["CRPIX2"] = rawhdu[i].header["CRPIX2"] - 0.5
            rawhdu.writeto(rawtmp, overwrite=True)

        if not skip_wcs:
            for image in run_images:
                if image == reference or "wfc3_ir" in p.get_instrument(image):
                    continue
                log.info("Updating image data for image: %s", image)
                rawtmp = image.replace(".fits", ".rawtmp.fits")
                rawhdu = fits.open(rawtmp, mode="readonly")
                hdu = fits.open(image, mode="readonly")
                newhdu = fits.HDUList()

                log.info("Current image info:")
                hdu.info()

                for i, h in enumerate(hdu):
                    if h.name == "SCI":
                        if "flc" in image or "flt" in image:
                            if len(rawhdu) >= i + 2 and rawhdu[i + 2].name == "DQ":
                                self.copy_wcs_keys(rawhdu[i], rawhdu[i + 2])
                        elif "c0m" in image:
                            maskfile = image.split("_")[0] + "_c1m.fits"
                            if os.path.exists(maskfile):
                                maskhdu = fits.open(maskfile)
                                self.copy_wcs_keys(rawhdu[i], maskhdu[i])
                                maskhdu.writeto(maskfile, overwrite=True)

                    if "wfpc2" in p.get_instrument(image).lower() and h.name == "WCSCORR":
                        continue

                    ver = int(h.ver)
                    name = str(h.name).strip()
                    idx = -1
                    for j, rawh in enumerate(rawhdu):
                        if str(rawh.name).strip() == name and int(rawh.ver) == ver:
                            idx = j
                    if idx < 0:
                        log.debug("Skip extension %s,%s,%s - no match in %s", i, name, ver, rawtmp)
                        continue

                    if h.name != "DQ":
                        if "data" in dir(h) and "data" in dir(rawhdu[idx]):
                            if rawhdu[idx].data is not None and h.data is not None:
                                if rawhdu[idx].data.dtype == h.data.dtype:
                                    rawhdu[idx].data = h.data

                    log.debug("Copy extension %s,%s,%s", idx, name, ver)
                    newhdu.append(copy.copy(rawhdu[idx]))

                if "wfpc2" in p.get_instrument(image).lower():
                    newhdu[0].header["NEXTEND"] = 4

                log.info("New image info:")
                newhdu.info()
                newhdu.writeto(image, output_verify="silentfix", overwrite=True)

                if os.path.isfile(rawtmp) and not p.options["args"].cleanup:
                    os.remove(rawtmp)

        if os.path.isfile("dummy.fits"):
            os.remove("dummy.fits")

        if not p.options["args"].keep_objfile:
            for file in glob.glob("*.coo"):
                os.remove(file)

        if modified:
            p.sanitize_reference(reference)

        return (tweakreg_success, shift_table)
