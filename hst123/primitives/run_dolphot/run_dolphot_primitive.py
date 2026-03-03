"""Run DOLPHOT, prepare images (calcsky, splitgroups, *mask), param files, fake stars. Scraping is in ScrapeDolphotPrimitive."""
import glob
import logging
import os
import shutil
import time

import numpy as np
from astropy.io import fits
import astropy.wcs as wcs

from hst123.primitives.base import BasePrimitive

log = logging.getLogger(__name__)

DOLPHOT_REQUIRED_SCRIPTS = [
    "dolphot",
    "calcsky",
    "acsmask",
    "wfc3mask",
    "wfpc2mask",
    "splitgroups",
]


class DolphotPrimitive(BasePrimitive):
    """Run DOLPHOT, prepare images, param files, fake stars; scraping is in ScrapeDolphotPrimitive."""

    def check_for_dolphot(self):
        """Return True if all DOLPHOT_REQUIRED_SCRIPTS are on PATH."""
        for s in DOLPHOT_REQUIRED_SCRIPTS:
            if not shutil.which(s):
                return False
        return True

    def make_dolphot_dict(self, dolphot):
        """
        Build dict of dolphot file paths and parameters (param, log, base, colfile, etc.).

        Parameters
        ----------
        dolphot : str
            Base name for dolphot output (e.g. "dp" -> dp.param, dp.output, dp.phot).

        Returns
        -------
        dict
            Keys: base, param, log, total_objs, colfile, fake, fakelist, fakelog,
            radius, final_phot, limit_radius, original.
        """
        return {
            "base": dolphot,
            "param": dolphot + ".param",
            "log": dolphot + ".output",
            "total_objs": 0,
            "colfile": dolphot + ".columns",
            "fake": dolphot + ".fake",
            "fakelist": dolphot + ".fakelist",
            "fakelog": dolphot + ".fake.output",
            "radius": 12,
            "final_phot": dolphot + ".phot",
            "limit_radius": 10.0,
            "original": dolphot + ".orig",
        }

    def needs_to_calc_sky(self, image, check_wcs=False):
        """
        Return True if image needs calcsky (no .sky.fits or WCS mismatch when check_wcs=True).

        Parameters
        ----------
        image : str
            Path to science FITS file.
        check_wcs : bool, optional
            If True, compare image and sky file WCS; return False if mismatch. Default False.

        Returns
        -------
        bool
            True if calcsky should be run for this image.
        """
        p = self._p
        log.info("Checking for %s", image.replace(".fits", ".sky.fits"))
        files = glob.glob(image.replace(".fits", ".sky.fits"))
        if len(files) == 0:
            return True
        if check_wcs:
            imhdu = fits.open(image)
            skhdu = fits.open(files[0])
            if len(imhdu) != len(skhdu):
                return False
            check_keys = [
                "CRVAL1",
                "CRVAL2",
                "CRPIX1",
                "CRPIX2",
                "CD1_1",
                "CD1_2",
                "CD2_1",
                "CD2_2",
                "NAXIS1",
                "NAXIS2",
            ]
            for imh, skh in zip(imhdu, skhdu):
                for key in check_keys:
                    if key in list(imh.header.keys()):
                        if key not in list(skh.header.keys()):
                            return False
                        if imh.header[key] != skh.header[key]:
                            return False
            return True
        return False

    def needs_to_split_groups(self, image):
        """Return True if split chip files are missing (count SCI extensions vs .chip?.fits)."""
        hdu = fits.open(image, mode="readonly")
        total = sum(1 for h in hdu if h.name == "SCI")
        hdu.close()
        return len(glob.glob(image.replace(".fits", ".chip?.fits"))) != total

    def needs_to_be_masked(self, image):
        """
        Return True if image should be masked (DOL* header not set to 0 for this instrument).

        Parameters
        ----------
        image : str
            Path to FITS image.

        Returns
        -------
        bool
            True if *mask should be run (WFPC2, WFC3, ACS).
        """
        p = self._p
        hdulist = fits.open(image)
        header = hdulist[0].header
        inst = p._fits.get_instrument(image).split("_")[0].upper()
        if inst == "WFPC2":
            if "DOLWFPC2" in header.keys() and header["DOLWFPC2"] == 0:
                return False
        if inst == "WFC3":
            if "DOL_WFC3" in header.keys() and header["DOL_WFC3"] == 0:
                return False
        if inst == "ACS":
            if "DOL_ACS" in header.keys() and header["DOL_ACS"] == 0:
                return False
        return True

    def split_groups(self, image, delete_non_science=True):
        """
        Run splitgroups on image; optionally remove non-science chip files.

        Parameters
        ----------
        image : str
            Path to FITS file.
        delete_non_science : bool, optional
            If True, remove split files that are not science extensions. Default True.
        """
        log.info("Running split groups for %s", image)
        os.system(f"splitgroups {image}")
        if delete_non_science:
            split_images = glob.glob(
                image.replace(".fits", ".chip*.fits")
            )
            for split in split_images:
                hdu = fits.open(split)
                info = hdu[0]._summary()
                if info[0].upper() != "SCI":
                    log.warning(
                        "WARNING: deleting %s, not a science extension.",
                        split,
                    )
                    os.remove(split)

    def mask_image(self, image, instrument):
        """
        Run instrument-specific mask (acsmask, wfc3mask, wfpc2mask) for dolphot DQ.

        Parameters
        ----------
        image : str
            Path to science FITS file.
        instrument : str
            Instrument name (e.g. "acs", "wfc3", "wfpc2") for *mask command.
        """
        p = self._p
        maskimage = p.get_dq_image(image)
        cmd = f"{instrument}mask {image} {maskimage}"
        log.info("Executing: %s", cmd)
        os.system(cmd)

    def calc_sky(self, image, options):
        """
        Run calcsky for image using detector options (r_in, r_out, step, sigma_low, sigma_high).

        Parameters
        ----------
        image : str
            Path to science FITS (without .fits for calcsky command).
        options : dict
            detector_defaults dict with [det]["dolphot_sky"] keys.
        """
        p = self._p
        det = "_".join(p._fits.get_instrument(image).split("_")[:2])
        opt = options[det]["dolphot_sky"]
        cmd = "calcsky {image} {rin} {rout} {step} {sigma_low} {sigma_high}"
        calc_sky_cmd = cmd.format(
            image=image.replace(".fits", ""),
            rin=opt["r_in"],
            rout=opt["r_out"],
            step=opt["step"],
            sigma_low=opt["sigma_low"],
            sigma_high=opt["sigma_high"],
        )
        log.info("Executing: %s", calc_sky_cmd)
        os.system(calc_sky_cmd)

    def generate_base_param_file(self, param_file, options, n):
        """
        Write Nimg and global dolphot options to param file.

        Parameters
        ----------
        param_file : file-like
            Open file for writing.
        options : dict
            global_defaults or detector_defaults with "dolphot" key.
        n : int
            Number of images (Nimg).
        """
        param_file.write("Nimg = {n}\n".format(n=n))
        for par, value in options["dolphot"].items():
            param_file.write("{par} = {value}\n".format(par=par, value=value))

    def get_dolphot_instrument_parameters(self, image, options):
        """
        Get dolphot instrument parameters for image from detector_defaults.

        Parameters
        ----------
        image : str
            Path to FITS image.
        options : dict
            detector_defaults (keys like acs_wfc, wfc3_uvis).

        Returns
        -------
        dict
            Dolphot parameters for this detector.
        """
        p = self._p
        instrument_string = p._fits.get_instrument(image)
        detector_string = "_".join(instrument_string.split("_")[:2])
        return options[detector_string]["dolphot"]

    def add_image_to_param_file(
        self, param_file, image, i, options, is_wfpc2=False
    ):
        """
        Write imgXXXX_file and imgXXXX_* parameters for one image to param file.

        Parameters
        ----------
        param_file : file-like
            Open param file.
        image : str
            Path to FITS image (base name without .fits used in file).
        i : int
            Image index (zero-padded as img0001, etc.).
        options : dict
            detector_defaults for this detector.
        is_wfpc2 : bool, optional
            If True, log WFPC2-specific parameter adjustments. Default False.
        """
        param_file.write(
            "img{i}_file = {file}\n".format(
                i=str(i).zfill(4), file=os.path.splitext(image)[0]
            )
        )
        params = self.get_dolphot_instrument_parameters(image, options)
        for par, val in params.items():
            if is_wfpc2 and par in ["RAper", "RPSF", "apsize"]:
                log.info("Adjusting for WFPC2 %s = %s", par, val)
            elif is_wfpc2 and par in ["apsky", "RSky", "RSky2"]:
                log.info("Adjusting for WFPC2 %s = %s", par, val)
            param_file.write(
                "img{i}_{par} = {val}\n".format(
                    i=str(i).zfill(4), par=par, val=val
                )
            )

    def make_dolphot_file(self, images, reference):
        """
        Write full dolphot parameter file (reference as img0000, then images).

        Parameters
        ----------
        images : list of str
            Paths to split/science images to include.
        reference : str
            Reference image path (written as img0000).
        """
        p = self._p
        dopt = p.options["detector_defaults"]
        gopt = p.options["global_defaults"]
        with open(p.dolphot["param"], "w") as dolphot_file:
            self.generate_base_param_file(dolphot_file, gopt, len(images))
            inst = p._fits.get_instrument(reference)
            is_wfpc2 = "wfpc2" in inst.lower()
            log.info("Checking reference %s instrument type %s", reference, inst)
            log.info("WFPC2=%s", is_wfpc2)
            self.add_image_to_param_file(
                dolphot_file, reference, 0, dopt, is_wfpc2=is_wfpc2
            )
            for i, image in enumerate(images):
                self.add_image_to_param_file(
                    dolphot_file, image, i + 1, dopt
                )

    def run_dolphot(self):
        """
        Execute dolphot using pipeline dolphot param file and base name.

        Writes output to dolphot["base"] and log to dolphot["log"]. Requires
        dolphot["param"] to exist (e.g. from make_dolphot_file).
        """
        from hst123.utils.logging import make_banner

        p = self._p
        if os.path.isfile(p.dolphot["param"]):
            if os.path.exists(p.dolphot["base"]):
                os.remove(p.dolphot["base"])
            if os.path.exists(p.dolphot["original"]):
                os.remove(p.dolphot["original"])
            cmd = "dolphot {base} -p{par} > {log}".format(
                base=p.dolphot["base"],
                par=p.dolphot["param"],
                log=p.dolphot["log"],
            )
            make_banner("Running dolphot with cmd={cmd}".format(cmd=cmd))
            os.system(cmd)
            time.sleep(10)
            log.info("dolphot is finished (whew)!")
            if os.path.exists(p.dolphot["base"]):
                filesize = (
                    os.stat(p.dolphot["base"]).st_size / 1024 / 1024
                )
                log.info(
                    "Output dolphot file size is %s MB",
                    "%.3f" % filesize,
                )
        else:
            log.error(
                "ERROR: dolphot parameter file %s does not exist! "
                "Generate a parameter file first.",
                p.dolphot["param"],
            )

    def prepare_dolphot(self, image):
        """
        Mask, split, and calcsky image as needed; return list of split images to use.

        Parameters
        ----------
        image : str
            Path to science FITS file.

        Returns
        -------
        list of str
            Paths to split chip images that contain the coordinate (or all if include_all_splits).
        """
        p = self._p
        if self.needs_to_be_masked(image):
            inst = p._fits.get_instrument(image).split("_")[0]
            self.mask_image(image, inst)
        if self.needs_to_split_groups(image):
            self.split_groups(image)
        outimg = []
        split_images = glob.glob(image.replace(".fits", ".chip?.fits"))
        for im in split_images:
            if not p.split_image_contains(
                im, p.coord
            ) and not p.options["args"].include_all_splits:
                os.remove(im)
            else:
                if self.needs_to_calc_sky(im):
                    self.calc_sky(im, p.options["detector_defaults"])
                outimg.append(im)
        return outimg

    def get_dolphot_photometry(self, split_images, reference):
        """
        Scrape photometry from dolphot catalog for pipeline coord and print final photometry.

        Requires dolphot base and colfile to exist. Delegates to pipeline scrapedolphot
        and print_final_phot.

        Parameters
        ----------
        split_images : list of str
            Split image paths (for obstable).
        reference : str
            Reference image path for WCS and scraping.
        """
        from hst123.utils.logging import make_banner

        p = self._p
        ra = p.coord.ra.degree
        dec = p.coord.dec.degree
        make_banner(f"Starting scrape dolphot for: {ra} {dec}")
        opt = p.options["args"]
        dp = p.dolphot
        if (
            os.path.exists(dp["colfile"])
            and os.path.exists(dp["base"])
            and os.stat(dp["base"]).st_size > 0
        ):
            phot = p._scrape_dolphot.scrapedolphot(
                p.coord,
                reference,
                split_images,
                dp,
                get_limits=True,
                scrapeall=opt.scrape_all,
                brightest=opt.brightest,
            )
            p.final_phot = phot
            if phot:
                make_banner(
                    "Printing out the final photometry for: {ra} {dec}\n"
                    "There is photometry for {n} sources".format(
                        ra=ra, dec=dec, n=len(phot)
                    )
                )
                allphot = p.options["args"].scrape_all
                p._scrape_dolphot.print_final_phot(phot, p.dolphot, allphot=allphot)
            else:
                make_banner(
                    f"WARNING: did not find a source for: {ra} {dec}"
                )
        else:
            log.warning(
                "WARNING: dolphot did not run. Use the --run-dolphot flag "
                "or check your dolphot output for errors before using "
                "--scrape-dolphot"
            )

    def do_fake(self, obstable, refname):
        """
        Run dolphot fake star injection at pipeline coord.

        Uses existing dolphot param and base; writes fakelist and fake output.
        No-op if dolphot base is missing or empty.

        Parameters
        ----------
        obstable : astropy.table.Table or list
            Rows with "image" (and optionally filter/zeropoint) for fake mags.
        refname : str
            Reference image path for WCS (fake position).

        Returns
        -------
        None
        """
        from hst123.utils.logging import make_banner

        p = self._p
        dp = p.dolphot
        gopt = p.options["global_defaults"]["fake"]
        if not os.path.exists(dp["base"]) or os.path.getsize(dp["base"]) == 0:
            log.warning(
                "WARNING: option --do-fake used but dolphot has not been run."
            )
            return None
        flines = 0
        if os.path.exists(dp["fake"]):
            with open(dp["fake"], "r") as fake:
                for i, line in enumerate(fake):
                    flines = i + 1
        if not os.path.exists(dp["fake"]) or flines < gopt["nstars"]:
            images = []
            imgnums = []
            with open(dp["param"], "r") as param_file:
                for line in param_file:
                    if "_file" in line and "img0000" not in line:
                        filename = (
                            line.split("=")[1].strip() + ".fits"
                        )
                        imgnum = line.split("=")[0]
                        imgnum = int(
                            imgnum.replace("img", "").replace(
                                "_file", ""
                            )
                        )
                        images.append(filename)
                        imgnums.append(imgnum)
            hdu = fits.open(refname)
            w = wcs.WCS(hdu[0].header)
            x, y = wcs.utils.skycoord_to_pixel(p.coord, w, origin=1)
            with open(dp["fakelist"], "w") as fakelist:
                magmin = gopt["mag_min"]
                dm = (gopt["mag_max"] - magmin) / gopt["nstars"]
                for i in np.arange(gopt["nstars"]):
                    line = "0 1 {x} {y} ".format(x=x, y=y)
                    for row in obstable:
                        line += str("{mag} ".format(mag=magmin + i * dm))
                    fakelist.write(line + "\n")
            p.options["global_defaults"]["FakeStars"] = dp["fakelist"]
            p.options["global_defaults"]["FakeOut"] = dp["fake"]
            defaults = p.options["detector_defaults"]
            Nimg = len(obstable)
            with open(dp["param"], "w") as dfile:
                self.generate_base_param_file(dfile, defaults, Nimg)
                inst = p._fits.get_instrument(refname)
                is_wfpc2 = "wfpc2" in inst.lower()
                self.add_image_to_param_file(
                    dfile, refname, 0, defaults, is_wfpc2=is_wfpc2
                )
                for i, row in enumerate(obstable):
                    self.add_image_to_param_file(
                        dfile, row["image"], i + 1, defaults
                    )
            cmd = "dolphot {base} -p{param} > {log}".format(
                base=dp["base"],
                param=dp["param"],
                log=dp["fakelog"],
            )
            log.info(cmd)
            os.system(cmd)
            log.info("dolphot fake stars is finished (whew)!")
        return None
