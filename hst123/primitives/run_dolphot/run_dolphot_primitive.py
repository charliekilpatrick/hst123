"""Run DOLPHOT, prepare images (calcsky, splitgroups, *mask), param files, fake stars. Scraping is in ScrapeDolphotPrimitive."""
import glob
import os
import shutil
import signal
import sys
import tempfile
import time

import numpy as np
from astropy.io import fits
import astropy.wcs as wcs

from hst123.primitives.base import BasePrimitive
from hst123.utils.dolphot_sky import (
    calcsky_max_pixels_external,
    primary_array_pixel_count,
    sky_fits_path,
    summarize_primary_for_calcsky,
    write_calcsky_sanitized_input,
    write_sky_fits_fallback,
)
from hst123.utils.logging import get_logger, log_calls, make_banner, run_external_command
from hst123.utils.paths import pipeline_chip_output_dir
from hst123.utils.wcs_utils import wcs_from_fits_hdu

log = get_logger(__name__)

# After first full WARNING for missing DOLPHOT source tree, repeat fallbacks at DEBUG.
_dolphot_python_mask_tree_warned = False

DOLPHOT_REQUIRED_SCRIPTS = [
    "dolphot",
    "calcsky",
    # acsmask / wfc3mask / wfpc2mask: hst123.utils.dolphot_mask (Python)
    # splitgroups: hst123.utils.dolphot_splitgroups (Python)
]


def dolphot_subprocess_env() -> dict[str, str]:
    """
    Environment for ``dolphot`` (and similar OpenMP) subprocesses.

    On **macOS**, multithreaded DOLPHOT often prints ``Using N threads`` then exits
    with **SIGTRAP** (``zsh: trace trap``) when OpenMP + ``libomp`` do not match the
    binary (common on Apple Silicon or mixed x86_64/arm64). Forcing a single thread
    avoids most of these crashes.

    Resolution order:

    1. If ``OMP_NUM_THREADS`` is already set in the environment, it is unchanged.
    2. Else if ``HST123_DOLPHOT_OMP_THREADS`` is set (e.g. ``4``), use it as
       ``OMP_NUM_THREADS``.
    3. Else on ``sys.platform == "darwin"``, set ``OMP_NUM_THREADS=1``.

    For a permanent multi-threaded run after fixing **Homebrew** ``libomp`` and
    rebuilding DOLPHOT (see README), export ``OMP_NUM_THREADS`` or
    ``HST123_DOLPHOT_OMP_THREADS`` before running the pipeline.

    Note: ``calcsky`` uses :func:`calcsky_subprocess_env` instead, because an inherited
    shell ``OMP_NUM_THREADS`` (e.g. from conda) commonly breaks **calcsky** on macOS
    even when ``dolphot`` would be fine.
    """
    env = os.environ.copy()
    if "OMP_NUM_THREADS" in os.environ:
        return env
    explicit = os.environ.get("HST123_DOLPHOT_OMP_THREADS")
    if explicit is not None and str(explicit).strip() != "":
        env["OMP_NUM_THREADS"] = str(explicit).strip()
        return env
    if sys.platform == "darwin":
        env["OMP_NUM_THREADS"] = "1"
    return env


def calcsky_subprocess_env() -> dict[str, str]:
    """
    Environment for the ``calcsky`` binary only.

    On **macOS**, ``calcsky`` frequently aborts with **SIGTRAP** if ``OMP_NUM_THREADS>1``
    and OpenMP libraries are mismatched. Unlike :func:`dolphot_subprocess_env`, this
    **does not** inherit a global ``OMP_NUM_THREADS`` from the parent process: conda
    and other stacks often set it to the CPU count, which triggers the crash. We
    default to a single thread on Darwin.

    Override with ``HST123_CALCSKY_OMP_THREADS`` (e.g. ``4``) after verifying
    ``calcsky`` is linked against the correct Homebrew ``libomp``.
    """
    env = os.environ.copy()
    if sys.platform != "darwin":
        return dolphot_subprocess_env()
    explicit = os.environ.get("HST123_CALCSKY_OMP_THREADS")
    if explicit is not None and str(explicit).strip() != "":
        env["OMP_NUM_THREADS"] = str(explicit).strip()
    else:
        env["OMP_NUM_THREADS"] = "1"
    return env


class DolphotPrimitive(BasePrimitive):
    """Run DOLPHOT, prepare images, param files, fake stars; scraping is in ScrapeDolphotPrimitive."""

    def check_for_dolphot(self):
        """Return True if all DOLPHOT_REQUIRED_SCRIPTS are on PATH."""
        for s in DOLPHOT_REQUIRED_SCRIPTS:
            if not shutil.which(s):
                return False
        return True

    def make_dolphot_dict(self, dolphot, work_dir=None):
        """
        Build dict of dolphot file paths and parameters (param, log, base, colfile, etc.).

        Parameters
        ----------
        dolphot : str
            Base name for dolphot output (e.g. "dp" -> dp.param, dp.output, dp.phot).
        work_dir : str | None, optional
            Pipeline work directory. If provided, DOLPHOT products are written
            under ``<work_dir>/dolphot/``.

        Returns
        -------
        dict
            Keys: base, param, log, total_objs, colfile, fake, fakelist, fakelog,
            radius, final_phot, limit_radius, original.
        """
        base = os.fspath(dolphot)
        if work_dir:
            wd = os.path.abspath(os.path.expanduser(work_dir))
            out = os.path.join(wd, "dolphot")
            os.makedirs(out, exist_ok=True)
            # Keep only the leaf name under work_dir/dolphot.
            base = os.path.join(out, os.path.basename(base))
        return {
            "base": base,
            "param": base + ".param",
            "log": base + ".output",
            "total_objs": 0,
            "colfile": base + ".columns",
            "fake": base + ".fake",
            "fakelist": base + ".fakelist",
            "fakelog": base + ".fake.output",
            "radius": 12,
            "final_phot": base + ".phot",
            "limit_radius": 10.0,
            "original": base + ".orig",
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
        sky_path = sky_fits_path(image)
        log.debug("calcsky check sky_path=%s science=%s", sky_path, image)
        if not os.path.exists(sky_path):
            return True
        files = [sky_path]
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

    def _chip_stem(self, image):
        return os.path.splitext(os.path.basename(image))[0]

    def _chip_out_dir(self):
        opt = getattr(self._p, "options", None)
        if not opt:
            return None
        args = opt.get("args") if isinstance(opt, dict) else getattr(opt, "args", None)
        if args is None:
            return None
        wd = getattr(args, "work_dir", None)
        return pipeline_chip_output_dir(wd)

    def _chip_glob_pattern(self, image):
        """Glob for per-chip FITS (under base work_dir when set)."""
        stem = self._chip_stem(image)
        cod = self._chip_out_dir()
        if cod:
            return os.path.join(cod, f"{stem}.chip?.fits")
        return image.replace(".fits", ".chip?.fits")

    def _input_fits_newer_than_chip_products(self, image) -> bool:
        """
        Return True if the science FITS was modified after split chip FITS were written.

        Used to invalidate a full set of ``*.chipN.fits`` when the input exposure was
        replaced or re-calibrated (same stem, newer mtime than all chips).
        """
        chips = glob.glob(self._chip_glob_pattern(image))
        if not chips:
            return False
        try:
            im_mtime = os.path.getmtime(image)
        except OSError:
            return False
        try:
            newest_chip = max(os.path.getmtime(c) for c in chips)
        except OSError:
            return False
        return im_mtime > newest_chip

    def needs_to_split_groups(self, image):
        """
        Return True if splitgroups should run.

        Splitting is skipped when the expected number of ``<stem>.chipN.fits`` files
        already exist next to the exposure (or under the pipeline chip output directory)
        **and** the science FITS is not newer than those chip files (see
        :meth:`_input_fits_newer_than_chip_products`).
        """
        from hst123.utils.dolphot_splitgroups import count_expected_split_outputs

        expected = count_expected_split_outputs(image)
        if expected == 0:
            return False
        n = len(glob.glob(self._chip_glob_pattern(image)))
        if n != expected:
            return True
        return self._input_fits_newer_than_chip_products(image)

    def dolphot_mask_markers_done_in_primary(self, image) -> bool:
        """
        Return True if the primary HDU indicates DOLPHOT masking already ran.

        WFPC2 / WFC3 / ACS write DOLWFPC2, DOL_WFC3, or DOL_ACS = 0 in the science
        file after ``*mask``; that is the on-disk status used to skip re-masking.
        Other instruments are not checked here (see :meth:`needs_to_be_masked`).
        """
        p = self._p
        with fits.open(image) as hdulist:
            header = hdulist[0].header
        inst = p._fits.get_instrument(image).split("_")[0].upper()
        if inst == "WFPC2":
            return "DOLWFPC2" in header and header["DOLWFPC2"] == 0
        if inst == "WFC3":
            return "DOL_WFC3" in header and header["DOL_WFC3"] == 0
        if inst == "ACS":
            return "DOL_ACS" in header and header["DOL_ACS"] == 0
        return False

    def needs_to_be_masked(self, image):
        """
        Return True if the DOLPHOT mask step should run for this exposure.

        For ACS / WFC3 / WFPC2, masking is **skipped** when the primary header already
        has the corresponding DOL* keyword set to 0 (mask already applied on disk).
        """
        p = self._p
        inst = p._fits.get_instrument(image).split("_")[0].upper()
        if inst in ("WFPC2", "WFC3", "ACS"):
            return not self.dolphot_mask_markers_done_in_primary(image)
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
        cod = self._chip_out_dir()
        chip_out_dir = cod if cod else None
        use_ext = os.environ.get(
            "HST123_DOLPHOT_SPLITGROUPS_EXTERNAL", ""
        ).strip().lower() in ("1", "true", "yes")
        if not use_ext:
            try:
                from hst123.utils.dolphot_splitgroups import apply_splitgroups

                apply_splitgroups(image, chip_out_dir=chip_out_dir, log_=log)
            except Exception as exc:
                log.warning(
                    "Python splitgroups failed (%s); falling back to splitgroups",
                    exc,
                )
                use_ext = True
        if use_ext:
            run_external_command(["splitgroups", image], log=log)
            if cod:
                img_dir = os.path.dirname(os.path.abspath(image))
                stem = self._chip_stem(image)
                for f in sorted(
                    glob.glob(os.path.join(img_dir, f"{stem}.chip*.fits"))
                ):
                    dest = os.path.join(cod, os.path.basename(f))
                    if os.path.abspath(f) != os.path.abspath(dest):
                        shutil.move(f, dest)
        if delete_non_science:
            stem = self._chip_stem(image)
            gdir = cod if cod else os.path.dirname(os.path.abspath(image))
            split_images = glob.glob(os.path.join(gdir, f"{stem}.chip*.fits"))
            for split in split_images:
                hdu = fits.open(split)
                info = hdu[0]._summary()
                if info[0].upper() != "SCI":
                    log.warning(
                        "Deleting %s; not a science extension.",
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
        inst_l = (instrument or "").lower()
        mask_exe = f"{inst_l}mask"
        # acsmask reads WFC PAM FITS from the DOLPHOT tree; fail fast with a clear message.
        if inst_l == "acs":
            from hst123.dolphot_install import verify_acs_wfc_pam_files

            ok, msgs = verify_acs_wfc_pam_files()
            if not ok:
                raise RuntimeError(
                    "DOLPHOT ACS acsmask needs WFC PAM files (wfc1_pam.fits, wfc2_pam.fits) "
                    "under .../dolphot3.1/acs/data/. Problems: "
                    + "; ".join(msgs)
                    + ". Fix: run hst123-install-dolphot without --no-psfs, or extract "
                    "ACS_WFC_PAM.tar.gz into the DOLPHOT source tree. "
                    "If the install lives on cloud storage, ensure files are fully local "
                    "(not zero-byte placeholders)."
                )
        if inst_l == "wfc3":
            from hst123.dolphot_install import ensure_wfc3_mask_support_files

            ok, msgs = ensure_wfc3_mask_support_files()
            if not ok:
                raise RuntimeError(
                    "DOLPHOT WFC3 wfc3mask needs distortion map FITS (UVIS1wfc3_map.fits, "
                    "UVIS2wfc3_map.fits, ir_wfc3_map.fits) under .../dolphot3.1/wfc3/data/. "
                    "Problems: "
                    + "; ".join(msgs)
                    + ". Fix: install WFC3 PAM archives (WFC3_UVIS_PAM.tar.gz and "
                    "WFC3_IR_PAM.tar.gz supply these maps) — run "
                    "hst123-install-dolphot --wfc3-maps-only or a full install with "
                    "PSFs, or run hst123-install-dolphot without --no-psfs. "
                    "If the tree is on cloud storage, ensure maps are fully synced "
                    "(not zero-byte placeholders)."
                )
        maskimage = p._fits.get_dq_image(image)
        use_ext = os.environ.get("HST123_DOLPHOT_MASK_EXTERNAL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        dq_arg = maskimage if maskimage and os.path.isfile(maskimage) else None
        if not use_ext:
            try:
                from hst123.utils.dolphot_mask import apply_dolphot_mask_instrument

                log.info("DOLPHOT mask (Python): %s %s", inst_l, os.path.basename(image))
                apply_dolphot_mask_instrument(
                    inst_l,
                    image,
                    dq_arg,
                    log_=log,
                )
            except Exception as exc:
                global _dolphot_python_mask_tree_warned
                err_l = str(exc).lower()
                tree_missing = (
                    "source tree" in err_l
                    or "hst123_dolphot_root" in err_l
                    or "need acs/data" in err_l
                )
                if tree_missing and _dolphot_python_mask_tree_warned:
                    log.debug(
                        "Python DOLPHOT mask (no DOLPHOT source tree): falling back "
                        "to %s for %s",
                        mask_exe,
                        os.path.basename(image),
                    )
                else:
                    if tree_missing:
                        _dolphot_python_mask_tree_warned = True
                    log.warning(
                        "Python DOLPHOT mask failed (%s); falling back to %s",
                        exc,
                        mask_exe,
                    )
                use_ext = True
        if use_ext:
            cmd = [mask_exe, image]
            if maskimage:
                if os.path.isfile(maskimage):
                    cmd.append(maskimage)
                else:
                    log.warning(
                        "DQ path from get_dq_image is not an existing file (%r); "
                        "running %s with science image only",
                        maskimage,
                        mask_exe,
                    )
            log.info("Executing: %s", " ".join(cmd))
            run_external_command(cmd, log=log)
        wd = os.path.dirname(os.path.abspath(image)) or "."
        self._primitive_cleanup(
            "mask_image",
            work_dir=wd,
            validate_fits_paths=[os.path.abspath(image)]
            if os.path.isfile(image)
            else [],
        )

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
        det = "_".join(self._p._fits.get_instrument(image).split("_")[:2]).lower()
        opt = options[det]["dolphot_sky"]
        img = os.fspath(image)
        final_sky = sky_fits_path(img)
        work_dir = os.path.dirname(os.path.abspath(img)) or "."
        base = os.path.basename(img)
        tmp_img = ""
        tmp_sky_out = ""
        calcsky_ok = False
        cp = None
        try:
            try:
                summ = summarize_primary_for_calcsky(img)
            except Exception as exc:
                summ = f"(could not summarize: {exc})"
            log.info("calcsky: %s | %s", base, summ)

            n1, n2, npx = primary_array_pixel_count(img)
            max_px = calcsky_max_pixels_external()
            if max_px > 0 and npx > max_px:
                log.info(
                    "calcsky: skip external binary %dx%d=%d px > HST123_CALCSKY_MAX_PIXELS=%d "
                    "(large drizzles often crash calcsky; Python sky map)",
                    n1,
                    n2,
                    npx,
                    max_px,
                )
                write_sky_fits_fallback(
                    img,
                    final_sky,
                    r_in=int(opt["r_in"]),
                    r_out=int(opt["r_out"]),
                    step=int(opt["step"]),
                    sigma_low=float(opt["sigma_low"]),
                    sigma_high=float(opt["sigma_high"]),
                )
                log.info("Wrote sky map (Python, large-image path): %s", final_sky)
                return

            # Short temp path: upstream calcsky.c uses char str[81] with
            # sprintf("%s.sky.fits", argv[1]); a long work_dir overflows → SIGTRAP
            # (__chk_fail_overflow) on macOS. System temp + prefix "cs" stays safe
            # even before hst123's calcsky.c source patch (see dolphot_install).
            fd, tmp_img = tempfile.mkstemp(
                suffix=".fits", prefix="cs", dir=tempfile.gettempdir()
            )
            os.close(fd)
            tmp_root = tmp_img[:-5] if tmp_img.lower().endswith(".fits") else tmp_img
            tmp_sky_out = tmp_root + ".sky.fits"
            calcsky_ok = False
            cp = None
            try:
                try:
                    write_calcsky_sanitized_input(img, tmp_img)
                except (OSError, ValueError, TypeError) as exc:
                    log.warning(
                        "Could not prepare calcsky input (%s); skipping external calcsky",
                        exc,
                    )
                else:
                    calc_argv = [
                        "calcsky",
                        tmp_root,
                        str(opt["r_in"]),
                        str(opt["r_out"]),
                        str(opt["step"]),
                        str(opt["sigma_low"]),
                        str(opt["sigma_high"]),
                    ]
                    log.info("calcsky: exec %s", " ".join(calc_argv))
                    log.info(
                        "calcsky: external binary has no progress API; "
                        "row-level progress is logged for the Python/Numba sky map path."
                    )
                    try:
                        cp = run_external_command(
                            calc_argv,
                            log=log,
                            check=False,
                            env=calcsky_subprocess_env(),
                        )
                    except OSError as exc:
                        log.warning(
                            "calcsky could not be executed (%s); using Python sky fallback",
                            exc,
                        )
                        cp = None

                    if (
                        cp is not None
                        and cp.returncode == 0
                        and os.path.isfile(tmp_sky_out)
                    ):
                        shutil.move(tmp_sky_out, final_sky)
                        log.info("calcsky: ok -> %s", os.path.basename(final_sky))
                        calcsky_ok = True
                    elif cp is not None and cp.returncode == 0:
                        log.warning(
                            "calcsky exited 0 but output missing (%s); using Python sky fallback",
                            tmp_sky_out,
                        )
            finally:
                for path in (tmp_img, tmp_sky_out):
                    try:
                        if path and os.path.isfile(path):
                            os.unlink(path)
                    except OSError:
                        pass

            if calcsky_ok:
                return

            if cp is not None and cp.returncode < 0:
                sig = -cp.returncode
                _Signals = getattr(signal, "Signals", None)
                if _Signals is not None:
                    try:
                        sig_name = _Signals(sig).name
                    except ValueError:
                        sig_name = "?"
                else:
                    sig_name = "?"
                log.warning(
                    "calcsky: abnormal exit %s (signal %s=%s); Python sky fallback",
                    cp.returncode,
                    sig,
                    sig_name,
                )
                if sig == 5 and tmp_root and len(tmp_root) > 72:
                    log.warning(
                        "calcsky: SIGTRAP with long base path (%d chars) often means an "
                        "**unpatched** calcsky.c stack buffer (81 B) overflow, not OpenMP — "
                        "re-run hst123-install-dolphot (applies calcsky.c patch) or rely "
                        "on this Python sky map.",
                        len(tmp_root),
                    )
            elif cp is not None:
                log.warning(
                    "calcsky: exit %s; Python sky fallback",
                    cp.returncode,
                )

            write_sky_fits_fallback(
                img,
                final_sky,
                r_in=int(opt["r_in"]),
                r_out=int(opt["r_out"]),
                step=int(opt["step"]),
                sigma_low=float(opt["sigma_low"]),
                sigma_high=float(opt["sigma_high"]),
            )
            log.info("Wrote sky map (Python fallback): %s", final_sky)
        finally:
            vfit = [os.path.abspath(img)]
            if os.path.isfile(final_sky):
                vfit.append(os.path.abspath(final_sky))
            self._primitive_cleanup(
                "calc_sky",
                work_dir=work_dir,
                remove_globs=(".hst123_calcsky_*.fits",),
                validate_fits_paths=vfit,
            )

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
        detector_string = "_".join(instrument_string.split("_")[:2]).lower()
        return options[detector_string]["dolphot"]

    def add_image_to_param_file(
        self,
        param_file,
        image,
        i,
        options,
        is_wfpc2=False,
        work_dir=None,
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
        work_dir : str, optional
            Pipeline ``--work-dir``. When set, write ``imgXXXX_file`` paths **relative
            to this directory** so DOLPHOT is not fed very long absolute strings (some
            builds abort with ``*** buffer overflow detected ***`` when parsing long
            ``img*_file`` lines). :meth:`run_dolphot` runs ``dolphot`` with ``cwd`` set
            to *work_dir*, so relative stems resolve correctly.
        """
        abs_img = os.path.abspath(os.path.expanduser(os.fspath(image)))
        if work_dir and str(work_dir).strip():
            wd = os.path.abspath(os.path.expanduser(str(work_dir).strip()))
            try:
                rel = os.path.relpath(abs_img, wd).replace("\\", "/")
                root_no_ext = os.path.splitext(rel)[0]
            except ValueError:
                root_no_ext = os.path.splitext(abs_img)[0].replace("\\", "/")
        else:
            root_no_ext = os.path.splitext(abs_img)[0].replace("\\", "/")
        param_file.write(
            "img{i}_file = {file}\n".format(i=str(i).zfill(4), file=root_no_ext)
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
        args = p.options.get("args")
        wd = None
        if args is not None:
            raw = getattr(args, "work_dir", None)
            if isinstance(raw, str) and raw.strip():
                wd = os.path.abspath(os.path.expanduser(raw))
        param_rel = p.dolphot["param"]
        param_path = (
            os.path.join(wd, param_rel) if wd else param_rel
        )
        # DOLPHOT expects Nimg == number of science images (img0001..imgNNNN).
        # The drizzled reference/template is img0000_file and is not counted.
        nimg = len(images)
        with open(param_path, "w", encoding="utf-8") as dolphot_file:
            self.generate_base_param_file(dolphot_file, gopt, nimg)
            inst = p._fits.get_instrument(reference)
            is_wfpc2 = "wfpc2" in inst.lower()
            log.info("Checking reference %s instrument type %s", reference, inst)
            log.info("WFPC2=%s", is_wfpc2)
            self.add_image_to_param_file(
                dolphot_file,
                reference,
                0,
                dopt,
                is_wfpc2=is_wfpc2,
                work_dir=wd,
            )
            for i, image in enumerate(images):
                self.add_image_to_param_file(
                    dolphot_file, image, i + 1, dopt, work_dir=wd
                )
        self._primitive_cleanup(
            "make_dolphot_file",
            validate_text_paths=[param_path]
            if os.path.isfile(param_path)
            else [],
            validation_notes={
                "n_images": len(images),
                "Nimg": nimg,
                "reference": os.path.basename(reference),
            },
        )

    @log_calls
    def run_dolphot(self):
        """
        Execute dolphot using pipeline dolphot param file and base name.

        Writes output to dolphot["base"] and log to dolphot["log"]. Requires
        dolphot["param"] to exist (e.g. from make_dolphot_file).
        """
        p = self._p
        args = p.options.get("args")
        wd = None
        if args is not None:
            raw_wd = getattr(args, "work_dir", None)
            if isinstance(raw_wd, str) and raw_wd.strip():
                wd = os.path.abspath(os.path.expanduser(raw_wd))
        param_path = os.path.join(wd, p.dolphot["param"]) if wd else p.dolphot["param"]
        log_path = os.path.join(wd, p.dolphot["log"]) if wd else p.dolphot["log"]
        base_name = p.dolphot["base"]
        try:
            if os.path.isfile(param_path):
                base_fp = os.path.join(wd, base_name) if wd else base_name
                orig_fp = (
                    os.path.join(wd, p.dolphot["original"]) if wd else p.dolphot["original"]
                )
                if os.path.exists(base_fp):
                    os.remove(base_fp)
                if os.path.exists(orig_fp):
                    os.remove(orig_fp)
                # Prefer paths relative to --work-dir to avoid legacy DOLPHOT buffer
                # overflows on long absolute paths (common on shared /data trees).
                # We still run with cwd=wd so relative paths resolve correctly.
                rel_base = base_name
                rel_param = param_path
                if wd:
                    try:
                        rel_base = os.path.relpath(os.path.abspath(base_name), wd)
                        rel_param = os.path.relpath(os.path.abspath(param_path), wd)
                    except Exception:
                        rel_base = base_name
                        rel_param = param_path

                dolphot_argv = ["dolphot", rel_base, f"-p{rel_param}"]
                banner_cmd = "dolphot {base} -p{par} (log -> {log})".format(
                    base=rel_base,
                    par=rel_param,
                    log=log_path,
                )
                make_banner("Running dolphot: {cmd}".format(cmd=banner_cmd))
                run_external_command(
                    dolphot_argv,
                    log=log,
                    tee_path=log_path,
                    cwd=wd,
                    env=dolphot_subprocess_env(),
                )
                time.sleep(10)
                log.info("dolphot is finished (whew)!")
                out_cat = os.path.join(wd, base_name + ".phot") if wd else base_name + ".phot"
                if os.path.isfile(out_cat):
                    filesize = os.stat(out_cat).st_size / 1024 / 1024
                    log.info(
                        "Output dolphot file size is %s MB",
                        "%.3f" % filesize,
                    )
                elif os.path.isfile(os.path.join(wd, base_name) if wd else base_name):
                    filesize = (
                        os.stat(os.path.join(wd, base_name) if wd else base_name).st_size
                        / 1024
                        / 1024
                    )
                    log.info(
                        "Output dolphot file size is %s MB",
                        "%.3f" % filesize,
                    )
            else:
                log.error(
                    "ERROR: dolphot parameter file %s does not exist! "
                    "Generate a parameter file first.",
                    param_path,
                )
        finally:
            vtxt = []
            phot_out = os.path.join(wd, base_name + ".phot") if wd else base_name + ".phot"
            if os.path.isfile(phot_out):
                vtxt.append(phot_out)
            elif os.path.isfile(os.path.join(wd, base_name) if wd else base_name):
                vtxt.append(os.path.join(wd, base_name) if wd else base_name)
            if os.path.isfile(log_path):
                vtxt.append(log_path)
            cleanup_wd = wd
            args = p.options.get("args")
            if not cleanup_wd and args is not None:
                raw_wd = getattr(args, "work_dir", None)
                if isinstance(raw_wd, str) and raw_wd.strip():
                    cleanup_wd = os.path.abspath(os.path.expanduser(raw_wd))
            if not cleanup_wd and os.path.isfile(param_path):
                cleanup_wd = os.path.dirname(os.path.abspath(param_path))
            if not cleanup_wd:
                cleanup_wd = os.getcwd()
            self._primitive_cleanup(
                "run_dolphot",
                work_dir=cleanup_wd,
                remove_globs=("*drc.noise.fits",),
                validate_text_paths=vtxt,
                text_min_size=0,
                validation_notes={
                    "param_exists": os.path.isfile(param_path),
                },
                # Ephemeral sky sidecars; not drizzle debug artifacts
                respect_keep_artifacts=False,
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
        else:
            log.info(
                "Skipping DOLPHOT mask for %s: primary header indicates mask "
                "already applied (DOL* marker).",
                os.path.basename(image),
            )
        from hst123.utils.dolphot_splitgroups import count_expected_split_outputs

        if self.needs_to_split_groups(image):
            self.split_groups(image)
        elif count_expected_split_outputs(image) > 0:
            log.info(
                "Skipping splitgroups for %s: expected chip FITS present and "
                "not older than the science file.",
                os.path.basename(image),
            )
        outimg = []
        split_images = glob.glob(self._chip_glob_pattern(image))
        for im in split_images:
            if not p.split_image_contains(
                im, p.coord
            ) and not p.options["args"].include_all_splits:
                os.remove(im)
            else:
                if self.needs_to_calc_sky(im):
                    self.calc_sky(im, p.options["detector_defaults"])
                outimg.append(im)
        wd = os.path.dirname(os.path.abspath(image)) or "."
        vfit = [os.path.abspath(x) for x in outimg]
        for im in outimg:
            sp = sky_fits_path(im)
            if os.path.isfile(sp):
                vfit.append(os.path.abspath(sp))
        self._primitive_cleanup(
            "prepare_dolphot",
            work_dir=wd,
            remove_globs=(".hst123_calcsky_*.fits",),
            validate_fits_paths=vfit,
        )
        return outimg

    def collect_existing_split_images(self, image):
        """
        Return chip FITS paths if a complete, up-to-date chip set exists without
        re-running mask, splitgroups, or per-chip calcsky.

        Returns ``None`` if chips are missing, incomplete, or newer science data
        implies a full :meth:`prepare_dolphot` run.
        """
        p = self._p
        from hst123.utils.dolphot_splitgroups import count_expected_split_outputs

        expected = count_expected_split_outputs(image)
        if expected == 0:
            return []
        chips = glob.glob(self._chip_glob_pattern(image))
        if len(chips) != expected:
            return None
        if self._input_fits_newer_than_chip_products(image):
            return None
        outimg = []
        for im in sorted(chips):
            if p.split_image_contains(
                im, p.coord
            ) or p.options["args"].include_all_splits:
                outimg.append(im)
        if not outimg:
            return None
        return outimg

    def get_dolphot_photometry(self, split_images, reference, visit_obstable=None):
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
        visit_obstable : astropy.table.Table, optional
            Per-visit observation table for fast chip-level metadata (see
            :meth:`hst123.hst123.expand_obstable_for_split_images`).
        """
        from hst123.utils.logging import make_banner

        p = self._p
        ra = p.coord.ra.degree
        dec = p.coord.dec.degree
        make_banner(f"Starting scrape dolphot for: {ra} {dec}")
        opt = p.options["args"]
        dp = p.dolphot
        phot = None
        try:
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
                    visit_obstable=visit_obstable,
                )
                p.final_phot = phot
                cat_for_h5 = getattr(p, "_last_dolphot_catalog_array", None)
                try:
                    if getattr(opt, "write_dolphot_hdf5", True):
                        try:
                            from pathlib import Path

                            from hst123.utils.dolphot_catalog_hdf5 import (
                                append_scraped_final_phot_hdf5,
                                write_dolphot_catalog_hdf5,
                            )

                            base = Path(dp["base"])
                            wda = os.path.abspath(
                                os.path.expanduser(getattr(opt, "work_dir", None) or ".")
                            )
                            out_h5 = Path(wda) / (base.name + ".h5")
                            dolphot_dir = base.parent
                            fast_h5 = getattr(opt, "dolphot_hdf5_fast", True)
                            write_dolphot_catalog_hdf5(
                                out_h5,
                                base,
                                include_raw_sidecars=not fast_h5,
                                dolphot_dir=dolphot_dir,
                                embed_dolphot_directory_text=False,
                                catalog_array=cat_for_h5,
                                compression=not fast_h5,
                                include_directory_manifest=not fast_h5,
                                serialize_meta=not fast_h5,
                            )
                            if phot:
                                try:
                                    append_scraped_final_phot_hdf5(
                                        out_h5,
                                        phot,
                                        compression=not fast_h5,
                                        serialize_meta=False,
                                    )
                                except Exception as exc:
                                    log.warning(
                                        "Could not append stacked scrape photometry to HDF5: %s",
                                        exc,
                                    )
                            if fast_h5:
                                log.info(
                                    "Wrote compact DOLPHOT HDF5: %s "
                                    "(full catalog + stacked scraped photometry; use "
                                    "--dolphot-hdf5-full for gzip/manifest/raw sidecars)",
                                    out_h5,
                                )
                            else:
                                log.info(
                                    "Wrote full DOLPHOT HDF5 archive: %s "
                                    "(gzipped table, parsed sidecars, manifest of %s)",
                                    out_h5,
                                    dolphot_dir,
                                )
                        except ImportError as exc:
                            log.info(
                                "DOLPHOT HDF5 not written (install h5py: pip install h5py): %s",
                                exc,
                            )
                        except Exception as exc:
                            log.warning("DOLPHOT HDF5 not written: %s", exc)
                finally:
                    p.__dict__.pop("_last_dolphot_catalog_array", None)
                if phot:
                    p._scrape_dolphot.log_scrape_summary(phot)
                    write_per_source = getattr(
                        opt, "scrape_write_per_source_phot", False
                    )
                    skip_ascii = bool(opt.scrape_all and not write_per_source)
                    if not skip_ascii:
                        make_banner(
                            "Printing out the final photometry for: {ra} {dec}\n"
                            "There is photometry for {n} sources".format(
                                ra=ra, dec=dec, n=len(phot)
                            )
                        )
                        allphot = p.options["args"].scrape_all
                        p._scrape_dolphot.print_final_phot(
                            phot, p.dolphot, allphot=allphot
                        )
                else:
                    make_banner(
                        f"Did not find a source for: {ra} {dec}"
                    )
            else:
                log.warning(
                    "WARNING: dolphot did not run. Use the --run-dolphot flag "
                    "or check your dolphot output for errors before using "
                    "--scrape-dolphot"
                )
        finally:
            raw_wd = getattr(opt, "work_dir", None)
            wd = raw_wd if isinstance(raw_wd, str) and raw_wd.strip() else "."
            wda = os.path.abspath(os.path.expanduser(wd))
            vfit = []
            if reference and os.path.isfile(reference):
                vfit.append(reference)
            for s in split_images or []:
                if s and os.path.isfile(str(s)):
                    vfit.append(str(s))
            tmp_left = os.path.join(wda, "tmp")
            rpaths = [tmp_left] if os.path.isfile(tmp_left) else []
            self._primitive_cleanup(
                "get_dolphot_photometry",
                work_dir=wda,
                remove_globs=("*drc.noise.fits",),
                remove_paths=rpaths,
                validate_fits_paths=vfit,
                validate_tables=phot if phot is not None else [],
                respect_keep_artifacts=False,
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
        args = p.options.get("args")
        wd = None
        if args is not None:
            raw_wd = getattr(args, "work_dir", None)
            if isinstance(raw_wd, str) and raw_wd.strip():
                wd = os.path.abspath(os.path.expanduser(raw_wd))
        try:
            if not os.path.exists(dp["base"]) or os.path.getsize(dp["base"]) == 0:
                log.warning(
                    "Option --do-fake used but dolphot has not been run."
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
                with fits.open(refname) as hdu:
                    w = wcs_from_fits_hdu(hdu, 0)
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
                        dfile,
                        refname,
                        0,
                        defaults,
                        is_wfpc2=is_wfpc2,
                        work_dir=wd,
                    )
                    for i, row in enumerate(obstable):
                        self.add_image_to_param_file(
                            dfile, row["image"], i + 1, defaults, work_dir=wd
                        )
                param_path = os.path.join(wd, dp["param"]) if wd else dp["param"]
                base_name = dp["base"]
                rel_base = base_name
                rel_param = param_path
                if wd:
                    try:
                        rel_base = os.path.relpath(os.path.abspath(base_name), wd)
                        rel_param = os.path.relpath(os.path.abspath(param_path), wd)
                    except Exception:
                        rel_base = base_name
                        rel_param = param_path
                fake_argv = ["dolphot", rel_base, f"-p{rel_param}"]
                log.info("Running: %s (log -> %s)", " ".join(fake_argv), dp["fakelog"])
                run_external_command(
                    fake_argv,
                    log=log,
                    tee_path=dp["fakelog"],
                    cwd=wd,
                    env=dolphot_subprocess_env(),
                )
                log.info("dolphot fake stars is finished (whew)!")
        finally:
            vtxt = []
            for key in ("fakelog", "fakelist", "fake"):
                pth = dp.get(key)
                if pth and os.path.isfile(pth):
                    vtxt.append(pth)
            self._primitive_cleanup(
                "do_fake",
                validate_text_paths=vtxt,
                text_min_size=0,
            )
        return None
