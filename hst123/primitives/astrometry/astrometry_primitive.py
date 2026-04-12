"""Image alignment (TweakReg/JHAT), WCS updates, reference prep. parse_coord for RA/Dec."""

import copy
import functools
import glob
import logging
import os
import re
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
from hst123.utils.options import want_redo_astrometry
from hst123.utils.paths import normalize_fits_path
from hst123.utils.stdio import suppress_stdout
from hst123.utils.logging import format_hdu_list_summary, log_calls
from hst123.utils.workdir_cleanup import cleanup_after_tweakreg
from hst123.utils.alignment_validation import log_tweakreg_shift_metrics
from hst123.primitives.base import BasePrimitive
from hst123.primitives.astrometry.alignment_meta import (
    alignment_is_redundant,
    alignment_method_token,
    normalize_alignment_ref_id,
    read_alignment_provenance,
    write_alignment_provenance,
)

# Logger names emitted by STScI ``stwcs`` (PyPI) when TweakReg/headerlet runs.
_STSCI_WCS_LOGGERS = (
    "stwcs",
    "stwcs.wcsutil",
    "stwcs.wcsutil.headerlet",
)


def _quiet_headerlet_loggers(fn):
    """Raise headerlet-related loggers to WARNING for the duration of *fn* (less noise)."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        prev = {n: logging.getLogger(n).level for n in _STSCI_WCS_LOGGERS}
        try:
            for n in _STSCI_WCS_LOGGERS:
                logging.getLogger(n).setLevel(logging.WARNING)
            return fn(*args, **kwargs)
        finally:
            for n, lev in prev.items():
                logging.getLogger(n).setLevel(lev)

    return wrapped


def _resolve_work_dir_chdir(work_dir):
    """
    Return absolute directory used for alignment (``<work-dir>/workspace``), chdir into it.

    TweakReg, JHAT, and AstroDrizzle scratch/logs use this tree; the main drizzled
    reference may still live in the base ``--work-dir`` when given as an absolute path.

    Relative ``work_dir`` values must not be joined again after ``chdir`` —
    e.g. ``os.chdir("test_data")`` then ``join("test_data", "x.txt")`` would
    resolve to ``test_data/test_data/x.txt`` and raise FileNotFoundError.
    """
    from hst123.utils.paths import pipeline_workspace_dir

    base = work_dir if work_dir else "."
    path = os.path.abspath(os.path.expanduser(base))
    ws = pipeline_workspace_dir(path)
    target = ws if ws else path
    os.makedirs(target, exist_ok=True)
    os.chdir(target)
    return target


def _workspace_rawtmp_path(workspace_fits: str) -> str:
    """Partner ``*.rawtmp.fits`` for a workspace ``*.fits`` (basename-safe)."""
    if workspace_fits.endswith(".rawtmp.fits"):
        return workspace_fits
    if workspace_fits.endswith(".fits"):
        return workspace_fits[:-5] + ".rawtmp.fits"
    return workspace_fits + ".rawtmp.fits"


def _is_number(num):
    """
    Return True if num can be interpreted as a number.

    Parameters
    ----------
    num : any
        Value to check (e.g. str or float).

    Returns
    -------
    bool
        True if float(num) does not raise ValueError or TypeError.
    """
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

try:
    from drizzlepac import tweakreg, catalogs

    from hst123.utils.stsci_wcs import hstwcs_class

    HSTWCS = hstwcs_class()
except ImportError:
    tweakreg = None
    catalogs = None
    HSTWCS = None  # type: ignore[misc,assignment]


class AstrometryPrimitive(BasePrimitive):
    """TweakReg (HST) or JHAT (JWST/Gaia) alignment, reference prep, WCS updates."""

    def prepare_reference_tweakreg(self, reference):
        """
        Prepare reference image for TweakReg (specific HDU layout).

        Parameters
        ----------
        reference : str
            Path to reference FITS file.

        Returns
        -------
        bool
            True if preparation succeeded and file was written.
        """
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

    def _pipeline_reference_filter(self, ref_path: str | None) -> str | None:
        """
        Filter name for the pipeline reference, matching :meth:`_build_tweakreg_batches`.

        Drizzled ``*.drc.fits`` products may have empty ``FILTNAM1`` on PRIMARY while
        ``FILTER`` is set; :meth:`~hst123.primitives.fits.FitsHelper.get_filter`
        may still recover the band. As a last resort, parse ``inst.<filt>.ref_*.drc.fits``.
        """
        if not ref_path or not str(ref_path).strip():
            return None
        rp = normalize_fits_path(str(ref_path).strip())
        if not os.path.isfile(rp):
            return None
        try:
            f = self._p._fits.get_filter(rp)
            f = (f or "").strip()
            if f:
                return f
        except Exception:
            pass
        base = os.path.basename(rp)
        m = re.match(r"^[^.]+\.([a-z0-9]+)\.ref_\d+\.drc\.fits$", base, re.I)
        if m:
            return m.group(1).lower()
        return None

    def _primary_header_for_alignment_probe(self, file: str):
        """
        Primary header to evaluate ALIGN* provenance.

        TweakReg records provenance on working copies (often ``*.rawtmp.fits``).
        When *file* is workspace ``*.fits`` and has no block, probe the partner
        rawtmp if present.
        """
        fp = normalize_fits_path(file)
        candidates = [fp]
        if fp.endswith(".fits") and not fp.endswith(".rawtmp.fits"):
            rt = _workspace_rawtmp_path(fp)
            if rt != fp:
                candidates.append(rt)
        existing = [p for p in candidates if os.path.isfile(p)]
        if not existing:
            with fits.open(fp, mode="readonly") as hdu:
                return hdu[0].header
        for path in existing:
            with fits.open(path, mode="readonly") as hdu:
                hdr = hdu[0].header
                if read_alignment_provenance(hdr) is not None:
                    return hdr
        with fits.open(existing[0], mode="readonly") as hdu:
            return hdu[0].header

    def _effective_tweakreg_reference_for_image(
        self,
        image: str,
        all_images: list[str],
        reference: str | None,
    ) -> str | None:
        """
        Match :meth:`_build_tweakreg_batches` reference choice per filter.

        The pipeline ``handle_reference`` path is often a drizzled ``*.drc.fits``;
        per-filter batches still use the deepest exposure (as ``*.rawtmp.fits``)
        when the drizzle filter does not match. Stored ``ALIGNRF`` must be
        compared against that same path.
        """
        fp = normalize_fits_path(image)
        if not all_images:
            return reference

        def _filt_of(p: str) -> str:
            try:
                return self._p._fits.get_filter(normalize_fits_path(p)).strip()
            except Exception:
                return ""

        by_filt: dict[str, list[str]] = {}
        for p in all_images:
            p2 = normalize_fits_path(p)
            if not os.path.isfile(p2):
                continue
            by_filt.setdefault(_filt_of(p2), []).append(p2)

        filt = _filt_of(fp)
        paths = by_filt.get(filt, [])
        if len(paths) < 2:
            return reference

        ref_ok = bool(
            reference
            and str(reference).strip()
            and reference != "dummy.fits"
            and os.path.isfile(normalize_fits_path(reference))
        )
        ref_path = normalize_fits_path(reference) if ref_ok else None
        ref_filt = self._pipeline_reference_filter(reference) if ref_ok else None

        if ref_filt is not None and filt == ref_filt:
            return reference
        deepest_fits = sorted(paths, key=lambda im: fits.getval(im, "EXPTIME"))[-1]
        raw = _workspace_rawtmp_path(deepest_fits)
        if os.path.isfile(raw):
            return normalize_fits_path(raw)
        return normalize_fits_path(deepest_fits)

    def check_images_for_tweakreg(
        self,
        run_images,
        *,
        alignment_method: str | None = None,
        alignment_ref_id: str | None = None,
        force_realign: bool = False,
        tweakreg_reference_images: list[str] | None = None,
        tweakreg_pipeline_reference: str | None = None,
    ):
        """
        Return images that still need TweakReg-style alignment.

        Skips files that are missing on disk. When *alignment_ref_id* is set and
        *force_realign* is false, skips files whose primary header already records
        a successful alignment with the same ``HIERARCH HST123 ALIGN*`` method and
        reference (redundant re-run).

        Parameters
        ----------
        run_images : list of str
            Paths to FITS images.
        alignment_method : str, optional
            ``tweakreg`` or ``jhat`` (default: pipeline ``--align-with``).
        alignment_ref_id : str, optional
            Reference path, ``GAIA``, etc. If unset, redundancy is not evaluated.
        force_realign : bool, optional
            If True (``--clobber``, ``--redo``, ``--redo-astrometry``), never skip
            based on provenance.
        tweakreg_reference_images : list of str, optional
            Full workspace image list (e.g. obstable ``image`` column). When set
            with :meth:`_effective_tweakreg_reference_for_image`, the reference
            id compared to provenance matches :meth:`_build_tweakreg_batches`
            (pipeline drizzle vs deepest exposure per filter), not only the
            pipeline reference argument.
        tweakreg_pipeline_reference : str, optional
            Pipeline reference path (same as ``handle_reference``). Used with
            *tweakreg_reference_images* for per-filter batch ref resolution.

        Returns
        -------
        list of str or None
            Subset needing alignment; None if nothing left to process.
        """
        if not run_images:
            return None

        images = []
        for file in run_images:
            fp = normalize_fits_path(file)
            if not os.path.isfile(fp):
                log.warning(
                    "Skipping missing or unreadable file before tweakreg: %s", file
                )
                continue
            images.append(fp)

        if not images:
            log.warning("No on-disk images left for tweakreg after filtering.")
            return None

        if alignment_method is None:
            alignment_method = getattr(
                self._p.options["args"], "align_with", "tweakreg"
            )

        method_tok = alignment_method_token(alignment_method)
        use_batch_ref = tweakreg_reference_images is not None

        for file in list(images):
            fp = normalize_fits_path(file)
            ref_id = alignment_ref_id
            if use_batch_ref:
                ref_id = self._effective_tweakreg_reference_for_image(
                    fp,
                    tweakreg_reference_images,
                    tweakreg_pipeline_reference,
                )
                if ref_id is None or str(ref_id).strip() == "":
                    ref_id = alignment_ref_id
            ref_for_compare = (
                normalize_alignment_ref_id(ref_id) if ref_id else None
            )
            hdr = self._primary_header_for_alignment_probe(fp)
            prov = read_alignment_provenance(hdr)
            if (
                not force_realign
                and ref_for_compare
                and ref_id is not None
                and str(ref_id).strip() != ""
                and alignment_is_redundant(
                    hdr,
                    method=method_tok,
                    ref_id=ref_id,
                    require_success=True,
                )
            ):
                log.info(
                    "Skipping alignment for %s: already aligned "
                    "(success=True, method=%s, reference_id=%s)",
                    os.path.basename(file),
                    prov["method"] if prov else method_tok,
                    prov["ref"] if prov else ref_for_compare,
                )
                images.remove(file)
                continue
            log.debug(
                "Alignment needed for %s (provenance=%s, ref_id=%s)",
                file,
                prov,
                ref_for_compare,
            )

        if len(images) == 0:
            return None
        return images

    def get_nsources(self, image, thresh):
        """
        Return number of sources detected in image at given threshold.

        Parameters
        ----------
        image : str
            Path to FITS image.
        thresh : float
            Detection threshold for catalog generation.

        Returns
        -------
        int
            Total number of sources from catalog.
        """
        if catalogs is None or HSTWCS is None:
            log.warning("drizzlepac or stsci_wcs (bundled STScI WCS) unavailable; cannot count sources")
            return 0
        nsources = 0
        log.debug(
            "Source count %s threshold=%s",
            os.path.basename(image),
            thresh,
        )
        with fits.open(image, mode="readonly") as imghdu:
            for i, h in enumerate(imghdu):
                if h.name == "SCI" or (len(imghdu) == 1 and h.name == "PRIMARY"):
                    filename = "{:s}[{:d}]".format(image, i)
                    wcs = HSTWCS(filename)
                    catalog_mode = "automatic"
                    catalog = catalogs.generateCatalog(
                        wcs,
                        mode=catalog_mode,
                        catalog=filename,
                        threshold=thresh,
                        **self._p.options["catalog"],
                    )
                    try:
                        with suppress_stdout():
                            catalog.buildCatalogs()
                        nsources += catalog.num_objects
                    except Exception:
                        pass

        log.debug("Got %s total sources for %s", nsources, os.path.basename(image))
        return nsources

    def count_nsources(self, images):
        """
        Count catalog sources from coo files (tagged with threshold).

        Parameters
        ----------
        images : list of str
            Base image paths; coo files are found by replacing .fits with _sci*_xy_catalog.coo.

        Returns
        -------
        int
            Total line count from matching coo files (excluding "threshold" lines).
        """
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
        """
        Estimate threshold vs. source count for image to reach target nobj.

        Parameters
        ----------
        image : str
            Path to FITS image.
        target : int or float
            Target number of sources.

        Returns
        -------
        list of tuple
            (nobj, threshold) pairs from sampling thresholds.
        """
        trd = settings.tweakreg_defaults
        t_min = float(trd["threshold_min"])
        t_hi = max(80.0, t_min * 2.0)
        # Fixed small grid (was up to 12 catalog passes per image)
        grid = np.geomspace(t_hi, t_min, num=5)
        inp_data = []
        log.debug(
            "TweakReg threshold samples for %s target_nobj=%s",
            os.path.basename(image),
            target,
        )
        for t in grid:
            nobj = self.get_nsources(image, float(t))
            inp_data.append((float(nobj), float(t)))
        inp_data.sort(key=lambda x: x[1])
        log.info(
            "TweakReg threshold curve %s: %d samples, nobj range %.0f–%.0f",
            os.path.basename(image),
            len(inp_data),
            min(x[0] for x in inp_data),
            max(x[0] for x in inp_data),
        )
        return inp_data

    def add_thresh_data(self, thresh_data, image, inp_data):
        """
        Append threshold/source-count row to thresh_data table.

        Parameters
        ----------
        thresh_data : astropy.table.Table or None
            Existing table; if None, a new table is created.
        image : str
            Image path for the "file" column.
        inp_data : list of tuple
            (nobj, threshold) pairs.

        Returns
        -------
        astropy.table.Table
            Table with file column and one column per threshold.
        """
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
        """
        Interpolate threshold for target source count; clamp to settings.

        Parameters
        ----------
        thresh_data : astropy.table.Table
            Table with threshold keys and source counts.
        target : float
            Target number of sources.

        Returns
        -------
        float
            Threshold value (clamped to tweakreg_defaults threshold_min/max).
        """
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

        trd = settings.tweakreg_defaults
        if len(thresh) == 0:
            return float(trd["threshold_min"])
        if len(thresh) == 1:
            threshold = float(thresh[0])
        else:
            thresh_func = interp1d(
                nsources,
                thresh,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            threshold = float(thresh_func(target))

        if threshold < trd["threshold_min"]:
            threshold = trd["threshold_min"]
        if threshold > trd["threshold_max"]:
            threshold = trd["threshold_max"]

        log.debug("Interpolated TweakReg threshold: %s", threshold)
        return threshold

    def _build_tweakreg_batches(self, tmp_images, reference):
        """
        Split working copies by filter so TweakReg uses a spectrally matched reference.

        Aligning e.g. F555W exposures to an F814W reference yields too few cross-band
        matches; each filter aligns to the user reference when filters match, else to
        the deepest exposure in that filter.
        """
        if not tmp_images:
            return []

        def _filt_of(tp):
            orig = tp.replace(".rawtmp.fits", ".fits")
            try:
                return self._p._fits.get_filter(orig).strip()
            except Exception:
                return ""

        by_filt = {}
        for tp in tmp_images:
            by_filt.setdefault(_filt_of(tp), []).append(tp)

        ref_ok = bool(
            reference
            and reference != "dummy.fits"
            and os.path.isfile(reference)
        )
        ref_filt = self._pipeline_reference_filter(reference) if ref_ok else None

        batches = []
        for filt in sorted(by_filt.keys(), key=str):
            paths = by_filt[filt]
            if len(paths) < 2:
                log.info(
                    "TweakReg: skip filter %s — need ≥2 images (have %d)",
                    filt,
                    len(paths),
                )
                continue
            if ref_filt is not None and filt == ref_filt:
                batches.append((reference, paths))
            else:
                deepest = sorted(
                    paths, key=lambda im: fits.getval(im, "EXPTIME")
                )[-1]
                batches.append((deepest, paths))
        return batches

    def get_shallow_param(self, image):
        """
        Return (filter, pivot_wavelength, exptime) for shallow-image checks.

        Parameters
        ----------
        image : str
            Path to FITS image.

        Returns
        -------
        tuple
            (filter_name, pivot_wavelength, exptime).
        """
        filt = self._p._fits.get_filter(image)
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
        """
        Log TweakReg failure banner.

        Parameters
        ----------
        exception : Exception
            The exception raised by TweakReg.
        """
        log.warning(
            "tweakreg failed: %s\n%s\nAdjusting thresholds and images...",
            exception.__class__.__name__,
            exception,
        )

    def apply_tweakreg_success(
        self,
        shifts,
        *,
        ref_id: str,
        method: str = "tweakreg",
    ):
        """
        Set TWEAKSUC=1 and HST123 alignment provenance for files with valid shifts.

        Parameters
        ----------
        shifts : astropy.table.Table or iterable
            Rows with "file", "xoffset", "yoffset"; non-NaN offsets get TWEAKSUC=1.
        ref_id : str
            Reference image path or ``dummy.fits`` / batch reference (stored normalized).
        method : str
            Alignment engine label (default ``tweakreg``).
        """
        for row in shifts:
            xo, yo = row["xoffset"], row["yoffset"]
            if np.isnan(xo) or np.isnan(yo):
                continue
            file = row["file"]
            if not os.path.exists(file):
                log.warning("%s does not exist", file)
                continue
            with fits.open(file, mode="update") as hdu:
                hdu[0].header["TWEAKSUC"] = 1
                write_alignment_provenance(
                    hdu[0].header,
                    method=method,
                    ref_id=ref_id,
                    success=True,
                )
                hdu.flush()
            log.debug(
                "TweakReg alignment recorded for %s (success=True, method=%s, reference_id=%s)",
                os.path.basename(file),
                alignment_method_token(method),
                normalize_alignment_ref_id(ref_id),
            )

    def copy_wcs_keys(self, from_hdu, to_hdu):
        """
        Copy WCS header keys from one HDU to another.

        Parameters
        ----------
        from_hdu : astropy.io.fits.HDU
            Source HDU (e.g. CRPIX1, CRVAL1, CD matrix, CTYPE).
        to_hdu : astropy.io.fits.HDU
            Target HDU to update.
        """
        for key in [
            "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "CTYPE1", "CTYPE2",
        ]:
            if key in from_hdu.header.keys():
                to_hdu.header[key] = from_hdu.header[key]

    @log_calls
    def run_alignment(
        self,
        obstable,
        reference,
        do_cosmic=True,
        skip_wcs=False,
        search_radius=None,
        update_hdr=True,
    ):
        """
        Run alignment (TweakReg or JHAT) per --align-with; return (message, shift_table).

        Parameters
        ----------
        obstable : astropy.table.Table or dict
            Table with "image" column of paths.
        reference : str
            Reference image path.
        do_cosmic : bool, optional
            Run cosmic-ray rejection on non-reference images. Default True.
        skip_wcs : bool, optional
            Skip WCS update step. Default False.
        search_radius : float, optional
            Override search radius (arcsec). Default from options.
        update_hdr : bool, optional
            Update FITS headers with new WCS. Default True.

        Returns
        -------
        tuple
            (message_str, shift_table or None). e.g. ("tweakreg success", shift_table).
        """
        align_with = getattr(
            self._p.options["args"], "align_with", "tweakreg"
        ).lower()
        log.info(
            "Alignment requested: method=%s pipeline_reference_argument=%r",
            align_with,
            reference,
        )
        if align_with == "jhat":
            result = self.run_jhat_align(obstable, reference)
        else:
            result = self.run_tweakreg(
                obstable,
                reference,
                do_cosmic=do_cosmic,
                skip_wcs=skip_wcs,
                search_radius=search_radius,
                update_hdr=update_hdr,
            )
        msg = result[0] if result else None
        log.info(
            "Alignment step complete: outcome=%r method=%s pipeline_reference_argument=%r",
            msg,
            align_with,
            reference,
        )
        self._primitive_cleanup(
            "run_alignment",
            validation_notes={
                "align_with": align_with,
                "message": msg,
            },
        )
        return result

    def run_jhat_align(self, obstable, reference):
        """
        Run JHAT to align each image in obstable to Gaia.

        Parameters
        ----------
        obstable : astropy.table.Table or dict
            Table with "image" column of paths.
        reference : str
            Reference image (unused; JHAT aligns to Gaia).

        Returns
        -------
        tuple
            (message_str, None) for compatibility with run_tweakreg.
        """
        from hst123.primitives.astrometry.jhat import run_jhat

        p = self._p
        outdir = _resolve_work_dir_chdir(p.options["args"].work_dir)
        params = getattr(settings, "jhat_params", None) or {}
        force_realign = want_redo_astrometry(p.options["args"])
        jhat_ref_id = "GAIA"

        log.info(
            "JHAT setup: method=jhat reference_catalog=%s (unused pipeline ref=%r)",
            jhat_ref_id,
            reference,
        )

        ran_any = False
        for image in obstable["image"]:
            if not os.path.exists(image):
                log.warning("Skipping missing image for JHAT: %s", image)
                continue
            if not force_realign:
                with fits.open(image, mode="readonly") as hdu:
                    if alignment_is_redundant(
                        hdu[0].header,
                        method="jhat",
                        ref_id=jhat_ref_id,
                        require_success=True,
                    ):
                        prov = read_alignment_provenance(hdu[0].header)
                        log.info(
                            "Skipping alignment for %s: already aligned "
                            "(success=True, method=jhat, reference_id=%s)",
                            os.path.basename(image),
                            prov["ref"] if prov else jhat_ref_id,
                        )
                        continue
            log.info(
                "Running JHAT on %s (method=jhat, reference=%s)",
                os.path.basename(image),
                jhat_ref_id,
            )
            try:
                run_jhat(
                    image,
                    outdir=outdir,
                    params=params,
                    gaia=True,
                    verbose=False,
                )
            except Exception as e:
                log.error(
                    "JHAT alignment failed for %s (method=jhat, reference=%s): %s",
                    os.path.basename(image),
                    jhat_ref_id,
                    e,
                )
                self._primitive_cleanup(
                    "run_jhat_align",
                    work_dir=outdir,
                    validation_notes={"status": "failed", "image": str(image)},
                )
                return ("jhat failure", None)
            ran_any = True
            with fits.open(image, mode="update") as hdu:
                hdu[0].header["TWEAKSUC"] = 1
                write_alignment_provenance(
                    hdu[0].header,
                    method="jhat",
                    ref_id=jhat_ref_id,
                    success=True,
                )
                hdu.flush()
            log.info(
                "JHAT alignment succeeded for %s (method=jhat, reference=%s)",
                os.path.basename(image),
                jhat_ref_id,
            )
        if not ran_any:
            log.info(
                "JHAT: no images required alignment (missing files or already aligned to %s).",
                jhat_ref_id,
            )
        vpaths = [
            normalize_fits_path(str(im))
            for im in obstable["image"]
            if os.path.isfile(str(im))
        ]
        self._primitive_cleanup(
            "run_jhat_align",
            work_dir=outdir,
            validate_fits_paths=vpaths,
        )
        return ("jhat success", None)

    @log_calls
    @_quiet_headerlet_loggers
    def run_tweakreg(
        self,
        obstable,
        reference,
        do_cosmic=True,
        skip_wcs=False,
        search_radius=None,
        update_hdr=True,
    ):
        """
        Run TweakReg on images in obstable; return (success, shift_table).

        Parameters
        ----------
        obstable : astropy.table.Table or dict
            Table with "image" column of paths.
        reference : str
            Reference image path.
        do_cosmic : bool, optional
            Run cosmic-ray rejection. Default True.
        skip_wcs : bool, optional
            Skip WCS update. Default False.
        search_radius : float, optional
            Override search radius. Default from options.
        update_hdr : bool, optional
            Update FITS headers. Default True.

        Returns
        -------
        tuple
            (success_bool, shift_table).
        """
        p = self._p
        outdir = _resolve_work_dir_chdir(p.options["args"].work_dir)
        outshifts = os.path.join(outdir, "drizzle_shifts.txt")
        options = p.options["global_defaults"]
        force_realign = want_redo_astrometry(p.options["args"])

        def _tweakreg_cleanup_and_return(result):
            vpaths = [
                normalize_fits_path(str(im))
                for im in obstable["image"]
                if os.path.isfile(str(im))
            ]
            self._primitive_cleanup(
                "run_tweakreg",
                work_dir=outdir,
                remove_globs=(".hst123_calcsky_*.fits",),
                validate_fits_paths=vpaths,
            )
            return result

        initial_ref = None
        if reference and str(reference).strip():
            initial_ref = str(reference).strip()
        run_images = self.check_images_for_tweakreg(
            list(obstable["image"]),
            alignment_method="tweakreg",
            alignment_ref_id=initial_ref,
            force_realign=force_realign,
            tweakreg_reference_images=list(obstable["image"]),
            tweakreg_pipeline_reference=initial_ref,
        )
        if not run_images:
            log.info(
                "TweakReg skipped: no images require alignment "
                "(missing, or already aligned for current method/reference when known)."
            )
            return _tweakreg_cleanup_and_return(("tweakreg success", None))
        if reference in run_images:
            run_images.remove(reference)

        shift_table = Table(
            [run_images, [np.nan] * len(run_images), [np.nan] * len(run_images)],
            names=("file", "xoffset", "yoffset"),
        )

        if not run_images:
            log.warning("All images have been run through tweakreg.")
            return _tweakreg_cleanup_and_return((True, shift_table))

        log.info("Need to run tweakreg for images:")
        p.input_list(obstable["image"], show=True, save=False)

        tmp_images = []
        for image in run_images:
            if p.updatewcs and not skip_wcs:
                det = "_".join(p._fits.get_instrument(image).split("_")[:2])
                wcsoptions = p.options["detector_defaults"][det]
                # Match run_astrodrizzle: skip MAST AstrometryDB here. DB updates call
                # archive_wcs(..., QUIET_ABORT) and emit duplicate-WCSNAME warnings
                # on FLCs that already carry IDC/TWEAK alternates; makecorr still runs.
                p.update_image_wcs(image, wcsoptions, use_db=False)

            if not do_cosmic:
                tmp_images.append(image)
                continue

            if image == reference or "wfc3_ir" in p._fits.get_instrument(image):
                log.info("Skipping adjustments for %s as WFC3/IR or reference", image)
                tmp_images.append(image)
                continue

            rawtmp = image.replace(".fits", ".rawtmp.fits")
            tmp_images.append(rawtmp)
            if os.path.exists(rawtmp):
                log.info("%s exists. Skipping...", rawtmp)
                continue

            shutil.copyfile(image, rawtmp)
            inst = p._fits.get_instrument(image).split("_")[0]
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

        if tweakreg is None:
            log.error("drizzlepac.tweakreg is not installed; skipping alignment")
            return _tweakreg_cleanup_and_return(("tweakreg failure", shift_table))

        log.info("Tweakreg is executing...")
        start_tweak = time.time()

        batches = self._build_tweakreg_batches(tmp_images, reference)
        if not batches:
            log.warning(
                "TweakReg: no filter group with ≥2 images; aligning full list once"
            )
            batches = [(reference, tmp_images)]

        if len(batches) > 1:
            log.info(
                "TweakReg: %d filter batch(es) (same-band reference per batch)",
                len(batches),
            )

        tweakreg_success = True
        thresh_data = None
        max_tries = 6

        for bi, (ref_use, batch_imgs) in enumerate(batches):
            out_this = (
                outshifts
                if len(batches) == 1
                else os.path.join(outdir, "drizzle_shifts_%d.txt" % bi)
            )
            batch_ok = False
            tweak_img = copy.copy(batch_imgs)
            ithresh = p.threshold
            rthresh = p.threshold
            shallow_img = []
            tries = 0

            while not batch_ok and tries < max_tries:
                tweak_img = self.check_images_for_tweakreg(
                    tweak_img,
                    alignment_method="tweakreg",
                    alignment_ref_id=ref_use,
                    force_realign=force_realign,
                )
                if not tweak_img:
                    batch_ok = True
                    break

                if shallow_img:
                    for img in shallow_img:
                        if img in tweak_img:
                            tweak_img.remove(img)

                if len(tweak_img) == 0:
                    log.error("removed all images as shallow in batch %d", bi)
                    tweak_img = copy.copy(batch_imgs)
                    tweak_img = self.check_images_for_tweakreg(
                        tweak_img,
                        alignment_method="tweakreg",
                        alignment_ref_id=ref_use,
                        force_realign=force_realign,
                    )

                success = list(set(batch_imgs) ^ set(tweak_img))
                if tries > 1 and ref_use == "dummy.fits" and len(success) > 0:
                    n = len(success) - 1
                    shutil.copyfile(success[random.randint(0, n)], "dummy.fits")

                log.info(
                    "TweakReg batch %d: ref=%s (reference_id=%s) method=tweakreg n=%d  %s",
                    bi,
                    os.path.basename(ref_use),
                    normalize_alignment_ref_id(ref_use),
                    len(tweak_img),
                    ", ".join(os.path.basename(x) for x in tweak_img[:6])
                    + (" …" if len(tweak_img) > 6 else ""),
                )

                deepest = sorted(tweak_img, key=lambda im: fits.getval(im, "EXPTIME"))[
                    -1
                ]

                if not thresh_data or deepest not in thresh_data["file"]:
                    inp_data = self.get_tweakreg_thresholds(
                        deepest, options["nbright"] * 4
                    )
                    thresh_data = self.add_thresh_data(thresh_data, deepest, inp_data)
                mask = thresh_data["file"] == deepest
                inp_thresh = thresh_data[mask][0]
                log.debug("Image threshold fit for %s", os.path.basename(deepest))
                new_ithresh = self.get_best_tweakreg_threshold(
                    inp_thresh, options["nbright"] * 4
                )

                if not thresh_data or ref_use not in thresh_data["file"]:
                    inp_data = self.get_tweakreg_thresholds(
                        ref_use, options["nbright"] * 4
                    )
                    thresh_data = self.add_thresh_data(thresh_data, ref_use, inp_data)
                mask = thresh_data["file"] == ref_use
                inp_thresh = thresh_data[mask][0]
                log.debug("Reference threshold fit for %s", os.path.basename(ref_use))
                new_rthresh = self.get_best_tweakreg_threshold(
                    inp_thresh, options["nbright"] * 4
                )

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
                    if detkey in p._fits.get_instrument(ref_use):
                        rconv = overrides["conv_width"]
                        break
                for detkey, overrides in trd["detector_overrides"].items():
                    if all(detkey in p._fits.get_instrument(i) for i in tweak_img):
                        iconv = overrides["conv_width"]
                        tol = overrides["tolerance"]
                        break

                if (new_ithresh >= ithresh or new_rthresh >= rthresh) and tries > 1:
                    log.debug(
                        "Relaxing TweakReg thresholds (try %d): image/ref thresh, tol, searchrad",
                        tries,
                    )
                    ithresh = np.max(
                        [new_ithresh * (0.95 ** tries), trd["threshold_min"]]
                    )
                    rthresh = np.max(
                        [new_rthresh * (0.95 ** tries), trd["threshold_min"]]
                    )
                    tol = tol * 1.3 ** tries
                    search_rad = search_rad * 1.2 ** tries
                else:
                    ithresh = new_ithresh
                    rthresh = new_rthresh

                if tries > 4:
                    minobj = trd["minobj_fallback"]

                log.debug(
                    "TweakReg params: ref_thresh=%.4f img_thresh=%.4f tol=%.4f searchrad=%.4f minobj=%s",
                    rthresh,
                    ithresh,
                    tol,
                    search_rad,
                    minobj,
                )

                try:
                    with suppress_stdout():
                        tweakreg.TweakReg(
                            files=tweak_img,
                            refimage=ref_use,
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
                            imagefindcfg={
                                "threshold": ithresh,
                                "conv_width": iconv,
                                "use_sharp_round": True,
                            },
                            refimagefindcfg={
                                "threshold": rthresh,
                                "conv_width": rconv,
                                "use_sharp_round": True,
                            },
                            shiftfile=True,
                            outshifts=out_this,
                        )
                    shallow_img = []

                except AssertionError as e:
                    self.tweakreg_error(e)
                    max_et = max(
                        fits.getval(im, "EXPTIME") for im in tweak_img
                    )
                    shallow_img = [
                        im
                        for im in tweak_img
                        if fits.getval(im, "EXPTIME") < 0.5 * max_et
                    ]
                    if not shallow_img:
                        shallow_img = [
                            sorted(
                                tweak_img,
                                key=lambda im: fits.getval(im, "EXPTIME"),
                            )[0]
                        ]
                    log.debug(
                        "Removing %d shallow exposure(s) from batch (EXPTIME heuristic)",
                        len(shallow_img),
                    )

                except TypeError as e:
                    self.tweakreg_error(e)

                log.debug("Reading shift file: %s", out_this)
                if not os.path.isfile(out_this):
                    log.warning("Missing shift file %s after TweakReg", out_this)
                    tries += 1
                    continue

                shifts = Table.read(
                    out_this,
                    format="ascii",
                    names=(
                        "file",
                        "xoffset",
                        "yoffset",
                        "rotation1",
                        "rotation2",
                        "scale1",
                        "scale2",
                    ),
                )

                _ref_metrics = ref_use if os.path.isfile(ref_use) else deepest
                if not os.path.isfile(_ref_metrics):
                    _ref_metrics = deepest
                log_tweakreg_shift_metrics(
                    shifts,
                    ref_path=os.path.abspath(_ref_metrics),
                    log=log,
                    tolerance_arcsec=tol,
                    batch_index=bi if len(batches) > 1 else None,
                )

                self.apply_tweakreg_success(
                    shifts, ref_id=ref_use, method="tweakreg"
                )

                for row in shifts:
                    filename = os.path.basename(row["file"])
                    filename = filename.replace(".rawtmp.fits", "").replace(
                        ".fits", ""
                    )
                    idx = [
                        i
                        for i, r in enumerate(shift_table)
                        if filename in r["file"]
                    ]
                    if len(idx) == 1:
                        shift_table[idx[0]]["xoffset"] = row["xoffset"]
                        shift_table[idx[0]]["yoffset"] = row["yoffset"]

                if not self.check_images_for_tweakreg(
                    batch_imgs,
                    alignment_method="tweakreg",
                    alignment_ref_id=ref_use,
                    force_realign=force_realign,
                ):
                    batch_ok = True

                tries += 1

            if not batch_ok:
                log.warning(
                    "TweakReg batch %d did not converge after %d tries (method=tweakreg, ref=%s)",
                    bi,
                    max_tries,
                    os.path.basename(ref_use),
                )
                tweakreg_success = False

        log.info("Tweakreg finished in %.2fs", time.time() - start_tweak)
        try:
            _agg = Table(
                [shift_table["xoffset"], shift_table["yoffset"]],
                names=("xoffset", "yoffset"),
            )
            _agg_ref = (
                os.path.abspath(reference)
                if reference and os.path.isfile(reference)
                else os.path.abspath(deepest)
            )
            log_tweakreg_shift_metrics(
                _agg,
                ref_path=_agg_ref,
                log=log,
                tolerance_arcsec=settings.tweakreg_defaults["tolerance"],
                summary_prefix="TweakReg aggregate",
            )
        except Exception as exc:
            log.debug("TweakReg aggregate shift validation: %s", exc)

        log.info(
            "TweakReg alignment summary: success=%s batches=%d method=tweakreg",
            tweakreg_success,
            len(batches),
        )
        log.debug("Shift table: %s", shift_table)
        try:
            parts = []
            for row in shift_table:
                bn = os.path.basename(str(row["file"]))
                parts.append(
                    "%s dx=%s dy=%s" % (bn, row["xoffset"], row["yoffset"])
                )
            log.info("Tweakreg offsets (%d): %s", len(parts), "; ".join(parts[:8]) + (" …" if len(parts) > 8 else ""))
        except Exception:
            log.info("Tweakreg: %d shift row(s)", len(shift_table))

        cleanup_after_tweakreg(
            outdir,
            log=log,
            keep_artifacts=getattr(
                p.options["args"], "keep_drizzle_artifacts", False
            ),
        )

        # Fix CRVAL/CRPIX indexing after tweakreg
        for image in tmp_images:
            rawtmp = image
            rawhdu = fits.open(rawtmp, mode="readonly")
            tweaksuc = (
                "TWEAKSUC" in rawhdu[0].header.keys() and rawhdu[0].header["TWEAKSUC"] == 1
            )
            if "wfc3_ir" in p._fits.get_instrument(image):
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
                if image == reference or "wfc3_ir" in p._fits.get_instrument(image):
                    continue
                rawtmp = image.replace(".fits", ".rawtmp.fits")
                rawhdu = fits.open(rawtmp, mode="readonly")
                hdu = fits.open(image, mode="readonly")
                log.info(
                    "Merge WCS/data %s | %s",
                    os.path.basename(image),
                    format_hdu_list_summary(hdu),
                )
                newhdu = fits.HDUList()

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

                    if "wfpc2" in p._fits.get_instrument(image).lower() and h.name == "WCSCORR":
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

                if "wfpc2" in p._fits.get_instrument(image).lower():
                    newhdu[0].header["NEXTEND"] = 4

                log.debug("After merge: %s", format_hdu_list_summary(newhdu))
                newhdu.writeto(image, output_verify="silentfix", overwrite=True)
                rawhdu.close()
                hdu.close()

                if os.path.isfile(rawtmp) and not p.options["args"].cleanup:
                    os.remove(rawtmp)

        if os.path.isfile("dummy.fits"):
            os.remove("dummy.fits")

        if not p.options["args"].keep_objfile:
            for file in glob.glob("*.coo"):
                os.remove(file)

        if modified:
            p.sanitize_reference(reference)

        return _tweakreg_cleanup_and_return((tweakreg_success, shift_table))
