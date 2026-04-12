#!/usr/bin/env python3
"""
HST pipeline driver: ingest, WCS, AstroDrizzle, DOLPHOT, and catalog scraping.

The public API is the :class:`hst123` instance constructor and its methods,
plus :func:`main` for CLI execution. Implementation is split across helpers in
``hst123.primitives`` (e.g. :class:`~hst123.primitives.FitsHelper`). Changelog and
zeropoint notes live in ``docs/``.

Code map (this module)
----------------------
Rough order of definitions: CLI and options → ingest / MAST → observation table →
reference selection → WCS updates → AstroDrizzle → cosmic-ray rejection → DOLPHOT
wrappers → :func:`main` driver.
"""
import copy
import filecmp
import glob
import logging as py_logging
import os
from concurrent.futures import ThreadPoolExecutor
import random
import shutil
import sys
import time
import uuid
import warnings

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table, Column, unique
from astropy.time import Time
from astropy.utils.data import clear_download_cache
import astropy.wcs as wcs
from astropy.wcs import NoConvergence
from astroscrappy import detect_cosmics
from hst123.utils.stsci_wcs import run_updatewcs

import astroquery
warnings.filterwarnings("ignore")

from hst123 import __version__, settings
from hst123.utils import options
from hst123.utils.logging import (
    ASTRODRIZZLE_DETAIL_LOGGER,
    PHOTEQ_DETAIL_LOGGER,
    attach_work_dir_log_file,
    ensure_cli_logging_configured,
    format_hdu_list_summary,
    get_logger,
    ephemeral_pipeline_runfile,
    ingest_text_file_to_logger,
    log_calls,
    log_pipeline_configuration,
    log_pipeline_phase_summary,
    make_banner,
)
from hst123.utils.stdio import (
    limit_blas_threads_when_parallel,
    suppress_stdout,
    suppress_stdout_fd,
)
from hst123.utils.options import (
    dolphot_catalog_already_present,
    want_redo_astrodrizzle,
    want_redo_dolphot,
)
from hst123.utils.paths import (
    normalize_fits_path,
    normalize_work_and_raw_dirs,
    pipeline_workspace_dir,
)
from hst123.utils.astrodrizzle_paths import (
    astrodrizzle_output_exists,
    logical_driz_to_internal_astrodrizzle,
    normalize_astrodrizzle_output_path,
    recover_drizzlepac_linear_output,
)
from hst123.utils.astrodrizzle_helpers import (
    build_astrodrizzle_keyword_args,
    build_wfpc2_skymask_catalog,
    combine_type_and_nhigh,
    wfpc2_astrodrizzle_scratch_paths,
    drizzle_product_catalog_header,
    drizzle_sidecar_paths,
    remove_internal_linear_drizzle_products,
    rename_astrodrizzle_sidecars,
    resolve_drizzle_clean_flag,
    ensure_wcsname_tweak_on_image,
    wcs_image_hdu_index,
    write_drc_multis_extension_if_requested,
)
from hst123.utils.workdir_cleanup import (
    cleanup_after_astrodrizzle,
    remove_files_matching_globs,
    remove_superseded_instrument_mask_reference_drizzle,
)
from hst123.utils.reference_download import (
    fetch_calibration_reference,
    ref_prefix_for_header,
)
from hst123.utils.visit import add_visit_info as add_visit_info_util
from hst123.utils.wcs_utils import (
    fix_sip_ctype_headers_fits,
    make_meta_wcs_header as make_meta_wcs_header_util,
    remove_conflicting_alt_wcs_duplicate_names,
    wcs_from_fits_hdu,
)
from hst123.primitives import FitsHelper, PhotometryHelper
from hst123.primitives.astrometry import AstrometryPrimitive, parse_coord
from hst123.primitives.run_dolphot import DolphotPrimitive
from hst123.primitives.scrape_dolphot import ScrapeDolphotPrimitive

log = get_logger(__name__)

with suppress_stdout():
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Observations
    from drizzlepac import tweakreg, astrodrizzle, catalogs, photeq


class hst123(object):
  """
  End-to-end HST reduction and photometry pipeline.

  Construct once, then call :meth:`handle_args` (or set ``self.options['args']``
  programmatically) before invoking download, alignment, drizzle, or DOLPHOT steps.

  Attributes
  ----------
  input_images : list of str
      Paths to science FITS after ingest.
  obstable : astropy.table.Table or None
      Per-exposure metadata table from :meth:`input_list`.
  reference : str
      Path to the drizzled reference image when set.
  options : dict
      ``global_defaults``, ``detector_defaults``, ``instrument_defaults``,
      ``acceptable_filters``, ``drizzle_defaults``, and ``args`` (CLI namespace).
  coord : astropy.coordinates.SkyCoord or None
      Target position from the command line.

  See Also
  --------
  hst123.main : CLI entry point.
  """

  def __init__(self):
    """
    Initialize pipeline state, load :mod:`hst123.settings`, and attach primitives.

    Primitives (``_fits``, ``_phot``, ``_astrom``, ``_dolphot``, ``_scrape_dolphot``)
    carry instrument-specific logic.
    """
    self.input_images = []
    self.split_images = []
    self.fake_images = []
    self.obstable = None

    self.reference = ''
    self.root_dir = '.'
    gd = settings.global_defaults
    self.rawdir = gd['rawdir']

    self.usagestring = 'hst123.py ra dec'
    self.command = ''

    self.before = None
    self.after = None
    self.coord = None

    self.productlist = None

    self.cleanup = False
    self.updatewcs = True
    self.archive = False
    self.keep_objfile = False

    self.magsystem = gd['magsystem']

    # Detection threshold used for image alignment by tweakreg
    self.threshold = gd['default_threshold']

    # S/N limit for calculating limiting magnitude
    self.snr_limit = gd['snr_limit']

    self.dolphot = {}

    # Names for input image table
    self.names = settings.names
    # Names for the final output photometry table
    final_names = settings.final_names

    # Make an empty table with above column names for output photometry table
    self.final_phot = Table([[0.],['INSTRUMENT'],['FILTER'],[0.],[0.],[0.]],
        names=final_names)[:0].copy()

    # List of options (drizzle_defaults is a copy so CLI can set num_cores per run)
    self.options = {
        "global_defaults": settings.global_defaults,
        "detector_defaults": settings.detector_defaults,
        "instrument_defaults": settings.instrument_defaults,
        "acceptable_filters": settings.acceptable_filters,
        "catalog": settings.catalog_pars,
        "drizzle_defaults": copy.deepcopy(settings.drizzle_defaults),
        "args": None,
    }

    # List of pipeline products in case they need to be cleaned at start
    self.pipeline_products = settings.pipeline_products
    self.pipeline_images = settings.pipeline_images

    # Helpers from primitives/ (logic split by responsibility)
    self._fits = FitsHelper(self)
    self._phot = PhotometryHelper(self)
    self._astrom = AstrometryPrimitive(self)
    self._dolphot = DolphotPrimitive(self)
    self._scrape_dolphot = ScrapeDolphotPrimitive(self)
    log.debug(
        "Pipeline ready (FitsHelper, PhotometryHelper, astrometry, DOLPHOT, scrape)."
    )

  def add_options(self, parser=None, usage=None):
    """
    Add hst123 command-line arguments to an argparse parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser, optional
        Parser to add arguments to; if None, a new parser is created.
    usage : str, optional
        Usage string when parser is None.

    Returns
    -------
    argparse.ArgumentParser
        Parser with all hst123 options registered.
    """
    return options.add_options(parser=parser, usage=usage, version=__version__)

  # ---------------------------------------------------------------------------
  # Cache, MAST staging, and archive recovery
  # ---------------------------------------------------------------------------

  @log_calls
  def clear_downloads(self, options):
    """
    Clear Astropy's download cache to avoid failures when the cache is full.

    Parameters
    ----------
    options : dict
        Must include ``astropath`` (suffix under ``$HOME`` for the cache location).

    Notes
    -----
    No-op when ``--no-clear-downloads`` is set.
    """
    if self.options['args'].no_clear_downloads:
      return None
    log.debug('clear_downloads: astropy cache')
    try:
        if "HOME" in os.environ:
            astropath = options['astropath']
            astropy_cache = os.environ['HOME'] + astropath
            log.debug('clear_download_cache %s', astropy_cache)
            if os.path.exists(astropy_cache):
                with suppress_stdout():
                    clear_download_cache()
    except RuntimeError:
        log.warning("Runtime error in clear_download_cache(); continuing.")
    except FileNotFoundError:
        log.warning("Cannot run full clear_download_cache(); continuing.")

  @log_calls
  def try_to_get_image(self, image):
    """
    Recover a missing exposure from the local archive layout (``--archive``).

    Parameters
    ----------
    image : str
        Destination path for the FITS file.

    Returns
    -------
    bool
        True if a file was copied from the archive, False otherwise.
    """
    if not self.coord:
        return False

    dest = normalize_fits_path(image)
    base = os.path.basename(image).lower()

    data = {'productFilename': os.path.basename(image), 'ra': self.coord.ra.degree}

    inst = ''
    if base.startswith('i') and base.endswith('flc.fits'):
        inst = 'WFC3/UVIS'
    elif base.startswith('i') and base.endswith('flt.fits'):
        inst = 'WFC3/IR'
    elif base.startswith('j') and base.endswith('flt.fits'):
        inst = 'ACS/HRC'
    elif base.startswith('j') and base.endswith('flc.fits'):
        inst = 'ACS/WFC'
    elif base.startswith('u'):
        inst = 'WFPC2'
    else:
        return False

    data["instrument_name"] = inst

    success, fullfile = self.check_archive(data)

    if success:
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copyfile(fullfile, dest)
        return True
    return False

  # ---------------------------------------------------------------------------
  # Observation table: per-image metadata, visits, drizname, optional listing file
  # ---------------------------------------------------------------------------

  def input_list(self, img, show=True, save=False, file=None, image_number=[]):
    """
    Build the observation metadata table for a list of FITS paths.

    Parameters
    ----------
    img : list of str
        Input image paths; missing files may be filled via :meth:`try_to_get_image`.
    show : bool, optional
        If True, log a one-line summary of inputs (and a debug table).
    save : bool, optional
        If True, store the table on ``self.obstable``.
    file : str, optional
        If set, write a formatted text summary under ``work_dir``.
    image_number : list, optional
        Optional per-row image indices; default is zeros.

    Returns
    -------
    astropy.table.Table or None
        Columns match :mod:`hst123.settings` ``names`` (image, exptime, datetime,
        filter, instrument, detector, zeropoint, chip, ...). Returns None if no
        valid inputs remain.
    """
    zptype = self.magsystem

    good = []
    for image in img:
        success = True
        if not os.path.exists(image):
            success = self.try_to_get_image(image)
        if success:
            good.append(image)

    if not good:
        return None
    else:
        img = copy.copy(good)

    img = [normalize_fits_path(p) for p in img]
    for p in img:
        if not os.path.isfile(p):
            log.warning("Dropping missing or unreadable FITS from observation table: %s", p)
    img = [p for p in img if os.path.isfile(p)]
    if not img:
        log.error("No input FITS paths exist on disk after resolve (missing files?).")
        return None

    hdu = fits.open(img[0])
    h = hdu[0].header

    exp = [fits.getval(image,'EXPTIME') for image in img]
    if 'DATE-OBS' in h.keys() and 'TIME-OBS' in h.keys():
        dat = [fits.getval(image,'DATE-OBS') + 'T' +
               fits.getval(image,'TIME-OBS') for image in img]
    # This should work if image is missing DATE-OBS, e.g., for drz images
    elif 'EXPSTART' in h.keys():
        dat = [Time(fits.getval(image, 'EXPSTART'),
            format='mjd').datetime.strftime('%Y-%m-%dT%H:%M:%S')
            for image in img]
    fil = [self._fits.get_filter(image) for image in img]
    ins = [self._fits.get_instrument(image) for image in img]
    det = ['_'.join(self._fits.get_instrument(image).split('_')[:2]) for image in img]
    chip = [self._fits.get_chip(image) for image in img]
    zpt = [self._fits.get_zpt(i, ccdchip=c, zptype=zptype) for i, c in zip(img, chip)]

    if not image_number:
        image_number = [0 for image in img]

    obstable = Table([img,exp,dat,fil,ins,det,zpt,chip,image_number],
        names=self.names)

    obstable.sort('datetime')

    obstable = add_visit_info_util(
        obstable, self.options["global_defaults"]["visit"], log=log
    )

    # One compact line (avoid N+1 INFO lines per input_list call)
    form = '{file: <36} {inst: <18} {filt: <10} '
    form += '{exp: <12} {date: <10} {time: <10}'
    if show:
        bits = [
            "%s|%s/%s"
            % (
                os.path.basename(row["image"]),
                row["instrument"].upper(),
                row["filter"].upper(),
            )
            for row in obstable
        ]
        log.info("inputs n=%d: %s", len(obstable), " ".join(bits))
        log.debug(
            "inputs table:\n%s",
            "\n".join(
                form.format(
                    file=os.path.basename(row["image"]),
                    inst=row["instrument"].upper(),
                    filt=row["filter"].upper(),
                    exp="%7.4f" % row["exptime"],
                    date=Time(row["datetime"]).datetime.strftime("%Y-%m-%d"),
                    time=Time(row["datetime"]).datetime.strftime("%H:%M:%S"),
                )
                for row in obstable
            ),
        )

    # Iterate over visit, instrument, filter to add group-specific info.
    # Use object dtype so full paths are never truncated (fixed-width U99
    # previously cut long work_dir + basename, breaking .drz.fits handling).
    obstable.add_column(
        Column(np.empty(len(obstable), dtype=object), name="drizname")
    )
    for i,row in enumerate(obstable):
        visit = row['visit']
        n = str(visit).zfill(4)
        inst = row['instrument']
        filt = row['filter']

        # Visit should correspond to first image so they're all the same
        visittable = obstable[obstable['visit']==visit]
        refimage = visittable['image'][0]
        if 'DATE-OBS' in h.keys():
            date_obj = Time(fits.getval(refimage, 'DATE-OBS'))
        else:
            date_obj = Time(fits.getval(refimage, 'EXPSTART'), format='mjd')
        date_str = date_obj.datetime.strftime('%y%m%d')

        # Make a photpipe-like image name
        drizname = ''
        objname = self.options['args'].object
        if objname:
            drizname = '{obj}.{inst}.{filt}.ut{date}_{n}.drc.fits'
            drizname = drizname.format(inst=inst.split('_')[0],
                filt=filt, n=n, date=date_str, obj=objname)
        else:
            drizname = '{inst}.{filt}.ut{date}_{n}.drc.fits'
            drizname = drizname.format(inst=inst.split('_')[0],
                filt=filt, n=n, date=date_str)

        if self.options["args"].work_dir:
            wd_base = os.path.abspath(
                os.path.expanduser(self.options["args"].work_dir)
            )
            driz_root = pipeline_workspace_dir(wd_base) or wd_base
            # Per-epoch drizzle products live under workspace/; --drizzle-all
            # consolidated outputs use <work-dir>/drizzle/ (base, not workspace/).
            if self.options["args"].drizzle_all:
                drizzle_dir = os.path.join(wd_base, "drizzle")
                os.makedirs(drizzle_dir, exist_ok=True)
                drizname = os.path.join(drizzle_dir, drizname)
            else:
                drizname = os.path.join(driz_root, drizname)

        obstable[i]['drizname'] = drizname

    if len(obstable):
        log.debug(
            "drizname paths assigned (%d row(s)); example: %s",
            len(obstable),
            obstable[0]["drizname"],
        )

    if file:

        form = '{inst: <10} {filt: <10} {exp: <12} {date: <16}'
        header = form.format(inst='INSTRUMENT', filt='FILTER', exp='EXPTIME',
            date='DATE')

        if self.options['args'].work_dir:
            file = os.path.join(self.options['args'].work_dir, file)

        outfile = open(file, 'w')
        outfile.write(header+'\n')

        for visit in list(set(obstable['visit'])):
            visittable = obstable[obstable['visit'] == visit]
            for inst in list(set(visittable['instrument'])):
                insttable = visittable[visittable['instrument'] == inst]
                for filt in list(set(insttable['filter'])):
                    ftable = insttable[insttable['filter'] == filt]

                    # Format instrument name
                    if 'wfc3' in inst and 'uvis' in inst: instname='WFC3/UVIS'
                    if 'wfc3' in inst and 'ir' in inst: instname='WFC3/IR'
                    if 'acs' in inst and 'wfc' in inst: instname='ACS/WFC'
                    if 'acs' in inst and 'hrc' in inst: instname='ACS/HRC'
                    if 'acs' in inst and 'sbc' in inst: instname='ACS/SBC'
                    if 'wfpc2' in inst: instname='WFPC2'
                    if '_' in instname:
                        instname=instname.upper()
                        instname=instname.replace('_full','').replace('_','/')

                    mjd = [Time(r['datetime']).mjd for r in ftable]
                    time = Time(np.mean(mjd), format='mjd')

                    exptime = np.sum(ftable['exptime'])

                    date_decimal='%1.5f'% (time.mjd % 1.0)
                    date = time.datetime.strftime('%Y-%m-%d')
                    date += date_decimal[1:]

                    line=form.format(date=date, inst=instname,
                        filt=filt.upper(), exp='%7.4f' % exptime)
                    outfile.write(line+'\n')

        outfile.close()

    # Save as the primary obstable for this reduction?
    if save:
        self.obstable = obstable

    return obstable

  # ---------------------------------------------------------------------------
  # Sync between raw cache and work_dir (see also copy_raw_data_archive)
  # ---------------------------------------------------------------------------

  @log_calls
  def copy_raw_data(self, rawdir, reverse=False, check_for_coord=False):
    """
    Sync FITS between a raw directory and the working directory.

    Parameters
    ----------
    rawdir : str
        Directory containing ``*.fits`` (used as the source when *reverse* is True).
    reverse : bool, optional
        If False, copy pipeline ``input_images`` into *rawdir*. If True, copy from
        *rawdir* into ``work_dir``, keeping a copy under ``work_dir/raw/`` and a
        symlink (or copy) at the work-dir top level.
    check_for_coord : bool, optional
        If True with *reverse*, only copy exposures that pass :meth:`needs_to_be_reduced`.

    Notes
    -----
    Used after downloads or when ingesting an existing raw cache.
    """
    if not reverse:
        if not os.path.exists(rawdir):
            os.mkdir(rawdir)
        for f in self.input_images:
            if not os.path.isfile(rawdir+'/'+f):
                # Create new file and change permissions
                shutil.copyfile(f, rawdir+'/'+f)
    else:
        dest_dir = "."
        if (
            getattr(self, "options", None)
            and self.options.get("args") is not None
            and self.options["args"].work_dir
        ):
            dest_dir = self.options["args"].work_dir
        dest_abs = os.path.abspath(dest_dir)
        os.makedirs(dest_abs, exist_ok=True)
        raw_sub = os.path.join(dest_abs, "raw")
        os.makedirs(raw_sub, exist_ok=True)
        ws_dir = pipeline_workspace_dir(dest_abs)
        if ws_dir:
            os.makedirs(ws_dir, exist_ok=True)
        for file in glob.glob(os.path.join(rawdir, "*.fits")):
            if check_for_coord:
                warning, check = self.needs_to_be_reduced(file, save_c1m=True)
                if not check:
                    log.warning(warning)
                    continue
            base = os.path.basename(file)
            real_tgt = os.path.join(raw_sub, base)
            dest = (
                os.path.join(ws_dir, base)
                if ws_dir
                else os.path.join(dest_abs, base)
            )
            if not os.path.isfile(real_tgt):
                log.info("%s -> %s (under raw/)", file, real_tgt)
                shutil.copyfile(file, real_tgt)
            elif not filecmp.cmp(file, real_tgt):
                log.info("%s -> %s (refresh under raw/)", file, real_tgt)
                shutil.copyfile(file, real_tgt)
            if os.path.lexists(dest):
                try:
                    if os.path.islink(dest) or os.path.isfile(dest):
                        if os.path.samefile(dest, real_tgt):
                            log.info("%s == %s", real_tgt, dest)
                            continue
                except OSError:
                    pass
                try:
                    os.remove(dest)
                except OSError:
                    pass
            try:
                rel = os.path.relpath(real_tgt, os.path.dirname(dest))
                os.symlink(rel, dest)
                log.info("Symlink %s -> %s", dest, rel)
            except OSError:
                log.warning(
                    "Could not symlink %s -> raw/%s; copying file to work dir",
                    dest,
                    base,
                )
                shutil.copyfile(real_tgt, dest)

  def _archive_path_for_product(self, product, archivedir):
    """Return full path (archivedir/inst/det/ra/name) for a product row."""
    filefmt = '{inst}/{det}/{ra}/{name}'
    filename = '_'.join(product['productFilename'].split('_')[-2:])
    instrument = product['instrument_name']
    ra = str(int(np.round(product['ra'])))
    if 'WFPC2' in instrument.upper() or 'PC/WFC' in instrument.upper():
        inst, det = 'WFPC2', 'WFPC2'
    else:
        inst, det = instrument.split('/')
    return archivedir + '/' + filefmt.format(inst=inst, det=det, ra=ra, name=filename)

  @log_calls
  def check_archive(self, product, archivedir=None):
    """
    Ensure the archive tree exists and report whether the product file is present.

    Parameters
    ----------
    product : dict-like
        Row with ``productFilename``, ``instrument_name``, ``ra``, etc.
    archivedir : str, optional
        Root archive directory; defaults to ``args.archive``.

    Returns
    -------
    tuple (bool, str)
        ``(file_exists, full_path)`` where *full_path* is the target archive path.
    """
    if not archivedir:
        archivedir = self.options['args'].archive
    if not os.path.exists(archivedir):
        try:
            os.makedirs(archivedir)
        except OSError:
            error = (
                'Could not make archive dir {dir}\n'
                'Enable write permissions to this location\n'
                'Exiting...'
            )
            log.error(error.format(dir=archivedir))
            return False, None

    fullfile = self._archive_path_for_product(product, archivedir)
    path, basefile = os.path.split(fullfile)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            error = (
                'Could not make archive dir {0}\n'
                'Enable write permissions to this location\n'
                'Exiting...'
            )
            log.error(error.format(path))
            return False, None

        return False, fullfile

    else:
        if os.path.exists(fullfile):
            return True, fullfile
        else:
            return False, fullfile

  @log_calls
  def copy_raw_data_archive(self, product, archivedir=None, workdir=None,
    check_for_coord = False):
    """
    Copy one MAST product from the archive into the working directory.

    Parameters
    ----------
    product : dict-like
        Row from :meth:`get_productlist` (used with :meth:`_archive_path_for_product`).
    archivedir, workdir : str, optional
        Archive root and destination directory (defaults from ``args``).
    check_for_coord : bool, optional
        If True, skip copying when :meth:`needs_to_be_reduced` rejects the file.

    Returns
    -------
    int or None
        ``0`` on success or skip-when-identical; None if missing or filtered out.
    """
    if not archivedir:
        archivedir = self.options['args'].archive
    if not os.path.exists(archivedir):
        log.warning("Archive directory does not exist: %s", archivedir)
        return None

    fullfile = self._archive_path_for_product(product, archivedir)
    path, basefile = os.path.split(fullfile)

    if not os.path.exists(fullfile):
        log.warning("Archive file not found: %s", fullfile)
        return None
    else:
        if check_for_coord:
            warning, check = self.needs_to_be_reduced(fullfile, save_c1m=True)
            if not check:
                log.warning(warning)
                return None
        if workdir:
            fulloutfile = os.path.join(workdir, basefile)
        else:
            fulloutfile = basefile

        # Check whether fulloutfile exists and if files are the same
        if os.path.exists(fulloutfile) and filecmp.cmp(fullfile, fulloutfile):
            message = '{file} == {base}'
            log.info(message.format(file=fullfile,base=fulloutfile))
            return 0
        else:
            message = '{file} != {base}'
            log.info(message.format(file=fullfile,base=fulloutfile))
            shutil.copyfile(fullfile, fulloutfile)
            return 0

  # ---------------------------------------------------------------------------
  # Reference image products for DOLPHOT (sanitize, compress, WFPC2 layout)
  # ---------------------------------------------------------------------------

  @log_calls
  def sanitize_reference(self, reference):
    """
    Normalize a drizzled reference FITS for DOLPHOT (single science HDU, headers).

    Promotes SCI data to PRIMARY when needed, strips COMMENT/HISTORY, applies
    mask-based NaN/median masking, and fills detector calibration keywords.

    Parameters
    ----------
    reference : str
        Path to the reference image (updated in place).

    Returns
    -------
    None
        Returns None if the file is missing.
    """
    if not os.path.exists(reference):
        error = 'Reference {ref} does not exist!'
        log.error(error.format(ref=reference))
        return None

    hdu = fits.open(reference, mode='readonly')

    newhdu = fits.HDUList()

    hdr = hdu[0].header
    newhdu.append(hdu[0])
    newhdu[0].name='PRIMARY'

    if newhdu[0].data is None and len(hdu) > 1:
        ext1 = hdu[1]
        ext1_name = str(ext1.name).strip().upper()
        d1 = getattr(ext1, "data", None)
        ndim = int(getattr(d1, "ndim", 0) or 0)
        if ext1_name == "SCI" and ndim >= 2 and d1 is not None:
            newhdu[0].data = ext1.data

    if 'COMMENT' in newhdu[0].header.keys():
        del newhdu[0].header['COMMENT']
    if 'HISTORY' in newhdu['PRIMARY'].header.keys():
        del newhdu[0].header['HISTORY']

    newhdu[0].header['EXTEND']=False

    inst = newhdu[0].header['INSTRUME'].lower()
    opt  = self.options['instrument_defaults'][inst]['crpars']
    for key in ['saturate','rdnoise','gain']:
        if key not in newhdu[0].header.keys():
            newhdu[0].header[key.upper()] = opt[key]

    maskfile = reference.replace('.fits', '.mask.fits')
    if os.path.exists(maskfile):
        maskhdu = fits.open(maskfile)
        mask = maskhdu[0].data
        if (mask is not None and len(mask.shape)>1):
            if len(mask.shape)==3:
                newmask = np.ones((mask.shape[1], mask.shape[2]), dtype=bool)
                for j in np.arange(mask.shape[0]):
                    newmask = (newmask) & (mask[j,:,:]==0)
            else:
                newmask = mask == 0

            minmask = newhdu[0].data < -5000.0
            newmask = newmask | minmask

            if not self.options['args'].no_nan:
                newhdu[0].data[newmask] = float('NaN')
            # Otherwise set to median pixel value
            else:
                medpix = np.median(newhdu[0].data[~newmask])
                newhdu[0].data[newmask] = medpix

    newhdu[0].header['SANITIZE']=1

    newhdu.writeto(reference, output_verify='silentfix', overwrite=True)

  @log_calls
  def compress_reference(self, reference):
    """
    Collapse multi-extension drizzle products to a single PRIMARY for DOLPHOT.

    AstroDrizzle often writes **PRIMARY (image) + HDRTAB**. Older logic treated
    any 2-HDU file as PRIMARY+SCI and replaced PRIMARY with extension 1, which
    swaps the science image for a binary table and breaks ``calcsky`` (crash /
    SIGTRAP). Only **SCI** is promoted when the primary HDU has no image array.

    Parameters
    ----------
    reference : str
        Path to the drizzled FITS file (updated in place).

    Returns
    -------
    None
        Returns None if *reference* does not exist.
    """
    if not os.path.exists(reference):
        error = "Reference {ref} does not exist!"
        log.error(error.format(ref=reference))
        return None

    hdu = fits.open(reference)
    try:
        prim = hdu[0]
        naxis = int(prim.header.get("NAXIS", 0) or 0)
        prim_has_image = naxis > 0 and prim.data is not None

        newhdu = fits.HDUList()
        if len(hdu) == 1:
            newhdu.append(copy.copy(prim))
        elif len(hdu) == 2:
            ext1 = hdu[1]
            ext1_name = str(ext1.name).strip().upper()
            if ext1_name == "SCI" and not prim_has_image:
                sci = copy.copy(ext1)
                sci.name = "PRIMARY"
                for key in prim.header.keys():
                    if key not in sci.header.keys():
                        sci.header[key] = prim.header[key]
                newhdu.append(sci)
                log.debug(
                    "compress_reference: promoted SCI to PRIMARY for %s",
                    reference,
                )
            else:
                newhdu.append(copy.copy(prim))
                log.debug(
                    "compress_reference: kept PRIMARY (%s + %s) for %s",
                    prim.name,
                    ext1.name,
                    reference,
                )
        else:
            # MEF drizzle (e.g. .drc: PRIMARY + SCI + WHT + CTX): promote SCI only.
            ext1 = hdu[1] if len(hdu) > 1 else None
            ext1_name = str(ext1.name).strip().upper() if ext1 is not None else ""
            if ext1_name == "SCI" and not prim_has_image:
                sci = copy.copy(ext1)
                sci.name = "PRIMARY"
                for key in prim.header.keys():
                    if key not in sci.header.keys():
                        sci.header[key] = prim.header[key]
                newhdu.append(sci)
                log.debug(
                    "compress_reference: promoted SCI to PRIMARY (MEF, %d ext) for %s",
                    len(hdu),
                    reference,
                )
            else:
                newhdu.append(copy.copy(prim))
                if len(hdu) > 2:
                    log.debug(
                        "compress_reference: kept first HDU only (%d extensions) for %s",
                        len(hdu),
                        reference,
                    )

        newhdu[0].name = "PRIMARY"
        newhdu.writeto(reference, output_verify="silentfix", overwrite=True)
    finally:
        hdu.close()


  def sanitize_wfpc2(self, image):
    """
    Rewrite a WFPC2 MEF so PRIMARY and SCI extensions match DOLPHOT expectations.

    Parameters
    ----------
    image : str
        Path to a WFPC2 ``*_c0m.fits`` file (modified in place).
    """
    hdu = fits.open(image, mode='readonly')
    newhdu = fits.HDUList()
    newhdu.append(hdu['PRIMARY'])

    n = len([h.name for h in hdu if h.name == 'SCI'])
    newhdu['PRIMARY'].header['NEXTEND'] = n

    for h in hdu:
        if h.name == 'SCI':
            newhdu.append(h)

    newhdu.writeto(image, output_verify='silentfix', overwrite=True)

  # ---------------------------------------------------------------------------
  # Input validation: quality, filters, dates, coordinate in field
  # ---------------------------------------------------------------------------

  def needs_to_be_reduced(self, image, save_c1m=False):
    """
    Decide whether an exposure should enter the reduction pipeline.

    Parameters
    ----------
    image : str
        Path to the FITS file.
    save_c1m : bool, optional
        If True, allow WFPC2 ``*_c1m.fits`` through the instrument-shape checks.

    Returns
    -------
    warning : str
        Human-readable reason when the file is rejected (may be empty when accepted).
    accept : bool
        True if the file passes quality, filter, date, and coordinate checks.

    Notes
    -----
    Applies ``--before`` / ``--after``, ``EXPFLAG``, exposure time, filter allow-list,
    and optional on-sky containment tests for ``self.coord``.
    """
    keep_short = self.options['args'].keep_short
    keep_tdf_down = self.options['args'].keep_tdf_down
    keep_indt = self.options['args'].keep_indt

    if not os.path.exists(image):
        success = self.try_to_get_image(image)
        if not success:
            return '{image} does not exist'.format(image=image), False

    try:
        hdu = fits.open(image, mode='readonly')
        check = False
        for h in hdu:
            if h.data is not None and h.name.upper()=='SCI':
                check = True
    except (OSError, TypeError, AttributeError):
        msg = '{img} is empty or corrupted. Trying to download again...'
        log.warning(msg.format(img=image))

        success = False
        if not self.productlist:
            return 'could not find or download {img}'.format(img=image), False

        mask = self.productlist['productFilename']==image
        if self.productlist[mask]==0:
            return 'could not find or download {img}'.format(img=image), False

        self.download_files(
            self.productlist,
            archivedir=self.options["args"].archive,
            clobber=True,
            work_dir=self.options["args"].work_dir,
        )

        for product in self.productlist[mask]:
            self.copy_raw_data_archive(product, 
                archivedir=self.options['args'].archive,
                workdir=self.options['args'].work_dir, 
                check_for_coord=True)

        if os.path.exists(image):
            try:
                hdu = fits.open(image, mode='readonly')
                check = False
                for h in hdu:
                    if h.data is not None and h.name.upper()=='SCI':
                        check = True
            except (OSError, TypeError, AttributeError):
                return 'could not find or download {img}'.format(img=image), False

    is_not_hst_image = False
    warning = ''
    detector = ''

    # Check for header keys that we need
    for key in ['INSTRUME','EXPTIME','DATE-OBS','TIME-OBS']:
        if key not in hdu[0].header.keys():
            msg = '{key} not in {img} header'
            return msg.format(key=key, img=image), False

    instrument = hdu[0].header['INSTRUME'].lower()
    if 'c1m.fits' in image and not save_c1m:
        # We need the c1m.fits files, but they aren't reduced as science data
        return 'do not need to reduce c1m.fits files.', False

    if ('DETECTOR' in hdu[0].header.keys()):
        detector = hdu[0].header['DETECTOR'].lower()

    # Check for EXPFLAG=='INDETERMINATE', usually indicating a bad exposure
    if 'EXPFLAG' in hdu[0].header.keys():
        flag = hdu[0].header['EXPFLAG']
        if flag=='INDETERMINATE':
            if not keep_indt:
                return f'{image} has EXPFLAG==INDETERMINATE', False
        elif 'TDF-DOWN' in flag:
            if not keep_tdf_down:
                return f'{image} has EXPFLAG==TDF-DOWN AT EXPSTART', False
        elif flag!='NORMAL':
            return f'{image} has EXPFLAG=={flag}.', False

    # Get rid of exposures with exptime < 20s
    if not keep_short:
        exptime = hdu[0].header['EXPTIME']
        if (exptime < 15):
            return f'{image} EXPTIME is {exptime} < 20.', False

    # Now check date and compare to self.before
    mjd_obs = Time(hdu[0].header['DATE-OBS']+'T'+hdu[0].header['TIME-OBS']).mjd
    if self.before is not None:
        mjd_before = Time(self.before).mjd
        dbefore = self.before.strftime('%Y-%m-%d')
        if mjd_obs > mjd_before:
            return f'{image} after the input before date {dbefore}.', False

    # Same with self.after
    if self.after is not None:
        mjd_after = Time(self.after).mjd
        dafter = self.after.strftime('%Y-%m-%d')
        if mjd_obs < mjd_after:
            return f'{image} before the input after date {dafter}.', False

    # Get rid of data where input coordinate not in any extension
    if self.coord:
        for h in hdu:
            if h.data is not None and h.name.upper()=='SCI':
                # This method doesn't need to be very precise and fails if
                # certain variables (e.g., distortion terms) are missing, so
                # construct a very basic dummy header with base terms
                dummy_header = {'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                        'CRPIX1': h.header['CRPIX1'],
                        'CRPIX2': h.header['CRPIX2'],
                        'CRVAL1': h.header['CRVAL1'],
                        'CRVAL2': h.header['CRVAL2'],
                        'CD1_1': h.header['CD1_1'], 'CD1_2': h.header['CD1_2'],
                        'CD2_1': h.header['CD2_1'], 'CD2_2': h.header['CD2_2']}
                w = wcs.WCS(dummy_header)
                # This can be rough and sometimes WCS will choke on images
                # if mode='all', so using mode='wcs' (only core WCS)
                x,y = wcs.utils.skycoord_to_pixel(self.coord, w,
                                                      origin=1, mode='wcs')
                if (x > 0 and y > 0 and
                    x < h.header['NAXIS1'] and y < h.header['NAXIS2']):
                    is_not_hst_image = True

        if not is_not_hst_image:
            ra = self.coord.ra.degree
            dec = self.coord.dec.degree
            return f'{image} does not contain: {ra} {dec}', False

    filt = self._fits.get_filter(image).upper()
    if not (filt in self.options['acceptable_filters']):
        msg = (
            f'{image} with FILTER={filt} does not have an acceptable filter.'
        )
        return msg, False

    # Get rid of images that don't match one of the allowed instrument/detector
    # types and images whose extensions don't match the allowed type for those
    # instrument/detector types
    is_not_hst_image = False
    nextend = hdu[0].header['NEXTEND']
    warning = (
        f'{image} with INSTRUME={instrument}, '
        f'DETECTOR={detector}, NEXTEND={nextend} is bad.'
    )
    if (instrument.upper() == 'WFPC2' and 'c0m.fits' in image and nextend==4):
        is_not_hst_image = True
    if (instrument.upper() == 'ACS' and
        detector.upper() == 'WFC' and 'flc.fits' in image):
        is_not_hst_image = True
    if (instrument.upper() == 'ACS' and
        detector.upper() == 'HRC' and 'flt.fits' in image):
        is_not_hst_image = True
    if (instrument.upper() == 'WFC3' and
        detector.upper() == 'UVIS' and 'flc.fits' in image):
        is_not_hst_image = True
    if (instrument.upper() == 'WFC3' and
        detector.upper() == 'IR' and 'flt.fits' in image):
        is_not_hst_image = True
    if save_c1m:
        if (instrument.upper() == 'WFPC2' and 'c1m.fits' in image):
            is_not_hst_image = True

    return warning, is_not_hst_image

  # ---------------------------------------------------------------------------
  # DOLPHOT chip images: quick on-sky containment for split groups
  # ---------------------------------------------------------------------------

  def split_image_contains(self, image, coord):
    """
    Return whether a DOLPHOT chip image's WCS encloses the target sky position.

    Uses linear WCS only (``mode='wcs'``) for robustness near chip edges.

    Parameters
    ----------
    image : str
        Path to a single-HDU chip file (e.g. ``*_c0m.chip1.fits``).
    coord : astropy.coordinates.SkyCoord
        Target position.

    Returns
    -------
    bool
        True if the projected pixel coordinates lie inside the image bounds.
    """
    log.info('Analyzing split image: %s', image)
    with fits.open(image) as hdul:
        prim = hdul[0]
        if prim.data is None:
            log.warning(
                "split_image_contains: no data in primary HDU of %s", image
            )
            return False
        # fobj=hdul: required when primary WCS references CPDIS / D2IM lookup
        # tables (ACS/WFC FLC split chips); WCS(header) alone raises ValueError.
        w = wcs_from_fits_hdu(hdul, 0)

        try:
            y, x = wcs.utils.skycoord_to_pixel(
                coord, w, origin=1, mode="wcs"
            )
        except (NoConvergence, TypeError, ValueError) as exc:
            log.warning(
                "split_image_contains: WCS sky->pixel failed for %s (%s); "
                "treating chip as not containing the coordinate",
                os.path.basename(image),
                exc,
            )
            return False

        naxis1, naxis2 = prim.data.shape

        inside_im = False
        if (
            x > 0
            and x < naxis1 - 1
            and y > 0
            and y < naxis2 - 1
        ):
            inside_im = True

        return inside_im

  # ---------------------------------------------------------------------------
  # Reference stack: deepest exposures, drizzle, and reference path
  # ---------------------------------------------------------------------------

  @log_calls
  def pick_deepest_images(self, images, reffilter=None, avoid_wfpc2=False,
    refinst=None):
    """
    Select the longest-exposure images in preferred filters for a reference stack.

    Parameters
    ----------
    images : list of str
        Paths to calibrated FITS files.
    reffilter : str, optional
        If set, restrict to this filter (must be in ``acceptable_filters``).
    avoid_wfpc2 : bool, optional
        If True, skip WFPC2 exposures.
    refinst : str, optional
        If set, restrict to this instrument name.

    Returns
    -------
    list of str
        Deepest exposures matching the constraints.
    """
    best_filters = ['f606w','f555w','f814w','f350lp','f110w','f105w',
        'f336w']

    # If we gave an input filter for reference, override best_filters
    if reffilter:
        if reffilter.upper() in self.options['acceptable_filters']:
            # Automatically set the best filter to only this value
            best_filters = [reffilter.lower()]

    # Best filter suffixes in the approximate order we would want to use to
    # generate a template.
    best_types = ['lp', 'w', 'x', 'm', 'n']

    # First group images together by filter/instrument
    filts = [self._fits.get_filter(im) for im in images]
    insts = [self._fits.get_instrument(im).replace('_full','').replace('_sub','')
        for im in images]

    if refinst:
        mask = [refinst.lower() in i for i in insts]
        if any(mask):
            filts = list(np.array(filts)[mask])
            insts = list(np.array(insts)[mask])

    # Group images together by unique instrument/filter pairs and then
    # calculate the total exposure time for all pairs.
    unique_filter_inst = list(set(['{}_{}'.format(a_, b_)
        for a_, b_ in zip(filts, insts)]))

    # Don't construct reference image from acs/hrc if avoidable
    if any(['hrc' not in val for val in unique_filter_inst]):
        # remove all elements with hrc
        new = [val for val in unique_filter_inst if 'hrc' not in val]
        unique_filter_inst = new

    # Do same for WFPC2 if avoid_wfpc2=True
    if avoid_wfpc2:
        if any(['wfpc2' not in val for val in unique_filter_inst]):
            # remove elements with WFPC2
            new = [val for val in unique_filter_inst if 'wfpc2' not in val]
            unique_filter_inst = new

    total_exposure = []
    for val in unique_filter_inst:
        exposure = 0
        for im in self.input_images:
            if (self._fits.get_filter(im) in val and
                self._fits.get_instrument(im).split('_')[0] in val):
                exposure += fits.getval(im,'EXPTIME')
        total_exposure.append(exposure)

    best_filt_inst = ''
    best_exposure = 0

    # First type to generate a reference image from the 'best' filters.
    for filt in best_filters:
        if any(filt in s for s in unique_filter_inst):
            vals = filter(lambda x: filt in x, unique_filter_inst)
            for v in vals:
                exposure = total_exposure[unique_filter_inst.index(v)]
                if exposure > best_exposure:
                    best_filt_inst = v
                    best_exposure = exposure

    # Now try to generate a reference image for types in best_types.
    for filt_type in best_types:
        if not best_filt_inst:
            if any(filt_type in s for s in unique_filter_inst):
                vals = filter(lambda x: filt_type in x, unique_filter_inst)
                for v in vals:
                    exposure = total_exposure[unique_filter_inst.index(v)]
                    if exposure > best_exposure:
                        best_filt_inst = v
                        best_exposure = exposure

    # Now get list of images with best_filt_inst.
    reference_images = []
    for im in images:
        filt = self._fits.get_filter(im)
        inst = self._fits.get_instrument(im).replace('_full','').replace('_sub','')
        if (filt+'_'+inst == best_filt_inst):
            reference_images.append(im)

    return reference_images

  @log_calls
  def pick_reference(self, obstable):
    """
    Build or reuse the astrometric reference drizzle for alignment and DOLPHOT.

    Chooses the deepest exposures in the preferred filter, optionally runs an
    instrument-mask drizzle when ``n<3``, then drizzles to the visit reference
    filename (``{inst}.{filt}.ref_{visit}.drc.fits``) under ``work_dir``.

    Parameters
    ----------
    obstable : `astropy.table.Table`
        Observation table with at least an ``image`` column.

    Returns
    -------
    str or None
        Path to the drizzled reference ``.drc.fits``, or None on failure.
    """
    reference_images = self.pick_deepest_images(list(obstable['image']),
        reffilter=self.options['args'].reference_filter,
        avoid_wfpc2=self.options['args'].avoid_wfpc2,
        refinst=self.options['args'].reference_instrument)

    if len(reference_images)==0:
        error = 'Could not pick a reference image'
        log.error(error)
        return None

    best_filt = self._fits.get_filter(reference_images[0])
    best_inst = self._fits.get_instrument(reference_images[0]).split('_')[0]

    vnum = np.min(obstable['visit'].data)
    vnum = str(vnum).zfill(4)

    # Generate photpipe-like output name for the drizzled image
    if self.options['args'].object:
        drizname = '{obj}.{inst}.{filt}.ref_{num}.drc.fits'
        drizname = drizname.format(inst=best_inst, filt=best_filt,
            obj=self.options['args'].object, num=vnum)
    else:
        drizname = '{inst}.{filt}.ref_{num}.drc.fits'
        drizname = drizname.format(inst=best_inst, filt=best_filt, num=vnum)

    wd = self.options["args"].work_dir
    if wd:
        drizname = os.path.join(wd, drizname)

    reference_images = sorted(reference_images)

    if os.path.isfile(drizname) and not want_redo_astrodrizzle(
        self.options["args"]
    ):
        hdu = fits.open(drizname)
        try:
            hdr = drizzle_product_catalog_header(hdu)
            # Check for NINPUT and INPUT names
            if "NINPUT" in hdr and "INPUT" in hdr:
                ninput = len(reference_images)
                str_input = ",".join([s.split(".")[0] for s in reference_images])

                if hdr["INPUT"].startswith(str_input) and hdr["NINPUT"] == ninput:
                    log.warning(
                        "Drizzled image %s exists; skipping astrodrizzle.",
                        drizname,
                    )
                    return drizname
        finally:
            hdu.close()

    log.info(
        "Reference drizzle: %s | inputs=%s",
        drizname,
        ",".join(os.path.basename(str(p)) for p in reference_images),
    )

    if self.options['args'].drizzle_add:
        add_images = list(str(self.options['args'].drizzle_add).split(','))
        for image in add_images:
            ip = normalize_fits_path(image)
            if os.path.isfile(ip) and ip not in reference_images:
                reference_images.append(ip)

    if "wfpc2" in best_inst and len(obstable) < 3:
        wd = self.options["args"].work_dir or "."
        ws = pipeline_workspace_dir(wd) or wd
        reference_images = [
            normalize_fits_path(p)
            for p in glob.glob(os.path.join(ws, "u*c0m.fits"))
            if os.path.isfile(p)
        ]

    obstable = self.input_list(reference_images, show=True, save=False)
    if not obstable or len(obstable)==0:
        return None

    # Interstitial `{inst}.ref.drc.fits` when n<3 (instrument-mask pass); removed
    # after the final filter-named reference drizzle succeeds.
    instrument_mask_ref_path: str | None = None

    # If number of images is small, try to use imaging from the same instrument
    # and detector for masking
    if len(obstable)<3 and not self.options['args'].no_mask:
        inst = obstable['instrument'][0]
        det = obstable['detector'][0]
        mask = (obstable['instrument']==inst) & (obstable['detector']==det)

        outimage = "{inst}.ref.drc.fits".format(inst=inst)
        if self.options["args"].work_dir:
            ws = pipeline_workspace_dir(self.options["args"].work_dir)
            outimage = os.path.join(
                ws or self.options["args"].work_dir, outimage
            )

        instrument_mask_ref_path = outimage

        if not self.options['args'].skip_tweakreg:
            error, shift_table = self._astrom.run_tweakreg(obstable[mask], '')
        if not self.run_astrodrizzle(
            obstable[mask], output_name=outimage, clean=False, save_fullfile=True
        ):
            log.error("Reference drizzle failed for %s", outimage)
            return None

        # Add cosmic ray mask to static image mask
        if self.options['args'].add_crmask:
            for row in obstable[mask]:
                file = row['image']
                crmasks = glob.glob(file.replace('.fits','*crmask.fits'))

                for i,crmaskfile in enumerate(sorted(crmasks)):
                    crmaskhdu = fits.open(crmaskfile)
                    crmask = crmaskhdu[0].data==0
                    if 'c0m' in file:
                        maskfile = file.split('_')[0]+'_c1m.fits'
                        if os.path.exists(maskfile):
                            maskhdu = fits.open(maskfile)
                            maskhdu[i+1].data[crmask]=4096
                            maskhdu.writeto(maskfile, overwrite=True)
                    else:
                        maskhdu = fits.open(file)
                        if maskhdu[3*i+1].name=='DQ':
                            maskhdu[3*i+1].data[crmask]=4096
                        maskhdu.writeto(file, overwrite=True)

    if not self.options['args'].skip_tweakreg:
        error, shift_table = self._astrom.run_tweakreg(obstable, '')
    if not self.run_astrodrizzle(
        obstable, output_name=drizname, save_fullfile=True
    ):
        log.error("Reference drizzle failed for %s", drizname)
        return None

    if instrument_mask_ref_path:
        remove_superseded_instrument_mask_reference_drizzle(
            instrument_mask_ref_path,
            log=log,
            keep_artifacts=getattr(
                self.options["args"], "keep_drizzle_artifacts", False
            ),
        )

    return drizname

  # ---------------------------------------------------------------------------
  # Per-image calibration headers and WCS (before / after updatewcs)
  # ---------------------------------------------------------------------------

  def fix_idcscale(self, image):
    """
    Ensure WFC3 IDC headers carry ``IDCSCALE`` from detector defaults.

    Parameters
    ----------
    image : str
        Path to a FITS file updated in place.
    """
    det = '_'.join(self._fits.get_instrument(image).split('_')[:2])

    if 'wfc3' in det:
        hdu = fits.open(image)
        idcscale = self.options['detector_defaults'][det]['idcscale']
        for i,h in enumerate(hdu):
            if 'IDCSCALE' not in hdu[i].header.keys():
                hdu[i].header['IDCSCALE']=idcscale

        hdu.writeto(image, overwrite=True, output_verify='silentfix')

  def fix_phot_keys(self, image):
    """
    Propagate the first ``PHOTPLAM`` / ``PHOTFLAM`` pair to every HDU.

    Parameters
    ----------
    image : str
        Path to a FITS file updated in place.
    """
    hdu = fits.open(image)
    photplam=None
    photflam=None

    for i,h in enumerate(hdu):

        if 'PHOTPLAM' in h.header.keys() and 'PHOTFLAM' in h.header.keys():
            photplam = h.header['PHOTPLAM']
            photflam = h.header['PHOTFLAM']
            break

    if photflam and photplam:
        for i,h in enumerate(hdu):
            hdu[i].header['PHOTPLAM']=photplam
            hdu[i].header['PHOTFLAM']=photflam

        hdu.writeto(image, overwrite=True, output_verify='silentfix')

  def fix_hdu_wcs_keys(self, image, change_keys, ref_url):
    """
    Rewrite ``WCSNAME``-like keys and expand CRDS reference paths for alignment.

    Parameters
    ----------
    image : str
        FITS path opened in update mode.
    change_keys : iterable of str
        Header keys to normalize (strip ``WCSNAME`` values, resolve ``ref$`` paths).
    ref_url : str
        CRDS / reference root URL prefix (see ``ref_prefix_for_header`` in
        ``hst123.utils.reference_download``).
    """
    hdu = fits.open(image, mode='update')
    ref = ref_prefix_for_header(ref_url)
    outdir = self.options['args'].work_dir or os.path.abspath(".")
    cals_dir = os.path.join(outdir, "cals")
    os.makedirs(cals_dir, exist_ok=True)

    for i,h in enumerate(hdu):
        for key in hdu[i].header.keys():
            if 'WCSNAME' in key:
                hdu[i].header[key] = hdu[i].header[key].strip()
        for key in change_keys:
            if key in list(hdu[i].header.keys()):
                val = hdu[i].header[key]
            else:
                continue
            if val == 'N/A':
                continue
            if (ref+'$' in val):
                ref_file = val.split('$')[1]
            else:
                ref_file = val
            # Header may store paths like test_data/foo.fits; CRDS URLs need basename only
            ref_file = os.path.basename(ref_file.strip())

            fullfile = os.path.join(cals_dir, ref_file)
            if not os.path.exists(fullfile):
                log.debug("ref fetch: %s", os.path.basename(fullfile))
                ok, err = fetch_calibration_reference(
                    self.options['global_defaults'],
                    ref_url,
                    ref_file,
                    fullfile,
                    log=log,
                )
                if not ok:
                    log.warning(
                        'Could not download reference %s (last error: %s)',
                        ref_file,
                        err or 'unknown',
                    )

            if os.path.exists(fullfile):
                log.debug("Setting %s ext%d %s=%s", image, i, key, fullfile)
                hdu[i].header[key] = fullfile
            else:
                log.error(
                    'Missing reference file %s; not updating header key %s',
                    fullfile,
                    key,
                )

        # WFPC2 does not have residual distortion corrections and astrodrizzle
        # choke if DGEOFILE is in header but not NPOLFILE.  So do a final check
        # for this part of the WCS keys
        if 'wfpc2' in self._fits.get_instrument(image).lower():
            keys = list(h.header.keys())
            if 'DGEOFILE' in keys and 'NPOLFILE' not in keys:
                del hdu[i].header['DGEOFILE']

    hdu.writeto(image, overwrite=True, output_verify='silentfix')
    hdu.close()

  def update_image_wcs(self, image, options, use_db=True):
    """
    Run STScI ``updatewcs`` on a calibrated image after header hygiene.

    Skips images already marked ``TWEAKSUC`` or hierarchically aligned. Otherwise
    resolves CRDS paths, optionally calls AstrometryDB (``use_db``), then applies
    ``fix_hdu_wcs_keys`` and ``fix_idcscale``.

    Parameters
    ----------
    image : str
        Path to the FITS file.
    options : dict
        Unused legacy parameter (detector defaults come from ``self.options``).
    use_db : bool, optional
        Forwarded to :func:`~hst123.utils.stsci_wcs.run_updatewcs`.

    Returns
    -------
    bool or None
        True on success, None if ``updatewcs`` raised.
    """
    hdu = fits.open(image, mode='readonly')
    if 'TWEAKSUC' in hdu[0].header.keys() and hdu[0].header['TWEAKSUC']==1:
        return True

    # Check for hierarchical alignment.  If image has been shifted with
    # hierarchical alignment, we don't want to shift it again
    if 'HIERARCH' in hdu[0].header.keys() and hdu[0].header['HIERARCH']==1:
        return True

    hdu.close()

    message = 'Updating WCS for {file}'
    log.info(message.format(file=image))

    self.clear_downloads(self.options['global_defaults'])

    change_keys = self.options['global_defaults']['keys']
    inst = self._fits.get_instrument(image).split('_')[0]
    ref_url = self.options['instrument_defaults'][inst]['env_ref']

    self.fix_hdu_wcs_keys(image, change_keys, ref_url)

    _n_sip = fix_sip_ctype_headers_fits(image, logger=log)
    if _n_sip:
        log.debug("SIP/CTYPE aligned before updatewcs: %s (%d HDU(s))", image, _n_sip)

    _n_stale_alt = remove_conflicting_alt_wcs_duplicate_names(image, logger=log)
    if _n_stale_alt:
        log.debug(
            "Cleared %d stale alternate WCS key(s) (duplicate WCSNAME vs primary) before updatewcs: %s",
            _n_stale_alt,
            os.path.basename(image),
        )

    _quiet = (
        "astropy",
        "astropy.wcs",
        "astropy.wcs.wcs",
        "astropy.io",
        "astropy.io.fits",
        "stwcs",
        "stwcs.wcsutil",
        "stwcs.wcsutil.altwcs",
        "stwcs.wcsutil.headerlet",
    )
    _prev_lv = {}
    for _name in _quiet:
        _lg = py_logging.getLogger(_name)
        _prev_lv[_name] = _lg.level
        _lg.setLevel(py_logging.ERROR)
    _ap_log = None
    _ap_level0 = None
    try:
        import astropy as _astropy_pkg

        _ap_log = _astropy_pkg.log
        _ap_level0 = _ap_log.level
        _ap_log.setLevel(py_logging.ERROR)
    except Exception:
        _ap_log = None
    try:
        try:
            try:
                from astropy.wcs.wcs import FITSFixedWarning as _FFW
            except ImportError:
                _FFW = None
            with warnings.catch_warnings():
                if _FFW is not None:
                    warnings.simplefilter("ignore", _FFW)
                with suppress_stdout_fd():
                    run_updatewcs(image, use_db=use_db)
        finally:
            for _name, _lv in _prev_lv.items():
                py_logging.getLogger(_name).setLevel(_lv)
            if _ap_log is not None and _ap_level0 is not None:
                _ap_log.setLevel(_ap_level0)
        hdu = fits.open(image, mode='update')
        log.info(
            "updatewcs ok %s | %s",
            os.path.basename(image),
            format_hdu_list_summary(hdu),
        )
        hdu.close()
        self.fix_hdu_wcs_keys(image, change_keys, ref_url)
        self.fix_idcscale(image)
        return True
    except Exception:
        error = 'Failed to update WCS for image {file}'
        log.error(error.format(file=image))
        return None

  # ---------------------------------------------------------------------------
  # DrizzlePac (AstroDrizzle): scratch inputs, photeq, combine, DRC sidecars
  # ---------------------------------------------------------------------------

  @log_calls
  def run_astrodrizzle(self, obstable, output_name=None, ra=None, dec=None,
    clean=None, save_fullfile=False):
    """
    Drizzle a stack of exposures with DrizzlePac (sky match, CR rejection, combine).

    Copies inputs to scratch paths (WFPC2 ``*_c0m`` / ``*_c1m``), runs ``photeq``
    when applicable, invokes :func:`drizzlepac.astrodrizzle.AstroDrizzle`, then
    renames sidecars and optionally writes a multi-extension ``.drc.fits``.

    Parameters
    ----------
    obstable : `astropy.table.Table`
        Must include an ``image`` column listing calibrated FITS paths.
    output_name : str, optional
        Desired output path (``*.drc.fits`` logical name); default ``drizzled.drc.fits``.
    ra, dec : float, optional
        Field center for output WCS (degrees); default from ``self.coord``.
    clean : bool or None
        DrizzlePac ``clean`` flag; resolved via :func:`~hst123.utils.astrodrizzle_helpers.resolve_drizzle_clean_flag`.
    save_fullfile : bool, optional
        If True, build a multi-extension DRC product when supported.

    Returns
    -------
    bool
        True if the drizzle product exists on disk; False otherwise.
    """
    n = len(obstable)
    # Unique scratch names so concurrent AstroDrizzle calls never share dup/temp FITS.
    scratch_tag = uuid.uuid4().hex[:16]

    wd_arg = self.options["args"].work_dir
    if wd_arg:
        outdir = pipeline_workspace_dir(wd_arg) or os.path.abspath(wd_arg)
    else:
        outdir = os.path.abspath(".")
    os.makedirs(outdir, exist_ok=True)

    if output_name is None:
        output_name = os.path.join(outdir, "drizzled.drc.fits")

    logical_output = normalize_astrodrizzle_output_path(output_name, log)
    internal_output = normalize_astrodrizzle_output_path(
        logical_driz_to_internal_astrodrizzle(os.fspath(logical_output)), log
    )

    combine_type, combine_nhigh = combine_type_and_nhigh(
        n, self.options["args"].combine_type or None
    )

    wcskey = "TWEAK"

    det = "_".join(self._fits.get_instrument(obstable[0]["image"]).split("_")[:2])
    options = self.options['detector_defaults'][det]

    # Make a copy of each input image so drizzlepac doesn't edit base headers
    tmp_input = []
    wfpc2_c1m_scratch: list[str] = []
    for image in obstable['image']:
        inst_l = self._fits.get_instrument(image).lower()
        base_l = os.path.basename(image).lower()
        if "wfpc2" in inst_l and base_l.endswith("_c0m.fits"):
            tmp_c0m, tmp_c1m = wfpc2_astrodrizzle_scratch_paths(image, scratch_tag)
            shutil.copyfile(image, tmp_c0m)
            if tmp_c1m is not None:
                c1m_src = os.path.join(
                    os.path.dirname(os.path.abspath(image)),
                    os.path.basename(image)[:-9] + "_c1m.fits",
                )
                shutil.copyfile(c1m_src, tmp_c1m)
                wfpc2_c1m_scratch.append(tmp_c1m)
            else:
                log.warning(
                    "WFPC2 DQ file missing next to %s; drizzle may fail without *_c1m.fits",
                    os.path.basename(image),
                )
            tmp_input.append(tmp_c0m)
        else:
            tmp = image.replace(".fits", ".drztmp.fits")
            shutil.copyfile(image, tmp)
            tmp_input.append(tmp)

    if self.updatewcs:
        for image in tmp_input:
            det = '_'.join(self._fits.get_instrument(image).split('_')[:2])
            wcsoptions = self.options['detector_defaults'][det]
            self.update_image_wcs(image, wcsoptions, use_db=False)
    elif self.options['args'].skip_tweakreg:
        for image in tmp_input:
            self.clear_downloads(self.options['global_defaults'])

            change_keys = self.options['global_defaults']['keys']
            inst = self._fits.get_instrument(image).split('_')[0]
            ref_url = self.options['instrument_defaults'][inst]['env_ref']

            self.fix_hdu_wcs_keys(image, change_keys, ref_url)

    if not ra or not dec:
        ra = self.coord.ra.degree if self.coord else None
        dec = self.coord.dec.degree if self.coord else None

    if self.options['args'].keep_short and not self.options['args'].sky_sub:
        skysub = False
    else:
        skysub = True

    if self.options['args'].drizzle_scale:
        pixscale = self.options['args'].drizzle_scale
    else:
        pixscale = options['pixel_scale']

    wht_type = self.options["args"].wht_type

    drizzle_clean = resolve_drizzle_clean_flag(clean, self.options["args"].cleanup)
    if save_fullfile:
        drizzle_clean = False

    if len(tmp_input) == 1:
        # Duplicate input so AstroDrizzle has ≥2 frames; keep under outdir so it is
        # found when CWD is not work_dir (e.g. reference drizzle before run_alignment chdir).
        first = tmp_input[0]
        base_l = os.path.basename(first).lower()
        inst_l = self._fits.get_instrument(first).lower()
        if "wfpc2" in inst_l and base_l.endswith("_c0m.fits"):
            # DrizzlePac WFPC2 derives *_c1m.fits from the science name via _c0m→_c1m;
            # a generic dup_input.fits name breaks that and sky/DQ steps fail.
            dup_c0m = os.path.join(
                outdir, f"hst123_astrodrizzle_dup_{scratch_tag}_c0m.fits"
            )
            shutil.copyfile(first, dup_c0m)
            tmp_input.append(dup_c0m)
            c1m_src = os.path.join(
                os.path.dirname(os.path.abspath(first)),
                os.path.basename(first)[:-9] + "_c1m.fits",
            )
            if os.path.isfile(c1m_src):
                dup_c1m = os.path.join(
                    outdir, f"hst123_astrodrizzle_dup_{scratch_tag}_c1m.fits"
                )
                shutil.copyfile(c1m_src, dup_c1m)
                wfpc2_c1m_scratch.append(dup_c1m)
            else:
                log.warning(
                    "WFPC2 single-input duplicate: missing paired DQ %s next to %s",
                    os.path.basename(c1m_src),
                    os.path.basename(first),
                )
        else:
            dup_path = os.path.join(
                outdir, f"hst123_astrodrizzle_dup_{scratch_tag}_input.fits"
            )
            shutil.copyfile(first, dup_path)
            tmp_input.append(dup_path)

    self.input_list(obstable["image"], show=True, save=False)

    # If drizmask, then edit tmp_input masks for everything except for drizadd
    # files
    if self.options['args'].drizzle_mask and self.options['args'].drizzle_add:
        add_im_base = [im.split('.')[0]
            for im in self.options['args'].drizzle_add.split(',')]

        if ',' in self.options['args'].drizzle_mask:
            ramask, decmask = self.options['args'].drizzle_mask.split(',')
        else:
            ramask, decmask = self.options['args'].drizzle_mask.split()

        maskcoord = parse_coord(ramask, decmask)

        for image in tmp_input:
            added = any(base in image for base in add_im_base)
            with fits.open(image, mode="update") as imhdu:
                for i, h in enumerate(imhdu):
                    if h.name != "DQ":
                        continue

                    w = wcs_from_fits_hdu(imhdu, i)
                    y, x = wcs.utils.skycoord_to_pixel(maskcoord, w, origin=1)

                    size = self.options["global_defaults"]["mask_region_size"]
                    naxis1, naxis2 = h.data.shape

                    outside_im = False
                    if (
                        x + size < 0
                        or x - size > naxis1 - 1
                        or y + size < 0
                        or y - size > naxis2 - 1
                    ):
                        if not added:
                            continue
                        outside_im = True

                    xmin = int(np.max([x - size, 0]))
                    ymin = int(np.max([y - size, 0]))
                    xmax = int(np.min([x + size, naxis2 - 1]))
                    ymax = int(np.min([y + size, naxis1 - 1]))

                    imhdu[i].data[xmin:xmax, ymin:ymax]

                    if any(base in image for base in add_im_base):
                        log.info("Making outside drizmask: %s", image)
                        if outside_im:
                            imhdu[i].data[:, :] = 128
                        else:
                            data = copy.copy(imhdu[i].data[xmin:xmax, ymin:ymax])
                            imhdu[i].data[:, :] = 128
                            imhdu[i].data[xmin:xmax, ymin:ymax] = data
                    else:
                        log.info("Making inside drizmask: %s", image)
                        imhdu[i].data[xmin:xmax, ymin:ymax] = 128

                imhdu.flush()

    for image in tmp_input:
        ensure_wcsname_tweak_on_image(image, log)

    start_drizzle = time.time()

    skymask_cat = build_wfpc2_skymask_catalog(tmp_input, outdir, log)

    # Equalize sensitivities (non-WFPC2); WFPC2 skips photeq in original logic
    photeq_log = ephemeral_pipeline_runfile(outdir, "photeq")
    n_photeq = 0
    for image in tmp_input:
        if "wfpc2" in self._fits.get_instrument(image).lower():
            continue
        n_photeq += 1
        log.debug("photeq %s", os.path.basename(image))
        self.fix_phot_keys(image)
        with suppress_stdout_fd():
            with suppress_stdout():
                photeq.photeq(
                    files=image,
                    readonly=False,
                    ref_phot_ext=1,
                    logfile=photeq_log,
                )
    if n_photeq:
        log.info(
            "photeq: %d image(s); replaying runfile into session log (not kept in work dir)",
            n_photeq,
        )
        ingest_text_file_to_logger(
            photeq_log,
            get_logger(PHOTEQ_DETAIL_LOGGER),
            log_tag="photeq",
            replay_full=True,
            begin_end_markers=False,
            compact_ws=True,
            delete_after=True,
        )

    rotation = 0.0
    if self.options["args"].no_rotation:
        rotation = None

    logfile_name = ephemeral_pipeline_runfile(outdir, "astrodrizzle")
    dd = self.options["drizzle_defaults"]
    ad_kwargs = build_astrodrizzle_keyword_args(
        output_name=internal_output,
        logfile_name=logfile_name,
        wcskey=wcskey,
        options=options,
        dd=dd,
        ra=ra,
        dec=dec,
        rotation=rotation,
        combine_type=combine_type,
        combine_nhigh=combine_nhigh,
        skysub=skysub,
        skymask_cat=skymask_cat,
        wht_type=wht_type,
        pixscale=pixscale,
        clean=drizzle_clean,
    )

    tries = 0
    ad_detail_log = get_logger(ASTRODRIZZLE_DETAIL_LOGGER)
    try:
        while tries < 3:
            try:
                log.info(
                    "AstroDrizzle: %d tmp input(s), combine=%s, output=%s (internal %s)",
                    len(tmp_input),
                    combine_type,
                    os.path.basename(logical_output),
                    os.path.basename(internal_output),
                )
                log.debug("AstroDrizzle inputs: %s", tmp_input)
                # C extensions may printf to fd 1; ingest runfile in finally.
                with limit_blas_threads_when_parallel(int(ad_kwargs.get("num_cores", 1))):
                    with suppress_stdout_fd():
                        with suppress_stdout():
                            astrodrizzle.AstroDrizzle(tmp_input, **ad_kwargs)
                break
            except FileNotFoundError:
                # Usually happens because of a file missing in astropy cache.
                # Try clearing the download cache and then re-try
                self.clear_downloads(self.options['global_defaults'])
                tries += 1
    finally:
        ingest_text_file_to_logger(
            logfile_name,
            ad_detail_log,
            replay_full=True,
            begin_end_markers=False,
            compact_ws=True,
            delete_after=True,
        )

    if tries >= 3:
        log.error(
            "AstroDrizzle failed after retries (often a missing CRDS/cache file). "
            "See session log for AstroDrizzle replay."
        )

    log.info("Astrodrizzle finished in %.2fs", time.time() - start_drizzle)

    if self.options['args'].cleanup:
        for image in tmp_input:
            if os.path.isfile(image):
                os.remove(image)
        for c1m_tmp in wfpc2_c1m_scratch:
            if os.path.isfile(c1m_tmp):
                os.remove(c1m_tmp)

    for dup_suffix in ("input", "c0m", "c1m"):
        dup_p = os.path.join(
            outdir, f"hst123_astrodrizzle_dup_{scratch_tag}_{dup_suffix}.fits"
        )
        if os.path.exists(dup_p):
            os.remove(dup_p)

    internal_output = recover_drizzlepac_linear_output(internal_output, log)

    science_file, _, _ = drizzle_sidecar_paths(internal_output)
    if not astrodrizzle_output_exists(internal_output):
        log.error(
            "AstroDrizzle did not produce expected drizzle product %r "
            "(missing sidecar %r). Truncated drizname (no .drz.fits) was a "
            "legacy table issue — see astrodrizzle_paths.normalize.",
            internal_output,
            science_file,
        )
        cleanup_after_astrodrizzle(
            outdir,
            log=log,
            keep_artifacts=getattr(
                self.options["args"], "keep_drizzle_artifacts", False
            ),
            base_work_dir=self.options["args"].work_dir,
        )
        return False

    if not os.path.isfile(internal_output) and os.path.isfile(science_file):
        log.info(
            "AstroDrizzle sidecar %s -> canonical %s",
            os.path.basename(science_file),
            os.path.basename(internal_output),
        )

    weight_file, mask_file = rename_astrodrizzle_sidecars(internal_output, log)

    # Get comma-separated list of base input files
    ninput = len(tmp_input)
    tmp_input = sorted(tmp_input)
    str_input = ','.join([s.split('.')[0] for s in tmp_input])

    origzpt = self._fits.get_zpt(internal_output)

    # Add header keys on drizzled file
    hdu = fits.open(internal_output, mode='update')
    filt = obstable['filter'][0]
    hdu[0].header['FILTER'] = filt.upper()
    hdu[0].header['TELID'] = 'HST'
    hdu[0].header['OBSTYPE'] = 'OBJECT'
    hdu[0].header['EXTVER'] = 1
    hdu[0].header['ORIGZPT']=origzpt
    # Format the header time variable for MJD-OBS, DATE-OBS, TIME-OBS
    time_start = Time(hdu[0].header['EXPSTART'], format='mjd')
    hdu[0].header['MJD-OBS'] = time_start.mjd
    hdu[0].header['DATE-OBS'] = time_start.datetime.strftime('%Y-%m-%d')
    hdu[0].header['TIME-OBS'] = time_start.datetime.strftime('%H:%M:%S')
    # These keys are useful for auditing drz image later
    hdu[0].header['NINPUT'] = ninput
    hdu[0].header['INPUT'] = str_input
    hdu[0].header['BUNIT'] = 'ELECTRONS'
    # Add object name if it was input from command line
    if self.options['args'].object:
        hdu[0].header['TARGNAME'] = self.options['args'].object
        hdu[0].header['OBJECT'] = self.options['args'].object

    if self.options['args'].fix_zpt:
        # Get current zeropoint of drizzled image
        fixzpt = self.options['args'].fix_zpt
        zpt = origzpt
        exptime = hdu[0].header['EXPTIME']
        effzpt = zpt + 2.5*np.log10(exptime)
        fixscale = 10**(0.4 * (fixzpt - effzpt))
        fluxscale = self.options['global_defaults']['ab_flux_zero_mjy'] * 10**(-0.4 * fixzpt) # mJy/pix scale

        # Adjust header values for context
        inst = self._fits.get_instrument(internal_output).split('_')[0]
        crpars = self.options['instrument_defaults'][inst]['crpars']
        det = '_'.join(self._fits.get_instrument(internal_output).split('_')[:2])
        instopt = self.options['detector_defaults'][det]

        hdu[0].header['FIXZPT']   = fixzpt
        hdu[0].header['FIXFLUX']  = fluxscale
        hdu[0].header['EFFZPT']   = effzpt
        hdu[0].header['FIXSCALE'] = fixscale
        # rescaled by EXPTIME, so essentially cps
        hdu[0].header['BUNIT']    = 'cps'
        hdu[0].header['SCALSAT']  = crpars['saturate'] * fixscale

        # Finally do data scaling
        data = hdu[0].data * fixscale
        hdu[0].data = data

    hdu.close()

    logical_drc = (
        os.fspath(logical_output)
        if str(logical_output).lower().endswith(".drc.fits")
        else None
    )
    drc_written = write_drc_multis_extension_if_requested(
        internal_output,
        weight_file,
        mask_file,
        save_fullfile,
        log,
        format_hdu_list_summary=format_hdu_list_summary,
        logical_drc_path=logical_drc,
    )
    if drc_written:
        remove_internal_linear_drizzle_products(internal_output, log)
    elif logical_drc:
        log.warning(
            "Keeping internal drizzle product %s (could not build %s; missing weight/mask?)",
            internal_output,
            logical_drc,
        )

    cleanup_after_astrodrizzle(
        outdir,
        log=log,
        keep_artifacts=getattr(
            self.options["args"], "keep_drizzle_artifacts", False
        ),
        base_work_dir=self.options["args"].work_dir,
    )

    return True

  # ---------------------------------------------------------------------------
  # Cosmic rays (astroscrappy); optional DQ / WFPC2 c1m updates
  # ---------------------------------------------------------------------------

  def run_cosmic(self, image, options, output=None):
    """
    Run LAcosmic-style cosmic-ray detection (astroscrappy) on SCI extensions.

    Parameters
    ----------
    image : str
        Input FITS path.
    options : dict
        Detector block with ``rdnoise``, ``gain``, ``saturate``, ``sig_clip``,
        ``sig_frac``, and ``obj_lim`` (see ``settings.detector_defaults``).
    output : str, optional
        Output path; defaults to overwriting *image*.

    Notes
    -----
    When ``--add-crmask`` is set, updates matching DQ extensions or WFPC2 ``*_c1m``.
    """
    message = "Cleaning cosmic rays in image: {image}"
    log.info(message.format(image=image))
    hdulist = fits.open(image, mode="readonly")

    if output is None:
        output = image

    for i, hdu in enumerate(hdulist):
        if hdu.name == "SCI":
            mask = np.zeros(hdu.data.shape, dtype=np.bool_)

            crmask, crclean = detect_cosmics(hdu.data.copy().astype('<f4'),
                inmask=mask, readnoise=options['rdnoise'], gain=options['gain'],
                satlevel=options['saturate'], sigclip=options['sig_clip'],
                sigfrac=options['sig_frac'], objlim=options['obj_lim'])

            hdulist[i].data[:, :] = crclean[:, :]

            if self.options["args"].add_crmask:
                if "flc" in image or "flt" in image:
                    if len(hdulist) >= i + 2 and hdulist[i + 2].name == "DQ":
                        hdulist[i + 2].data[np.where(crmask)] = 4096
                elif "c0m" in image:
                    maskfile = image.split("_")[0] + "_c1m.fits"
                    if os.path.exists(maskfile):
                        maskhdu = fits.open(maskfile)
                        maskhdu[i].data[np.where(crmask)] = 4096
                        maskhdu.writeto(maskfile, overwrite=True)

    hdulist.writeto(output, overwrite=True, output_verify="silentfix")
    hdulist.close()

  # ---------------------------------------------------------------------------
  # MAST: query_region, product table, download_files
  # ---------------------------------------------------------------------------

  @log_calls
  def get_productlist(self, coord, search_radius):
    """
    Query MAST for HST science products near a sky position.

    Parameters
    ----------
    coord : `astropy.coordinates.SkyCoord`
        Field center.
    search_radius : float or str
        Search radius passed to ``astroquery.mast.Observations.query_region``.

    Returns
    -------
    astropy.table.Table or None
        Table with ``productFilename``, ``downloadFilename``, ``obsID``, etc., or
        None if no rows match the configured filters.
    """
    self.clear_downloads(self.options['global_defaults'])

    productlist = None

    # Check for coordinate and exit if it does not exist
    if not coord:
        error = 'Coordinate was not provided.'
        return productlist

    make_banner("MAST catalog query")
    log.info(
        "query_region center=%s radius=%s",
        coord.to_string("hmsdms"),
        search_radius,
    )

    # Define search params and grab all files from MAST
    try:
        if self.options['args'].token:
            log.info("MAST Observations.login(token=...)")
            Observations.login(token=self.options['args'].token)
    except Exception:
        log.warning("Could not log in with input username/password")

    try:
        obsTable = Observations.query_region(coord, radius=search_radius)
    except (astroquery.exceptions.RemoteServiceError,
        requests.exceptions.ConnectionError,
        astroquery.exceptions.TimeoutError,
        requests.exceptions.ChunkedEncodingError):
        log.error("MAST is not currently working. Try again later.")
        return productlist

    # Get rid of all masked rows (they aren't HST data anyway)
    obsTable = obsTable.filled()
    log.info(
        "MAST query_region returned %i row(s) (before HST/image/detector filters).",
        len(obsTable),
    )

    # Construct masks for telescope, data type, detector, and data rights
    masks = []
    masks.append([t.upper()=='HST' for t in obsTable['obs_collection']])
    masks.append([p.upper()=='IMAGE' for p in obsTable['dataproduct_type']])
    masks.append([any(l) for l in list(map(list,zip(*[[det in inst.upper()
                for inst in obsTable['instrument_name']]
                for det in ['ACS','WFC','WFPC2']])))])
    # Added mask to remove calibration data from search
    masks.append([f.upper()!='DETECTION' for f in obsTable['filters']])
    masks.append([i.upper()!='CALIBRATION' for i in obsTable['intentType']])

    # Time constraint masks (before and after MJD)
    if self.before:
        masks.append([t < Time(self.before).mjd for t in obsTable['t_min']])
    if self.after:
        masks.append([t > Time(self.after).mjd for t in obsTable['t_min']])

    # Get rid of short exposures (defined as 15s or less)
    if not self.options['args'].keep_short:
        masks.append([t > 15. for t in obsTable['t_exptime']])

    # Apply the masks to the observation table
    mask = [all(l) for l in list(map(list, zip(*masks)))]
    obsTable = obsTable[mask]
    log.info(
        "After HST / IMAGE / ACS·WFC·WFPC2 / non-calibration / exposure-time filters: "
        "%i observation(s).",
        len(obsTable),
    )

    if self.options['args'].only_filter:
        get_filts = self.options['args'].only_filter.split(',')
        get_filts = [f.lower() for f in get_filts]
        mask = np.array([any([f in row['filters'].lower() for f in get_filts])
            for row in obsTable])
        obsTable = obsTable[mask]
        log.info(
            "After --only-filter (%s): %i observation(s).",
            self.options['args'].only_filter,
            len(obsTable),
        )

    # Get product lists in order of observation time
    obsTable.sort('t_min')

    # Iterate through each observation and download the correct product
    # depending on the filename and instrument/detector of the observation
    for obs in obsTable:
        try:
            productList = Observations.get_product_list(obs)
            # Ignore the 'C' type products
            mask = productList['type']=='S'
            productList = productList[mask]
        except Exception:
            log.error("MAST is not currently working. Try again later.")
            return productlist

        instrument = obs['instrument_name']
        s_ra = obs['s_ra']
        s_dec = obs['s_dec']

        instcol = Column([instrument]*len(productList), name='instrument_name')
        racol = Column([s_ra]*len(productList), name='ra')
        deccol = Column([s_dec]*len(productList), name='dec')

        productList.add_column(instcol)
        productList.add_column(racol)
        productList.add_column(deccol)

        for prod in productList:
            filename = prod['productFilename']

            if (('c0m.fits' in filename and 'WFPC2' in instrument) or
                ('c1m.fits' in filename and 'WFPC2' in instrument) or
                ('c0m.fits' in filename and 'PC/WFC' in instrument) or
                ('c1m.fits' in filename and 'PC/WFC' in instrument) or
                ('flc.fits' in filename and 'ACS/WFC' in instrument) or
                ('flt.fits' in filename and 'ACS/HRC' in instrument) or
                ('flc.fits' in filename and 'WFC3/UVIS' in instrument) or
                ('flt.fits' in filename and 'WFC3/IR' in instrument)):

                if not productlist:
                    productlist = Table(prod)
                else:
                    productlist.add_row(prod)

    if not productlist:
        log.warning(
            "MAST: no science products (FLT/FLC/C0M/C1M) matched filters for this field."
        )
        return None

    downloadFilenames = []
    for prod in productlist:
        filename = prod['productFilename']

        # Cut down new HST filenames that start with hst_PROGID
        filename = '_'.join(filename.split('_')[-2:])
        downloadFilenames.append(filename)

    productlist.add_column(Column(downloadFilenames, name='downloadFilename'))

    # Check that all files to download are unique
    if productlist and len(productlist)>1:
        productlist = unique(productlist, keys='downloadFilename')

    # Sort by obsID in case we need to reference
    productlist.sort('obsID')

    log.info(
        "MAST: final product list has %i unique file(s) (by downloadFilename, obsID-sorted).",
        len(productlist),
    )

    return productlist

  def _mast_download_one_product_row(self, item):
    """
    Download one MAST product row (used by :meth:`download_files` thread pool).

    Parameters
    ----------
    item : tuple
        ``(i, prod, filename, n, mast_staging_parent)`` where *i* is the 0-based
        index in the full product list, *n* is ``len(productlist)``, and
        *filename* is the destination path after archive / dest resolution.
    """
    i, prod, filename, n, mast_staging_parent = item
    message = f"({i+1}/{n}) {filename}"
    log.debug("MAST download try %s", message)
    try:
        with suppress_stdout():
            download = Observations.download_products(
                Table(prod),
                download_dir=mast_staging_parent,
                cache=False,
            )
        shutil.move(download['Local Path'][0], filename)

        log.info("MAST ok %s", message)

    except Exception as e:
        log.warning("MAST fail %s: %s", message, e)
        log.debug("Download exception detail", exc_info=True)

  @log_calls
  def download_files(
      self,
      productlist,
      dest=None,
      archivedir=None,
      clobber=False,
      work_dir=None,
  ):
    """
    Download MAST products via ``astroquery.mast.Observations.download_products``.

    Parameters
    ----------
    productlist : table-like
        Rows from :meth:`get_productlist` (must include ``downloadFilename``, ``obsID``).
    dest : str, optional
        Directory for downloaded FITS when not using an archive layout.
    archivedir : str, optional
        When set, ``check_archive`` places files under ``{inst}/{det}/{ra}/...``.
    clobber : bool, optional
        If False, skip when the destination file already exists.
    work_dir : str, optional
        Parent for ``.mast_download_staging`` (isolates astroquery temp from CWD).

    Returns
    -------
    bool
        True when the download loop completes (individual files may warn on failure).

    Notes
    -----
    Staging always lives under *work_dir* so ``mastDownload`` is not created in an
    unrelated shell current directory.

    When multiple files need downloading, fetches run in parallel (thread pool)
    with worker count ``min(--max-cores, number of files)``.
    """
    if not productlist:
        log.error('Product list is empty. Cannot download files.')
        return False

    work_abs = os.path.abspath(os.path.expanduser(work_dir or os.getcwd()))
    mast_staging_parent = os.path.join(work_abs, ".mast_download_staging")
    os.makedirs(mast_staging_parent, exist_ok=True)
    mast_download_tree = os.path.join(mast_staging_parent, "mastDownload")

    n = len(productlist)
    log.info("MAST download: %i file(s) in product list.", n)
    log.info(
        "MAST download staging (astroquery temp): %s",
        mast_staging_parent,
    )
    if dest:
        log.info("Download target directory (dest): %s", os.path.abspath(dest))
    if archivedir:
        log.info("Archive directory: %s", os.path.abspath(archivedir))
    if not dest and not archivedir:
        log.info("Download target: current directory (%s)", os.getcwd())

    pending = []
    for i, prod in enumerate(productlist):
        filename = prod['downloadFilename']

        if dest:
            filename = dest + '/' + filename

        if archivedir:
            check, fullfile = self.check_archive(prod, archivedir=archivedir)
            filename = fullfile
            if check and not clobber:
                message = '{file} exists. Skipping...'
                log.info(message.format(file=filename))
                continue
        elif os.path.isfile(filename):
            message = '{file} exists. Skipping...'
            log.info(message.format(file=filename))
            continue

        pending.append((i, prod, filename))

    n_pending = len(pending)
    opt = getattr(self, "options", None)
    args = opt.get("args") if opt else None
    mc = getattr(args, "drizzle_num_cores", None) if args else None
    if mc is None:
        mc = settings.default_astrodrizzle_cores()
    max_workers = max(1, min(int(mc), n_pending) if n_pending else 1)

    if n_pending:
        log.info(
            "MAST download: fetching %i file(s) with up to %i thread(s) "
            "(--max-cores).",
            n_pending,
            max_workers,
        )
        items = [
            (i, prod, filename, n, mast_staging_parent)
            for i, prod, filename in pending
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pool.map(self._mast_download_one_product_row, items)

    # Remove astroquery staging under work-dir (not the shell cwd when --work-dir is set)
    if os.path.isdir(mast_download_tree):
        log.info("Removing temporary MAST staging: %s", mast_download_tree)
        shutil.rmtree(mast_download_tree)
    try:
        if os.path.isdir(mast_staging_parent) and not os.listdir(mast_staging_parent):
            os.rmdir(mast_staging_parent)
    except OSError:
        pass

    return True

  # ---------------------------------------------------------------------------
  # DOLPHOT: param file, binary, scrape; visit splitting; CLI arg bridge
  # ---------------------------------------------------------------------------

  def make_dolphot_file(self, images, reference):
    """
    Write the DOLPHOT parameter file listing chip images and reference.

    Delegates to :meth:`hst123.primitives.run_dolphot.DolphotPrimitive.make_dolphot_file`.
    """
    self._dolphot.make_dolphot_file(images, reference)

  def run_dolphot(self):
    """
    Execute DOLPHOT using the prepared parameter and image list.

    Delegates to :meth:`hst123.primitives.run_dolphot.DolphotPrimitive.run_dolphot`.
    """
    self._dolphot.run_dolphot()

  @log_calls
  def organize_reduction_tables(self, obstable, byvisit=False):
    """
    Split a full observation table into per-visit sub-tables when requested.

    Parameters
    ----------
    obstable : `astropy.table.Table`
        Table including a ``visit`` column.
    byvisit : bool, optional
        If True, return one table per visit; if False, return a single-element list.

    Returns
    -------
    list of `astropy.table.Table`
        Tables in processing order.
    """
    tables = []
    if byvisit:
        for visit in list(set(obstable['visit'].data)):
            mask = obstable['visit'] == visit
            tables.append(obstable[mask])
    else:
        tables.append(obstable)

    return tables

  @log_calls
  def handle_reference(self, obstable, refname):
    """
    Resolve the drizzled reference image path, building it via :meth:`pick_reference` if needed.

    Parameters
    ----------
    obstable : `astropy.table.Table`
        Current visit table.
    refname : str or None
        User-supplied reference path, or None to generate.

    Returns
    -------
    str or None
        Path to the reference FITS, or None if missing or drizzle failed.
    """
    banner = 'Handling reference image: {0}'
    if refname:
        refname = normalize_fits_path(refname)
    if refname and os.path.isfile(refname):
        make_banner(banner.format(refname))
    else:
        make_banner(banner.format('generating from input files'))
        refname = self.pick_reference(obstable)

    if not refname or not os.path.isfile(refname):
        log.error(
            "Could not build or find reference image (missing inputs or drizzle failure)."
        )
        return None

    banner = 'Sanitizing reference image: {ref}'
    make_banner(banner.format(ref=refname))
    self.sanitize_reference(refname)

    return refname

  def prepare_dolphot(self, image):
    """
    Mask, split, and sky-subtract one exposure for DOLPHOT.

    Delegates to :meth:`hst123.primitives.run_dolphot.DolphotPrimitive.prepare_dolphot`.
    """
    return self._dolphot.prepare_dolphot(image)

  @log_calls
  def drizzle_all(self, obstable, hierarchical=False, clobber=False,
    do_tweakreg=True):
    """
    Drizzle each unique ``drizname`` group (visit/filter/epoch) from *obstable*.

    Parameters
    ----------
    obstable : `astropy.table.Table`
        Must include ``drizname`` and ``image`` columns (from :meth:`input_list`).
    hierarchical : bool, optional
        If True, after per-epoch drizzles, run a second TweakReg pass and apply
        shifts tied to the deepest stacked image.
    clobber : bool, optional
        If False, skip drizzle when the output file already exists. Set True when
        using ``--redo-astrodrizzle``, ``--redo``, or ``--clobber`` (see
        :func:`~hst123.utils.options.want_redo_astrodrizzle`).
    do_tweakreg : bool, optional
        Run TweakReg on each group before AstroDrizzle when alignment is enabled.

    Notes
    -----
    Optional per-drizzle ``calcsky`` for non-reference products is controlled by
    ``HST123_DOLPHOT_SKY_FOR_DRIZZLE_ALL``.

    Groups are processed **sequentially**: each step calls :meth:`run_tweakreg`, which
    sets the process working directory to ``workspace/``; parallelizing groups in
    threads would race on ``chdir``. DOLPHOT prep thread count is set by
    ``--max-cores`` (same as AstroDrizzle workers).
    """
    opt = self.options['args']

    for name in np.unique(obstable['drizname'].data):
        mask = obstable['drizname']==name
        driztable = obstable[mask]

        if os.path.exists(name) and not clobber:
            message = 'Drizzled image {im} exists.  Skipping...'
            log.info(message.format(im=name))
        else:
            message = 'Constructing drizzled image: {im}'
            log.info(message.format(im=name))
            # Run tweakreg on the sub-table to make sure frames are aligned
            if do_tweakreg:
                error, shift_table = self._astrom.run_tweakreg(driztable, '')
            # Next run astrodrizzle to construct the drizzled frame
            if not self.run_astrodrizzle(
                driztable, output_name=name, save_fullfile=True
            ):
                log.error("Drizzle failed for %s; skipping sanitize/sky for this product.", name)
                continue

        self.sanitize_reference(name)

        # By default, only generate DOLPHOT sky for the main reference image.
        # Set HST123_DOLPHOT_SKY_FOR_DRIZZLE_ALL=1 to restore older behavior.
        if opt.run_dolphot and str(
            os.environ.get("HST123_DOLPHOT_SKY_FOR_DRIZZLE_ALL", "")
        ).strip().lower() in ("1", "true", "yes", "on"):
            if self._dolphot.needs_to_calc_sky(name):
                self.compress_reference(name)
                self._dolphot.calc_sky(name, self.options['detector_defaults'])
                sky_image = name.replace('.fits', '.sky.fits')
                noise_name = name.replace('.fits', '.noise.fits')
                shutil.copy(sky_image, noise_name)

    if hierarchical:
        driztable = unique(obstable, keys='drizname')
        # Construct new table with drizzled frames
        driztable = driztable['drizname','instrument','detector','filter',
            'visit']
        driztable.rename_column('drizname','image')

        # Create a dictionary of the central pixel RA/Dec for comparison to
        # post-alignment images
        central = {}

        # Overwrite WCSNAME 'TWEAK' if it exists
        for file in driztable['image']:
            hdu = fits.open(file, mode='update')
            try:
                wi = wcs_image_hdu_index(hdu)
                hdu[wi].header["WCSNAME"] = "TWEAK-ORIG"
                hdu[wi].header["TWEAKSUC"] = 0

                x = hdu[wi].header["NAXIS1"] / 2
                y = hdu[wi].header["NAXIS2"] / 2
                w = wcs_from_fits_hdu(hdu, wi)
                coord = wcs.utils.pixel_to_skycoord(x, y, w, origin=1)

                central[file] = {}
                central[file]["cra"] = coord.ra.degree
                central[file]["cdec"] = coord.dec.degree
            finally:
                hdu.close()

        # Pick deepest drizzled image for reference and run tweakreg
        img = self.pick_deepest_images(driztable['image'])
        deepest = sorted(img, key=lambda im: fits.getval(im, 'EXPTIME'))[-1]

        error, shift_table = self._astrom.run_tweakreg(driztable, deepest,
            do_cosmic=False, skip_wcs=True, search_radius=5.0)

        # Convert xoffset and yoffset values to RAoffset and DECoffset
        log.info("Applying hierarchical shifts to individual frames")
        for row in shift_table:
            # Read-only open + full HDUList for WCS (lookup-table distortion).
            with fits.open(row["file"], mode="readonly") as hdu_ro:
                x = hdu_ro[0].header["NAXIS1"] / 2
                y = hdu_ro[0].header["NAXIS2"] / 2
                w = wcs_from_fits_hdu(hdu_ro, 0)
                coord = wcs.utils.pixel_to_skycoord(x, y, w, origin=1)

            nra = coord.ra.degree
            ndec = coord.dec.degree

            dra = central[row['file']]['cra']-nra
            ddec = central[row['file']]['cdec']-ndec

            mask = obstable['drizname']==row['file']
            filetable = obstable[mask]

            for file in filetable['image']:
                log.info('Applying shift to %s', file)
                hdu = fits.open(file, mode='update')

                # Set HIERARCH=1 so other methods will recognize that the
                # image has been hierarchically aligned and do not run tweakreg
                hdu[0].header['HIERARCH']=1
                hdu[0].header['TWEAKSUC']=1

                for i,h in enumerate(hdu):
                    if ('CRVAL1' in h.header.keys() and
                        'CRVAL2' in h.header.keys()):
                        corig = SkyCoord(h.header['CRVAL1'],
                                         h.header['CRVAL2'], unit='deg')
                        newdec = corig.dec.degree - ddec
                        newra = corig.ra.degree - dra

                        # Update header variable
                        hdu[i].header['CRVAL1PR']=corig.ra.degree
                        hdu[i].header['CRVAL2PR']=corig.dec.degree
                        hdu[i].header['CRVAL1']=newra
                        hdu[i].header['CRVAL2']=newdec
                        hdu[i].header['SHIFTRA']=dra
                        hdu[i].header['SHIFTDEC']=ddec
                        hdu[i].header['SHIFTREF']=row['file']

                    # If wfpc2 copy WCS keys over to mask file
                    if 'c0m' in self._fits.get_instrument(file).lower():
                        maskfile = file.split('_')[0]+'_c1m.fits'
                        if os.path.exists(maskfile):
                            maskhdu = fits.open(maskfile)
                            self._astrom.copy_wcs_keys(hdu[i], maskhdu[i])
                            maskhdu.writeto(maskfile, overwrite=True)

                hdu.close()

        # Flag for testing - exits after hierarchical alignment on drz frames
        # has been performed
        if opt.hierarch_test:
            sys.exit()

        # Now that WCS corrections have been applied, we want to skip this for
        # future runs of tweakreg and astrodrizzle
        self.updatewcs = False


  def get_dolphot_photometry(self, split_images, reference):
    """
    Parse DOLPHOT output for the pipeline coordinate and print summary photometry.

    Delegates to :meth:`hst123.primitives.run_dolphot.DolphotPrimitive.get_dolphot_photometry`.
    """
    self._dolphot.get_dolphot_photometry(split_images, reference)

  @log_calls
  def handle_args(self, parser):
    """
    Parse CLI arguments, normalize work/raw dirs, and apply overrides to ``self.options``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser from :meth:`add_options`.

    Returns
    -------
    argparse.Namespace
        Parsed options (also stored as ``self.options['args']``).
    """
    opt = parser.parse_args()
    if getattr(opt, "redo", False):
        opt.redo_astrometry = True
        opt.redo_astrodrizzle = True
    opt.work_dir, opt.raw_dir = normalize_work_and_raw_dirs(
        opt.work_dir, opt.raw_dir
    )
    ws = pipeline_workspace_dir(opt.work_dir)
    if ws:
        os.makedirs(ws, exist_ok=True)
    self.options['args'] = opt

    # Handle other options
    self.reference = self.options['args'].reference
    if opt.align_only: self.options['global_defaults']['dolphot']['AlignOnly']=1
    if opt.before: self.before=Time(self.options['args'].before)
    if opt.after: self.after=Time(self.options['args'].after)
    if opt.skip_tweakreg: self.updatewcs = False

    # Override drizzled image dimensions
    dim = opt.drizzle_dim
    for det in self.options['detector_defaults'].keys():
        self.options['detector_defaults'][det]['nx']=dim
        self.options['detector_defaults'][det]['ny']=dim

    # If only wide, modify acceptable_filters to those with W, X, or LP
    if opt.only_wide:
        self.options['acceptable_filters'] = [filt for filt in
            self.options['acceptable_filters'] if (filt.upper().endswith('X')
                or filt.upper().endswith('W') or filt.upper().endswith('LP'))]

    if opt.only_filter:
        filts = [f.lower() for f in list(opt.only_filter.split(','))]
        self.options['acceptable_filters'] = [filt for filt in
            self.options['acceptable_filters'] if filt.lower() in filts]

    if opt.fit_sky:
        if opt.fit_sky in [1,2,3,4]:
            self.options['global_defaults']['dolphot']['FitSky']=opt.fit_sky
        else:
            log.warning('--fit-sky %s not allowed.', opt.fit_sky)

    if opt.tweak_search:
        self.options['global_defaults']['search_rad']=opt.tweak_search
    if opt.tweak_min_obj:
        self.options['global_defaults']['minobj']=opt.tweak_min_obj
    if opt.tweak_nbright:
        self.options['global_defaults']['nbright']=opt.tweak_nbright

    if opt.tweak_thresh:
        self.threshold = opt.tweak_thresh

    if opt.dolphot_lim < self.options['global_defaults']['dolphot']['SigFinal']:
        lim = opt.dolphot_lim
        self.options["global_defaults"]["dolphot"]["SigFinal"] = lim
        if lim < self.options["global_defaults"]["dolphot"]["SigFind"]:
            self.options["global_defaults"]["dolphot"]["SigFind"] = lim

    # AstroDrizzle (DrizzlePac) parallelism: default from settings when --max-cores omitted.
    mc = getattr(opt, "max_cores", None)
    if mc is None:
        opt.drizzle_num_cores = settings.default_astrodrizzle_cores()
    else:
        opt.drizzle_num_cores = max(1, int(mc))
    self.options["drizzle_defaults"]["num_cores"] = opt.drizzle_num_cores

    # Check for dolphot scripts and set run-dolphot to False if any of them is
    # not available.  This will prevent errors due to scripts not being in path
    if not self._dolphot.check_for_dolphot():
        log.warning(
            "Dolphot scripts not in path; setting --run-dolphot to False. "
            "If you want to run dolphot, download and compile scripts."
        )
        opt.run_dolphot = False

    return opt


# =============================================================================
# Command-line driver (parse RA/Dec, run pipeline stages from flags)
# =============================================================================


def main():
    """
    Command-line entry point for the full HST123 pipeline.

    Reads ``sys.argv[1:3]`` as right ascension and declination, then applies all
    flags from :meth:`hst123.add_options`. Orchestrates MAST download or raw ingest,
    builds the observation table, alignment, optional drizzle-all, DOLPHOT, and
    catalog scraping according to the selected options.

    Notes
    -----
    Calling ``--help`` prints usage and exits. The function may call ``sys.exit``
    on fatal configuration errors or after ``--hierarch-test``.

    See Also
    --------
    hst123.hst123 : Programmatic use of the same pipeline steps.
    """
    ensure_cli_logging_configured()
    wall_start = time.perf_counter()
    phase_rows: list[tuple[str, float]] = []
    _phase_t = [time.perf_counter()]

    def _phase(name: str) -> None:
        now = time.perf_counter()
        phase_rows.append((name, now - _phase_t[0]))
        _phase_t[0] = now

    hst = hst123()

    # Handle the --help option
    if '-h' in sys.argv or '--help' in sys.argv:
        parser = hst.add_options(usage=hst.usagestring)
        options = parser.parse_args()
        sys.exit()

    # Starting banner
    hst.command = ' '.join(sys.argv)
    make_banner(f'Starting: {hst.command}')

    # Try to parse the coordinate and check if it's acceptable
    if len(sys.argv) < 3:
        log.warning(hst.usagestring)
        sys.exit(1)
    else:
        coord = parse_coord(sys.argv[1], sys.argv[2])
        hst.coord = coord
    if not hst.coord:
        log.warning(hst.usagestring)
        sys.exit(1)
    
    # This is to prevent argparse from choking if dec was not degrees as float
    sys.argv[1] = str(coord.ra.degree) ; sys.argv[2] = str(coord.dec.degree)
    ra = '%7.8f' % hst.coord.ra.degree
    dec = '%7.8f' % hst.coord.dec.degree

    # Handle other options
    opt = hst.handle_args(hst.add_options(usage=hst.usagestring))
    default = hst.options['global_defaults']

    attach_work_dir_log_file(opt.work_dir, process_name="pipeline")

    coord_str = hst.coord.to_string("hmsdms")
    log_pipeline_configuration(
        log,
        opt,
        version=__version__,
        coord_hmsdms=coord_str,
    )
    _phase("config")

    # Handle file downloads - first check what products are available
    hst.productlist = hst.get_productlist(hst.coord, default['radius'])
    _phase("mast_catalog_query")
    if opt.download:
        banner = f'Downloading HST data from MAST for: {ra} {dec} ({coord_str})'
        make_banner(banner)

        if opt.archive:
            hst.dest=None
        else:
            hst.dest = opt.raw_dir

        if opt.raw_dir and not os.path.exists(opt.raw_dir):
            os.makedirs(opt.raw_dir)

        hst.download_files(
            hst.productlist,
            archivedir=opt.archive,
            dest=hst.dest,
            clobber=opt.clobber,
            work_dir=opt.work_dir,
        )
        _phase("mast_download")
    else:
        _phase("mast_download_skipped")

    if opt.archive and not opt.skip_copy:
        log.info("Ingest: archive → work_dir (%s)", opt.work_dir)
        if hst.productlist:
            for product in hst.productlist:
                hst.copy_raw_data_archive(product, archivedir=opt.archive,
                    workdir=opt.work_dir, check_for_coord=True)
        else:
            log.warning("No MAST products to copy from archive.")
    else:
        log.info("Ingest: raw_dir (%s) → work_dir", opt.raw_dir)
        hst.copy_raw_data(opt.raw_dir, reverse=True, check_for_coord=True)
    _phase("ingest")

    # Get input images
    hst.input_images = hst._fits.get_input_images(workdir=opt.work_dir)

    # Check which are HST images that need to be reduced
    make_banner('Checking which images need to be reduced')
    for file in list(hst.input_images):
        warning, needs_reduce = hst.needs_to_be_reduced(file)
        if not needs_reduce:
            log.warning(warning)
            hst.input_images.remove(file)
    _phase("input_filter")

    # Check there are still images that need to be reduced
    if len(hst.input_images)>0:

        # Get metadata on all input images and put them into an obstable
        make_banner('Organizing input images by visit')
        # Going forward, we'll refer everything to obstable for imgs + metadata
        table = hst.input_list(hst.input_images, show=True)
        if table is None or len(table) == 0:
            log.error(
                "No valid FITS inputs after filtering (missing files or bad headers). Exiting."
            )
            sys.exit(1)
        tables = hst.organize_reduction_tables(table, byvisit=opt.by_visit)
        _phase("organize_visits")

        for i,obstable in enumerate(tables):

            vnum = str(i).zfill(4)
            if opt.run_dolphot or opt.scrape_dolphot:
                hst.dolphot = hst._dolphot.make_dolphot_dict(
                    opt.dolphot + vnum, work_dir=opt.work_dir
                )

            hst.reference = hst.handle_reference(obstable, opt.reference)
            if not hst.reference:
                log.error(
                    "Skipping visit %s: no valid reference image (check FITS on disk).",
                    i,
                )
                continue

            # Run main alignment (tweakreg or jhat per --align-with)
            if not opt.skip_tweakreg:
                make_banner('Running main alignment ({})'.format(opt.align_with))
                error, _ = hst._astrom.run_alignment(obstable, hst.reference)
            _phase(f"visit_{i}_alignment")

            # Drizzle all visit/filter pairs if drizzleall
            # Handle this first, especially if doing hierarchical alignment
            if ((opt.drizzle_all or opt.hierarchical) and
                'drizname' in obstable.keys()):
                do_tweakreg = not opt.skip_tweakreg
                hst.drizzle_all(
                    obstable,
                    hierarchical=opt.hierarchical,
                    do_tweakreg=do_tweakreg,
                    clobber=want_redo_astrodrizzle(opt),
                )

            if opt.redrizzle:
                make_banner('Performing redrizzle of all epochs/filters')
                hst.updatewcs = False
                do_tweakreg = not opt.skip_tweakreg
                hst.drizzle_all(obstable, clobber=True,
                    do_tweakreg=do_tweakreg)
            _phase(f"visit_{i}_drizzle_astrodrizzle")

            # dolphot image preparation: mask_image, split_groups, calc_sky
            split_images = []
            if opt.run_dolphot or opt.scrape_dolphot:
                message = 'Preparing dolphot data for files={files}.'
                log.info(message.format(files=','.join(map(str,
                    obstable['image']))))
                prep_list = list(obstable['image'])
                cores = max(1, int(getattr(opt, "drizzle_num_cores", 1) or 1))
                if cores > 1 and len(prep_list) > 1:
                    n = min(cores, len(prep_list))
                    log.info(
                        "DOLPHOT prepare using %d thread(s) for %d image(s) "
                        "(--max-cores)",
                        n,
                        len(prep_list),
                    )
                    with ThreadPoolExecutor(max_workers=n) as pool:
                        split_lists = list(pool.map(hst._dolphot.prepare_dolphot, prep_list))
                    for outimg in split_lists:
                        split_images.extend(outimg)
                else:
                    for image in prep_list:
                        split_images.extend(hst._dolphot.prepare_dolphot(image))
            _phase(f"visit_{i}_dolphot_prepare")

            if os.path.exists(hst.reference):
                hst.compress_reference(hst.reference)
                if opt.run_dolphot:
                    if hst._dolphot.needs_to_calc_sky(hst.reference, check_wcs=True):
                        message = 'Running calcsky for reference image: {ref}'
                        log.info(message.format(ref=hst.reference))
                        hst.compress_reference(hst.reference)
                        hst._dolphot.calc_sky(
                            hst.reference,
                            hst.options["detector_defaults"],
                        )
            _phase(f"visit_{i}_ref_calcsky")

            # Construct dolphot param file from split images and reference
            if opt.run_dolphot:
                banner = 'Adding images to dolphot parameter file: {file}.'
                make_banner(banner.format(file=hst.dolphot['param']))
                hst._dolphot.make_dolphot_file(split_images, hst.reference)

                skip_dolphot = (
                    not want_redo_dolphot(opt)
                    and dolphot_catalog_already_present(hst.dolphot)
                )
                if skip_dolphot:
                    log.info(
                        "Skipping DOLPHOT run: catalog already present at %s. "
                        "Use --redo-dolphot or --redo to re-run.",
                        hst.dolphot["base"],
                    )
                else:
                    log.info("Starting dolphot run...")
                    hst._dolphot.run_dolphot()
            _phase(f"visit_{i}_dolphot_execute")

            # Scrape data from the dolphot catalog for the input coordinates
            if opt.scrape_dolphot: hst._dolphot.get_dolphot_photometry(split_images,
                hst.reference)

            # Do fake star injection if --do-fake is passed
            if opt.do_fake: hst._dolphot.do_fake(obstable, hst.reference)

        _phase("post_visit")

    # Write out a list of the input images with metadata for easy reference
    make_banner('Complete list of input images')
    hst.input_list(hst.input_images, show=True, save=False, file=None)

    # Clean up interstitial files in working directory
    if opt.cleanup:
        message = 'Cleaning up {n} input images.'
        make_banner(message.format(n=len(hst.input_images)))
        for image in hst.input_images:
            message = 'Removing image: {im}'
            log.info(message.format(im=image))
            if os.path.isfile(image):
                os.remove(image)
        # DOLPHOT sky sidecars (*.drc.noise.fits) are not listed in input_images
        n_extra = remove_files_matching_globs(opt.work_dir, settings.cleanup_extra_globs)
        if n_extra:
            log.info(
                "Removed %d DOLPHOT noise sidecar(s) (*.drc.noise.fits) under work dir",
                n_extra,
            )
    _phase("finalize_list_cleanup")

    wall_s = time.perf_counter() - wall_start
    log_pipeline_phase_summary(log, phase_rows, wall_seconds=wall_s)
    message = 'Finished with: {cmd}\n'
    message += 'It took {time} seconds to complete this script.'
    make_banner(message.format(cmd=hst.command, time=wall_s))


if __name__ == '__main__':
    main()
