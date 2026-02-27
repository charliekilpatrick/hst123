#!/usr/bin/env python3

"""
By C. D. Kilpatrick 2019-02-07

v1.00: 2019-02-07. Base hst123 download, tweakreg, drizzle, dolphot param
v1.01: 2019-02-15. Added running dolphot, scraping dolphot output
v1.02: 2019-02-22. Added fake star injection
v1.03: 2019-06-02. Added drizzleall options and cleaned up options/syntax
v1.04: 2020-02-10. Updated to python=3.7
v1.05: 2020-03-04. Fixes to sanitizing reference images for tweakreg/dolphot
v1.06: 2020-03-07. Added archiving capability (--archive dir) for analyzing
                   large volumes of data (see description in "def archive")
v1.07: 2020-03-08. Updates to archiving capability and added the script
                   construct_hst_archive.py to help generate archive.

hst123.py: An all-in-one script for downloading, registering, drizzling,
running dolphot, and scraping data from dolphot catalogs.

Pipeline logic is split into helper classes (_FitsHelper, _PhotometryHelper,
etc.) so the main hst123 class stays a thin orchestrator. Additional helpers
(Download, Reference, Tweakreg, Drizzle, Dolphot, Reduction) can be added
following the same delegation pattern.
"""

"""
Additional note for zero points:
Zero points in hst123 are calculated in the AB mag system by default (this is
different from dolphot, which uses the Vega mag system).  hst123 takes
advantage of the fact that all HST images contain the PHOTFLAM and PHOTPLAM
header keys for each chip, and:
         
    ZP_AB = -2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408

For reference, see the WFPC2, ACS, and WFC3 zero point pages:

WFPC2: http://www.stsci.edu/instruments/wfpc2/Wfpc2_dhb/wfpc2_ch52.html
ACS: http://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
WFC3: http://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration
"""

# Dependencies and settings
import warnings
warnings.filterwarnings('ignore')
import stwcs
import glob
import sys
import os
import shutil
import time
import filecmp
import astroquery
import progressbar
import copy
import requests
import random
import astropy.wcs as wcs
import numpy as np
from contextlib import contextmanager
from astropy import units as u
from astropy.utils.data import clear_download_cache,download_file
from astropy.io import fits
from astropy.table import Table, Column, unique
from astropy.time import Time
from astroscrappy import detect_cosmics
from stwcs import updatewcs
from scipy.interpolate import interp1d

# Internal dependencies
from hst123.utils import options
from hst123 import settings
from hst123.primitives import FitsHelper, PhotometryHelper
from hst123.primitives.astrometry import AstrometryPrimitive, parse_coord
from hst123.utils.logging import get_logger, make_banner

log = get_logger(__name__)

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

with suppress_stdout():
    from drizzlepac import tweakreg,astrodrizzle,catalogs,photeq
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord


# Helpers live in primitives/ (FitsHelper, PhotometryHelper, etc.)


class hst123(object):

  def __init__(self):

    # Basic parameters
    self.input_images = []
    self.split_images = []
    self.fake_images = []
    self.obstable = None

    self.reference = ''
    self.root_dir = '.'
    gd = settings.global_defaults
    self.rawdir = gd['rawdir']
    self.summary = gd['summary']

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

    # List of options
    self.options = {'global_defaults': settings.global_defaults,
                    'detector_defaults': settings.detector_defaults,
                    'instrument_defaults': settings.instrument_defaults,
                    'acceptable_filters': settings.acceptable_filters,
                    'catalog': settings.catalog_pars,
                    'args': None}

    # List of pipeline products in case they need to be cleaned at start
    self.pipeline_products = settings.pipeline_products
    self.pipeline_images = settings.pipeline_images

    # Helpers from primitives/ (logic split by responsibility)
    self._fits = FitsHelper(self)
    self._phot = PhotometryHelper(self)
    self._astrom = AstrometryPrimitive(self)

  def add_options(self, parser=None, usage=None):
    return(options.add_options(parser=parser, usage=usage))

  def clear_downloads(self, options):
    if self.options['args'].no_clear_downloads:
        return(None)
    log.info('Trying to clear downloads')
    try:
        # utils.data.download_file can get buggy if the cache is
        # full.  Clear the cache even though we aren't using caching
        # to prevent download method from choking
        if 'HOME' in os.environ.keys():
            astropath = options['astropath']
            astropy_cache = os.environ['HOME'] + astropath
            log.info('Clearing cache: %s', astropy_cache)
            if os.path.exists(astropy_cache):
                with suppress_stdout():
                    clear_download_cache()
    except RuntimeError:
        warning = 'WARNING: Runtime Error in clear_download_cache().\n'
        warning += 'Passing...'
        pass
    except FileNotFoundError:
        warning = 'WARNING: Cannot run full clear_download_cache().\n'
        warning += 'Passing...'
        pass

  def make_clean(self):
    question = 'Are you sure you want to delete hst123 data? [y/n] '
    var = input(question)
    if var != 'y' and var != 'yes':
        warning = 'WARNING: input={inp}. Exiting...'
        log.warning(warning.format(inp=var))
        sys.exit(1)
    else:
        for pattern in self.pipeline_products:
            for file in glob.glob(pattern):
                if os.path.isfile(file):
                    os.remove(file)
        sys.exit(0)

  def get_zpt(self, image, ccdchip=1, zptype='abmag'):
    return self._fits.get_zpt(image, ccdchip=ccdchip, zptype=zptype)

  def avg_magnitudes(self, magerrs, counts, exptimes, zpt):
    return self._phot.avg_magnitudes(magerrs, counts, exptimes, zpt)

  # Given a column description key and image name, return column number from
  # dolphot column file.  Offset is for interpreting fakes file, which is
  # formatted the same way but has 4 + 2*Nimg extra columns at start
  def get_dolphot_column(self, colfile, key, image, offset=0):
    coldata = ''
    with open(colfile) as colfile_data:
        for line in colfile_data:
            if (image.replace('.fits','') in line and key in line):
                coldata = line.strip().strip('\n')
                break
    if not coldata:
        return(None)
    else:
        try:
            colnum = int(coldata.split('.')[0].strip()) - 1 + offset
            return colnum
        except Exception:
            return None

  # Given a row of a dolphot file and a columns file, return the data key
  # described by 'key' (e.g., 'VEGAMAG', 'Total counts',
  # 'Magnitude uncertainty') for the input image name
  def get_dolphot_data(self, row, colfile, key, image):
    # First get the column number from the start of the column file
    colnum = self.get_dolphot_column(colfile, key, image)
    rdata = row.split()

    # Now grab that piece of data from the row and return
    if colnum is not None:
        if colnum < 0 or colnum > len(rdata)-1:
            error = 'ERROR: tried to use bad column {n} in dolphot output'
            log.error(error.format(n=colnum))
            return(None)
        else:
            return(row.split()[colnum])
    else:
        return(None)

  def print_final_phot(self, final_phot, dolphot, allphot=True):
    for i,phot in enumerate(final_phot):
        outfile = dolphot['final_phot']
        base = dolphot['base']
        if not allphot:
            out = outfile
        else:
            num = str(i).zfill(len(str(len(hst.final_phot))))
            out = outfile.replace('.phot', '_'+num+'.phot')

        snana = out.replace('.phot', '.snana')

        message = 'Photometry for source {n} '
        message = message.format(n=i)
        keys = phot.meta.keys()
        if 'x' in keys and 'y' in keys and 'separation' in keys:
            message += 'at x,y={x},{y}.\n'
            message += 'Separated from input coordinate by {sep} pix.'
            message = message.format(x=phot.meta['x'], y=phot.meta['y'],
                sep=phot.meta['separation'])

        log.info(message)

        with open(out, 'w') as f:
            self.show_photometry(phot, f=f)
        with open(snana, 'w') as f:
            self.show_photometry(phot, f=f, snana=True, show=False)
        log.info('')

  def show_data(self, phottable, form, header, units, f=None, avg=False):

    if avg:
        log.info('\n# Average Photometry')
        if f: f.write('\n# Average Photometry \n')
    else:
        for key in phottable.meta.keys():
            out = '# {key} = {val}'
            if f: f.write(out.format(key=key, val=phottable.meta[key])+'\n')
            log.info(out.format(key=key, val=phottable.meta[key]))

    if header: log.info(header)
    if f and header: f.write(header+'\n')
    if units: log.info(units)
    if f and units: f.write(units+'\n')

    for row in phottable:

        # Format instrument name
        inst = row['INSTRUMENT'].lower()
        if 'wfc3' in inst and 'uvis' in inst: inst='WFC3/UVIS'
        if 'wfc3' in inst and 'ir' in inst: inst='WFC3/IR'
        if 'acs' in inst and 'wfc' in inst: inst='ACS/WFC'
        if 'acs' in inst and 'hrc' in inst: inst='ACS/HRC'
        if 'acs' in inst and 'sbc' in inst: inst='ACS/SBC'
        if 'wfpc2' in inst: inst='WFPC2'
        if '_' in inst: inst.upper().replace('_full','').replace('_','/')

        date='%7.5f'% row['MJD']
        if row['MJD']==99999.0: date='-----------'

        datakeys = {'date':date, 'inst':inst, 'filt':row['FILTER'].upper(),
            'exp':'%7.4f' % row['EXPTIME'], 'mag':'%3.4f' % row['MAGNITUDE'],
            'err':'%3.4f' % row['MAGNITUDE_ERROR']}

        if 'lim' in form:
            datakeys['lim'] = '%3.4f' % row['LIMIT']

        line=form.format(**datakeys)

        log.info(line)
        if f: f.write(line+'\n')

    log.info('')
    f.write('\n')

  def snana(self, phottable, file):

    objname = 'dummy'
    if self.options['args'].object:
        objname = self.options['args'].object

    header = 'SNID: {obj} \nRA: {ra} \nDECL: {dec} \n\n'
    header = header.format(obj=objname, ra=self.coord.ra.degree,
        dec=self.coord.dec.degree)

    file.write(header)
    file.write('VARLIST: MJD FLT FLUXCAL MAG MAGERR \n')

    form = 'OBS: {date: <16} {instfilt: <20} {flux: <16} {fluxerr: <16} '
    form += '{mag: <16} {magerr: <6} \n'
    zpt = self.options['global_defaults']['output_zpt']

    for row in phottable:

        # Format instrument name
        inst = row['INSTRUMENT'].lower()
        if 'wfc3' in inst and 'uvis' in inst: inst='WFC3/UVIS'
        if 'wfc3' in inst and 'ir' in inst: inst='WFC3/IR'
        if 'acs' in inst and 'wfc' in inst: inst='ACS/WFC'
        if 'acs' in inst and 'hrc' in inst: inst='ACS/HRC'
        if 'acs' in inst and 'sbc' in inst: inst='ACS/SBC'
        if 'wfpc2' in inst: inst='WFPC2'
        if '_' in inst: inst.upper().replace('_full','').replace('_','/')

        date='%7.5f'% row['MJD']
        mag = row['MAGNITUDE']
        magerr = row['MAGNITUDE_ERROR']
        flux = 10**(0.4 * (zpt - mag))
        fluxerr = 1./1.086 * magerr * flux

        datakeys = {'date':date, 'instfilt':inst+'-'+row['FILTER'].upper(),
            'flux': flux, 'fluxerr': fluxerr, 'mag': mag, 'magerr': magerr}

        line=form.format(**datakeys)
        file.write(line)

    file.write('\n')

  def show_photometry(self, final_photometry, latex=False, show=True, f=None,
    snana=False):

    # Check to make sure dictionary is set up properly
    keys = final_photometry.keys()
    if ('INSTRUMENT' not in keys or 'FILTER' not in keys
        or 'MJD' not in keys or 'MAGNITUDE' not in keys
        or 'MAGNITUDE_ERROR' not in keys or 'EXPTIME' not in keys):
       error = 'ERROR: photometry table has a key error'
       log.error(error)
       return(None)

    # Split photometry table into the average photometry and everything else
    avg_photometry = final_photometry[final_photometry['IS_AVG']==1]
    if len(avg_photometry)>0:
        final_photometry=final_photometry[final_photometry['IS_AVG']==0]

    if snana: self.snana(avg_photometry, f)

    if latex:

        form = '{date: <10} & {inst: <10} & {filt: <10} '
        form += '{exp: <10} & {mag: <8} & {err: <8} \\\\'
        header = form.format(date='MJD', inst='Instrument', filt='Filter',
            exp='Exposure', mag='Magnitude', err='Uncertainty')
        units = form.format(date='(MJD)', inst='', filt='', exp='(s)', mag='',
            err='') + '\\hline\\hline'

        if final_photometry:
            self.show_data(final_photometry, form, header, units, f=f)
        if avg_photometry:
            self.show_data(avg_photometry, form, header, units, f=f, avg=True)

    if show:

        form = '{date: <12} {inst: <10} {filt: <8} '
        form += '{exp: <14} {mag: <9} {err: <11}'

        headkeys = {'date':'# MJD', 'inst':'Instrument','filt':'Filter',
            'exp':'Exposure','mag':'Magnitude','err':'Uncertainty'}

        if 'LIMIT' in keys:
            form += ' {lim: <10}'
            headkeys['lim']='Limit'

        header = form.format(**headkeys)

        if final_photometry:
            self.show_data(final_photometry, form, header, '', f=f)
        if avg_photometry:
            self.show_data(avg_photometry, form, header, '', f=f, avg=True)

  def get_chip(self, image):
    return self._fits.get_chip(image)

  def try_to_get_image(self, image):

    image = image.lower()

    if not self.coord:
        return(False)

    # Need to guess image properties from input data and image name
    data = {'productFilename': image, 'ra': self.coord.ra.degree}

    inst = ''
    if image.startswith('i') and image.endswith('flc.fits'):
        inst = 'WFC3/UVIS'
    elif image.startswith('i') and image.endswith('flt.fits'):
        inst = 'WFC3/IR'
    elif image.startswith('j') and image.endswith('flt.fits'):
        inst = 'ACS/HRC'
    elif image.startswith('j') and image.endswith('flc.fits'):
        inst = 'ACS/WFC'
    elif image.startswith('u'):
        inst = 'WFPC2'
    else:
        return(False)

    data['instrument_name']=inst

    success, fullfile = self.check_archive(data)

    if success:
        shutil.copyfile(fullfile, image)
    else:
        return(False)


  def input_list(self, img, show=True, save=False, file=None, image_number=[]):
    # Input variables
    zptype = self.magsystem

    # To prevent FileNotFoundError - make sure all images exist and if not then
    # try to download them
    good = []
    for image in img:
        success = True
        if not os.path.exists(image):
            success = self.try_to_get_image(image)
        if success:
            good.append(image)

    if not good:
        return(None)
    else:
        img = copy.copy(good)

    hdu = fits.open(img[0])
    h = hdu[0].header

    # Make a table with all of the metadata for each image.
    exp = [fits.getval(image,'EXPTIME') for image in img]
    if 'DATE-OBS' in h.keys() and 'TIME-OBS' in h.keys():
        dat = [fits.getval(image,'DATE-OBS') + 'T' +
               fits.getval(image,'TIME-OBS') for image in img]
    # This should work if image is missing DATE-OBS, e.g., for drz images
    elif 'EXPSTART' in h.keys():
        dat = [Time(fits.getval(image, 'EXPSTART'),
            format='mjd').datetime.strftime('%Y-%m-%dT%H:%M:%S')
            for image in img]
    fil = [self.get_filter(image) for image in img]
    ins = [self.get_instrument(image) for image in img]
    det = ['_'.join(self.get_instrument(image).split('_')[:2]) for image in img]
    chip= [self.get_chip(image) for image in img]
    zpt = [self.get_zpt(i, ccdchip=c, zptype=zptype) for i,c in zip(img,chip)]

    if not image_number:
        image_number = [0 for image in img]

    # Save this obstable.  Useful for other methods
    obstable = Table([img,exp,dat,fil,ins,det,zpt,chip,image_number],
        names=self.names)

    # Look at the table in order of date
    obstable.sort('datetime')

    # Automatically add visit info
    obstable = self.add_visit_info(obstable)

    # Show the obstable in a column formatted style
    form = '{file: <36} {inst: <18} {filt: <10} '
    form += '{exp: <12} {date: <10} {time: <10}'
    if show:
        header = form.format(file='FILE',inst='INSTRUMENT',filt='FILTER',
                             exp='EXPTIME',date='DATE-OBS',time='TIME-OBS')
        log.info('')
        log.info(header)

        for row in obstable:
            line = form.format(file=os.path.basename(row['image']),
                    inst=row['instrument'].upper(),
                    filt=row['filter'].upper(),
                    exp='%7.4f' % row['exptime'],
                    date=Time(row['datetime']).datetime.strftime('%Y-%m-%d'),
                    time=Time(row['datetime']).datetime.strftime('%H:%M:%S'))
            log.info(line)

        log.info('')

    # Iterate over visit, instrument, filter to add group-specific info
    obstable.add_column(Column([' '*99]*len(obstable), name='drizname'))
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
            drizname = '{obj}.{inst}.{filt}.ut{date}_{n}.drz.fits'
            drizname = drizname.format(inst=inst.split('_')[0],
                filt=filt, n=n, date=date_str, obj=objname)
        else:
            drizname = '{inst}.{filt}.ut{date}_{n}.drz.fits'
            drizname = drizname.format(inst=inst.split('_')[0],
                filt=filt, n=n, date=date_str)

        if self.options['args'].work_dir:
            drizname = os.path.join(self.options['args'].work_dir, drizname)

        obstable[i]['drizname'] = drizname

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

    return(obstable)

  # Check for necessary dolphot scripts in the path
  def check_for_dolphot(self):
    # Complete list of scripts that hst123 needs to run all dolphot cmds
    scripts = ['dolphot','calcsky','acsmask','wfc3mask','wfpc2mask',
        'splitgroups']

    dolphot = True
    for s in scripts:
        if not shutil.which(s):
            dolphot = False

    return(dolphot)

  # Make dictionary with dolphot information
  def make_dolphot_dict(self, dolphot):
    return({ 'base': dolphot, 'param': dolphot+'.param',
             'log': dolphot+'.output', 'total_objs': 0,
             'colfile': dolphot+'.columns', 'fake': dolphot+'.fake' ,
             'fakelist': dolphot+'.fakelist', 'fakelog': dolphot+'.fake.output',
             'radius': 12, 'final_phot': dolphot+'.phot',
             'limit_radius': 10.0,
             'original': dolphot+'.orig'})

  def scrapedolphot(self, coord, reference, images, dolphot, scrapeall=False,
    get_limits=False, brightest=False):

    # Check input variables
    base = dolphot['base'] ; colfile = dolphot['colfile']
    if not os.path.isfile(base) or not os.path.isfile(colfile):
        error = 'ERROR: dolphot output {dp} does not exist.  Use --run-dolphot '
        error += 'or check your dolphot output for errors'
        log.error(error.format(dp=dolphot['base']))
        return(None)

    # Check for reference image and x,y coordinates to scrape data
    if (not reference or not coord):
        error = 'ERROR: Need a reference image and coordinate to '
        error += 'scrape data from the dolphot catalog. Exiting...'
        log.error(error)
        return(None)

    # Copy dolphot catalog into a file if it doesn't exist already
    if not os.path.exists(dolphot['original']):
        shutil.copyfile(dolphot['base'], dolphot['original'])

    if not self.options['args'].no_cuts:

        message = 'Cutting bad sources from dolphot catalog.'
        log.info(message)

        # Cut bad sources
        f = open('tmp', 'w')
        numlines = sum(1 for line in open(base))

        with open(dolphot['base']) as dolphot_file:
            message = 'There are {n} sources in dolphot file {dp}. '
            message += 'Cutting bad sources...'
            log.info(message.format(n=numlines, dp=dolphot['base']))

            bar = progressbar.ProgressBar(maxval=numlines).start()
            typecol = self.get_dolphot_column(colfile, 'Object type', '')
            for i,line in enumerate(dolphot_file):
                bar.update(i)
                if (int(line.split()[typecol])==1): # Obj type
                    f.write(line)
            bar.finish()

        f.close()

        message = 'Done cutting bad sources'
        log.info(message)

        if filecmp.cmp(dolphot['base'], 'tmp'):
            message = 'No changes to dolphot file {dp}.'
            log.info(message.format(dp=dolphot['base']))
            os.remove('tmp')
        else:
            message = 'Updating dolphot file {dp}.'
            log.info(message.format(dp=dolphot['base']))
            shutil.move('tmp', dolphot['base'])

    else:
        # Copy original file back over if it already exists
        if os.path.exists(dolphot['original']):
            shutil.copyfile(dolphot['original'], dolphot['base'])

    # Get x,y coordinate from reference image of potential source
    hdu = fits.open(reference)
    w = wcs.WCS(hdu[0].header)
    x,y = wcs.utils.skycoord_to_pixel(coord, w, origin=1)
    ra = coord.ra.degree
    dec = coord.dec.degree
    radius = dolphot['radius']

    # Check if we need to override dolphot scrape radius
    if self.options['args'].scrape_radius:
        # Calculate what the radius will be for reference image in pixels
        # we can do this pretty easily with a trick from dec
        angradius = self.options['args'].scrape_radius/3600.
        if dec < 89:
            dec1 = coord.dec.degree + angradius
        else:
            dec1 = coord.dec.degree - angradius
        coord1 = SkyCoord(ra, dec1, unit='deg')
        x1,y1 = wcs.utils.skycoord_to_pixel(coord1, w, origin=1)
        radius = np.sqrt((x-x1)**2+(y-y1)**2)

    log.info(
        'Looking for a source around x=%s, y=%s in %s with a radius of %s',
        '%7.2f' % float(x), '%7.2f' % float(y), reference, '%7.4f' % float(radius)
    )

    data = []
    colfile = dolphot['colfile']
    with open(dolphot['base']) as dp:
        for line in dp:
            xcol = self.get_dolphot_column(colfile, 'Object X', '')
            ycol = self.get_dolphot_column(colfile, 'Object Y', '')
            xline = float(line.split()[xcol]) + 0.5
            yline = float(line.split()[ycol]) + 0.5
            dist = np.sqrt((xline-x)**2 + (yline-y)**2)
            sn = self.get_dolphot_data(line, colfile, 'Signal-to-noise', '')
            counts = self.get_dolphot_data(line, colfile,
                'Normalized count rate', '')
            if (dist < radius):
                data.append({'sep':dist, 'data':line, 'sn': float(sn),
                    'counts': float(counts)})

    limit_data = []
    if get_limits:
        limit_radius = dolphot['limit_radius']
        limit_data = self.get_limit_data(dolphot, coord, w, x, y, colfile,
            limit_radius)

    message = 'Done looking for sources in dolphot file {dp}. '
    message += 'hst123 found {n} sources around: {ra} {dec}'
    log.info(message.format(dp=dolphot['base'], n=len(data), ra=ra, dec=dec))

    # What to do for n sources?
    if len(data)==0:
        return(None)
    else:
        warning = 'WARNING: found more than one source. '
        warning += 'Picking {0}'
        # If brightest, reverse sort by counts
        if brightest:
            data = sorted(data, key=lambda obj: obj['counts'], reverse=True)
            warning = warning.format('brightest object')
        # Otherwise sort by separation so closest object is first
        else:
            data = sorted(data, key=lambda obj: obj['sep'])
            warning = warning.format('closest to '+str(ra)+' '+str(dec))
        if not scrapeall:
            log.warning(warning)
            data = [data[0]]
            m='Separation={0}, Signal-to-noise={1}'
            log.info(m.format(data[0]['sep'], data[0]['sn']))

    # Now make an obstable out of images and sort into visits
    obstable = self.input_list(images, show=False, save=False)

    if not obstable:
        return(None)

    # Now get the final photometry for the source and sort
    final_phot=[]
    finished = False
    for i,dat in enumerate(data):

        finished = False
        source_phot = None

        while not finished:
            source_phot = self.parse_phot(obstable, dat['data'], colfile,
                limit_data=limit_data)

            finished = True

            if 'LIMIT' in source_phot.keys():
                if any([np.isnan(row['LIMIT']) for row in source_phot]):
                    if limit_radius < 80:
                        # Re-try with larger radius
                        limit_radius = 2 * limit_radius
                        finished = False

                        limit_data = self.get_limit_data(dolphot, coord, w, x,
                            y, colfile, limit_radius)

        source_phot.meta['separation'] = dat['sep']
        source_phot.meta['magsystem']=self.magsystem

        # Add final data to meta
        if 'x' in source_phot.meta.keys() and 'y' in source_phot.meta.keys():
            x = [float(source_phot.meta['x'])]
            y = [float(source_phot.meta['y'])]
            coord = wcs.utils.pixel_to_skycoord(x, y, w, origin=1)[0]

            source_phot.meta['ra']=coord.ra.degree
            source_phot.meta['dec']=coord.dec.degree

        source_phot.sort(['MJD','FILTER'])

        final_phot.append(source_phot)

    return(final_phot)

  def get_limit_data(self, dolphot, coord, w, x, y, colfile, limit_radius):
    # limit radius in arcseconds
    limit_data = []

    # Get limit radius in pixels
    if coord.dec.degree < 89:
        dec1 = coord.dec.degree + limit_radius/3600.
    else:
        dec1 = coord.dec.degree - limit_radius/3600.

    coord1 = SkyCoord(coord.ra.degree, dec1, unit='deg')
    x1,y1 = wcs.utils.skycoord_to_pixel(coord1, w, origin=1)
    radius = np.sqrt((x-x1)**2+(y-y1)**2)

    with open(dolphot['base']) as dp:
        for line in dp:
            xcol = self.get_dolphot_column(colfile, 'Object X', '')
            ycol = self.get_dolphot_column(colfile, 'Object Y', '')
            xline = float(line.split()[xcol]) + 0.5
            yline = float(line.split()[ycol]) + 0.5
            dist = np.sqrt((xline-x)**2 + (yline-y)**2)
            if (dist < radius):
                limit_data.append([dist, line])

    return(limit_data)

  # Copy raw data into raw data dir
  def copy_raw_data(self, rawdir, reverse=False, check_for_coord=False):
    # reverse=False will backup data in the working directory to rawdir
    if not reverse:
        if not os.path.exists(rawdir):
            os.mkdir(rawdir)
        for f in self.input_images:
            if not os.path.isfile(rawdir+'/'+f):
                # Create new file and change permissions
                shutil.copyfile(f, rawdir+'/'+f)
    # reverse=True will copy files from the rawdir to the working dir
    else:
        for file in glob.glob(rawdir+'/*.fits'):
            # If check_for_coord, only copy files that have target coord
            if check_for_coord:
                warning, check = self.needs_to_be_reduced(file, save_c1m=True)
                if not check:
                    log.warning(warning)
                    continue
            path, base = os.path.split(file)
            # Should catch error where 'base' does not exist
            if os.path.isfile(base) and filecmp.cmp(file, base):
                message = '{file} == {base}'
                log.info(message.format(file=file, base=base))
                continue
            else:
                message = '{file} != {base}'
                log.info(message.format(file=file, base=base))
                shutil.copyfile(file, base)

  # Check the archivedir provided by --archive option.  Allows for organizing,
  # avoiding multiple copies of data files and enable speed ups
  def check_archive(self, product, archivedir=None):

    if not archivedir:
        archivedir = self.options['args'].archive
    if not os.path.exists(archivedir):
        try:
            os.makedirs(archivedir)
        except OSError:
            error = 'ERROR: could not make archive dir {dir}\n'
            error += 'Enable write permissions to this location\n'
            error += 'Exiting...'
            log.error(error.format(dir=archivedir))
            return(False, None)

    # Resolve the filename for product row object
    filefmt = '{inst}/{det}/{ra}/{name}'

    # Parse the product['instrument_name'] value into inst/det
    filename = product['productFilename']
    filename = '_'.join(filename.split('_')[-2:])
    instrument = product['instrument_name']
    ra = str(int(np.round(product['ra'])))

    if 'WFPC2' in instrument.upper() or 'PC/WFC' in instrument.upper():
        inst = 'WFPC2'; det='WFPC2'
        file = filefmt.format(inst=inst, det=det, ra=ra, name=filename)
    else:
        # Then filename should just be inst/det
        inst,det = instrument.split('/')
        file = filefmt.format(inst=inst, det=det, ra=ra, name=filename)

    fullfile = archivedir + '/' + file

    path, basefile = os.path.split(fullfile)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            error = 'ERROR: could not make archive dir {0}\n'
            error += 'Enable write permissions to this location\n'
            error += 'Exiting...'
            log.error(error.format(path))
            return(False, None)

        return(False, fullfile)

    else:
        if os.path.exists(fullfile):
            return(True, fullfile)
        else:
            return(False, fullfile)

  def copy_raw_data_archive(self, product, archivedir=None, workdir=None,
    check_for_coord = False):

    if not archivedir:
        archivedir = self.options['args'].archive
    if not os.path.exists(archivedir):
        warning = 'WARNING: could find archive dir {0}'
        log.warning(warning.format(archivedir))
        return(None)

    # Resolve the filename for product row object
    filefmt = '{inst}/{det}/{ra}/{name}'

    # Parse the product['instrument_name'] value into inst/det
    filename = product['productFilename']
    filename = '_'.join(filename.split('_')[-2:])
    instrument = product['instrument_name']
    ra = str(int(np.round(product['ra'])))

    if 'WFPC2' in instrument.upper() or 'PC/WFC' in instrument.upper():
        inst = 'WFPC2'; det='WFPC2'
        file = filefmt.format(inst=inst, det=det, ra=ra, name=filename)
    else:
        # Then filename should just be inst/det
        inst,det = instrument.split('/')
        file = filefmt.format(inst=inst, det=det, ra=ra, name=filename)

    fullfile = archivedir + '/' + file
    path, basefile = os.path.split(fullfile)

    if not os.path.exists(fullfile):
        warning = 'WARNING: could not find file {0}'
        log.warning(warning.format(fullfile))
        return(None)
    else:
        if check_for_coord:
            warning, check = self.needs_to_be_reduced(fullfile, save_c1m=True)
            if not check:
                log.warning(warning)
                return(None)
        if workdir:
            fulloutfile = os.path.join(workdir, basefile)
        else:
            fulloutfile = basefile

        # Check whether fulloutfile exists and if files are the same
        if os.path.exists(fulloutfile) and filecmp.cmp(fullfile, fulloutfile):
            message = '{file} == {base}'
            log.info(message.format(file=fullfile,base=fulloutfile))
            return(0)
        else:
            message = '{file} != {base}'
            log.info(message.format(file=fullfile,base=fulloutfile))
            shutil.copyfile(fullfile, fulloutfile)
            return(0)

  # For an input obstable, sort all files into instrument, visit, and filter
  # so we can group them together for the final output from dolphot
  def add_visit_info(self, obstable):
    # First add empty 'visit' column to obstable
    obstable['visit'] = [int(0)] * len(obstable)

    # Sort obstable by date so we assign visits in chronological order
    obstable.sort('datetime')

    # Time tolerance for a 'visit' -- how many days apart are obs?
    tol = self.options['global_defaults']['visit']

    # Iterate through each file in the obstable
    for row in obstable:
        inst = row['instrument']
        mjd = Time(row['datetime']).mjd
        filt = row['filter']
        filename = row['image']

        # If this is the first one we're making, assign it to visit 1
        if all([obs['visit'] == 0 for obs in obstable]):
            row['visit'] = int(1)
        else:
            instmask = obstable['instrument'] == inst
            timemask = [abs(Time(obs['datetime']).mjd - mjd) < tol
                            for obs in obstable]
            filtmask = [f == filt for f in obstable['filter']]
            nzero = obstable['visit'] != 0 # Ignore unassigned visits
            mask = [all(l) for l in zip(instmask, timemask, filtmask, nzero)]

            # If no matches, then we need to define a new visit
            if not any(mask):
                # no matches. create new visit number = max(visit)+1
                row['visit'] = int(np.max(obstable['visit']) + 1)
            else:
                # otherwise, confirm that all visits of obstable[mask] are same
                if (len(list(set(obstable[mask]['visit']))) != 1):
                    error = 'ERROR: visit numbers are incorrectly assigned.'
                    log.error(error)
                    return(None)
                else:
                    # visit number is equal to other values in set
                    row['visit'] = list(set(obstable[mask]['visit']))[0]

    return(obstable)

  # Given an input obstable, dolphot data, and colfile, calculate the average
  # mjd, magnitude, and magnitude error
  def calc_avg_stats(self, obstable, data, colfile):

    estr = 'Magnitude uncertainty'
    cstr = 'Measured counts'

    mjds = [] ; err = [] ; counts = [] ; exptime = [] ; filts = []
    det = [] ; zpt = []
    for row in obstable:
        error = self.get_dolphot_data(data, colfile, estr, row['image'])
        count = self.get_dolphot_data(data, colfile, cstr, row['image'])

        if error and count:
            mjds.append(Time(row['datetime']).mjd)
            err.append(error)
            counts.append(count)
            exptime.append(row['exptime'])
            filts.append(row['filter'])
            det.append(row['detector'])
            zpt.append(row['zeropoint'])

    if len(mjds) > 0:
        avg_mjd = np.mean(mjds)
        total_exptime = np.sum(exptime)
        mag, magerr = self.avg_magnitudes(err, counts, exptime, zpt)

        return(avg_mjd, mag, magerr, total_exptime)
    else:
        return(np.nan, np.nan, np.nan, np.nan)

  # Given an input dictionary of files sorted by visits, a row from a dolphot
  # output file, and a column file corresponding to that dolphot file, parse the
  # output into a photometry table
  def parse_phot(self, obstable, row, cfile, limit_data=[]):
    # Names for the final output photometry table
    fnames = ['MJD', 'INSTRUMENT', 'FILTER',
              'EXPTIME', 'MAGNITUDE', 'MAGNITUDE_ERROR', 'IS_AVG']
    init_row = [[0.],['X'*24],['X'*12],[0.],[0.],[0.], [0]]

    if limit_data:
        fnames += ['LIMIT']
        init_row += [[0.]]

    # Make an empty table with above column names for output photometry table
    final_phot = Table(init_row, names=fnames)
    final_phot = final_phot[:0].copy()

    # Get general object statistics to put in final_phot metadata
    x=self.get_dolphot_data(row, cfile, 'Object X', '')
    y=self.get_dolphot_data(row, cfile, 'Object Y', '')
    sharpness=self.get_dolphot_data(row, cfile, 'Object sharpness', '')
    roundness=self.get_dolphot_data(row, cfile, 'Object roundness', '')

    final_phot.meta['x']=x
    final_phot.meta['y']=y
    final_phot.meta['sharpness']=sharpness
    final_phot.meta['roundness']=roundness

    if limit_data:
        limit_message = '{0}-sigma estimate'.format(int(self.snr_limit))
        final_phot.meta['limit']=limit_message

    # Iterate through the dictionary to get all photometry from files
    for inst in list(set(obstable['instrument'])):
        insttable = obstable[obstable['instrument'] == inst]
        for visit in list(set(insttable['visit'])):
            visittable = insttable[insttable['visit'] == visit]
            for filt in list(set(visittable['filter'])):
                ftable = visittable[visittable['filter'] == filt]

                mjd, mag, err, exptime = self.calc_avg_stats(ftable, row, cfile)
                new_row = (mjd, inst, filt, exptime, mag, err, 0)

                if limit_data:
                    mags = [] ; errs = []
                    for data in limit_data:
                        mjd, limmag, limerr, exp = self.calc_avg_stats(ftable,
                            data[1], cfile)

                        if (not np.isnan(limmag) and not np.isnan(limerr)
                            and limmag < 99):
                            mags.append(limmag)
                            errs.append(limerr)

                    if len(mags)>30:
                        limit = self.snr_limit
                        maglimit = self.estimate_mag_limit(mags, errs,
                            limit=limit)
                    else:
                        maglimit = np.nan

                    new_row = (mjd, inst, filt, exptime, mag, err, 0, maglimit)

                final_phot.add_row(new_row)

    # Finally, combine all magnitudes with the same inst/filt into one
    # average magnitude (regardless of visit)
    for inst in list(set(obstable['instrument'])):
        insttable = obstable[obstable['instrument'] == inst]
        for filt in list(set(insttable['filter'])):
            ftable = insttable[insttable['filter'] == filt]

            mjd, mag, err, exptime = self.calc_avg_stats(ftable, row, cfile)

            # Give this row a dummy MJD variable so it will show up last
            new_row = (mjd, inst, filt, exptime, mag, err, 1)

            if limit_data:
                mags = [] ; errs = []
                for data in limit_data:
                    mjd, limmag, limerr, exptime = self.calc_avg_stats(ftable,
                        data[1], cfile)

                    if not np.isnan(limmag) and not np.isnan(limerr):
                        mags.append(limmag)
                        errs.append(limerr)


                if len(mags)>30:
                    limit = self.snr_limit
                    maglimit = self.estimate_mag_limit(mags, errs,
                        limit=limit)
                else:
                    maglimit = np.nan

                limit = self.snr_limit
                maglimit = self.estimate_mag_limit(mags, errs, limit=limit)
                new_row = (mjd, inst, filt, exptime, mag, err, 1, maglimit)

            final_phot.add_row(new_row)

    return(final_phot)

  def estimate_mag_limit(self, mags, errs, limit=3.0):
    return self._phot.estimate_mag_limit(mags, errs, limit=limit)

  # Sanitizes reference header, gets rid of multiple extensions and only
  # preserves science data.
  def sanitize_reference(self, reference):

    # If the reference image does not exist, print an error and return
    if not os.path.exists(reference):
        error = 'ERROR: reference {ref} does not exist!'
        log.error(error.format(ref=reference))
        return(None)

    hdu = fits.open(reference, mode='readonly')

    # Going to write out newhdu
    # Only want science extension from orig reference
    newhdu = fits.HDUList()

    hdr = hdu[0].header
    newhdu.append(hdu[0])
    newhdu[0].name='PRIMARY'

    # Copy over if missing data
    if newhdu[0].data is None:
        if hdu[1].data is not None:
            newhdu[0].data = hdu[1].data

    # COMMENT and HISTORY keys are annoying, so get rid of those
    if 'COMMENT' in newhdu[0].header.keys():
        del newhdu[0].header['COMMENT']
    if 'HISTORY' in newhdu['PRIMARY'].header.keys():
        del newhdu[0].header['HISTORY']

    # Make sure that reference header reflects one extension
    newhdu[0].header['EXTEND']=False

    # Add header variables that dolphot needs: GAIN, RDNOISE, SATURATE
    inst = newhdu[0].header['INSTRUME'].lower()
    opt  = self.options['instrument_defaults'][inst]['crpars']
    for key in ['saturate','rdnoise','gain']:
        if key not in newhdu[0].header.keys():
            newhdu[0].header[key.upper()] = opt[key]

    # Adjust the value of masked pixels to NaN or median pixel value
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

    # Mark as SANITIZE
    newhdu[0].header['SANITIZE']=1

    # Write out to same file w/ overwrite
    newhdu.writeto(reference, output_verify='silentfix', overwrite=True)

  # Compresses reference image into single extension file
  def compress_reference(self, reference):

    # If the reference image does not exist, print an error and return
    if not os.path.exists(reference):
        error = 'ERROR: reference {ref} does not exist!'
        log.error(error.format(ref=reference))
        return(None)

    hdu = fits.open(reference)

    # Going to write out newhdu
    # Only want science extension from orig reference
    newhdu = fits.HDUList()

    hdr = hdu[0].header
    newhdu.append(hdu[0])
    newhdu[0].name='PRIMARY'

    if len(hdu)==1:
        newhdu[0] = hdu[0]
    elif len(hdu)==2:
        newhdu[0] = hdu[1]
        # Add header keys from hdu[0] to newhdu[0]
        for key in hdu[0].header.keys():
            if key not in newhdu[0].header.keys():
                newhdu[0].header[key] = hdu[0].header[key]

    newhdu.writeto(reference, output_verify='silentfix', overwrite=True)


  # Sanitizes wfpc2 data by getting rid of extensions and changing variables.
  def sanitize_wfpc2(self, image):
    hdu = fits.open(image, mode='readonly')

    # Going to write out newhdu Only want science extension from orig reference
    newhdu = fits.HDUList()
    newhdu.append(hdu['PRIMARY'])

    # Change number of extensions to number of science extensions in wfpc2 image
    n = len([h.name for h in hdu if h.name == 'SCI'])
    newhdu['PRIMARY'].header['NEXTEND'] = n

    # Now append science extensions to newhdu
    for h in hdu:
        if h.name == 'SCI':
            newhdu.append(h)

    # Write out to same file w/ overwrite
    newhdu.writeto(image, output_verify='silentfix', overwrite=True)

  def needs_to_be_reduced(self, image, save_c1m=False):

    keep_short = self.options['args'].keep_short
    keep_tdf_down = self.options['args'].keep_tdf_down
    keep_indt = self.options['args'].keep_indt

    if not os.path.exists(image):
        success = self.try_to_get_image(image)
        if not success:
            warning = 'WARNING: {image} does not exist'
            return(warning.format(img=image), False)

    try:
        hdu = fits.open(image, mode='readonly')
        check = False
        for h in hdu:
            if h.data is not None and h.name.upper()=='SCI':
                check = True
    except (OSError, TypeError, AttributeError):
        warning = 'WARNING: {img} is empty or corrupted.  '
        warning += 'Trying to download again...'
        log.warning(warning.format(img=image))

        success = False
        if not self.productlist:
            warning = 'WARNING: could not find or download {img}'
            return(warning.format(img=image), False)

        mask = self.productlist['productFilename']==image
        if self.productlist[mask]==0:
            warning = 'WARNING: could not find or download {img}'
            return(warning.format(img=image), False)

        self.download_files(self.productlist,
            archivedir=self.options['args'].archive, 
            clobber=True)

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
                warning = 'WARNING: could not find or download {img}'
                return(warning.format(img=image), False)

    is_not_hst_image = False
    warning = ''
    detector = ''

    # Check for header keys that we need
    for key in ['INSTRUME','EXPTIME','DATE-OBS','TIME-OBS']:
        if key not in hdu[0].header.keys():
            warning = 'WARNINGS: {key} not in {img} header'
            warning = warning.format(key=key, img=image)
            return(warning, False)

    instrument = hdu[0].header['INSTRUME'].lower()
    if 'c1m.fits' in image and not save_c1m:
        # We need the c1m.fits files, but they aren't reduced as science data
        warning = 'WARNING: do not need to reduce c1m.fits files.'
        return(warning, False)

    if ('DETECTOR' in hdu[0].header.keys()):
        detector = hdu[0].header['DETECTOR'].lower()

    # Check for EXPFLAG=='INDETERMINATE', usually indicating a bad exposure
    if 'EXPFLAG' in hdu[0].header.keys():
        flag = hdu[0].header['EXPFLAG']
        if flag=='INDETERMINATE':
            if not keep_indt:
                warning = f'WARNING: {image} has EXPFLAG==INDETERMINATE'
                return(warning, False)
        elif 'TDF-DOWN' in flag:
            if not keep_tdf_down:
                warning = f'WARNING: {image} has EXPFLAG==TDF-DOWN AT EXPSTART'
                return(warning, False)
        elif flag!='NORMAL':
            warning = f'WARNING: {image} has EXPFLAG=={flag}.'
            return(warning, False)

    # Get rid of exposures with exptime < 20s
    if not keep_short:
        exptime = hdu[0].header['EXPTIME']
        if (exptime < 15):
            warning = f'WARNING: {image} EXPTIME is {exptime} < 20.'
            return(warning, False)

    # Now check date and compare to self.before
    mjd_obs = Time(hdu[0].header['DATE-OBS']+'T'+hdu[0].header['TIME-OBS']).mjd
    if self.before is not None:
        mjd_before = Time(self.before).mjd
        dbefore = self.before.strftime('%Y-%m-%d')
        if mjd_obs > mjd_before:
            warning = f'WARNING: {image} after the input before date {dbefore}.'
            return(warning, False)

    # Same with self.after
    if self.after is not None:
        mjd_after = Time(self.after).mjd
        dafter = self.after.strftime('%Y-%m-%d')
        if mjd_obs < mjd_after:
            warning = f'WARNING: {image} before the input after date {dafter}.'
            return(warning, False)

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
            warning = f'WARNING: {image} does not contain: {ra} {dec}'
            return(warning, False)

    filt = self.get_filter(image).upper()
    if not (filt in self.options['acceptable_filters']):
        warning = f'WARNING: {image} with FILTER={filt} '
        warning += 'does not have an acceptable filter.'
        return(warning, False)

    # Get rid of images that don't match one of the allowed instrument/detector
    # types and images whose extensions don't match the allowed type for those
    # instrument/detector types
    is_not_hst_image = False
    nextend = hdu[0].header['NEXTEND']
    warning = f'WARNING: {image} with INSTRUME={instrument}, '
    warning += f'DETECTOR={detector}, NEXTEND={nextend} is bad.'
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

    return(warning, is_not_hst_image)

  def needs_to_split_groups(self,image):
    # Count number of science extensions in image
    hdu = fits.open(image, mode='readonly')
    total=0
    for h in hdu:
        if h.name=='SCI': total+=1
    return(len(glob.glob(image.replace('.fits', '.chip?.fits'))) != total)

  def needs_to_calc_sky(self, image, check_wcs=False):
    log.info('Checking for %s', image.replace('.fits','.sky.fits'))
    files = glob.glob(image.replace('.fits','.sky.fits'))
    if (len(files) == 0):
        return(True)
    else:
        if check_wcs:
            # Look at the image and sky wcs and check that they are the same
            imhdu = fits.open(image)
            skhdu = fits.open(files[0])

            if len(imhdu)!=len(skhdu):
                return(False)

            check_keys = ['CRVAL1','CRVAL2','CRPIX1','CRPIX2','CD1_1','CD1_2',
                'CD2_1','CD2_2','NAXIS1','NAXIS2']

            for imh,skh in zip(imhdu,skhdu):
                for key in check_keys:
                    if key in list(imh.header.keys()):
                        if key not in list(skh.header.keys()):
                            return(False)
                        if imh.header[key]!=skh.header[key]:
                            return(False)

            return(True)
        else:
            return(False)

  # Make a stripped down version of a wcs header to override WCS header errors
  def make_meta_wcs_header(self, header):
    meta_header = {}
    for key in ['NAXIS','NAXIS1','NAXIS2','CD1_1','CD1_2','CD2_1','CD2_2',
        'CRVAL1','CRVAL2','CRPIX1','CRPIX2','CTYPE1','CTYPE2']:
        meta_header[key]=header[key]

    if meta_header['CTYPE1']=='RA---TAN-SIP': meta_header['CTYPE1']='RA---TAN'
    if meta_header['CTYPE2']=='DEC--TAN-SIP': meta_header['CTYPE2']='DEC--TAN'

    return(meta_header)

  # Check if the image contains input coordinate.  This is somewhat complicated
  # as an image might cover input coordinates, but they land on a bad part of
  # the detector.  So 1) check if coordinate is in image, and 2) check if
  # corresponding DQ file lists this part of image as good pixels.
  def split_image_contains(self, image, coord):

    log.info('Analyzing split image: %s', image)
    hdu = fits.open(image)
    try:
        w = wcs.WCS(hdu[0].header)
    except MemoryError:
        w = wcs.WCS(self.make_meta_wcs_header(hdu[0].header))

    y,x = wcs.utils.skycoord_to_pixel(coord, w, origin=1)

    naxis1,naxis2 = hdu[0].data.shape

    inside_im = False
    if (x > 0 and x < naxis1-1 and
        y > 0 and y < naxis2-1):
        inside_im = True

    return(inside_im)

  # Determine if we need to run the dolphot mask routine
  def needs_to_be_masked(self,image):
    # Masking should remove all of the DQ arrays etc, so make sure that any
    # extensions with data in in them are only SCI extensions. This might not be
    # 100% robust, but should be good enough.
    hdulist = fits.open(image)
    header = hdulist[0].header
    inst = self.get_instrument(image).split('_')[0].upper()
    if inst=='WFPC2':
        if 'DOLWFPC2' in header.keys():
            if header['DOLWFPC2']==0:
                return(False)
    if inst=='WFC3':
        if 'DOL_WFC3' in header.keys():
            if header['DOL_WFC3']==0:
                return(False)
    if inst=='ACS':
        if 'DOL_ACS' in header.keys():
            if header['DOL_ACS']==0:
                return(False)
    return(True)

  def get_filter(self, image):
    return self._fits.get_filter(image)

  def get_instrument(self, image):
    return self._fits.get_instrument(image)

  def get_input_images(self, pattern=None, workdir=None):
    return self._fits.get_input_images(pattern=pattern, workdir=workdir)

  def get_split_images(self, pattern=None, workdir=None):
    return self._fits.get_split_images(pattern=pattern, workdir=workdir)

  def get_dq_image(self, image):
    return self._fits.get_dq_image(image)

  # Run the dolphot splitgroups routine
  def split_groups(self, image, delete_non_science=True):
    log.info('Running split groups for %s', image)
    splitgroups = f'splitgroups {image}'

    log.info('Executing: %s', splitgroups)
    os.system(splitgroups)

    # Delete images that aren't from science extensions
    if delete_non_science:
        split_images = glob.glob(image.replace('.fits','.chip*.fits'))

        for split in split_images:
            hdu = fits.open(split)
            info = hdu[0]._summary()

            if info[0].upper()!='SCI':
                warning = f'WARNING: deleting {split}, not a science extension.'
                log.warning(warning)
                os.remove(split)

  # Run the dolphot mask routine for the input image
  def mask_image(self, image, instrument):
    maskimage = self.get_dq_image(image)
    cmd = f'{instrument}mask {image} {maskimage}'

    log.info('Executing: %s', cmd)
    os.system(cmd)

  # Run the dolphot calcsky routine
  def calc_sky(self, image, options):
    det = '_'.join(self.get_instrument(image).split('_')[:2])
    opt = options[det]['dolphot_sky']
    cmd = 'calcsky {image} {rin} {rout} {step} {sigma_low} {sigma_high}'
    calc_sky = cmd.format(image=image.replace('.fits',''), rin=opt['r_in'],
                            rout=opt['r_out'], step=opt['step'],
                            sigma_low=opt['sigma_low'],
                            sigma_high=opt['sigma_high'])

    log.info('Executing: %s', calc_sky)
    os.system(calc_sky)

  # Write the global dolphot parameters to the dolphot parameter file
  def generate_base_param_file(self, param_file, options, n):
    param_file.write('Nimg = {n}\n'.format(n=n))
    for par, value in options['dolphot'].items():
          param_file.write('{par} = {value}\n'.format(par=par, value=value))

  # Get instrument/detector specific params for input image
  def get_dolphot_instrument_parameters(self, image, options):
    instrument_string = self.get_instrument(image)
    detector_string = '_'.join(instrument_string.split('_')[:2])
    return(options[detector_string]['dolphot'])

  # Write image and image-specific parameters to dolphot parameter file
  def add_image_to_param_file(self,param_file, image, i, options,
    is_wfpc2=False):
    # Add image name to param file
    image_name = 'img{i}_file = {file}\n'
    param_file.write(image_name.format(i=str(i).zfill(4),
        file=os.path.splitext(image)[0]))

    # Now add all image-specific params to param file
    params = self.get_dolphot_instrument_parameters(image, options)
    for par, val in params.items():
        if is_wfpc2:
            # For reference images, rescale these values by 1.5 as on p. 8 of
            # the dolphot/WFPC2 documentation, because dolphot will not
            # automatically recognize the reference image as WFPC2 with WFC
            # pixel scale (0.0996"/pix)
            # http://americano.dolphinsim.com/dolphot/dolphotWFPC2.pdf
            if par in ['RAper','RPSF','apsize']:
                log.info('Adjusting for WFPC2 %s = %s', par, val)
            elif par in ['apsky','RSky','RSky2']:
                log.info('Adjusting for WFPC2 %s = %s', par, val)
        image_par_value = 'img{i}_{par} = {val}\n'
        param_file.write(image_par_value.format(i=str(i).zfill(4),
            par=par, val=val))

  # Given an input list of images, pick images from the same filter and
  # instrument in the best filter for a reference image (wide-band or LP, and
  # preferably in a sensitive filter like F606W or F814W), that has the
  # longest cumulative exposure time.  Optional parameter to avoid images from
  # WFPC2 if possible
  def pick_deepest_images(self, images, reffilter=None, avoid_wfpc2=False,
    refinst=None):
    # Best possible filter for a dolphot reference image in the approximate
    # order I would want to use for a reference image.  You can also use
    # to force the script to pick a reference image from a specific filter.
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
    filts = [self.get_filter(im) for im in images]
    insts = [self.get_instrument(im).replace('_full','').replace('_sub','')
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
            if (self.get_filter(im) in val and
                self.get_instrument(im).split('_')[0] in val):
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
        filt = self.get_filter(im)
        inst = self.get_instrument(im).replace('_full','').replace('_sub','')
        if (filt+'_'+inst == best_filt_inst):
            reference_images.append(im)

    return(reference_images)

  # Pick the best reference out of input images.  Returns the filter of the
  # reference image. Also generates a drizzled image corresponding reference
  def pick_reference(self, obstable):
    # If we haven't defined input images, catch error

    reference_images = self.pick_deepest_images(list(obstable['image']),
        reffilter=self.options['args'].reference_filter,
        avoid_wfpc2=self.options['args'].avoid_wfpc2,
        refinst=self.options['args'].reference_instrument)

    if len(reference_images)==0:
        error = 'ERROR: could not pick a reference image'
        log.error(error)
        return(None)

    best_filt = self.get_filter(reference_images[0])
    best_inst = self.get_instrument(reference_images[0]).split('_')[0]

    vnum = np.min(obstable['visit'].data)
    vnum = str(vnum).zfill(4)

    # Generate photpipe-like output name for the drizzled image
    if self.options['args'].object:
        drizname = '{obj}.{inst}.{filt}.ref_{num}.drz.fits'
        drizname = drizname.format(inst=best_inst, filt=best_filt,
            obj=self.options['args'].object, num=vnum)
    else:
        drizname = '{inst}.{filt}.ref_{num}.drz.fits'
        drizname = drizname.format(inst=best_inst, filt=best_filt, num=vnum)

    reference_images = sorted(reference_images)

    if os.path.exists(drizname):
        hdu = fits.open(drizname)

        # Check for NINPUT and INPUT names
        if 'NINPUT' in hdu[0].header.keys() and 'INPUT' in hdu[0].header.keys():
            # Check that NINPUT and INPUT match what we expect
            ninput = len(reference_images)
            str_input = ','.join([s.split('.')[0] for s in reference_images])

            if (hdu[0].header['INPUT'].startswith(str_input) and
                hdu[0].header['NINPUT']==ninput):
                warning='WARNING: drizzled image {drz} exists.\n'
                warning+='Skipping astrodrizzle...'
                log.warning(warning.format(drz=drizname))
                return(drizname)

    message = 'Reference image name will be: {reference}.\n'
    message += 'Generating from input files: {img}\n\n'
    log.info(message.format(reference=drizname, img=reference_images))

    if self.options['args'].drizzle_add:
        add_images = list(str(self.options['args'].drizzle_add).split(','))
        for image in add_images:
            if os.path.exists(image) and image not in reference_images:
                reference_images.append(image)

    if 'wfpc2' in best_inst and len(obstable)<3:
        reference_images = glob.glob('u*c0m.fits')

    obstable = self.input_list(reference_images, show=True, save=False)
    if not obstable or len(obstable)==0:
        return(None)

    # If number of images is small, try to use imaging from the same instrument
    # and detector for masking
    if len(obstable)<3 and not self.options['args'].no_mask:
        inst = obstable['instrument'][0]
        det = obstable['detector'][0]
        mask = (obstable['instrument']==inst) & (obstable['detector']==det)

        outimage = '{inst}.ref.drz.fits'.format(inst=inst)

        if not opt.skip_tweakreg:
            error, shift_table = self._astrom.run_tweakreg(obstable[mask], '')
        self.run_astrodrizzle(obstable[mask], output_name=outimage,
            clean=False, save_fullfile=True)

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

    if not opt.skip_tweakreg:
        error, shift_table = self._astrom.run_tweakreg(obstable, '')
    self.run_astrodrizzle(obstable, output_name=drizname, save_fullfile=True)

    return(drizname)

  def fix_idcscale(self, image):

    det = '_'.join(self.get_instrument(image).split('_')[:2])

    if 'wfc3' in det:
        hdu = fits.open(image)
        idcscale = self.options['detector_defaults'][det]['idcscale']
        for i,h in enumerate(hdu):
            if 'IDCSCALE' not in hdu[i].header.keys():
                hdu[i].header['IDCSCALE']=idcscale

        hdu.writeto(image, overwrite=True, output_verify='silentfix')

  def fix_phot_keys(self, image):

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

    hdu = fits.open(image, mode='update')
    ref = ref_url.strip('.old')
    outdir = self.options['args'].work_dir
    if not outdir:
        outdir = '.'

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

            fullfile = os.path.join(outdir, ref_file)
            if not os.path.exists(fullfile):
                log.info('Grabbing: %s', fullfile)
                # Try using both old cdbs database and new crds link
                urls = []
                url = self.options['global_defaults']['crds']
                urls.append(url+ref_file)

                url = self.options['global_defaults']['cdbs']
                urls.append(url+ref_url+'/'+ref_file)

                for url in urls:
                    message = f'Downloading file: {url}'
                    log.info(message)
                    try:
                        dat = download_file(url, cache=False,
                            show_progress=False, timeout=120)
                        shutil.move(dat, fullfile)
                        log.info('%s [SUCCESS]', message)
                        break
                    except Exception:
                        log.warning('%s [FAILURE]', message)

            message = f'Setting {image},{i} {key}={fullfile}'
            log.info(message)
            hdu[i].header[key] = fullfile

        # WFPC2 does not have residual distortion corrections and astrodrizzle
        # choke if DGEOFILE is in header but not NPOLFILE.  So do a final check
        # for this part of the WCS keys
        if 'wfpc2' in self.get_instrument(image).lower():
            keys = list(h.header.keys())
            if 'DGEOFILE' in keys and 'NPOLFILE' not in keys:
                del hdu[i].header['DGEOFILE']

    hdu.writeto(image, overwrite=True, output_verify='silentfix')
    hdu.close()

  # Update image wcs using updatewcs routine
  def update_image_wcs(self, image, options, use_db=True):

    hdu = fits.open(image, mode='readonly')
    # Check if tweakreg was successfully run.  If so, then skip
    if 'TWEAKSUC' in hdu[0].header.keys() and hdu[0].header['TWEAKSUC']==1:
        return(True)

    # Check for hierarchical alignment.  If image has been shifted with
    # hierarchical alignment, we don't want to shift it again
    if 'HIERARCH' in hdu[0].header.keys() and hdu[0].header['HIERARCH']==1:
        return(True)

    hdu.close()

    message = 'Updating WCS for {file}'
    log.info(message.format(file=image))

    self.clear_downloads(self.options['global_defaults'])

    change_keys = self.options['global_defaults']['keys']
    inst = self.get_instrument(image).split('_')[0]
    ref_url = self.options['instrument_defaults'][inst]['env_ref']

    self.fix_hdu_wcs_keys(image, change_keys, ref_url)

    # Usually if updatewcs fails, that means it's already been done
    try:
        updatewcs.updatewcs(image, use_db=use_db)
        hdu = fits.open(image, mode='update')
        message = '\n\nupdatewcs success.  File info:'
        log.info(message)
        hdu.info()
        hdu.close()
        self.fix_hdu_wcs_keys(image, change_keys, ref_url)
        self.fix_idcscale(image)
        return True
    except Exception:
        error = 'ERROR: failed to update WCS for image {file}'
        log.error(error.format(file=image))
        return(None)

  # Run the drizzlepac astrodrizzle routine using detector parameters.
  def run_astrodrizzle(self, obstable, output_name = None, ra=None, dec=None,
    clean=None, save_fullfile=False):

    log.info('Starting astrodrizzle')

    n = len(obstable)

    if self.options['args'].work_dir:
        outdir = self.options['args'].work_dir
    else:
        outdir = '.'

    if output_name is None:
        output_name = os.path.join(outdir, 'drizzled.fits')

    if n < 7:
        combine_type = 'minmed'
        combine_nhigh = 0
    else:
        combine_type = 'median'
        combine_nhigh = np.max(int((n-4)/2), 0)

    if self.options['args'].combine_type:
        combine_type = self.options['args'].combine_type

    wcskey = 'TWEAK'

    inst = list(set(obstable['instrument']))
    det = '_'.join(self.get_instrument(obstable[0]['image']).split('_')[:2])
    options = self.options['detector_defaults'][det]

    # Make a copy of each input image so drizzlepac doesn't edit base headers
    tmp_input = []
    for image in obstable['image']:
        tmp = image.replace('.fits','.drztmp.fits')

        # Copy the raw data into a temporary file
        shutil.copyfile(image, tmp)
        tmp_input.append(tmp)

    if self.updatewcs:
        for image in tmp_input:
            det = '_'.join(self.get_instrument(image).split('_')[:2])
            wcsoptions = self.options['detector_defaults'][det]
            self.update_image_wcs(image, wcsoptions, use_db=False)
    elif self.options['args'].skip_tweakreg:
        for image in tmp_input:
            self.clear_downloads(self.options['global_defaults'])

            change_keys = self.options['global_defaults']['keys']
            inst = self.get_instrument(image).split('_')[0]
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

    wht_type = self.options['args'].wht_type

    if clean is not None:
        clean = self.options['args'].cleanup

    if len(tmp_input)==1:
        shutil.copy(tmp_input[0], 'dummy.fits')
        tmp_input.append('dummy.fits')

    log.info('Need to run astrodrizzle for images:')
    self.input_list(obstable['image'], show=True, save=False)

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
            imhdu = fits.open(image)
            added = any([base in image for base in add_im_base])

            for i,h in enumerate(imhdu):
                if not h.name=='DQ':
                    continue

                w = wcs.WCS(h.header)
                y,x = wcs.utils.skycoord_to_pixel(maskcoord, w, origin=1)

                size = self.options['global_defaults']['mask_region_size']
                naxis1,naxis2 = h.data.shape

                outside_im = False
                if ((x+size < 0 or x-size > naxis1-1 or
                    y+size < 0 or y-size > naxis2-1)):
                    if not added:
                        continue
                    else:
                        outside_im = True

                xmin = int(np.max([x-size, 0]))
                ymin = int(np.max([y-size, 0]))
                xmax = int(np.min([x+size, naxis2-1]))
                ymax = int(np.min([y+size, naxis1-1]))

                imhdu[i].data[xmin:xmax, ymin:ymax]

                if any([base in image for base in add_im_base]):
                    log.info('Making outside drizmask: %s', image)
                    if outside_im: imhdu[i].data[:,:]=128
                    else:
                        data = copy.copy(imhdu[i].data[xmin:xmax,ymin:ymax])
                        imhdu[i].data[:,:]=128
                        imhdu[i].data[xmin:xmax,ymin:ymax]=data
                else:
                    log.info('Making inside drizmask: %s', image)
                    imhdu[i].data[xmin:xmax,ymin:ymax]=128

            imhdu.writeto(image, overwrite=True, output_verify='silentfix')

    # Check for TWEAK key in hdu.  If WCSNAME in header but not TWEAK
    # then rename WCSNAME to TWEAK
    for image in tmp_input:
        imhdu = fits.open(image)
        for i,h in enumerate(imhdu):
            head = h.header
            tweak = False ; wcsname = False
            log.info('Checking for tweak keys in header...')
            for key in head.keys():
                if 'WCSNAME' in key and head[key].strip()=='TWEAK': tweak = True

            if not tweak:
                # Rename 'WCSNAME' to 'TWEAK' in rawhdu
                log.info('Changing WCSNAME to TWEAK for %s,%s', image, i)
                imhdu[i].header['WCSNAME']='TWEAK'

        imhdu.writeto(image, overwrite=True, output_verify='silentfix')

    start_drizzle = time.time()

    # Make astrodrizzle use c1m masks when drizzling c0m files
    skymask_cat = None
    with open('skymask_cat','w') as f:
        for file in tmp_input:
            if 'c0m' in file:
                maskfile = file.split('_')[0]+'_c1m.fits'
                if os.path.exists(maskfile):
                    hdu = fits.open(maskfile)
                    for i,h in enumerate(hdu):
                        if h.name=='SCI':
                            # Reset DQ mask for bad columns
                            hdu[i].data[np.where(hdu[i].data==258)]=256
                    hdu.writeto(maskfile, overwrite=True)

                    skymask_cat='skymask_cat'

                    hdu = fits.open(file)
                    exts = []
                    for i,h in enumerate(hdu):
                        if h.name=='SCI':
                            exts.append(str(i))
                    line = file+'{'+','.join(exts)+'},'
                    line += ','.join([maskfile+'['+ext+']' for ext in exts])

                    f.write(line+' \n')

    # Equalize sensitivities for WFPC2 data
    for image in tmp_input:
        if 'wfpc2' not in self.get_instrument(image).lower():
            log.info('Equalizing photometric calibration in %s', image)
            self.fix_phot_keys(image)

            with suppress_stdout():
                photeq.photeq(files=image, readonly=False, ref_phot_ext=1,
                    logfile=os.path.join(outdir, 'photeq.log'))

    rotation = 0.0
    if self.options['args'].no_rotation:
        rotation = None

    logfile_name = os.path.join(outdir, 'astrodrizzle.log')

    if save_fullfile:
        clean=False

    tries = 0
    while tries < 3:
        try:
            log.info('Running astrodrizzle on: %s', ','.join(tmp_input))
            log.info('Output image: %s', output_name)
            dd = settings.drizzle_defaults
            astrodrizzle.AstroDrizzle(tmp_input, output=output_name,
                runfile=logfile_name,
                wcskey=wcskey, context=True, group='', build=False,
                num_cores=dd['num_cores'], preserve=False, clean=clean, skysub=skysub,
                skymethod='globalmin+match', skymask_cat=skymask_cat,
                skystat='mode', skylower=0.0, skyupper=None, updatewcs=False,
                driz_sep_fillval=None, driz_sep_bits=options['driz_bits'],
                driz_sep_wcs=True, driz_sep_rot=rotation,
                driz_sep_scale=options['driz_sep_scale'],
                driz_sep_outnx=options['nx'], driz_sep_outny=options['ny'],
                driz_sep_ra=ra, driz_sep_dec=dec, driz_sep_pixfrac=dd['driz_sep_pixfrac'],
                combine_maskpt=dd['combine_maskpt'], combine_type=combine_type,
                combine_nlow=0, combine_nhigh=combine_nhigh,
                combine_lthresh=-10000, combine_hthresh=None,
                combine_nsigma=dd['combine_nsigma'], driz_cr_corr=True,
                driz_cr=True, driz_cr_snr=dd['driz_cr_snr'], driz_cr_grow=dd['driz_cr_grow'],
                driz_cr_ctegrow=dd['driz_cr_ctegrow'], driz_cr_scale=dd['driz_cr_scale'],
                final_pixfrac=dd['final_pixfrac'], final_fillval=None,
                final_bits=options['driz_bits'], final_units='counts',
                final_wcs=True, final_refimage=None, final_wht_type=wht_type,
                final_rot=rotation, final_scale=pixscale,
                final_outnx=options['nx'], final_outny=options['ny'],
                final_ra=ra, final_dec=dec)
            break
        except FileNotFoundError:
            # Usually happens because of a file missing in astropy cache.
            # Try clearing the download cache and then re-try
            self.clear_downloads(self.options['global_defaults'])
            tries += 1


    message = 'Astrodrizzle took {time} seconds to execute.\n\n'
    log.info(message.format(time = time.time()-start_drizzle))

    if self.options['args'].cleanup:
        for image in tmp_input:
            os.remove(image)

    # Get rid of extra dummy files
    if os.path.exists('dummy.fits'):
        os.remove('dummy.fits')

    # Rename science (sci), weight (wht), and mask (ctx) files
    weight_file = output_name.replace('.fits', '_wht.fits')
    mask_file = output_name.replace('.fits', '_ctx.fits')
    science_file = output_name.replace('.fits', '_sci.fits')

    if os.path.exists(weight_file):
        os.rename(weight_file, output_name.replace('.fits', '.weight.fits'))
        weight_file = output_name.replace('.fits', '.weight.fits')
    if os.path.exists(mask_file):
        os.rename(mask_file, output_name.replace('.fits', '.mask.fits'))
        mask_file = output_name.replace('.fits', '.mask.fits')
    if os.path.exists(science_file):
        os.rename(science_file, output_name)

    # Get comma-separated list of base input files
    ninput = len(tmp_input)
    tmp_input = sorted(tmp_input)
    str_input = ','.join([s.split('.')[0] for s in tmp_input])

    origzpt = self.get_zpt(output_name)

    # Add header keys on drizzled file
    hdu = fits.open(output_name, mode='update')
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
        inst = self.get_instrument(output_name).split('_')[0]
        crpars = self.options['instrument_defaults'][inst]['crpars']
        det = '_'.join(self.get_instrument(output_name).split('_')[:2])
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

    log.info(
        'save_fullfile=%s weight_file=%s (exists=%s) mask_file=%s (exists=%s)',
        save_fullfile, weight_file, os.path.exists(weight_file),
        mask_file, os.path.exists(mask_file),
    )
    if (save_fullfile and os.path.exists(weight_file) and
        os.path.exists(mask_file)):

        hdu = fits.open(output_name)
        hdu.info()

        newhdu = fits.HDUList()

        # Make PRIMARY HDU
        newhdu.append(copy.copy(hdu[0]))
        newhdu[0].data = None
        newhdu[0].header['EXTNAME']='PRIMARY'

        # Make SCI HDU
        newhdu.append(copy.copy(hdu[0]))
        newhdu[1].header['EXTNAME']='SCI'

        # Make WHT HDU
        weight_hdu = fits.open(weight_file)
        newhdu.append(weight_hdu[0])
        newhdu[2].header['EXTNAME']='WHT'
        newhdu[2].header['BUNIT'] = 'UNITLESS'

        # Make CTX HDU
        mask_hdu = fits.open(mask_file)
        newhdu.append(mask_hdu[0])
        newhdu[3].header['EXTNAME']='CTX'
        newhdu[3].header['BUNIT'] = 'UNITLESS'

        # Make HDRTAB HDU
        if 'HDRTAB' in [h.name for h in hdu]:
            newhdu.append(copy.copy(hdu['HDRTAB']))

        newhdu.writeto(output_name.replace('.drz.fits','.drc.fits'),
            overwrite=True, output_verify='silentfix')

    return(True)

  # Run cosmic ray clean
  def run_cosmic(self, image, options, output=None):
    message = 'Cleaning cosmic rays in image: {image}'
    log.info(message.format(image=image))
    hdulist = fits.open(image,mode='readonly')

    if output is None:
        output = image

    for i,hdu in enumerate(hdulist):
        if hdu.name=='SCI':
            mask = np.zeros(hdu.data.shape, dtype=np.bool_)

            crmask, crclean = detect_cosmics(hdu.data.copy().astype('<f4'),
                inmask=mask, readnoise=options['rdnoise'], gain=options['gain'],
                satlevel=options['saturate'], sigclip=options['sig_clip'],
                sigfrac=options['sig_frac'], objlim=options['obj_lim'])

            hdulist[i].data[:,:] = crclean[:,:]

            # Add crmask data to DQ array or DQ image
            if self.options['args'].add_crmask:
                if 'flc' in image or 'flt' in image:
                    # Assume this hdu is corresponding DQ array
                    if len(hdulist)>=i+2 and hdulist[i+2].name=='DQ':
                        hdulist[i+2].data[np.where(crmask)]=4096
                elif 'c0m' in image:
                    maskfile = image.split('_')[0]+'_c1m.fits'
                    if os.path.exists(maskfile):
                        maskhdu = fits.open(maskfile)
                        maskhdu[i].data[np.where(crmask)]=4096
                        maskhdu.writeto(maskfile, overwrite=True)

    # This writes in place
    hdulist.writeto(output, overwrite=True, output_verify='silentfix')
    hdulist.close()


  # Construct a product list from the input coordinate
  def get_productlist(self, coord, search_radius):

    self.clear_downloads(self.options['global_defaults'])

    productlist = None

    # Check for coordinate and exit if it does not exist
    if not coord:
        error = 'ERROR: coordinate was not provided.'
        return(productlist)

    # Define search params and grab all files from MAST
    try:
        if self.options['args'].token:
            log.info('Logging in with token...')
            Observations.login(token=self.options['args'].token)
    except Exception:
        warning = 'WARNING: could not log in with input username/password'
        log.warning(warning)

    try:
        obsTable = Observations.query_region(coord, radius=search_radius)
    except (astroquery.exceptions.RemoteServiceError,
        requests.exceptions.ConnectionError,
        astroquery.exceptions.TimeoutError,
        requests.exceptions.ChunkedEncodingError):
        error = 'ERROR: MAST is not currently working.\nTry again later...'
        log.error(error)
        return productlist

    # Get rid of all masked rows (they aren't HST data anyway)
    obsTable = obsTable.filled()

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

    if self.options['args'].only_filter:
        get_filts = self.options['args'].only_filter.split(',')
        get_filts = [f.lower() for f in get_filts]
        mask = np.array([any([f in row['filters'].lower() for f in get_filts])
            for row in obsTable])
        obsTable = obsTable[mask]

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
            log.error('ERROR: MAST is not currently working. Try again later.')
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
        return(None)

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

    return(productlist)

  def download_files(self, productlist, dest=None, archivedir=None,
    clobber=False):

    if not productlist:
        error = 'ERROR: product list is empty.  Cannot download files.'
        log.error(error)
        return(False)

    n = len(productlist)
    log.info('We need to download %s files', n)

    for i,prod in enumerate(productlist):
        filename = prod['downloadFilename']

        outdir = ''
        if dest:
            outdir = dest
            filename = dest + '/' + filename

        # Check if we enabled the archive option
        if archivedir:
            check, fullfile = self.check_archive(prod, archivedir=archivedir)
            filename = fullfile
            outdir, basefile = os.path.split(fullfile)
            if check and not clobber:
                message = '{file} exists. Skipping...'
                log.info(message.format(file=filename))
                continue
        elif os.path.isfile(filename):
            message = '{file} exists. Skipping...'
            log.info(message.format(file=filename))
            continue

        obsid = prod['obsID']

        message = f'Trying to download ({i+1}/{n}) {filename}'
        log.info(message)

        try:
            with suppress_stdout():
                cache = '.'
                download = Observations.download_products(Table(prod),
                    download_dir=cache, cache=False)
                shutil.move(download['Local Path'][0], filename)

            log.info('%s [SUCCESS]', message)

        except Exception as e:
            log.warning('%s [FAILURE]', message)

    # Clean up mastDownload directory
    if os.path.exists('mastDownload'):
        shutil.rmtree('mastDownload')

    return(True)

  def check_large_reduction(self):
    n = len(self.input_images)
    m = self.options['args'].large_num
    if n > m:
        error = 'ERROR: --no_large_reduction and input list size={n}>{m}'
        log.error(error.format(n=n, m=m))

        # Clean up any files in directory
        for pattern in self.pipeline_products+self.pipeline_images:
            for file in glob.glob(pattern):
                if os.path.isfile(file):
                    os.remove(file)

  def make_dolphot_file(self, images, reference):
    dopt = self.options['detector_defaults']
    gopt = self.options['global_defaults']
    with open(self.dolphot['param'], 'w') as dolphot_file:
        self.generate_base_param_file(dolphot_file, gopt, len(images))

        # Write reference image to param file
        inst = self.get_instrument(reference)
        is_wfpc2 = 'wfpc2' in inst.lower()
        # Check if image is WFPC2 to adjust dolphot parameters
        log.info('Checking reference %s instrument type %s', reference, inst)
        log.info('WFPC2=%s', is_wfpc2)
        self.add_image_to_param_file(dolphot_file, reference, 0, dopt,
            is_wfpc2=is_wfpc2)

        # Write out image-specific params to dolphot file
        for i,image in enumerate(images):
            self.add_image_to_param_file(dolphot_file, image, i+1, dopt)

  def run_dolphot(self):
    if os.path.isfile(self.dolphot['param']):
        # Clean previous runs of dolphot if they exist to avoid conflicts
        if os.path.exists(self.dolphot['base']):
            os.remove(self.dolphot['base'])
        if os.path.exists(self.dolphot['original']):
            os.remove(self.dolphot['original'])

        cmd = 'dolphot {base} -p{par} > {log}'
        cmd = cmd.format(base=self.dolphot['base'], par=self.dolphot['param'],
            log=self.dolphot['log'])
        banner = 'Running dolphot with cmd={cmd}'
        make_banner(banner.format(cmd=cmd))
        os.system(cmd)
        time.sleep(10)
        log.info('dolphot is finished (whew)!')
        if os.path.exists(self.dolphot['base']):
            filesize = os.stat(self.dolphot['base']).st_size/1024/1024
            filesize = '%.3f'%(filesize)
            log.info('Output dolphot file size is %s MB', filesize)
    else:
        error = 'ERROR: dolphot parameter file {file} does not exist!'
        error += ' Generate a parameter file first.'
        log.error(error.format(file=self.dolphot['param']))

  def organize_reduction_tables(self, obstable, byvisit=False):

    tables = []
    if byvisit:
        for visit in list(set(obstable['visit'].data)):
            mask = obstable['visit'] == visit
            tables.append(obstable[mask])
    else:
        tables.append(obstable)

    return(tables)

  def handle_reference(self, obstable, refname):
    # If reference image was not provided then make one
    banner = 'Handling reference image: {0}'
    if refname and os.path.exists(refname):
        make_banner(banner.format(refname))
    else:
        make_banner(banner.format('generating from input files'))
        refname = self.pick_reference(obstable)

    # Sanitize extensions and header variables in reference
    banner = 'Sanitizing reference image: {ref}'
    make_banner(banner.format(ref=refname))
    self.sanitize_reference(refname)

    return(refname)

  def prepare_dolphot(self, image):
    # Mask if needs to be masked
    if self.needs_to_be_masked(image):
        inst = self.get_instrument(image).split('_')[0]
        self.mask_image(image, inst)

    # Split groups if needs split groups
    if self.needs_to_split_groups(image):
        self.split_groups(image)

    # Add split images to the list of split images
    outimg = []
    split_images = glob.glob(image.replace('.fits', '.chip?.fits'))
    for im in split_images:
        if not self.split_image_contains(im, self.coord) and not self.options['args'].include_all_splits:
            # If the image doesn't have coord, delete that file
            os.remove(im)
        else:
            if self.needs_to_calc_sky(im):
                self.calc_sky(im, self.options['detector_defaults'])

            outimg.append(im)

    return(outimg)

  # Drizzle all images in obstable based on individual instruments/filters and
  # observation epochs.  These are segmented by drizname defined in input_list.
  # There is an additional option to perform hierarchical alignment on the
  # sub-frames after all sub-images are drizzled.
  def drizzle_all(self, obstable, hierarchical=False, clobber=False,
    do_tweakreg=True):

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
            self.run_astrodrizzle(driztable, output_name=name,
                save_fullfile=True)

        self.sanitize_reference(name)

        # Make a sky file for the drizzled image and rename 'noise'
        if opt.run_dolphot:
            if (self.needs_to_calc_sky(name)):
                self.compress_reference(name)
                self.calc_sky(name, self.options['detector_defaults'])
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
            hdu[0].header['WCSNAME']='TWEAK-ORIG'
            hdu[0].header['TWEAKSUC']=0

            x = hdu[0].header['NAXIS1']/2
            y = hdu[0].header['NAXIS2']/2
            w = wcs.WCS(hdu[0].header)
            coord = wcs.utils.pixel_to_skycoord(x, y, w, origin=1)

            central[file]={}
            central[file]['cra']=coord.ra.degree
            central[file]['cdec']=coord.dec.degree

            hdu.close()

        # Pick deepest drizzled image for reference and run tweakreg
        img = self.pick_deepest_images(driztable['image'])
        deepest = sorted(img, key=lambda im: fits.getval(im, 'EXPTIME'))[-1]

        error, shift_table = self._astrom.run_tweakreg(driztable, deepest,
            do_cosmic=False, skip_wcs=True, search_radius=5.0)

        # Convert xoffset and yoffset values to RAoffset and DECoffset
        message = '\n\nApplying shifts to individual image frames'
        log.info(message)
        for row in shift_table:
            hdu = fits.open(row['file'], mode='readonly')

            # Repeat process from above to get central RA/Dec and get offset
            x = hdu[0].header['NAXIS1']/2
            y = hdu[0].header['NAXIS2']/2
            w = wcs.WCS(hdu[0].header)
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
                    if 'c0m' in self.get_instrument(file).lower():
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
    ra = self.coord.ra.degree ; dec = self.coord.dec.degree
    make_banner(f'Starting scrape dolphot for: {ra} {dec}')

    opt = self.options['args']
    dp = self.dolphot
    if (os.path.exists(dp['colfile']) and
        os.path.exists(dp['base']) and
        os.stat(dp['base']).st_size>0):

        phot = self.scrapedolphot(self.coord, reference, split_images, dp,
            get_limits=True, scrapeall=opt.scrape_all, brightest=opt.brightest)

        self.final_phot = phot

        if phot:
            message = 'Printing out the final photometry for: {ra} {dec}\n'
            message += 'There is photometry for {n} sources'
            message = message.format(ra=ra, dec=dec, n=len(phot))
            make_banner(message)

            allphot = self.options['args'].scrape_all
            self.print_final_phot(phot, self.dolphot, allphot=allphot)

        else:
            make_banner(f'WARNING: did not find a source for: {ra} {dec}')

    else:
        message = 'WARNING: dolphot did not run.  Use the --run-dolphot flag'
        message += ' or check your dolphot output for errors before using '
        message += '--scrape-dolphot'
        log.warning(message)

  def handle_args(self, parser):
    opt = parser.parse_args()
    self.options['args'] = opt

    # If we're cleaning up a previous run, execute that here then exit
    if self.options['args'].make_clean: self.make_clean()

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
            warning = f'WARNING: --fit-sky {opt.fit_sky} not allowed.'
            log.warning(warning)

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
        self.options['global_defaults']['dolphot']['SigFinal']=lim
        if lim < self.options['global_defaults']['dolphot']['SigFind']:
            self.options['global_defaults']['dolphot']['SigFind']=lim

    # Check for dolphot scripts and set run-dolphot to False if any of them is
    # not available.  This will prevent errors due to scripts not being in path
    if not self.check_for_dolphot():
        warning = 'WARNING: dolphot scripts not in path!  Setting --run-dolphot'
        warning += ' to False.  If you want to run dolphot, download and '
        warning += 'compile scripts!'
        log.warning(warning)
        opt.run_dolphot = False

    return(opt)

  def do_fake(self, obstable, refname):
    dp = self.dolphot
    gopt = self.options['global_defaults']['fake']
    if not os.path.exists(dp['base'] or os.path.getsize(dp['base'])==0):
        warning = 'WARNING: option --do-fake used but dolphot has not been run.'
        log.warning(warning)
        return None

    # If fakefile already exists, check that number of lines==Nstars
    if os.path.exists(dp['fake']):
        flines=0
        with open(dp['fake'], 'r') as fake:
            for i,line in enumerate(fake):
                flines=i+1

    if not os.path.exists(dp['fake']) or flines<gopt['nstars']:
        # Create the fakestar file given input observations
        images = [] ; imgnums = []
        with open(dp['param'], 'r') as param_file:
            for line in param_file:
                if ('_file' in line and 'img0000' not in line):
                    filename = line.split('=')[1].strip()+'.fits'

                    imgnum = line.split('=')[0]
                    imgnum = int(imgnum.replace('img','').replace('_file',''))

                    images.append(filename)
                    imgnums.append(imgnum)

        # Get x,y coordinates of ra,dec
        hdu = fits.open(refname)
        w = wcs.WCS(hdu[0].header)
        x,y = wcs.utils.skycoord_to_pixel(self.coord, w, origin=1)

        with open(dp['fakelist'], 'w') as fakelist:
            magmin = gopt['mag_min']
            dm = (gopt['mag_max'] - magmin)/(gopt['nstars'])
            for i in np.arange(par['nstars']):
                # Each row needs to look like "0 1 x y mag1 mag2... magN" where
                # N=Nimg in original dolphot param file
                line = '0 1 {x} {y} '.format(x=x, y=y)
                for row in obstable:
                    line += str('{mag} '.format(mag=magmin+i*dm))

        # Rewrite the param file with FakeStars and FakeOut param
        self.options['global_defaults']['FakeStars']=dp['fakelist']
        self.options['global_defaults']['FakeOut']=dp['fake']

        with open(dp['param'], 'w') as dfile:
            defaults = self.options['detector_defaults']
            Nimg = len(obstable)
            self.generate_base_param_file(dfile, defaults, Nimg)

            # Write reference image to param file
            self.add_image_to_param_file(dfile, ref, 0, defaults)

            # Write out image-specific params to dolphot file
            for i,row in enumerate(obstable):
                self.add_image_to_param_file(dfile, row['image'], i+1, defaults)

        # Now run dolphot again
        cmd = 'dolphot {base} -p{param} > {log}'
        cmd = cmd.format(base=dp['base'], param=dp['param'], log=dp['fakelog'])
        log.info(cmd)
        os.system(cmd)
        log.info('dolphot fake stars is finished (whew)!')

        # Now parse the output fake file.  We basically want to fit a generic
        # function to the signal-to-noise and magnitude in output fakes
        # Make a fakes table with image name, filter, and 3-sigma limit
        names = ('image', 'filter', 'limit')
        faketable = Table([['X'*100],['X'*20],[0.]], names=names)
        off = 4+2*len(obstable)

        # First iterate through all individual images and get limit
        for row in obstable:
            sn = [] ; mags = []
            sstr = 'Signal-to-noise'
            mstr = 'Instrumental VEGAMAG magnitude'
            scol = self.get_dolphot_column(dp['colfile'], sstr, row['image'],
                offset=off)
            mcol = self.get_dolphot_column(dp['colfile'], mstr, row['image'],
                offset=off)

            if not scol or not mcol:
                continue

            with open(dp['fake'], 'r') as fakefile:
                for line in fakefile:
                    data = line.split()
                    sn.append(float(data[scol]))
                    mags.append(float(data[mcol]))

        for filt in list(set(obstable['filter'])):
            sn = [] ; mags = []
            sstr = 'Signal-to-noise, '+filt.upper()
            mstr = 'Instrumental VEGAMAG magnitude, '+filt.upper()
            scol = self.get_dolphot_column(dp['colfile'], sstr, '', offset=off)
            mcol = self.get_dolphot_column(dp['colfile'], mstr, '', offset=off)

            if not scol or not mcol:
                continue

            with open(dp['fake'], 'r') as fakefile:
                for line in fakefile:
                    data = line.split()
                    sn.append(float(data[scol]))
                    mags.append(float(data[mcol]))

def main():
    # Start timer, create hst123 class obj, parse args
    start = time.time()
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

    # Handle file downloads - first check what products are available
    hst.productlist = hst.get_productlist(hst.coord, default['radius'])
    if opt.download:
        banner = f'Downloading HST data from MAST for: {ra} {dec}'
        make_banner(banner)

        if opt.raw_dir:
            opt.raw_dir = os.path.join(opt.raw_dir, 'raw')

        if opt.archive:
            hst.dest=None
        else:
            hst.dest = opt.raw_dir

        if opt.raw_dir and not os.path.exists(opt.raw_dir):
            os.makedirs(opt.raw_dir)

        hst.download_files(hst.productlist, archivedir=opt.archive,
            dest=hst.dest, clobber=opt.clobber)

    if opt.archive and not opt.skip_copy:
        make_banner('Copying raw data to working dir')
        if hst.productlist:
            for product in hst.productlist:
                hst.copy_raw_data_archive(product, archivedir=opt.archive,
                    workdir=opt.work_dir, check_for_coord=True)
        else:
            make_banner('WARNING: no products to download!')
    else:
        # Assume that all files are in the raw/ data directory
        make_banner('Copying raw data to working dir')
        hst.copy_raw_data(opt.raw_dir, reverse=True, check_for_coord=True)

    # Get input images
    hst.input_images = hst.get_input_images(workdir=opt.work_dir)

    # Check which are HST images that need to be reduced
    make_banner('Checking which images need to be reduced')
    for file in list(hst.input_images):
        warning, needs_reduce = hst.needs_to_be_reduced(file)
        if not needs_reduce:
            log.warning(warning)
            hst.input_images.remove(file)

    # Quit if the number of input files exceeds large reduction limit
    if opt.no_large_reduction: hst.check_large_reduction()

    # Check there are still images that need to be reduced
    if len(hst.input_images)>0:

        # Get metadata on all input images and put them into an obstable
        make_banner('Organizing input images by visit')
        # Going forward, we'll refer everything to obstable for imgs + metadata
        table = hst.input_list(hst.input_images, show=True)
        tables = hst.organize_reduction_tables(table, byvisit=opt.by_visit)

        for i,obstable in enumerate(tables):

            vnum = str(i).zfill(4)
            if opt.run_dolphot or opt.scrape_dolphot:
                hst.dolphot = hst.make_dolphot_dict(opt.dolphot+vnum)

            hst.reference = hst.handle_reference(obstable, opt.reference)

            # Run main tweakreg to register to the reference.  Skipping tweakreg
            # will speed up analysis if only running scrape-dolphot
            if not opt.skip_tweakreg:
                make_banner('Running main tweakreg')
                error, _ = hst._astrom.run_tweakreg(obstable, hst.reference)

            # Drizzle all visit/filter pairs if drizzleall
            # Handle this first, especially if doing hierarchical alignment
            if ((opt.drizzle_all or opt.hierarchical) and
                'drizname' in obstable.keys()):
                do_tweakreg = not opt.skip_tweakreg
                hst.drizzle_all(obstable, hierarchical=opt.hierarchical,
                    do_tweakreg=do_tweakreg, clobber=opt.clobber)

            if opt.redrizzle:
                make_banner('Performing redrizzle of all epochs/filters')
                hst.updatewcs = False
                do_tweakreg = not opt.skip_tweakreg
                hst.drizzle_all(obstable, clobber=True,
                    do_tweakreg=do_tweakreg)

            # dolphot image preparation: mask_image, split_groups, calc_sky
            split_images = []
            if opt.run_dolphot or opt.scrape_dolphot:
                message = 'Preparing dolphot data for files={files}.'
                log.info(message.format(files=','.join(map(str,
                    obstable['image']))))
                for image in obstable['image']:
                    outimg = hst.prepare_dolphot(image)
                    split_images.extend(outimg)

            if os.path.exists(hst.reference):
                hst.compress_reference(hst.reference)
                if opt.run_dolphot:
                    if hst.needs_to_calc_sky(hst.reference, check_wcs=True):
                        message = 'Running calcsky for reference image: {ref}'
                        log.info(message.format(ref=hst.reference))
                        hst.compress_reference(hst.reference)
                        hst.calc_sky(hst.reference,
                            hst.options['detector_defaults'])

            # Construct dolphot param file from split images and reference
            if opt.run_dolphot:
                banner = 'Adding images to dolphot parameter file: {file}.'
                make_banner(banner.format(file=hst.dolphot['param']))
                hst.make_dolphot_file(split_images, hst.reference)

                # Preparing to start dolphot...
                log.info('Preparing to start dolphot run...')
                time.sleep(10)
                hst.run_dolphot()

            # Scrape data from the dolphot catalog for the input coordinates
            if opt.scrape_dolphot: hst.get_dolphot_photometry(split_images,
                hst.reference)

            # Do fake star injection if --do-fake is passed
            if opt.do_fake: hst.do_fake(obstable, hst.reference)

    # Write out a list of the input images with metadata for easy reference
    make_banner('Complete list of input images')
    hst.input_list(hst.input_images, show=True, save=False, file=hst.summary)

    # Clean up interstitial files in working directory
    if opt.cleanup:
        message = 'Cleaning up {n} input images.'
        make_banner(message.format(n=len(hst.input_images)))
        for image in hst.input_images:
            message = 'Removing image: {im}'
            log.info(message.format(im=image))
            if os.path.isfile(image):
                os.remove(image)

    message = 'Finished with: {cmd}\n'
    message += 'It took {time} seconds to complete this script.'
    make_banner(message.format(cmd=hst.command, time=time.time()-start))


if __name__ == '__main__':
    main()
