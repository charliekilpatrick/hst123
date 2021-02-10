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
"""

# Dependencies and settings
import warnings
warnings.filterwarnings('ignore')
import stwcs
import glob, sys, os, shutil, time, filecmp, astroquery, progressbar, copy
import smtplib, datetime, requests, random
import astropy.wcs as wcs
import numpy as np
from contextlib import contextmanager
from scipy import interpolate
from astropy import units as u
from astropy.utils.data import clear_download_cache,download_file
from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time
from astroscrappy import detect_cosmics
from dateutil.parser import parse
from stwcs import updatewcs
from shapely.geometry import Polygon, Point
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# Color strings for download messages
green = '\033[1;32;40m'
red = '\033[1;31;40m'
end = '\033[0;0m'

# Dictionaries with all options for instruments, detectors, and filters
global_defaults = {
    'archive': '/data2/ckilpatrick/hst/archive',
    'astropath': '/.astropy/cache/download/py3/urlmap.dir',
    'keys': ['IDCTAB','DGEOFILE','NPOLEXT',
             'NPOLFILE','D2IMFILE', 'D2IMEXT','OFFTAB'],
    'badkeys': ['ATODFILE','WF4TFILE','BLEVFILE','BLEVDFIL','BIASFILE',
        'BIASDFIL','DARKFILE','DARKFDIL','FLATFILE','FLATDFIL','SHADFILE',
        'PHOTTAB','GRAPHTAB','COMPTAB','IDCTAB','OFFTAB','DGEOFILE',
        'MASKCORR','ATODCORR','WF4TCORR','BLEVCORR','DARKCORR','FLATCORR',
        'SHADCORR','DOSATMAP','DOPHOTOM','DOHISTOS','DRIZCORR','OUTDTYPE'],
    'cdbs': 'ftp://ftp.stsci.edu/cdbs/',
    'mast': 'https://mast.stsci.edu/api/v0/download/file?uri=',
    'crds': 'https://hst-crds.stsci.edu/unchecked_get/references/hst/',
    'visit': 1,
    'search_rad': 3.0, # for tweakreg in arcsec
    'radius': 5 * u.arcmin,
    'nbright': 5000,
    'minobj': 10,
    'dolphot': {'FitSky': 2,
                'SkipSky': 2,
                'RCombine': 1.5,
                'SkySig': 2.25,
                'SecondPass': 5,
                'SigFindMult': 0.85,
                'MaxIT': 25,
                'NoiseMult': 0.10,
                'FSat': 0.999,
                'ApCor': 1,
                'RCentroid': 2,
                'PosStep': 0.25,
                'dPosMax': 2.5,
                'SigPSF': 10.0,
                'PSFres': 1,
                'Align': 4,
                'Rotate': 1,
                'ACSuseCTE': 0,
                'WFC3useCTE': 0,
                'WFPC2useCTE': 1,
                'FlagMask': 7,
                'SigFind': 2.5,
                'SigFinal': 3.5,
                'UseWCS': 1,
                'AlignOnly': 0,
                'AlignIter': 3,
                'AlignTol': 0.5,
                'AlignStep': 0.2,
                'VerboseData': 1,
                'NegSky': 0,
                'Force1': 0,
                'DiagPlotType': 'PNG',
                'InterpPSFlib': 1,
                'FakeMatch': 3.0,
                'FakePSF': 2.0,
                'FakeStarPSF': 1,
                'FakePad': 0},
    'fake': {'mag_min': 18.0,
             'mag_max': 29.5,
             'nstars': 50000}}

catalog_pars = {
    'skysigma':3.,
    'computesig':False,
    'conv_width':3.5,
    'sharplo':0.2,
    'sharphi':1.0,
    'roundlo':-1.0,
    'roundhi':1.0,
    'peakmin':None,
    'peakmax':None,
    'fluxmin':None,
    'fluxmax':None,
    'nsigma':1.5,
    'ratio':1.0,
    'theta':0.0
}

instrument_defaults = {
    'wfc3': {'env_ref': 'iref.old',
             'crpars': {'rdnoise': 6.5,
                        'gain': 1.0,
                        'saturate': 70000.0,
                        'sig_clip': 4.0,
                        'sig_frac': 0.2,
                        'obj_lim': 6.0}},
    'acs': {'env_ref': 'jref.old',
            'crpars': {'rdnoise': 6.5,
                       'gain': 1.0,
                       'saturate': 70000.0,
                       'sig_clip': 3.0,
                       'sig_frac': 0.1,
                       'obj_lim': 5.0}},
    'wfpc2': {'env_ref': 'uref',
              'crpars': {'rdnoise': 10.0,
                         'gain': 7.0,
                         'saturate': 27000.0,
                         'sig_clip': 4.0,
                         'sig_frac': 0.3,
                         'obj_lim': 6.0}}}

detector_defaults = {
    'wfc3_uvis': {'driz_bits': 96, 'nx': 5200, 'ny': 5200,
                  'input_files': '*_flc.fits', 'pixel_scale': 0.04,
                  'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                  'sigma_low': 2.25, 'sigma_high': 2.00},
                  'dolphot': {'apsky': '15 25', 'RAper': 3, 'RChi': 2.0,
                              'RPSF': 13, 'RSky': '15 35',
                              'RSky2': '4 10'},
                   'idcscale': 0.03962000086903572},
    'wfc3_ir': {'driz_bits': 576, 'nx': 5200, 'ny': 5200,
                'input_files': '*_flt.fits', 'pixel_scale': 0.0642,
                'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '8 20', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 15, 'RSky': '8 20',
                            'RSky2': '3 10'},
                   'idcscale': 0.1282500028610229},
    'acs_wfc': {'driz_bits': 96, 'nx': 5200, 'ny': 5200,
                'input_files': '*_flc.fits', 'pixel_scale': 0.05,
                'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 10, 'RSky': '15 35',
                            'RSky2': '3 6'}},
    'acs_hrc': {'driz_bits': 0, 'nx': 5200, 'ny': 5200,
                'input_files': '*_flt.fits', 'pixel_scale': 0.05,
                'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 10, 'RSky': '15 35',
                            'RSky2': '3 6'}},
    'wfpc2_wfpc2': {'driz_bits': 1032, 'nx': 5200, 'ny': 5200,
                    'input_files': '*_c0m.fits', 'pixel_scale': 0.046,
                    'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                    'sigma_low': 2.25, 'sigma_high': 2.00},
                    'dolphot': {'apsky': '15 25', 'RAper': 3, 'RChi': 2,
                                'RPSF': 13, 'RSky': '15 35',
                                'RSky2': '4 10'}}}

acceptable_filters = {
    'F220W','F250W','F330W','F344N','F435W','F475W','F550M','F555W',
    'F606W','F625W','F660N','F660N','F775W','F814W','F850LP','F892N',
    'F098M','F105W','F110W','F125W','F126N','F127M','F128N','F130N','F132N',
    'F139M','F140W','F153M','F160W','F164N','F167N','F200LP','F218W','F225W',
    'F275W','F280N','F300X','F336W','F343N','F350LP','F373N','F390M','F390W',
    'F395N','F410M','F438W','F467M','F469N','F475X','F487N','F547M',
    'F600LP','F621M','F625W','F631N','F645N','F656N','F657N','F658N','F665N',
    'F673N','F680N','F689M','F763M','F845M','F953N','F122M','F160BW','F185W',
    'F218W','F255W','F300W','F375N','F380W','F390N','F437N','F439W','F450W',
    'F569W','F588N','F622W','F631N','F673N','F675W','F702W','F785LP','F791W',
    'F953N','F1042M'}

"""
Zeropoints in hst123 are calculated in the AB mag system by default (this is
different from dolphot, which uses the Vega mag system).  hst123 takes
advantage of the fact that all HST images contain the PHOTFLAM and PHOTPLAM
header keys for each chip, and:
         ZP_AB = -2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408

For reference, see the WFPC2, ACS, and WFC3 zero point pages:

    WFPC2: http://www.stsci.edu/instruments/wfpc2/Wfpc2_dhb/wfpc2_ch52.html
    ACS: http://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
    WFC3: http://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/
          photometric-calibration
"""

class hst123(object):

  def __init__(self):

    # Basic parameters
    self.input_images = []
    self.split_images = []
    self.fake_images = []
    self.obstable = None

    self.reference = ''
    self.root_dir = '.'
    self.rawdir = 'raw'
    self.summary = 'exposure_summary.out'

    self.usagestring = 'hst123.py ra dec'
    self.command = ''

    self.before = None
    self.after = None
    self.coord = None

    self.productlist = None

    self.keepshort = False
    self.nocleanup = False
    self.updatewcs = True
    self.archive = False
    self.keep_objfile = False

    self.magsystem = 'abmag'

    # Detection threshold used for image alignment by tweakreg
    self.threshold = 15.

    # S/N limit for calculating limiting magnitude
    self.snr_limit = 3.0

    self.dolphot = {}

    # Names for input image table
    self.names = ['image','exptime','datetime','filter','instrument',
        'detector','zeropoint','chip','imagenumber']

    # Names for the final output photometry table
    final_names = ['MJD', 'INSTRUMENT', 'FILTER',
                   'EXPTIME', 'MAGNITUDE', 'MAGNITUDE_ERROR']

    # Make an empty table with above column names for output photometry table
    self.final_phot = Table([[0.],['INSTRUMENT'],['FILTER'],[0.],[0.],[0.]],
        names=final_names)[:0].copy()

    # List of options
    self.options = {'global_defaults': global_defaults,
                    'detector_defaults': detector_defaults,
                    'instrument_defaults': instrument_defaults,
                    'acceptable_filters': acceptable_filters,
                    'catalog': catalog_pars,
                    'args': None}

    # List of pipeline products in case they need to be cleaned at start
    self.pipeline_products = ['*chip?.fits', '*chip?.sky.fits',
                              '*rawtmp.fits', '*drz.fits', '*drz.sky.fits',
                              '*idc.fits', '*dxy.fits', '*off.fits',
                              '*d2im.fits', '*d2i.fits', '*npl.fits',
                              'dp*', '*.log', '*.output','*sci?.fits',
                              '*wht.fits','*sci.fits','*StaticMask.fits']

    self.pipeline_images = ['*flc.fits','*flt.fits','*c0m.fits','*c1m.fits']

  def add_options(self, parser=None, usage=None):
    import argparse
    if parser == None:
        parser = argparse.ArgumentParser(usage=usage,conflict_handler='resolve')
    parser.add_argument('ra', type=str,
        help='Right ascension to reduce the HST images')
    parser.add_argument('dec', type=str,
        help='Declination to reduce the HST images')
    parser.add_argument('--makeclean', default=False, action='store_true',
        help='Clean up all output files from previous runs then exit.')
    parser.add_argument('--download', default=False, action='store_true',
        help='Download the raw data files given input ra and dec.')
    parser.add_argument('--keepshort', default=False, action='store_true',
        help='Keep image files that are shorter than 20 seconds.')
    parser.add_argument('--before', default=None, type=str,
        help='Reject obs after this date.')
    parser.add_argument('--after', default=None, type=str,
        help='Reject obs before this date.')
    parser.add_argument('--clobber', default=False, action='store_true',
        help='Overwrite files when using download mode.')
    parser.add_argument('--reference','--ref', default='',
        type=str, help='Name of the reference image.')
    parser.add_argument('--rundolphot', default=False, action='store_true',
        help='Run dolphot as part of this hst123 run.')
    parser.add_argument('--alignonly', default=False, action='store_true',
        help='Set AlignOnly=1 when running dolphot.')
    parser.add_argument('--dolphot','--dp', default='dp', type=str,
        help='Name of the dolphot output file.')
    parser.add_argument('--scrapedolphot','--sd', default=False,
        action='store_true', help='Scrape photometry from the dolphot '+\
        'catalog from the input RA/Dec.')
    parser.add_argument('--reffilter', default=None, type=str,
        help='Use this filter for the reference image if available.')
    parser.add_argument('--refinst', default=None, type=str,
        help='Use this instrument for the reference image if available.')
    parser.add_argument('--dofake','--df', default=False,
        action='store_true', help='Run fake star injection into dolphot. '+\
        'Requires that dolphot has been run, and so files are taken from the '+\
        'parameters in dolphot output from the current directory rather than '+\
        'files derived from the current run.')
    parser.add_argument('--nocleanup', default=False, action='store_true',
        help='Dont clean up interstitial image files (i.e., flt,flc,c1m,c0m).')
    parser.add_argument('--drizzleall', default=False, action='store_true',
        help='Drizzle all visit/filter pairs together.')
    parser.add_argument('--object', default=None, type=str,
        help='Change the object name in all science files to value and '+\
        'use that value in the filenames for drizzled images.')
    parser.add_argument('--archive', default=None, type=str,
        help='Download and save raw data to an archive directory instead of '+\
        'same folder as reduction (see global_defaults[\'archive\'])')
    parser.add_argument('--workdir', default=None, type=str,
        help='Use the input working directory rather than the current dir')
    parser.add_argument('--keep_objfile', default=False, action='store_true',
        help='Keep the object file output from tweakreg.')
    parser.add_argument('--skip_tweakreg', default=False, action='store_true',
        help='Skip running tweakreg.')
    parser.add_argument('--drizdim', default=5200, type=int,
        help='Override the dimensions of drizzled images.')
    parser.add_argument('--drizscale', default=None, type=float,
        help='Override the pixel scale of drizzled images (units are arcsec).')
    parser.add_argument('--scrapeall', default=False, action='store_true',
        help='Scrape all candidate counterparts within the scrape radius '+\
        '(default=2 pixels) and output to files dpXXX.phot where XXX is '+\
        'zero-padded integer for labeling sources in order of proximity to '+\
        'input coordinate.')
    parser.add_argument('--token', default=None, type=str,
        help='Input a token for astroquery.mast.Observations.')
    parser.add_argument('--scraperadius', default=None, type=float,
        help='Override the dolphot scrape radius (units are arcsec).')
    parser.add_argument('--alert', default=None, type=str,
        help='Include a configuration file to send an alert via email '+\
        'once the script has finished.')
    parser.add_argument('--nocuts', default=False, action='store_true',
        help='Skip cuts to dolphot output file.')
    parser.add_argument('--onlywide', default=False, action='store_true',
        help='Only reduce wide-band filters.')
    parser.add_argument('--brightest', default=False, action='store_true',
        help='Sort output source files by signal-to-noise in reference image.')
    parser.add_argument('--no_large_reduction', default=False,
        action='store_true', help='Exit if input list is >large_num images.')
    parser.add_argument('--large_num', default=200, type=int,
        help='Large number of images to skip when --no_large_reduction used.')
    parser.add_argument('--combine_type', default=None, type=str,
        help='Override astrodrizzle combine_type with input.')
    parser.add_argument('--sky_sub', default=False, action='store_true',
        help='Use sky subtraction in astrodrizzle.')
    parser.add_argument('--wht_type', default='EXP', type=str,
        help='final_wht_type parameter for astrodrizzle.')
    parser.add_argument('--drizadd', default=None, type=str,
        help='Comma-separated list of images to add to the drizzled reference'+\
        ' image.  Use this to inject data from other instruments, filters, '+\
        'etc. if they would not be selected by pick_best_reference.')
    parser.add_argument('--drizmask', default=None, type=str,
        help='Mask out pixels in a box around input mask coordinate in '+\
        'drizzled images but outside box in images from drizadd.')
    parser.add_argument('--no_clear_downloads', default=False,
        action='store_true', help='Suppress the clear_downloads method.')
    parser.add_argument('--fixzpt', default=None, type=float,
        help='Fix the zero point of drizzled images to input value '+\
        '(accounting for combined EXPTIME in header).')
    parser.add_argument('--no_nan', default=False, action='store_true',
        help='Set nan values in drizzled images to median pixel value.')
    parser.add_argument('--skip_copy', default=False, action='store_true',
        help='Skip copying files from archive if --archive is used.')
    parser.add_argument('--byvisit', default=False, action='store_true',
        help='Reduce images by visit number.')
    parser.add_argument('--no_rotation', default=False, action='store_true',
        help='When drizzling, do not rotate to PA=0 degrees but preserve '+\
        'the original position angle.')
    parser.add_argument('--only_filter', default=None, type=str,
        help='List of filters that will be used to update acceptable filters.')
    parser.add_argument('--fitsky', default=None, type=int,
        help='Change the dolphot FitSky parameter to something other than 2.')
    parser.add_argument('--no_mask', default=False, action='store_true',
        help='Do not add extra masking based on other input files.')
    return(parser)

  def clear_downloads(self, options):
    if not self.options['args'].no_clear_downloads:
        return(None)
    print('Trying to clear downloads')
    try:
        # utils.data.download_file can get buggy if the cache is
        # full.  Clear the cache even though we aren't using caching
        # to prevent download method from choking
        if 'HOME' in os.environ.keys():
            astropath = options['astropath']
            astropy_cache = os.environ['HOME'] + astropath
            print('Clearing cache: {0}'.format(astropy_cache))
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

  # Make sure all standard output is formatted in the same way with banner
  # messages for each module
  def make_banner(self, message):
    print('\n\n'+message+'\n'+'#'*80+'\n'+'#'*80+'\n\n')

  def make_clean(self):
    question = 'Are you sure you want to delete hst123 data? [y/n] '
    var = input(question)
    if var != 'y' and var != 'yes':
        warning = 'WARNING: input={inp}. Exiting...'
        print(warning)
    else:
        for pattern in self.pipeline_products:
            for file in glob.glob(pattern):
                if os.path.isfile(file):
                    os.remove(file)

    sys.exit(0)

  def get_zpt(self, image, ccdchip=1, zptype='abmag'):
    # For a given image and optional ccdchip, determine the photometric zero
    # point in AB mag from PHOTFLAM and PHOTPLAM.
    # ZP_AB = -2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408
    hdu = fits.open(image, mode='readonly')
    inst = self.get_instrument(image).lower()
    use_hdu = None
    zpt = None

    # Get hdus that contain PHOTPLAM and PHOTFLAM
    sci = []
    for i,h in enumerate(hdu):
        keys = list(h.header.keys())
        if ('PHOTPLAM' in keys and 'PHOTFLAM' in keys):
            sci.append(h)

    if len(sci)==1: use_hdu=sci[0]
    elif len(sci)>1:
        chips = []
        for h in sci:
            if 'acs' in inst or 'wfc3' in inst:
                if 'CCDCHIP' in h.header.keys():
                    if h.header['CCDCHIP']==ccdchip:
                        chips.append(h)
            else:
                if 'DETECTOR' in h.header.keys():
                    if h.header['DETECTOR']==ccdchip:
                        chips.append(h)
        if len(chips)>0: use_hdu = chips[0]

    if use_hdu:
        photplam = float(use_hdu.header['PHOTPLAM'])
        photflam = float(use_hdu.header['PHOTFLAM'])
        if 'ab' in zptype:
            zpt = -2.5*np.log10(photflam)-5*np.log10(photplam)-2.408
        elif 'st' in zptype:
            zpt = -2.5*np.log10(photflam)-21.1

    return(zpt)

  def parse_polygon(self, line, closed=True):
    repd = {'j2000 ': '', 'gsc1 ': '', 'icrs ': '', 'multi': '',
            'polygon': '', ')': '', '(': '', 'other': ''}

    line = line.lower()
    for old, new in repd.items():
        line = line.replace(old, new)

    line = line.strip().replace(' ', ',')
    line = line.replace(',,,', ',')
    line = line.replace(',,',',')

    polyline = np.array(line.strip().split(','), dtype=float)

    if closed:
        if False in polyline[:2] == polyline[-2:]:
            polyline = np.append(polyline, polyline[:2])

    poly = Polygon(polyline.reshape(len(polyline) // 2, 2))

    return(poly)

  def parse_circle(self, line):
    repd = {'j2000 ': '', 'gsc1 ': '', 'icrs ': '', 'multi': '',
            'circle': '', ')': '', '(': '', 'other': ''}

    line = line.lower()
    for old, new in repd.items():
        line = line.replace(old, new)

    line = line.strip().replace(' ', ',')
    line = line.replace(',,,', ',')
    line = line.replace(',,',',')

    polyline = np.array(line.strip().split(','), dtype=float)

    if len(polyline)==3:
        p = Point(polyline[0], polyline[1])
        circle = p.buffer(polyline[2])

        return(circle)

    else:
        return(None)

  def region_contains_coord(self, s_region, coord):
    s_region = s_region.strip().lower()
    region = None
    if 'polygon' in s_region:
        region = self.parse_polygon(s_region)
    if 'circle' in s_region:
        region = self.parse_circle(s_region)

    if not region:
        return(False)

    point = Point(coord.ra.degree, coord.dec.degree)

    return(region.contains(point))

  def avg_magnitudes(self, magerrs, counts, exptimes, zpt):
    # Mask out bad values
    idx = []
    for i in np.arange(len(magerrs)):
        try:
            if (float(magerrs[i]) < 0.5 and float(counts[i]) > 0.0 and
                float(exptimes[i]) > 0.0 and float(zpt[i]) > 0.0):
                idx.append(i)
        except:
            pass

    if not idx:
        return (float('NaN'),float('NaN'))

    magerrs = np.array([float(m) for m in magerrs])[idx]
    counts = np.array([float(c) for c in counts])[idx]
    exptimes = np.array([float(e) for e in exptimes])[idx]
    zpt = np.array([float(z) for z in zpt])[idx]

    # Normalize flux and fluxerr to a common zero point
    flux = counts / exptimes * 10**(0.4 * (27.5 - zpt))
    fluxerr = 1./1.086 * magerrs * flux

    # Calculate average and propagate uncertainties weighted by fluxerr
    average_flux = np.sum(flux*1/fluxerr**2)/np.sum(1/fluxerr**2)
    average_fluxerr = np.sqrt(np.sum(fluxerr**2))/len(fluxerr)

    # Transform back to magnitude and magnitude error
    final_mag = 27.5 - 2.5 * np.log10(average_flux)
    final_magerr = 1.086 * average_fluxerr / average_flux

    return(final_mag, final_magerr)

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
            colnum = int(coldata.split('.')[0].strip())-1+offset
            return(colnum)
        except:
            return(None)

  # Given a row of a dolphot file and a columns file, return the data key
  # described by 'key' (e.g., 'VEGAMAG', 'Total counts',
  # 'Magnitude uncertainty') for the input image name
  def get_dolphot_data(self, row, colfile, key, image):
    # First get the column number from the start of the column file
    colnum = self.get_dolphot_column(colfile, key, image)
    rdata = row.split()

    # Now grab that piece of data from the row and return
    if colnum:
        if colnum < 0 or colnum > len(rdata)-1:
            error = 'ERROR: tried to use bad column {n} in dolphot output'
            print(error.format(n=colnum))
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

        print(message)

        with open(out, 'w') as f:
            self.show_photometry(phot, f=f)
        with open(snana, 'w') as f:
            self.show_photometry(phot, f=f, snana=True, show=False)
        print('\n')

  def show_data(self, phottable, form, header, units, f=None, avg=False):

    if avg:
        print('\n# Average Photometry')
        if f: f.write('\n# Average Photometry \n')
    else:
        for key in phottable.meta.keys():
            out = '# {key} = {val}'
            if f: f.write(out.format(key=key, val=phottable.meta[key])+'\n')
            print(out.format(key=key, val=phottable.meta[key]))

    if header: print(header)
    if f and header: f.write(header+'\n')
    if units: print(units)
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

        print(line)
        if f: f.write(line+'\n')

    print('\n')
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
    zpt = 27.5

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
       print(error)
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
    # Returns the chip (i.e., 1 for UVIS1, 2 for UVIS2, 1 for WFPC2/PC, 2-4 for
    # WFPC2/WFC 2-4, etc., default=1 for ambiguous images with more than one
    # chip in the hdu)
    hdu = fits.open(image)
    chip = None
    for h in hdu:
        if 'CCDCHIP' in h.header.keys():
            if not chip: chip=h.header['CCDCHIP']
            else: chip=1
        elif 'DETECTOR' in h.header.keys():
            if not chip: chip=h.header['DETECTOR']
            else: chip=1

    if not chip: chip=1

    return(chip)

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

    # Make a table with all of the metadata for each image.
    exp = [fits.getval(image,'EXPTIME') for image in img]
    dat = [fits.getval(image,'DATE-OBS') + 'T' +
           fits.getval(image,'TIME-OBS') for image in img]
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
    form = '{file: <26} {inst: <18} {filt: <10} '
    form += '{exp: <12} {date: <10} {time: <10}'
    if show:
        header = form.format(file='FILE',inst='INSTRUMENT',filt='FILTER',
                             exp='EXPTIME',date='DATE-OBS',time='TIME-OBS')
        print('\n\n')
        print(header)

        for row in obstable:
            line = form.format(file=row['image'],
                    inst=row['instrument'].upper(),
                    filt=row['filter'].upper(),
                    exp='%7.4f' % row['exptime'],
                    date=Time(row['datetime']).datetime.strftime('%Y-%m-%d'),
                    time=Time(row['datetime']).datetime.strftime('%H:%M:%S'))
            print(line)

        print('\n\n')

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
        date_obj = Time(fits.getval(refimage, 'DATE-OBS'))
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

        obstable[i]['drizname'] = drizname

    if file:

        form = '{inst: <10} {filt: <10} {exp: <12} {date: <16}'
        header = form.format(inst='INSTRUMENT', filt='FILTER', exp='EXPTIME',
            date='DATE')
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

  # Check if num is a number
  def is_number(self, num):
    try:
        num = float(num)
    except ValueError:
        return(False)
    return(True)

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
        error = 'ERROR: dolphot output {dp} does not exist.  Use --rundolphot '
        error += 'or check your dolphot output for errors'
        print(error.format(dp=dolphot['base']))
        return(None)

    # Check for reference image and x,y coordinates to scrape data
    if (not reference or not coord):
        error = 'ERROR: Need a reference image and coordinate to '
        error += 'scrape data from the dolphot catalog. Exiting...'
        print(error)
        return(None)

    # Copy dolphot catalog into a file if it doesn't exist already
    if not os.path.exists(dolphot['original']):
        shutil.copyfile(dolphot['base'], dolphot['original'])

    if not self.options['args'].nocuts:

        message = 'Cutting bad sources from dolphot catalog.'
        print(message)

        # Cut bad sources
        f = open('tmp', 'w')
        numlines = sum(1 for line in open(base))

        with open(dolphot['base']) as dolphot_file:
            message = 'There are {n} sources in dolphot file {dp}. '
            message += 'Cutting bad sources...'
            print(message.format(n=numlines, dp=dolphot['base']))

            bar = progressbar.ProgressBar(maxval=numlines).start()
            typecol = self.get_dolphot_column(colfile, 'Object type', '')
            for i,line in enumerate(dolphot_file):
                bar.update(i)
                if (int(line.split()[typecol])==1): # Obj type
                    f.write(line)
            bar.finish()

        f.close()

        message = 'Done cutting bad sources'
        print(message)

        if filecmp.cmp(dolphot['base'], 'tmp'):
            message = 'No changes to dolphot file {dp}.'
            print(message.format(dp=dolphot['base']))
            os.remove('tmp')
        else:
            message = 'Updating dolphot file {dp}.'
            print(message.format(dp=dolphot['base']))
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
    if self.options['args'].scraperadius:
        # Calculate what the radius will be for reference image in pixels
        # we can do this pretty easily with a trick from dec
        angradius = self.options['args'].scraperadius/3600.
        if dec < 89:
            dec1 = coord.dec.degree + angradius
        else:
            dec1 = coord.dec.degree - angradius
        coord1 = SkyCoord(ra, dec1, unit='deg')
        x1,y1 = wcs.utils.skycoord_to_pixel(coord1, w, origin=1)
        radius = np.sqrt((x-x1)**2+(y-y1)**2)

    message = 'Looking for a source around x={x}, y={y} in {file} '
    message += 'with a radius of {rad}'
    print(message.format(x='%7.2f'%float(x), y='%7.2f'%float(y), file=reference,
        rad='%7.4f'%float(radius)))

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
            if (dist < radius):
                data.append({'sep':dist, 'data':line, 'sn': float(sn)})

    limit_data = []
    if get_limits:
        limit_radius = dolphot['limit_radius']
        limit_data = self.get_limit_data(dolphot, coord, w, x, y, colfile,
            limit_radius)

    message = 'Done looking for sources in dolphot file {dp}. '
    message += 'hst123 found {n} sources around: {ra} {dec}'
    print(message.format(dp=dolphot['base'], n=len(data), ra=ra, dec=dec))

    # What to do for n sources?
    if len(data)==0:
        return(None)
    else:
        warning = 'WARNING: found more than one source. '
        warning += 'Picking {0}'
        # If brightest, sort by 1/sn so highest sn is first
        if brightest:
            data = sorted(data, key=lambda obj: 1./obj['sn'])
            warning = warning.format('brightest object')
        # Otherwise sort by separation so closest object is first
        else:
            data = sorted(data, key=lambda obj: obj['sep'])
            warning = warning.format('closest to '+str(ra)+' '+str(dec))
        if not scrapeall:
            print(warning)
            data = [data[0]]
            m='Separation={0}, Signal-to-noise={1}'
            print(m.format(data[0]['sep'], data[0]['sn']))

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
                    print(warning)
                    continue
            path, base = os.path.split(file)
            # Should catch error where 'base' does not exist
            if os.path.isfile(base) and filecmp.cmp(file, base):
                message = '{file} == {base}'
                print(message.format(file=file, base=base))
                continue
            else:
                message = '{file} != {base}'
                print(message.format(file=file, base=base))
                shutil.copyfile(file, base)

  # Check the archivedir provided by --archive option.  Allows for organizing,
  # avoiding multiple copies of data files and enable speed ups
  def check_archive(self, product, archivedir=None):

    if not archivedir:
        archivedir = self.options['args'].archive
    if not os.path.exists(archivedir):
        try:
            os.makedirs(archivedir)
        except:
            error = 'ERROR: could not make archive dir {dir}\n'
            error += 'Enable write permissions to this location\n'
            error += 'Exiting...'
            print(error.format(dir=archivedir))
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
        except:
            error = 'ERROR: could not make archive dir {0}\n'
            error += 'Enable write permissions to this location\n'
            error += 'Exiting...'
            print(error.format(path))
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
        print(warning.format(archivedir))
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
        print(warning.format(fullfile))
        return(None)
    else:
        if check_for_coord:
            warning, check = self.needs_to_be_reduced(fullfile, save_c1m=True)
            if not check:
                print(warning)
                return(None)
        if workdir:
            fulloutfile = workdir + '/' + basefile
        else:
            fulloutfile = basefile

        # Check whether fulloutfile exists and if files are the same
        if os.path.exists(fulloutfile) and filecmp.cmp(fullfile, fulloutfile):
            message = '{file} == {base}'
            print(message.format(file=fullfile,base=fulloutfile))
            return(0)
        else:
            message = '{file} != {base}'
            print(message.format(file=fullfile,base=fulloutfile))
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
                    print(error)
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

  # Estimate the limiting magnitude given an input list of magnitudes and
  # errors.  Default is 3-sigma limit.
  def estimate_mag_limit(self, mags, errs, limit=3.0):

    warning = 'WARNING: cannot sample a wide enough range of magnitudes '
    warning += 'to estimate a limit'

    # First bin signal-to-noise in magnitude, then extrapolate to get 3-sigma
    try:
        mags = np.array(mags) ; errs = np.array(errs)
        bin_mag = np.linspace(np.min(mags), np.max(mags), 100)
        snr = np.zeros(100)
    except ValueError:
        print(warning)
        return(np.nan)

    for i in np.arange(100):
        if i==99:
            snr[i]=snr[i-1]
        else:
            idx = np.where((mags > bin_mag[i]) & (mags < bin_mag[i+1]))
            snr[i] = np.median(1./errs[idx])

    mask = np.array([np.isnan(val) for val in snr])
    bin_mag = bin_mag[~mask]
    snr = snr[~mask]

    if len(snr)>10:
        snr_func = interpolate.interp1d(snr, bin_mag, fill_value='extrapolate',
            bounds_error=False)
        return(snr_func(limit))
    else:
        return(np.nan)

  # Sanitizes reference header, gets rid of multiple extensions and only
  # preserves science data.
  def sanitize_reference(self, reference):

    # If the reference image does not exist, print an error and return
    if not os.path.exists(reference):
        error = 'ERROR: reference {ref} does not exist!'
        print(error.format(ref=reference))
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
        print(error.format(ref=reference))
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

  def parse_coord(self, ra, dec):
    if (not (hst.is_number(ra) and hst.is_number(dec)) and
        (':' not in ra and ':' not in dec)):
        error = 'ERROR: cannot interpret: {ra} {dec}'
        print(error.format(ra=ra, dec=dec))
        return(None)

    if (':' in ra and ':' in dec):
        # Input RA/DEC are sexagesimal
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        coord = SkyCoord(ra, dec, frame='icrs', unit=unit)
        return(coord)
    except ValueError:
        error = 'ERROR: Cannot parse coordinates: {ra} {dec}'
        print(error.format(ra=ra,dec=dec))
        return(None)

  def needs_to_be_reduced(self, image, save_c1m=False, keep_indt=False):
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
        print(warning.format(img=image))

        success = False
        if not self.productlist:
            warning = 'WARNING: could not find or download {img}'
            return(warning.format(img=image), False)

        mask = self.productlist['productFilename']==image
        if self.productlist[mask]==0:
            warning = 'WARNING: could not find or download {img}'
            return(warning.format(img=image), False)

        archivedir = self.options['args'].archive
        workdir = self.options['args'].workdir

        self.download_files(self.productlist,
            archivedir=self.options['args'].archive, clobber=True)

        for product in self.productlist[mask]:
            self.copy_raw_data_archive(product, archivedir=archivedir,
                workdir=workdir, check_for_coord=True)

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
    for key in ['INSTRUME','EXPFLAG','EXPTIME',
        'DATE-OBS','TIME-OBS']:
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
    if not keep_indt:
        if 'EXPFLAG' in hdu[0].header.keys():
            if hdu[0].header['EXPFLAG']!='NORMAL':
                warning = 'WARNING: {img} has EXPFLAG!=NORMAL.'
                return(warning.format(img=image), False)

    # Get rid of exposures with exptime < 20s
    if not self.options['args'].keepshort:
        exptime = hdu[0].header['EXPTIME']
        if (exptime < 15):
            warning = 'WARNING: {img} EXPTIME is {exp} < 20.'
            return(warning.format(img=image, exp=exptime), False)

    # Now check date and compare to self.before
    mjd_obs = Time(hdu[0].header['DATE-OBS']+'T'+hdu[0].header['TIME-OBS']).mjd
    if self.before is not None:
        mjd_before = Time(self.before).mjd
        dbefore = self.before.strftime('%Y-%m-%d')
        if mjd_obs > mjd_before:
            warning = 'WARNING: {img} is after the input before date {date}.'
            return(warning.format(img=image, date=dbefore), False)

    # Same with self.after
    if self.after is not None:
        mjd_after = Time(self.after).mjd
        dafter = self.after.strftime('%Y-%m-%d')
        if mjd_obs < mjd_after:
            warning = 'WARNING: {img} is before the input after date {date}.'
            return(warning.format(img=image, date=dafter), False)

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
            warning = 'WARNING: {img} does not contain: {ra} {dec}'
            ra = self.coord.ra.degree
            dec = self.coord.dec.degree
            return(warning.format(img=image, ra=ra, dec=dec), False)

    filt = self.get_filter(image).upper()
    if not (filt in self.options['acceptable_filters']):
        warning = 'WARNING: {img} with FILTER={filt} '
        warning += 'does not have an acceptable filter.'
        return(warning.format(img=image, filt=filt), False)

    # Get rid of images that don't match one of the allowed instrument/detector
    # types and images whose extensions don't match the allowed type for those
    # instrument/detector types
    is_not_hst_image = False
    warning = 'WARNING: {img} with INSTRUME={inst}, DETECTOR={det}, '
    warning += 'NEXTEND={N} is bad.'
    nextend = hdu[0].header['NEXTEND']
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

    return(warning.format(img=image, inst=instrument, det=detector,
        N=nextend), is_not_hst_image)

  def needs_to_split_groups(self,image):
    return(len(glob.glob(image.replace('.fits', '.chip?.fits'))) == 0)

  def needs_to_calc_sky(self, image, check_wcs=False):
    print('Checking for',image.replace('.fits','.sky.fits'))
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

  # Check if the image contains input coordinate.  This is somewhat complicated
  # as an image might cover input coordinates, but they land on a bad part of
  # the detector.  So 1) check if coordinate is in image, and 2) check if
  # corresponding DQ file lists this part of image as good pixels.
  def image_contains(self, image, coord):

    cmd = 'sky2xy {image} {ra} {dec}'
    cmd = cmd.format(image=image, ra=coord.ra.degree, dec=coord.dec.degree)
    result = os.popen(cmd).read()
    if 'off image' in result:
        return(False)
    else:
        return(True)

  # Determine if we need to run the dolphot mask routine
  def needs_to_be_masked(self,image):
    # Masking should remove all of the DQ arrays etc, so make sure that any
    # extensions with data in in them are only SCI extensions. This might not be
    # 100% robust, but should be good enough.
    hdulist = fits.open(image)
    header = hdulist[0].header
    inst = self.get_instrument(image).split('_')[0].upper()
    if inst is 'WFPC2':
        if 'DOLWFPC2' in header.keys():
            if header['DOLWFPC2']==0:
                return(False)
    if inst is 'WFC3':
        if 'DOL_WFC3' in header.keys():
            if header['DOL_WFC3']==0:
                return(False)
    if inst is 'ACS':
        if 'DOL_ACS' in header.keys():
            if header['DOL_ACS']==0:
                return(False)
    return(True)

  # Get a string representing the filter for the input image
  def get_filter(self, image):
    if 'wfpc2' in str(fits.getval(image, 'INSTRUME')).lower():
        f = str(fits.getval(image, 'FILTNAM1'))
        if len(f.strip()) == 0:
            f = str(fits.getval(image, 'FILTNAM2'))
    else:
        try:
            f = str(fits.getval(image, 'FILTER'))
        except:
            f = str(fits.getval(image, 'FILTER1'))
            if 'clear' in f.lower():
                f = str(fits.getval(image, 'FILTER2'))
    return(f.lower())

  # Get string representing instrument, detectory, and subarray for the input
  # image
  def get_instrument(self, image):
    hdu = fits.open(image, mode='readonly')
    inst = hdu[0].header['INSTRUME'].lower()
    if inst.upper() == 'WFPC2':
        det = 'wfpc2'
        sub = 'full'
    else:
        det = hdu[0].header['DETECTOR'].lower()
        if hdu[0].header['SUBARRAY'] == 'T':
            sub = 'sub'
        else:
            sub = 'full'
    out = '{inst}_{det}_{sub}'
    return(out.format(inst=inst, det=det, sub=sub))

  # Glob all of the input images from working directory
  def get_input_images(self, pattern=None):
    if pattern == None:
        pattern = ['*c1m.fits','*c0m.fits','*flc.fits','*flt.fits']
    return([s for p in pattern for s in glob.glob(p)])

  def get_split_images(self, pattern=None):
    if pattern == None:
        pattern = ['*c0m.chip?.fits', '*flc.chip?.fits', '*flt.chip?.fits']
    return([s for p in pattern for s in glob.glob(p)])

  # Get the data quality image for dolphot mask routine.  This is entirely for
  # WFPC2 since mask routine requires additional input for this instrument
  def get_dq_image(self,image):
    if self.get_instrument(image).split('_')[0].upper() == 'WFPC2':
        return(image.replace('c0m.fits','c1m.fits'))
    else:
        return('')

  # Run the dolphot splitgroups routine
  def split_groups(self, image, delete_non_science=True):
    print('Running split groups for {image}'.format(image=image))
    splitgroups = 'splitgroups {filename}'.format(filename=image)

    print('\n\nExecuting: {0}\n\n'.format(splitgroups))
    os.system(splitgroups)

    # Delete images that aren't from science extensions
    if delete_non_science:
        split_images = glob.glob(image.replace('.fits','.chip*.fits'))

        for split in split_images:
            hdu = fits.open(split)
            info = hdu[0]._summary()

            if info[0].upper()!='SCI':
                warning = 'WARNING: deleting {im}, not a science extension.'
                print(warning.format(im=split))
                os.remove(split)

  # Run the dolphot mask routine for the input image
  def mask_image(self, image, instrument):
    maskimage = self.get_dq_image(image)
    cmd = '{instrument}mask {image} {maskimage}'
    mask = cmd.format(instrument=instrument, image=image, maskimage=maskimage)

    print('\n\nExecuting: {0}\n\n'.format(mask))
    os.system(mask)

  # Run the dolphot calcsky routine
  def calc_sky(self, image, options):
    det = '_'.join(self.get_instrument(image).split('_')[:2])
    opt = options[det]['dolphot_sky']
    cmd = 'calcsky {image} {rin} {rout} {step} {sigma_low} {sigma_high}'
    calc_sky = cmd.format(image=image.replace('.fits',''), rin=opt['r_in'],
                            rout=opt['r_out'], step=opt['step'],
                            sigma_low=opt['sigma_low'],
                            sigma_high=opt['sigma_high'])

    print('\n\nExecuting: {0}\n\n'.format(calc_sky))
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
  def add_image_to_param_file(self,param_file, image, i, options):
    # Add image name to param file
    image_name = 'img{i}_file = {file}\n'
    param_file.write(image_name.format(i=str(i).zfill(4),
        file=os.path.splitext(image)[0]))

    # Now add all image-specific params to param file
    params = self.get_dolphot_instrument_parameters(image, options)
    for par, val in params.items():
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
        reffilter=self.options['args'].reffilter, avoid_wfpc2=True,
        refinst=self.options['args'].refinst)

    if len(reference_images)==0:
        error = 'ERROR: could not pick a reference image'
        print(error)
        return(None)

    best_filt = self.get_filter(reference_images[0])
    best_inst = self.get_instrument(reference_images[0]).split('_')[0]

    vnum = np.min(obstable['visit'].data)
    vnum = str(vnum).zfill(3)

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
                print(warning.format(drz=drizname))
                return(drizname)

    message = 'Reference image name will be: {reference}.\n'
    message += 'Generating from input files: {img}\n\n'
    print(message.format(reference=drizname, img=reference_images))

    if self.options['args'].drizadd:
        add_images = list(str(self.options['args'].drizadd).split(','))
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

        self.run_tweakreg(obstable[mask], '')
        self.run_astrodrizzle(obstable[mask], output_name=outimage,
            clean=False)

        # Add cosmic ray mask to static image mask
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

    self.run_tweakreg(obstable, '')
    self.run_astrodrizzle(obstable, output_name=drizname)

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

  def fix_hdu_wcs_keys(self, image, change_keys, ref_url):

    hdu = fits.open(image, mode='update')
    ref = ref_url.strip('.old')

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
            if not os.path.exists(ref_file):
                # Try using both old cdbs database and new crds link
                urls = []
                url = self.options['global_defaults']['cdbs']
                urls.append(url+ref_url+'/'+ref_file)

                url = self.options['global_defaults']['crds']
                urls.append(url+ref_file)
                for url in urls:
                    message = 'Downloading file: {url}'
                    sys.stdout.write(message.format(url=url))
                    sys.stdout.flush()
                    try:
                        dat = download_file(url, cache=False,
                            show_progress=False, timeout=120)
                        shutil.move(dat, ref_file)
                        message = '\r' + message
                        message += green+' [SUCCESS]'+end+'\n'
                        sys.stdout.write(message.format(url=url))
                        break
                    except:
                        message = '\r' + message
                        message += red+' [FAILURE]'+end+'\n'
                        sys.stdout.write(message.format(url=url))
                        print(message.format(url=url))

            message = 'Setting {im},{i} {key}={val}'
            print(message.format(im=image, i=i, key=key, val=ref_file))
            hdu[i].header[key] = ref_file

    hdu.writeto(image, overwrite=True, output_verify='silentfix')
    hdu.close()

  # Update image wcs using updatewcs routine
  def update_image_wcs(self, image, options, use_db=True):

    message = 'Updating WCS for {file}'
    print(message.format(file=image))

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
        print(message)
        hdu.info()
        hdu.close()
        self.fix_hdu_wcs_keys(image, change_keys, ref_url)
        self.fix_idcscale(image)
        return(True)
    except:
        error = 'ERROR: failed to update WCS for image {file}'
        print(error.format(file=image))
        return(None)

  # Run the drizzlepac astrodrizzle routine using detector parameters.
  def run_astrodrizzle(self, obstable, output_name = None, ra=None, dec=None,
    clean=None):

    print('Starting astrodrizzle')

    n = len(obstable)

    if output_name is None:
        output_name = 'drizzled.fits'

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
    if len(inst) > 1 and not self.options['args'].drizadd:
        error = 'ERROR: Cannot drizzle together images from detectors: {det}.'
        error += 'Exiting...'
        print(error.format(det=','.join(map(str,inst))))
        return(False)

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

    if not ra or not dec:
        ra = self.coord.ra.degree if self.coord else None
        dec = self.coord.dec.degree if self.coord else None

    if self.options['args'].keepshort and not self.options['args'].sky_sub:
        skysub = False
    else:
        skysub = True

    if self.options['args'].drizscale:
        pixscale = self.options['args'].drizscale
    else:
        pixscale = options['pixel_scale']

    wht_type = self.options['args'].wht_type

    if clean is not None:
        clean = not self.options['args'].nocleanup

    if len(tmp_input)==1:
        shutil.copy(tmp_input[0], 'dummy.fits')
        tmp_input.append('dummy.fits')

    print('Need to run astrodrizzle for images:')
    self.input_list(obstable['image'], show=True, save=False)

    # If drizmask, then edit tmp_input masks for everything except for drizadd
    # files
    if self.options['args'].drizmask and self.options['args'].drizadd:
        add_im_base = [im.split('.')[0]
            for im in self.options['args'].drizadd.split(',')]

        if ',' in self.options['args'].drizmask:
            ramask, decmask = self.options['args'].drizmask.split(',')
        else:
            ramask, decmask = self.options['args'].drizmask.split()
        maskcoord = self.parse_coord(ramask, decmask)

        for image in tmp_input:
            imhdu = fits.open(image)
            added = any([base in image for base in add_im_base])

            for i,h in enumerate(imhdu):
                if not h.name=='DQ':
                    continue

                w = wcs.WCS(h.header)
                y,x = wcs.utils.skycoord_to_pixel(maskcoord, w, origin=1)

                size=200
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
                    print('Making outside drizmask:',image)
                    if outside_im: imhdu[i].data[:,:]=128
                    else:
                        data = copy.copy(imhdu[i].data[xmin:xmax,ymin:ymax])
                        imhdu[i].data[:,:]=128
                        imhdu[i].data[xmin:xmax,ymin:ymax]=data
                else:
                    print('Making inside drizmask:',image)
                    imhdu[i].data[xmin:xmax,ymin:ymax]=128

            imhdu.writeto(image, overwrite=True, output_verify='silentfix')

    # Check for TWEAK key in hdu.  If WCSNAME in header but not TWEAK
    # then rename WCSNAME to TWEAK
    for image in tmp_input:
        imhdu = fits.open(image)
        for i,h in enumerate(imhdu):
            head = h.header
            tweak = False ; wcsname = False
            print('Checking for tweak keys in header...')
            for key in head.keys():
                if 'WCSNAME' in key: wcsname = True
                if 'WCSNAME' in key and head[key].strip()=='TWEAK': tweak = True

            if wcsname and not tweak:
                # Rename 'WCSNAME' to 'TWEAK' in rawhdu
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
    hdu = fits.open(tmp_input[0])
    photflam = None
    for h in hdu:
        if h.name=='SCI' and 'PHOTFLAM' in h.header.keys():
            photflam = h.header['PHOTFLAM']

    for image in tmp_input:
        if 'c0m' in image:
            photeq.photeq(files=image, readonly=False, ref_phot=photflam,
                phot_kwd='PHOTFLAM')

    rotation = 0.0
    if self.options['args'].no_rotation:
        rotation = None

    tries = 0
    while tries < 3:
        try:
            astrodrizzle.AstroDrizzle(tmp_input, output=output_name, runfile='',
                wcskey=wcskey, context=True, group='', build=False,
                num_cores=8, preserve=False, clean=clean, skysub=skysub,
                skymethod='globalmin+match', skymask_cat=skymask_cat,
                skystat='mode', skylower=0.0, skyupper=None, updatewcs=False,
                driz_sep_fillval=None, driz_sep_bits=options['driz_bits'],
                driz_sep_wcs=True, driz_sep_rot=rotation, driz_sep_scale=None,
                driz_sep_outnx=options['nx'], driz_sep_outny=options['ny'],
                driz_sep_ra=ra, driz_sep_dec=dec, driz_sep_pixfrac=0.8,
                combine_maskpt=0.2, combine_type=combine_type,
                combine_nlow=0, combine_nhigh=combine_nhigh,
                combine_lthresh=-10000, combine_hthresh=None,
                combine_nsigma='4 3', driz_cr_corr=True,
                driz_cr=True, driz_cr_snr='3.5 3.0', driz_cr_grow=1,
                driz_cr_ctegrow=0, driz_cr_scale='1.2 0.7',
                final_pixfrac=0.8, final_fillval=None,
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
    print(message.format(time = time.time()-start_drizzle))

    if not self.options['args'].nocleanup:
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
    if os.path.exists(mask_file):
        os.rename(mask_file, output_name.replace('.fits', '.mask.fits'))
    if os.path.exists(science_file):
        os.rename(science_file, output_name)

    # Get comma-separated list of base input files
    ninput = len(tmp_input)
    tmp_input = sorted(tmp_input)
    str_input = ','.join([s.split('.')[0] for s in tmp_input])

    # Add header keys on drizzled file
    hdu = fits.open(output_name, mode='update')
    filt = obstable['filter'][0]
    hdu[0].header['FILTER'] = filt.upper()
    hdu[0].header['TELID'] = 'HST'
    hdu[0].header['OBSTYPE'] = 'OBJECT'
    # These keys are useful for auditing drz image later
    hdu[0].header['NINPUT'] = ninput
    hdu[0].header['INPUT'] = str_input
    # Add object name if it was input from command line
    if self.options['args'].object:
        hdu[0].header['TARGNAME'] = self.options['args'].object
        hdu[0].header['OBJECT'] = self.options['args'].object

    if self.options['args'].fixzpt:
        # Get current zeropoint of drizzled image
        fixzpt = self.options['args'].fixzpt
        zpt = self.get_zpt(output_name)
        exptime = hdu[0].header['EXPTIME']
        effzpt = zpt + 2.5*np.log10(exptime)
        fixscale = 10**(0.4 * (fixzpt - effzpt))
        fluxscale = 3631e-3 * 10**(-0.4 * fixzpt) # mJy/pix scale

        # Adjust header values for context
        inst = self.get_instrument(output_name).split('_')[0]
        crpars = self.options['instrument_defaults'][inst]['crpars']
        det = '_'.join(self.get_instrument(output_name).split('_')[:2])
        instopt = self.options['detector_defaults'][det]

        hdu[0].header['FIXZPT']   = fixzpt
        hdu[0].header['FIXFLUX']  = fluxscale
        hdu[0].header['EFFZPT']   = effzpt
        hdu[0].header['ORIGZPT']  = zpt
        hdu[0].header['FIXSCALE'] = fixscale
        # rescaled by EXPTIME, so essentially cps
        hdu[0].header['BUNIT']    = 'cps'
        hdu[0].header['SCALSAT']  = crpars['saturate'] * fixscale

        # Finally do data scaling
        data = hdu[0].data * fixscale
        hdu[0].data = data

    hdu.close()

    return(True)

  # Run cosmic ray clean
  def run_cosmic(self, image, options, output=None):
    message = 'Cleaning cosmic rays in image: {image}'
    print(message.format(image=image))
    hdulist = fits.open(image,mode='readonly')

    if output is None:
        output = image

    for i,hdu in enumerate(hdulist):
        if hdu.name=='SCI':
            mask = np.zeros(hdu.data.shape, dtype=np.bool)

            crmask, crclean = detect_cosmics(hdu.data.copy().astype('<f4'),
                inmask=mask, readnoise=options['rdnoise'], gain=options['gain'],
                satlevel=options['saturate'], sigclip=options['sig_clip'],
                sigfrac=options['sig_frac'], objlim=options['obj_lim'])

            hdulist[i].data[:,:] = crclean[:,:]

            # Add crmask data to DQ array or DQ image
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

  # Prepare reference image for tweakreg.  Requires a specific organization of
  # data and header keys for tweakreg to parse
  def prepare_reference_tweakreg(self, reference):
    if not os.path.exists(reference):
        error = 'ERROR: tried to sanitize non-existence ref {ref}'
        print(error.format(ref=reference))
        return(False)

    hdu = fits.open(reference)

    # Summary is organized: EXTNAME, EXTVER, TYPE, CARDS, DIMENSIONS, FORMAT
    data = [h._summary() for h in hdu]

    if len(data)==1:

        newhdu = fits.HDUList()
        newhdu.append(hdu[0])
        newhdu.append(hdu[0])
        newhdu[0].data = None

        # Update EXTVER
        newhdu[0].header['EXTVER']=1
        newhdu[1].header['EXTVER']=1

        # Set the EXTNAMEs
        newhdu[0].header['EXTNAME']='PRIMARY'
        newhdu[1].header['EXTNAME']='SCI'

        # Write out to same file and return True
        newhdu.writeto(reference, output_verify='silentfix', overwrite=True)

        return(True)

    else:
        # Get the smallest index extension that contains a data/image array
        idxIm = [i for i,d in enumerate(data) if (d[2].strip()=='ImageHDU')]

        # Get index of the primary extension
        idxPr = [i for i,d in enumerate(data)
            if (d[0].strip().upper()=='PRIMARY')]

        if len(idxIm)>0:
            newhdu = fits.HDUList()
            newhdu.append(hdu[np.min(idxIm)])
            newhdu.append(hdu[np.min(idxIm)])
            newhdu[0].data = None

            # If there is a primary extension, overwrite the header vars
            if len(idxPr)>0:
                primary = hdu[np.min(idxPr)]
                hkeys = list(newhdu[0].header.keys())
                for n,key in enumerate(primary.header.keys()):
                    if not key.strip():
                        continue
                    if isinstance(hdu[0].header[key], str):
                        if '\n' in hdu[0].header[key]:
                            continue
                    if key=='FILETYPE':
                        newhdu[0].header[key]='SCI'
                    elif key=='FILENAME':
                        newhdu[0].header[key]=reference
                    elif key=='EXTEND':
                        newhdu[0].header[key]=True
                    else:
                        val = hdu[0].header[key]
                        if isinstance(val, str):
                            val = val.strip().replace('\n',' ')
                        newhdu[0].header[key] = val

            # Update image name
            newhdu[0].header['FILENAME']=reference
            newhdu[1].header['FILENAME']=reference

            # Update EXTVER
            newhdu[0].header['EXTVER']=1
            newhdu[1].header['EXTVER']=1

            # Set the EXTNAMEs
            newhdu[0].header['EXTNAME']='PRIMARY'
            newhdu[1].header['EXTNAME']='SCI'

            # We'll need to resanitize the images
            if 'SANITIZE' in newhdu[0].header.keys():
                del newhdu[0].header['SANITIZE']
            if 'SANITIZE' in newhdu[1].header.keys():
                del newhdu[1].header['SANITIZE']

            # We'll need to regenerate the sky image
            if os.path.exists(reference.replace('.fits','.sky.fits')):
                os.remove(reference.replace('.fits','.sky.fits'))

            # Write out to same file and return True
            newhdu.writeto(reference, output_verify='silentfix', overwrite=True)

            return(True)

        else:
            # If there is no image index, we can't create a good reference image
            return(False)

  # Check each image in the list to see if tweakreg has been run
  def check_images_for_tweakreg(self, run_images):

    if not run_images:
        return(None)

    images = copy.copy(run_images)

    for file in list(images):
        print('Checking {0} for WCSNAME=TWEAK'.format(file))
        hdu = fits.open(file, mode='readonly')
        remove_image = False
        for i,h in enumerate(hdu):
            if h.name == 'SCI':
                header = h.header
                if 'WCSNAME' in header.keys():
                    if header['WCSNAME'].strip()=='TWEAK':
                        remove_image = True

        if remove_image:
            images.remove(file)

    # If run_images is now empty, return None instead
    if len(images)==0:
        return(None)

    return(images)

  # Returns the number of sources detected in an image at the thresh value
  def get_nsources(self, image, thresh):
    imghdu = fits.open(image)
    nsources = 0
    message = '\n\nGetting number of sources in {im} at threshold={thresh}'
    print(message.format(im=image, thresh=thresh))
    for i,h in enumerate(imghdu):
        if h.name=='SCI':
            filename="{:s}[{:d}]".format(image, i)
            wcs = stwcs.wcsutil.HSTWCS(filename)
            catalog_mode = 'automatic'
            catalog = catalogs.generateCatalog(wcs, mode=catalog_mode,
                catalog=filename, threshold=thresh,
                **self.options['catalog'])
            try:
                catalog.buildCatalogs()
                nsources += catalog.num_objects
            except:
                pass

    message = 'Got {n} total sources'
    print(message.format(n=nsources))

    return(nsources)

  def count_nsources(self, image):
    cat_str = '_sci*_xy_catalog.coo'
    # Tag cat files with the threshold so we can reference it later
    n = 0
    for image in images:
        for catalog in glob.glob(image.replace('.fits',cat_str)):
            with open(catalog, 'r+') as f:
                for line in f:
                    if 'threshold' not in line:
                        n += 1

    return(n)

  # Given an input image, look for a matching catalog and estimate what the
  # threshold should be for this image.  If no catalog exists, generate one
  # on the fly and estimate threshold
  def get_tweakreg_thresholds(self, image, thresh, runcat=True, inp_data=[]):

    # Analyze the input images
    cat_str = '_sci*_xy_catalog.coo'

    # Look at all science images
    data={'name':image, 'nobjects':0, 'threshold': []}
    catfiles = glob.glob(image.replace('.fits',cat_str))
    if len(catfiles)>0:
        for catfile in catfiles:
            with open(catfile) as f:
                line = f.readline()
                if (line and 'threshold=' in line):
                    threshold = float(line.split('=')[1])
                    data['threshold'].append(threshold)
            table = Table.read(catfile, format='ascii')
            data['nobjects']=data['nobjects']+len(table)

        best_thresh = list(set(data['threshold']))
        if len(best_thresh)>0:
            if len(best_thresh)>1:
                warning = 'WARNING: thresholds for {im} catalogs do not match\n'
                warning += 'Picking largest value'
                print(warning.format(im=image))
            thresh=np.max(best_thresh)

    else:
        if runcat:
            try:
                data['nobjects'] = self.get_nsources(image, thresh)
            except:
                return(None)
        else:
            return(None)

    def thresh_func(x, p):
        threshold = (x/p[0])**p[1]
        return(threshold)

    # Scale threshold to the image with the largest number of objects
    if len(inp_data)<2:
        threshold = thresh * thresh_func(data['nobjects'],(8000,0.4))
    else:
        try:
            popt, pcov = curve_fit(thresh_func, [d[0] for d in inp_data],
                [d[1] for d in inp_data], p0=(8000, 0.4))
            threshold = thresh_func(8000, *popt)
            return(threshold)
        except:
            threshold = thresh * thresh_func(data['nobjects'],(8000,0.6))

    # Set minimum and maximum threshold
    if threshold<3.0: threshold=3.0
    if threshold>5000.0: threshold=1000.0

    return(threshold)

  # Check if images are too shallow for running with deep images in tweakreg
  def get_shallow_param(self, image):
    # Get filters and the pivot wavelength
    filt = self.get_filter(image)
    hdu = fits.open(image)

    pivot = 0.0
    for h in hdu:
        if 'PHOTPLAM' in h.header.keys():
            pivot = float(h.header['PHOTPLAM'])
            break

    exptime = 0.0
    for h in hdu:
        if 'EXPTIME' in h.header.keys():
            exptime = float(h.header['EXPTIME'])
            break

    return(filt, pivot, exptime)

  # Error message for tweakreg
  def tweakreg_error(self, exception):
    message = '\n\n' + '#'*80 + '\n'
    message += 'WARNING: tweakreg failed: {e}\n'
    message += '#'*80 + '\n'
    print(message.format(e=exception.__class__.__name__))
    print('Error:', exception)
    print('Adjusting thresholds and images...')

  def add_thresholds(self, images, thresh):
    cat_str = '_sci*_xy_catalog.coo'
    # Tag cat files with the threshold so we can reference it later
    for image in images:
        for catalog in glob.glob(image.replace('.fits',cat_str)):
            with open(catalog, 'r+') as f:
                line = f.readline()
                if 'threshold' not in line:
                    f.seek(0,0) ; content = f.read() ; f.seek(0, 0)
                    newline='#threshold={t}'.format(t=thresh)
                    f.write(newline + '\n' + content)

  # Run tweakreg on all input images
  def run_tweakreg(self, obstable, reference):

    # Get options from object
    options = self.options['global_defaults']
    # Check if tweakreg has already been run on each image
    run_images = self.check_images_for_tweakreg(list(obstable['image']))

    # Check if we just removed all of the images
    if not run_images:
        warning = 'WARNING: All images have been run through tweakreg.'
        print(warning)
        return(True)

    print('Need to run tweakreg for images:')
    self.input_list(obstable['image'], show=True, save=False)

    tmp_images = []
    for image in run_images:
        if self.updatewcs:
            det = '_'.join(self.get_instrument(image).split('_')[:2])
            wcsoptions = self.options['detector_defaults'][det]
            self.update_image_wcs(image, wcsoptions)

        # wfc3_ir doesn't need cosmic clean and assume reference is cleaned
        if (image == reference or 'wfc3_ir' in self.get_instrument(image)):
            message = 'Skipping adjustments for {file} as WFC3/IR or reference'
            print(message.format(file=image))
            tmp_images.append(image)
            continue

        rawtmp = image.replace('.fits','rawtmp.fits')
        tmp_images.append(rawtmp)

        # Check if rawtmp already exists
        if os.path.exists(rawtmp):
            message = '{file} exists. Skipping...'
            print(message.format(file=rawtmp))
            continue

        # Copy the raw data into a temporary file
        shutil.copyfile(image, rawtmp)

        # Clean cosmic rays so they aren't used for alignment
        inst = self.get_instrument(image).split('_')[0]
        crpars = self.options['instrument_defaults'][inst]['crpars']
        self.run_cosmic(rawtmp, crpars)

    modified = False
    ref_images = self.pick_deepest_images(list(obstable['image']))
    deepest = sorted(ref_images, key=lambda im: fits.getval(im, 'EXPTIME'))[-1]
    if (not reference or reference=='dummy.fits'):
        reference = 'dummy.fits'
        shutil.copyfile(deepest, reference)
    elif not self.prepare_reference_tweakreg(reference):
        # Can't use this reference image, just use one of the input
        reference = 'dummy.fits'
        shutil.copyfile(deepest, reference)
    else:
        modified = True

    message = 'Tweakreg is executing...'
    print(message)

    start_tweak = time.time()

    tweakreg_success = False
    tweak_img = copy.copy(tmp_images)
    ithresh = self.threshold ; rthresh = self.threshold
    shallow_img = []
    tries = 0

    while (not tweakreg_success and tries < 7):
        tweak_img = self.check_images_for_tweakreg(tweak_img)
        if not tweak_img: break
        if tweak_img:
            # Remove images from tweak_img if they are too shallow
            if shallow_img:
                for img in shallow_img:
                    if img in tweak_img:
                        tweak_img.remove(img)

            if len(tweak_img)==0:
                error = 'ERROR: removed all images as shallow'
                print(error)
                tweak_img = copy.copy(tmp_images)
                tweak_img = self.check_images_for_tweakreg(tweak_img)

            # If we've tried multiple runs and there are images in input
            # list with TWEAK and reference image=dummy.fits, we might need
            # to try a different reference image
            success = list(set(tmp_images) ^ set(tweak_img))
            if tries > 1 and reference=='dummy.fits' and len(success)>0:
                # Make random success image new dummy image
                n = len(success)-1
                shutil.copyfile(success[random.randint(0,n)],'dummy.fits')

            # This estimates what the input threshold should be and cuts
            # out images based on number of detected sources from previous
            # rounds of tweakreg
            message = '\n\nReference image: {ref} \n'
            message += 'Images: {im}'
            print(message.format(ref=reference, im=','.join(tweak_img)))

            thresholds = np.array([self.get_tweakreg_thresholds(im, ithresh,
                runcat=False) for im in tweak_img])
            thresholds = [t for t in thresholds if t is not None]

            if thresholds:
                ithresh = np.max(thresholds)
            else:
                deepest = self.pick_deepest_images(tweak_img)
                deepimg = sorted(deepest,
                    key=lambda im: fits.getval(im, 'EXPTIME'))[-1]
                ithresh = self.get_tweakreg_thresholds(deepimg, ithresh)

            rthresh = self.get_tweakreg_thresholds(reference, rthresh)

            # Other input options
            nbright = options['nbright']
            minobj = options['minobj']
            search_rad = int(np.round(options['search_rad']))

            message = '\nAdjusting thresholds:\n'
            message += 'Reference threshold={rthresh}\n'
            message += 'Image threshold={ithresh}\n'
            print(message.format(ithresh=ithresh, rthresh=rthresh))

            rconv = 3.5 ; iconv = 3.5 ; tol = 0.5
            if 'wfc3_ir' in self.get_instrument(reference):
                rconv = 2.5
            if all(['wfc3_ir' in self.get_instrument(i)
                for i in tweak_img]): iconv = 2.5 ; tol = 1.2

            try:
                tweakreg.TweakReg(files=tweak_img, refimage=reference,
                    verbose=False, interactive=False, clean=True,
                    writecat=True, updatehdr=True, reusename=True,
                    rfluxunits='counts', minobj=minobj, wcsname='TWEAK',
                    searchrad=search_rad, searchunits='arcseconds', runfile='',
                    tolerance=tol, refnbright=nbright, nbright=nbright,
                    separation=0.5, residplot='No plot', see2dplot=False,
                    imagefindcfg = {'threshold': ithresh,
                        'conv_width': iconv, 'use_sharp_round': True},
                    refimagefindcfg = {'threshold': rthresh,
                        'conv_width': rconv, 'use_sharp_round': True})

                # Reset shallow_img list
                shallow_img = []

            except AssertionError as e:
                self.tweakreg_error(e)

                message = 'Re-running tweakreg with shallow images removed:'
                print(message)
                for img in tweak_img:
                    nsources = self.get_nsources(img, ithresh)
                    if nsources < 1000:
                        shallow_img.append(img)

            except (MemoryError,TypeError,UnboundLocalError,RuntimeError) as e:
                self.tweakreg_error(e)

            self.add_thresholds(tweak_img, ithresh)
            self.add_thresholds([reference], rthresh)

            # Check that everything made it through tweakreg
            if not self.check_images_for_tweakreg(tmp_images):
                tweakreg_success = True

            tries += 1

    message = 'Tweakreg took {time} seconds to execute.\n\n'
    print(message.format(time = time.time()-start_tweak))

    for image in run_images:
        # Copy image over now to perform other image header updates
        if (image == reference or 'wfc3_ir' in self.get_instrument(image)):
            continue

        message = '\n\nUpdating image data for image: {im}'
        print(message.format(im=image))
        rawtmp = image.replace('.fits','rawtmp.fits')

        rawhdu = fits.open(rawtmp, mode='readonly')
        hdu    = fits.open(image, mode='readonly')
        newhdu = fits.HDUList()

        print('Current image info:')
        hdu.info()

        for i, h in enumerate(hdu):
            if h.name=='SCI':
                if 'flc' in image or 'flt' in image:
                    if len(rawhdu)>=i+2 and rawhdu[i+2].name=='DQ':
                        self.copy_wcs_keys(rawhdu[i], rawhdu[i+2])
                elif 'c0m' in image:
                    maskfile = image.split('_')[0]+'_c1m.fits'
                    if os.path.exists(maskfile):
                        maskhdu = fits.open(maskfile)
                        self.copy_wcs_keys(rawhdu[i], maskhdu[i])
                        maskhdu.writeto(maskfile, overwrite=True)

            # Skip WCSCORR for WFPC2 as non-standard hdu
            if 'wfpc2' in self.get_instrument(image).lower():
                if h.name=='WCSCORR':
                    continue

            # Get the index of the corresponding extension in rawhdu.  This
            # can be different from "i" if extensions were added or rearranged
            ver = int(h.ver) ; name = str(h.name).strip()
            idx = -1

            for j,rawh in enumerate(rawhdu):
                if str(rawh.name).strip()==name and int(rawh.ver)==ver:
                    idx = j

            # If there is no corresponding extension, then continue
            if idx < 0:
                message = 'Skip extension {i},{ext},{ver} '
                message += '- no match in {f}'
                print(message.format(i=i, ext=name, ver=ver, f=rawtmp))
                continue

            # If we can access the data in both extensions, copy from orig file
            if h.name!='DQ':
                if 'data' in dir(h) and 'data' in dir(rawhdu[idx]):
                    if (rawhdu[idx].data is not None and h.data is not None):
                        if rawhdu[idx].data.dtype==h.data.dtype:
                            rawhdu[idx].data = h.data

            # Copy the rawtmp extension into the new file
            message = 'Copy extension {i},{ext},{ver}'
            print(message.format(i=idx, ext=name, ver=ver))
            newhdu.append(copy.copy(rawhdu[idx]))

        if 'wfpc2' in self.get_instrument(image).lower():
            # Adjust number of extensions to 4
            newhdu[0].header['NEXTEND']=4

        print('\n\nNew image info:')
        newhdu.info()

        newhdu.writeto(image, output_verify='silentfix', overwrite=True)

        if (os.path.isfile(rawtmp) and not self.options['args'].nocleanup):
            os.remove(rawtmp)

    # Clean up temporary files and output
    if os.path.isfile('dummy.fits'):
        os.remove('dummy.fits')

    if not self.options['args'].keep_objfile:
        for file in glob.glob('*.coo'):
            os.remove(file)

    if modified:
        # Re-sanitize reference using main sanitize function
        self.sanitize_reference(reference)

    return(tweakreg_success)

  def copy_wcs_keys(self, from_hdu, to_hdu):
    for key in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1',
        'CD2_2']:
        if key in from_hdu.header.keys():
            to_hdu.header[key]=from_hdu.header[key]

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
            Observations.login(token=self.options['args'].token)
    except:
        warning = 'WARNING: could not log in with input username/password'
        print(warning)

    try:
        obsTable = Observations.query_region(coord, radius=search_radius)
    except (astroquery.exceptions.RemoteServiceError,
        requests.exceptions.ConnectionError,
        astroquery.exceptions.TimeoutError,
        requests.exceptions.ChunkedEncodingError):
        error = 'ERROR: MAST is not working currently working\n'
        error += 'Try again later...'
        print(error)
        return(productlist)

    # Get rid of all masked rows (they aren't HST data anyway)
    obsTable = obsTable.filled()

    # Construct masks for telescope, data type, detector, and data rights
    masks = []
    masks.append([t.upper()=='HST' for t in obsTable['obs_collection']])
    masks.append([p.upper()=='IMAGE' for p in obsTable['dataproduct_type']])
    masks.append([any(l) for l in list(map(list,zip(*[[det in inst.upper()
                for inst in obsTable['instrument_name']]
                for det in ['ACS','WFC','WFPC2']])))])

    # Time constraint masks (before and after MJD)
    if self.before:
        masks.append([t < Time(self.before).mjd for t in obsTable['t_min']])
    if self.after:
        masks.append([t > Time(self.after).mjd for t in obsTable['t_min']])

    # Get rid of short exposures (defined as 15s or less)
    if not self.options['args'].keepshort:
        masks.append([t > 15. for t in obsTable['t_exptime']])

    # Apply the masks to the observation table
    mask = [all(l) for l in list(map(list, zip(*masks)))]
    obsTable = obsTable[mask]

    # Iterate through each observation and download the correct product
    # depending on the filename and instrument/detector of the observation
    for obs in obsTable:
        try:
            productList = Observations.get_product_list(obs)
            # Ignore the 'C' type products
            mask = productList['type']=='S'
            productList = productList[mask]
        except:
            error = 'ERROR: MAST is not working currently working\n'
            error += 'Try again later...'
            print(error)
            return(productlist)
        instrument = obs['instrument_name']
        s_ra = obs['s_ra']
        s_dec = obs['s_dec']
        productList.add_column([instrument]*len(productList),
            name='instrument_name')
        productList.add_column([s_ra]*len(productList), name='ra')
        productList.add_column([s_dec]*len(productList), name='dec')

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

    return(productlist)

  def download_files(self, productlist, dest=None, archivedir=None,
    clobber=False):

    if not productlist:
        error = 'ERROR: product list is empty.  Cannot download files.'
        print(error)
        return(False)

    for prod in productlist:
        filename = prod['productFilename']

        # Cut down new HST filenames that start with hst_PROGID
        filename = '_'.join(filename.split('_')[-2:])

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
                print(message.format(file=filename))
                continue
        elif os.path.isfile(filename):
            message = '{file} exists. Skipping...'
            print(message.format(file=filename))
            continue

        obsid = prod['obsID']

        message = 'Trying to download {image}'
        sys.stdout.write(message.format(image=filename))
        sys.stdout.flush()

        try:
            with suppress_stdout():
                cache = '.'
                download = Observations.download_products(Table(prod),
                    download_dir=cache, cache=False)
            shutil.move(download['Local Path'][0], filename)
            message = '\r' + message
            message += green+' [SUCCESS]'+end+'\n'
            sys.stdout.write(message.format(image=filename))
        except Exception as e:
            message = '\r' + message
            message += red+' [FAILURE]'+end+'\n'
            sys.stdout.write(message.format(image=filename))
            print('Error:', e)

    return(True)

  def sendEmail(self, alert, outfiles=None):

    try:
        to_addr = alert['to_addr'][0]
        login = alert['login'][0]
        password = alert['password'][0]
        smtpserver = alert['smtpserver'][0]
    except:
        return(False)

    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'hst123.py has finished'
    msg['From'] = 'Supernova Alerts: hst123.py'
    msg['To'] = to_addr

    kwargs = {'cmd': self.command, 'dir': os.getcwd(),
        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    message = '''<html><body><p>Finished with hst123.py!</p>
                <p>Command: {cmd}</p>
                <p>Directory: {dir}</p>
                <p>Time finished: {time}</p>
                <p>{text}</p>
                <p>CDK</p>
                </body></html>'''

    kwargs['text']=''
    if outfiles:
        with open(outfiles[0]) as f:
            for line in f:
                kwargs['text']+=line+'<br>'

    if not kwargs['text']: kwargs['text']='No sources detected by dolphot!'

    message = message.format(**kwargs)

    payload = MIMEText(message, 'html')
    msg.attach(payload)

    with smtplib.SMTP(smtpserver) as server:
        try:
            server.starttls()
            server.login(login, password)
            resp = server.sendmail(msg['From'], [to_addr], msg.as_string())
            return(True)
        except:
            return(False)

    return(False)

  def check_large_reduction(self):
    n = len(self.input_images)
    m = self.options['args'].large_num
    if n > m:
        error = 'ERROR: --no_large_reduction and input list size={n}>{m}'
        print(error.format(n=n, m=m))

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
        self.add_image_to_param_file(dolphot_file, reference, 0, dopt)

        # Write out image-specific params to dolphot file
        for i,image in enumerate(images):
            self.add_image_to_param_file(dolphot_file, image, i+1, dopt)

  def run_dolphot(self):
    if os.path.isfile(self.dolphot['param']):
        cmd = 'dolphot {base} -p{par} > {log}'
        cmd = cmd.format(base=self.dolphot['base'], par=self.dolphot['param'],
            log=self.dolphot['log'])
        banner = 'Running dolphot with cmd={cmd}'
        self.make_banner(banner.format(cmd=cmd))
        os.system(cmd)
        print('dolphot is finished (whew)!')
    else:
        error = 'ERROR: dolphot parameter file {file} does not exist!'
        error += ' Generate a parameter file first.'
        print(error.format(file=self.dolphot['param']))

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
        self.make_banner(banner.format(refname))
    else:
        self.make_banner(banner.format('generating from input files'))
        refname = self.pick_reference(obstable)

    # Sanitize extensions and header variables in reference
    banner = 'Sanitizing reference image: {ref}'
    self.make_banner(banner.format(ref=refname))
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
        if not self.image_contains(im, self.coord):
            # If the image doesn't have coord, delete that file
            os.remove(im)
        else:
            if self.needs_to_calc_sky(im):
                self.calc_sky(im, self.options['detector_defaults'])

            outimg.append(im)

    return(outimg)

  def drizzle_all(self, obstable):
    for name in obstable['drizname']:
        mask = obstable['drizname']==name
        driztable = obstable[mask]

        message = 'Constructing drizzled image: {im}'
        print(message.format(im=name))
        self.run_astrodrizzle(driztable, output_name=name)
        self.sanitize_reference(name)

        # Make a sky file for the drizzled image and rename 'noise'
        if (self.needs_to_calc_sky(name)):
            self.compress_reference(name)
            self.calc_sky(name, self.options['detector_defaults'])
            sky_image = name.replace('.fits', '.sky.fits')
            noise_name = name.replace('.fits', '.noise.fits')
            shutil.copy(sky_image, noise_name)

  def get_dolphot_photometry(self, split_images, reference):
    message = 'Starting scrape dolphot for: {ra} {dec}'
    self.make_banner(message.format(ra=ra, dec=dec))

    opt = self.options['args']
    dp = self.dolphot
    if (os.path.exists(dp['colfile']) and
        os.path.exists(dp['base']) and
        os.stat(dp['base']).st_size>0):

        phot = self.scrapedolphot(self.coord, reference, split_images, dp,
            get_limits=True, scrapeall=opt.scrapeall, brightest=opt.brightest)

        self.final_phot = phot

        if phot:
            message = 'Printing out the final photometry for: {ra} {dec}\n'
            message += 'There is photometry for {n} sources'
            message = message.format(ra=ra, dec=dec, n=len(phot))
            self.make_banner(message)

            allphot = self.options['args'].scrapeall
            self.print_final_phot(phot, self.dolphot, allphot=allphot)

        else:
            message = 'WARNING: did not find a source for: {ra} {dec}'
            self.make_banner(message.format(ra=ra, dec=dec))

    else:
        message = 'WARNING: dolphot did not run.  Use the --rundolphot flag'
        message += ' or check your dolphot output for errors before using '
        message += '--scrapedolphot'
        print(message)

  def handle_args(self, parser):
    opt = parser.parse_args()
    self.options['args'] = opt
    default = self.options['global_defaults']

    # If we're cleaning up a previous run, execute that here then exit
    if self.options['args'].makeclean: self.make_clean()

    # Handle other options
    self.reference = self.options['args'].reference
    if opt.alignonly: default['dolphot']['AlignOnly']=1
    if opt.before: self.before=parse(self.options['args'].before)
    if opt.after: self.after=parse(self.options['args'].after)

    # Override drizzled image dimensions
    dim = opt.drizdim
    for det in self.options['detector_defaults'].keys():
        self.options['detector_defaults'][det]['nx']=dim
        self.options['detector_defaults'][det]['ny']=dim

    # If only wide, modify acceptable_filters to those with W, X, or LP
    if opt.onlywide:
        self.options['acceptable_filters'] = [filt for filt in
            self.options['acceptable_filters'] if (filt.upper().endswith('X')
                or filt.upper().endswith('W') or filt.upper().endswith('LP'))]

    if opt.only_filter:
        filts = [f.lower() for f in list(opt.only_filter.split(','))]
        self.options['acceptable_filters'] = [filt for filt in
            self.options['acceptable_filters'] if filt.lower() in filts]

    if opt.fitsky:
        if opt.fitsky in [1,2,3,4]:
            self.options['global_defaults']['dolphot']['FitSky']=opt.fitsky
        else:
            warning = 'WARNING: --fitsky {0} not allowed.  Setting fitsky=2.'
            print(warning.format(opt.fitsky))

    return(opt)

  def do_fake(self, obstable, refname):
    dp = self.dolphot
    gopt = self.options['global_defaults']['fake']
    if not os.path.exists(dp['base'] or os.path.getsize(dp['base'])==0):
        warning = 'WARNING: option --dofake used but dolphot has not been run.'
        print(warning)
        return(None)

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
        print(cmd)
        os.system(cmd)
        print('dolphot fake stars is finished (whew)!')

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

if __name__ == '__main__':
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
    hst.make_banner('Starting: {cmd}'.format(cmd=hst.command))

    # Try to parse the coordinate and check if it's acceptable
    if len(sys.argv) < 3: print(hst.usagestring) ; sys.exit(1)
    else: coord = hst.parse_coord(sys.argv[1], sys.argv[2]) ; hst.coord = coord
    if not hst.coord: print(hst.usagestring) ; sys.exit(1)
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
        banner = 'Downloading HST data from MAST for: {ra} {dec}'
        hst.make_banner(banner.format(ra=ra, dec=dec))
        if opt.archive: hst.dest=None
        if hst.rawdir and not os.path.exists(hst.rawdir):
            os.makedirs(hst.rawdir)
        hst.download_files(hst.productlist, archivedir=opt.archive,
            dest=hst.dest, clobber=opt.clobber)

    if opt.archive and not opt.skip_copy:
        hst.make_banner('Copying raw data to working dir')
        for product in hst.productlist:
            hst.copy_raw_data_archive(product, archivedir=opt.archive,
                workdir=opt.workdir, check_for_coord=True)
    else:
        # Assume that all files are in the raw/ data directory
        hst.make_banner('Copying raw data to working dir')
        hst.copy_raw_data(hst.rawdir, reverse=True, check_for_coord=True)

    # Get input images
    hst.input_images = hst.get_input_images()

    # Check which are HST images that need to be reduced
    hst.make_banner('Checking which images need to be reduced')
    for file in list(hst.input_images):
        warning, needs_reduce = hst.needs_to_be_reduced(file)
        if not needs_reduce:
            print(warning)
            hst.input_images.remove(file)

    # Quit if the number of input files exceeds large reduction limit
    if opt.no_large_reduction: hst.check_large_reduction()

    # Check there are still images that need to be reduced
    if len(hst.input_images)>0:

        # Get metadata on all input images and put them into an obstable
        hst.make_banner('Organizing input images by visit')
        # Going forward, we'll refer everything to obstable for imgs + metadata
        table = hst.input_list(hst.input_images, show=True)
        tables = hst.organize_reduction_tables(table, byvisit=opt.byvisit)

        for i,obstable in enumerate(tables):

            vnum = str(i).zfill(3)
            hst.dolphot = hst.make_dolphot_dict(opt.dolphot+vnum)

            hst.reference = hst.handle_reference(obstable, opt.reference)

            # Run main tweakreg to register to the reference.  Skipping tweakreg
            # will speed up analysis if only running scrapedolphot
            if not opt.skip_tweakreg:
                hst.make_banner('Running main tweakreg')
                error = hst.run_tweakreg(obstable, hst.reference)

            # Drizzle all visit/filter pairs if drizzleall
            if opt.drizzleall and 'drizname' in obstable.keys():
                hst.drizzle_all(obstable)

            # dolphot image preparation: mask_image, split_groups, calc_sky
            message = 'Preparing dolphot data for files={files}.'
            print(message.format(files=','.join(map(str,obstable['image']))))
            split_images = []
            for image in obstable['image']:
                outimg = hst.prepare_dolphot(image)
                split_images.extend(outimg)

            if os.path.exists(hst.reference):
                if hst.needs_to_calc_sky(hst.reference, check_wcs=True):
                    message = 'Running calcsky for reference image: {ref}'
                    print(message.format(ref=hst.reference))
                    hst.compress_reference(hst.reference)
                    hst.calc_sky(hst.reference,hst.options['detector_defaults'])

            # Construct dolphot param file from split images and reference
            banner = 'Adding images to dolphot parameter file: {file}.'
            hst.make_banner(banner.format(file=hst.dolphot['param']))
            hst.make_dolphot_file(split_images, hst.reference)

            # Dolphot using param file and input parameters
            if opt.rundolphot: hst.run_dolphot()

            # Scrape data from the dolphot catalog for the input coordinates
            if opt.scrapedolphot: hst.get_dolphot_photometry(split_images,
                hst.reference)

            # Do fake star injection if --dofake is passed
            if opt.dofake: hst.do_fake(obstable, hst.reference)

    # Write out a list of the input images with metadata for easy reference
    hst.make_banner('Complete list of input images')
    hst.input_list(hst.input_images, show=True, save=False, file=hst.summary)

    # Clean up interstitial files in working directory
    if not opt.nocleanup:
        message = 'Cleaning up {n} input images.'
        hst.make_banner(message.format(n=len(hst.input_images)))
        for image in hst.input_images:
            message = 'Removing image: {im}'
            print(message.format(im=image))
            if os.path.isfile(image):
                os.remove(image)

    if opt.alert:
        hst.make_banner('Sending alert about completed hst123.py script')
        outfiles = glob.glob('dp*.phot')
        hst.sendEmail(Table.read(opt.alert, format='ascii'), outfiles=outfiles)

    message = 'Finished with: {cmd}\n'
    message += 'It took {time} seconds to complete this script.'
    hst.make_banner(message.format(cmd=hst.command, time=time.time()-start))
