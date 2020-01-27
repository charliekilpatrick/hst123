#!/usr/bin/env python

"""
By C. D. Kilpatrick 2019-02-07

v1.00: 2019-02-07. Base hst123 download, tweakreg, drizzle, dolphot param
v1.01: 2019-02-15. Added running dolphot, scraping dolphot output
v1.02: 2019-02-22. Added fake star injection
v1.03: 2019-06-02. Added drizzleall options and cleaned up options/syntax

hst123.py: An all-in-one script for downloading, registering, drizzling,
running dolphot, and scraping data from dolphot catalogs.
"""

# Python 2/3 compatibility
from __future__ import print_function

try:
    input = raw_input # use 'input' function in both Python 2 and 3
except NameError:
    pass

# Dependencies and settings
import glob, sys, os, shutil, time
import re, subprocess, warnings, filecmp, astroquery
import astropy.wcs as wcs
import numpy as np
from astropy import units as u
from astropy import utils
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astroquery.mast import Observations
from astroscrappy import detect_cosmics
from datetime import datetime
from dateutil.parser import parse
from drizzlepac import tweakreg,astrodrizzle
from photutils import SkyCircularAperture,aperture_photometry
from stwcs import updatewcs
warnings.filterwarnings('ignore')

# Color strings for download messages
green = '\033[1;32;40m'
red = '\033[1;31;40m'
end = '\033[0;0m'

# Dictionaries with all options for instruments, detectors, and filters
global_defaults = {
    'keys': ['IDCTAB','DGEOFILE','NPOLEXT',
             'NPOLFILE','D2IMFILE', 'D2IMEXT','OFFTAB'],
    'cdbs': 'ftp://ftp.stsci.edu/cdbs/',
    'mast': 'https://mast.stsci.edu/api/v0/download/file?uri=',
    'visit': 1,
    'radius': 5 * u.arcmin,
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
    'fake': {'filter1': None,
             'filter2': None,
             'filter1_min': 18.0,
             'filter1_max': 30.0,
             'color_min': -5.0,
             'color_max': 5.0,
             'nstars': 50000}}

instrument_defaults = {
    'wfc3': {'env_ref': 'iref.old',
             'crpars': {'rdnoise': 6.5,
                        'gain': 1.0,
                        'saturation': 70000.0,
                        'sig_clip': 4.0,
                        'sig_frac': 0.2,
                        'obj_lim': 6.0}},
    'acs': {'env_ref': 'jref.old',
            'crpars': {'rdnoise': 6.5,
                       'gain': 1.0,
                       'saturation': 70000.0,
                       'sig_clip': 3.0,
                       'sig_frac': 0.1,
                       'obj_lim': 5.0}},
    'wfpc2': {'env_ref': 'uref',
              'crpars': {'rdnoise': 10.0,
                         'gain': 7.0,
                         'saturation': 27000.0,
                         'sig_clip': 4.0,
                         'sig_frac': 0.3,
                         'obj_lim': 6.0}}}

detector_defaults = {
    'wfc3_uvis': {'driz_bits': 0, 'nx': 5200, 'ny': 5200,
                  'input_files': '*_flc.fits', 'pixel_scale': 0.04,
                  'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                  'sigma_low': 2.25, 'sigma_high': 2.00},
                  'dolphot': {'apsky': '15 25', 'RAper': 3, 'RChi': 2.0,
                              'RPSF': 13, 'RSky': '15 35',
                              'RSky2': '4 10'}},
    'wfc3_ir': {'driz_bits': 512, 'nx': 5200, 'ny': 5200,
                'input_files': '*_flt.fits', 'pixel_scale': 0.09,
                'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '8 20', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 15, 'RSky': '8 20',
                            'RSky2': '3 10'}},
    'acs_wfc': {'driz_bits': 0, 'nx': 5200, 'ny': 5200,
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
    'wfpc2_wfpc2': {'driz_bits': 0, 'nx': 5200, 'ny': 5200,
                    'input_files': '*_c0m.fits', 'pixel_scale': 0.05,
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
    'F600LP','F621M','F625W','F631N','F645N','F656N','F657N','F665N',
    'F673N','F680N','F689M','F763M','F845M','F953N','F122M','F160BW','F185W',
    'F218W','F255W','F300W','F375N','F380W','F390N','F437N','F439W','F450W',
    'F569W','F588N','F622W','F631N','F673N','F675W','F702W','F785LP','F791W',
    'F953N','F1042M'}

# AB mag zero points for all HST imaging filters
zeropoint = {
    'wfc3_uvis': {'f200lp': 27.430, 'f218w': 22.952, 'f225w': 24.078,
        'f275w': 24.169, 'f280n': 20.943, 'f300x': 24.985, 'f336w': 24.708,
        'f343n': 23.906, 'f350lp': 26.982, 'f373n': 21.920, 'f390m': 23.641,
        'f390w': 25.394, 'f395n': 22.688, 'f410m': 23.614, 'f438w': 24.856,
        'f467m': 23.702, 'f469n': 21.827, 'f475w': 25.722, 'f475x': 26.178,
        'f487n': 22.244, 'f502n': 22.326, 'f547m': 24.771, 'f555w': 25.824,
        'f600lp': 25.911, 'f606w': 26.103, 'f621m': 24.626, 'f625w': 25.550,
        'f631n': 21.904, 'f645n': 22.260, 'f656n': 20.486, 'f657n': 22.670,
        'f658n': 21.056, 'f665n': 22.747, 'f673n': 22.596, 'f680n': 23.813,
        'f689m': 24.493, 'f763m': 24.238, 'f775w': 24.890, 'f814w': 25.139,
        'f845m': 23.823, 'f850lp': 23.888, 'f953n': 20.452},
    'wfc3_ir': {'f105w': 26.2687, 'f110w': 26.8223, 'f125w': 26.2303,
        'f140w': 26.4524, 'f160w': 25.9463, 'f098m': 25.6674, 'f127m': 24.6412,
        'f139m': 24.4793, 'f153m': 24.4635, 'f126n': 22.8609, 'f128n': 22.9726,
        'f130n': 22.9900, 'f132n': 22.9472, 'f164n': 22.9089, 'f167n': 22.9568},
    'acs_wfc': {'f435w': 25.660, 'f475w': 26.052,'f502n': 22.283,
        'f550m': 24.852, 'f555w': 25.710, 'f606w': 26.493, 'f625w': 25.899,
        'f658n': 22.757, 'f660n': 21.707, 'f775w': 25.659, 'f814w': 25.937,
        'f850lp': 24.851, 'f892n': 22.393},
    'acs_hrc': {'f220w': 23.523, 'f250w': 23.734, 'f330w': 24.085,
        'f344n': 21.546, 'f435w': 25.097, 'f475w': 25.538, 'f502n': 21.824,
        'f550m': 24.441, 'f555w': 25.251, 'f606w': 25.984, 'f625w': 25.366,
        'f658n': 22.186, 'f660n': 21.129, 'f775w': 24.951, 'f814w': 25.282,
        'f850lp': 24.404, 'f892n': 21.871},
    'wfpc2_wfpc2': {'f122m': 13.752, 'f160bw': 14.946, 'f170w': 16.313,
        'f185w': 16.014, 'f218w': 16.558, 'f255w': 17.037, 'f300w': 19.433,
        'f336w': 19.460, 'f343n': 14.023, 'f375n': 15.238, 'f380w': 20.972,
        'f390n': 17.537, 'f410m': 19.669, 'f437n': 17.297, 'f439w': 20.916,
        'f450w': 22.016, 'f467m': 20.012, 'f469n': 17.573, 'f487n': 17.380,
        'f502n': 17.988, 'f547m': 21.676, 'f555w': 22.561, 'f569w': 22.253,
        'f588n': 19.179, 'f606w': 22.896, 'f622w': 22.368, 'f631n': 18.516,
        'f656n': 17.564, 'f658n': 18.115, 'f673n': 18.753, 'f675w': 22.042,
        'f702w': 22.431, 'f785lp': 20.738, 'f791w': 21.512, 'f814w': 21.659,
        'f850lp': 20.018, 'f953n': 16.186, 'f1042m': 16.326}
}

class hst123(object):

  def __init__(self):

    # Basic parameters
    self.input_images = []
    self.split_images = []
    self.fake_images = []
    self.obstable = None

    self.reference = ''
    self.root_dir = '.'
    self.raw_dir = 'raw/'

    self.before = None
    self.after = None
    self.coord = None

    self.reffilter = None
    self.object = None

    self.keepshort = False
    self.clobber = False
    self.nocleanup = False

    self.dolphot = {}

    # Names for input image table
    self.names = ['image','exptime','datetime','filter','instrument']

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
                    'zeropoint': zeropoint}

    # List of pipeline products in case they need to be cleaned at start
    self.pipeline_products = ['*chip?.fits', '*chip?.sky.fits',
                              '*rawtmp.fits', '*drz.fits', '*drz.sky.fits',
                              '*idc.fits', '*dxy.fits', '*off.fits',
                              '*d2im.fits', '*d2i.fits', '*npl.fits',
                              'dp*', '*.log', '*.output','*sci?.fits',
                              '*wht.fits','*sci.fits','*StaticMask.fits']

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
    return(parser)

  # Make sure all standard output is formatted in the same way with banner
  # messages for each module
  def make_banner(self, message):
    n = 80
    print('')
    print('')
    print(message)
    # Print part below the message in the banner
    print('#' * n)
    print('#' * n)
    print('')
    print('')

  def make_clean(self):
    question = 'Are you sure you want to delete '
    question += 'output from previous hst123 runs? [y/n] '
    var = input(question)
    if var != 'y' and var != 'yes':
        warning = 'WARNING: input={inp}. Exiting...'
        print(warning)
        sys.exit(1)
    for pattern in self.pipeline_products:
        for file in glob.glob(pattern):
            if os.path.isfile(file):
                os.remove(file)
    self.copy_raw_data(reverse = True)
    sys.exit(0)

  def get_zpt(self, det, filt):
    # Get zero point given an input instrument and filter
    return(self.options['zeropoint'][det][filt])


  def avg_magnitudes(self, magerrs, counts, exptimes, zpt):
    # Mask out bad values
    idx = []
    for i,data in enumerate(magerrs):
        if data < 0.5:
            idx.append(i)

    if not idx:
        return (float('NaN'),float('NaN'))

    magerrs = np.array(magerrs)[idx]
    counts = np.array(counts)[idx]
    exptimes = np.array(exptimes)[idx]
    zpt = np.array(zpt)[idx]

    # Now normalize flux and fluxerr to a common zero point
    flux = counts / exptimes * 10**(0.4 * (27.5 - zpt))
    fluxerr = 1./1.086 * magerrs * flux

    # Now calculate average and propagate uncertainties weighted by fluxerr
    average_flux = np.sum(flux*1/fluxerr**2)/np.sum(1/fluxerr**2)
    average_fluxerr = np.sqrt(np.sum(fluxerr**2))/len(fluxerr)

    # Now transform back to magnitude and magnitude error
    final_mag = 27.5 - 2.5 * np.log10(average_flux)
    final_magerr = 1.086 * average_fluxerr / average_flux

    return(final_mag, final_magerr)

  # Given a row of a dolphot file and a columns file, return the data key
  # described by 'data' (e.g., 'VEGAMAG', 'Total counts',
  # 'Magnitude uncertainty') for the input image name
  def get_dolphot_data(self, row, columns, data, image):
    # First get the column number from the start of the column file
    l = ''
    with open(columns) as f:
        for line in f:
            if image.strip('.fits') in line and data in line:
                l = line.strip().strip('\n')
                break
    # Now parse the line to get the column number, indexed to 0
    colnum = int(l.split('.')[0])-1

    # Now grab that piece of data from the row and return
    return(row.split()[colnum])

  def show_photometry(self, final_photometry, latex=False, file=None):
    # Check to make sure dictionary is set up properly
    keys = final_photometry.keys()
    if ('INSTRUMENT' not in keys or 'FILTER' not in keys
        or 'MJD' not in keys or 'MAGNITUDE' not in keys
        or 'MAGNITUDE_ERROR' not in keys or 'EXPTIME' not in keys):
       error = 'ERROR: photometry table has a key error'
       print(error)
       sys.exit(1)
    else:
        form = ''
        if latex:
            form = '{date: <10} & {inst: <10} & {filt: <10} '
            form += '{exp: <10} & {mag: <8} & {err: <8} \\\\'
            header = form.format(date='MJD', inst='Instrument',
                                 filt='Filter', exp='Exposure Time',
                                 mag='Magnitude', err='Uncertainty')
            units = form.format(date='(MJD)', inst='', filt='', exp='(s)',
                                mag='', err='')
            print(header)
            print(units)
            if file is not None:
                file.write(header+'\n')
                file.write(units+'\n')
        else:
            form = '{date: <13} {inst: <10} {filt: <8} '
            form += '{exp: <14} {mag: <9} {err: <9}'
            header = form.format(date='# MJD', inst='Instrument',
                                 filt='Filter', exp='Exposure Time',
                                 mag='Magnitude', err='Uncertainty')
            print(header)
            if file is not None:
                file.write(header+'\n')

        for row in final_photometry:
            line = form.format(date=row['MJD'], inst=row['INSTRUMENT'],
                               filt=row['FILTER'], exp='%7.4f' % row['EXPTIME'],
                               mag='%3.4f' % row['MAGNITUDE'],
                               err='%3.4f' % row['MAGNITUDE_ERROR'])
            print(line)
            if file is not None:
                file.write(line+'\n')

  def input_list(self, img, show=True):
    # Make a table with all of the metadata for each image.
    exp = [fits.getval(image,'EXPTIME') for image in img]
    dat = [fits.getval(image,'DATE-OBS') + 'T' +
           fits.getval(image,'TIME-OBS') for image in img]
    fil = [self.get_filter(image) for image in img]
    ins = [self.get_instrument(image) for image in img]

    # Save this obstable.  Useful for other methods
    self.obstable = Table([img,exp,dat,fil,ins], names=self.names)

    # Look at the table in order of date
    self.obstable.sort('datetime')

    # Show the obstable in a column formatted style
    if show:
        form = '{file: <19} {inst: <18} {filt: <10} '
        form += '{exp: <10} {date: <10} {time: <10}'
        header = form.format(file='FILE',inst='INSTRUMENT',filt='FILTER',
                             exp='EXPTIME',date='DATE-OBS',time='TIME-OBS')
        print(header)

        for row in self.obstable:
            line = form.format(file=row['image'],
                    inst=row['instrument'].upper(),
                    filt=row['filter'].upper(), exp=row['exptime'],
                    date=Time(row['datetime']).datetime.strftime('%Y-%m-%d'),
                    time=Time(row['datetime']).datetime.strftime('%H:%M:%S'))
            print(line)

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
             'fake_out': dolphot+'.fakeout', 'radius': 2,
             'final_phot': dolphot+'.phot'})

  # Copy raw data into raw data dir
  def copy_raw_data(self, reverse = False):
    # reverse = False will simply backup data in the working directory to the
    # raw dir
    if not reverse:
        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)
        for f in self.input_images:
            if not os.path.isfile(self.raw_dir+f):
                # Create new file and change permissions
                shutil.copyfile(f, self.raw_dir+f)
    # reverse = True will copy files from the raw dir to the working directory
    # if the working files are different from the raw dir.  This is necessary
    # for the pipeline to work properly, as some procedures (esp. WCS checking,
    # tweakreg, astrodrizzle) require un-edited files.
    else:
        for file in glob.glob(self.raw_dir+'*.fits'):
            path, base = os.path.split(file)
            # Should catch error where 'base' does not exist
            if os.path.isfile(base) and filecmp.cmp(file,base):
                message = '{file} == {base}'
                print(message.format(file=file,base=base))
                continue
            else:
                message = '{file} != {base}'
                print(message.format(file=file,base=base))
                shutil.copyfile(file, base)

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
                    return(1)
                else:
                    # visit number is equal to other values in set
                    row['visit'] = list(set(obstable[mask]['visit']))[0]
    return(0)

  # Given an input dictionary of files sorted by visits, a row from a dolphot
  # output file, and a column file corresponding to that dolphot file, parse the
  # output into a photometry table
  def parse_phot(self, obstable, row, column_file):
    # Names for the final output photometry table
    final_names = ['MJD', 'INSTRUMENT', 'FILTER',
                   'EXPTIME', 'MAGNITUDE', 'MAGNITUDE_ERROR']
    # Make an empty table with above column names for output photometry table
    final_phot = Table([[0.],['INSTRUMENT'],['FILTER'],[0.],[0.],[0.]],
        names=final_names)[:0].copy()

    # Iterate through the dictionary to get all photometry from files
    for inst in list(set(obstable['instrument'])):
        insttable = obstable[obstable['instrument'] == inst]
        for visit in list(set(insttable['visit'])):
            visittable = insttable[insttable['visit'] == visit]
            for filt in list(set(visittable['filter'])):
                filttable = visittable[visittable['filter'] == filt]
                mjds, mags, magerrs, counts, exptimes, zpts = ([] for i in range(6))
                for im in filttable['image']:
                    mjds.append(Time(str(fits.getval(im,'DATE-OBS')) + 'T' +
                             str(fits.getval(im,'TIME-OBS'))).mjd)
                    mags.append(float(self.get_dolphot_data(row[1], 'dp.columns',
                            'Instrumental', im)))
                    magerrs.append(float(self.get_dolphot_data(row[1],
                        'dp.columns', 'Magnitude uncertainty', im)))
                    counts.append(float(self.get_dolphot_data(row[1],
                              'dp.columns', 'Measured counts', im)))
                    exptimes.append(float(fits.getval(im, 'EXPTIME')))
                    det = det = '_'.join(self.get_instrument(im).split('_')[:2])
                    zpts.append(float(self.get_zpt(det, filt)))

                # Calculate average of mjds, total of exptimes
                avg_mjd = np.mean(mjds)
                total_exptime = np.sum(exptimes)

                # Average magnitude and magnitude errors
                mag, magerr = self.avg_magnitudes(magerrs,
                    counts, exptimes, zpts)

                final_phot.add_row((avg_mjd, inst, filt,
                                    total_exptime, mag, magerr))
    return(final_phot)

  # Sanitizes reference header, gets rid of multiple extensions and only
  # preserves science data.
  def sanitize_reference(self, reference):
    hdu = fits.open(reference, mode='readonly')

    # Going to write out newhdu
    # Only want science extension from orig reference
    newhdu = fits.HDUList()
    newhdu.append(hdu['PRIMARY'])

    # Want to preserve header info, so combine PRIMARY headers
    for key in hdu['PRIMARY'].header.keys():
        if (key not in newhdu['PRIMARY'].header.keys()+['COMMENT','HISTORY']):
            newhdu['PRIMARY'].header[key] = hdu['PRIMARY'].header[key]

    # Make sure that reference header reflects one extension
    newhdu['PRIMARY'].header['EXTEND']=False

    # Add header variables that dolphot needs: GAIN, RDNOISE, SATURATE
    inst = newhdu['PRIMARY'].header['INSTRUME'].lower()
    opt  = self.options['instrument_defaults'][inst]['crpars']
    newhdu['PRIMARY'].header['SATURATE'] = opt['saturation']
    newhdu['PRIMARY'].header['RDNOISE']  = opt['rdnoise']
    newhdu['PRIMARY'].header['GAIN']     = opt['gain']

    if 'WHT' in [h.name for h in hdu]:
        wght = hdu['WHT'].data
        newhdu['PRIMARY'].data[np.where(wght == 0)] = float('NaN')
    elif os.path.exists(reference.replace('.fits', '.weight.fits')):
        wght = fits.open(reference.replace('.fits', '.weight.fits'))[0].data
        newhdu['PRIMARY'].data[np.where(wght == 0)] = float('NaN')

    # Write out to same file w/ overwrite
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
    if (':' in ra and ':' in dec):
        # Input RA/DEC are sexagesimal
        return(SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg)))
    elif (self.is_number(ra) and self.is_number(dec)):
        # Assume input coordiantes are decimal degrees
        return(SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg)))
    else:
        # Throw an error and exit
        error = 'ERROR: Cannot parse coordinates ra={ra}, dec={dec}'
        print(error.format(ra=ra,dec=dec))
        return(None)

  def needs_to_be_reduced(self,image):
    hdu = fits.open(image, mode='readonly')
    is_not_hst_image = False
    warning = ''
    detector = ''
    instrument = hdu[0].header['INSTRUME'].lower()
    if 'c1m.fits' in image:
        # We need the c1m.fits files, but they aren't reduced as science data
        warning = 'WARNING: do not need to reduce c1m.fits files.'
        return(warning, False)

    if ('DETECTOR' in hdu[0].header.keys()):
        detector = hdu[0].header['DETECTOR'].lower()

    # Get rid of exposures with exptime < 20s
    if not options.keepshort:
        exptime = hdu[0].header['EXPTIME']
        if (exptime < 20):
            warning = 'WARNING: {img} EXPTIME is {exp} < 20.'
            return(warning.format(img=image, exp=exptime), False)

    # Now check date and compare to self.before
    mjd_obs = Time(hdu[0].header['DATE-OBS']+'T'+hdu[0].header['TIME-OBS']).mjd
    if self.before is not None:
        mjd_before = Time(self.before).mjd
        if mjd_obs > mjd_before:
            warning = 'WARNING: {img} is after the input before date {date}.'
            return(warning.format(img=image,
                                  date=self.before.strftime('%Y-%m-%d')), False)
    # Same with self.after
    if self.after is not None:
        mjd_after = Time(self.after).mjd
        if mjd_obs < mjd_after:
            warning = 'WARNING: {img} is before the input after date {date}.'
            return(warning.format(img=image,
                                  date=self.after.strftime('%Y-%m-%d')), False)

    # Get rid of data where the input coordinates do
    # not land in any of the sub-images
    if self.coord:
        for h in hdu:
            if h.data is not None and 'EXTNAME' in h.header:
                if h.header['EXTNAME'] == 'SCI':
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
            warning = 'WARNING: {img} does not contain ra={ra}, dec={dec}.'
            ra = self.coord.ra.degree
            dec = self.coord.dec.degree
            return(warning.format(img=image, ra=ra, dec=dec), False)

    # Get rid of images that don't match one of the allowed instrument/detector
    # types and images whose extensions don't match the allowed type for those
    # instrument/detector types
    is_not_hst_image = False
    warning = 'WARNING: {img} with INSTRUME={inst}, DETECTOR={det} is bad.'
    if (instrument.upper() == 'WFPC2' and 'c0m.fits' in image):
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
    return(warning.format(img=image, inst=instrument, det=detector),
        is_not_hst_image)

  def needs_to_split_groups(self,image):
    return(len(glob.glob(image.replace('.fits', '.chip?.fits'))) == 0)

  def needs_to_calc_sky(self, image):
    files = glob.glob(image.replace('.fits','.sky.fits'))
    if (len(files) == 0):
        return(True)

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

  # Get the data quality image for dolphot mask routine.  This is entirely for
  # WFPC2 since mask routine requires additional input for this instrument
  def get_dq_image(self,image):
    if self.get_instrument(image).split('_')[0].upper() == 'WFPC2':
        return(image.replace('c0m.fits','c1m.fits'))
    else:
        return('')

  # Run the dolphot splitgroups routine
  def split_groups(self,image):
    print('Running split groups for {image}'.format(image=image))
    os.system('splitgroups {filename}'.format(filename=image))
    # Modify permissions
    #for file in glob.glob(image.replace('.fits', '.chip?.fits')):
    #    os.chmod(file, 0775)

  # Run the dolphot mask routine for the input image
  def mask_image(self, image, instrument):
    maskimage = self.get_dq_image(image)
    cmd = '{instrument}mask {image} {maskimage}'
    mask = cmd.format(instrument=instrument, image=image, maskimage=maskimage)
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
    os.system(calc_sky)
    #for file in glob.glob(image.replace('.fits','.sky.fits')):
    #    os.chmod(file, 0775)


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

  # Pick the best reference out of input images.  Returns the filter of the
  # reference image. Also generates a drizzled image corresponding reference
  def pick_reference(self):
    # If we haven't defined input images, catch error
    if not self.input_images:
        warning = 'ERROR: No input images. Exiting...'
        print(warning)
        sys.exit(1)
    else:
        # Best possible filter for a dolphot reference image in the approximate
        # order I would want to use for a reference image.  You can also use
        # to force the script to pick a reference image from a specific filter.
        best_filters = ['f606w','f555w','f814w','f350lp']

        if self.reffilter:
            if self.reffilter.upper() in self.options['acceptable_filters']:
                best_filters = [self.reffilter.lower()]

        # Best filter suffixes in the approximate order we would want to use to
        # generate a templatea.
        best_types = ['lp', 'w', 'x', 'm', 'n']

        # First group images together by filter/instrument
        filts = [self.get_filter(im) for im in self.input_images]
        insts = [self.get_instrument(im).split('_')[0]
                 for im in self.input_images]

        # Group images together by unique instrument/filter pairs and then
        # calculate the total exposure time for all pairs.
        unique_filter_inst = list(set(['{}_{}'.format(a_, b_)
                                       for a_, b_ in zip(filts, insts)]))

        # Don't construct reference image from acs/hrc if avoidable
        if any(['hrc' not in val for val in unique_filter_inst]):
            # remove all elements with hrc
            new = [val for val in unique_filter_inst if 'hrc' not in val]
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
        for im in self.input_images:
            if (self.get_filter(im)+'_'+
                self.get_instrument(im).split('_')[0] == best_filt_inst):
                reference_images.append(im)
        best_filt,best_inst = best_filt_inst.split('_')

        # Generate output name like other drizzled image
        # Make a photpipe-like image name
        drizname = ''
        if self.object:
            drizname = '{obj}.{inst}.{filt}.ref.drz.fits'
            drizname = drizname.format(inst=best_inst, filt=best_filt,
                obj=self.object)
        else:
            drizname = '{inst}.{filt}.ref.drz.fits'
            drizname = drizname.format(inst=best_inst, filt=best_filt)

        output_name = drizname
        message = 'Reference image name will be: {reference}. '
        message += 'Generating from input files {files}.'
        print(message.format(reference=output_name, files=reference_images))
        self.reference = output_name

        self.run_tweakreg(reference_images, '')
        self.run_astrodrizzle(reference_images, output_name = output_name)

  # Run the drizzlepac astrodrizzle routine using detector parameters.
  def run_astrodrizzle(self, images, output_name = None):
    if output_name is None:
        output_name = 'drizzled.fits'

    num_images = len(images)
    if num_images == 1:
        combine_type = 'mean'
    else:
        combine_type = 'minmed'

    wcskey = 'TWEAK'

    inst = list(set([self.get_instrument(im).split('_')[0] for im in images]))
    det = '_'.join(self.get_instrument(images[0]).split('_')[:2])
    if len(inst) > 1:
        error = 'ERROR: Cannot drizzle together images from detectors: {det}.'
        error += 'Exiting...'
        print(error.format(det=','.join(map(str,inst))))
        sys.exit(1)

    options = self.options['detector_defaults'][det]
    change_keys = self.options['global_defaults']['keys']
    for image in images:
        inst = self.get_instrument(image).split('_')[0]
        ref_url = self.options['instrument_defaults'][inst]['env_ref']
        ref = ref_url.strip('.old')
        for key in change_keys:
            try:
                val = fits.getval(image, key, extname='PRIMARY')
            except KeyError:
                try:
                    val = fits.getval(image, key, extname='SCI')
                except KeyError:
                    error = 'WARNING: {key} is not part of {image} header.'
                    print(error.format(key=key,image=image))
                    continue
            if val == 'N/A':
                continue
            if (ref+'$' in val):
                ref_file = val.split('$')[1]
            else:
                ref_file = val
            if not os.path.exists(ref_file):
                url = self.options['global_defaults']['cdbs']
                url += ref_url+'/'+ref_file
                message = 'Downloading file: {url}'
                sys.stdout.write(message.format(url=url))
                sys.stdout.flush()
                try:
                    # utils.data.download_file can get buggy if the cache is
                    # full.  Clear the cache even though we aren't using
                    # caching to prevent download method from choking
                    utils.data.clear_download_cache()
                except RuntimeError:
                    pass
                try:
                    dat = utils.data.download_file(url, cache=False,
                        show_progress=False, timeout=120)
                    shutil.move(dat, ref_file)
                    message = '\r' + message
                    message += green+' [SUCCESS]'+end+'\n'
                    sys.stdout.write(message.format(url=url))
                except:
                    message = '\r' + message
                    message += red+' [FAILURE]'+end+'\n'
                    sys.stdout.write(message.format(url=url))
                    print(message.format(url=url))
            fits.setval(image, key, extname='PRIMARY', value=ref_file,
                output_verify='silentfix')
            fits.setval(image, key, extname='SCI', value=ref_file,
                output_verify='silentfix')
        updatewcs.updatewcs(image)

    ra = self.coord.ra.degree if self.coord else None
    dec = self.coord.dec.degree if self.coord else None

    if self.keepshort:
        skysub = False
    else:
        skysub = True

    astrodrizzle.AstroDrizzle(images, output=output_name, runfile='',
                wcskey=wcskey, context=True, group='', build=False,
                num_cores=8, preserve=False, clean=True, skysub=skysub,
                skystat='mode', skylower=0.0, skyupper=None, updatewcs=True,
                driz_sep_fillval=-50000, driz_sep_bits=0, driz_sep_wcs=True,
                driz_sep_rot=0.0, driz_sep_scale=options['pixel_scale'],
                driz_sep_outnx=options['nx'], driz_sep_outny=options['ny'],
                driz_sep_ra=ra, driz_sep_dec=dec,
                combine_maskpt=0.2, combine_type=combine_type,
                combine_nlow=0, combine_nhigh=0, combine_lthresh=-10000,
                combine_hthresh=None, combine_nsigma='4 3',
                driz_cr=True, driz_cr_snr='3.5 3.0', driz_cr_grow=1,
                driz_cr_ctegrow=0, driz_cr_scale='1.2 0.7',
                final_pixfrac=1.0, final_fillval=-50000,
                final_bits=options['driz_bits'], final_units='counts',
                final_wcs=True, final_refimage=None,
                final_rot=0.0, final_scale=options['pixel_scale'],
                final_outnx=options['nx'], final_outny=options['ny'],
                final_ra=ra, final_dec=dec)

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

    # Add header keys on drizzled file
    hdu = fits.open(output_name, mode='update')
    filt = self.get_filter(images[0])
    hdu[0].header['FILTER'] = filt.upper()
    hdu[0].header['TELID'] = 'HST'
    hdu[0].header['OBSTYPE'] = 'OBJECT'
    if self.object:
        hdu[0].header['TARGNAME'] = self.object
        hdu[0].header['OBJECT'] = self.object
    hdu.close()

  # Run cosmic ray clean
  def run_cosmic(self,image,options,output=None):
    message = 'Cleaning cosmic rays in image: {image}'
    print(message.format(image=image))
    hdulist = fits.open(image,mode='readonly')

    if output is None:
        output = image

    for hdu in hdulist:
        if ('EXTNAME' in hdu.header and hdu.header['EXTNAME'].upper() == 'SCI'):
            mask = np.zeros(hdu.data.shape, dtype=np.bool)

            _crmask, crclean = detect_cosmics(hdu.data.copy().astype('<f4'),
                                              inmask=mask,
                                              readnoise=options['rdnoise'],
                                              gain=options['gain'],
                                              satlevel=options['saturation'],
                                              sigclip=options['sig_clip'],
                                              sigfrac=options['sig_frac'],
                                              objlim=options['obj_lim'])
            hdu.data[:,:] = crclean[:,:]

    # This writes in place
    hdulist.writeto(output, overwrite=True, output_verify='silentfix')
    hdulist.close()

  # Run tweakreg on all input images
  def run_tweakreg(self, images, reference):

    # Get options from object
    options = self.options['global_defaults']

    # Make a copy of the input image list
    run_images = list(images)

    # Check if tweakreg has already been run on each image and if it has then
    # assume images are aligned
    for file in list(run_images):
        hdu = fits.open(file, mode='readonly')
        remove_image = False
        for h in hdu:
            if h.name == 'SCI':
                header = h.header
                for key in header.keys():
                    if 'WCSNAME' in key:
                        if header[key] == 'TWEAK':
                                remove_image = True
        if remove_image:
            run_images.remove(file)


    # Check if we just removed all of the images
    if (len(run_images) == 0):
        warning = 'WARNING: All images have been run through tweakreg.'
        print(warning)
        return(1)

    for image in run_images:
        # wfc3_ir doesn't need cosmic clean and assume reference is cleaned
        if (image == reference or 'wfc3_ir' in self.get_instrument(image)):
            continue
        rawtmp = image.replace('.fits','rawtmp.fits')

        # Copy the raw data into a temporary file
        shutil.copyfile(image, rawtmp)

        # Clean cosmic rays on the image in place. We do this so we don't
        # accidentally pick cosmic rays for alignment
        inst = self.get_instrument(image).split('_')[0]
        crpars = self.options['instrument_defaults'][inst]['crpars']
        self.run_cosmic(image,crpars)

    if (reference == '' or reference == None):
        reference = 'dummy.fits'
        shutil.copyfile(images[0], reference)

    message = 'Executing tweakreg with images: {images} \n'
    message += 'Reference image: {reference} \n'
    message += 'Tweakreg is executing...'
    print(message.format(images = ','.join(map(str,run_images)),
                         reference = reference))
    start_tweak = time.time()
    tweakreg.TweakReg(files=run_images, refimage=reference, verbose=False,
            interactive=False, clean=True, writecat = True, updatehdr=True,
            wcsname='TWEAK', reusename=True, rfluxunits='counts', minobj=10,
            searchrad=2.0, searchunits='arcseconds', runfile='',
            see2dplot=False, separation=0.5, residplot='No plot',
            imagefindcfg = {'threshold': 5, 'use_sharp_round': True},
            refimagefindcfg = {'threshold': 5, 'use_sharp_round': True})

    message = 'Tweakreg took {time} seconds to execute.'
    print(message.format(time = time.time()-start_tweak))

    for image in run_images:
        if (image == reference or 'wfc3_ir' in self.get_instrument(image)):
            continue
        rawtmp = image.replace('.fits','rawtmp.fits')
        hdu_from = fits.open(rawtmp)
        hdu_to = fits.open(image)

        for i, hdu in enumerate(hdu_from):
            if ('EXTNAME' in hdu.header and
                hdu.header['EXTNAME'].upper() == 'SCI'):
                hdu_to[i].data[:,:] = hdu.data[:,:]

        hdu_from.close()
        hdu_to.writeto(image,overwrite = True)
        hdu_to.close()
        if os.path.isfile(rawtmp):
            os.remove(rawtmp)

    if os.path.isfile('dummy.fits'):
        os.remove('dummy.fits')

    return(0)

  # download files for input self.coord
  def download_files(self):
    # Check for coordinate and exit if it does not exist
    if not self.coord:
        error = 'ERROR: coordinate was not provided.  Exiting...'
        print(error)
        sys.exit(1)

    # Define the search coord/radius and grab all files from MAST that
    # correspond to this region
    search_radius = self.options['global_defaults']['radius']
    try:
        obsTable = Observations.query_region(self.coord, radius=search_radius)
    except astroquery.exceptions.RemoteServiceError:
        error = 'ERROR: MAST is not working for some reason. '
        error += 'Try again later.  Exiting...'
        print(error)
        sys.exit(1)


    # Get rid of all masked rows as (pretty sure) they aren't HST data anyway
    obsTable = obsTable.filled()

    # Check for before and after to mask data
    if self.before is not None:
        obsTable = obsTable[obsTable['t_min'] < Time(self.before).mjd]
    if self.after is not None:
        obsTable = obsTable[obsTable['t_min'] > Time(self.after).mjd]

    # Construct masks for telescope, data type, detector, and data rights
    telmask = [tel.upper() == 'HST' for tel in obsTable['obs_collection']]
    promask = [pro.upper() == 'IMAGE' for pro in obsTable['dataproduct_type']]
    detmask = [any(l) for l in list(map(list,zip(*[[det in inst.upper()
                for inst in obsTable['instrument_name']]
                for det in ['ACS','WFC','WFPC2']])))]
    ritmask = [str(rit).upper() == 'PUBLIC' for rit in obsTable['dataRights']]

    # Apply the masks to the observation table
    mask = [all(l) for l in zip(telmask,promask,detmask,ritmask)]
    obsTable = obsTable[mask]

    # Iterate through each observation and download the correct product
    # depending on the filename and instrument/detector of the observation
    for obs in obsTable:
        productList = Observations.get_product_list(obs)
        instrument = obs['instrument_name']
        for prod in productList:
            filename = prod['productFilename']
            if (os.path.isfile(filename) and not self.clobber):
                message = '{file} exists and clobber = False. Skipping...'
                print(message.format(file = filename))
                continue
            if (('c0m.fits' in filename and 'WFPC2' in instrument) or
                ('c1m.fits' in filename and 'WFPC2' in instrument) or
                ('c0m.fits' in filename and 'PC/WFC' in instrument) or
                ('c1m.fits' in filename and 'PC/WFC' in instrument) or
                ('flc.fits' in filename and 'ACS/WFC' in instrument) or
                ('flt.fits' in filename and 'ACS/HRC' in instrument) or
                ('flc.fits' in filename and 'WFC3/UVIS' in instrument) or
                ('flt.fits' in filename and 'WFC3/IR' in instrument)):
                obsid = prod['obsID']
                uri = prod['dataURI']

                message = 'Trying to download {image}'
                sys.stdout.write(message.format(image=filename))
                sys.stdout.flush()

                url = self.options['global_defaults']['mast'] + uri
                try:
                    # utils.data.download_file can get buggy if the cache is
                    # full.  Clear the cache even though we aren't using caching
                    # to prevent download method from choking
                    utils.data.clear_download_cache()
                except RuntimeError:
                    pass
                try:
                    dat = utils.data.download_file(url, cache=False,
                        show_progress=False, timeout=120)
                    shutil.move(dat, filename)
                    message = '\r' + message
                    message += green+' [SUCCESS]'+end+'\n'
                    sys.stdout.write(message.format(image=filename))
                except:
                    message = '\r' + message
                    message += red+' [FAILURE]'+end+'\n'
                    sys.stdout.write(message.format(image=filename))

if __name__ == '__main__':
    # Start timer, create hst123 class obj, parse args
    start = time.time()
    usagestring='hst123.py ra dec'
    hst = hst123()

    # Handle ra/dec in command line so there's no ambiguity about declination
    if '-h' in sys.argv or '--help' in sys.argv:
        parser = hst.add_options(usage=usagestring)
        options = parser.parse_args()
        sys.exit()
    if len(sys.argv) < 3:
        print(usagestring)
        sys.exit(1)
    else:
        ra = sys.argv[1]
        dec = sys.argv[2]
        if (not (hst.is_number(ra) and hst.is_number(dec)) and
           (':' not in ra and ':' not in dec)):
            error = 'ERROR: cannot interpret ra={ra}, dec={dec}.'
            print(error.format(ra=ra, dec=dec))
            sys.exit(1)
        else:
            hst.coord = hst.parse_coord(ra, dec)
            sys.argv[1] = str(hst.coord.ra.degree)
            sys.argv[2] = str(hst.coord.dec.degree)

    parser = hst.add_options(usage=usagestring)
    options = parser.parse_args()

    # Starting banner
    banner = 'Starting hst123.'
    hst.make_banner(banner)

    # If we're cleaning up a previous run, execute that here
    if options.makeclean:
        banner = 'Cleaning output from previous runs of hst123.'
        hst.make_banner(banner)
        hst.make_clean()

    # Handle all other options
    hst.reference = options.reference
    hst.clobber   = options.clobber
    hst.keepshort = options.keepshort
    hst.reffilter = options.reffilter
    hst.object    = options.object
    hst.dolphot   = hst.make_dolphot_dict(options.dolphot)
    if options.alignonly:
        hst.options['global_defaults']['dolphot']['AlignOnly']=1
    if options.before is not None:
        hst.before = parse(options.before)
    if options.after is not None:
        hst.after = parse(options.after)

    # Copy files that don't exist from the raw data directory
    hst.copy_raw_data(reverse=True)

    # If we need to download images, handle that here
    if options.download:
        banner = 'Downloading HST data from MAST for ra={ra}, dec={dec}.'
        ra = '%7.6f' % hst.coord.ra.degree
        dec = '%7.6f' % hst.coord.dec.degree
        hst.make_banner(banner.format(ra=ra, dec=dec))
        hst.download_files()

    # Get input images
    hst.input_images = hst.get_input_images()

    # Make raw/ in current dir and copy files into raw/ and copy un-edited
    # versions of files back into working dir
    banner = 'Copying raw data to raw data folder: {dir}.'
    hst.make_banner(banner.format(dir=hst.raw_dir))
    hst.copy_raw_data()

    # Check which are HST images that need to be reduced
    banner = 'Checking which images need to be reduced by hst123.'
    hst.make_banner(banner)
    for file in list(hst.input_images):
        warning, needs_reduce = hst.needs_to_be_reduced(file)
        if not needs_reduce:
            print(warning)
            hst.input_images.remove(file)
            continue

        filt = hst.get_filter(file).upper()
        if not (filt in hst.options['acceptable_filters']):
            warning = 'WARNING: {img} with FILTER={filt} '
            warning += 'does not have an acceptable filter.'
            print(warning.format(img=file, filt=filt))
            hst.input_images.remove(file)

    # Check there are still images that need to be reduced
    if len(hst.input_images) == 0:
        error = 'ERROR: No input images.  Exiting...'
        print(error)
        sys.exit(1)

    # Get metadata on all input images and put them into an obstable
    banner = 'Organizing input images by visit.'
    hst.make_banner(banner)
    hst.input_list(hst.input_images, show=False)
    hst.add_visit_info(hst.obstable)

    # Update object name if it is passed as an argument
    if hst.object:
        for file in hst.input_images:
            hdu = fits.open(file, mode='update')
            hdu[0].header['TARGNAME'] = hst.object
            hdu[0].header['OBJECT'] = hst.object
            hdu.close()

    # If reference image was not provided then make one
    banner = 'Handling reference image: '
    if not hst.reference:
        banner += 'generating from input files.'
        hst.make_banner(banner)
        hst.pick_reference()
    else:
        banner += '{ref}.'
        hst.make_banner(banner.format(ref=hst.reference))
        # Check to make sure reference image file actually exists
        if not os.path.isfile(hst.reference):
            error = 'ERROR: input reference image {ref} does not exist. '
            error += 'Exiting...'
            print(error.format(ref=hst.reference))
            sys.exit(1)

    # Sanitize extensions and header variables in reference
    banner = 'Sanitizing reference image: {ref}.'
    hst.make_banner(banner.format(ref=hst.reference))
    hst.sanitize_reference(hst.reference)

    # Run main tweakreg to register to the reference
    banner = 'Running main tweakreg.'
    hst.make_banner(banner)
    errorcode = hst.run_tweakreg(hst.input_images, hst.reference)

    # Drizzle all visit/filter pairs if drizzleall
    if options.drizzleall:
        for visit in list(set(hst.obstable['visit'].data)):
            vismask = hst.obstable['visit'] == visit
            visittable = hst.obstable[vismask]
            for filt in list(set(visittable['filter'])):
                filmask = visittable['filter'] == filt
                imagetable = visittable[filmask]

                # Construct a name for the drizzled image
                refimage = imagetable['image'][0]
                inst = hst.get_instrument(refimage).split('_')[0]
                n = str(visit).zfill(4)
                ex_img = imagetable['image'].data[0]
                date_obj = Time(fits.getval(ex_img, 'DATE-OBS'))
                date_str = date_obj.datetime.strftime('%y%m%d')

                # Make a photpipe-like image name
                drizname = ''
                if hst.object:
                    drizname = '{obj}.{inst}.{filt}.ut{date}_{n}.drz.fits'
                    drizname = drizname.format(inst=inst, filt=filt, n=n,
                        date=date_str, obj=hst.object)
                else:
                    drizname = '{inst}.{filt}.ut{date}_{n}.drz.fits'
                    drizname = drizname.format(inst=inst, filt=filt, n=n,
                        date=date_str)

                print(list(imagetable['image'].data))
                hst.run_astrodrizzle(list(imagetable['image'].data),
                    output_name = drizname)
                hst.sanitize_reference(drizname)

                # Make a sky file for the drizzled image and rename 'noise'
                if (hst.needs_to_calc_sky(drizname)):
                    hst.calc_sky(drizname, hst.options['detector_defaults'])
                    sky_image = drizname.replace('.fits', '.sky.fits')
                    noise_name = drizname.replace('.fits', '.noise.fits')
                    shutil.copy(sky_image, noise_name)

    # Sanitize any WFPC2 images
    banner = 'Sanitizing WFPC2 images.'
    hst.make_banner(banner)
    for file in hst.input_images:
        if 'wfpc2' in hst.get_instrument(file):
            hst.sanitize_wfpc2(file)

    # dolphot image preparation: mask_image, split_groups, calc_sky
    message = 'Preparing dolphot data for files={files}.'
    print(message.format(files=','.join(map(str,hst.input_images))))
    for image in hst.input_images:
        # Mask if needs to be masked
        if hst.needs_to_be_masked(image):
            hst.mask_image(image, hst.get_instrument(image).split('_')[0])

        # Split groups if needs split groups
        if hst.needs_to_split_groups(image):
            hst.split_groups(image)

        # Add split images to the list of split images
        split_images = glob.glob(image.replace('.fits','.chip?.fits'))
        if hst.coord:
            for im in list(split_images):
                if not hst.image_contains(im, hst.coord):
                    split_images.remove(im)
        hst.split_images.extend(split_images)

        # Run calc sky on split images if they need calc sky
        for split in split_images:
            if hst.needs_to_calc_sky(split):
                hst.calc_sky(split, hst.options['detector_defaults'])

    # Generate reference image sky file
    if (hst.needs_to_calc_sky(hst.reference)):
        hst.calc_sky(hst.reference, hst.options['detector_defaults'])

    # Write out a list of the input images with metadata for easy reference
    banner = 'Complete list of input images'
    hst.make_banner(banner)
    hst.input_list(hst.input_images)

    # Start constructing dolphot param file from split images and reference
    if not options.dofake:
        banner = 'Adding images to dolphot parameter file: {file}.'
        hst.make_banner(banner.format(file = hst.dolphot['param']))
        dolphot_file = open(hst.dolphot['param'], 'w')
        hst.generate_base_param_file(dolphot_file,
                                 hst.options['global_defaults'],
                                 len(hst.split_images))
        # Write reference image to param file
        hst.add_image_to_param_file(dolphot_file, hst.reference, 0,
                                    hst.options['detector_defaults'])

        # Write out image-specific params to dolphot file
        for i,image in enumerate(hst.split_images):
            hst.add_image_to_param_file(dolphot_file, image, i+1,
                                     hst.options['detector_defaults'])
        dolphot_file.close()
        message = 'Added {n} images to the dolphot parameter file {dp}.'
        print(message.format(n=len(hst.split_images), dp=hst.dolphot['param']))

    # Run dolphot using the param file we constructed and input parameters
    if options.rundolphot:
        if os.path.isfile(hst.dolphot['param']):
            cmd = 'dolphot '+hst.dolphot['base']
            cmd += ' -p'+hst.dolphot['param']
            cmd += ' > '+hst.dolphot['log']
            banner = 'Running dolphot with cmd={cmd}'
            hst.make_banner(banner.format(cmd=cmd))
            os.system(cmd)

            message = 'dolphot is finished (whew)!'
            print(message)
        else:
            error = 'ERROR: dolphot parameter file {file} does not exist!'
            error += ' Generate a parameter file first (no --dofake).'
            print(error.format(file=hst.dolphot['param']))
            sys.exit(1)


    # Scrape data from the dolphot catalog for the input coordinates
    if options.scrapedolphot:
        # Check if dolphot output exists
        if not os.path.isfile(hst.dolphot['base']):
            error = 'ERROR: dolphot output {dp} does not exist.  Run dolphot '
            error += '(using hst123.py --rundp).  Exiting...'
            print(error.format(dp=hst.dolphot['base']))
            sys.exit(1)
        else:
            message = 'Cutting bad sources from dolphot catalog.'
            hst.make_banner(message)
            # Cut bad sources
            f = open('tmp', 'w')
            numlines = sum(1 for line in open(hst.dolphot['base']))
            with open(hst.dolphot['base']) as dolphot_file:
                message = 'There are {n} sources in dolphot file {dp}. '
                message += 'Cutting bad sources...'
                sys.stdout.write(message.format(n=numlines,
                    dp=hst.dolphot['base']))
                width = 77
                sys.stdout.write('\n')
                sys.stdout.write('[%s]' % (' ' * (width-2)))
                sys.stdout.flush()
                sys.stdout.write('\b' * (width-1))
                for i,line in enumerate(dolphot_file):
                    if (i % int(numlines/(width-1)) == 0):
                        sys.stdout.write('-')
                        sys.stdout.flush()
                    if (int(line.split()[10]) == 1): # Obj type
                        f.write(line)
                sys.stdout.write('\n')

            f.close()
            message = 'Done cutting bad sources'
            print(message)
            if filecmp.cmp(hst.dolphot['base'], 'tmp'):
                message = 'No changes to dolphot file {dp}.'
                print(message.format(dp=hst.dolphot['base']))
                os.remove('tmp')
            else:
                message = 'Updating dolphot file {dp}.'
                print(message.format(dp=hst.dolphot['base']))
                shutil.move('tmp', hst.dolphot['base'])

            # Check for reference image and x,y coordinates to scrape data
            if hst.reference is '' or hst.coord is None:
                error = 'ERROR: Need a reference image and coordinate to '
                error = 'scrape data from the dolphot catalog. Exiting...'
                print(error)
                sys.exit(1)
            else:
                # Get x,y coordinate from reference image of potential source
                hdu = fits.open(hst.reference)
                w = wcs.WCS(hdu[0].header)
                x,y = wcs.utils.skycoord_to_pixel(hst.coord, w, origin=1)
                ra = hst.coord.ra.degree
                dec = hst.coord.dec.degree
                radius = hst.dolphot['radius']
                message = 'Looking for a source around x={x}, y={y} in {file} '
                message += ' with a radius of {rad}'
                hst.make_banner(message.format(x=x, y=y, file=hst.reference,
                    rad=radius))

                data = []
                with open(hst.dolphot['base']) as dolphot_file:
                    for line in dolphot_file:
                        xline = float(line.split()[2]) + 0.5
                        yline = float(line.split()[3]) + 0.5
                        dist = np.sqrt((xline-x)**2 + (yline-y)**2)
                        if (dist < radius):
                            data.append([dist, line])
                message = 'Done looking for sources in dolphot file {dp}. '
                message += 'hst123 found {n} sources around ra={ra}, dec={dec}.'
                print(message.format(dp='dp', n=len(data), ra=ra, dec=dec))
                f.close()

                # What to do for n sources?
                best = []
                if len(data) == 0:
                    error = 'ERROR: did not find any candidate sources.'
                    error += ' Exiting...'
                    print(error)
                    sys.exit(1)
                elif len(data) == 1:
                    best = data[0]
                elif len(data) > 1:
                    warning = 'WARNING: found more than one source. '
                    warning += 'Picking object closest to ra={ra}, dec={dec}.'
                    print(warning.format(ra=ra, dec=dec))
                    for obj in data:
                        if not best:
                            best = obj
                        else:
                            if best[0] > obj[0]:
                                best = obj

                # First grab all images from dolphot parameter file then
                # construct an input list
                images = []
                f = open(hst.dolphot['param'], 'r')
                for line in f:
                    if ('_file' in line):
                        if 'img0000' in line:
                            continue
                        else:
                            filename = line.split('=')[1].strip()+'.fits'
                            images.append(filename)
                f.close()

                # Now make an obstable out of images and sort into visits
                hst.input_list(images, show=False)
                hst.add_visit_info(hst.obstable)

                # Now get the final photometry for the source
                colfile = hst.dolphot['colfile']
                hst.final_phot = hst.parse_phot(hst.obstable, best, colfile)

                # Sort the final photometry table by mjd
                hst.final_phot.sort('MJD')

                # Show photometry and write out to file
                message = 'Printing out the final photometry '
                message += 'for ra={ra}, dec={dec}'
                hst.make_banner(message.format(ra=ra, dec=dec))
                phot_file = open(hst.dolphot['final_phot'], 'w')
                hst.show_photometry(hst.final_phot, file=phot_file)
                phot_file.close()

    # Run fake star injection into dolphot. Therefore uses the dolphot parameter
    # and output file in the current directory rather than deriving new dolphot
    # parameters.  This method will therefore prevent generation of a new
    # dolphot parameter file.
    if options.dofake:
        # Need to change global parameters to allow fake star injection/create a
        # fake star injection output file
        dp_param = hst.options['global_defaults']['dolphot']
        dp_param['FakeStars'] = hst.dolphot['fake']
        dp_param['FakeOut'] = hst.dolphot['fake_out']

        # Create the fakestar file given input observations
        images = []
        f = open(hst.dolphot['param'], 'r')
        for line in f:
            if ('_file' in line):
                if 'img0000' in line:
                    continue
                else:
                    filename = line.split('=')[1].strip()+'.fits'
                    images.append(filename)
        f.close()

        hst.input_list(images, show=False)
        hst.add_visit_info(hst.obstable)

        filters = [hst.get_filter(i) for i in hst.obstable['image']]
        wavelength = []
        for i in images:
            try:
                wavelength.append(fits.getval(i,'PHOTPLAM',extname='PRIMARY'))
            except:
                wavelength.append(fits.getval(i,'PHOTPLAM',extname='SCI'))

        # Get the names of the bluest and reddest filters in the dolphot red
        bluest_filter = ''
        bluest_wavelength = 2.0e10
        reddest_filter = ''
        reddest_wavelength = 0.0
        for i,filt in enumerate(filters):
            if (wavelength[i] < bluest_wavelength):
                bluest_filter = filt
                bluest_wavelength = wavelength[i]
            if (wavelength[i] > reddest_wavelength):
                reddest_filter = filt
                reddest_wavelength = wavelength[i]

        hst.options['global_defaults']['fake']['filter1'] = bluest_filter
        hst.options['global_defaults']['fake']['filter2'] = reddest_filter

        cmd = 'fakelist {dp} {filter1} {filter2} {filter1_min} {filter1_max} '
        cmd += '{color_min} {color_max} -nstar={nstar} > {fake_out}'

        os.system(cmd.format(fake_out=hst.dolphot['fake_out'],
            **hst.options['global_defaults']['fake']))

    # Clean up interstitial files in working directory
    if not options.nocleanup:
        for file in hst.input_images:
            os.remove(file)

    message = 'It took {time} seconds to complete this script.'
    hst.make_banner(message.format(time=time.time()-start))
