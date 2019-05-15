#!/usr/bin/env python
# CDK v1.00: 2019-02-07. Base hst123 download, tweakreg, drizzle, dolphot param
# CDK v1.01: 2019-02-15. Added running dolphot, scraping dolphot output
# CDK v1.02: 2019-02-22. Added fake star injection
#
# hst123.py: An all-in-one script for downloading, registering, drizzling,
# running dolphot, and scraping data from dolphot catalogs.
#
# Python 2/3 compatibility
from __future__ import print_function

try:
    input = raw_input # use 'input' function in both Python 2 and 3
except NameError:
    pass

# Dependencies and settings
import glob, sys, os, shutil, time, subprocess, warnings, filecmp
import astroquery
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import astropy.wcs as wcs
from stwcs import updatewcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import utils
from astropy.table import Table
from astropy.time import Time
from astroquery.mast import Observations
from drizzlepac import tweakreg,astrodrizzle
from astroscrappy import detect_cosmics
warnings.filterwarnings('ignore')

# Color strings for download messages
green = '\033[1;32;40m'
red = '\033[1;31;40m'
end = '\033[0;0m'

# Dictionaries with all options for instruments, detectors, and filters
global_defaults = {
    'change_keys': ['IDCTAB','DGEOFILE','NPOLEXT','NPOLFILE','D2IMFILE',
                    'D2IMEXT','OFFTAB'],
    'cdbs_ftp': 'ftp://ftp.stsci.edu/cdbs/',
    'mast_radius': 5 * u.arcmin,
    'mast_uri': 'https://mast.stsci.edu/api/v0/download/file?uri=',
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
    'wfc3': {'env_ref': 'iref',
             'crpars': {'rdnoise': 6.5,
                        'gain': 1.0,
                        'saturation': 70000.0,
                        'sig_clip': 4.0,
                        'sig_frac': 0.2,
                        'obj_lim': 6.0}},
    'acs': {'env_ref': 'jref',
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
                'dolphot': {'apsky': '15 25', 'RAper': 3, 'RChi': 2,
                            'RPSF': 13, 'RSky': '15 35',
                            'RSky2': '4 10'}},
    'acs_hrc': {'driz_bits': 0, 'nx': 5200, 'ny': 5200,
                'input_files': '*_flt.fits', 'pixel_scale': 0.05,
                'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 10, 'RSky': '15 35',
                            'RSky2': '3 6'}},
    'wfpc2_wfpc2': {'driz_bits': 0, 'nx': 5200, 'ny': 5200,
                    'input_files': '*_c0m.fits', 'pixel_scale': 0.09,
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

    self.keepshort = False
    self.before = None
    self.after = None

    self.coord = None
    self.redo = False
    self.download = False
    self.clobber = False
    self.nocleanup = False

    self.max_dolphot_images = 9999999999
    self.run_dolphot = False
    self.scrape_dolphot = False
    self.nocuts = False
    self.do_fake = False
    self.dolphot = {}

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
                    'acceptable_filters': acceptable_filters}

    # List of pipeline products in case they need to be cleaned at start
    self.pipeline_products = ['*chip?.fits', '*chip?.sky.fits',
                              '*rawtmp.fits', '*drz.fits', '*drz.sky.fits',
                              '*idc.fits', '*dxy.fits', '*off.fits',
                              '*d2im.fits', '*d2i.fits', '*npl.fits',
                              'dp*', '*.log', '*.output','*sci?.fits',
                              '*wht.fits','*sci.fits','*StaticMask.fits']

  def add_options(self, parser=None, usage=None):
    import optparse
    if parser == None:
        parser = optparse.OptionParser(usage=usage,
            conflict_handler='resolve')
    parser.add_option('--redo', default=False, action='store_true',
        help='Redo the hst123 reduction by re-copying files from the raw dir.')
    parser.add_option('--makeclean','--mc', default=False, action='store_true',
        help='Clean up all output files from previous runs then exit.')
    parser.add_option('--download', default=False, action='store_true',
        help='Download the raw data files given input ra and dec.')
    parser.add_option('--keepshort','--ks', default=False, action='store_true',
        help='Keep image files that are shorter than 20 seconds.')
    parser.add_option('--before', default=None, type='string',
        metavar='YYYY-MM-DD', help='Date after which we should reject all '+\
        'HST observations for reduction.')
    parser.add_option('--after', default=None, type='string',
        metavar='YYYY-MM-DD', help='Date before which we should reject all '+\
        'HST observations for reduction.')
    parser.add_option('--clobber', default=False, action='store_true',
        help='Overwrite files when using download mode.')
    parser.add_option('--ra', default=None, metavar='deg/HH:MM:SS',
        type='string', help='RA of interest.')
    parser.add_option('--dec', default=None, metavar='deg/DD:MM:SS',
        type='string', help='DEC of interest.')
    parser.add_option('--reference','--ref', default='', metavar='ref.fits',
        type='string', help='Name of the reference image.')
    # WARNING: --rootdir is not yet implemented.  hst123 will always run in the
    # current directory
    #parser.add_option('--root','--rootdir', default=None,
    #    type='string', help='Directory where hst123.py should run/pathway '+\
    #    'to input files.')
    parser.add_option('--rundolphot','--rd', default=False, action='store_true',
        help='Run dolphot as part of this hst123 run.')
    parser.add_option('--maxdolphot','--mdp','--maxdp', default=9999999,
        type='int', metavar=9999999,
        help='Maximum number of images per dolphot run.')
    parser.add_option('--alignonly','--ao', default=False, action='store_true',
        help='When running dolphot, set AlignOnly=1 so dolphot stops after '+\
        'image alignment.')
    parser.add_option('--dolphot','--dp', default='dp', type='string',
        metavar='dp', help='Name of the dolphot output file.')
    parser.add_option('--nocuts','--nc', default=False, action='store_true',
        help='Do not mask bad sources from the dolphot output catalog before '+\
        'scraping data.')
    parser.add_option('--scrapedolphot','--sd', default=False,
        action='store_true', help='Scrape photometry from the dolphot '+\
        'catalog from the input RA/Dec.')
    parser.add_option('--dofake','--df', default=False,
        action='store_true', help='Run fake star injection into dolphot. '+\
        'Requires that dolphot has been run, and so files are taken from the '+\
        'parameters in dolphot output from the current directory rather than '+\
        'files derived from the current run.')
    parser.add_option('--nocleanup','--nc', default=False, action='store_true',
        help='Dont clean up interstitial image files (i.e., flt,flc,c1m,c0m).')
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

  def avg_magnitudes(self, mags, magerrs, counts, exptimes):
    # First create a mask and apply to input values
    idx = []
    for i,data in enumerate(zip(mags,magerrs,counts,exptimes)):
        if data[1] < 1.0 and data[2] > 0:
            idx.append(i)

    if not idx:
        return (float('NaN'),float('NaN'))

    mags = np.array(mags)[idx]
    magerrs = np.array(magerrs)[idx]
    counts = np.array(counts)[idx]
    exptimes = np.array(exptimes)[idx]

    # First calculate zero points for these fluxes
    flux = counts / exptimes
    zps = mags + 2.5 * np.log10(flux)

    # Now normalize flux and fluxerr to a common zero point
    flux = flux * 10**(0.4 * (27.5 - zps))
    fluxerr = 1./1.086 * magerrs * flux

    # Now calculate average and propagate uncertainties weighted by fluxerr
    average_flux = np.sum(flux*1/fluxerr**2)/np.sum(1/fluxerr**2)
    average_fluxerr = np.sqrt(np.sum(fluxerr**2))/len(fluxerr)

    # Now transform back to magnitude and magnitude error
    final_mag = 27.5 - 2.5*np.log10(average_flux)
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
       sys.exit(3)
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
                               filt=row['FILTER'], exp='%7.3f' % row['EXPTIME'],
                               mag='%3.3f' % row['MAGNITUDE'],
                               err='%3.3f' % row['MAGNITUDE_ERROR'])
            print(line)
            if file is not None:
                file.write(line+'\n')

  def input_list(self, images, show=True):
    # Make a table with all of the metadata for each image.
    exptime = [fits.getval(image,'EXPTIME') for image in images]
    datetim = [fits.getval(image,'DATE-OBS') + 'T' +
               fits.getval(image,'TIME-OBS') for image in images]
    filters = [self.get_filter(image) for image in images]
    instrum = [self.get_instrument(image) for image in images]

    # Save this obstable.  Useful for other methods
    self.obstable = Table([images,exptime,datetim,filters,instrum],
                   names=('image','exptime','datetime','filter','instrument'))
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
            os.chmod(self.raw_dir, 0775)
        for f in self.input_images:
            if not os.path.isfile(self.raw_dir+f):
                # Create new file and change permissions
                shutil.copyfile(f, self.raw_dir+f)
                os.chmod(self.raw_dir+f, 0775)
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
            os.chmod(base, 0775)

  # For an input obstable, sort all files into instrument, visit, and filter
  # so we can group them together for the final output from dolphot
  def add_visit_info(self, obstable):
    # First add empty 'visit' column to obstable
    obstable['visit'] = np.zeros(len(obstable))

    # Sort obstable by date so we assign visits in chronological order
    obstable.sort('datetime')

    # Time tolerance for defining a 'visit'.  How many days apart were the obs
    # separated
    tol = 1

    # Iterate through each file in the obstable
    for row in obstable:
        inst = row['instrument'].upper()
        mjd = Time(row['datetime']).mjd
        filt = row['filter'].upper()
        filename = row['image']

        # If this is the first one we're making, assign it to visit 1
        if all([obs['visit'] == 0 for obs in obstable]):
            row['visit'] = 1
        else:
            instmask = obstable['instrument'] == inst
            timemask = [abs(Time(obs['datetime']).mjd - mjd) < tol
                            for obs in obstable]
            filtmask = obstable['filt'] == filt
            mask = [all(l) for l in zip(instmask, timemask, filtmask)]

            # If no matches, then we need to define a new visit
            if not any(mask):
                # no matches. create new visit number = max(visit)+1
                row['visit'] = np.max(obstable['visit']) + 1
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
        instname = dictionary[inst]['NAME']
        for visit in list(set(insttable['visit'])):
            visittable = insttable[insttable['visit'] == visit]
            for filt in list(set(visittable['filter'])):
                filttable = visittable[visittable['filter'] == filt]
                mjds, mags, magerrs, counts, exptimes = ([] for i in range(5))
                for im in filttable['image']:
                    mjds = [Time(str(fits.getval(im,'DATE-OBS')) + 'T' +
                             str(fits.getval(im,'TIME-OBS'))).mjd]
                    mags = [float(self.get_dolphot_data(row[1], 'dp.columns',
                            'Instrumental', im))]
                    magerrs = [float(self.get_dolphot_data(row[1], 'dp.columns',
                               'Magnitude uncertainty', im))]
                    counts = [float(self.get_dolphot_data(row[1], 'dp.columns',
                              'Measured counts', im))]
                    exptimes = [float(fits.getval(im, 'EXPTIME'))]

                # Calculate average of mjds, total of exptimes
                avg_mjd = np.mean(mjds)
                total_exptime = np.sum(exptimes)

                # Average magnitude and magnitude errors
                mag, magerr = self.avg_magnitudes(mags, magerrs,
                    counts, exptimes)

                final_phot.add_row((avg_mjd, instname, filt,
                                    total_exptime, mag, magerr))
    return(final_phot)

  # Sanitizes reference header, gets rid of multiple extensions and only
  # preserves science data.
  def sanitize_reference(self, reference):
    hdu = fits.open(reference, mode='readonly')

    # Going to write out newhdu
    # Only want science extension from orig reference
    newhdu = fits.HDUList()
    newhdu.append(hdu['SCI'])

    # Want to preserve header info, so combine SCI+PRIMARY headers
    for key in hdu['PRIMARY'].header.keys():
        if (key not in newhdu[0].header.keys()+['COMMENT','HISTORY']):
            newhdu[0].header[key] = hdu['PRIMARY'].header[key]

    # Make sure that reference header reflects one extension
    newhdu[0].header['EXTEND']=False

    # Add header variables that dolphot needs: GAIN, RDNOISE, SATURATE
    inst = newhdu[0].header['INSTRUME'].lower()
    opt  = self.options['instrument_defaults'][inst]['crpars']
    newhdu[0].header['SATURATE'] = opt['saturation']
    newhdu[0].header['RDNOISE']  = opt['rdnoise']
    newhdu[0].header['GAIN']     = opt['gain']

    if 'WHT' in [h.name for h in hdu]:
        wght = hdu['WHT'].data
        newhdu['SCI'].data[np.where(wght == 0)] = float('NaN')

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
    if not self.keepshort:
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

  def needs_to_calc_sky(self,image):
    files = glob.glob(image.replace('.fits','.sky.fits'))
    if (len(files) == 0):
        if self.coord:
            return(self.image_contains(image, self.coord))
        else:
            return(True)

  # Quick script to determine if the input image contains input coordinate.
  # This routine should only be run on chip?.fits files.
  def image_contains(self, image, coord):
    # sky2xy is way better/faster than astropy.wcs. Also, astropy.wcs has a
    # problem with chip? files generated by splitgroups because the WCSDVARR
    # gets split off from the data, which is required by WCS (see fobj).
    ra = coord.ra.degree
    dec = coord.dec.degree
    cmd = 'sky2xy {image} {ra} {dec}'.format(image=image, ra=ra, dec=dec)
    result = subprocess.check_output(cmd, shell=True)
    if ('off image' in result):
        message = '{image} does not contain ra={ra}, dec={dec}'
        print(message.format(image=image, ra=ra, dec=dec))
        return(False)
    else:
        return(True)

  # Determine if we need to run the dolphot mask routine
  def needs_to_be_masked(self,image):
    # Masking should remove all of the DQ arrays etc, so make sure that any
    # extensions with data in in them are only SCI extensions. This might not be
    # 100% robust, but should be good enough.
    hdulist = fits.open(image)
    needs_masked = False
    for hdu in hdulist:
        if hdu.data is not None and 'EXTNAME' in hdu.header:
            if hdu.header['EXTNAME'].upper() != 'SCI':
                needs_masked = True
    return(needs_masked)

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
  def get_instrument(self,image):
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
    for file in glob.glob(image.replace('.fits', '.chip?.fits')):
        os.chmod(file, 0775)

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
    for file in glob.glob(image.replace('.fits','.sky.fits')):
        os.chmod(file, 0775)


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
        output_name = best_inst+'_'+best_filt+'_drz.fits'
        message = 'Reference image name will be: {reference}. '
        message += 'Generating from input files {files}.'
        print(message.format(reference=output_name, files=reference_images))
        self.reference = output_name

        self.run_tweakreg(reference_images,'',self.options['global_defaults'])
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
    change_keys = self.options['global_defaults']['change_keys']
    for image in images:
        inst = self.get_instrument(image).split('_')[0]
        ref = self.options['instrument_defaults'][inst]['env_ref']
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
            if (ref+'$' in val):
                ref_file = val.split('$')[1]
                if not os.path.exists(ref_file):
                    url = self.options['global_defaults']['cdbs_ftp']
                    url += ref+'/'+ref_file
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
                        os.chmod(ref_file, 0775)
                        message = '\r' + message
                        message += green+' [SUCCESS]'+end+'\n'
                        sys.stdout.write(message.format(url=url))
                    except:
                        message = '\r' + message
                        message += red+' [FAILURE]'+end+'\n'
                        sys.stdout.write(message.format(url=url))
                        print(message.format(url=url))
                fits.setval(image, key, extname='PRIMARY', value=ref_file)
                fits.setval(image, key, extname='SCI', value=ref_file)
        updatewcs.updatewcs(image)

    ra = self.coord.ra.degree if self.coord else None
    dec = self.coord.dec.degree if self.coord else None

    if self.keepshort:
        skysub = False
    else:
        skysub = True

    astrodrizzle.AstroDrizzle(images, output=output_name, runfile='',
                wcskey=wcskey, context=True, group='', build=True,
                num_cores=8, preserve=False, clean=True, skysub=skysub,
                skystat='mode', skylower=0.0, skyupper=None, updatewcs=True,
                driz_sep_fillval=-100000, driz_sep_bits=0, driz_sep_wcs=True,
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

    # Change permissions on drizzled file
    os.chmod(output_name, 0775)

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
  def run_tweakreg(self, images, reference, options):

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
        os.chmod(rawtmp, 0775)

        # Clean cosmic rays on the image in place. We do this so we don't
        # accidentally pick cosmic rays for alignment
        inst = self.get_instrument(image).split('_')[0]
        crpars = self.options['instrument_defaults'][inst]['crpars']
        self.run_cosmic(image,crpars)

    if (reference == '' or reference == None):
        reference = 'dummy.fits'
        shutil.copyfile(images[0], reference)
        os.chmod(reference, 0775)

    message = 'Executing tweakreg with images: {images} \n'
    message += 'Reference image: {reference} \n'
    message += 'Tweakreg is executing...'
    print(message.format(images = ','.join(map(str,run_images)),
                         reference = reference))
    start_tweak = time.time()
    tweakreg.TweakReg(files=run_images, refimage=reference, verbose=False,
            interactive=False, clean=True, writecat = False, updatehdr=True,
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
    # Define the search coord/radius and grab all files from MAST that
    # correspond to this region
    search_radius = self.options['global_defaults']['mast_radius']
    try:
        obsTable = Observations.query_region(self.coord, radius=search_radius)
    except astroquery.exceptions.RemoteServiceError:
        error = 'ERROR: MAST is not working for some reason. '
        error += 'Try again later.  Exiting...'
        print(error)
        sys.exit(2)


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

                url = self.options['global_defaults']['mast_uri'] + uri
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
                    os.chmod(filename, 0775)
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
    usagestring='USAGE: hst123.py'
    hst = hst123()
    parser = hst.add_options(usage=usagestring)
    options, args = parser.parse_args()

    # Starting banner
    banner = 'Starting hst123.py'
    hst.make_banner(banner)

    # If we're cleaning up a previous run, execute that here
    if (options.makeclean):
        question = 'Are you sure you want to delete output from previous hst123'
        question += ' runs? [y/n] '
        var = input(question)
        if var != 'y' and var != 'yes':
            warning = 'WARNING: input={inp}. Exiting...'
            print(warning)
            sys.exit(3)
        banner = 'Cleaning output from previous runs of hst123.'
        hst.make_banner(banner)
        for pattern in hst.pipeline_products:
            for file in glob.glob(pattern):
                if os.path.isfile(file):
                    os.remove(file)
        hst.copy_raw_data(reverse = True)
        sys.exit(0)

    # Handle all other options
    hst.reference = options.reference
    hst.download = options.download
    hst.clobber = options.clobber
    hst.redo = options.redo
    hst.keepshort = options.keepshort
    hst.max_dolphot_images = options.maxdolphot
    hst.run_dolphot = options.rundolphot
    hst.scrape_dolphot = options.scrapedolphot
    hst.nocuts = options.nocuts
    hst.do_fake = options.dofake
    hst.dolphot = hst.make_dolphot_dict(options.dolphot)
    if options.alignonly:
        hst.options['global_defaults']['dolphot']['AlignOnly']=1
    if (options.ra is not None and options.dec is not None):
        hst.coord = hst.parse_coord(options.ra, options.dec)
    if options.before is not None:
        hst.before = parse(options.before)
    if options.after is not None:
        hst.after = parse(options.after)

    # If we're re-doing the reduction, copy files that don't exist from the raw
    # data directory
    hst.copy_raw_data(reverse=True)

    # If we need to download images, handle that here
    if options.download:
        if not hst.coord:
            error = 'ERROR: hst123.py --download requires input ra and dec.'
            error += 'Exiting...'
            print(error)
            sys.exit(2)
        else:
            banner = 'Downloading HST data from MAST for ra={ra}, dec={dec}.'
            hst.make_banner(banner.format(ra=hst.coord.ra.degree,
                                          dec=hst.coord.dec.degree))
            hst.download_files()

    # Get input images
    hst.input_images = hst.get_input_images()

    # Make raw/ in current dir and copy files into raw/ then copy un-edited
    # versions of files back into working dir
    banner = 'Copying raw data to raw data folder: {dir}.'
    hst.make_banner(banner.format(dir=hst.raw_dir))
    hst.copy_raw_data()

    # Check which are HST images that need to be reduced
    banner = 'Checking which images need to be reduced by hst123.'
    hst.make_banner(banner)
    for file in list(hst.input_images):
        warning, needs_reduce = hst.needs_to_be_reduced(file)
        if not (needs_reduce):
            print(warning)
            hst.input_images.remove(file)

            # WARNING: these lines delete the actual data and raw data file
            if 'c1m.fits' not in file:
                os.remove(file)
                os.remove(hst.raw_dir+'/'+file)

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
    errorcode = hst.run_tweakreg(hst.input_images,hst.reference,
                     hst.options['global_defaults'])

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

    # Check if the total number of images exceeds the maximum number of dolphot
    # images per run (--maxdolphot).  If so, cut up the list of input images
    # into separate dolphot param files with Nimg = maxdolphot.  This is mostly
    # so you don't kill your computer's RAM running dolphot.  Technically, this
    # also saves time on big (>>a few) dolphot runs.
    if len(hst.split_images) > hst.max_dolphot_images:
        # TODO: need to implement this routine and separate dolphot runs and
        # scraping data from all of the separate dolphot runs
        message = 'WARNING: total number of input images for dolphot={Nimg} and'
        message += ' max dolphot images per dolphot run={Nmax}.  Chopping '
        message += 'images into separate dolphot parameter files.'
        print(message.format(Nimg=len(hst.split_images),
                             Nmax=hst.max_dolphot_images))

    # Start constructing dolphot param file from split images and reference
    if not hst.do_fake:
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
    if hst.run_dolphot:
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
            sys.exit(2)


    # Scrape data from the dolphot catalog for the input coordinates
    if hst.scrape_dolphot:
        # Check if dolphot output exists
        if not os.path.isfile(hst.dolphot['base']):
            error = 'ERROR: dolphot output {dp} does not exist.  Run dolphot '
            error += '(using hst123.py --rundp).  Exiting...'
            print(error.format(dp=hst.dolphot['base']))
            sys.exit(2)
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
                width = 80
                sys.stdout.write('\n')
                sys.stdout.write('[%s]' % (' ' * (width-2)))
                sys.stdout.flush()
                sys.stdout.write('\b' * (width-1))
                for i,line in enumerate(dolphot_file):
                    if (i % int(numlines/(width-1)) == 0):
                        sys.stdout.write('-')
                        sys.stdout.flush()
                    if (int(line.split()[10]) == 1):# and         # Obj type
                        #abs(float(line.split()[6])) < 0.3 and  # Sharpness
                        #float(line.split()[9]) < 0.5):         # Crowding
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
                    sys.exit(3)
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
    if hst.do_fake:
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

        # Re-run dolphot with new parameter file
        #cmd = 'dolphot '+hst.dolphot['base']
        #cmd += ' -p'+hst.dolphot['param']
        #cmd += ' > '+hst.dolphot['log']
        #banner = 'Running dolphot with cmd={cmd}'
        #hst.make_banner(banner.format(cmd=cmd))
        #os.system(cmd)

    # Clean up interstitial files in working directory
    if not hst.nocleanup:
        for file in hst.input_images:
            os.remove(file)

    message = 'It took {time} seconds to complete this script.'
    hst.make_banner(message.format(time=time.time()-start))
