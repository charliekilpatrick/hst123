#!/usr/bin/env python
# CDK v1.00: 2019-02-07
#
# hst123.py: An all-in-one pipeline for downloading,
# registering, drizzling, and preparing HST data for
# dolphot reductions
#
# Python 2/3 compatibility
from __future__ import print_function # to use print() as a function in Python 2

# Dependencies and settings
import glob, sys, os, shutil, time, io, urllib, subprocess, contextlib, warnings, filecmp
from astropy.io import fits
from stwcs import updatewcs
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astroquery.mast import Observations
from astropy import utils
from drizzlepac import tweakreg,astrodrizzle
from contextlib import contextmanager
from astroscrappy import detect_cosmics
import numpy as np

# Suppresses warnings
warnings.filterwarnings('ignore')

# Suppresses output when loading iraf
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

global_defaults = {'template_image': None,
                   'ra': None,
                   'dec': None,
                   'rotation': 0.0,
                   'pixel_fraction': 1.0,
                   'clean': True,
                   'num_cores': 8,
                   'sky_subtract': True,
                   'use_tweakshifts': True,
                   'tweakshifts_threshold': 5,
                   'ftp_uref_server': 'ftp://ftp.stsci.edu/cdbs/uref/',
                   'ftp_iref_server': 'ftp://ftp.stsci.edu/cdbs/iref/',
                   'ftp_jref_server': 'ftp://ftp.stsci.edu/cdbs/jref/',
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
                               'AlignIter': 3,
                               'AlignTol': 0.5,
                               'AlignStep': 0.2,
                               'VerboseData': 1,
                               'NegSky': 0,
                               'Force1': 0,
                               'DiagPlotType': 'PNG',
                               'InterpPSFlib': 1,
                               'AlignOnly': 1}}

instrument_defaults = {'wfc3': {'env_ref': 'iref',
                                'ftp_server': 'ftp://ftp.stsci.edu/cdbs/iref/',
                                'driz_bits': 0,
                                'pixel_scale': 0.04,
                                'crpars': {'rdnoise': 6.5,
                                           'gain': 1.0,
                                           'saturation': 70000.0,
                                           'sig_clip': 4.0,
                                           'sig_frac': 0.2,
                                           'obj_lim': 6.0}},
                       'acs': {'env_ref': 'jref',
                               'ftp_server': 'ftp://ftp.stsci.edu/cdbs/jref/',
                               'driz_bits': 0,
                               'pixel_scale': 0.05,
                               'crpars': {'rdnoise': 6.5,
                                          'gain': 1.0,
                                          'saturation': 70000.0,
                                          'sig_clip': 3.0,
                                          'sig_frac': 0.1,
                                          'obj_lim': 5.0}},
                       'wfpc2': {'env_ref': 'uref',
                                 'ftp_server': 'ftp://ftp.stsci.edu/cdbs/uref/',
                                 'driz_bits': 0,
                                 'pixel_scale': 0.046,
                                 'crpars': {'rdnoise': 10.0,
                                            'gain': 7.0,
                                            'saturation': 27000.0,
                                            'sig_clip': 4.0,
                                            'sig_frac': 0.3,
                                            'obj_lim': 6.0}
                                 }}


detector_defaults = {'wfc3_uvis': {'nx': 5200, 'ny': 5200, 'input_files': '*_flc.fits',
                                   'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                                   'sigma_low': 2.25, 'sigma_high': 2.00},
                                   'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                                               'RPSF': 10, 'RSky': '15 35',
                                               'RSky2': '3 6'}},
                     'wfc3_ir': {'driz_bits': 512, 'nx': 5200, 'input_files': '*_flt.fits',
                                  'ny': 5200, 'pixel_scale': 0.09,
                                 'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                                 'sigma_low': 2.25, 'sigma_high': 2.00},
                                 'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                                             'RPSF': 10, 'RSky': '8 20',
                                             'RSky2': '3 6'}},
                     'acs_wfc': {'nx': 5200, 'ny': 5200, 'input_files': '*_flc.fits',
                                 'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                                 'sigma_low': 2.25, 'sigma_high': 2.00},
                                 'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                                             'RPSF': 10, 'RSky': '15 35',
                                             'RSky2': '3 6'}},
                    'acs_hrc': {'nx': 5200, 'ny': 5200, 'input_files': '*_flt.fits',
                                 'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                                 'sigma_low': 2.25, 'sigma_high': 2.00},
                                 'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                                             'RPSF': 10, 'RSky': '15 35',
                                             'RSky2': '3 6'}},
                     'wfpc2_wfpc2': {'nx': 5200, 'ny': 5200, 'input_files': '*_c0m.fits',
                                     'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                                     'sigma_low': 2.25, 'sigma_high': 2.00},
                                     'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                                                 'RPSF': 10, 'RSky': '15 35',
                                                 'RSky2': '3 6'}}}

subarray_defaults = {'wfc3_uvis_full': {},
                     'wfc3_uvis_sub': {'nx': 1400, 'ny': 1400},
                     # Note that wfc3_ir does not ever do cosmic ray removal
                     # because it automatically flags cosmic rays using up the ramp sampling
                     'wfc3_ir_full': {},

                     'acs_wfc_full': {},
                     'acs_wfc_sub': {'nx': 1400, 'ny': 1400, 'input_files': '*_flt.fits'},

                     'wfpc2_wfpc2_full': {}}

acceptable_filters = {'F220W','F250W','F330W','F344N','F435W','F475W','F502N',
                      'F550M','F555W','F606W','F625W','F658N','F660N','F660N',
                      'F775W','F814W','F850LP','F892N','F098M','F105W','F110W',
                      'F125W','F126N','F127M','F128N','F130N','F132N','F139M',
                      'F140W','F153M','F160W','F164N','F167N','F200LP','F218W',
                      'F225W','F275W','F280N','F300X','F336W','F343N','F350LP',
                      'F373N','F390M','F390W','F395N','F410M','F438W','F467M',
                      'F469N','F475X','F487N','F502N','F547M','F600LP','F621M',
                      'F625W','F631N','F645N','F656N','F657N','F658N','F665N',
                      'F673N','F680N','F689M','F763M','F845M','F953N','F122M',
                      'F160BW','F185W','F218W','F255W','F300W','F375N','F380W',
                      'F390N','F437N','F439W','F450W','F569W','F588N','F622W',
                      'F631N','F673N','F675W','F702W','F785LP','F791W','F953N',
                      'F1042M'}

class hst123(object):

  def __init__(self):

      # Basic parameters
      self.total_images = 0
      self.total_split_images = 0
      self.input_images = []
      self.split_images = []
      self.filt_list = []
      self.inst_list = []

      self.options = {'global_defaults': global_defaults,
                      'detector_defaults': detector_defaults,
                      'acceptable_filters': acceptable_filters,
                      'instrument_defaults': instrument_defaults,
                      'subarray_defaults': subarray_defaults}

      self.ra = None
      self.dec = None

      self.download = False
      self.clobber = False

      self.cen_coord = None
      self.ref_coord = None

      self.good_detector = ['WFPC2/WFC','PC/WFC',
                            'ACS/WFC','ACS/HRC',
                            'ACS/SBC','WFC3/UVIS',
                            'WFC3/IR']

      self.download_uri = 'https://mast.stsci.edu/api/v0/download/file?uri='

      self.reference_image_name = ''
      self.global_object_name = ''

      self.dolphot_param_file = 'dp.param'
      self.raw_dir = 'raw/'
      self.input_root_dir = ''

      self.is_reference_hst = True

  def add_options(self, parser=None, usage=None):
        import optparse
        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler='resolve')
        parser.add_option('--ra', default=None, type='string', help='RA of interest')
        parser.add_option('--dec', default=None, type='string', help='DEC of interest')
        parser.add_option('--objname','--obj', default='test', type='string',
                          help='Object name to use for dolphot/output files.  Default=test.')
        parser.add_option('--reference','--ref', default='', type='string',
                          help='Name of the reference image.')
        parser.add_option('--inputrootdir','--input', default=None, type='string',
                          help='Pathway to input HST files.')
        parser.add_option('--makeclean', default=False, action='store_true',
                          help='Clean up all output files from previous runs then exit.')
        parser.add_option('--download', default=False, action='store_true',
                          help='Download the raw data files given input ra and dec.')
        parser.add_option('--clobber', default=False, action='store_true',
                          help='Overwrite files when using download mode.')
        return(parser)

  def copy_raw_data(self):
    if not os.path.exists(self.raw_dir):
      os.mkdir(self.raw_dir)
    for f in self.input_images:
      if not os.path.isfile(self.raw_dir+f):
        shutil.copy(f, self.raw_dir)

  # Sanitizes template header, gets rid of
  # multiple extensions and only preserves
  # science data.
  def sanitize_template(self, template):
    hdu = fits.open(template)

    # Going to write out newhdu
    # Only want science extension from orig template
    newhdu = fits.HDUList()
    newhdu.append(hdu['SCI'])

    # Want to preserve header info, so combine
    # SCI+PRIMARY headers (except COMMENT/HISTORY keys)
    for key in hdu['PRIMARY'].header.keys():
      if (key not in newhdu[0].header.keys()
        and key != 'COMMENT'
        and key != 'HISTORY'):
          newhdu[0].header[key] = hdu['PRIMARY'].header[key]

    # Make sure that template header reflects one extension
    newhdu[0].header['EXTEND']=False

    # Add header variables that dolphot needs:
    # GAIN, RDNOISE, SATURATE
    inst = newhdu[0].header['INSTRUME'].lower()
    inst_options = self.options['instrument_defaults'][inst]['crpars']
    newhdu[0].header['SATURATE'] = inst_options['saturation']
    newhdu[0].header['RDNOISE'] = inst_options['rdnoise']
    newhdu[0].header['GAIN'] = inst_options['gain']

    # Write out to same file w/ overwrite
    newhdu.writeto(template, output_verify='silentfix', overwrite=True)

  # Sanitizes wfpc2 data by getting rid of
  # excess extensions and changing header.
  def sanitize_wfpc2(self, image):
    hdu = fits.open(image)

    # Going to write out newhdu
    # Only want science extension from orig template
    newhdu = fits.HDUList()
    newhdu.append(hdu['PRIMARY'])

    # change number of extensions to number
    # of science extensions in wfpc2 image
    n = len([h.name for h in hdu if h.name == 'SCI'])
    newhdu['PRIMARY'].header['NEXTEND'] = n

    # Now append science extensions to newhdu
    for h in hdu:
      if h.name == 'SCI':
        newhdu.append(h)

    # Write out to same file w/ overwrite
    newhdu.writeto(image, output_verify='silentfix', overwrite=True)

  def parse_center_coord(self):
    if (not self.ra or not self.dec):
      print('No input RA or DEC!')
      sys.exit()

    if (':' in self.ra and ':' in self.dec):
      # Input RA/DEC are sexagesimal
      self.cen_coord = SkyCoord(self.ra,self.dec,frame='ircs')
    else:
      # Assume input coordiantes are decimal degrees
      self.cen_coord = SkyCoord(self.ra,self.dec,frame='ircs',unit='deg')

  def needs_to_be_reduced(self,image):
    hdu = fits.open(image, mode='readonly')
    is_not_hst_image = False
    detector = ''
    instrument = hdu[0].header['INSTRUME'].lower()
    if ('DETECTOR' in hdu[0].header.keys()):
      detector = hdu[0].header['DETECTOR'].lower()

    # Get rid of exposures with exptime < 20s
    exptime = hdu[0].header['EXPTIME']
    if (exptime < 20):
      return False

    # Get rid of data where the input coordinates do
    # not land in any of the sub-images
    if (self.ra is not None and self.dec is not None):
      for h in hdu:
        if h.data is not None and 'EXTNAME' in h.header:
          if h.header['EXTNAME'] == 'SCI':
            w = WCS(h.header,hdu)
            world_coord = np.array([[float(self.ra),float(self.dec)]])
            pixcrd = w.wcs_world2pix(world_coord,1)
            if (pixcrd[0][0] > 0 and
                pixcrd[0][1] > 0 and
                pixcrd[0][0] < h.header['NAXIS1'] and
                pixcrd[0][1] < h.header['NAXIS2']):
                is_not_hst_image = True
      if not is_not_hst_image:
        return False

    # Get rid of images that don't match one of the
    # allowed instrument/detector types and images
    # whose extensions don't match the allowed type
    # for those instrument/detector types
    is_not_hst_image = False
    if (instrument.upper() == 'WFPC2' and 'c0m.fits' in image):
        is_not_hst_image = True
    if (instrument.upper() == 'ACS' and detector.upper() == 'WFC' and 'flc.fits' in image):
        is_not_hst_image = True
    if (instrument.upper() == 'ACS' and detector.upper() == 'HRC' and 'flt.fits' in image):
        is_not_hst_image = True
    if (instrument.upper() == 'WFC3' and detector.upper() == 'UVIS' and 'flc.fits' in image):
        is_not_hst_image = True
    if (instrument.upper() == 'WFC3' and detector.upper() == 'IR' and 'flt.fits' in image):
        is_not_hst_image = True
    return is_not_hst_image

  def needs_to_split_groups(self,image):
    return len(glob.glob(image.replace('.fits', '.chip?.fits'))) == 0

  def needs_to_calc_sky(self,image):
    files = glob.glob(image.replace('.fits','.sky.fits'))
    if (len(files) == 0):
      return True
    else:
      hdu1 = fits.open(image)
      hdu2 = fits.open(files[0])
      if (hdu1[0].header == hdu2[0].header):
        return False
      else:
        return True


  def image_contains(self, image, ra, dec):
    # sky2xy is way better/faster than astropy.wcs
    # Also, astropy.wcs has a problem with chip?
    # files generated by splitgroups because the WCSDVARR
    # gets split off from the data, which is required
    # by WCS (see fobj).
    cmd = 'sky2xy {image} {ra} {dec}'.format(image=image,ra=ra,dec=dec)
    result = subprocess.check_output(cmd, shell=True)
    if ('off image' in result):
        print('{image} does not contain RA={ra}, DEC={dec}!'.format(image=image,
                                                                    ra=ra,dec=dec))
        return False
    else:
        return True


  def needs_to_be_masked(self,image):
    # Masking should remove all of the DQ arrays etc,
    # so make sure that any extensions with data in
    # in them are only SCI extensions. This might not be
    # 100% robust, but should be good enough.
    hdulist = fits.open(image)
    needs_masked = False
    for hdu in hdulist:
        if hdu.data is not None and 'EXTNAME' in hdu.header:
            if hdu.header['EXTNAME'].upper() != 'SCI':
                needs_masked = True
    return needs_masked

  def get_filter(self,image):
    if 'c0m.fits' in image:
        f = fits.getval(image, 'FILTNAM1')
        if len(f) == 0:
            f = fits.getval(image, 'FILTNAM2')
    else:
        try:
            f = fits.getval(image, 'FILTER')
        except:
            f = fits.getval(image, 'FILTER1')
            if 'clear' in f.lower():
                f = fits.getval(image, 'FILTER2')
    return f.lower()

  def get_instrument(self,image):
    hdu = fits.open(image, mode='readonly')
    instrument = hdu[0].header['INSTRUME'].lower()
    if instrument.upper() == 'WFPC2':
        detector = 'wfpc2'
        subarray = 'full'
    else:
        detector = hdu[0].header['DETECTOR'].lower()
        if hdu[0].header['SUBARRAY'] == 'T':
            subarray = 'sub'
        else:
            subarray = 'full'
    return '{instrument}_{detector}_{subarray}'.format(instrument=instrument,
                                                       detector=detector,
                                                       subarray=subarray)

  def get_input_images(self,pattern=None):
    if pattern == None:
      pattern = ['*c1m.fits','*c0m.fits','*flc.fits','*flt.fits']
    pattern = [self.input_root_dir+s for s in pattern]
    images = []
    for files in pattern:
      images.extend(glob.glob(files))
    return(images)

  def get_dq_image(self,image):
    instrument = self.get_instrument(image).split('_')[0]
    maskimage = ''
    if instrument.upper() == 'WFPC2':
      maskimage = image.replace('c0m.fits','c1m.fits')
    return maskimage

  def split_groups(self,image):
    os.system('splitgroups {filename}'.format(filename=image))

  def mask_image(self,instrument, image):
    maskimage = self.get_dq_image(image)
    cmd = '{instrument}mask {image} {maskimage}'.format(instrument=instrument,
                                                        image=image, maskimage=maskimage)
    os.system(cmd)

  def calc_sky(self,image, options):
    calcsky_opts = self.get_calcsky_parameters(image, options)
    cmd = 'calcsky {image} {rin} {rout} {step} {sigma_low} {sigma_high}'.format(
                        image=image.replace('.fits',''),
                        rin=calcsky_opts['r_in'], rout=calcsky_opts['r_out'],
                        step=calcsky_opts['step'], sigma_low=calcsky_opts['sigma_low'],
                        sigma_high=calcsky_opts['sigma_high'])
    os.system(cmd)

  def get_calcsky_parameters(self,image, options):
    instrument_string = self.get_instrument(image)
    detector_string = '_'.join(instrument_string.split('_')[:2])
    print('Getting calcsky parameters for:',detector_string)
    return options[detector_string]['dolphot_sky']

  def generate_base_param_file(self,param_file,options):
    for par, value in options['dolphot'].items():
        param_file.write('{par} = {value}\n'.format(par=par, value=value))

  def get_dolphot_instrument_parameters(self,image, options):
    instrument_string = self.get_instrument(image)
    detector_string = '_'.join(instrument_string.split('_')[:2])
    return options[detector_string]['dolphot']

  def generate_image_param_file(self,param_file, image, i, options):
    number = str(i+1).zfill(4)
    param_file.write('img{i}_file = {file}\n'.format(i=number,
                                                     file=os.path.splitext(image)[0]))
    for par, value in self.get_dolphot_instrument_parameters(image, options).items():
      param_file.write('img{i}_{option} = {value}\n'.format(i=number,
                                                            option=par, value=value))

  def generate_template_param_file(self, param_file, template, options):
    number = str(0).zfill(4)
    param_file.write('img{i}_file = {file}\n'.format(i=number,
                                                     file=os.path.splitext(template)[0]))
    for par, value in self.get_dolphot_instrument_parameters(template, options).items():
      param_file.write('img{i}_{option} = {value}\n'.format(i=number,
                                                            option=par, value=value))

  # Pick the best reference out of input images
  # Returns the filter of the reference image.
  # Also generates a drizzled image corresponding
  # to the reference and assigns that drizzled image
  # to self.reference_image_name
  def pick_reference(self):
    # If we haven't defined input images yet,
    # we want to return something to signify
    if not self.input_images:
        print('No input images!!!')
        return None
    else:
        # List of best filters roughly in the order I would
        # want for a decent reference image
        best_filters = ['f606w','f555w','f814w','f350lp',
                        'f450w','f439w','f110w','f160w','f550m']

        # First group images together by filter/instrument
        if not self.filt_list:
            self.filt_list = [self.get_filter(im) for im in self.input_images]
        if not self.inst_list:
            self.inst_list = [self.get_instrument(im).split('_')[0]
                                for im in self.input_images]

        unique_filter_inst = list(set(['{}_{}'.format(a_, b_)
            for a_, b_ in zip(self.filt_list, self.inst_list)]))
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
        for filt in best_filters:
            if any(filt in s for s in unique_filter_inst):
                vals = filter(lambda x: filt in x, unique_filter_inst)
                for v in vals:
                    exposure = total_exposure[unique_filter_inst.index(v)]
                    if exposure > best_exposure:
                        best_filt_inst = v
                        best_exposure = exposure

        # Now check for any long-pass filter
        if not best_filt_inst:
            if any('lp' in s for s in unique_filter_inst):
                vals = filter(lambda x: 'lp' in x, unique_filter_inst)
                for v in vals:
                    exposure = total_exposure[unique_filter_inst.index(v)]
                    if exposure > best_exposure:
                        best_filt_inst = v
                        best_exposure = exposure

        # Now check for any wide-band filter
        if not best_filt_inst:
            if any('w' in s for s in unique_filter_inst):
                vals = filter(lambda x: 'w' in x, unique_filter_inst)
                for v in vals:
                    exposure = total_exposure[unique_filter_inst.index(v)]
                    if exposure > best_exposure:
                        best_filt_inst = v
                        best_exposure = exposure

        # Now check for medium filters
        if not best_filt_inst:
            if any('m' in s for s in unique_filter_inst):
                vals = filter(lambda x: 'm' in x, unique_filter_inst)
                for v in vals:
                    exposure = total_exposure[unique_filter_inst.index(v)]
                    if exposure > best_exposure:
                        best_filt_inst = v
                        best_exposure = exposure

        # Now check for all remaining filters
        if not best_filt_inst:
            for v in unique_filter_inst:
                exposure = total_exposure[unique_filter_inst.index(v)]
                if exposure > best_exposure:
                    best_filt_inst = v
                    best_exposure = exposure

        # Now get list of images with best_filt_inst
        reference_images = []
        for im in self.input_images:
          if (self.get_filter(im)+'_'+
              self.get_instrument(im).split('_')[0] == best_filt_inst):
            reference_images.append(im)
        best_filt,best_inst = best_filt_inst.split('_')
        output_name = best_inst+'_'+best_filt+'_drz.fits'
        print('Reference file will be: ',output_name)
        self.reference_image_name = output_name

        self.run_tweakreg(reference_images,'',self.options['global_defaults'])
        self.run_astrodrizzle(reference_images, output_name = output_name)
        # Astrodrizzle sets saturated/other masked pixels to a weird negative
        # value.  Reset these pixels to nan
        with fits.open(output_name, mode='update') as hdu:
          hdu[1].data[np.where(hdu[1].data == np.min(hdu[1].data))] = float('NaN')
          hdu.flush()

  def run_astrodrizzle(self, images, output_name = None):
    if output_name is None:
        output_name = 'drizzled.fits'

    num_images = len(images)
    if num_images == 1:
        combine_type = 'mean'
    else:
        combine_type = 'minmed'

    wcskey = 'TWEAK'

    options = self.options['global_defaults']
    inst_list = self.get_instrument(images[0]).split('_')
    inst = inst_list[0]
    inst_det = inst_list[0]+'_'+inst_list[1]
    inst_options = self.options['instrument_defaults'][inst]
    det_options = self.options['detector_defaults'][inst_det]
    if ('wfc3_ir' in inst_det):
      driz_bits = 576
    else:
      driz_bits = 0

    for image in images:
      ref = ''
      change_keys = []
      if 'wfpc2' in self.get_instrument(image):
        ref = 'uref'
        change_keys = ['DGEOFILE','IDCTAB','OFFTAB']
      if 'wfc3_uvis' in self.get_instrument(image):
        ref = 'iref'
        change_keys = ['IDCTAB','NPOLFILE','NPOLEXT','D2IMFILE','D2IMEXT']
      if 'wfc3_ir' in self.get_instrument(image):
        ref = 'iref'
        change_keys = ['IDCTAB']
      if 'acs_wfc' in self.get_instrument(image):
        ref = 'jref'
        change_keys = ['IDCTAB','DGEOFILE','NPOLEXT','NPOLFILE','D2IMFILE','D2IMEXT']
      if 'acs_hrc' in self.get_instrument(image):
        ref = 'jref'
        change_keys = ['IDCTAB','DGEOFILE','NPOLEXT','NPOLFILE']
      for key in change_keys:
        try:
          val = fits.getval(image,key,ext=0)
        except:
          try:
            val = fits.getval(image,key,ext=1)
          except:
            print(key,' is not part of header...')
            continue
        if (ref+'$' in val):
          new_val = val.split('$')[1]
          if not os.path.exists(new_val):
            url = options['ftp_'+ref+'_server'] + new_val
            print('Downloading file: ',url)
            urllib.urlretrieve(url,new_val)
            urllib.urlcleanup()
          exts = [0,1]
          for ext in exts:
            fits.setval(image,key,ext=ext,value=new_val)
      updatewcs.updatewcs(image)

    astrodrizzle.AstroDrizzle(images, output=output_name, runfile='astrodrizzle.log',
                wcskey=wcskey, context=True, group='', build=True,
                num_cores=options['num_cores'], preserve=False,
                clean=options['clean'], skysub=options['sky_subtract'],
                skystat='mode', skylower=0.0, skyupper=None, driz_sep_fillval=-100000,
                driz_sep_bits=0, driz_sep_wcs=True, updatewcs=True,
                driz_sep_rot=options['rotation'],
                driz_sep_scale=inst_options['pixel_scale'],
                driz_sep_outnx=det_options['nx'], driz_sep_outny=det_options['ny'],
                driz_sep_ra=self.ra, driz_sep_dec=self.dec,
                combine_maskpt=0.2, combine_type=combine_type, combine_nsigma='4 3',
                combine_nlow=0, combine_nhigh=0, combine_lthresh=-10000,
                combine_hthresh=None, driz_cr=True, driz_cr_snr='3.5 3.0',
                driz_cr_grow=1, driz_cr_ctegrow=0, driz_cr_scale='1.2 0.7',
                final_pixfrac=options['pixel_fraction'], final_fillval=-50000,
                final_bits=driz_bits, final_units='counts',
                final_wcs=True, final_refimage=None,
                final_rot=options['rotation'],
                final_scale=inst_options['pixel_scale'],
                final_outnx=det_options['nx'], final_outny=det_options['ny'],
                final_ra=self.ra, final_dec=self.dec)

  # Run cosmic ray clean
  def run_cosmic(self,image,options,output=None):
    print('Cleaning cosmic rays in image: ',image)
    hdulist = fits.open(image,mode='readonly')

    for hdu in hdulist:
        if ('EXTNAME' in hdu.header and hdu.header['EXTNAME'].upper() == 'SCI'):
            mask = np.zeros(hdu.data.shape, dtype=np.bool)

            _crmask, crclean = detect_cosmics(hdu.data.copy().astype('<f4'), inmask=mask,
                                              readnoise=options['rdnoise'],
                                              gain=options['gain'],
                                              satlevel=options['saturation'],
                                              sigclip=options['sig_clip'],
                                              sigfrac=options['sig_frac'],
                                              objlim=options['obj_lim'])
            hdu.data[:,:] = crclean[:,:]

        if output is None:
            output = image

    # This writes in place
    hdulist.writeto(output,overwrite=True,output_verify='silentfix')
    hdulist.close()

  # Run tweakreg on all input images
  def run_tweakreg(self,images,template,options):

    run_images = images[:]
    # Check if images have already been run through
    # tweakreg.  Could already have tweaked WCS param
    #remove_images = []
    #for image in images:
    #  try:
    #    hdr = fits.open(image)['SCI'].header
    #    keys = hdr.keys()
    #    if ('WCSNAME' in keys):
    #      if (hdr['WCSNAME'] == 'TWEAK'):
    #        # tweakreg has already been run, remove image
    #        remove_images.append(image)
    #  except:
    #    print('Something went wrong checking ',image)

    #for image in remove_images:
    #  run_images.remove(image)

    # Check if we just removed all of the images
    if (len(run_images) == 0):
      print('All images have been run through tweakreg!')
      return(1)

    for image in run_images:
        # wfc3_ir doesn't need cosmic clean and assume template is cleaned
        if (image == template or 'wfc3_ir' in self.get_instrument(image)):
          continue
        rawtmp = image.replace('.fits','rawtmp.fits')

        # Copy the raw data into a temporary file
        shutil.copy(image, rawtmp)

        # Clean cosmic rays on the image in place.
        # We do this so we don't accidentally pick
        # cosmic rays for alignment
        inst = self.get_instrument(image).split('_')[0]
        crpars = self.options['instrument_defaults'][inst]['crpars']
        self.run_cosmic(image,crpars)

    if (template == '' or template == None):
      shutil.copy(images[0],'dummy.fits')
      template = 'dummy.fits'

    print('Executing tweakreg with images: ',run_images)
    print('Template image: ',template)
    print('Tweakreg is executing...')
    start_tweak = time.time()
    tweakreg.TweakReg(files = run_images, refimage = template, verbose=False,
            interactive=False, clean=True, writecat = False, updatehdr=True,
            wcsname='TWEAK', reusename=True, rfluxunits='counts', see2dplot=False,
            separation=0.5, residplot='No plot', runfile='',
            imagefindcfg = {'threshold': options['tweakshifts_threshold'],
                'use_sharp_round': True},
            refimagefindcfg = {'threshold': options['tweakshifts_threshold'],
                'use_sharp_round': True})

    print('Tweakreg took',time.time()-start_tweak,'seconds to execute.')

    for image in run_images:
        if (image == template or 'wfc3_ir' in self.get_instrument(image)):
          continue
        rawtmp = image.replace('.fits','rawtmp.fits')
        hdu_from = fits.open(rawtmp)
        hdu_to = fits.open(image)

        for i, hdu in enumerate(hdu_from):
            if ('EXTNAME' in hdu.header and hdu.header['EXTNAME'].upper() == 'SCI'):
                hdu_to[i].data[:,:] = hdu.data[:,:]

        hdu_from.close()
        hdu_to.writeto(image,overwrite = True)
        hdu_to.close()
        os.remove(rawtmp)

    if os.path.isfile('dummy.fits'):
      os.remove('dummy.fits')

  # download files for input ra and dec
  def download_files(self):
    coord = SkyCoord(self.ra,self.dec,unit='deg')

    obsTable = Observations.query_region(coord,radius=5*u.arcmin)
    colmask = obsTable['obs_collection'] == 'HST'
    immask = obsTable['dataproduct_type'] == 'image'
    detmask = [obsTable['instrument_name'] == good for good in self.good_detector]
    detmask = [any(l) for l in list(map(list,zip(*detmask)))]
    rightmask = obsTable['dataRights'] == 'PUBLIC'

    good = [a and b and c and d for a,b,c,d in zip(colmask,immask,detmask,rightmask)]
    obsTable = obsTable[good]

    for obs in obsTable:
      productList = Observations.get_product_list(obs)
      for prod in productList:
        filename = prod['productFilename']
        if ('c0m.fits' in filename or
            'c1m.fits' in filename or
            'flt.fits' in filename or
            'flc.fits' in filename):
            obsid = prod['obsID']
            uri = prod['dataURI']

            if (os.path.isfile(filename) and not self.clobber):
              print(filename,'already exists and clobber is False')
              continue

            print('Trying to download: ',filename)

            url = self.download_uri + uri
            utils.data.clear_download_cache()
            try:
              dat = utils.data.download_file(url,
                  cache=False,show_progress=False,timeout=120)
              shutil.move(dat,filename)
            except:
              print('Timed out when downloading',filename)


if __name__ == '__main__':

    start = time.time()
    print('Starting hst123...')
    print('###################################')
    print('###################################')
    print('')
    print('')
    time.sleep(1)

    usagestring='USAGE: hst123.py'
    hst = hst123()
    parser = hst.add_options(usage=usagestring)
    options, args = parser.parse_args()

    hst.reference_image_name = options.reference
    hst.ra = options.ra
    hst.dec = options.dec
    hst.download = options.download
    hst.clobber = options.clobber

    if (options.download):
      if (not hst.ra or not hst.dec):
        print('hst123.py needs input ra and dec for the download option!')
        sys.exit(2)
      else:
        print('Downloading HST data from MAST for ra=',hst.ra,'dec=',hst.dec)
        print('###################################')
        print('###################################')
        print('')
        print('')
        hst.download_files()

    if (options.makeclean):
      print('Cleaning output from previous runs of hst123')
      print('###################################')
      print('###################################')
      print('')
      print('')
      os.system('rm -rf '+hst.dolphot_param_file)
      os.system('rm -rf *chip?.fits *chip?.sky.fits')
      os.system('rm -rf *rawtmp.fits')
      os.system('rm -rf *drz.fits *drz.sky.fits')
      os.system('rm -rf *idc.fits *dxy.fits *off.fits *d2im.fits')
      os.system('rm -rf *d2i.fits *npl.fits')
      os.system('rm -rf dp*')
      os.system('rm -rf *.log *.output')
      for file in glob.glob(hst.raw_dir+'*.fits'):
        path, base = os.path.split(file)
        if filecmp.cmp(file,base):
          print('{file} and {base} are identical'.format(file=file,base=base))
          continue
        else:
          print('Copying {file} to {base}'.format(file=file,base=base))
          shutil.copy(file, base)
        os.chmod(base,0775)
      sys.exit(0)

    # Unzip input_images
    if (len(glob.glob(hst.input_root_dir+'*.fits.gz')) != 0):
      os.system('gunzip '+hst.input_root_dir+'*.fits.gz')

    # Get input images
    hst.input_images = hst.get_input_images()

    print('Copying raw data into raw data folder: ',hst.raw_dir)
    print('###################################')
    print('###################################')
    print('')
    print('')

    # Make raw/ in current dir and copy files into raw/
    hst.copy_raw_data()

    # Check which are HST images that need to be reduced
    remove_list = []
    for file in hst.input_images:
      if (not hst.needs_to_be_reduced(file)):
        remove_list.append(file)
        continue
      if (hst.get_filter(file).upper() not in hst.options['acceptable_filters']):
        remove_list.append(file)

    for file in remove_list:
      hst.input_images.remove(file)

    if not hst.reference_image_name:
      print('No reference image was provided.')
      print('Generating reference image from input files.')
      print('###################################')
      print('###################################')
      print('')
      print('')
      hst.pick_reference()

    print('Sanitizing reference image:',hst.reference_image_name)
    print('###################################')
    print('###################################')
    print('')
    print('')
    hst.sanitize_template(hst.reference_image_name)

    print('Running main tweakreg')
    print('###################################')
    print('###################################')
    print('')
    print('')
    hst.run_tweakreg(hst.input_images,
                     hst.reference_image_name,
                     hst.options['global_defaults'])

    print('Sanitizing WFPC2 images:')
    print('###################################')
    print('###################################')
    print('')
    print('')
    for file in hst.input_images:
      if 'wfpc2' in hst.get_instrument(file):
        hst.sanitize_wfpc2(file)

    f = open(hst.dolphot_param_file,'w')
    hst.generate_base_param_file(f,hst.options['global_defaults'])

    for file in hst.input_images:
      hst.filt_list.append(hst.get_filter(file))
      hst.inst_list.append(hst.get_instrument(file))
      if (hst.needs_to_be_masked(file)):
        instrument = hst.get_instrument(file).split('_')[0]
        hst.mask_image(instrument,file)
      if (hst.needs_to_split_groups(file)):
        hst.split_groups(file)
      split_images = glob.glob(file.replace('.fits', '.chip?.fits'))
      hst.split_images.extend(split_images)
      for split in split_images:
        if hst.needs_to_calc_sky(split):
          hst.calc_sky(split,hst.options['detector_defaults'])
      hst.total_images += 1

    # Generate template image sky file
    if (hst.needs_to_calc_sky(hst.reference_image_name)):
      hst.calc_sky(hst.reference_image_name,hst.options['detector_defaults'])

    print('')
    print('')
    print('###################################')
    print('###################################')
    print('Complete list of input images: ')
    line_format = '{0: <18} {1: <14} {2: <8} {3: <10} {4: <10} {5: <10}'
    line = line_format.format('FILE','INSTRUMENT','FILTER',
                              'EXPTIME','DATE-OBS','TIME-OBS')
    print(line)
    for file,inst,filt in zip(hst.input_images,hst.inst_list,hst.filt_list):
        exptime = fits.getval(file,'EXPTIME')
        dateobs = fits.getval(file,'DATE-OBS')
        timeobs = fits.getval(file,'TIME-OBS')
        line=line_format.format(file,inst.upper(),filt.upper(),exptime,dateobs,timeobs)
        print(line)

    print('###################################')
    print('###################################')
    print('')
    print('')

    if (hst.ra != None and hst.dec != None):
      print('Getting rid of images that do'+\
            ' not contain RA={ra}, DEC={dec}'.format(ra=hst.ra,dec=hst.dec))
      print('###################################')
      print('###################################')
      print('')
      print('')

      remove_list = []
      for image in hst.split_images:
        if not hst.image_contains(image, hst.ra, hst.dec):
          remove_list.append(image)

      for image in remove_list:
        hst.split_images.remove(image)

    print('Adding images to dolphot parameter file: ',hst.dolphot_param_file)
    print('###################################')
    print('###################################')
    print('')
    print('')

    f.write('Nimg = {n}\n'.format(n=len(hst.split_images)))
    hst.generate_template_param_file(f, hst.reference_image_name,
                                        hst.options['detector_defaults'])
    for i,file in enumerate(hst.split_images):
      hst.generate_image_param_file(f, file, i, hst.options['detector_defaults'])
    f.close()

    print('Total time elapsed to complete this script: ',time.time()-start,'seconds')
