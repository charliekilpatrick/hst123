#!/usr/bin/env python
# CDK v1.00: 2019-02-07
#
# hst123.py: An all-in-one pipeline for downloading,
# registering, drizzling, and preparing HST data for
# dolphot reductions
#
# Python 2/3 compatibility
from __future__ import print_function

# Dependencies and settings
import glob, sys, os, shutil, time, urllib, subprocess, warnings, filecmp
import numpy as np
from datetime import datetime
from stwcs import updatewcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
from astropy import utils
from astropy.table import Table
from astropy.time import Time
from astroquery.mast import Observations
from drizzlepac import tweakreg,astrodrizzle
from astroscrappy import detect_cosmics
warnings.filterwarnings('ignore')

green = '\033[1;32;40m'
red = '\033[1;31;40m'
end = '\033[0;0m'

global_defaults = {
    'change_keys': ['IDCTAB','DGEOFILE','NPOLEXT','NPOLFILE','D2IMFILE',
                    'D2IMEXT','OFFTAB'],
    'cdbs_ftp': 'ftp://ftp.stsci.edu/cdbs/',
    'mast_radius': 6*u.arcmin,
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
                'InterpPSFlib': 1}}

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
                  'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                              'RPSF': 10, 'RSky': '15 35',
                              'RSky2': '3 6'}},
    'wfc3_ir': {'driz_bits': 512, 'nx': 5200, 'ny': 5200,
                'input_files': '*_flt.fits', 'pixel_scale': 0.09,
                'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 10, 'RSky': '8 20',
                            'RSky2': '3 6'}},
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
                    'input_files': '*_c0m.fits', 'pixel_scale': 0.046,
                    'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                    'sigma_low': 2.25, 'sigma_high': 2.00},
                    'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                                'RPSF': 10, 'RSky': '15 35',
                                'RSky2': '3 6'}}}

acceptable_filters = {
    'F220W','F250W','F330W','F344N','F435W','F475W','F502N','F550M','F555W',
    'F606W','F625W','F658N','F660N','F660N','F775W','F814W','F850LP','F892N',
    'F098M','F105W','F110W','F125W','F126N','F127M','F128N','F130N','F132N',
    'F139M','F140W','F153M','F160W','F164N','F167N','F200LP','F218W','F225W',
    'F275W','F280N','F300X','F336W','F343N','F350LP','F373N','F390M','F390W',
    'F395N','F410M','F438W','F467M','F469N','F475X','F487N','F502N','F547M',
    'F600LP','F621M','F625W','F631N','F645N','F656N','F657N','F658N','F665N',
    'F673N','F680N','F689M','F763M','F845M','F953N','F122M','F160BW','F185W',
    'F218W','F255W','F300W','F375N','F380W','F390N','F437N','F439W','F450W',
    'F569W','F588N','F622W','F631N','F673N','F675W','F702W','F785LP','F791W',
    'F953N','F1042M'}

class hst123(object):

  def __init__(self):

    # Basic parameters
    self.input_images = []
    self.split_images = []
    self.template = ''
    self.coord = None
    self.download = False
    self.clobber = False
    self.dolphot = {}
    self.root_dir = '.'
    self.raw_dir = 'raw/'

    self.options = {'global_defaults': global_defaults,
                    'detector_defaults': detector_defaults,
                    'instrument_defaults': instrument_defaults,
                    'acceptable_filters': acceptable_filters}

    # Complete list of pipeline products in case
    # they need to be cleaned at start
    self.pipeline_products = ['*chip?.fits','*chip?.sky.fits',
                              '*rawtmp.fits','*drz.fits','*drz.sky.fits',
                              '*idc.fits','*dxy.fits','*off.fits',
                              '*d2im.fits','*d2i.fits','*npl.fits',
                              'dp*','*.log','*.output']

  def add_options(self, parser=None, usage=None):
    import optparse
    if parser == None:
        parser = optparse.OptionParser(usage=usage,
            conflict_handler='resolve')
    parser.add_option('--ra', default=None,
        type='string', help='RA of interest')
    parser.add_option('--dec', default=None,
        type='string', help='DEC of interest')
    parser.add_option('--reference','--ref', default='',
        type='string', help='Name of the reference image.')
    parser.add_option('--inputrootdir','--input', default=None,
        type='string', help='Pathway to input HST files.')
    parser.add_option('--makeclean', default=False, action='store_true',
        help='Clean up all output files from previous runs then exit.')
    parser.add_option('--download', default=False, action='store_true',
        help='Download the raw data files given input ra and dec.')
    parser.add_option('--clobber', default=False, action='store_true',
        help='Overwrite files when using download mode.')
    parser.add_option('--dolphot','--dp', default='dp', type='string',
        help='Base name for dolphot parameter file and output.')
    return(parser)

  # Make sure all standard output is
  # formatted in the same way with banner
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

  def show_input_list(self):
    form = '{file: <19} {inst: <18} {filt: <10} '
    form += '{exp: <10} {date: <10} {time: <10}'
    header = form.format(file='FILE',inst='INSTRUMENT',filt='FILTER',
                              exp='EXPTIME',date='DATE-OBS',time='TIME-OBS')
    print(header)

    # Make a table with all of the metadata for each image.
    # In future, it might be good to make this at start so
    # other functions can reference the metadata.
    exptime = [fits.getval(image,'EXPTIME') for image in self.input_images]
    datetim = [fits.getval(image,'DATE-OBS') + 'T' +
               fits.getval(image,'TIME-OBS') for image in self.input_images]
    filters = [self.get_filter(image) for image in self.input_images]
    instrum = [self.get_instrument(image) for image in self.input_images]

    table = Table([self.input_images,exptime,datetim,filters,instrum],
                   names=('image','exptime','datetime','filter','instrument'))
    table.sort('datetime')

    for row in table:
        line = form.format(file=row['image'], inst=row['instrument'].upper(),
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

  def make_dolphot_dict(self, dolphot):
    return({ 'base': dolphot, 'param': dolphot+'.param' })

  def copy_raw_data(self, reverse = False):
    # reverse = False will simply backup data
    # in the working directory to the raw dir
    if not reverse:
        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)
        for f in self.input_images:
            if not os.path.isfile(self.raw_dir+f):
                shutil.copy(f, self.raw_dir)
    # reverse = True will copy files from the
    # raw dir to the working directory if the
    # working files are different from the raw
    # dir.  This is necessary for the pipeline
    # to work properly, as some procedures
    # (esp. WCS checking, tweakreg, astrodrizzle)
    # require un-edited files.
    else:
        for file in glob.glob(self.raw_dir+'*.fits'):
            path, base = os.path.split(file)
            if filecmp.cmp(file,base):
                message = '{file} == {base}'
                print(message.format(file=file,base=base))
                continue
            else:
                message = '{file} != {base}'
                print(message.format(file=file,base=base))
                shutil.copy(file, base)
            os.chmod(base,0775)

  # Sanitizes template header, gets rid of
  # multiple extensions and only preserves
  # science data.
  def sanitize_template(self, template):
    hdu = fits.open(template, mode='readonly')

    # Going to write out newhdu
    # Only want science extension from orig template
    newhdu = fits.HDUList()
    newhdu.append(hdu['SCI'])

    # Want to preserve header info, so combine
    # SCI+PRIMARY headers (except COMMENT/HISTORY keys)
    for key in hdu['PRIMARY'].header.keys():
        if (key not in newhdu[0].header.keys()+['COMMENT','HISTORY']):
            newhdu[0].header[key] = hdu['PRIMARY'].header[key]

    # Make sure that template header reflects one extension
    newhdu[0].header['EXTEND']=False

    # Add header variables that dolphot needs:
    # GAIN, RDNOISE, SATURATE
    inst = newhdu[0].header['INSTRUME'].lower()
    opt  = self.options['instrument_defaults'][inst]['crpars']
    newhdu[0].header['SATURATE'] = opt['saturation']
    newhdu[0].header['RDNOISE']  = opt['rdnoise']
    newhdu[0].header['GAIN']     = opt['gain']

    if 'WHT' in [h.name for h in hdu]:
        wght = hdu['WHT'].data
        newhdu['SCI'].data[np.where(wght == 0)] = float('NaN')

    # Write out to same file w/ overwrite
    newhdu.writeto(template, output_verify='silentfix', overwrite=True)

  # Sanitizes wfpc2 data by getting rid of
  # extra extensions and changing variables.
  def sanitize_wfpc2(self, image):
    hdu = fits.open(image, mode='readonly')

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

  def parse_coord(self, ra, dec):
    if (':' in ra and ':' in dec):
        # Input RA/DEC are sexagesimal
        return(SkyCoord(ra,dec,frame='fk5'))
    elif (self.is_number(ra) and self.is_number(dec)):
        # Assume input coordiantes are decimal degrees
        return(SkyCoord(ra,dec,frame='fk5',unit='deg'))
    else:
        # Throw an error and exit
        error = 'ERROR: Cannot parse coordinates ra={ra}, dec={dec}'
        print(error.format(ra=ra,dec=dec))
        return(None)

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
        return(False)

    # Get rid of data where the input coordinates do
    # not land in any of the sub-images
    if self.coord:
        ra = self.coord.ra.degree
        dec = self.coord.dec.degree
        for h in hdu:
            if h.data is not None and 'EXTNAME' in h.header:
                if h.header['EXTNAME'] == 'SCI':
                    w = WCS(h.header,hdu)
                    world_coord = np.array([[ra,dec]])
                    pixcrd = w.wcs_world2pix(world_coord,1)
                    if (pixcrd[0][0] > 0 and
                        pixcrd[0][1] > 0 and
                        pixcrd[0][0] < h.header['NAXIS1'] and
                        pixcrd[0][1] < h.header['NAXIS2']):
                        is_not_hst_image = True
    if not is_not_hst_image:
        return(False)

    # Get rid of images that don't match one of the
    # allowed instrument/detector types and images
    # whose extensions don't match the allowed type
    # for those instrument/detector types
    is_not_hst_image = False
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
    return(is_not_hst_image)

  def needs_to_split_groups(self,image):
    return(len(glob.glob(image.replace('.fits', '.chip?.fits'))) == 0)

  def needs_to_calc_sky(self,image):
    files = glob.glob(image.replace('.fits','.sky.fits'))
    if (len(files) == 0):
        if self.coord:
            return(self.image_contains(image, self.coord))
        else:
            return(True)


  def image_contains(self, image, coord):
    # sky2xy is way better/faster than astropy.wcs
    # Also, astropy.wcs has a problem with chip?
    # files generated by splitgroups because the WCSDVARR
    # gets split off from the data, which is required
    # by WCS (see fobj).
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
    return(needs_masked)

  def get_filter(self, image):
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
    return(f.lower())

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

  def get_input_images(self,pattern=None):
    if pattern == None:
        pattern = ['*c1m.fits','*c0m.fits','*flc.fits','*flt.fits']
    return([s for p in pattern for s in glob.glob(p)])

  def get_dq_image(self,image):
    if self.get_instrument(image).split('_')[0].upper() == 'WFPC2':
        return(image.replace('c0m.fits','c1m.fits'))
    else:
        return('')

  def split_groups(self,image):
    print('Running split groups for {image}'.format(image=image))
    os.system('splitgroups {filename}'.format(filename=image))

  def mask_image(self, image, instrument):
    maskimage = self.get_dq_image(image)
    cmd = '{instrument}mask {image} {maskimage}'
    mask = cmd.format(instrument=instrument, image=image, maskimage=maskimage)
    print(mask)
    os.system(mask)

  def calc_sky(self, image, options):
    det = '_'.join(self.get_instrument(image).split('_')[:2])
    opt = options[det]['dolphot_sky']
    cmd = 'calcsky {image} {rin} {rout} {step} {sigma_low} {sigma_high}'
    calc_sky = cmd.format(image=image.replace('.fits',''), rin=opt['r_in'],
                            rout=opt['r_out'], step=opt['step'],
                            sigma_low=opt['sigma_low'],
                            sigma_high=opt['sigma_high'])
    os.system(calc_sky)

  def generate_base_param_file(self,param_file,options):
      for par, value in options['dolphot'].items():
          param_file.write('{par} = {value}\n'.format(par=par, value=value))

  def get_dolphot_instrument_parameters(self,image, options):
    instrument_string = self.get_instrument(image)
    detector_string = '_'.join(instrument_string.split('_')[:2])
    return(options[detector_string]['dolphot'])

  def add_image_param_file(self,param_file, image, i, options):
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

  # Pick the best reference out of input images
  # Returns the filter of the reference image.
  # Also generates a drizzled image corresponding
  # to the reference
  def pick_reference(self):
    # If we haven't defined input images yet,
    # we want to return something to signify
    if not self.input_images:
        warning = 'WARNING: No input images.'
        print(warning)
        sys.exit(1)
    else:
        # List of best filters roughly in the order I would
        # want for a decent reference image
        best_filters = ['f606w','f555w','f814w','f350lp',
                        'f450w','f439w','f110w','f160w','f550m']

        # First group images together by filter/instrument
        filts = [self.get_filter(im) for im in self.input_images]
        insts = [self.get_instrument(im).split('_')[0]
                 for im in self.input_images]

        unique_filter_inst = list(set(['{}_{}'.format(a_, b_)
                                       for a_, b_ in zip(filts, insts)]))
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
        message = 'Reference image name will be: {template}'
        print(message.format(template=output_name))
        self.template = output_name

        self.run_tweakreg(reference_images,'',self.options['global_defaults'])
        self.run_astrodrizzle(reference_images, output_name = output_name)

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
        error = 'ERROR: Cannot drizzle together images from detectors: {det}'
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
                    print(message.format(url=url))
                    urllib.urlretrieve(url,ref_file)
                    urllib.urlcleanup()
            fits.setval(image, key, extname='PRIMARY', value=ref_file)
            fits.setval(image, key, extname='SCI', value=ref_file)
        updatewcs.updatewcs(image)

    ra = self.coord.ra.degree if self.coord else None
    dec = self.coord.dec.degree if self.coord else None

    astrodrizzle.AstroDrizzle(images, output=output_name, runfile='',
                wcskey=wcskey, context=True, group='', build=True,
                num_cores=8, preserve=False, clean=True, skysub=True,
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
  def run_tweakreg(self, images, template, options):

    run_images = images[:]

    # Check if we just removed all of the images
    if (len(run_images) == 0):
        warning = 'WARNING: All images have been run through tweakreg'
        print(warning)
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

    message = 'Executing tweakreg with images: {images} \n'
    message += 'Template image: {template} \n'
    message += 'Tweakreg is executing...'
    print(message.format(images = ','.join(map(str,run_images)),
                         template = template))
    start_tweak = time.time()
    tweakreg.TweakReg(files = run_images, refimage = template, verbose=False,
            interactive=False, clean=True, writecat = False, updatehdr=True,
            wcsname='TWEAK', reusename=True, rfluxunits='counts',
            see2dplot=False, separation=0.5, residplot='No plot', runfile='',
            imagefindcfg = {'threshold': 5, 'use_sharp_round': True},
            refimagefindcfg = {'threshold': 5, 'use_sharp_round': True})

    message = 'Tweakreg took {time} seconds to execute'
    print(message.format(time = time.time()-start_tweak))

    for image in run_images:
        if (image == template or 'wfc3_ir' in self.get_instrument(image)):
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
        os.remove(rawtmp)

    if os.path.isfile('dummy.fits'):
      os.remove('dummy.fits')

  # download files for input ra and dec
  def download_files(self):
    search_radius = self.options['global_defaults']['mast_radius']
    obsTable = Observations.query_region(self.coord, radius=search_radius)
    telmask = [tel.upper() == 'HST' for tel in obsTable['obs_collection']]
    promask = [pro.upper() == 'IMAGE' for pro in obsTable['dataproduct_type']]
    detmask = [any(l) for l in list(map(list,zip(*[[det in inst.upper()
                for inst in obsTable['instrument_name']]
                for det in ['ACS','WFC','WFPC2']])))]
    ritmask = [rit.upper() == 'PUBLIC' for rit in obsTable['dataRights']]

    mask = [all(l) for l in zip(telmask,promask,detmask,ritmask)]
    obsTable = obsTable[mask]

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
                    # utils.data.download_file can get buggy
                    # if the cache is full.  Clear the cache
                    # even though we aren't using caching to
                    # prevent this method from choking
                    utils.data.clear_download_cache()
                except RuntimeError:
                    pass
                try:
                    dat = utils.data.download_file(url, cache=False,
                        show_progress=False, timeout=120)
                    shutil.move(dat,filename)
                    message = '\r' + message
                    message += green+' [SUCCESS]'+end+'\n'
                    sys.stdout.write(message.format(image=filename))
                except:
                    message = '\r' + message
                    message += red+' [FAILURE]'+end+'\n'
                    sys.stdout.write(message.format(image=filename))

if __name__ == '__main__':

    start = time.time()
    time.sleep(1)

    usagestring='USAGE: hst123.py'
    hst = hst123()
    banner = 'Starting hst123.py'
    hst.make_banner(banner)
    parser = hst.add_options(usage=usagestring)
    options, args = parser.parse_args()

    hst.template = options.reference
    hst.download = options.download
    hst.clobber = options.clobber
    hst.dolphot = hst.make_dolphot_dict(options.dolphot)
    if (options.ra is not None and options.dec is not None):
        hst.coord = hst.parse_coord(options.ra, options.dec)

    if (options.download):
        if not hst.coord:
            error = 'ERROR: hst123.py --download requires input ra and dec'
            print(error)
            sys.exit(2)
        else:
            banner = 'Downloading HST data from MAST for ra={ra}, dec={dec}'
            hst.make_banner(banner.format(ra=hst.coord.ra.degree,
                                          dec=hst.coord.dec.degree))
            hst.download_files()

    if (options.makeclean):
        banner = 'Cleaning output from previous runs of hst123'
        hst.make_banner(banner)
        for pattern in hst.pipeline_products:
            for file in glob.glob(pattern):
                os.remove(file)
        hst.copy_raw_data(reverse = True)
        sys.exit(0)

    # Get input images
    hst.input_images = hst.get_input_images()

    # Make raw/ in current dir and copy files into raw/
    # then copy un-edited versions of files back into
    # working dir
    banner = 'Copying raw data to and re-copying from raw data folder: {dir}'
    hst.make_banner(banner.format(dir=hst.raw_dir))
    hst.copy_raw_data()
    hst.copy_raw_data(reverse=True)

    # Check which are HST images that need to be reduced
    remove_list = []
    for file in hst.input_images:
        if not (hst.needs_to_be_reduced(file) and
            hst.get_filter(file).upper() in hst.options['acceptable_filters']):
            remove_list.append(file)

    for file in remove_list:
        hst.input_images.remove(file)

    if len(hst.input_images) == 0:
        error = 'ERROR: No input images.  Exiting...'
        print(error)
        sys.exit(1)

    # If not reference image was provided then make one
    if not hst.template:
        banner = 'Generating reference image from input files.'
        hst.make_banner(banner)
        hst.pick_reference()

    # Sanitize extensions and header variables in reference
    banner = 'Sanitizing reference image: {image}'
    hst.make_banner(banner.format(image = hst.template))
    hst.sanitize_template(hst.template)

    # Run main tweakreg to register to the reference
    banner = 'Running main tweakreg'
    hst.make_banner(banner)
    hst.run_tweakreg(hst.input_images,hst.template,
                     hst.options['global_defaults'])

    # Sanitize any WFPC2 images
    banner = 'Sanitizing WFPC2 images'
    hst.make_banner(banner)
    for file in hst.input_images:
        if 'wfpc2' in hst.get_instrument(file):
            hst.sanitize_wfpc2(file)

    # Start all of the dolphot preparation.
    # Write out a parameter file and do all
    # the mask, splitgroups, calcsky stuff
    dolphot_file = open(hst.dolphot['param'],'w')
    hst.generate_base_param_file(dolphot_file,hst.options['global_defaults'])

    # dolphot file preparation
    message = 'Preparing dolphot data for files={files}'
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

        for split in split_images:
            if hst.needs_to_calc_sky(split):
                # run calc sky with calc_sky parameters
                hst.calc_sky(split, hst.options['detector_defaults'])

    # Generate template image sky file
    if (hst.needs_to_calc_sky(hst.template)):
        hst.calc_sky(hst.template,hst.options['detector_defaults'])

    # Write out a complete list of the input
    # images with metadata for easy reference
    banner = 'Complete list of input images'
    hst.make_banner(banner)
    hst.show_input_list()

    # Start adding the sub-image info to dolphot param file
    banner = 'Adding images to dolphot parameter file: {file}'
    hst.make_banner(banner.format(file = hst.dolphot['param']))

    # Add the number of images to reduce
    dolphot_file.write('Nimg = {n}\n'.format(n=len(hst.split_images)))

    # Write template image to param file
    hst.add_image_param_file(dolphot_file, hst.template, 0,
                             hst.options['detector_defaults'])

    # Write out image-specific params to dolphot file
    for i,image in enumerate(hst.split_images):
        hst.add_image_param_file(dolphot_file, image, i+1,
                                 hst.options['detector_defaults'])
    dolphot_file.close()

    message = 'It took {time} seconds to complete this script'
    print(message.format(time = time.time()-start))
