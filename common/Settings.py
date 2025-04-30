from astropy import units as u

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
    'rmap': {
        'base_url': 'https://hst-crds.stsci.edu/unchecked_get/mappings/hst',
        'acs': {
            'wfc': {
                'idctab': 'hst_acs_idctab_0256.rmap',

            },
            'hrc': {

            }
        },
        'wfc3': {
            'uvis': {

            },
            'ir': {

            }
        },
        'wfpc2': {

        }
    },
    'visit': 1,
    'search_rad': 1.0, # for tweakreg in arcsec
    'radius': 5 * u.arcmin,
    'nbright': 4000,
    'minobj': 10,
    'dolphot': {'FitSky': 2,
                'SkipSky': 2,
                'RCombine': 1.5,
                'SkySig': 2.25,
                'SecondPass': 1,
                'SigFindMult': 0.85,
                'MaxIT': 25,
                'NoiseMult': 0.10,
                'FSat': 0.999,
                'ApCor': 1,
                'RCentroid': 2,
                'PosStep': 0.25,
                'dPosMax': 2.5,
                'SigPSF': 5.0,
                'PSFres': 1,
                'Align': 2,
                'Rotate': 1,
                'ACSuseCTE': 0,
                'WFC3useCTE': 0,
                'WFPC2useCTE': 1,
                'FlagMask': 7,
                'SigFind': 2.5,
                'SigFinal': 3.5,
                'UseWCS': 1,
                'AlignOnly': 0,
                'AlignIter': 5,
                'AlignTol': 0.5,
                'AlignStep': 4.0,
                'VerboseData': 1,
                'NegSky': 1,
                'Force1': 1,
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
    'skysigma':0.0,
    'computesig':True,
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
    'theta':0.0,
    'use_sharp_round': True,
    'expand_refcat': False,
    'enforce_user_order': True,
    'clean': True,
    'interactive': False,
    'verbose': False,
    'updatewcs': False,
    'xyunits': 'pixels',
    '_RULES_': {'_rule_1':'True', '_rule2_':'False'}
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
                  'driz_sep_scale': None,
                  'input_files': '*_flc.fits', 'pixel_scale': 0.04,
                  'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                  'sigma_low': 2.25, 'sigma_high': 2.00},
                  'dolphot': {'apsky': '15 25', 'RAper': 3, 'RChi': 2.0,
                              'RPSF': 13, 'RSky': '15 35',
                              'RSky2': '4 10'},
                   'idcscale': 0.03962000086903572},
    'wfc3_ir': {'driz_bits': 576, 'nx': 5200, 'ny': 5200,
                'driz_sep_scale': None,
                'input_files': '*_flt.fits', 'pixel_scale': 0.128,
                'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '8 20', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 15, 'RSky': '8 20',
                            'RSky2': '3 10'},
                   'idcscale': 0.1282500028610229},
    'acs_wfc': {'driz_bits': 96, 'nx': 5200, 'ny': 5200,
                'driz_sep_scale': None,
                'input_files': '*_flc.fits', 'pixel_scale': 0.05,
                'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 10, 'RSky': '15 35',
                            'RSky2': '3 6'}},
    'acs_hrc': {'driz_bits': 0, 'nx': 5200, 'ny': 5200,
                'driz_sep_scale': None,
                'input_files': '*_flt.fits', 'pixel_scale': 0.05,
                'dolphot_sky': {'r_in': 15, 'r_out': 35, 'step': 4,
                                'sigma_low': 2.25, 'sigma_high': 2.00},
                'dolphot': {'apsky': '15 25', 'RAper': 2, 'RChi': 1.5,
                            'RPSF': 10, 'RSky': '15 35',
                            'RSky2': '3 6'}},
    'wfpc2_wfpc2': {'driz_bits': 1032, 'nx': 5200, 'ny': 5200,
                    'driz_sep_scale': 0.046,
                    'input_files': '*_c0m.fits', 'pixel_scale': 0.046,
                    'dolphot_sky': {'r_in': 10, 'r_out': 25, 'step': 2,
                                    'sigma_low': 2.25, 'sigma_high': 2.00},
                    'dolphot': {'apsky': '15 25', 'RAper': 3, 'RChi': 2,
                                'RPSF': 13, 'RSky': '15 35',
                                'RSky2': '4 10'}}}

acceptable_filters = {'jwst': ['F444W'],
    'hst': [
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
    'F953N','F1042M']}

pipeline_products = ['*chip?.fits', '*chip?.sky.fits',
                              '*rawtmp.fits', '*drz.fits', '*drz.sky.fits',
                              '*idc.fits', '*dxy.fits', '*off.fits',
                              '*d2im.fits', '*d2i.fits', '*npl.fits',
                              'dp*', '*.log', '*.output','*sci?.fits',
                              '*wht.fits','*sci.fits','*StaticMask.fits']

pipeline_images = ['*flc.fits','*flt.fits','*c0m.fits','*c1m.fits']

# Names for input image table
names = ['image','exptime','datetime','filter','instrument',
         'detector','zeropoint','chip','imagenumber']

# Names for the final output photometry table
final_names = ['MJD', 'INSTRUMENT', 'FILTER', 'EXPTIME', 'MAGNITUDE', 
               'MAGNITUDE_ERROR']
