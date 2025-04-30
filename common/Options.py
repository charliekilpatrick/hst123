# Arguments and options for hst123
def add_options(parser=None, usage=None):
    import argparse
    if parser == None:
        parser = argparse.ArgumentParser(usage=usage,conflict_handler='resolve')
    
    # Basic arguments and options
    parser.add_argument('ra', type=str,
        help='Right ascension to reduce the HST images')
    parser.add_argument('dec', type=str,
        help='Declination to reduce the HST images')
    parser.add_argument('--work-dir', default=None, type=str,
        help='Use the input working directory rather than the current dir.')
    parser.add_argument('--raw-dir', default='./', type=str,
        help='Use the input raw data directory rather than the current dir.')
    parser.add_argument('--make-clean', default=False, action='store_true',
        help='Clean up all output files from previous runs then exit.')
    parser.add_argument('--download', default=False, action='store_true',
        help='Download the raw data files given input ra and dec.')
    parser.add_argument('--token', default=None, type=str,
        help='Input a token for astroquery.mast.Observations.')
    parser.add_argument('--archive', default=None, type=str,
        help='Download and save raw data to an archive directory instead of '+\
        'same folder as reduction (see global_defaults[\'archive\'])')
    parser.add_argument('--no-clear-downloads', default=False,
        action='store_true', help='Suppress the clear_downloads method.')
    parser.add_argument('--clobber', default=False, action='store_true',
        help='Overwrite files when using download mode.')
    parser.add_argument('--cleanup', default=False, action='store_true',
        help='Clean up interstitial image files (i.e., flt,flc,c1m,c0m).')
    parser.add_argument('--skip-copy', default=False, action='store_true',
        help='Skip copying files from archive if --archive is used.')
    parser.add_argument('--by-visit', default=False, action='store_true',
        help='Reduce images by visit number.')
    
    # Options for selecting files to reduce/analyze
    parser.add_argument('--before', default=None, type=str,
        help='Reject obs after this date.')
    parser.add_argument('--after', default=None, type=str,
        help='Reject obs before this date.')
    parser.add_argument('--only-filter', default=None, type=str,
        help='List of filters that will be used to update acceptable filters.')
    parser.add_argument('--only-wide', default=False, action='store_true',
        help='Only reduce wide-band filters.')
    parser.add_argument('--keep-short', default=False, action='store_true',
        help='Keep image files that are shorter than 20 seconds.')
    parser.add_argument('--keep-indt', default=False, action='store_true',
        help='Keep images with EXPFLAG==INDETERMINATE.')
    parser.add_argument('--keep-tdf-down', default=False, action='store_true',
        help='Keep images with EXPFLAG==TDF-DOWN AT START.')
    parser.add_argument('--no-large-reduction', default=False,
        action='store_true', help='Exit if input list is >large_num images.')
    parser.add_argument('--large-num', default=200, type=int,
        help='Large number of images to skip when --no_large_reduction used.')
    
    # Reference image parameters
    parser.add_argument('--reference','--ref', default='',
        type=str, help='Name of the reference image.')
    parser.add_argument('--reference-filter','--ref-filter', default=None, type=str,
        help='Use this filter for the reference image if available.')
    parser.add_argument('--reference-instrument', default=None, type=str,
        help='Use this instrument for the reference image if available.')
    parser.add_argument('--avoid-wfpc2', default=False, action='store_true',
        help='Avoid using WFPC2 images as the reference image.')

    # Tweakreg options
    parser.add_argument('--tweak-search', default=None, type=float,
        help='Default search radius for tweakreg.')
    parser.add_argument('--tweak-min-obj', default=None, type=int,
        help='Default search radius for tweakreg.')
    parser.add_argument('--tweak-nbright', default=None, type=int,
        help='Default number of bright sources to try to use for alignment.')
    parser.add_argument('--tweak-thresh', default=None, type=float,
        help='Initial threshold for finding sources in tweakreg.')
    parser.add_argument('--keep-objfile', default=False, action='store_true',
        help='Keep the object file output from tweakreg.')
    parser.add_argument('--skip-tweakreg', default=False, action='store_true',
        help='Skip running tweakreg.')
    parser.add_argument('--hierarchical', default=False, action='store_true',
        help='Drizzle all visit/filter pairs then use them as basis to'+\
        ' perform alignment on the sub-frames.')
    parser.add_argument('--hierarchical-test', default=False, 
        action='store_true',
        help='Testing for hierarchical alignment mode so the script exits'+\
        ' after tweakreg alignment is performed on drz files.')

    # Drizzling options
    parser.add_argument('--drizzle-all', default=False, action='store_true',
        help='Drizzle all visit/filter pairs together.')
    parser.add_argument('--drizzle-add', default=None, type=str,
        help='Comma-separated list of images to add to the drizzled reference'+\
        ' image.  Use this to inject data from other instruments, filters, '+\
        'etc. if they would not be selected by pick_best_reference.')
    parser.add_argument('--drizzle-mask', default=None, type=str,
        help='Mask out pixels in a box around input mask coordinate in '+\
        'drizzled images but outside box in images from drizadd.')
    parser.add_argument('--object', default=None, type=str,
        help='Change the object name in all science files to value and '+\
        'use that value in the filenames for drizzled images.')
    parser.add_argument('--drizzle-dim', default=5200, type=int,
        help='Override the dimensions of drizzled images.')
    parser.add_argument('--drizzle-scale', default=None, type=float,
        help='Override the pixel scale of drizzled images (units are arcsec).')
    parser.add_argument('--sky-sub', default=False, action='store_true',
        help='Use sky subtraction in astrodrizzle.')
    parser.add_argument('--combine-type', default=None, type=str,
        help='Override astrodrizzle combine_type with input.')
    parser.add_argument('--wht-type', default='EXP', type=str,
        help='final_wht_type parameter for astrodrizzle.')
    parser.add_argument('--no-nan', default=False, action='store_true',
        help='Set nan values in drizzled images to median pixel value.')
    parser.add_argument('--redrizzle', default=False, action='store_true',
        help='Redrizzle all epochs/filters once the master reference image '+\
        'is created and all images are aligned to that frame.')
    parser.add_argument('--fix-zpt', default=None, type=float,
        help='Fix the zero point of drizzled images to input value '+\
        '(accounting for combined EXPTIME in header).')
    parser.add_argument('--no-rotation', default=False, action='store_true',
        help='When drizzling, do not rotate to PA=0 degrees but preserve '+\
        'the original position angle.')
    parser.add_argument('--no-mask', default=False, action='store_true',
        help='Do not add extra masking based on other input files.')
    
    # dolphot options
    parser.add_argument('--run-dolphot', default=False, action='store_true',
        help='Run dolphot as part of this hst123 run.')
    parser.add_argument('--align-only', default=False, action='store_true',
        help='Set AlignOnly=1 when running dolphot.')
    parser.add_argument('--dolphot','--dp', default='dp', type=str,
        help='Name of the dolphot output file.')
    parser.add_argument('--dolphot-lim', default=3.5, type=float,
        help='Detection threshold for sources detected by dolphot.')
    parser.add_argument('--fit-sky', default=None, type=int,
        help='Change the dolphot FitSky parameter to something other than 2.')
    parser.add_argument('--do-fake','--df', default=False,
        action='store_true', help='Run fake star injection into dolphot. '+\
        'Requires that dolphot has been run, and so files are taken from the '+\
        'parameters in dolphot output from the current directory rather than '+\
        'files derived from the current run.')
    parser.add_argument('--add-crmask', default=False, action='store_true',
        help='Add the cosmic ray mask to the image DQ mask for dolphot.')
    parser.add_argument('--include-all-splits', default=False, action='store_true',
        help='Include all split images in the dolphot parameter files.')
    
    # Scrape dolphot options
    parser.add_argument('--scrape-dolphot','--sd', default=False,
        action='store_true', help='Scrape photometry from the dolphot '+\
        'catalog from the input RA/Dec.')
    parser.add_argument('--scrape-all', default=False, action='store_true',
        help='Scrape all candidate counterparts within the scrape radius '+\
        '(default=2 pixels) and output to files dpXXX.phot where XXX is '+\
        'zero-padded integer for labeling sources in order of proximity to '+\
        'input coordinate.')
    parser.add_argument('--scrape-radius', default=None, type=float,
        help='Override the dolphot scrape radius (units are arcsec).')
    parser.add_argument('--no-cuts', default=False, action='store_true',
        help='Skip cuts to dolphot output file.')
    parser.add_argument('--brightest', default=False, action='store_true',
        help='Sort output source files by signal-to-noise in reference image.')

    
    return(parser)
