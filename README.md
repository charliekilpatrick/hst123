# hst123

An all-in-one script for downloading, registering, and drizzling HST images, running dolphot, and scraping data from dolphot catalogs.  This script is optimized to obtain photometry of point sources across multiple HST images.

## Description

hst123.py is a single script designed to be run in a working directory that contains your images.

If you use the `--download` flag, the script will automatically download publicly-available HST image files at the input right ascension and declination.  Use the `--token` command-line argument with your MAST authorization token (see: https://auth.mast.stsci.edu/info) to download private files only available to you.  The script downloads all files within a radius of 5 arcminutes but only reduces images where the input coordinate is inside the image.  If you do not use the download flag, the script will reduce images in the current directory.

Currently, the script is capable of reducing all instrument and detector types supported by dolphot assuming they have the following file types:

```
WFPC2: c0m.fits, c1m.fits (requires both)
ACS/WFC: flc.fits
ACS/HRC: flt.fits
WFC3/UVIS: flc.fits
WFC3/IR: flt.fits
```

You can provide your own reference image (`--reference`), but hst123 functions best with a HST reference image.  If you do not provide one, hst123 will drizzle a reference image from the downloaded or input images.

When drizzling the reference image and during final image alignment, hst123 aligns the input images using drizzlepac.tweakreg. The alignment from this method is often suboptimal if the input images are not very deep or in an ultraviolet or narrow-band filter, resulting in too few sources for relative alignment.

dolphot parameters have been tuned for each HST instrument and detector and from the advice of Andrew Dolphin. It is not recommended that you adjust any of these parameters before running dolphot (`--rundolphot`).

Final photometry is scraped (`--scrapedolphot`) from the dolphot output and split into instruments, filters, and visits, where visits are defined by observation date/time with 0.5 day separations. The final photometry is then formatted into an output file (`--photoutput`) for convenience.  If you want photometry from each instrument and filter without separating by visit, use the `--novisit` flag.

## Options

```
usage: hst123.py ra dec

positional arguments:
  ra                    Right ascension to reduce the HST images
  dec                   Declination to reduce the HST images

optional arguments:
  -h, --help            show this help message and exit
  --makeclean           Clean up all output files from previous runs then
                        exit.
  --download            Download the raw data files given input ra and dec.
  --keepshort           Keep image files that are shorter than 20 seconds.
  --before BEFORE       Reject obs after this date.
  --after AFTER         Reject obs before this date.
  --clobber             Overwrite files when using download mode.
  --reference REFERENCE, --ref REFERENCE
                        Name of the reference image.
  --rundolphot          Run dolphot as part of this hst123 run.
  --alignonly           Set AlignOnly=1 when running dolphot.
  --dolphot DOLPHOT, --dp DOLPHOT
                        Name of the dolphot output file.
  --scrapedolphot, --sd
                        Scrape photometry from the dolphot catalog from the
                        input RA/Dec.
  --reffilter REFFILTER
                        Use this filter for the reference image if available.
  --avoidwfpc2          Avoid using WFPC2 images as the reference image.
  --refinst REFINST     Use this instrument for the reference image if
                        available.
  --dofake, --df        Run fake star injection into dolphot. Requires that
                        dolphot has been run, and so files are taken from the
                        parameters in dolphot output from the current
                        directory rather than files derived from the current
                        run.
  --cleanup             Clean up interstitial image files (i.e.,
                        flt,flc,c1m,c0m).
  --drizzleall          Drizzle all visit/filter pairs together.
  --hierarchical        Drizzle all visit/filter pairs then use them as basis
                        to perform alignment on the sub-frames.
  --hierarch_test       Testing for hierarchical alignment mode so the script
                        exits after tweakreg alignment is performed on drz
                        files.
  --object OBJECT       Change the object name in all science files to value
                        and use that value in the filenames for drizzled
                        images.
  --archive ARCHIVE     Download and save raw data to an archive directory
                        instead of same folder as reduction (see
                        global_defaults['archive'])
  --workdir WORKDIR     Use the input working directory rather than the
                        current dir
  --keep_objfile        Keep the object file output from tweakreg.
  --skip_tweakreg       Skip running tweakreg.
  --drizdim DRIZDIM     Override the dimensions of drizzled images.
  --drizscale DRIZSCALE
                        Override the pixel scale of drizzled images (units are
                        arcsec).
  --scrapeall           Scrape all candidate counterparts within the scrape
                        radius (default=2 pixels) and output to files
                        dpXXX.phot where XXX is zero-padded integer for
                        labeling sources in order of proximity to input
                        coordinate.
  --token TOKEN         Input a token for astroquery.mast.Observations.
  --scraperadius SCRAPERADIUS
                        Override the dolphot scrape radius (units are arcsec).
  --nocuts              Skip cuts to dolphot output file.
  --onlywide            Only reduce wide-band filters.
  --brightest           Sort output source files by signal-to-noise in
                        reference image.
  --no_large_reduction  Exit if input list is >large_num images.
  --large_num LARGE_NUM
                        Large number of images to skip when
                        --no_large_reduction used.
  --combine_type COMBINE_TYPE
                        Override astrodrizzle combine_type with input.
  --sky_sub             Use sky subtraction in astrodrizzle.
  --wht_type WHT_TYPE   final_wht_type parameter for astrodrizzle.
  --drizadd DRIZADD     Comma-separated list of images to add to the drizzled
                        reference image. Use this to inject data from other
                        instruments, filters, etc. if they would not be
                        selected by pick_best_reference.
  --drizmask DRIZMASK   Mask out pixels in a box around input mask coordinate
                        in drizzled images but outside box in images from
                        drizadd.
  --no_clear_downloads  Suppress the clear_downloads method.
  --fixzpt FIXZPT       Fix the zero point of drizzled images to input value
                        (accounting for combined EXPTIME in header).
  --no_nan              Set nan values in drizzled images to median pixel
                        value.
  --skip_copy           Skip copying files from archive if --archive is used.
  --byvisit             Reduce images by visit number.
  --no_rotation         When drizzling, do not rotate to PA=0 degrees but
                        preserve the original position angle.
  --only_filter ONLY_FILTER
                        List of filters that will be used to update acceptable
                        filters.
  --fitsky FITSKY       Change the dolphot FitSky parameter to something other
                        than 2.
  --no_mask             Do not add extra masking based on other input files.
  --keep_indt           Keep images with EXPFLAG==INDETERMINATE.
  --keep_tdf_down       Keep images with EXPFLAG==TDF-DOWN AT START.
  --tweak_thresh TWEAK_THRESH
                        Initial threshold for finding sources in tweakreg.
  --add_crmask          Add the cosmic ray mask to the image DQ mask for
                        dolphot.
  --redrizzle           Redrizzle all epochs/filters once the master reference
                        image is created and all images are aligned to that
                        frame.
  --dolphot_lim DOLPHOT_LIM
                        Detection threshold for sources detected by dolphot.
  --tweak_search TWEAK_SEARCH
                        Default search radius for tweakreg.
  --tweak_min_obj TWEAK_MIN_OBJ
                        Default search radius for tweakreg.
```

## Requirements

See requirements.txt.  The main requirements are numpy, astropy, drizzlepac, and astroquery.  hst123 also uses astroscrappy (based on LACosmic) for the cosmic ray rejection routine and stwcs for updating the geometric distortion terms in HST image headers.

hst123 also requires a complete installation of dolphot, including all instrument-specific modules and filter PSFs.  To obtain these files, visit: http://americano.dolphinsim.com/dolphot/.

## Contact

For all questions, comments, suggestions, and bugs related to this script, please contact Charlie Kilpatrick at ckilpatrick@northwestern.edu.
