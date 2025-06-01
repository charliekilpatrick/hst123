# hst123

[![GitHub Actions](https://github.com/charliekilpatrick/hst123/actions/workflows/build-test.yml/badge.svg)](https://github.com/charliekilpatrick/hst123/actions/workflows/build-test.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://charliekilpatrick.github.io/hst123/)

An all-in-one script for downloading, registering, and drizzling HST images, running dolphot, and scraping data from dolphot catalogs.  This script is optimized to obtain photometry of point sources across multiple HST images.

## Installation

### Mac OS X

It is easiest to install hst123 dependencies using conda and pip:

```
conda create -n hst python=3.10 astropy pip astroquery astroscrappy numpy progressbar33 requests scipy
conda activate hst
pip install drizzlepac stwcs
```

On recent installations, I received a HDF5 error when installing the "tables" dependency of drizzlepac:

```
ERROR:: Could not find a local HDF5 installation.
You may need to explicitly state where your local HDF5 headers and
library can be found by setting the ``HDF5_DIR`` environment
variable or by using the ``--hdf5`` command-line option.
```

To solve this issue, use homebrew to install hdf5 and c-blosc (see: https://stackoverflow.com/questions/73029883/could-not-find-hdf5-installation-for-pytables-on-m1-mac):

```
pip install cython
brew install hdf5
brew install c-blosc
export HDF5_DIR=/opt/homebrew/opt/hdf5 
export BLOSC_DIR=/opt/homebrew/opt/c-blosc
```

Then re-run `pip install drizzlepac stwcs`.

### Linux

Follow the same instructions above with:

```
conda create -n hst python=3.10 astropy pip astroquery astroscrappy numpy progressbar33 requests scipy
conda activate hst
pip install drizzlepac stwcs
```

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

options:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   Use the input working directory rather than the
                        current dir.
  --make-clean          Clean up all output files from previous runs then
                        exit.
  --download            Download the raw data files given input ra and dec.
  --token TOKEN         Input a token for astroquery.mast.Observations.
  --archive ARCHIVE     Download and save raw data to an archive directory
                        instead of same folder as reduction (see
                        global_defaults['archive'])
  --no-clear-downloads  Suppress the clear_downloads method.
  --clobber             Overwrite files when using download mode.
  --cleanup             Clean up interstitial image files (i.e.,
                        flt,flc,c1m,c0m).
  --skip-copy           Skip copying files from archive if --archive is used.
  --by-visit            Reduce images by visit number.
  --before BEFORE       Reject obs after this date.
  --after AFTER         Reject obs before this date.
  --only-filter ONLY_FILTER
                        List of filters that will be used to update acceptable
                        filters.
  --only-wide           Only reduce wide-band filters.
  --keep-short          Keep image files that are shorter than 20 seconds.
  --keep-indt           Keep images with EXPFLAG==INDETERMINATE.
  --keep-tdf-down       Keep images with EXPFLAG==TDF-DOWN AT START.
  --no-large-reduction  Exit if input list is >large_num images.
  --large-num LARGE_NUM
                        Large number of images to skip when
                        --no_large_reduction used.
  --reference REFERENCE, --ref REFERENCE
                        Name of the reference image.
  --reference-filter REFERENCE_FILTER
                        Use this filter for the reference image if available.
  --reference-instrument REFERENCE_INSTRUMENT
                        Use this instrument for the reference image if
                        available.
  --avoid-wfpc2         Avoid using WFPC2 images as the reference image.
  --tweak-search TWEAK_SEARCH
                        Default search radius for tweakreg.
  --tweak-min-obj TWEAK_MIN_OBJ
                        Default search radius for tweakreg.
  --tweak-thresh TWEAK_THRESH
                        Initial threshold for finding sources in tweakreg.
  --keep-objfile        Keep the object file output from tweakreg.
  --skip-tweakreg       Skip running tweakreg.
  --hierarchical        Drizzle all visit/filter pairs then use them as basis
                        to perform alignment on the sub-frames.
  --hierarchical-test   Testing for hierarchical alignment mode so the script
                        exits after tweakreg alignment is performed on drz
                        files.
  --drizzle-all         Drizzle all visit/filter pairs together.
  --drizzle-add DRIZZLE_ADD
                        Comma-separated list of images to add to the drizzled
                        reference image. Use this to inject data from other
                        instruments, filters, etc. if they would not be
                        selected by pick_best_reference.
  --drizzle-mask DRIZZLE_MASK
                        Mask out pixels in a box around input mask coordinate
                        in drizzled images but outside box in images from
                        drizadd.
  --object OBJECT       Change the object name in all science files to value
                        and use that value in the filenames for drizzled
                        images.
  --drizzle-dim DRIZZLE_DIM
                        Override the dimensions of drizzled images.
  --drizzle-scale DRIZZLE_SCALE
                        Override the pixel scale of drizzled images (units are
                        arcsec).
  --sky-sub             Use sky subtraction in astrodrizzle.
  --combine-type COMBINE_TYPE
                        Override astrodrizzle combine_type with input.
  --wht-type WHT_TYPE   final_wht_type parameter for astrodrizzle.
  --no-nan              Set nan values in drizzled images to median pixel
                        value.
  --redrizzle           Redrizzle all epochs/filters once the master reference
                        image is created and all images are aligned to that
                        frame.
  --fix-zpt FIX_ZPT     Fix the zero point of drizzled images to input value
                        (accounting for combined EXPTIME in header).
  --no-rotation         When drizzling, do not rotate to PA=0 degrees but
                        preserve the original position angle.
  --no-mask             Do not add extra masking based on other input files.
  --run-dolphot         Run dolphot as part of this hst123 run.
  --align-only          Set AlignOnly=1 when running dolphot.
  --dolphot DOLPHOT, --dp DOLPHOT
                        Name of the dolphot output file.
  --dolphot-lim DOLPHOT_LIM
                        Detection threshold for sources detected by dolphot.
  --fit-sky FIT_SKY     Change the dolphot FitSky parameter to something other
                        than 2.
  --do-fake, --df       Run fake star injection into dolphot. Requires that
                        dolphot has been run, and so files are taken from the
                        parameters in dolphot output from the current
                        directory rather than files derived from the current
                        run.
  --add-crmask          Add the cosmic ray mask to the image DQ mask for
                        dolphot.
  --scrape-dolphot, --sd
                        Scrape photometry from the dolphot catalog from the
                        input RA/Dec.
  --scrape-all          Scrape all candidate counterparts within the scrape
                        radius (default=2 pixels) and output to files
                        dpXXX.phot where XXX is zero-padded integer for
                        labeling sources in order of proximity to input
                        coordinate.
  --scrape-radius SCRAPE_RADIUS
                        Override the dolphot scrape radius (units are arcsec).
  --no-cuts             Skip cuts to dolphot output file.
  --brightest           Sort output source files by signal-to-noise in
                        reference image.
```

## External dependencies

hst123 requires a complete installation of dolphot to run PSF photometry, including all instrument-specific modules and filter PSFs.  To obtain these files, visit: http://americano.dolphinsim.com/dolphot/.

## Contact

For all questions, comments, suggestions, and bugs related to this script, please contact Charlie Kilpatrick at ckilpatrick@northwestern.edu.
