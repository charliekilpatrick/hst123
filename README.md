# hst123

An all-in-one script for downloading, registering, and drizzling HST images, running dolphot, and scraping data from dolphot catalogs.  This script is optimized to obtain photometry of point sources across multiple HST images.

## Description

hst123.py is a single script designed to be run in a working directory that contains your images.

If you use the --download flag, the script will automatically download HST image files assuming you provided a right ascension and declination.  Currently, the script only downloads public data (options for private data in a future update).  The script will then check to make sure that the input RA/Dec is in each image.  If you do not use the download flag, the script will reduce images in the current directory.

Currently, the script is capable of reducing all instrument and detector types supported by dolphot assuming they have the following file types:
    - WFPC2: c0m.fits, c1m.fits (requires both)
    - ACS/WFC: flc.fits
    - ACS/HRC: flt.fits
    - WFC3/UVIS: flc.fits
    - WFC3/IR: flt.fits

You can provide your own reference image (--reference), but hst123 functions best with a HST reference image.  If you do not provide one, hst123 will drizzle a reference image from the downloaded or input images.

When drizzling the reference image and during final image alignment, hst123 aligns the input images using drizzlepac.tweakreg. The alignment from this method is often suboptimal if the input images are not very deep or in an ultraviolet or narrow-band filter, resulting in too few sources for relative alignment.

dolphot parameters have been tuned for each HST instrument and detector and from the advice of Andrew Dolphin. It is not recommended that you adjust any of these parameters before running dolphot (--rundolphot).

Final photometry is scraped (--scrapedolphot) from the dolphot output and split into instruments, filters, and visits, where visits are defined by observation date/time with 0.5 day separations. The final photometry is then formatted into an output file (--photoutput) for convenience.  If you want photometry from each instrument and filter without separating by visit, use the --novisit/--nv flag.

## Options

Usage: hst123.py

Options:

  -h, --help<br/>            show this help message and exit

  --redo<br/>                Redo the hst123 reduction by re-copying files from the
                        raw dir.

  --makeclean, --mc<br/>     Clean up all output files from previous runs then
                        exit.

  --download<br/>            Download the raw data files given input ra and dec.

  --before=YYYY-MM-DD<br/>   Date after which we should reject all HST observations
                        for reduction.

  --after=YYYY-MM-DD<br/>    Date before which we should reject all HST
                        observations for reduction.

  --clobber<br/>             Overwrite files when using download mode.

  --ra=deg/HH:MM:SS<br/>     RA of interest.

  --dec=deg/DD:MM:SS<br/>    DEC of interest.

  --reference=ref.fits, --ref=ref.fits<br/>
                        Name of the reference image.

  --rundolphot, --rd<br/>    Run dolphot as part of this hst123 run.

  --maxdolphot=9999999, --mdp=9999999, --maxdp=9999999<br/>
                        Maximum number of images per dolphot run.

  --alignonly, --ao<br/>     When running dolphot, set AlignOnly=1 so dolphot stops
                        after image alignment.

  --dolphot=dp, --dp=dp<br/>
                        Name of the dolphot output file.

  --nocuts, --nc<br/>        Do not mask bad sources from the dolphot output
                        catalog before scraping data.

  --scrapedolphot, --sd<br/>
                        Scrape photometry from the dolphot catalog from the
                        input RA/Dec.

## Requirements

See requirements.txt.  The main requirements are numpy, astropy, drizzlepac, and astroquery.  hst123 also uses astroscrappy (based on LACosmic) for the cosmic ray rejection routine and stwcs for updating the geometric distortion terms in HST image headers.

hst123 also requires a complete installation of dolphot, including all instrument-specific modules and filter PSFs.  To obtain these files, visit: http://americano.dolphinsim.com/dolphot/.

## Contact

For all questions, comments, suggestions, and bugs related to this script, please contact Charlie Kilpatrick at cdkilpat@ucsc.edu.
