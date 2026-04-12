""" wfpc2_dgeo - Functions to convert WFPC2 DGEOFILE into D2IMFILE

"""
import os
import datetime

import astropy
from astropy.io import fits
import numpy as np

from stsci.tools import fileutil

import logging
logger = logging.getLogger("stwcs.updatewcs.apply_corrections")


def update_wfpc2_d2geofile(filename, fhdu=None):
    """
    Creates a D2IMFILE from the DGEOFILE for a WFPC2 image (input), and
    modifies the header to reflect the new usage.

    Parameters
    ----------
    filename: string
        Name of WFPC2 file to be processed.  This file will be updated
        to delete any reference to a DGEOFILE and add a D2IMFILE to replace
        that correction when running updatewcs.
    fhdu: object
        FITS object for WFPC2 image.  If user has already opened the WFPC2
        file, they can simply pass that FITS object in for direct processing.

    Returns
    -------
    d2imfile: string
        Name of D2IMFILE created from DGEOFILE.  The D2IMFILE keyword in the
        image header will be updated/added to point to this newly created file.

    """
    if isinstance(filename, fits.HDUList):
        fhdu = filename
        filename = fhdu.filename()
        close_fhdu = False
    else:
        fhdu = fileutil.openImage(filename, mode='update')
        close_fhdu = True

    dgeofile = fhdu['PRIMARY'].header.get('DGEOFILE', None)
    already_converted = dgeofile not in [None, "N/A", "", " "]
    if already_converted or 'ODGEOFIL' in fhdu['PRIMARY'].header:
        if not already_converted:
            dgeofile = fhdu['PRIMARY'].header.get('ODGEOFIL', None)
        logger.info('Converting DGEOFILE %s into D2IMFILE...' % dgeofile)
        rootname = filename[:filename.find('.fits')]
        d2imfile = convert_dgeo_to_d2im(dgeofile, rootname)
        fhdu['PRIMARY'].header['ODGEOFIL'] = dgeofile
        fhdu['PRIMARY'].header['DGEOFILE'] = 'N/A'
        fhdu['PRIMARY'].header['D2IMFILE'] = d2imfile
    else:
        d2imfile = None
        fhdu['PRIMARY'].header['DGEOFILE'] = 'N/A'
        if 'D2IMFILE' not in fhdu['PRIMARY'].header:
            fhdu['PRIMARY'].header['D2IMFILE'] = 'N/A'

    # Only close the file handle if opened in this function
    if close_fhdu:
        fhdu.close()

    # return the d2imfile name so that calling routine can keep
    # track of the new file created and delete it later if necessary
    # (multidrizzle clean=True mode of operation)
    return d2imfile


def convert_dgeo_to_d2im(dgeofile, output, clobber=True):
    """ Routine that converts the WFPC2 DGEOFILE into a D2IMFILE.
    """
    dgeo = fileutil.openImage(dgeofile)
    outname = output + '_d2im.fits'

    removeFileSafely(outname)
    data = np.array([dgeo['dy', 1].data[:, 0]])
    scihdu = fits.ImageHDU(data=data)
    dgeo.close()
    # add required keywords for D2IM header
    scihdu.header['EXTNAME'] = ('DY', 'Extension name')
    scihdu.header['EXTVER'] = (1, 'Extension version')
    fits_str = 'PYFITS Version ' + str(astropy.__version__)
    scihdu.header['ORIGIN'] = (fits_str, 'FITS file originator')
    scihdu.header['INHERIT'] = (False, 'Inherits global header')

    dnow = datetime.datetime.now()
    scihdu.header['DATE'] = (str(dnow).replace(' ', 'T'),
                             'Date FITS file was generated')

    scihdu.header['CRPIX1'] = (0, 'Distortion array reference pixel')
    scihdu.header['CDELT1'] = (1, 'Grid step size in first coordinate')
    scihdu.header['CRVAL1'] = (0, 'Image array pixel coordinate')
    scihdu.header['CRPIX2'] = (0, 'Distortion array reference pixel')
    scihdu.header['CDELT2'] = (1, 'Grid step size in second coordinate')
    scihdu.header['CRVAL2'] = (0, 'Image array pixel coordinate')

    phdu = fits.PrimaryHDU()
    phdu.header['INSTRUME'] = 'WFPC2'
    d2imhdu = fits.HDUList()
    d2imhdu.append(phdu)
    scihdu.header['DETECTOR'] = (1, 'CCD number of the detector: PC 1, WFC 2-4 ')
    d2imhdu.append(scihdu.copy())
    scihdu.header['EXTVER'] = (2, 'Extension version')
    scihdu.header['DETECTOR'] = (2, 'CCD number of the detector: PC 1, WFC 2-4 ')
    d2imhdu.append(scihdu.copy())
    scihdu.header['EXTVER'] = (3, 'Extension version')
    scihdu.header['DETECTOR'] = (3, 'CCD number of the detector: PC 1, WFC 2-4 ')
    d2imhdu.append(scihdu.copy())
    scihdu.header['EXTVER'] = (4, 'Extension version')
    scihdu.header['DETECTOR'] = (4, 'CCD number of the detector: PC 1, WFC 2-4 ')
    d2imhdu.append(scihdu.copy())
    d2imhdu.writeto(outname)
    d2imhdu.close()

    return outname


def removeFileSafely(filename, clobber=True):
    """ Delete the file specified, but only if it exists and clobber is True.
    """
    if filename is not None and filename.strip() != '':
        if os.path.exists(filename) and clobber: os.remove(filename)
