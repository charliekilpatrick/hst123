import os.path

from astropy.io import fits
from stsci.tools import fileutil
from . import utils
from . import wfpc2_dgeo

import logging
logger = logging.getLogger("stwcs.updatewcs.apply_corrections")

# Note: The order of corrections is important


# A dictionary which lists the allowed corrections for each instrument.
# These are the default corrections applied also in the pipeline.

allowed_corrections = {'WFPC2': ['DET2IMCorr', 'MakeWCS', 'CompSIP', 'VACorr'],
                       'ACS': ['DET2IMCorr', 'TDDCorr', 'MakeWCS', 'CompSIP', 'VACorr', 'NPOLCorr'],
                       'STIS': ['MakeWCS', 'CompSIP', 'VACorr'],
                       'NICMOS': ['MakeWCS', 'CompSIP', 'VACorr'],
                       'WFC3': ['DET2IMCorr', 'MakeWCS', 'CompSIP', 'VACorr', 'NPOLCorr'],
                       }

cnames = {'DET2IMCorr': 'Detector to Image Correction',
          'TDDCorr': 'Time Dependent Distortion Correction',
          'MakeWCS': 'Recalculate basic WCS keywords based on the distortion model',
          'CompSIP': 'Given IDCTAB distortion model calculate the SIP coefficients',
          'VACorr': 'Velocity Aberration Correction',
          'NPOLCorr': 'Lookup Table Distortion'
          }


def setCorrections(fname, vacorr=True, tddcorr=True, npolcorr=True, d2imcorr=True):
    """
    Creates a list of corrections to be applied to a file
    based on user input paramters and allowed corrections
    for the instrument.

    Parameters
    ----------
    fname : str or `~astropy.io.fits.HDUList`
        Input file object or file name
    vacorr : bool
        Boolean flag indicating whether velocity aberration correction is requested.
    tddcorr : bool
        Boolean flag indicating whether time dependent distortion correction is requested.
    npolcorr : bool
        Boolean flag indicating whether time non-polynomial distortion correction is requested.
    d2imcorr : bool
        Boolean flag indicating whether d2im correction is requested.
    """
    fname, toclose = _toclose(fname)

    instrument = fname[0].header['INSTRUME']

    # make a copy of this list
    acorr = allowed_corrections[instrument][:]

    # For WFPC2 images, the old-style DGEOFILE needs to be
    # converted on-the-fly into a proper D2IMFILE here...
    if instrument == 'WFPC2':
        # check for DGEOFILE, and convert it to D2IMFILE if found
        _ = wfpc2_dgeo.update_wfpc2_d2geofile(fname)
    # Check if idctab is present on disk
    # If kw IDCTAB is present in the header but the file is
    # not found on disk, do not run TDDCorr, MakeCWS and CompSIP
    if not foundIDCTAB(fname):
        if 'TDDCorr' in acorr: acorr.remove('TDDCorr')
        if 'MakeWCS' in acorr: acorr.remove('MakeWCS')
        if 'CompSIP' in acorr: acorr.remove('CompSIP')

    if 'VACorr' in acorr and not vacorr:
        acorr.remove('VACorr')
    if 'TDDCorr' in acorr:
        tddcorr = applyTDDCorr(fname, tddcorr)
        if not tddcorr:
            acorr.remove('TDDCorr')

    if 'NPOLCorr' in acorr:
        npolcorr = applyNpolCorr(fname, npolcorr)
        if not npolcorr:
            acorr.remove('NPOLCorr')
    if 'DET2IMCorr' in acorr:
        d2imcorr = apply_d2im_correction(fname, d2imcorr)
        if not d2imcorr:
            acorr.remove('DET2IMCorr')
    logger.info("Corrections to be applied to {0} {1}".format(fname, acorr))
    if toclose:
        fname.close()
    return acorr


def foundIDCTAB(fname):
    """
    This functions looks for an "IDCTAB" keyword in the primary header.

    Parameters
    ----------
    fname : `~astropy.io.fits.HDUList`
        Input FITS file object.

    Returns
    -------
    status : bool
        If False : MakeWCS, CompSIP and TDDCorr should not be applied.
        If True : there's no restriction on corrections, they all should be applied.

    Raises
    ------
    IOError : If IDCTAB file not found on disk.
    """

    try:
        idctab = fname[0].header['IDCTAB'].strip()
        if idctab == 'N/A' or idctab == "":
            return False
    except KeyError:
        return False
    idctab = fileutil.osfn(idctab)
    if os.path.exists(idctab):
        return True
    else:
        raise IOError("IDCTAB file {0} not found".format(idctab))


def applyTDDCorr(fname, utddcorr):
    """
    The default value of tddcorr for all ACS images is True.
    This correction will be performed if all conditions below are True:
    - the user did not turn it off on the command line
    - the detector is WFC
    - the idc table specified in the primary header is available.

    Parameters
    ----------
    fname : `~astropy.io.fits.HDUList`
        Input FITS file object.

    """

    phdr = fname[0].header
    instrument = phdr['INSTRUME']
    try:
        detector = phdr['DETECTOR']
    except KeyError:
        detector = None
    try:
        tddswitch = phdr['TDDCORR']
    except KeyError:
        tddswitch = 'PERFORM'

    if instrument == 'ACS' and detector == 'WFC' and utddcorr and tddswitch == 'PERFORM':
        tddcorr = True
        try:
            idctab = phdr['IDCTAB']
        except KeyError:
            tddcorr = False
        if os.path.exists(fileutil.osfn(idctab)):
            tddcorr = True
        else:
            tddcorr = False
    else:
        tddcorr = False
    return tddcorr


def applyNpolCorr(fname, unpolcorr):
    """
    Determines whether non-polynomial distortion lookup tables should be added
    as extensions to the science file based on the 'NPOLFILE' keyword in the
    primary header and NPOLEXT kw in the first extension.
    This is a default correction and will always run in the pipeline.
    The file used to generate the extensions is
    recorded in the NPOLEXT keyword in the first science extension.
    If 'NPOLFILE' in the primary header is different from 'NPOLEXT' in the
    extension header and the file exists on disk and is a 'new type' npolfile,
    then the lookup tables will be updated as 'WCSDVARR' extensions.

    Parameters
    ----------
    fname : `~astropy.io.fits.HDUList`
        Input FITS file object.

    """
    applyNPOLCorr = True
    try:
        # get NPOLFILE kw from primary header
        fnpol0 = fname[0].header['NPOLFILE']
        if fnpol0 == 'N/A':
            utils.remove_distortion(fname, "NPOLFILE")
            return False
        fnpol0 = fileutil.osfn(fnpol0)
        if not fileutil.findFile(fnpol0):
            msg = (
                '"NPOLFILE" exists in primary header but file {0} not found. '
                "Non-polynomial distortion correction will not be applied."
            ).format(fnpol0)
            logger.critical(msg)
            raise IOError("NPOLFILE {0} not found".format(fnpol0))
        try:
            # get NPOLEXT kw from first extension header
            fnpol1 = fname[1].header['NPOLEXT']
            fnpol1 = fileutil.osfn(fnpol1)
            if fnpol1 and fileutil.findFile(fnpol1):
                if fnpol0 != fnpol1:
                    applyNPOLCorr = True
                else:
                    msg = (
                        "NPOLEXT with the same value as NPOLFILE found in first "
                        "extension; NPOL correction will not be applied."
                    )
                    logger.info(msg)
                    applyNPOLCorr = False
            else:
                # npl file defined in first extension may not be found
                # but if a valid kw exists in the primary header, non-polynomial
                # distortion correction should be applied.
                applyNPOLCorr = True
        except KeyError:
            # the case of "NPOLFILE" kw present in primary header but "NPOLEXT" missing
            # in first extension header
            applyNPOLCorr = True
    except KeyError:
        logger.info('"NPOLFILE" keyword not found in primary header')
        applyNPOLCorr = False
        return applyNPOLCorr

    if isOldStyleDGEO(fname, fnpol0):
            applyNPOLCorr = False
    return (applyNPOLCorr and unpolcorr)


def isOldStyleDGEO(fname, dgname):
    """
    Checks if the file defined in a NPOLFILE kw is a full size
    (old style) image.

    Parameters
    ----------
    fname : `~astropy.io.fits.HDUList`
        Input FITS file object.
    dgname : str
        Name of NPOL file.
    """

    sci_hdr = fname[1].header
    dgeo_hdr = fits.getheader(dgname, ext=1)
    sci_naxis1 = sci_hdr['NAXIS1']
    sci_naxis2 = sci_hdr['NAXIS2']
    dg_naxis1 = dgeo_hdr['NAXIS1']
    dg_naxis2 = dgeo_hdr['NAXIS2']
    if sci_naxis1 <= dg_naxis1 or sci_naxis2 <= dg_naxis2:
        msg = (
            "Only full size (old style) DGEO file was found; "
            "non-polynomial distortion correction will not be applied."
        )
        logger.critical(msg)
        return True
    else:
        return False


def apply_d2im_correction(fname, d2imcorr):
    """
    Logic to decide whether to apply the D2IM correction.

    Parameters
    ----------
    fname : `~astropy.io.fits.HDUList` or str
        Input FITS science file object.
    d2imcorr : bool
        Flag indicating if D2IM is should be enabled if allowed.

    Return
    ------
    applyD2IMCorr : bool
        Flag whether to apply the correction.

    The D2IM correction is applied to a science file if it is in the
    allowed corrections for the instrument. The name of the file
    with the correction is saved in the ``D2IMFILE`` keyword in the
    primary header. When the correction is applied the name of the
    file is saved in the ``D2IMEXT`` keyword in the 1st extension header.

    """
    fname, toclose = _toclose(fname)

    applyD2IMCorr = True
    if not d2imcorr:
        logger.info("D2IM correction not requested - not applying it.")
        return False
    # get D2IMFILE kw from primary header
    try:
        fd2im0 = fname[0].header['D2IMFILE']
    except KeyError:
        logger.info("D2IMFILE keyword is missing - D2IM correction will not be applied.")
        return False
    if fd2im0 == 'N/A':
        utils.remove_distortion(fname, "D2IMFILE")
        return False
    fd2im0 = fileutil.osfn(fd2im0)
    if not fileutil.findFile(fd2im0):
        message = "D2IMFILE {0} not found.".format(fd2im0)
        logger.critical(message)
        raise IOError(message)
    try:
        # get D2IMEXT kw from first extension header
        fd2imext = fname[1].header['D2IMEXT']

    except KeyError:
        # the case of D2IMFILE kw present in primary header but D2IMEXT missing
        # in first extension header
        return True
    fd2imext = fileutil.osfn(fd2imext)
    if fd2imext and fileutil.findFile(fd2imext):
        if fd2im0 != fd2imext:
            applyD2IMCorr = True
        else:
            applyD2IMCorr = False
    else:
        # D2IM file defined in first extension may not be found
        # but if a valid kw exists in the primary header,
        # detector to image correction should be applied.
        applyD2IMCorr = True
    if toclose:
        fname.close()
    return applyD2IMCorr


def _toclose(input, mode='readonly'):
    toclose = False
    if isinstance(input, str):
        toclose = True
        input = fits.open(input, mode=mode)
    return input, toclose