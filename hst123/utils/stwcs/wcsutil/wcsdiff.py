from astropy import wcs as pywcs
from collections import OrderedDict
from astropy.io import fits
from .headerlet import parse_filename
import numpy as np


def is_wcs_identical(scifile, file2, sciextlist, fextlist, scikey=" ",
                     file2key=" ", verbose=False):
    """
    Compares the WCS solution of 2 files.

    Parameters
    ----------
    scifile: string
             name of file1 (usually science file)
             IRAF style extension syntax is accepted as well
             for example scifile[1] or scifile[sci,1]
    file2:   string
             name of second file (for example headerlet)
    sciextlist - list
             a list of int or tuple ('SCI', 1), extensions in the first file
    fextlist - list
             a list of int or tuple ('SIPWCS', 1), extensions in the second file
    scikey:  string
             alternate WCS key in scifile
    file2key: string
             alternate WCS key in file2
    verbose: bool
             True: print to stdout

    Notes
    -----
    These can be 2 science observations or 2 headerlets
    or a science observation and a headerlet. The two files
    have the same WCS solution if the following are the same:

    - rootname/destim
    - primary WCS
    - SIP coefficients
    - NPOL distortion
    - D2IM correction

    """
    result = True
    diff = OrderedDict()
    fobj, fname, close_file = parse_filename(file2)
    sciobj, sciname, close_scifile = parse_filename(scifile)
    diff['file_names'] = [scifile, file2]
    if get_rootname(scifile) != get_rootname(file2):
        # logger.info('Rootnames do not match.')
        diff['rootname'] = ("%s: %s", "%s: %s") % (sciname, get_rootname(scifile), file2,
                                                   get_rootname(file2))
        result = False
    for i, j in zip(sciextlist, fextlist):
        w1 = pywcs.WCS(sciobj[i].header, sciobj, key=scikey)
        w2 = pywcs.WCS(fobj[j].header, fobj, key=file2key)
        diff['extension'] = [get_extname_extnum(sciobj[i]), get_extname_extnum(fobj[j])]
        if not np.allclose(w1.wcs.crval, w2.wcs.crval, rtol=10**(-7)):
            # logger.info('CRVALs do not match')
            diff['CRVAL'] = w1.wcs.crval, w2.wcs.crval
            result = False
        if not np.allclose(w1.wcs.crpix, w2.wcs.crpix, rtol=10**(-7)):
            # logger.info('CRPIX do not match')
            diff['CRPIX'] = w1.wcs.crpix, w2.wcs.crpix
            result = False
        if not np.allclose(w1.wcs.cd, w2.wcs.cd, rtol=10**(-7)):
            # logger.info('CDs do not match')
            diff['CD'] = w1.wcs.cd, w2.wcs.cd
            result = False
        if not (np.array(w1.wcs.ctype) == np.array(w2.wcs.ctype)).all():
            # logger.info('CTYPEs do not match')
            diff['CTYPE'] = w1.wcs.ctype, w2.wcs.ctype
            result = False
        if w1.sip or w2.sip:
            if (w2.sip and not w1.sip) or (w1.sip and not w2.sip):
                diff['sip'] = 'one sip extension is missing'
                result = False
            if not np.allclose(w1.sip.a, w2.sip.a, rtol=10**(-7)):
                diff['SIP_A'] = 'SIP_A differ'
                result = False
            if not np.allclose(w1.sip.b, w2.sip.b, rtol=10**(-7)):
                # logger.info('SIP coefficients do not match')
                diff['SIP_B'] = (w1.sip.b, w2.sip.b)
                result = False
        if w1.cpdis1 or w2.cpdis1:
            if w1.cpdis1 and not w2.cpdis1 or w2.cpdis1 and not w1.cpdis1:
                diff['CPDIS1'] = "CPDIS1 missing"
                result = False
            if w1.cpdis2 and not w2.cpdis2 or w2.cpdis2 and not w1.cpdis2:
                diff['CPDIS2'] = "CPDIS2 missing"
                result = False
            if not np.allclose(w1.cpdis1.data, w2.cpdis1.data, rtol=10**(-7)):
                # logger.info('NPOL distortions do not match')
                diff['CPDIS1_data'] = (w1.cpdis1.data, w2.cpdis1.data)
                result = False
            if not np.allclose(w1.cpdis2.data, w2.cpdis2.data, rtol=10**(-7)):
                # logger.info('NPOL distortions do not match')
                diff['CPDIS2_data'] = (w1.cpdis2.data, w2.cpdis2.data)
                result = False
        if w1.det2im1 or w2.det2im1:
            if w1.det2im1 and not w2.det2im1 or \
                    w2.det2im1 and not w1.det2im1:
                diff['DET2IM'] = "Det2im1 missing"
                result = False
            if not np.allclose(w1.det2im1.data, w2.det2im1.data, rtol=10**(-7)):
                # logger.info('Det2Im corrections do not match')
                diff['D2IM1_data'] = (w1.det2im1.data, w2.det2im1.data)
                result = False
        if w1.det2im2 or w2.det2im2:
            if w1.det2im2 and not w2.det2im2 or \
               w2.det2im2 and not w1.det2im2:
                diff['DET2IM2'] = "Det2im2 missing"
                result = False
            if not np.allclose(w1.det2im2.data, w2.det2im2.data, rtol=10**(-7)):
                # logger.info('Det2Im corrections do not match')
                diff['D2IM2_data'] = (w1.det2im2.data, w2.det2im2.data)
                result = False
    if not result and verbose:
        for key in diff:
            print(key, ":\t", diff[key][0], "\t", diff[key][1])
    if close_file:
        fobj.close()
    if close_scifile:
        sciobj.close()
    return result, diff


def get_rootname(fname):
    """
    Returns the value of ROOTNAME or DESTIM
    """

    hdr = fits.getheader(fname)
    try:
        rootname = hdr['ROOTNAME']
    except KeyError:
        try:
            rootname = hdr['DESTIM']
        except KeyError:
            rootname = fname
    return rootname


def get_extname_extnum(ext):
    """
    Return (EXTNAME, EXTNUM) of a FITS extension
    """
    extname = ""
    extnum = 1
    extname = ext.header.get('EXTNAME', extname)
    extnum = ext.header.get('EXTVER', extnum)
    return (extname, extnum)
