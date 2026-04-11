"""
This module implements headerlets.

A headerlet serves as a mechanism for encapsulating WCS information
which can be used to update the WCS solution of an image. The idea
came up first from the desire for passing improved astrometric
solutions for HST data and provide those solutions in a manner
that would not require getting entirely new images from the archive
when only the WCS information has been updated.

NOTE ::
  This module defines a FileHandler for the logging in the current
  working directory for the user when this module first gets imported.
  If that directory is removed later by the user, it will cause an
  Exception when performing headerlet operations later.

  The file handler can be identified using::

    rl = logging.getLogger('stwcs.wcsutil.headerlet')
    rl.handlers
    del rl.handlers[-1]  # if FileHandler was the last one, remove it

"""
import os
import sys
import functools
import logging
import textwrap
import copy
import time

import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs as pywcs
from astropy.utils import lazyproperty

from stsci.tools.fileutil import countExtn
from stsci.tools import fileutil as fu
from stsci.tools import parseinput

from . import altwcs
from . import wcscorr
from .hstwcs import HSTWCS
from ..updatewcs import utils
from .mappings import basic_wcs

"""
``clobber`` parameter in `astropy.io.fits.writeto()`` was renamed to
``overwrite`` in astropy v1.3.
"""
from astropy.utils import minversion
ASTROPY_13_MIN = minversion(astropy, "1.3")

from astropy import log
default_log_level = log.getEffectiveLevel()

# Logging support functions

class FuncNameLoggingFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        if '%(funcName)s' not in fmt:
            fmt = '%(funcName)s' + fmt
        logging.Formatter.__init__(self, fmt=fmt, datefmt=datefmt)

    def format(self, record):
        record = copy.copy(record)
        if hasattr(record, 'funcName') and record.funcName == 'init_logging':
            record.funcName = ''
        else:
            record.funcName += ' '
        return logging.Formatter.format(self, record)


logger = logging.getLogger(__name__)
formatter = FuncNameLoggingFormatter("%(levelname)s: %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.CRITICAL)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

FITS_STD_KW = ['XTENSION', 'BITPIX', 'NAXIS', 'PCOUNT',
               'GCOUNT', 'EXTNAME', 'EXTVER', 'ORIGIN',
               'INHERIT', 'DATE', 'IRAF-TLM']
DISTORTION_KEYWORDS = ['NPOLFILE', 'IDCTAB', 'D2IMFILE', 'SIPNAME', 'DISTNAME']

DEFAULT_SUMMARY_COLS = ['HDRNAME', 'WCSNAME', 'DISTNAME', 'AUTHOR', 'DATE',
                        'SIPNAME', 'NPOLFILE', 'D2IMFILE', 'DESCRIP']
COLUMN_DICT = {'vals': [], 'width': []}
COLUMN_FMT = '{:<{width}}'


def init_logging(funcname=None, level=100, mode='w', **kwargs):
    """

    Initialize logging for a function

    Parameters
    ----------
    funcname: string
            Name of function which will be recorded in log
    level:  int, or bool, or string
            int or string : Logging level
            bool: False - switch off logging
            Text logging level for the message ("DEBUG", "INFO",
            "WARNING", "ERROR", "CRITICAL")
    mode: 'w' or 'a'
            attach to logfile ('a' or start a new logfile ('w')

    """
    for hndl in logger.handlers:
        if isinstance(hndl, logging.FileHandler):
            has_file_handler = True
        else:
            has_file_handler = False
    if level:
        if not has_file_handler:
            logname = 'headerlet.log'
            fh = logging.FileHandler(logname, mode=mode)
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        logger.info("%s: Starting %s with arguments:\n\t %s" %
                    (time.asctime(), funcname, kwargs))


def with_logging(func):
    @functools.wraps(func)
    def wrapped(*args, **kw):
        level = kw.get('logging', 100)
        mode = kw.get('logmode', 'w')
        func_args = kw.copy()
        if sys.version_info[0] >= 3:
            argnames = func.__code__.co_varnames
        else:
            argnames = func.func_code.co_varnames

        for argname, arg in zip(argnames, args):
            func_args[argname] = arg

        init_logging(func.__name__, level, mode, **func_args)
        return func(*args, **kw)
    return wrapped

# Utility functions


def is_par_blank(par):
    return par in ['', ' ', 'INDEF', "None", None]


def parse_filename(fname, mode='readonly'):
    """
    Interprets the input as either a filename of a file that needs to be opened
    or a PyFITS object.

    Parameters
    ----------
    fname : str, `astropy.io.fits.HDUList`
        Input pointing to a file or `astropy.io.fits.HDUList` object.
        An input filename (str) will be expanded as necessary to
        interpret any environmental variables
        included in the filename.

    mode : string
        Specifies what mode to use when opening the file, if it needs
        to open the file at all [Default: 'readonly']

    Returns
    -------
    fobj : `astropy.io.fits.HDUList`
        FITS file handle for input

    fname : str
        Name of input file

    close_fobj : bool
        Flag specifying whether or not fobj needs to be closed since it was
        opened by this function. This allows a program to know whether they
        need to worry about closing the FITS object as opposed to letting
        the higher level interface close the object.

    """
    if isinstance(fname, str):
        fname = fu.osfn(fname)
        fobj = fits.open(fname, mode=mode)
        close_fobj = True
    elif isinstance(fname, list):
        fobj = fname
        if hasattr(fobj, 'filename'):
            fname = fobj.filename()
        else:
            fname = ''
        close_fobj = False
    else:
        raise ValueError(f"parse_filename expects a file name or HDUList, got {type(fname)}")
    return fobj, fname, close_fobj


def get_headerlet_kw_names(fobj, kw='HDRNAME'):
    """
    Returns a list of specified keywords from all HeaderletHDU
    extensions in a science file.

    Parameters
    ----------
    fobj : str, `astropy.io.fits.HDUList`
    kw : str
        Name of keyword to be read and reported
    """

    fobj, fname, open_fobj = parse_filename(fobj)

    hdrnames = []
    for ext in fobj:
        if isinstance(ext, fits.hdu.base.NonstandardExtHDU):
            hdrnames.append(ext.header[kw])

    if open_fobj:
        fobj.close()

    return hdrnames


def get_header_kw_vals(hdr, kwname, kwval, default=0):
    if kwval is None:
        if kwname in hdr:
            kwval = hdr[kwname]
        else:
            kwval = default
    return kwval


@with_logging
def find_headerlet_HDUs(fobj, hdrext=None, hdrname=None, distname=None,
                        strict=True, logging=False, logmode='w'):
    """
    Returns all HeaderletHDU extensions in a science file that matches
    the inputs specified by the user.  If no hdrext, hdrname or distname are
    specified, this function will return a list of all HeaderletHDU objects.

    Parameters
    ----------
    fobj : str, `astropy.io.fits.HDUList`
        Name of FITS file or open fits object (`astropy.io.fits.HDUList` instance)
    hdrext : int, tuple or None
        index number(EXTVER) or extension tuple of HeaderletHDU to be returned
    hdrname : string
        value of HDRNAME for HeaderletHDU to be returned
    distname : string
        value of DISTNAME for HeaderletHDUs to be returned
    strict : bool [Default: True]
        Specifies whether or not at least one parameter needs to be provided
        If False, all extension indices returned if hdrext, hdrname and distname
        are all None. If True and hdrext, hdrname, and distname are all None,
        raise an Exception requiring one to be specified.
    logging : bool
             enable logging to a file called headerlet.log
    logmode : 'w' or 'a'
             log file open mode

    Returns
    -------
    hdrlets : list
        A list of all matching HeaderletHDU extension indices (could be just one)

    """

    get_all = False
    if hdrext is None and hdrname is None and distname is None:
        if not strict:
            get_all = True
        else:
            mess = """\n
            =====================================================
            No valid Headerlet extension specified.
            Either "hdrname", "hdrext", or "distname" needs to be specified.
            =====================================================
            """
            logger.critical(mess)
            raise ValueError

    fobj, fname, open_fobj = parse_filename(fobj)

    hdrlets = []
    if hdrext is not None and isinstance(hdrext, int):
        if hdrext in range(len(fobj)):  # insure specified hdrext is in fobj
            if isinstance(fobj[hdrext], fits.hdu.base.NonstandardExtHDU) and \
                    fobj[hdrext].header['EXTNAME'] == 'HDRLET':
                hdrlets.append(hdrext)
    else:
        for ext in fobj:
            if isinstance(ext, fits.hdu.base.NonstandardExtHDU):
                if get_all:
                    hdrlets.append(fobj.index(ext))
                else:
                    if hdrext is not None:
                        if isinstance(hdrext, tuple):
                            hdrextname = hdrext[0]
                            hdrextnum = hdrext[1]
                        else:
                            hdrextname = 'HDRLET'
                            hdrextnum = hdrext
                    hdrext_match = ((hdrext is not None) and
                                    (hdrextnum == ext.header['EXTVER']) and
                                    (hdrextname == ext.header['EXTNAME']))
                    hdrname_match = ((hdrname is not None) and
                                     (hdrname == ext.header['HDRNAME']))
                    distname_match = ((distname is not None) and
                                      (distname == ext.header['DISTNAME']))
                    if hdrext_match or hdrname_match or distname_match:
                        hdrlets.append(fobj.index(ext))

    if open_fobj:
        fobj.close()

    if len(hdrlets) == 0:
        if hdrname:
            kwerr = 'hdrname'
            kwval = hdrname
        elif hdrext:
            kwerr = 'hdrext'
            kwval = hdrext
        else:
            kwerr = 'distname'
            kwval = distname
        message = f"""\n
        =======================================
        No valid Headerlet extension found!
        {kwerr} = {kwval} not found in {fname}.
        =======================================
        """
        logger.critical(message)
        raise ValueError

    return hdrlets


def verify_hdrname_is_unique(fobj, hdrname):
    """
    Verifies that no other HeaderletHDU extension has the specified hdrname.

    Parameters
    ----------
    fobj : str, `astropy.io.fits.HDUList`
        Name of FITS file or open fits file object
    hdrname : str
        value of HDRNAME for HeaderletHDU to be compared as unique

    Returns
    -------
    unique: bool
        If True, no other HeaderletHDU has the specified HDRNAME value
    """
    hdrnames_list = get_headerlet_kw_names(fobj)
    unique = not(hdrname in hdrnames_list)

    return unique


def update_versions(sourcehdr, desthdr):
    """
    Update keywords which store version numbers
    """
    phdukw = {'PYWCSVER': 'Version of PYWCS used to updated the WCS',
              'UPWCSVER': 'Version of STWCS used to updated the WCS'
              }
    for key in phdukw:
        try:
            desthdr[key] = (sourcehdr[key], sourcehdr.comments[key])
        except KeyError:
            desthdr[key] = (" ", phdukw[key])


def update_ref_files(source, dest):
    """
    Update the reference files name in the primary header of 'dest'
    using values from 'source'

    Parameters
    ----------
    source : `astropy.io.fits.Header`
    dest :   `astropy.io.fits.Header`
    """
    logger.info("Updating reference files")
    phdukw = {}

    for key in DISTORTION_KEYWORDS:
        if key in dest:
            dest.set(key, source[key], source.comments[key])
            phdukw[key] = True
        else:
            phdukw[key] = False

    return phdukw


def print_summary(summary_cols, summary_dict, pad=2, maxwidth=None, idcol=None,
                  output=None, clobber=True, quiet=False ):
    """
    Print out summary dictionary to STDOUT, and possibly an output file

    """
    nrows = None
    if idcol:
        nrows = len(idcol['vals'])

    # Find max width of each column
    column_widths = {}
    for kw in summary_dict:
        colwidth = np.array(summary_dict[kw]['width']).max()
        if maxwidth:
            colwidth = min(colwidth, maxwidth)
        column_widths[kw] = colwidth + pad
        if nrows is None:
            nrows = len(summary_dict[kw]['vals'])

    # print rows now
    outstr = ''
    # Start with column names
    if idcol:
        outstr += COLUMN_FMT.format(idcol['name'], width=idcol['width'] + pad)
    for kw in summary_cols:
        outstr += COLUMN_FMT.format(kw, width=column_widths[kw])
    outstr += '\n'
    # Now, add a row for each headerlet
    for row in range(nrows):
        if idcol:
            outstr += COLUMN_FMT.format(idcol['vals'][row],
                                        width=idcol['width'] + pad)
        for kw in summary_cols:
            val = summary_dict[kw]['vals'][row][:(column_widths[kw] - pad)]
            outstr += COLUMN_FMT.format(val, width=column_widths[kw])
        outstr += '\n'
    if not quiet:
        print(outstr)

    # If specified, write info to separate text file
    write_file = False
    if output:
        output = fu.osfn(output)  # Expand any environment variables in filename
        write_file = True
        if os.path.exists(output):
            if clobber:
                os.remove(output)
            else:
                print('WARNING: Not writing results to file!')
                print('         Output text file ', output, ' already exists.')
                print('         Set "clobber" to True or move file before trying again.')
                write_file = False
        if write_file:
            fout = open(output, mode='w')
            fout.write(outstr)
            fout.close()

# Private utility functions


def _create_primary_HDU(fobj, fname, wcsext, destim, hdrname, wcsname,
                        sipname, npolfile, d2imfile,
                        nmatch, catalog, wcskey,
                        author, descrip, history):
    # convert input values into valid FITS kw values
    if author is None:
        author = ''
    if descrip is None:
        descrip = ''

    sipname, idctab = utils.build_sipname(fobj, fname, sipname)
    logger.info("Setting sipname value to %s" % sipname)

    npolname, npolfile = utils.build_npolname(fobj, npolfile)
    logger.info("Setting npolfile value to %s" % npolname)

    d2imname, d2imfile = utils.build_d2imname(fobj, d2imfile)
    logger.info("Setting d2imfile value to %s" % d2imname)

    distname = utils.build_distname(sipname, npolname, d2imname)
    logger.info("Setting distname to %s" % distname)

    # open file and parse comments
    if history not in ['', ' ', None, 'INDEF'] and os.path.isfile(history):
        f = open(fu.osfn(history))
        history = f.readlines()
        f.close()
    else:
        history = ''

    rms_ra = fobj[wcsext].header.get("CRDER1" + wcskey, 0)
    rms_dec = fobj[wcsext].header.get("CRDER2" + wcskey, 0)
    if not nmatch:
        nmatch = fobj[wcsext].header.get("NMATCH" + wcskey, 0)
    if not catalog:
        catalog = fobj[wcsext].header.get('CATALOG' + wcskey, "")
    # get the version of STWCS used to create the WCS of the science file.

    upwcsver = fobj[0].header.get('UPWCSVER', "")
    pywcsver = fobj[0].header.get('PYWCSVER', "")
    # build Primary HDU
    phdu = fits.PrimaryHDU()
    phdu.header['DESTIM'] = (destim, 'Destination observation root name')
    phdu.header['HDRNAME'] = (hdrname, 'Headerlet name')
    fmt = "%Y-%m-%dT%H:%M:%S"
    phdu.header['DATE'] = (time.strftime(fmt), 'Date FITS file was generated')
    phdu.header['WCSNAME'] = (wcsname, 'Coordinate system title')
    phdu.header['DISTNAME'] = (distname, 'Distortion model name')
    phdu.header['SIPNAME'] = (sipname,
                              'origin of SIP polynomial distortion model')
    phdu.header['NPOLFILE'] = (npolfile,
                               'origin of non-polynmial distortion model')
    phdu.header['D2IMFILE'] = (d2imfile,
                               'origin of detector to image correction')
    phdu.header['IDCTAB'] = (idctab,
                             'origin of Polynomial Distortion')
    phdu.header['AUTHOR'] = (author, 'headerlet created by this user')
    phdu.header['DESCRIP'] = (descrip,
                              'Short description of headerlet solution')
    phdu.header['RMS_RA'] = (rms_ra,
                             'RMS in RA at ref pix of headerlet solution')
    phdu.header['RMS_DEC'] = (rms_dec,
                              'RMS in Dec at ref pix of headerlet solution')
    phdu.header['NMATCH'] = (nmatch,
                             'Number of sources used for headerlet solution')
    phdu.header['CATALOG'] = (catalog,
                              'Astrometric catalog used for headerlet '
                              'solution')
    phdu.header['UPWCSVER'] = (upwcsver, "Version of STWCS used to update the WCS")
    phdu.header['PYWCSVER'] = (pywcsver, "Version of PYWCS used to update the WCS")

    # clean up history string in order to remove whitespace characters that
    # would cause problems with FITS
    if isinstance(history, list):
        history_str = ''
        for line in history:
            history_str += line
    else:
        history_str = history
    history_lines = textwrap.wrap(history_str, width=70)
    for hline in history_lines:
        phdu.header.add_history(hline)

    return phdu


# Public Interface functions


@with_logging
def extract_headerlet(filename, output, extnum=None, hdrname=None,
                      clobber=False, logging=True):
    """
    Finds a headerlet extension in a science file and writes it out as
    a headerlet FITS file.

    If both hdrname and extnum are given they should match, if not
    raise an Exception

    Parameters
    ----------
    filename: string or HDUList or Python list
        This specifies the name(s) of science file(s) from which headerlets
        will be extracted.

        String input formats supported include use of wild-cards, IRAF-style
        '@'-files (given as '@<filename>') and comma-separated list of names.
        An input filename (str) will be expanded as necessary to interpret
        any environmental variables included in the filename.
        If a list of filenames has been specified, it will extract a
        headerlet from the same extnum from all filenames.
    output: string
           Filename or just rootname of output headerlet FITS file
           If string does not contain '.fits', it will create a filename with
           '_hlet.fits' suffix
    extnum: int
           Extension number which contains the headerlet to be written out
    hdrname: string
           Unique name for headerlet, stored as the HDRNAME keyword
           It stops if a value is not provided and no extnum has been specified
    clobber: bool
        If output file already exists, this parameter specifies whether or not
        to overwrite that file [Default: False]
    logging: bool
             enable logging to a file

    """

    if isinstance(filename, fits.HDUList):
        filename = [filename]
    else:
        filename, oname = parseinput.parseinput(filename)

    for f in filename:
        fobj, fname, close_fobj = parse_filename(f)
        frootname = fu.buildNewRootname(fname)
        if hdrname in ['', ' ', None, 'INDEF'] and extnum is None:
            if close_fobj:
                fobj.close()
                logger.critical("Expected a valid extnum or hdrname parameter")
                raise ValueError
        if hdrname is not None:
            extn_from_hdrname = find_headerlet_HDUs(fobj, hdrname=hdrname)[0]
            if extn_from_hdrname != extnum:
                logger.critical("hdrname and extnmu should refer to the same FITS extension")
                raise ValueError
            else:
                hdrhdu = fobj[extn_from_hdrname]
        else:
            hdrhdu = fobj[extnum]

        if not isinstance(hdrhdu, HeaderletHDU):
            logger.critical("Specified extension is not a headerlet")
            raise ValueError

        hdrlet = hdrhdu.headerlet

        if output is None:
            output = frootname

        if '.fits' in output:
            outname = output
        else:
            outname = '%s_hlet.fits' % output

        hdrlet.tofile(outname, clobber=clobber)

        if close_fobj:
            fobj.close()


@with_logging
def write_headerlet(filename, hdrname, output=None, sciext='SCI',
                    wcsname=None, wcskey=None, destim=None,
                    sipname=None, npolfile=None, d2imfile=None,
                    author=None, descrip=None, history=None,
                    nmatch=None, catalog=None,
                    attach=True, clobber=False, logging=False):

    """
    Save a WCS as a headerlet FITS file.

    This function will create a headerlet, write out the headerlet to a
    separate headerlet file, then, optionally, attach it as an extension
    to the science image (if it has not already been archived)

    Either wcsname or wcskey must be provided; if both are given, they must
    match a valid WCS.

    Updates wcscorr if necessary.

    Parameters
    ----------
    filename: string or HDUList or Python list
        This specifies the name(s) of science file(s) from which headerlets
        will be created and written out.
        String input formats supported include use of wild-cards, IRAF-style
        '@'-files (given as '@<filename>') and comma-separated list of names.
        An input filename (str) will be expanded as necessary to interpret
        any environmental variables included in the filename.
    hdrname: string
        Unique name for this headerlet, stored as HDRNAME keyword
    output: string or None
        Filename or just rootname of output headerlet FITS file
        If string does not contain '.fits', it will create a filename
        starting with the science filename and ending with '_hlet.fits'.
        If None, a default filename based on the input filename will be
        generated for the headerlet FITS filename
    sciext: string
        name (EXTNAME) of extension that contains WCS to be saved
    wcsname: string
        name of WCS to be archived, if " ": stop
    wcskey: one of A...Z or " " or "PRIMARY"
        if " " or "PRIMARY" - archive the primary WCS
    destim: string
        DESTIM keyword
        if  NOne, use ROOTNAME or science file name
    sipname: string or None (default)
         Name of unique file where the polynomial distortion coefficients were
         read from. If None, the behavior is:
         The code looks for a keyword 'SIPNAME' in the science header
         If not found, for HST it defaults to 'IDCTAB'
         If there is no SIP model the value is 'NOMODEL'
         If there is a SIP model but no SIPNAME, it is set to 'UNKNOWN'
    npolfile: string or None (default)
         Name of a unique file where the non-polynomial distortion was stored.
         If None:
         The code looks for 'NPOLFILE' in science header.
         If 'NPOLFILE' was not found and there is no npol model, it is set to 'NOMODEL'
         If npol model exists, it is set to 'UNKNOWN'
    d2imfile: string
         Name of a unique file where the detector to image correction was
         stored. If None:
         The code looks for 'D2IMFILE' in the science header.
         If 'D2IMFILE' is not found and there is no d2im correction,
         it is set to 'NOMODEL'
         If d2im correction exists, but 'D2IMFILE' is missing from science
         header, it is set to 'UNKNOWN'
    author: string
        Name of user who created the headerlet, added as 'AUTHOR' keyword
        to headerlet PRIMARY header
    descrip: string
        Short description of the solution provided by the headerlet
        This description will be added as the single 'DESCRIP' keyword
        to the headerlet PRIMARY header
    history: filename, string or list of strings
        Long (possibly multi-line) description of the solution provided
        by the headerlet. These comments will be added as 'HISTORY' cards
        to the headerlet PRIMARY header
        If filename is specified, it will format and attach all text from
        that file as the history.
    attach: bool
        Specify whether or not to attach this headerlet as a new extension
        It will verify that no other headerlet extension has been created with
        the same 'hdrname' value.
    clobber: bool
        If output file already exists, this parameter specifies whether or not
        to overwrite that file [Default: False]
    logging: bool
         enable file logging
    """

    if isinstance(filename, fits.HDUList):
        filename = [filename]
    else:
        filename, oname = parseinput.parseinput(filename)

    for f in filename:
        if isinstance(f, str):
            fname = f
        else:
            fname = f.filename()

        if wcsname in [None, ' ', '', 'INDEF'] and wcskey is None:
            message = """\n
            No valid WCS found found in %s.
            A valid value for either "wcsname" or "wcskey"
            needs to be specified.
            """ % fname
            logger.critical(message)
            raise ValueError

        # Translate 'wcskey' value for PRIMARY WCS to valid altwcs value of ' '
        if wcskey == 'PRIMARY':
            wcskey = ' '

        if attach:
            umode = 'update'
        else:
            umode = 'readonly'

        fobj, fname, close_fobj = parse_filename(f, mode=umode)

        # Interpret sciext input for this file
        if isinstance(sciext, int):
            sciextlist = [sciext]  # allow for specification of simple FITS header
        elif isinstance(sciext, str):
            numsciext = countExtn(fobj, sciext)
            if numsciext > 0:
                sciextlist = [tuple((sciext, i)) for i in range(1, numsciext + 1)]
            else:
                sciextlist = [0]
        elif isinstance(sciext, list):
            sciextlist = sciext
        else:
            errstr = "Expected sciext to be a list of FITS extensions with science data\n" + \
                "    a valid EXTNAME string, or an integer."
            logger.critical(errstr)
            raise ValueError

        # Insure that WCSCORR table has been created with all original
        # WCS's recorded prior to adding the headerlet WCS
        wcscorr.init_wcscorr(fobj)

        if wcsname is None:
            scihdr = fobj[sciextlist[0]].header
            wname = scihdr['wcsname' + wcskey]
        else:
            wname = wcsname
        if hdrname in [None, ' ', '']:
            hdrname = wcsname

        logger.critical('Creating the headerlet from image %s' % fname)
        hdrletobj = create_headerlet(fobj, sciext=sciextlist,
                                     wcsname=wname, wcskey=wcskey,
                                     hdrname=hdrname,
                                     sipname=sipname, npolfile=npolfile,
                                     d2imfile=d2imfile, author=author,
                                     descrip=descrip, history=history,
                                     nmatch=nmatch, catalog=catalog,
                                     logging=False)

        if attach:
            # Check to see whether or not a HeaderletHDU with
            # this hdrname already exists
            hdrnames = get_headerlet_kw_names(fobj)
            if hdrname not in hdrnames:
                hdrlet_hdu = HeaderletHDU.fromheaderlet(hdrletobj)

                if destim is not None:
                    hdrlet_hdu.header['destim'] = destim

                fobj.append(hdrlet_hdu)

                # Update the WCSCORR table with new rows from the headerlet's WCSs
                wcscorr.update_wcscorr(fobj, source=hdrletobj,
                                       extname='SIPWCS', wcs_id=wname)

                utils.updateNEXTENDKw(fobj)
                fobj.flush()
            else:
                message = """
                Headerlet with hdrname %s already archived for WCS %s.
                No new headerlet appended to %s.
                """ % (hdrname, wname, fname)
                logger.critical(message)

        if close_fobj:
            logger.info('Closing image in write_headerlet()...')
            fobj.close()

        frootname = fu.buildNewRootname(fname)

        if output is None:
            # Generate default filename for headerlet FITS file
            outname = '{0}_hlet.fits'.format(frootname)
        else:
            outname = output

        if not outname.endswith('.fits'):
            outname = '{0}_{1}_hlet.fits'.format(frootname, outname)

        # If user specifies an output filename for headerlet, write it out
        hdrletobj.tofile(outname, clobber=clobber)
        logger.critical('Created Headerlet file %s ' % outname)

        del hdrletobj


@with_logging
def create_headerlet(filename, sciext='SCI', hdrname=None, destim=None,
                     wcskey=" ", wcsname=None,
                     sipname=None, npolfile=None, d2imfile=None,
                     author=None, descrip=None, history=None,
                     nmatch=None, catalog=None,
                     logging=False, logmode='w'):
    """
    Create a headerlet from a WCS in a science file
    If both wcskey and wcsname are given they should match, if not
    raise an Exception

    Parameters
    ----------
    filename: string or HDUList
            Either a filename or PyFITS HDUList object for the input science file
            An input filename (str) will be expanded as necessary to interpret
            any environmental variables included in the filename.
    sciext: string or python list (default: 'SCI')
            Extension in which the science data with the linear WCS is.
            The headerlet will be created from these extensions.
            If string - a valid EXTNAME is expected
            If int - specifies an extension with a valid WCS, such as 0 for a
            simple FITS file
            If list - a list of FITS extension numbers or strings representing
            extension tuples, e.g. ('SCI, 1') is expected.
    hdrname: string
            value of HDRNAME keyword
            Takes the value from the HDRNAME<wcskey> keyword, if not available from WCSNAME<wcskey>
            It stops if neither is found in the science file and a value is not provided
    destim: string or None
            name of file this headerlet can be applied to
            if None, use ROOTNAME keyword
    wcskey: char (A...Z) or " " or "PRIMARY" or None
            a char representing an alternate WCS to be used for the headerlet
            if " ", use the primary (default)
            if None use wcsname
    wcsname: string or None
            if wcskey is None use wcsname specified here to choose an alternate WCS
            for the headerlet
    sipname: string or None (default)
            Name of unique file where the polynomial distortion coefficients were
            read from. If None, the behavior is:
            The code looks for a keyword 'SIPNAME' in the science header
            If not found, for HST it defaults to 'IDCTAB'
            If there is no SIP model the value is 'NOMODEL'
            If there is a SIP model but no SIPNAME, it is set to 'UNKNOWN'
    npolfile: string or None (default)
            Name of a unique file where the non-polynomial distortion was stored.
            If None:
            The code looks for 'NPOLFILE' in science header.
            If 'NPOLFILE' was not found and there is no npol model, it is set to 'NOMODEL'
            If npol model exists, it is set to 'UNKNOWN'
    d2imfile: string
            Name of a unique file where the detector to image correction was
            If None:
            The code looks for 'D2IMFILE' in the science header.
            If 'D2IMFILE' is not found and there is no d2im correction,
            it is set to 'NOMODEL'
            If d2im correction exists, but 'D2IMFILE' is missing from science
            header, it is set to 'UNKNOWN'
    author: string
            Name of user who created the headerlet, added as 'AUTHOR' keyword
            to headerlet PRIMARY header
    descrip: string
            Short description of the solution provided by the headerlet
            This description will be added as the single 'DESCRIP' keyword
            to the headerlet PRIMARY header
    history: filename, string or list of strings
            Long (possibly multi-line) description of the solution provided
            by the headerlet. These comments will be added as 'HISTORY' cards
            to the headerlet PRIMARY header
            If filename is specified, it will format and attach all text from
            that file as the history.
    nmatch: int (optional)
            Number of sources used in the new solution fit
    catalog: string (optional)
            Astrometric catalog used for headerlet solution
    logging: bool
            enable file logging
    logmode: 'w' or 'a'
            log file open mode

    Returns
    -------
    Headerlet object

    """
    if wcskey == 'O':
        message = "Warning: 'O' is a reserved key for the original WCS. Quitting..."
        logger.info(message)
        return None

    fobj, fname, close_file = parse_filename(filename)
    # based on `sciext` create a list of (EXTNAME, EXTVER) tuples
    # of extensions with WCS to be saved in a headerlet
    sciext = get_extname_extver_list(fobj, sciext)
    logger.debug("Data extensions from which to create headerlet:\n\t %s"
                 % (str(sciext)))
    if not sciext:
        logger.critical("No valid target extensions found in file %s" % fname)
        raise ValueError

    # Define extension to evaluate for verification of input parameters
    wcsext = sciext[0]
    logger.debug("sciext in create_headerlet is %s" % str(sciext))
    # Translate 'wcskey' value for PRIMARY WCS to valid altwcs value of ' '
    if wcskey == 'PRIMARY':
        wcskey = ' '
        logger.info("wcskey reset from 'PRIMARY' to ' '")
    wcskey = wcskey.upper()
    wcsnamekw = "".join(["WCSNAME", wcskey.upper()]).rstrip()
    hdrnamekw = "".join(["HDRNAME", wcskey.upper()]).rstrip()

    if not wcsname:
        # User did not specify a value for 'wcsname'
        if wcsnamekw in fobj[wcsext].header:
            # check if there's a WCSNAME for this wcskey in the header
            wcsname = fobj[wcsext].header[wcsnamekw]
            logger.info("Setting wcsname from header[%s] to %s" % (wcsnamekw, wcsname))
        else:
            if hdrname not in ['', ' ', None, "INDEF"]:
                """
                If wcsname for this wcskey was not provided
                and WCSNAME<wcskey> does not exist in the header
                and hdrname is provided, then
                use hdrname as WCSNAME for the headerlet.
                """
                wcsname = hdrname
                logger.debug("Setting wcsname from hdrname to %s" % hdrname)
            else:
                if hdrnamekw in fobj[wcsext].header:
                    wcsname = fobj[wcsext].header[hdrnamekw]
                    logger.debug("Setting wcsname from header[%s] to %s" % (hdrnamekw, wcsname))
                else:
                    message = """
                    Required keywords 'HDRNAME' or 'WCSNAME' not found!
                    Please specify a value for parameter 'hdrname'
                    or update header with 'WCSNAME' keyword.
                    """
                    logger.critical(message)
                    raise KeyError
    else:
        # Verify that 'wcsname' and 'wcskey' values specified by user reference
        # the same WCS
        wname = fobj[wcsext].header[wcsnamekw]
        if wcsname != wname:
            message = "\tInconsistent values for 'wcskey' and 'wcsname' specified!\n"
            message += "    'wcskey' = %s and 'wcsname' = %s. \n" % (wcskey, wcsname)
            message += "Actual value of %s found to be %s. \n" % (wcsnamekw, wname)
            logger.critical(message)
            raise KeyError(message)
    wkeys = altwcs.wcskeys(fobj, ext=wcsext)
    if wcskey != ' ':
        if wcskey not in wkeys:
            mess = "Skipping extension {0} - no WCS with wcskey={1} found.".format(wcsext, wcskey)
            logger.critical(mess)
            raise ValueError(mess)

    # get remaining required keywords
    if destim is None:
        if 'ROOTNAME' in fobj[0].header:
            destim = fobj[0].header['ROOTNAME']
            logger.info("Setting destim to rootname of the science file")
        else:
            destim = fname
            logger.info('DESTIM not provided')
            logger.info('Keyword "ROOTNAME" not found')
            logger.info('Using file name as DESTIM')

    if not hdrname:
        # check if HDRNAME<wcskey> is in header
        if hdrnamekw in fobj[wcsext].header:
            hdrname = fobj[wcsext].header[hdrnamekw]
        else:
            if wcsnamekw in fobj[wcsext].header:
                hdrname = fobj[wcsext].header[wcsnamekw]
                message = """
                Using default value for HDRNAME of "%s" derived from %s.
                """ % (hdrname, wcsnamekw)
                logger.info(message)
                logger.info("Setting hdrname to %s from header[%s]"
                            % (hdrname, wcsnamekw))
            else:
                message = "Required keywords 'HDRNAME' or 'WCSNAME' not found"
                logger.critical(message)
                raise KeyError

    hdul = []
    phdu = _create_primary_HDU(fobj, fname, wcsext, destim, hdrname, wcsname,
                               sipname, npolfile, d2imfile,
                               nmatch, catalog, wcskey,
                               author, descrip, history)
    hdul.append(phdu)

    """
    nd2i is a counter for d2i extensions to be used when the science file
    has an old d2i correction format. The old format did not write EXTVER
    kw for the d2i correction in the science header bu tthe new format expects
    them.
    """
    nd2i_extver = 1
    for ext in sciext:
        wkeys = altwcs.wcskeys(fobj, ext=ext)
        if wcskey != ' ':
            if wcskey not in wkeys:
                logger.debug(
                    'No WCS with wcskey=%s found in extension %s.  '
                    'Skipping...' % (wcskey, str(ext)))
                raise ValueError("")

        hwcs = HSTWCS(fobj, ext=ext, wcskey=wcskey)

        whdul = hwcs.to_fits(relax=True, key=" ")
        altwcs.exclude_hst_specific(whdul[0].header)

        if hasattr(hwcs, 'orientat'):
            orient_comment = "positions angle of image y axis (deg. e of n)"
            whdul[0].header['ORIENTAT'] = (hwcs.orientat, orient_comment)

        whdul[0].header.append(('TG_ENAME', ext[0], 'Target science data extname'))
        whdul[0].header.append(('TG_EVER', ext[1], 'Target science data extver'))

        if hwcs.wcs.has_cd():
            whdul[0].header = altwcs.pc2cd(whdul[0].header)

        idckw = hwcs._idc2hdr()
        whdul[0].header.extend(idckw)

        if hwcs.det2im1 or hwcs.det2im2:
            try:
                whdul[0].header.append(fobj[ext].header.cards['D2IMEXT'])
            except KeyError:
                pass
            whdul[0].header.extend(fobj[ext].header.cards['D2IMERR*'])
            if 'D2IM1.EXTVER' in whdul[0].header:
                try:
                    whdul[0].header['D2IM1.EXTVER'] = fobj[ext].header['D2IM1.EXTVER']
                except KeyError:
                    whdul[0].header['D2IM1.EXTVER'] = nd2i_extver
                    nd2i_extver += 1
            if 'D2IM2.EXTVER' in whdul[0].header:
                try:
                    whdul[0].header['D2IM2.EXTVER'] = fobj[ext].header['D2IM2.EXTVER']
                except KeyError:
                    whdul[0].header['D2IM2.EXTVER'] = nd2i_extver
                    nd2i_extver += 1

        if hwcs.cpdis1 or hwcs.cpdis2:
            whdul[0].header.extend(fobj[ext].header.cards['CPERR*'])
            try:
                whdul[0].header.append(fobj[ext].header.cards['NPOLEXT'])
            except KeyError:
                pass
            if 'DP1.EXTVER' in whdul[0].header:
                whdul[0].header['DP1.EXTVER'] = fobj[ext].header['DP1.EXTVER']
            if 'DP2.EXTVER' in whdul[0].header:
                whdul[0].header['DP2.EXTVER'] = fobj[ext].header['DP2.EXTVER']
        ihdu = fits.ImageHDU(header=whdul[0].header, name='SIPWCS')

        if ext[0] != "PRIMARY":
            ihdu.ver = int(fobj[ext].header['EXTVER'])

        hdul.append(ihdu)

        if hwcs.cpdis1:
            whdu = whdul[('WCSDVARR', 1)].copy()
            whdu.ver = int(fobj[ext].header['DP1.EXTVER'])
            hdul.append(whdu)
        if hwcs.cpdis2:
            whdu = whdul[('WCSDVARR', 2)].copy()
            whdu.ver = int(fobj[ext].header['DP2.EXTVER'])
            hdul.append(whdu)

        if hwcs.det2im1:
            whdu = whdul[('D2IMARR', 1)].copy()
            whdu.ver = int(ihdu.header['D2IM1.EXTVER'])
            hdul.append(whdu)
        if hwcs.det2im2:
            whdu = whdul[('D2IMARR', 2)].copy()
            whdu.ver = int(ihdu.header['D2IM2.EXTVER'])
            hdul.append(whdu)

    if close_file:
        fobj.close()

    hlet = Headerlet(hdul, logging=logging, logmode='a')
    hlet.init_attrs()
    return hlet


@with_logging
def apply_headerlet_as_primary(filename, hdrlet, attach=True, archive=True,
                               force=False, logging=False, logmode='a'):
    """
    Apply headerlet 'hdrfile' to a science observation 'destfile' as the primary WCS

    Parameters
    ----------
    filename: string or list of strings
             File name(s) of science observation whose WCS solution will be updated
    hdrlet: string or list of strings
             Headerlet file(s), must match 1-to-1 with input filename(s)
    attach: bool
            True (default): append headerlet to FITS file as a new extension.
    archive: bool
            True (default): before updating, create a headerlet with the
            WCS old solution.
    force: bool
            If True, this will cause the headerlet to replace the current PRIMARY
            WCS even if it has a different distortion model. [Default: False]
    logging: bool
            enable file logging
    logmode: 'w' or 'a'
             log file open mode
    """
    if not isinstance(filename, list):
        filename = [filename]
    if not isinstance(hdrlet, list):
        hdrlet = [hdrlet]
    if len(hdrlet) != len(filename):
        logger.critical("Filenames must have matching headerlets. "
                        "{0:d} filenames and {1:d} headerlets specified".format(len(filename),
                                                                                len(hdrlet)))

    for fname, h in zip(filename, hdrlet):
        print("Applying {0} as Primary WCS to {1}".format(h, fname))
        hlet = Headerlet.fromfile(h, logging=logging, logmode=logmode)
        hlet.apply_as_primary(fname, attach=attach, archive=archive,
                              force=force)


@with_logging
def apply_headerlet_as_alternate(filename, hdrlet, attach=True, wcskey=None,
                                 wcsname=None, logging=False, logmode='w'):
    """
    Apply headerlet to a science observation as an alternate WCS

    Parameters
    ----------
    filename: string or list of strings
             File name(s) of science observation whose WCS solution will be updated
    hdrlet: string or list of strings
             Headerlet file(s), must match 1-to-1 with input filename(s)
    attach: bool
          flag indicating if the headerlet should be attached as a
          HeaderletHDU to fobj. If True checks that HDRNAME is unique
          in the fobj and stops if not.
    wcskey: string
          Key value (A-Z, except O) for this alternate WCS
          If None, the next available key will be used
    wcsname: string
          Name to be assigned to this alternate WCS
          WCSNAME is a required keyword in a Headerlet but this allows the
          user to change it as desired.
    logging: bool
          enable file logging
    logmode: 'a' or 'w'
    """
    if not isinstance(filename, list):
        filename = [filename]
    if not isinstance(hdrlet, list):
        hdrlet = [hdrlet]
    if len(hdrlet) != len(filename):
        logger.critical("Filenames must have matching headerlets. "
                        "{0:d} filenames and {1:d} headerlets specified".format(len(filename),
                                                                                len(hdrlet)))

    for fname, h in zip(filename, hdrlet):
        print('Applying {0} as an alternate WCS to {1}'.format(h, fname))
        hlet = Headerlet.fromfile(h, logging=logging, logmode=logmode)
        hlet.apply_as_alternate(fname, attach=attach,
                                wcsname=wcsname, wcskey=wcskey)


@with_logging
def attach_headerlet(filename, hdrlet, logging=False, logmode='a'):
    """
    Attach Headerlet as an HeaderletHDU to a science file

    Parameters
    ----------
    filename: HDUList or list of HDULists
            science file(s) to which the headerlet should be applied
    hdrlet: string, Headerlet object or list of strings or Headerlet objects
            string representing a headerlet file(s), must match 1-to-1 input filename(s)
    logging: bool
            enable file logging
    logmode: 'a' or 'w'
    """
    if not isinstance(filename, list):
        filename = [filename]
    if not isinstance(hdrlet, list):
        hdrlet = [hdrlet]
    if len(hdrlet) != len(filename):
        logger.critical("Filenames must have matching headerlets. "
                        "{0:d} filenames and {1:d} headerlets specified".format(len(filename),
                                                                                len(hdrlet)))

    for fname, h in zip(filename, hdrlet):
        print('Attaching {0} as Headerlet extension to {1}'.format(h, fname))
        hlet = Headerlet.fromfile(h, logging=logging, logmode=logmode)
        hlet.attach_to_file(fname, archive=True)


@with_logging
def delete_headerlet(filename, hdrname=None, hdrext=None, distname=None,
                     keep_first=False, logging=False, logmode='w'):
    """
    Deletes all HeaderletHDUs with same HDRNAME from science files

    Notes
    -----
    One of hdrname, hdrext or distname should be given.
    If hdrname is given - delete all HeaderletHDUs with a name HDRNAME from fobj.
    If hdrext is given - delete specified HeaderletHDU(s) extension(s).
    If distname is given - deletes all HeaderletHDUs with a specific distortion model from fobj.
    Updates wcscorr

    Parameters
    ----------
    filename: string, HDUList or list of strings
            Filename can be specified as a single filename or HDUList, or
            a list of filenames
            Each input filename (str) will be expanded as necessary to interpret
            any environmental variables included in the filename.
    hdrname: string or None
        HeaderletHDU primary header keyword HDRNAME
    hdrext: int, tuple or None
        HeaderletHDU FITS extension number
        tuple has the form ('HDRLET', 1)
    distname: string or None
        distortion model as specified in the DISTNAME keyword
    keep_first: bool, optional
        If True, the first matching HeaderletHDU found will be NOT deleted.
    logging: bool
             enable file logging
    logmode: 'a' or 'w'
    """
    if not isinstance(filename, list):
        filename = [filename]

    for f in filename:
        print("Deleting Headerlet from ", f)
        _delete_single_headerlet(f, hdrname=hdrname, hdrext=hdrext,
                                 distname=distname, keep_first=keep_first,
                                 logging=logging, logmode='a')


def _delete_single_headerlet(filename, hdrname=None, hdrext=None, distname=None,
                             keep_first=True, logging=False, logmode='w'):
    """
    Deletes all matching HeaderletHDU(s) from a SINGLE science file

    Notes
    -----
    One of hdrname, hdrext or distname should be given.
    If hdrname is given - delete all HeaderletHDUs with a name HDRNAME from fobj.
    If hdrext is given - delete HeaderletHDU in extension.
    If distname is given - deletes all HeaderletHDUs with a specific distortion model from fobj.
    Updates wcscorr

    Parameters
    ----------
    filename: string or HDUList
           Either a filename or PyFITS HDUList object for the input science file
            An input filename (str) will be expanded as necessary to interpret
            any environmental variables included in the filename.
    hdrname: string or None
        HeaderletHDU primary header keyword HDRNAME
    hdrext: int, tuple or None
        HeaderletHDU FITS extension number
        tuple has the form ('HDRLET', 1)
    distname: string or None
        distortion model as specified in the DISTNAME keyword
    keep_first: bool, optional
        If True, all but the first duplicate extension will be deleted.
    logging: bool
             enable file logging
    logmode: 'a' or 'w'
    """
    hdrlet_ind = find_headerlet_HDUs(filename, hdrname=hdrname, hdrext=hdrext,
                                     distname=distname, logging=logging, logmode='a')
    if len(hdrlet_ind) == 0:
        message = """
        No HDUs deleted... No Headerlet HDUs found with '
        hdrname = %s
        hdrext  = %s
        distname = %s
        Please review input parameters and try again.
        """ % (hdrname, str(hdrext), distname)
        logger.critical(message)
        return

    fobj, fname, close_fobj = parse_filename(filename, mode='update')

    # delete row(s) from WCSCORR table now...
    #
    #
    if hdrname not in ['', ' ', None, 'INDEF']:
        selections = {'hdrname': hdrname}
    elif hdrname in ['', ' ', None, 'INDEF'] and hdrext is not None:
        selections = {'hdrname': fobj[hdrext].header['hdrname']}
    else:
        selections = {'distname': distname}
    wcscorr.delete_wcscorr_row(fobj['WCSCORR'].data, selections)

    # delete the headerlet extension now
    hdrlet_ind.reverse()
    del_all = 1 if keep_first else 0
    del_hdrlets = slice(0, len(hdrlet_ind) - del_all)
    for hdrind in hdrlet_ind[del_hdrlets]:
        del fobj[hdrind]

    utils.updateNEXTENDKw(fobj)
    # Update file object with changes
    fobj.flush()
    # close file, if was opened by this function
    if close_fobj:
        fobj.close()
    logger.critical('Deleted headerlet from extension(s) %s ' % str(hdrlet_ind))


def headerlet_summary(filename, columns=None, pad=2, maxwidth=None,
                      output=None, clobber=True, quiet=False):
    """

    Print a summary of all HeaderletHDUs in a science file to STDOUT, and
    optionally to a text file
    The summary includes:
    HDRLET_ext_number  HDRNAME  WCSNAME DISTNAME SIPNAME NPOLFILE D2IMFILE

    Parameters
    ----------
    filename: string or HDUList
            Either a filename or PyFITS HDUList object for the input science file
            An input filename (str) will be expanded as necessary to interpret
            any environmental variables included in the filename.
    columns: list
            List of headerlet PRIMARY header keywords to report in summary
            By default (set to None), it will use the default set of keywords
            defined as the global list DEFAULT_SUMMARY_COLS
    pad: int
            Number of padding spaces to put between printed columns
            [Default: 2]
    maxwidth: int
            Maximum column width(not counting padding) for any column in summary
            By default (set to None), each column's full width will be used
    output: string (optional)
            Name of optional output file to record summary. This filename
            can contain environment variables.
            [Default: None]
    clobber: bool
            If True, will overwrite any previous output file of same name
    quiet: bool
            If True, will NOT report info to STDOUT

    """
    if columns is None:
        summary_cols = DEFAULT_SUMMARY_COLS
    else:
        summary_cols = columns

    summary_dict = {}
    for kw in summary_cols:
        summary_dict[kw] = copy.deepcopy(COLUMN_DICT)

    # Define Extension number column
    extnums_col = copy.deepcopy(COLUMN_DICT)
    extnums_col['name'] = 'EXTN'
    extnums_col['width'] = 6

    fobj, fname, close_fobj = parse_filename(filename)
    # find all HDRLET extensions and combine info into a single summary
    for extn in fobj:
        if 'extname' in extn.header and extn.header['extname'] == 'HDRLET':
            hdrlet_indx = fobj.index_of(('hdrlet', extn.header['extver']))
            try:
                ext_cols, ext_summary = extn.headerlet.summary(columns=summary_cols)
                extnums_col['vals'].append(hdrlet_indx)
                for kw in summary_cols:
                    for key in COLUMN_DICT:
                        summary_dict[kw][key].extend(ext_summary[kw][key])
            except:
                print("Skipping headerlet")
                print("Could not read Headerlet from extension ", hdrlet_indx)

    if close_fobj:
        fobj.close()

    # Print out the summary dictionary
    print_summary(summary_cols, summary_dict, pad=pad, maxwidth=maxwidth,
                  idcol=extnums_col, output=output,
                  clobber=clobber, quiet=quiet)


@with_logging
def restore_from_headerlet(filename, hdrname=None, hdrext=None, archive=True,
                           force=False, logging=False, logmode='w'):
    """
    Restores a headerlet as a primary WCS

    Parameters
    ----------
    filename: string or HDUList
           Either a filename or PyFITS HDUList object for the input science file
            An input filename (str) will be expanded as necessary to interpret
            any environmental variables included in the filename.
    hdrname: string
        HDRNAME keyword of HeaderletHDU
    hdrext: int or tuple
        Headerlet extension number of tuple ('HDRLET',2)
    archive: bool (default: True)
        When the distortion model in the headerlet is the same as the distortion model of
        the science file, this flag indicates if the primary WCS should be saved as an alternate
        nd a headerlet extension.
        When the distortion models do not match this flag indicates if the current primary and
        alternate WCSs should be archived as headerlet extensions and alternate WCS.
    force: bool (default:False)
        When the distortion models of the headerlet and the primary do not match, and archive
        is False, this flag forces an update of the primary.
    logging: bool
           enable file logging
    logmode: 'a' or 'w'
    """

    hdrlet_ind = find_headerlet_HDUs(filename, hdrext=hdrext, hdrname=hdrname)

    fobj, fname, close_fobj = parse_filename(filename, mode='update')

    if len(hdrlet_ind) > 1:
        if hdrext:
            kwerr = 'hdrext'
            kwval = hdrext
        else:
            kwerr = 'hdrname'
            kwval = hdrname
        message = """
        Multiple Headerlet extensions found with the same name.
        %d Headerlets with "%s" = %s found in %s.
        """ % (len(hdrlet_ind), kwerr, kwval, fname)
        if close_fobj:
            fobj.close()
        logger.critical(message)
        raise ValueError

    hdrlet_indx = hdrlet_ind[0]

    # read headerlet from HeaderletHDU into memory
    if hasattr(fobj[hdrlet_ind[0]], 'hdulist'):
        hdrlet = Headerlet(fobj[hdrlet_indx].hdulist)
        hdrlet.init_attrs()
    else:
        hdrlet = fobj[hdrlet_indx].headerlet  # older convention in PyFITS

    # read in the names of the extensions which HeaderletHDU updates
    extlist = []
    for ext in hdrlet:
        if 'extname' in ext.header and ext.header['extname'] == 'SIPWCS':
            # convert from string to tuple or int
            if 'sciext' in ext.header:
                sciext = eval(ext.header['sciext'])
            elif 'tg_ename' in ext.header:
                sciext = (ext.header['tg_ename'],ext.header['tg_ever'])
            else:
                # Assume EXTNAME='sci' with matching EXTVER values
                sciext = ('sci',ext.header['extver'])
            extlist.append(fobj[sciext])
    # determine whether distortion is the same
    current_distname = hdrlet[0].header['distname']
    # same_dist = True
    if current_distname != fobj[0].header['distname']:
        # same_dist = False
        if not archive and not force:
            if close_fobj:
                fobj.close()
            message = """
            Headerlet does not have the same distortion as image!
            Set "archive"=True to save old distortion model, or
            set "force"=True to overwrite old model with new.
            """
            logger.critical(message)
            raise ValueError

    # check whether primary WCS has been archived already
    # Use information from first 'SCI' extension
    priwcs_name = None

    scihdr = extlist[0].header
    if 'hdrname' in scihdr:
        priwcs_hdrname = scihdr['hdrname']
    else:
        if 'wcsname' in scihdr:
            priwcs_hdrname = priwcs_name = scihdr['wcsname']
        else:
            if 'idctab' in scihdr:
                priwcs_hdrname = ''.join(['IDC_',
                                          utils.extract_rootname(scihdr['idctab'], suffix='_idc')])
            else:
                priwcs_hdrname = 'UNKNOWN'
            priwcs_name = priwcs_hdrname
            scihdr['WCSNAME'] = priwcs_name, 'Coordinate system title'

    priwcs_unique = verify_hdrname_is_unique(fobj, priwcs_hdrname)
    if archive and priwcs_unique:
        if priwcs_unique:
            newhdrlet = create_headerlet(fobj, sciext=scihdr['extname'],
                                         hdrname=priwcs_hdrname)
            newhdrlet.attach_to_file(fobj)
    #
    # copy hdrlet as a primary
    #
    hdrlet.apply_as_primary(fobj, attach=False, archive=archive, force=force)

    utils.updateNEXTENDKw(fobj)
    fobj.flush()
    if close_fobj:
        fobj.close()


@with_logging
def restore_all_with_distname(filename, distname, primary, archive=True,
                              sciext='SCI', logging=False, logmode='w'):
    """
    Restores all HeaderletHDUs with a given distortion model as alternate WCSs and a primary

    Parameters
    --------------
    filename: string or HDUList
           Either a filename or PyFITS HDUList object for the input science file
            An input filename (str) will be expanded as necessary to interpret
            any environmental variables included in the filename.
    distname: string
        distortion model as represented by a DISTNAME keyword
    primary: int or string or None
        HeaderletHDU to be restored as primary
        if int - a fits extension
        if string - HDRNAME
        if None - use first HeaderletHDU
    archive: bool (default True)
        flag indicating if HeaderletHDUs should be created from the
        primary and alternate WCSs in fname before restoring all matching
        headerlet extensions
    logging: bool
         enable file logging
    logmode: 'a' or 'w'
    """

    fobj, fname, close_fobj = parse_filename(filename, mode='update')

    hdrlet_ind = find_headerlet_HDUs(fobj, distname=distname)
    if len(hdrlet_ind) == 0:
        message = """
        No Headerlet extensions found with

        DISTNAME = %s in %s.

        For a full list of DISTNAMEs found in all headerlet extensions:

        get_headerlet_kw_names(fobj, kw='DISTNAME')
        """ % (distname, fname)
        if close_fobj:
            fobj.close()
        logger.critical(message)
        raise ValueError

    # Interpret 'primary' parameter input into extension number
    if primary is None:
        primary_ind = hdrlet_ind[0]
    elif isinstance(primary, int):
        primary_ind = primary
    else:
        primary_ind = None
        for ind in hdrlet_ind:
            if fobj[ind].header['hdrname'] == primary:
                primary_ind = ind
                break
        if primary_ind is None:
            if close_fobj:
                fobj.close()
            message = """
            No Headerlet extensions found with DISTNAME = %s in %s.
            """ % (primary, fname)
            logger.critical(message)
            raise ValueError
    # Check to see whether 'primary' HeaderletHDU has same distname as user
    # specified on input

    # read headerlet from HeaderletHDU into memory
    if hasattr(fobj[primary_ind], 'hdulist'):
        primary_hdrlet = fobj[primary_ind].hdulist
    else:
        primary_hdrlet = fobj[primary_ind].headerlet  # older convention in PyFITS
    pri_distname = primary_hdrlet[0].header['distname']
    if pri_distname != distname:
        if close_fobj:
            fobj.close()
        message = """
        Headerlet extension to be used as PRIMARY WCS
        has "DISTNAME" = %s
        "DISTNAME" = %s was specified on input.
        All updated WCSs must have same DISTNAME. Quitting...'
        """ % (pri_distname, distname)
        logger.critical(message)
        raise ValueError

    # read in headerletHDUs and update WCS keywords
    for hlet in hdrlet_ind:
        if fobj[hlet].header['distname'] == distname:
            if hasattr(fobj[hlet], 'hdulist'):
                hdrlet = fobj[hlet].hdulist
            else:
                hdrlet = fobj[hlet].headerlet  # older convention in PyFITS
            if hlet == primary_ind:
                hdrlet.apply_as_primary(fobj, attach=False,
                                        archive=archive, force=True)
            else:
                hdrlet.apply_as_alternate(fobj, attach=False,
                                          wcsname=hdrlet[0].header['wcsname'])

    utils.updateNEXTENDKw(fobj)
    fobj.flush()
    if close_fobj:
        fobj.close()


@with_logging
def archive_as_headerlet(filename, hdrname, sciext='SCI',
                         wcsname=None, wcskey=None, destim=None,
                         sipname=None, npolfile=None, d2imfile=None,
                         author=None, descrip=None, history=None,
                         nmatch=None, catalog=None,
                         logging=False, logmode='w'):
    """
    Save a WCS as a headerlet extension and write it out to a file.

    This function will create a headerlet, attach it as an extension to the
    science image (if it has not already been archived) then, optionally,
    write out the headerlet to a separate headerlet file.

    Either wcsname or wcskey must be provided, if both are given, they must match a valid WCS
    Updates wcscorr if necessary.

    Parameters
    ----------
    filename: string or HDUList
           Either a filename or PyFITS HDUList object for the input science file
            An input filename (str) will be expanded as necessary to interpret
            any environmental variables included in the filename.
    hdrname: string
        Unique name for this headerlet, stored as HDRNAME keyword
    sciext: string
        name (EXTNAME) of extension that contains WCS to be saved
    wcsname: string
        name of WCS to be archived, if " ": stop
    wcskey: one of A...Z or " " or "PRIMARY"
        if " " or "PRIMARY" - archive the primary WCS
    destim: string
        DESTIM keyword
        if  NOne, use ROOTNAME or science file name
    sipname: string or None (default)
             Name of unique file where the polynomial distortion coefficients were
             read from. If None, the behavior is:
             The code looks for a keyword 'SIPNAME' in the science header
             If not found, for HST it defaults to 'IDCTAB'
             If there is no SIP model the value is 'NOMODEL'
             If there is a SIP model but no SIPNAME, it is set to 'UNKNOWN'
    npolfile: string or None (default)
             Name of a unique file where the non-polynomial distortion was stored.
             If None:
             The code looks for 'NPOLFILE' in science header.
             If 'NPOLFILE' was not found and there is no npol model, it is set to 'NOMODEL'
             If npol model exists, it is set to 'UNKNOWN'
    d2imfile: string
             Name of a unique file where the detector to image correction was
             stored. If None:
             The code looks for 'D2IMFILE' in the science header.
             If 'D2IMFILE' is not found and there is no d2im correction,
             it is set to 'NOMODEL'
             If d2im correction exists, but 'D2IMFILE' is missing from science
             header, it is set to 'UNKNOWN'
    author: string
            Name of user who created the headerlet, added as 'AUTHOR' keyword
            to headerlet PRIMARY header
    descrip: string
            Short description of the solution provided by the headerlet
            This description will be added as the single 'DESCRIP' keyword
            to the headerlet PRIMARY header
    history: filename, string or list of strings
            Long (possibly multi-line) description of the solution provided
            by the headerlet. These comments will be added as 'HISTORY' cards
            to the headerlet PRIMARY header
            If filename is specified, it will format and attach all text from
            that file as the history.
    logging: bool
            enable file folling
    logmode: 'w' or 'a'
             log file open mode
    """

    fobj, fname, close_fobj = parse_filename(filename, mode='update')

    if wcsname in [None, ' ', '', 'INDEF'] and wcskey is None:
        message = """
        No valid WCS found found in %s.
        A valid value for either "wcsname" or "wcskey"
        needs to be specified.
        """ % fname
        if close_fobj:
            fobj.close()
        logger.critical(message)
        raise ValueError

    # Translate 'wcskey' value for PRIMARY WCS to valid altwcs value of ' '
    if wcskey == 'PRIMARY':
        wcskey = ' '
    wcskey = wcskey.upper()

    if wcsname is None:
        scihdr = fobj[sciext, 1].header
        wcsname = scihdr['wcsname' + wcskey]

    if hdrname in [None, ' ', '']:
        hdrname = wcsname

    # Check to see whether or not a HeaderletHDU with this hdrname already
    # exists
    hdrnames = get_headerlet_kw_names(fobj)
    if hdrname not in hdrnames:
        hdrletobj = create_headerlet(fobj, sciext=sciext,
                                     wcsname=wcsname, wcskey=wcskey,
                                     hdrname=hdrname,
                                     sipname=sipname, npolfile=npolfile,
                                     d2imfile=d2imfile, author=author,
                                     descrip=descrip, history=history,
                                     nmatch=nmatch, catalog=catalog,
                                     logging=False)
        hlt_hdu = HeaderletHDU.fromheaderlet(hdrletobj)

        if destim is not None:
            hlt_hdu[0].header['destim'] = destim

        fobj.append(hlt_hdu)

        utils.updateNEXTENDKw(fobj)
        fobj.flush()
    else:
        message = """
        Headerlet with hdrname %s already archived for WCS %s
        No new headerlet appended to %s .
        """ % (hdrname, wcsname, fname)
        logger.warning(message)

    if close_fobj:
        fobj.close()


# Headerlet Class definitions


class Headerlet(fits.HDUList):
    """
    A Headerlet class
    Ref: http://mediawiki.stsci.edu/mediawiki/index.php/Telescopedia:Headerlets
    """

    def __init__(self, hdus=[], file=None, logging=False, logmode='w'):
        """
        Parameters
        ----------
        hdus : list
                List of HDUs to be used to create the headerlet object itself
        file:  string
                File-like object from which HDUs should be read
        logging: bool
                 enable file logging
        logmode: 'w' or 'a'
                for internal use only, indicates whether the log file
                should be open in attach or write mode
        """
        self.logging = logging
        init_logging('class Headerlet', level=logging, mode=logmode)

        super(Headerlet, self).__init__(hdus, file=file)

    def init_attrs(self):
        self.fname = self.filename()
        self.hdrname = self[0].header["HDRNAME"]
        self.wcsname = self[0].header["WCSNAME"]
        self.upwcsver = self[0].header.get("UPWCSVER", "")
        self.pywcsver = self[0].header.get("PYWCSVER", "")
        self.destim = self[0].header["DESTIM"]
        self.sipname = self[0].header["SIPNAME"]
        self.idctab = self[0].header["IDCTAB"]
        self.npolfile = self[0].header["NPOLFILE"]
        self.d2imfile = self[0].header["D2IMFILE"]
        self.distname = self[0].header["DISTNAME"]

        try:
            self.vafactor = self[("SIPWCS", 1)].header.get("VAFACTOR", 1)  # None instead of 1?
        except (IndexError, KeyError):
            self.vafactor = self[0].header.get("VAFACTOR", 1)  # None instead of 1?
        self.author = self[0].header["AUTHOR"]
        self.descrip = self[0].header["DESCRIP"]

        self.fit_kws = ['HDRNAME', 'NMATCH', 'CATALOG']
        self.history = ''
        # header['HISTORY'] returns an iterable of all HISTORY values
        if 'HISTORY' in self[0].header:
            for hist in self[0].header['HISTORY']:
                self.history += hist + '\n'

        self.d2imerr = 0
        self.axiscorr = 1

    # Overridden to support the Headerlet logging features
    @classmethod
    def fromfile(cls, fileobj, mode='readonly', memmap=False,
                 save_backup=False, logging=False, logmode='w', **kwargs):
        hlet = super(cls, cls).fromfile(fileobj, mode, memmap, save_backup,
                                        **kwargs)
        if len(hlet) > 0:
            hlet.init_attrs()
        hlet.logging = logging
        init_logging('class Headerlet', level=logging, mode=logmode)
        return hlet

    @classmethod
    def fromstring(cls, data, **kwargs):
        hlet = super(cls, cls).fromstring(data, **kwargs)
        hlet.init_attrs()
        logmode = kwargs.get('logmode', 'w')
        hlet.logging = logging
        init_logging('class Headerlet', level=logging, mode=logmode)
        return hlet

    def apply_as_primary(self, fobj, attach=True, archive=True, force=False):
        """
        Copy this headerlet as a primary WCS to fobj

        Parameters
        ----------
        fobj: string, HDUList
              science file to which the headerlet should be applied
        attach: bool
              flag indicating if the headerlet should be attached as a
              HeaderletHDU to fobj. If True checks that HDRNAME is unique
              in the fobj and stops if not.
        archive: bool (default is True)
              When the distortion model in the headerlet is the same as the
              distortion model of the science file, this flag indicates if
              the primary WCS should be saved as an alternate and a headerlet
              extension.
              When the distortion models do not match this flag indicates if
              the current primary and alternate WCSs should be archived as
              headerlet extensions and alternate WCS.
        force: bool (default is False)
              When the distortion models of the headerlet and the primary do
              not match, and archive is False this flag forces an update
              of the primary
        """
        self.hverify()
        fobj, fname, close_dest = parse_filename(fobj, mode='update')
        if not self.verify_dest(fobj, fname):
            if close_dest:
                fobj.close()
            raise ValueError("Destination name does not match headerlet"
                             "Observation {0} cannot be updated with"
                             "headerlet {1}".format((fname, self.hdrname)))

        # Check to see whether the distortion model in the destination
        # matches the distortion model in the headerlet being applied

        dname = self.get_destination_model(fobj)
        dist_models_equal = self.equal_distmodel(dname)
        if not dist_models_equal and not force:
            raise ValueError("Distortion models do not match"
                             " To overwrite the distortion model, set force=True")

        orig_hlt_hdu = None
        numhlt = countExtn(fobj, 'HDRLET')
        hdrlet_extnames = list(map(str.upper, get_headerlet_kw_names(fobj)))

        # Insure that WCSCORR table has been created with all original
        # WCS's recorded prior to adding the headerlet WCS
        wcscorr.init_wcscorr(fobj)

        # start archive
        # If archive has been specified
        # regardless of whether or not the distortion models are equal...

        numsip = countExtn(self, 'SIPWCS')
        sciext_list = []
        alt_hlethdu = []
        for i in range(1, numsip + 1):
            sipheader = self[('SIPWCS', i)].header
            sciext_list.append((sipheader['TG_ENAME'], sipheader['TG_EVER']))

        if archive:
            target_ext = sciext_list[0]
            scihdr = fobj[target_ext].header

            if 'wcsname' in scihdr:
                hdrname = scihdr['WCSNAME']
                wcsname = hdrname
            else:
                hdrname = fobj[0].header['ROOTNAME'] + '_orig'
                wcsname = None

            if hdrname.upper() not in hdrlet_extnames:
                # -  if WCS has not been saved, write out WCS as headerlet extension
                # Create a headerlet for the original Primary WCS data in the file,
                # create an HDU from the original headerlet, and append it to
                # the file
                orig_hlt = create_headerlet(fobj, sciext=sciext_list,  # [target_ext],
                                            wcsname=wcsname,
                                            hdrname=hdrname,
                                            logging=self.logging)
                orig_hlt_hdu = HeaderletHDU.fromheaderlet(orig_hlt)
                numhlt += 1
                orig_hlt_hdu.header['EXTVER'] = numhlt
                logger.info(f"Created headerlet '{hdrname}' to be attached to file")
            else:
                logger.info(f"Headerlet with name '{hdrname}' is already attached")

            alt_wcs_names_dict = altwcs._alt_wcs_names(scihdr)
            alt_wcs_names = list(map(str.upper, altwcs._alt_wcs_names(scihdr).values()))

            mode = altwcs.ArchiveMode.OVERWRITE_KEY | altwcs.ArchiveMode.AUTO_RENAME

            if dist_models_equal:
                # Use the WCSNAME to determine whether or not to archive
                # Primary WCS as altwcs
                if 'hdrname' in scihdr:
                    archive_wcs = self.hdrname.upper() != scihdr['hdrname'].upper()
                    if 'wcsname' in scihdr:
                        wcsname = scihdr['wcsname'].upper()
                        archive_wcs = (archive_wcs or
                                       (self.wcsname.upper() != wcsname and
                                        wcsname not in alt_wcs_names))

                else:
                    if 'wcsname' in scihdr:
                        priwcs_name = None
                        wcsname = scihdr['wcsname'].upper()
                        archive_wcs = (self.wcsname.upper() != wcsname and
                                       wcsname not in alt_wcs_names)

                    else:
                        if 'idctab' in scihdr:
                            wcsname = ''.join(
                                ['IDC_',
                                 utils.extract_rootname(
                                     scihdr['idctab'], suffix='_idc')
                                ]
                            ).upper()
                            archive_wcs = wcsname not in alt_wcs_names

                        else:
                            wcsname = 'UNKNOWN'
                            archive_wcs = True

                if archive_wcs:
                    altwcs.archive_wcs(fobj, ext=sciext_list, mode=mode)

            else:
                # Add primary WCS to the list of WCS to be saved to headerlets:
                all_wcs_dict = alt_wcs_names_dict.copy()
                all_wcs_dict[' '] = scihdr.get('WCSNAME', ' ')

                for wcskey, hname in all_wcs_dict.items():
                    hname_u = hname.upper()
                    if hname_u not in hdrlet_extnames:
                        # create HeaderletHDU for alternate WCS now
                        alt_hlet = create_headerlet(fobj, sciext=sciext_list,
                                                    wcsname=hname, wcskey=wcskey,
                                                    hdrname=hname, sipname=None,
                                                    npolfile=None, d2imfile=None,
                                                    author=None, descrip=None, history=None,
                                                    logging=self.logging)
                        numhlt += 1
                        alt_hlet_hdu = HeaderletHDU.fromheaderlet(alt_hlet)
                        alt_hlet_hdu.header['EXTVER'] = numhlt
                        alt_hlethdu.append(alt_hlet_hdu)
                        hdrlet_extnames.append(hname_u)

                    altwcs.deleteWCS(fobj, sciext_list, wcskey=wcskey, wcsname=hname)
        self._del_dest_WCS_ext(fobj)
        for i in range(1, numsip + 1):
            target_ext = sciext_list[i - 1]
            self._del_dest_WCS(fobj, target_ext)
            sipwcs = HSTWCS(self, ('SIPWCS', i))
            idckw = sipwcs._idc2hdr()
            priwcs = sipwcs.to_fits(relax=True)
            altwcs.exclude_hst_specific(priwcs[0].header, wcskey=sipwcs.wcs.alt)
            numnpol = 1
            numd2im = 1
            if sipwcs.wcs.has_cd():
                priwcs[0].header = altwcs.pc2cd(priwcs[0].header)
            priwcs[0].header.extend(idckw)
            if 'crder1' in sipheader:
                for card in sipheader['crder*'].cards:
                    priwcs[0].header.set(card.keyword, card.value, card.comment,
                                         after='WCSNAME')
            # Update WCS with HDRNAME as well

            for kw in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND']:
                if kw in priwcs[0].header:
                    priwcs[0].header.remove(kw)

            priwcs[0].header.set('WCSNAME', self[0].header['WCSNAME'], "Coordinate system title")
            priwcs[0].header.set('WCSAXES', self[('SIPWCS', i)].header['WCSAXES'], "")
            priwcs[0].header.set('HDRNAME', self[0].header['HDRNAME'], "")
            if sipwcs.det2im1 or sipwcs.det2im2:
                try:
                    d2imerr = self[('SIPWCS', i)].header['D2IMERR*']
                    priwcs[0].header.extend(d2imerr)
                except KeyError:
                    pass
                try:
                    priwcs[0].header.append(self[('SIPWCS', i)].header.cards['D2IMEXT'])
                except KeyError:
                    pass
                if 'D2IM1.EXTVER' in priwcs[0].header:
                    priwcs[0].header['D2IM1.EXTVER'] = self[('SIPWCS', i)].header['D2IM1.EXTVER']
                    priwcs[('D2IMARR', 1)].header['EXTVER'] = self[('SIPWCS', i)].header['D2IM1.EXTVER']
                if 'D2IM2.EXTVER' in priwcs[0].header:
                    priwcs[0].header['D2IM2.EXTVER'] = self[('SIPWCS', i)].header['D2IM2.EXTVER']
                    priwcs[('D2IMARR', 2)].header['EXTVER'] = self[('SIPWCS', i)].header['D2IM2.EXTVER']
                    # D2IM1 will NOT exist for WFPC2 data...
                    if 'D2IM1.EXTVER' in priwcs[0].header:
                        # only set number of D2IM extensions to 2 if D2IM1 exists
                        numd2im = 2

            if sipwcs.cpdis1 or sipwcs.cpdis2:
                try:
                    cperr = self[('SIPWCS', i)].header['CPERR*']
                    priwcs[0].header.extend(cperr)
                except KeyError:
                    pass
                try:
                    priwcs[0].header.append(self[('SIPWCS', i)].header.cards['NPOLEXT'])
                except KeyError:
                    pass
                if 'DP1.EXTVER' in priwcs[0].header:
                    priwcs[0].header['DP1.EXTVER'] = self[('SIPWCS', i)].header['DP1.EXTVER']
                    priwcs[('WCSDVARR', 1)].header['EXTVER'] = self[('SIPWCS', i)].header['DP1.EXTVER']
                if 'DP2.EXTVER' in priwcs[0].header:
                    priwcs[0].header['DP2.EXTVER'] = self[('SIPWCS', i)].header['DP2.EXTVER']
                    priwcs[('WCSDVARR', 2)].header['EXTVER'] = self[('SIPWCS', i)].header['DP2.EXTVER']
                    numnpol = 2

            fobj[target_ext].header.update(priwcs[0].header)

            if sipwcs.cpdis1:
                whdu = priwcs[('WCSDVARR', (i - 1) * numnpol + 1)].copy()
                whdu.ver = int(self[('SIPWCS', i)].header['DP1.EXTVER'])
                fobj.append(whdu)
            if sipwcs.cpdis2:
                whdu = priwcs[('WCSDVARR', i * numnpol)].copy()
                whdu.ver = int(self[('SIPWCS', i)].header['DP2.EXTVER'])
                fobj.append(whdu)
            if sipwcs.det2im1:  # or sipwcs.det2im2:
                whdu = priwcs[('D2IMARR', (i - 1) * numd2im + 1)].copy()
                whdu.ver = int(self[('SIPWCS', i)].header['D2IM1.EXTVER'])
                fobj.append(whdu)
            if sipwcs.det2im2:
                whdu = priwcs[('D2IMARR', i * numd2im)].copy()
                whdu.ver = int(self[('SIPWCS', i)].header['D2IM2.EXTVER'])
                fobj.append(whdu)

        update_versions(self[0].header, fobj[0].header)
        #refs = update_ref_files(self[0].header, fobj[0].header)
        _ = update_ref_files(self[0].header, fobj[0].header)

        # Update the WCSCORR table with new rows from the headerlet's WCSs
        wcscorr.update_wcscorr(fobj, self, 'SIPWCS')

        # Append the original headerlet
        if archive and orig_hlt_hdu:
            fobj.append(orig_hlt_hdu)
        # Append any alternate WCS Headerlets
        if len(alt_hlethdu) > 0:
            for ahdu in alt_hlethdu:
                fobj.append(ahdu)
        if attach:
            # Finally, append an HDU for this headerlet
            self.attach_to_file(fobj)
            utils.updateNEXTENDKw(fobj)
        if close_dest:
            fobj.close()

    def apply_as_alternate(self, fobj, attach=True, wcskey=None, wcsname=None):
        """
        Copy this headerlet as an alternate WCS to fobj

        Parameters
        ----------
        fobj: string, HDUList
              science file/HDUList to which the headerlet should be applied
        attach: bool
              flag indicating if the headerlet should be attached as a
              HeaderletHDU to fobj. If True checks that HDRNAME is unique
              in the fobj and stops if not.
        wcskey: string
              Key value (A-Z, except O) for this alternate WCS
              If None, the next available key will be used
        wcsname: string
              Name to be assigned to this alternate WCS
              WCSNAME is a required keyword in a Headerlet but this allows the
              user to change it as desired.

        """
        self.hverify()
        fobj, fname, close_dest = parse_filename(fobj, mode='update')
        if not self.verify_dest(fobj, fname):
            if close_dest:
                fobj.close()
            raise ValueError("Destination name does not match headerlet"
                             "Observation %s cannot  be updated with"
                             "headerlet %s" % (fname, self.hdrname))

        # Verify whether this headerlet has the same distortion
        # found in the image being updated
        dname = self.get_destination_model(fobj)
        dist_models_equal = self.equal_distmodel(dname)
        if not dist_models_equal:
            raise ValueError("Distortion models do not match \n"
                             "Headerlet: %s \n"
                             "Destination file: %s\n"
                             "attach_to_file() can be used to append this headerlet" %
                             (self.distname, dname))

        # Insure that WCSCORR table has been created with all original
        # WCS's recorded prior to adding the headerlet WCS
        wcscorr.init_wcscorr(fobj)

        # determine value of WCSNAME to be used
        if wcsname is not None:
            wname = wcsname
        else:
            wname = self[0].header['WCSNAME']
        tg_ename = self[('SIPWCS', 1)].header['TG_ENAME']
        tg_ever = self[('SIPWCS', 1)].header['TG_EVER']
        # determine what alternate WCS this headerlet will be assigned to
        if wcskey is None:
            wkey = altwcs._next_wcskey(fobj[(tg_ename, tg_ever)].header)
        else:
            wcskey = wcskey.upper()
            available_keys = altwcs.available_wcskeys(fobj[(tg_ename, tg_ever)].header)
            if wcskey in available_keys:
                wkey = wcskey
            else:
                mess = "Observation %s already contains alternate WCS with key %s" % (fname, wcskey)
                logger.critical(mess)
                if close_dest:
                    fobj.close()
                raise ValueError(mess)
        numsip = countExtn(self, 'SIPWCS')

        log.setLevel('WARNING')
        for idx in range(1, numsip + 1):
            siphdr = self[('SIPWCS', idx)].header
            tg_ext = (siphdr['TG_ENAME'], siphdr['TG_EVER'])

            fhdr = fobj[tg_ext].header
            hwcs = pywcs.WCS(siphdr, self)
            hwcs_header = hwcs.to_header(key=wkey)
            altwcs.exclude_hst_specific(hwcs_header, wcskey=wkey)

            _idc2hdr(siphdr, fhdr, towkey=wkey)
            if hwcs.wcs.has_cd():
                hwcs_header = altwcs.pc2cd(hwcs_header, key=wkey)
            for ax in range(1, hwcs.naxis + 1):
                hwcs_header['CTYPE{0}{1}'.format(ax, wkey)] = \
                    self[('SIPWCS', 1)].header['CTYPE{0}'.format(ax)]
            fhdr.extend(hwcs_header)
            fhdr['WCSNAME' + wkey] = wname
            # also update with HDRNAME (a non-WCS-standard kw)
            for kw in self.fit_kws:
                # fhdr.insert(wind, pyfits.Card(kw + wkey,
                #                              self[0].header[kw]))
                fhdr.append(fits.Card(kw + wkey, self[0].header[kw]))

        log.setLevel(default_log_level)
        # Update the WCSCORR table with new rows from the headerlet's WCSs
        wcscorr.update_wcscorr(fobj, self, 'SIPWCS')

        if attach:
            self.attach_to_file(fobj)
            utils.updateNEXTENDKw(fobj)

        if close_dest:
            fobj.close()

    def attach_to_file(self, fobj, archive=False):
        """
        Attach Headerlet as an HeaderletHDU to a science file

        Parameters
        ----------
        fobj: string, HDUList
              science file/HDUList to which the headerlet should be applied
        archive: string
              Specifies whether or not to update WCSCORR table when attaching

        Notes
        -----
        The algorithm used by this method:
        - verify headerlet can be applied to this file (based on DESTIM)
        - verify that HDRNAME is unique for this file
        - attach as HeaderletHDU to fobj

        """
        self.hverify()
        fobj, fname, close_dest = parse_filename(fobj, mode='update')
        destver = self.verify_dest(fobj, fname)
        hdrver = self.verify_hdrname(fobj)
        if destver and hdrver:

            numhlt = countExtn(fobj, 'HDRLET')
            new_hlt = HeaderletHDU.fromheaderlet(self)
            new_hlt.header['extver'] = numhlt + 1
            fobj.append(new_hlt)
            utils.updateNEXTENDKw(fobj)
        else:
            message = "Headerlet %s cannot be attached to" % (self.hdrname)
            message += "observation %s" % (fname)
            if not destver:
                message += " * Image %s keyword ROOTNAME not equal to " % (fname)
                message += " DESTIM = '%s'\n" % (self.destim)
            if not hdrver:
                message += " * Image %s already has headerlet " % (fname)
                message += "with HDRNAME='%s'\n" % (self.hdrname)
            logger.critical(message)
        if close_dest:
            fobj.close()

    def info(self, columns=None, pad=2, maxwidth=None,
             output=None, clobber=True, quiet=False):
        """
        Prints a summary of this headerlet
        The summary includes:
        HDRNAME  WCSNAME DISTNAME SIPNAME NPOLFILE D2IMFILE

        Parameters
        ----------
        columns: list
                List of headerlet PRIMARY header keywords to report in summary
                By default (set to None), it will use the default set of keywords
                defined as the global list DEFAULT_SUMMARY_COLS
        pad: int
                Number of padding spaces to put between printed columns
                [Default: 2]
        maxwidth: int
                Maximum column width(not counting padding) for any column in summary
                By default (set to None), each column's full width will be used
        output: string (optional)
                Name of optional output file to record summary. This filename
                can contain environment variables.
                [Default: None]
        clobber: bool
                If True, will overwrite any previous output file of same name
        quiet: bool
                If True, will NOT report info to STDOUT

        """
        summary_cols, summary_dict = self.summary(columns=columns)
        print_summary(summary_cols, summary_dict, pad=pad, maxwidth=maxwidth,
                      idcol=None, output=output, clobber=clobber, quiet=quiet)

    def summary(self, columns=None):
        """
        Returns a summary of this headerlet as a dictionary

        The summary includes a summary of the distortion model as :
            HDRNAME  WCSNAME DISTNAME SIPNAME NPOLFILE D2IMFILE

        Parameters
        ----------
        columns: list
            List of headerlet PRIMARY header keywords to report in summary
            By default(set to None), it will use the default set of keywords
            defined as the global list DEFAULT_SUMMARY_COLS

        Returns
        -------
        summary: dict
            Dictionary of values for summary
        """
        if columns is None:
            summary_cols = DEFAULT_SUMMARY_COLS
        else:
            summary_cols = columns

        # Initialize summary dict based on requested columns
        summary = {}
        for kw in summary_cols:
            summary[kw] = copy.deepcopy(COLUMN_DICT)

        # Populate the summary with headerlet values
        for kw in summary_cols:
            if kw in self[0].header:
                val = self[0].header[kw]
            else:
                val = 'INDEF'
            summary[kw]['vals'].append(val)
            summary[kw]['width'].append(max(len(val), len(kw)))

        return summary_cols, summary

    def hverify(self):
        """
        Verify the headerlet file is a valid fits file and has
        the required Primary Header keywords
        """
        self.verify()
        header = self[0].header
        assert('DESTIM' in header and header['DESTIM'].strip())
        assert('HDRNAME' in header and header['HDRNAME'].strip())
        assert('UPWCSVER' in header)

    def verify_hdrname(self, dest):
        """
        Verifies that the headerlet can be applied to the observation

        Reports whether or not this file already has a headerlet with this
        HDRNAME.
        """
        unique = verify_hdrname_is_unique(dest, self.hdrname)
        logger.debug("verify_hdrname() returned %s" % unique)
        return unique

    def get_destination_model(self, dest):
        """
        Verifies that the headerlet can be applied to the observation

        Determines whether or not the file specifies the same distortion
        model/reference files.
        """
        destim_opened = False
        if not isinstance(dest, fits.HDUList):
            destim = fits.open(dest)
            destim_opened = True
        else:
            destim = dest
        dname = destim[0].header['DISTNAME'] if 'distname' in destim[0].header \
            else self.build_distname(dest)
        if destim_opened:
            destim.close()
        return dname

    def equal_distmodel(self, dmodel):
        if dmodel != self[0].header['DISTNAME']:
            if self.logging:
                    message = """
                    Distortion model in headerlet not the same as destination model
                    Headerlet model  : %s
                    Destination model: %s
                    """ % (self[0].header['DISTNAME'], dmodel)
                    logger.critical(message)
            return False
        else:
            return True

    def verify_dest(self, dest, fname):
        """
        verifies that the headerlet can be applied to the observation

        DESTIM in the primary header of the headerlet must match ROOTNAME
        of the science file (or the name of the destination file)
        """
        try:
            if not isinstance(dest, fits.HDUList):
                droot = fits.getval(dest, 'ROOTNAME')
            else:
                droot = dest[0].header['ROOTNAME']
        except KeyError:
            logger.debug("Keyword 'ROOTNAME' not found in destination file")
            droot = dest.split('.fits')[0]
        if droot == self.destim:
            logger.debug("verify_destim() returned True")
            return True
        else:
            logger.debug("verify_destim() returned False")
            logger.critical("Destination name does not match headerlet. "
                            "Observation %s cannot  be updated with"
                            "headerlet %s" % (fname, self.hdrname))
            return False

    def build_distname(self, dest):
        """
        Builds the DISTNAME for dest based on reference file names.
        """

        try:
            npolfile = dest[0].header['NPOLFILE']
        except KeyError:
            npolfile = None
        try:
            d2imfile = dest[0].header['D2IMFILE']
        except KeyError:
            d2imfile = None

        sipname, idctab = utils.build_sipname(dest, dest, None)
        npolname, npolfile = utils.build_npolname(dest, npolfile)
        d2imname, d2imfile = utils.build_d2imname(dest, d2imfile)
        dname = utils.build_distname(sipname, npolname, d2imname)
        return dname

    def tofile(self, fname, destim=None, hdrname=None, clobber=False):
        """
        Write this headerlet to a file

        Parameters
        ----------
        fname: string
               file name
        destim: string (optional)
                provide a value for DESTIM keyword
        hdrname: string (optional)
                provide a value for HDRNAME keyword
        clobber: bool
                a flag which allows to overwrte an existing file
        """
        if not destim or not hdrname:
            self.hverify()
        self.writeto(fname, overwrite=clobber)

    def _del_dest_WCS(self, dest, ext=None):
        """
        Delete the WCS of a science file extension
        """

        logger.info("Deleting all WCSs of file %s" % dest.filename())
        numext = len(dest)

        if ext:
            fext = dest[ext]
            self._remove_d2im(fext)
            self._remove_sip(fext)
            self._remove_lut(fext)
            self._remove_primary_WCS(fext)
            self._remove_idc_coeffs(fext)
            self._remove_fit_values(fext)
        else:
            for idx in range(numext):
                # Only delete WCS from extensions which may have WCS keywords
                if ('XTENSION' in dest[idx].header and dest[idx].header['XTENSION'] == 'IMAGE'):
                    self._remove_d2im(dest[idx])
                    self._remove_sip(dest[idx])
                    self._remove_lut(dest[idx])
                    self._remove_primary_WCS(dest[idx])
                    self._remove_idc_coeffs(dest[idx])
                    self._remove_fit_values(dest[idx])
        self._remove_ref_files(dest[0])

    def _del_dest_WCS_ext(self, dest):
        numwdvarr = countExtn(dest, 'WCSDVARR')
        numd2im = countExtn(dest, 'D2IMARR')
        if numwdvarr > 0:
            for idx in range(1, numwdvarr + 1):
                del dest[('WCSDVARR', idx)]
        if numd2im > 0:
            for idx in range(1, numd2im + 1):
                del dest[('D2IMARR', idx)]

    def _remove_ref_files(self, phdu):
        """
        phdu: Primary HDU
        """
        refkw = ['IDCTAB', 'NPOLFILE', 'D2IMFILE', 'SIPNAME', 'DISTNAME']
        for kw in refkw:
            try:
                phdu.header.set(kw, 'N/A')
            except KeyError:
                pass

    def _remove_fit_values(self, ext):
        """
        Remove the any existing astrometric fit values from a FITS extension
        """

        logger.debug("Removing astrometric fit values from (%s, %s)" %
                     (ext.name, ext.ver))
        dkeys = altwcs.wcskeys(ext.header)
        if 'O' in dkeys: dkeys.remove('O')  # Do not remove wcskey='O' values
        for fitkw in ['NMATCH', 'CATALOG']:
            for k in dkeys:
                fkw = (fitkw + k).rstrip()
                if fkw in ext.header:
                    del ext.header[fkw]

    def _remove_sip(self, ext):
        """
        Remove the SIP distortion of a FITS extension
        """

        logger.debug("Removing SIP distortion from (%s, %s)"
                     % (ext.name, ext.ver))
        for prefix in ['A', 'B', 'AP', 'BP']:
            try:
                order = ext.header[prefix + '_ORDER']
                del ext.header[prefix + '_ORDER']
            except KeyError:
                continue
            for i in range(order + 1):
                for j in range(order + 1):
                    key = prefix + '_%d_%d' % (i, j)
                    try:
                        del ext.header[key]
                    except KeyError:
                        pass
        try:
            ext.header.set('IDCTAB', 'N/A')
        except KeyError:
            pass

    def _remove_lut(self, ext):
        """
        Remove the Lookup Table distortion of a FITS extension
        """

        logger.debug("Removing LUT distortion from (%s, %s)"
                     % (ext.name, ext.ver))
        try:
            cpdis = ext.header['CPDIS*']
        except KeyError:
            return
        try:
            for c in range(1, len(cpdis) + 1):
                del ext.header['DP%s*...' % c]
                del ext.header[cpdis.cards[c - 1].keyword]
            del ext.header['CPERR*']
            ext.header.set('NPOLFILE', 'N/A')
            del ext.header['NPOLEXT']
        except KeyError:
            pass

    def _remove_d2im(self, ext):
        """
        Remove the Detector to Image correction of a FITS extension
        """

        logger.debug("Removing D2IM correction from (%s, %s)"
                     % (ext.name, ext.ver))
        try:
            d2imdis = ext.header['D2IMDIS*']
        except KeyError:
            return
        try:
            for c in range(1, len(d2imdis) + 1):
                del ext.header['D2IM%s*...' % c]
                del ext.header[d2imdis.cards[c - 1].keyword]
            del ext.header['D2IMERR*']
            ext.header.set('D2IMFILE', 'N/A')
            del ext.header['D2IMEXT']
        except KeyError:
            pass

    def _remove_alt_WCS(self, dest, ext):
        """
        Remove Alternate WCSs of a FITS extension.
        A WCS with wcskey 'O' is never deleted.
        """
        dkeys = altwcs.wcskeys(dest[('SCI', 1)].header)
        for val in ['O', '', ' ']:
            if val in dkeys:
                dkeys.remove(val)  # Never delete WCS with wcskey='O'

        logger.debug("Removing alternate WCSs with keys %s from %s"
                     % (dkeys, dest.filename()))
        for k in dkeys:
            altwcs.deleteWCS(dest, ext=ext, wcskey=k)

    def _remove_primary_WCS(self, ext):
        """
        Remove the primary WCS of a FITS extension
        """

        logger.debug("Removing Primary WCS from (%s, %s)"
                     % (ext.name, ext.ver))
        naxis = ext.header['NAXIS']
        for key in basic_wcs:
            for i in range(1, naxis + 1):
                try:
                    del ext.header[key + str(i)]
                except KeyError:
                    pass
        try:
            del ext.header['WCSAXES']
        except KeyError:
            pass
        try:
            del ext.header['WCSNAME']
        except KeyError:
            pass

    def _remove_idc_coeffs(self, ext):
        """
        Remove IDC coefficients of a FITS extension
        """

        logger.debug("Removing IDC coefficient from (%s, %s)"
                     % (ext.name, ext.ver))
        coeffs = ['OCX10', 'OCX11', 'OCY10', 'OCY11', 'IDCSCALE']
        for k in coeffs:
            try:
                del ext.header[k]
            except KeyError:
                pass


@with_logging
def _idc2hdr(fromhdr, tohdr, towkey=' '):
    """
    Copy the IDC (HST specific) keywords from one header to another

    """
    # save some of the idc coefficients
    coeffs = ['OCX10', 'OCX11', 'OCY10', 'OCY11', 'IDCSCALE']
    for c in coeffs:
        try:
            tohdr[c + towkey] = fromhdr[c]
            logger.debug("Copied %s to header")
        except KeyError:
            continue


def get_extname_extver_list(fobj, sciext):
    """
    Create a list of (EXTNAME, EXTVER) tuples

    Based on sciext keyword (see docstring for create_headerlet)
    walk throughh the file and convert extensions in `sciext` to
    valid (EXTNAME, EXTVER) tuples.
    """
    extlist = []
    if isinstance(sciext, int):
        if sciext == 0:
            extname = 'PRIMARY'
            extver = 1
        else:
            try:
                extname = fobj[sciext].header['EXTNAME']
            except KeyError:
                extname = ""
            try:
                extver = fobj[sciext].header['EXTVER']
            except KeyError:
                extver = 1
        extlist.append((extname, extver))
    elif isinstance(sciext, str):
        if sciext == 'PRIMARY':
            extname = "PRIMARY"
            extver = 1
            extlist.append((extname, extver))
        else:
            for ext in fobj:
                try:
                    extname = ext.header['EXTNAME']
                except KeyError:
                    continue
                if extname.upper() == sciext.upper():
                    try:
                        extver = ext.header['EXTVER']
                    except KeyError:
                        extver = 1
                    extlist.append((extname, extver))
    elif isinstance(sciext, list):
        if isinstance(sciext[0], int):
            for i in sciext:
                try:
                    extname = fobj[i].header['EXTNAME']
                except KeyError:
                    if i == 0:
                        extname = "PRIMARY"
                        extver = 1
                    else:
                        extname = ""
                try:
                    extver = fobj[i].header['EXTVER']
                except KeyError:
                    extver = 1
                extlist.append((extname, extver))
        else:
            extlist = sciext[:]
    else:
        errstr = "Expected sciext to be a list of FITS extensions with science data\n" + \
            "    a valid EXTNAME string, or an integer."
        logger.critical(errstr)
        raise ValueError
    return extlist


class HeaderletHDU(fits.hdu.nonstandard.FitsHDU):
    """
    A non-standard extension HDU for encapsulating Headerlets in a file.  These
    HDUs have an extension type of HDRLET and their EXTNAME is derived from the
    Headerlet's HDRNAME.

    The data itself is a FITS file embedded within the HDU data.  The file name
    is derived from the HDRNAME keyword, and should be in the form
    `<HDRNAME>_hdr.fits`.  If the COMPRESS keyword evaluates to `True`, the tar
    file is compressed with gzip compression.

    The structure of this HDU is the same as that proposed for the 'FITS'
    extension type proposed here:
    http://listmgr.cv.nrao.edu/pipermail/fitsbits/2002-April/thread.html

    The Headerlet contained in the HDU's data can be accessed by the
    `headerlet` attribute.
    """

    _extension = 'HDRLET'

    @lazyproperty
    def headerlet(self):
        """Return the encapsulated headerlet as a Headerlet object.

        This is similar to the hdulist property inherited from the FitsHDU
        class, though the hdulist property returns a normal HDUList object.
        """

        return Headerlet(self.hdulist)

    @classmethod
    def fromheaderlet(cls, headerlet, compress=False):
        """
        Creates a new HeaderletHDU from a given Headerlet object.

        Parameters
        ----------
        headerlet : `Headerlet`
            A valid Headerlet object.

        compress : bool, optional
            Gzip compress the headerlet data.

        Returns
        -------
        hlet : `HeaderletHDU`
            A `HeaderletHDU` object for the given `Headerlet` that can be
            attached as an extension to an existing `HDUList`.
        """

        # TODO: Perhaps check that the given object is in fact a valid
        # Headerlet
        hlet = cls.fromhdulist(headerlet, compress)

        # Add some more headerlet-specific keywords to the header
        phdu = headerlet[0]

        if 'SIPNAME' in phdu.header:
            sipname = phdu.header['SIPNAME']
        else:
            sipname = phdu.header['WCSNAME']

        hlet.header['HDRNAME'] = (phdu.header['HDRNAME'],
                                  phdu.header.comments['HDRNAME'])
        hlet.header['DATE'] = (phdu.header['DATE'],
                               phdu.header.comments['DATE'])
        hlet.header['SIPNAME'] = (sipname, 'SIP distortion model name')
        hlet.header['WCSNAME'] = (phdu.header['WCSNAME'], 'WCS name')
        hlet.header['DISTNAME'] = (phdu.header['DISTNAME'],
                                   'Distortion model name')
        hlet.header['NPOLFILE'] = (phdu.header['NPOLFILE'],
                                   phdu.header.comments['NPOLFILE'])
        hlet.header['D2IMFILE'] = (phdu.header['D2IMFILE'],
                                   phdu.header.comments['D2IMFILE'])
        hlet.header['EXTNAME'] = (cls._extension, 'Extension name')

        return hlet


fits.register_hdu(HeaderletHDU)
