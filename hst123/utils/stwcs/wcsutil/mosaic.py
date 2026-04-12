import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import string

from stsci.tools import parseinput, irafglob
from ..distortion import utils
from .. import wcsutil
from ..wcsutil import altwcs


def vmosaic(fnames, outwcs=None, ref_wcs=None, ext=None, extname=None, undistort=True,
            wkey='V', wname='VirtualMosaic', plot=False, clobber=False):
    """
    Create a virtual mosaic using the WCS of the input images.

    Parameters
    ----------
    fnames: a string or a list
              a string or a list of filenames, or a list of wcsutil.HSTWCS objects
    outwcs: an HSTWCS object
             if given, represents the output tangent plane
             if None, the output WCS is calculated from the input observations.
    ref_wcs: an HSTWCS object
             if output wcs is not given, this will be used as a reference for the
             calculation of the output WCS. If ref_wcs is None and outwcs is None,
             then the first observation in th einput list is used as reference.
    ext:    an int, a tuple or a list
              an int - represents a FITS extension, e.g. 0 is the primary HDU
              a tuple - uses the notation (extname, extver), e.g. ('sci',1)
              Can be a list of integers or tuples representing FITS extensions
    extname: string
              the value of the EXTNAME keyword for the extensions to be used in the mosaic
    undistort: bool (default: True)
               undistort (or not) the output WCS
    wkey:   string
              default: 'V'
              one character A-Z to be used to record the virtual mosaic WCS as
              an alternate WCS in the headers of the input files.
    wname:  string
              default: 'VirtualMosaic
              a string to be used as a WCSNAME value for the alternate WCS representign
              the virtual mosaic
    plot:   boolean
              if True and matplotlib is installed will make a plot of the tangent plane
              and the location of the input observations.
    clobber: bool
              This covers the case when an alternate WCS with the requested key
              already exists in the header of the input files.
              if clobber is True, it will be overwritten
              if False, it will compute the new one but will not write it to the headers.

    Notes
    -----
    The algorithm is:
    1. If output WCS is not given it is calculated from the input WCSes.
       The first image is used as a reference, if no reference is given.
       This represents the virtual mosaic WCS.
    2. For each input observation/chip, an HSTWCS object is created
       and its footprint on the sky is calculated (using only the four corners).
    3. For each input observation the footprint is projected on the output
       tangent plane and the virtual WCS is recorded in the header.
    """
    wcsobjects = readWCS(fnames, ext, extname)
    if outwcs is not None:
        outwcs = outwcs.deepcopy()
    else:
        if ref_wcs is not None:
            outwcs = utils.output_wcs(wcsobjects, ref_wcs=ref_wcs, undistort=undistort)
        else:
            outwcs = utils.output_wcs(wcsobjects, undistort=undistort)
    if plot:
        outc = np.array([[0., 0], [outwcs.pixel_shape[0], 0],
                         [outwcs.pixel_shape[0], outwcs.pixel_shape[1]],
                         [0, outwcs.pixel_shape[1]], [0, 0]])
        plt.plot(outc[:, 0], outc[:, 1])
    for wobj in wcsobjects:
        outcorners = outwcs.wcs_world2pix(wobj.calc_footprint(), 1)
        if plot:
            plt.plot(outcorners[:, 0], outcorners[:, 1])
        objwcs = outwcs.deepcopy()
        objwcs.wcs.crpix = objwcs.wcs.crpix - (outcorners[0])
        updatehdr(wobj.filename, objwcs, wkey=wkey, wcsname=wname, ext=wobj.extname,
                  clobber=clobber)
    return outwcs


def updatehdr(fname, wcsobj, wkey, wcsname, ext=1, clobber=False):
    hdr = fits.getheader(fname, ext=ext)
    wkey_hdr = wkey.upper().strip()
    if len(wkey) != 1 or wkey_hdr not in string.ascii_uppercase:
        raise KeyError("'wkey' must be one character: A-Z or space (' ')")

    if wkey not in altwcs.available_wcskeys(hdr):
        if not clobber:
            raise ValueError("'wkey' '{:.1s}' is already in use. "
                             "Use clobber=True to overwrite it or "
                             "specify a different key.".format(wkey))
        else:
            altwcs.deleteWCS(fname, ext=ext, wcskey='V')

    f = fits.open(fname, mode='update')
    hwcs = wcs2header(wcsobj)
    wcsnamekey = 'WCSNAME' + wkey_hdr
    f[ext].header[wcsnamekey] = wcsname
    for k in hwcs:
        f[ext].header[k[: 7] + wkey_hdr] = hwcs[k]
    f.close()


def wcs2header(wcsobj):
    h = altwcs.exclude_hst_specific(wcsobj.to_header(), wcskey=wcsobj.wcs.alt)

    if wcsobj.wcs.has_cd():
        altwcs.pc2cd(h)
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    orient = np.rad2deg(np.arctan2(h['CD1_2'], h['CD2_2']))
    h['ORIENT'] = orient
    return h


def readWCS(input, exts=None, extname=None):
    if isinstance(input, str):
        if input[0] == '@':
            # input is an @ file
            filelist = irafglob.irafglob(input)
        else:
            try:
                filelist, output = parseinput.parseinput(input)
            except IOError: raise
    elif isinstance(input, list):
        if isinstance(input[0], wcsutil.HSTWCS):
            # a list of HSTWCS objects
            return input
        else:
            filelist = input[:]
    wcso = []
    fomited = []
    # figure out which FITS extension(s) to use
    if exts is None and extname is None:
        # Assume it's simple FITS and the data is in the primary HDU
        for f in filelist:
            try:
                wcso = wcsutil.HSTWCS(f)
            except AttributeError:
                fomited.append(f)
                continue
    elif exts is not None and validateExt(exts):
        exts = [exts]
        for f in filelist:
            try:
                wcso.extend([wcsutil.HSTWCS(f, ext=e) for e in exts])
            except KeyError:
                fomited.append(f)
                continue
    elif extname is not None:
        for f in filelist:
            fobj = fits.open(f)
            for i in range(len(fobj)):
                try:
                    ename = fobj[i].header['EXTNAME']
                except KeyError:
                    continue
                if ename.lower() == extname.lower():
                    wcso.append(wcsutil.HSTWCS(f, ext=i))
                else:
                    continue
            fobj.close()
    if fomited != []:
        print("These files were skipped:")
        for f in fomited:
            print(f)
    return wcso


def validateExt(ext):
    if not isinstance(ext, int) and not isinstance(ext, tuple) \
       and not isinstance(ext, list):
        print("Ext must be integer, tuple, a list of int extension numbers, \
        or a list of tuples representing a fits extension, for example ('sci', 1).")
        return False
    else:
        return True

