import os
import warnings

import numpy as np
from numpy import linalg
from astropy import wcs as pywcs

from .. import updatewcs
from numpy import sqrt, arctan2
from stsci.tools import fileutil


def output_wcs(list_of_wcsobj, ref_wcs=None, owcs=None, undistort=True):
    """
    Create an output WCS.

    Parameters
    ----------
    list_of_wcsobj: Python list
                    a list of HSTWCS objects
    ref_wcs: an HSTWCS object
             to be used as a reference WCS, in case outwcs is None.
             if ref_wcs is None (default), the first member of the list
             is used as a reference
    outwcs:  an HSTWCS object
             the tangent plane defined by this object is used as a reference
    undistort: bool (default-True)
              a flag whether to create an undistorted output WCS
    """
    fra_dec = np.vstack([w.calc_footprint() for w in list_of_wcsobj])
    wcsname = list_of_wcsobj[0].wcs.name

    # This new algorithm may not be strictly necessary, but it may be more
    # robust in handling regions near the poles or at 0h RA.
    crval1, crval2 = computeFootprintCenter(fra_dec)

    crval = np.array([crval1, crval2], dtype=np.float64)  # this value is now zero-based
    if owcs is None:
        if ref_wcs is None:
            ref_wcs = list_of_wcsobj[0].deepcopy()
        if undistort:
            # outwcs = undistortWCS(ref_wcs)
            outwcs = make_orthogonal_cd(ref_wcs)
        else:
            outwcs = ref_wcs.deepcopy()
        outwcs.wcs.crval = crval
        outwcs.wcs.set()
        outwcs.pscale = sqrt(outwcs.wcs.cd[0, 0] ** 2 + outwcs.wcs.cd[1, 0] ** 2) * 3600.
        outwcs.orientat = arctan2(outwcs.wcs.cd[0, 1], outwcs.wcs.cd[1, 1]) * 180. / np.pi
    else:
        outwcs = owcs.deepcopy()
        outwcs.pscale = sqrt(outwcs.wcs.cd[0, 0] ** 2 + outwcs.wcs.cd[1, 0] ** 2) * 3600.
        outwcs.orientat = arctan2(outwcs.wcs.cd[0, 1], outwcs.wcs.cd[1, 1]) * 180. / np.pi

    tanpix = outwcs.wcs.s2p(fra_dec, 0)['pixcrd']

    _naxis1 = int(np.ceil(tanpix[:, 0].max() - tanpix[:, 0].min()))
    _naxis2 = int(np.ceil(tanpix[:, 1].max() - tanpix[:, 1].min()))
    outwcs.pixel_shape = (_naxis1, _naxis2)
    crpix = np.array([_naxis1 / 2., _naxis2 / 2.], dtype=np.float64)
    outwcs.wcs.crpix = crpix
    outwcs.wcs.set()
    tanpix = outwcs.wcs.s2p(fra_dec, 0)['pixcrd']

    # shift crpix to take into account (floating-point value of) position of
    # corner pixel relative to output frame size: no rounding necessary...
    newcrpix = np.array([crpix[0] + tanpix[:, 0].min(), crpix[1] +
                         tanpix[:, 1].min()])

    newcrval = outwcs.wcs.p2s([newcrpix], 1)['world'][0]
    outwcs.wcs.crval = newcrval
    outwcs.wcs.set()
    outwcs.wcs.name = wcsname  # keep track of label for this solution
    return outwcs


def computeFootprintCenter(edges):
    """ Geographic midpoint in spherical coords for points defined by footprints.
        Algorithm derived from: http://www.geomidpoint.com/calculation.html

        This algorithm should be more robust against discontinuities at the poles.
    """
    alpha = np.deg2rad(edges[:, 0])
    dec = np.deg2rad(edges[:, 1])

    xmean = np.mean(np.cos(dec) * np.cos(alpha))
    ymean = np.mean(np.cos(dec) * np.sin(alpha))
    zmean = np.mean(np.sin(dec))

    crval1 = np.rad2deg(np.arctan2(ymean, xmean)) % 360.0
    crval2 = np.rad2deg(np.arctan2(zmean, np.sqrt(xmean * xmean + ymean * ymean)))

    return crval1, crval2


def make_orthogonal_cd(wcs):
    """ Create a perfect (square, orthogonal, undistorted) CD matrix from the
        input WCS.
    """
    # get determinant of the CD matrix:
    cd = wcs.celestial.pixel_scale_matrix

    if hasattr(wcs, 'idcv2ref') and wcs.idcv2ref is not None:
        # Convert the PA_V3 orientation to the orientation at the aperture
        # This is for the reference chip only - we use this for the
        # reference tangent plane definition
        # It has the same orientation as the reference chip
        pv = updatewcs.makewcs.troll(wcs.pav3, wcs.wcs.crval[1], wcs.idcv2ref, wcs.idcv3ref)
        # Add the chip rotation angle
        if wcs.idctheta:
            pv += wcs.idctheta
        cs = np.cos(np.deg2rad(pv))
        sn = np.sin(np.deg2rad(pv))
        pvmat = np.dot(np.array([[cs, sn], [-sn, cs]]), wcs.parity)
        rot = np.arctan2(pvmat[0, 1], pvmat[1, 1])

        det = linalg.det(wcs.parity)
        if hasattr(wcs, 'idcscale') and wcs.idcscale is not None:
            scale = (wcs.idcscale) / 3600.  # HST pixel scale provided
    else:

        det = linalg.det(cd)

        # find pixel scale:
        if hasattr(wcs, 'idcscale') and wcs.idcscale is not None:
            scale = (wcs.idcscale) / 3600.  # HST pixel scale provided
        else:
            warnings.warn("IDCSCALE is missing, computing it from CD matrix.")
            scale = np.sqrt(np.abs(det))  # find as sqrt(pixel area)

        # find Y-axis orientation:
        if hasattr(wcs, 'orientat'):
            rot = np.deg2rad(wcs.orientat)  # use HST ORIENTAT
        else:
            rot = np.arctan2(wcs.wcs.cd[0, 1], wcs.wcs.cd[1, 1])  # angle of the Y-axis

    par = -1 if det < 0.0 else 1

    # create a perfectly square, orthogonal WCS
    sn = np.sin(rot)
    cs = np.cos(rot)
    orthogonal_cd = scale * np.array([[par * cs, sn], [-par * sn, cs]])

    lin_wcsobj = pywcs.WCS()
    lin_wcsobj.wcs.cd = orthogonal_cd
    lin_wcsobj.wcs.set()
    lin_wcsobj.orientat = arctan2(lin_wcsobj.wcs.cd[0, 1], lin_wcsobj.wcs.cd[1, 1]) * 180. / np.pi
    lin_wcsobj.pscale = sqrt(lin_wcsobj.wcs.cd[0, 0] ** 2 + lin_wcsobj.wcs.cd[1, 0] ** 2) * 3600.
    lin_wcsobj.wcs.crval = np.array([0., 0.])
    lin_wcsobj.wcs.crpix = np.array([0., 0.])
    lin_wcsobj.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    lin_wcsobj.wcs.set()

    return lin_wcsobj


def undistortWCS(wcsobj):
    """
    Creates an undistorted linear WCS by applying the IDCTAB distortion model
    to a 3-point square. The new ORIENTAT angle is calculated as well as the
    plate scale in the undistorted frame.
    """
    assert isinstance(wcsobj, pywcs.WCS)
    from . import coeff_converter

    cx, cy = coeff_converter.sip2idc(wcsobj)
    # cx, cy can be None because either there is no model available
    # or updatewcs was not run.
    if cx is None or cy is None:
        if foundIDCTAB(wcsobj.idctab):
            m = """IDCTAB is present but distortion model is missing.
            Run updatewcs() to update the headers or
            pass 'undistort=False' keyword to output_wcs().\n
            """
            raise RuntimeError(m)
        else:
            print('Distortion model is not available, using input reference image for output WCS.\n')
            return wcsobj.copy()
    crpix1 = wcsobj.wcs.crpix[0]
    crpix2 = wcsobj.wcs.crpix[1]
    xy = np.array([(crpix1, crpix2), (crpix1 + 1., crpix2),
                   (crpix1, crpix2 + 1.)], dtype=np.double)
    offsets = np.array([wcsobj.ltv1, wcsobj.ltv2])
    px = xy + offsets
    # order = wcsobj.sip.a_order
    pscale = wcsobj.idcscale
    # pixref = np.array([wcsobj.sip.SIPREF1, wcsobj.sip.SIPREF2])

    tan_pix = apply_idc(px, cx, cy, wcsobj.wcs.crpix, pscale, order=1)
    xc = tan_pix[:, 0]
    yc = tan_pix[:, 1]
    am = xc[1] - xc[0]
    bm = xc[2] - xc[0]
    cm = yc[1] - yc[0]
    dm = yc[2] - yc[0]
    cd_mat = np.array([[am, bm], [cm, dm]], dtype=np.double)

    # Check the determinant for singularity
    _det = (am * dm) - (bm * cm)
    if (_det == 0.0):
        print('Singular matrix in updateWCS, aborting ...')
        return

    lin_wcsobj = pywcs.WCS()
    cd_inv = np.linalg.inv(cd_mat)
    cd = np.dot(wcsobj.wcs.cd, cd_inv).astype(np.float64)
    lin_wcsobj.wcs.cd = cd
    lin_wcsobj.wcs.set()
    lin_wcsobj.orientat = arctan2(lin_wcsobj.wcs.cd[0, 1], lin_wcsobj.wcs.cd[1, 1]) * 180. / np.pi
    lin_wcsobj.pscale = sqrt(lin_wcsobj.wcs.cd[0, 0] ** 2 + lin_wcsobj.wcs.cd[1, 0] ** 2) * 3600.
    lin_wcsobj.wcs.crval = np.array([0., 0.])
    lin_wcsobj.wcs.crpix = np.array([0., 0.])
    lin_wcsobj.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    lin_wcsobj.wcs.set()
    return lin_wcsobj


def apply_idc(pixpos, cx, cy, pixref, pscale=None, order=None):
    """
    Apply the IDCTAB polynomial distortion model to pixel positions.
    pixpos must be already corrected for ltv1/2.

    Parameters
    ----------
    pixpos: a 2D numpy array of (x,y) pixel positions to be distortion corrected
    cx, cy: IDC model distortion coefficients
    pixref: reference opixel position

    """
    if cx is None:
        return pixpos

    if order is None:
        print('Unknown order of distortion model \n')
        return pixpos
    if pscale is None:
        print('Unknown model plate scale\n')
        return pixpos

    # Apply in the same way that 'drizzle' would...
    _cx = cx / pscale
    _cy = cy / pscale
    _p = pixpos

    # Do NOT include any zero-point terms in CX,CY here
    # as they should not be scaled by plate-scale like rest
    # of coeffs...  This makes the computations consistent
    # with 'drizzle'.  WJH 17-Feb-2004
    _cx[0, 0] = 0.
    _cy[0, 0] = 0.

    dxy = _p - pixref
    # Apply coefficients from distortion model here...

    c = _p * 0.
    for i in range(order + 1):
        for j in range(i + 1):
            c[:, 0] = c[:, 0] + _cx[i][j] * pow(dxy[:, 0], j) * pow(dxy[:, 1], (i - j))
            c[:, 1] = c[:, 1] + _cy[i][j] * pow(dxy[:, 0], j) * pow(dxy[:, 1], (i - j))

    return c


def foundIDCTAB(idctab):
    idctab_found = True
    try:
        idctab = fileutil.osfn(idctab)
        if idctab == 'N/A' or idctab == "":
            idctab_found = False
        if os.path.exists(idctab):
            idctab_found = True
        else:
            idctab_found = False
    except KeyError:
        idctab_found = False
    return idctab_found
