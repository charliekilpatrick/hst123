import numpy as np
import logging
import time
from math import sin, sqrt, pow, cos, asin, atan2, pi

from stsci.tools import fileutil
from . import utils

logger = logging.getLogger(__name__)


class MakeWCS:
    """
    Recompute basic WCS keywords based on PA_V3 and distortion model.

    Notes
    -----
    - Compute the reference chip WCS:

        -- CRVAL: transform the model XREF/YREF to the sky
        -- PA_V3 is calculated at the target position and adjusted
           for each chip orientation
        -- CD: PA_V3 and the model scale are used to cnstruct a CD matrix

    - Compute the second chip WCS:

        -- CRVAL: - the distance between the zero points of the two
                  chip models on the sky
        -- CD matrix: first order coefficients are added to the components
            of this distance and transfered on the sky. The difference
            between CRVAL and these vectors is the new CD matrix for each chip.
        -- CRPIX: chip's model zero point in pixel space (XREF/YREF)

    - Time dependent distortion correction is applied to both chips' V2/V3 values.

    """
    tdd_xyref = {1: [2048, 3072], 2: [2048, 1024]}

    def updateWCS(cls, ext_wcs, ref_wcs):
        """
        recomputes the basic WCS kw
        """
        logger.info("Starting MakeWCS: %s", time.asctime())
        if not ext_wcs.idcmodel:
            logger.info("IDC model not found, turning off Makewcs")
            return {}
        ltvoff, offshift = cls.getOffsets(ext_wcs)

        v23_corr = cls.zero_point_corr(ext_wcs)
        rv23_corr = cls.zero_point_corr(ref_wcs)

        cls.uprefwcs(ext_wcs, ref_wcs, rv23_corr, ltvoff, offshift)
        cls.upextwcs(ext_wcs, ref_wcs, v23_corr, rv23_corr, ltvoff, offshift)

        kw2update = {'CD1_1': ext_wcs.wcs.cd[0, 0],
                     'CD1_2': ext_wcs.wcs.cd[0, 1],
                     'CD2_1': ext_wcs.wcs.cd[1, 0],
                     'CD2_2': ext_wcs.wcs.cd[1, 1],
                     'CRVAL1': ext_wcs.wcs.crval[0],
                     'CRVAL2': ext_wcs.wcs.crval[1],
                     'CRPIX1': ext_wcs.wcs.crpix[0],
                     'CRPIX2': ext_wcs.wcs.crpix[1],
                     'IDCTAB': ext_wcs.idctab,
                     'OCX10': ext_wcs.idcmodel.cx[1, 0],
                     'OCX11': ext_wcs.idcmodel.cx[1, 1],
                     'OCY10': ext_wcs.idcmodel.cy[1, 0],
                     'OCY11': ext_wcs.idcmodel.cy[1, 1]
                     }
        return kw2update

    updateWCS = classmethod(updateWCS)

    def upextwcs(cls, ext_wcs, ref_wcs, v23_corr, rv23_corr, loff, offsh):
        """
        updates an extension wcs
        """
        ltvoffx, ltvoffy = loff[0], loff[1]
        offshiftx, offshifty = offsh[0], offsh[1]
        ltv1 = ext_wcs.ltv1
        ltv2 = ext_wcs.ltv2
        if ltv1 != 0. or ltv2 != 0.:
            offsetx = ext_wcs.wcs.crpix[0] - ltv1 - ext_wcs.idcmodel.refpix['XREF']
            offsety = ext_wcs.wcs.crpix[1] - ltv2 - ext_wcs.idcmodel.refpix['YREF']
            ext_wcs.idcmodel.shift(offsetx, offsety)

        fx, fy = ext_wcs.idcmodel.cx, ext_wcs.idcmodel.cy

        ext_wcs.setPscale()
        tddscale = (ref_wcs.pscale / fx[1, 1])
        v2 = ext_wcs.idcmodel.refpix['V2REF'] + v23_corr[0, 0] * tddscale
        v3 = ext_wcs.idcmodel.refpix['V3REF'] - v23_corr[1, 0] * tddscale
        v2ref = ref_wcs.idcmodel.refpix['V2REF'] + rv23_corr[0, 0] * tddscale
        v3ref = ref_wcs.idcmodel.refpix['V3REF'] - rv23_corr[1, 0] * tddscale

        R_scale = ref_wcs.idcmodel.refpix['PSCALE'] / 3600.0
        off = sqrt((v2 - v2ref) ** 2 + (v3 - v3ref) ** 2) / (R_scale * 3600.0)

        if v3 == v3ref:
            theta = 0.0
        else:
            theta = atan2(ext_wcs.parity[0][0] * (v2 - v2ref),
                          ext_wcs.parity[1][1] * (v3 - v3ref))

        if ref_wcs.idcmodel.refpix['THETA']: theta += ref_wcs.idcmodel.refpix['THETA'] * pi / 180.0

        dX = (off * sin(theta)) + offshiftx
        dY = (off * cos(theta)) + offshifty

        px = np.array([[dX, dY]])
        newcrval = ref_wcs.wcs.p2s(px, 1)['world'][0]
        newcrpix = np.array([ext_wcs.idcmodel.refpix['XREF'] + ltvoffx,
                             ext_wcs.idcmodel.refpix['YREF'] + ltvoffy])
        ext_wcs.wcs.crval = newcrval
        ext_wcs.wcs.crpix = newcrpix
        ext_wcs.wcs.set()

        # Create a small vector, in reference image pixel scale
        delmat = np.array([[fx[1, 1], fy[1, 1]],
                           [fx[1, 0], fy[1, 0]]]) / R_scale / 3600.

        # Account for subarray offset
        # Angle of chip relative to chip
        if ext_wcs.idcmodel.refpix['THETA']:
            dtheta = ext_wcs.idcmodel.refpix['THETA'] - ref_wcs.idcmodel.refpix['THETA']
        else:
            dtheta = 0.0

        rrmat = fileutil.buildRotMatrix(dtheta)
        # Rotate the vectors
        dxy = np.dot(delmat, rrmat)
        wc = ref_wcs.wcs.p2s((px + dxy), 1)['world']

        # Calculate the new CDs and convert to degrees
        cd11 = utils.diff_angles(wc[0, 0], newcrval[0]) * cos(newcrval[1] * pi / 180.0)
        cd12 = utils.diff_angles(wc[1, 0], newcrval[0]) * cos(newcrval[1] * pi / 180.0)
        cd21 = utils.diff_angles(wc[0, 1], newcrval[1])
        cd22 = utils.diff_angles(wc[1, 1], newcrval[1])
        cd = np.array([[cd11, cd12], [cd21, cd22]])
        ext_wcs.wcs.cd = cd
        ext_wcs.wcs.set()

    upextwcs = classmethod(upextwcs)

    def uprefwcs(cls, ext_wcs, ref_wcs, rv23_corr_tdd, loff, offsh):
        """
        Updates the reference chip
        """
        ltvoffx, ltvoffy = loff[0], loff[1]
        ltv1 = ref_wcs.ltv1
        ltv2 = ref_wcs.ltv2
        if ref_wcs.ltv1 != 0. or ref_wcs.ltv2 != 0.:
            offsetx = ref_wcs.wcs.crpix[0] - ltv1 - ref_wcs.idcmodel.refpix['XREF']
            offsety = ref_wcs.wcs.crpix[1] - ltv2 - ref_wcs.idcmodel.refpix['YREF']
            ref_wcs.idcmodel.shift(offsetx, offsety)

        # rfx, rfy = ref_wcs.idcmodel.cx, ref_wcs.idcmodel.cy

        # offshift = offsh
        dec = ref_wcs.wcs.crval[1]
        tddscale = (ref_wcs.pscale / ref_wcs.idcmodel.cx[1, 1])
        rv23 = [ref_wcs.idcmodel.refpix['V2REF'] + (rv23_corr_tdd[0, 0] * tddscale),
                ref_wcs.idcmodel.refpix['V3REF'] - (rv23_corr_tdd[1, 0] * tddscale)]
        # Get an approximate reference position on the sky
        rref = np.array([[ref_wcs.idcmodel.refpix['XREF'] + ltvoffx,
                          ref_wcs.idcmodel.refpix['YREF'] + ltvoffy]])

        crval = ref_wcs.wcs.p2s(rref, 1)['world'][0]
        # Convert the PA_V3 orientation to the orientation at the aperture
        # This is for the reference chip only - we use this for the
        # reference tangent plane definition
        # It has the same orientation as the reference chip
        pv = troll(ext_wcs.pav3, dec, rv23[0], rv23[1])
        # Add the chip rotation angle
        if ref_wcs.idcmodel.refpix['THETA']:
            pv += ref_wcs.idcmodel.refpix['THETA']

        # Set values for the rest of the reference WCS
        ref_wcs.wcs.crval = crval
        ref_wcs.wcs.crpix = np.array([0.0, 0.0]) + offsh
        parity = ref_wcs.parity
        R_scale = ref_wcs.idcmodel.refpix['PSCALE'] / 3600.0
        cd11 = parity[0][0] * cos(pv * pi / 180.0) * R_scale
        cd12 = parity[0][0] * -sin(pv * pi / 180.0) * R_scale
        cd21 = parity[1][1] * sin(pv * pi / 180.0) * R_scale
        cd22 = parity[1][1] * cos(pv * pi / 180.0) * R_scale

        rcd = np.array([[cd11, cd12], [cd21, cd22]])
        ref_wcs.wcs.cd = rcd
        ref_wcs.wcs.set()

    uprefwcs = classmethod(uprefwcs)

    def zero_point_corr(cls, hwcs):
        if hwcs.idcmodel.refpix['skew_coeffs'] is not None and 'TDD_CY_BETA' in hwcs.idcmodel.refpix['skew_coeffs']:
            v23_corr = np.array([[0.], [0.]])
            return v23_corr
        else:
            try:
                alpha = hwcs.idcmodel.refpix['TDDALPHA']
                beta = hwcs.idcmodel.refpix['TDDBETA']
            except KeyError:
                alpha = 0.0
                beta = 0.0
                v23_corr = np.array([[0.], [0.]])
                logger.debug("TDD Zero point correction for chip"
                             "{0} defaulted to: {1}".format(hwcs.chip, v23_corr))
                return v23_corr

        tdd = np.array([[beta, alpha], [alpha, -beta]])
        mrotp = fileutil.buildRotMatrix(2.234529) / 2048.
        xy0 = np.array([[cls.tdd_xyref[hwcs.chip][0] - 2048.],
                        [cls.tdd_xyref[hwcs.chip][1] - 2048.]])
        v23_corr = np.dot(mrotp, np.dot(tdd, xy0)) * 0.05
        logger.debug("TDD Zero point correction for chip {0}: {1}".format(hwcs.chip, v23_corr))
        return v23_corr

    zero_point_corr = classmethod(zero_point_corr)

    def getOffsets(cls, ext_wcs):
        ltv1 = ext_wcs.ltv1
        ltv2 = ext_wcs.ltv2

        offsetx = ext_wcs.wcs.crpix[0] - ltv1 - ext_wcs.idcmodel.refpix['XREF']
        offsety = ext_wcs.wcs.crpix[1] - ltv2 - ext_wcs.idcmodel.refpix['YREF']

        shiftx = ext_wcs.idcmodel.refpix['XREF'] + ltv1
        shifty = ext_wcs.idcmodel.refpix['YREF'] + ltv2
        if ltv1 != 0. or ltv2 != 0.:
            ltvoffx = ltv1 + offsetx
            ltvoffy = ltv2 + offsety
            offshiftx = offsetx + shiftx
            offshifty = offsety + shifty
        else:
            ltvoffx = 0.
            ltvoffy = 0.
            offshiftx = 0.
            offshifty = 0.

        ltvoff = np.array([ltvoffx, ltvoffy])
        offshift = np.array([offshiftx, offshifty])
        return ltvoff, offshift

    getOffsets = classmethod(getOffsets)


def troll(roll, dec, v2, v3):
    """ Computes the roll angle at the target position based on:
        the roll angle at the V1 axis(roll),
        the dec of the target(dec), and
        the V2/V3 position of the aperture (v2,v3) in arcseconds.

        Based on algorithm provided by Colin Cox and used in
        Generic Conversion at STScI.
    """
    # Convert all angles to radians
    _roll = np.deg2rad(roll)
    _dec = np.deg2rad(dec)
    _v2 = np.deg2rad(v2 / 3600.)
    _v3 = np.deg2rad(v3 / 3600.)

    # compute components
    sin_rho = sqrt((pow(sin(_v2), 2) + pow(sin(_v3), 2)) - (pow(sin(_v2), 2) * pow(sin(_v3), 2)))
    rho = asin(sin_rho)
    beta = asin(sin(_v3) / sin_rho)
    if _v2 < 0: beta = pi - beta
    gamma = asin(sin(_v2) / sin_rho)
    if _v3 < 0: gamma = pi - gamma
    A = pi / 2. + _roll - beta
    B = atan2(sin(A) * cos(_dec), (sin(_dec) * sin_rho - cos(_dec) * cos(rho) * cos(A)))

    # compute final value
    troll = np.rad2deg(pi - (gamma + B))

    return troll
