import copy
import datetime
import logging
import time
import numpy as np
from numpy import linalg
from stsci.tools import fileutil

from . import npol
from . import makewcs
from .utils import diff_angles

logger = logging.getLogger('stwcs.updatewcs.corrections')

MakeWCS = makewcs.MakeWCS
NPOLCorr = npol.NPOLCorr


class TDDCorr:
    """
    Apply time dependent distortion correction to distortion coefficients and basic
    WCS keywords. This correction **must** be done before any other WCS correction.

    Parameters
    ----------
    ext_wcs: HSTWCS object
             An HSTWCS object to be modified
    ref_wcs: HSTWCS object
             A reference HSTWCS object

    Notes
    -----
    Compute the ACS/WFC time dependent distortion terms as described
    in [1]_ and apply the correction to the WCS of the observation.

    The model coefficients are stored in the primary header of the IDCTAB.
    :math:`D_{ref}` is the reference date. The computed corrections are saved
    in the science extension header as TDDALPHA and TDDBETA keywords.

    .. math:: TDDALPHA = A_{0} + {A_{1}*(obsdate - D_{ref})}

    .. math:: TDDBETA =  B_{0} + B_{1}*(obsdate - D_{ref})


    The time dependent distortion affects the IDCTAB coefficients, and the
    relative location of the two chips. Because the linear order IDCTAB
    coefficients ar eused in the computatuion of the NPOL extensions,
    the TDD correction affects all components of the distortion model.

    Application of TDD to the IDCTAB polynomial coefficients:
    The TDD model is computed in Jay's frame, while the IDCTAB coefficients
    are in the HST V2/V3 frame. The coefficients are transformed to Jay's frame,
    TDD is applied and they are transformed back to the V2/V3 frame. This
    correction is performed in this class.

    Application of TDD to the relative location of the two chips is
    done in `makewcs`.

    References
    ----------
    .. [1] Jay Anderson, "Variation of the Distortion Solution of the WFC", ACS ISR 2007-08.

    """
    def updateWCS(cls, ext_wcs, ref_wcs):
        """
        - Calculates alpha and beta for ACS/WFC data.
        - Writes 2 new kw to the extension header: TDDALPHA and TDDBETA
        """
        logger.info("Starting TDDCorr: %s", time.asctime())
        ext_wcs.idcmodel.ocx = copy.deepcopy(ext_wcs.idcmodel.cx)
        ext_wcs.idcmodel.ocy = copy.deepcopy(ext_wcs.idcmodel.cy)

        ocxy_comment = "original linear term from IDCTAB"

        newkw = {'TDDALPHA': None,
                 'TDDBETA': None,
                 'TDD_CTA': None,
                 'TDD_CTB': None,
                 'TDD_CYA': None,
                 'TDD_CYB': None,
                 'TDD_CXA': None,
                 'TDD_CXB': None,
                 'OCX10': (ext_wcs.idcmodel.ocx[1, 0], ocxy_comment),
                 'OCX11': (ext_wcs.idcmodel.ocx[1, 1], ocxy_comment),
                 'OCY10': (ext_wcs.idcmodel.ocy[1, 0], ocxy_comment),
                 'OCY11': (ext_wcs.idcmodel.ocy[1, 1], ocxy_comment),
                 }

        if ext_wcs.idcmodel.refpix['skew_coeffs'] is not None and \
                ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CTB'] is not None:
            cls.apply_tdd2idc2015(ref_wcs)
            cls.apply_tdd2idc2015(ext_wcs)

            newkw.update({'TDD_CTA': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CTA'],
                          'TDD_CYA': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CYA'],
                          'TDD_CXA': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CXA'],
                          'TDD_CTB': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CTB'],
                          'TDD_CYB': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CYB'],
                          'TDD_CXB': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CXB']
                          })

        elif ext_wcs.idcmodel.refpix['skew_coeffs'] is not None and \
                ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CY_BETA'] is not None:
            logger.info("Applying 2014-calibrated TDD: {0}".format(time.asctime()))
            # We have 2014-calibrated TDD, not J.A.-style TDD
            cls.apply_tdd2idc2(ref_wcs)
            cls.apply_tdd2idc2(ext_wcs)
            newkw.update({'TDD_CYA': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CY_ALPHA'],
                          'TDD_CYB': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CY_BETA'],
                          'TDD_CXA': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CX_ALPHA'],
                          'TDD_CXB': ext_wcs.idcmodel.refpix['skew_coeffs']['TDD_CX_BETA']})
        else:
            alpha, beta = cls.compute_alpha_beta(ext_wcs)
            cls.apply_tdd2idc(ref_wcs, alpha, beta)
            cls.apply_tdd2idc(ext_wcs, alpha, beta)
            ext_wcs.idcmodel.refpix['TDDALPHA'] = alpha
            ext_wcs.idcmodel.refpix['TDDBETA'] = beta
            ref_wcs.idcmodel.refpix['TDDALPHA'] = alpha
            ref_wcs.idcmodel.refpix['TDDBETA'] = beta
            newkw.update({'TDDALPHA': alpha,
                          'TDDBETA': beta} )

        # add keyword comments
        newkw['TDDALPHA'] = (newkw['TDDALPHA'],
                             "time-dependent y-skew offset (pre-2014 IDCTAB)")
        newkw['TDDBETA'] = (newkw['TDDBETA'],
                            "time-dependent y-skew rate (pre-2014 IDCTAB)")
        newkw['TDD_CXB'] = (newkw['TDD_CXB'],
                            "time-dependent x-scale rate (>2015 IDCTAB)")
        newkw['TDD_CYB'] = (newkw['TDD_CYB'],
                            "time-dependent y-scale rate (>2015 IDCTAB)")
        newkw['TDD_CTA'] = (newkw['TDD_CTA'],
                            "time-dependent x-skew rate (>2015 IDCTAB)")
        newkw['TDD_CTB'] = (newkw['TDD_CTB'],
                            "time-dependent y-skew rate (>2015 IDCTAB)")
        newkw['TDD_CXA'] = (newkw['TDD_CXA'],
                            "time-dependent x-skew offset (2014 IDCTAB)")
        newkw['TDD_CYA'] = (newkw['TDD_CYA'],
                            "time-dependent y-skew offset (2014 IDCTAB)")

        return newkw

    updateWCS = classmethod(updateWCS)

    def apply_tdd2idc2015(cls, hwcs):
        """ Applies 2015-calibrated TDD correction to a couple of IDCTAB
            coefficients for ACS/WFC observations.
        """
        if not isinstance(hwcs.date_obs, float):
            year, month, day = hwcs.date_obs.split('-')
            rdate = datetime.datetime(int(year), int(month), int(day))
            rday = float(rdate.strftime("%j")) / 365.25 + rdate.year
        else:
            rday = hwcs.date_obs

        skew_coeffs = hwcs.idcmodel.refpix['skew_coeffs']
        delta_date = rday - skew_coeffs['TDD_DATE']

        if skew_coeffs['TDD_CXB'] is not None:
            hwcs.idcmodel.cx[1, 1] += skew_coeffs['TDD_CXB'] * delta_date
        if skew_coeffs['TDD_CTB'] is not None:
            hwcs.idcmodel.cy[1, 1] += skew_coeffs['TDD_CTB'] * delta_date
        if skew_coeffs['TDD_CYB'] is not None:
            hwcs.idcmodel.cy[1, 0] += skew_coeffs['TDD_CYB'] * delta_date

    apply_tdd2idc2015 = classmethod(apply_tdd2idc2015)

    def apply_tdd2idc2(cls, hwcs):
        """ Applies 2014-calibrated TDD correction to single IDCTAB coefficient
            of an ACS/WFC observation.
        """
        if not isinstance(hwcs.date_obs, float):
            year, month, day = hwcs.date_obs.split('-')
            rdate = datetime.datetime(int(year), int(month), int(day))
            rday = float(rdate.strftime("%j")) / 365.25 + rdate.year
        else:
            rday = hwcs.date_obs

        skew_coeffs = hwcs.idcmodel.refpix['skew_coeffs']
        cy_beta = skew_coeffs['TDD_CY_BETA']
        cy_alpha = skew_coeffs['TDD_CY_ALPHA']
        delta_date = rday - skew_coeffs['TDD_DATE']
        logger.info("DELTA_DATE: {0} based on rday: {1}, TDD_DATE: {2}".format(delta_date, rday,
                                                                               skew_coeffs['TDD_DATE']))

        if cy_alpha is None:
            hwcs.idcmodel.cy[1, 1] += cy_beta * delta_date
        else:
            new_beta = cy_alpha + cy_beta * delta_date
            hwcs.idcmodel.cy[1, 1] = new_beta
        logger.info("CY11: {0} based on alpha: {1}, beta: {2}".format(hwcs.idcmodel.cy[1, 1],
                                                                      cy_alpha, cy_beta))

        cx_beta = skew_coeffs['TDD_CX_BETA']
        cx_alpha = skew_coeffs['TDD_CX_ALPHA']
        if cx_alpha is not None:
            new_beta = cx_alpha + cx_beta * delta_date
            hwcs.idcmodel.cx[1, 1] = new_beta
            logger.info("CX11: {0} based on alpha: {1}, beta: {2}".format(new_beta,
                                                                          cx_alpha, cx_beta))

    apply_tdd2idc2 = classmethod(apply_tdd2idc2)

    def apply_tdd2idc(cls, hwcs, alpha, beta):
        """
        Applies TDD to the idctab coefficients of a ACS/WFC observation.
        This should be always the first correction.
        """
        theta_v2v3 = 2.234529
        mrotp = fileutil.buildRotMatrix(theta_v2v3)
        mrotn = fileutil.buildRotMatrix(-theta_v2v3)
        tdd_mat = np.array([[1 + (beta / 2048.), alpha / 2048.],
                            [alpha / 2048., 1 - (beta / 2048.)]], np.float64)
        abmat1 = np.dot(tdd_mat, mrotn)
        abmat2 = np.dot(mrotp, abmat1)
        xshape, yshape = hwcs.idcmodel.cx.shape, hwcs.idcmodel.cy.shape
        icxy = np.dot(abmat2, [hwcs.idcmodel.cx.ravel(), hwcs.idcmodel.cy.ravel()])
        hwcs.idcmodel.cx = icxy[0]
        hwcs.idcmodel.cy = icxy[1]
        hwcs.idcmodel.cx = np.reshape(hwcs.idcmodel.cx, xshape)
        hwcs.idcmodel.cy = np.reshape(hwcs.idcmodel.cy, yshape)

    apply_tdd2idc = classmethod(apply_tdd2idc)

    def compute_alpha_beta(cls, ext_wcs):
        """
        Compute the ACS time dependent distortion skew terms
        as described in ACS ISR 07-08 by J. Anderson.

        Jay's code only computes the alpha/beta values based on a decimal year
        with only 3 digits, so this line reproduces that when needed for comparison
        with his results.
        rday = float(('%0.3f')%rday)

        The zero-point terms account for the skew accumulated between
        2002.0 and 2004.5, when the latest IDCTAB was delivered.
        alpha = 0.095 + 0.090*(rday-dday)/2.5
        beta = -0.029 - 0.030*(rday-dday)/2.5
        """
        if not isinstance(ext_wcs.date_obs, float):
            year, month, day = ext_wcs.date_obs.split('-')
            rdate = datetime.datetime(int(year), int(month), int(day))
            rday = float(rdate.strftime("%j")) / 365.25 + rdate.year
        else:
            rday = ext_wcs.date_obs

        skew_coeffs = ext_wcs.idcmodel.refpix['skew_coeffs']
        if skew_coeffs is None:
            # Only print out warning for post-SM4 data where this may matter
            if rday > 2009.0:
                err_str = "------------------------------------------------------------------------  \n"
                err_str += "WARNING: the IDCTAB geometric distortion file specified in the image      \n"
                err_str += "         header did not have the time-dependent distortion coefficients.  \n"
                err_str += "         The pre-SM4 time-dependent skew solution will be used by default.\n"
                err_str += "         Please update IDCTAB with new reference file from HST archive.   \n"
                err_str += "------------------------------------------------------------------------  \n"
                print(err_str)
            # Using default pre-SM4 coefficients
            skew_coeffs = {'TDD_A': [0.095, 0.090 / 2.5],
                           'TDD_B': [-0.029, -0.030 / 2.5],
                           'TDD_DATE': 2004.5,
                           'TDDORDER': 1}

        alpha = 0
        beta = 0
        # Compute skew terms, allowing for non-linear coefficients as well
        for c in range(skew_coeffs['TDDORDER'] + 1):
            alpha += skew_coeffs['TDD_A'][c] * np.power((rday - skew_coeffs['TDD_DATE']), c)
            beta += skew_coeffs['TDD_B'][c] * np.power((rday - skew_coeffs['TDD_DATE']), c)

        return alpha, beta
    compute_alpha_beta = classmethod(compute_alpha_beta)


class VACorr:
    """
    Apply velocity aberation correction to WCS keywords.

    Notes
    -----
    Velocity Aberration is stored in the extension header keyword 'VAFACTOR'.
    The correction is applied to the CD matrix and CRVALs.

    """
    def updateWCS(cls, ext_wcs, ref_wcs):
        logger.info("Starting VACorr: %s" % time.asctime())
        if ext_wcs.vafactor != 1:
            ext_wcs.wcs.cd = ext_wcs.wcs.cd * ext_wcs.vafactor
            crval0 = ref_wcs.wcs.crval[0] + ext_wcs.vafactor * diff_angles(ext_wcs.wcs.crval[0],
                                                                           ref_wcs.wcs.crval[0])
            crval1 = ref_wcs.wcs.crval[1] + ext_wcs.vafactor * diff_angles(ext_wcs.wcs.crval[1],
                                                                           ref_wcs.wcs.crval[1])
            crval = np.array([crval0, crval1])
            ext_wcs.wcs.crval = crval
            ext_wcs.wcs.set()

        kw2update = {
            'CD1_1': ext_wcs.wcs.cd[0, 0],
            'CD1_2': ext_wcs.wcs.cd[0, 1],
            'CD2_1': ext_wcs.wcs.cd[1, 0],
            'CD2_2': ext_wcs.wcs.cd[1, 1],
            'CRVAL1': ext_wcs.wcs.crval[0],
            'CRVAL2': ext_wcs.wcs.crval[1]
        }

        return kw2update

    updateWCS = classmethod(updateWCS)


class CompSIP:
    """
    Compute Simple Imaging Polynomial (SIP) coefficients as defined in [2]_
    from IDC table coefficients.

    This class transforms the TDD corrected IDCTAB coefficients into SIP format.
    It also applies a binning factor to the coefficients if the observation was
    binned.

    References
    ----------
    .. [2] David Shupe, et al, "The SIP Convention of representing Distortion
       in FITS Image headers",  Astronomical Data Analysis Software And Systems, ASP
       Conference Series, Vol. 347, 2005

    """
    def updateWCS(cls, ext_wcs, ref_wcs):
        logger.info("Starting CompSIP: {0}".format(time.asctime()))
        kw2update = {}
        if not ext_wcs.idcmodel:
            logger.info("IDC model not found, SIP coefficient will not be computed.")
            return kw2update
        order = ext_wcs.idcmodel.norder
        kw2update['A_ORDER'] = order, 'SIP polynomial order, axis 1, detector to sky'
        kw2update['B_ORDER'] = order, 'SIP polynomial order, axis 2, detector to sky'
        # pscale = ext_wcs.idcmodel.refpix['PSCALE']

        cx = ext_wcs.idcmodel.cx
        cy = ext_wcs.idcmodel.cy

        matr = np.array([[cx[1, 1], cx[1, 0]], [cy[1, 1], cy[1, 0]]], dtype=np.float64)
        imatr = linalg.inv(matr)
        akeys1 = np.zeros((order + 1, order + 1), dtype=np.float64)
        bkeys1 = np.zeros((order + 1, order + 1), dtype=np.float64)
        sip_comment = 'SIP distortion coefficient'
        for n in range(order + 1):
            for m in range(order + 1):
                if n >= m and n >= 2:
                    idcval = np.array([[cx[n, m]], [cy[n, m]]])
                    sipval = np.dot(imatr, idcval)
                    akeys1[m, n - m] = sipval[0].item()
                    bkeys1[m, n - m] = sipval[1].item()
                    Akey = "A_%d_%d" % (m, n - m)
                    Bkey = "B_%d_%d" % (m, n - m)
                    kw2update[Akey] = sipval[0, 0] * ext_wcs.binned, sip_comment
                    kw2update[Bkey] = sipval[1, 0] * ext_wcs.binned, sip_comment
        kw2update['CTYPE1'] = 'RA---TAN-SIP'
        kw2update['CTYPE2'] = 'DEC--TAN-SIP'
        return kw2update

    updateWCS = classmethod(updateWCS)
