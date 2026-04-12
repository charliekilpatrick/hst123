import logging
import time

import numpy as np
from astropy.io import fits

from stsci.tools import fileutil

logger = logging.getLogger('stwcs.updatewcs.npol')


class NPOLCorr:
    """
    Defines a Lookup table prior distortion correction as per WCS paper IV.
    It uses a reference file defined by the NPOLFILE (suffix 'NPL') keyword
    in the primary header.

    Notes
    -----
    - Using extensions in the reference file create a WCSDVARR extensions
      and add them to the science file.
    - Add record-valued keywords to the science extension header to describe
      the lookup tables.
    - Add a keyword 'NPOLEXT' to the science extension header to store
      the name of the reference file used to create the WCSDVARR extensions.

    If WCSDVARR extensions exist and `NPOLFILE` is different from `NPOLEXT`,
    a subsequent update will overwrite the existing extensions.
    If WCSDVARR extensions were not found in the science file, they will be added.

    It is assumed that the NPL reference files were created to work with IDC tables
    but will be applied with SIP coefficients. A transformation is applied to correct
    for the fact that the lookup tables will be applied before the first order coefficients
    which are in the CD matrix when the SIP convention is used.
    """

    def updateWCS(cls, fobj):
        """
        Parameters
        ----------
        fobj : `astropy.io.fits.HDUList` object
            Science file, for which a distortion correction in a NPOLFILE is available

        """
        logger.info("Starting NPOL: %s", time.asctime())
        try:
            assert isinstance(fobj, fits.HDUList)
        except AssertionError:
            logger.exception("Input must be a fits.HDUList object")
            raise

        cls.applyNPOLCorr(fobj)
        nplfile = fobj[0].header['NPOLFILE']

        new_kw = {'NPOLEXT': nplfile}
        return new_kw

    updateWCS = classmethod(updateWCS)

    def applyNPOLCorr(cls, fobj):
        """
        For each science extension in a fits file object:
            - create a WCSDVARR extension
            - update science header
            - add/update NPOLEXT keyword
        """
        nplfile = fileutil.osfn(fobj[0].header['NPOLFILE'])
        # Map WCSDVARR EXTVER numbers to extension numbers
        wcsdvarr_ind = cls.getWCSIndex(fobj)
        for ext in fobj:
            try:
                extname = ext.header['EXTNAME'].lower()
            except KeyError:
                continue
            if extname == 'sci':
                extversion = ext.header['EXTVER']
                ccdchip = cls.get_ccdchip(fobj, extname='SCI', extver=extversion)
                header = ext.header
                # get the data arrays from the reference file and transform
                # them for use with SIP
                dx, dy = cls.getData(nplfile, ccdchip)
                idccoeffs = cls.getIDCCoeffs(header)

                if idccoeffs is not None:
                    dx, dy = cls.transformData(dx, dy, idccoeffs)

                # Determine EXTVER for the WCSDVARR extension from the
                # NPL file (EXTNAME, EXTVER) kw.
                # This is used to populate DPj.EXTVER kw
                wcsdvarr_x_version = 2 * extversion - 1
                wcsdvarr_y_version = 2 * extversion
                for ename in zip(['DX', 'DY'], [wcsdvarr_x_version, wcsdvarr_y_version], [dx, dy]):
                    error_val = ename[2].max()
                    cls.addSciExtKw(header, wdvarr_ver=ename[1], npol_extname=ename[0], error_val=error_val)
                    hdu = cls.createNpolHDU(header, npolfile=nplfile,
                                            wdvarr_ver=ename[1], npl_extname=ename[0],
                                            data=ename[2], ccdchip=ccdchip)
                    if wcsdvarr_ind:
                        fobj[wcsdvarr_ind[ename[1]]] = hdu
                    else:
                        fobj.append(hdu)

    applyNPOLCorr = classmethod(applyNPOLCorr)

    def getWCSIndex(cls, fobj):

        """
        If fobj has WCSDVARR extensions:
            returns a mapping of their EXTVER kw to file object extension numbers
        if fobj does not have WCSDVARR extensions:
            an empty dictionary is returned
        """
        wcsd = {}
        for e in range(len(fobj)):
            try:
                ename = fobj[e].header['EXTNAME']
            except KeyError:
                continue
            if ename == 'WCSDVARR':
                wcsd[fobj[e].header['EXTVER']] = e
        logger.debug("A map of WSCDVARR externsions %s" % wcsd)
        return wcsd

    getWCSIndex = classmethod(getWCSIndex)

    def addSciExtKw(cls, hdr, wdvarr_ver=None, npol_extname=None, error_val=0.0):
        """
        Adds kw to sci extension to define WCSDVARR lookup table extensions

        """
        j = 1 if npol_extname == 'DX' else 2

        npol = [
            (f'CPERR{j:1d}', error_val, f'Maximum error of NPOL correction for axis {j:d}'),
            (f'CPDIS{j:1d}', 'Lookup', 'Prior distortion function type'),
            (f'DP{j:1d}.EXTVER', wdvarr_ver, 'Version number of WCSDVARR extension'),
            (f'DP{j:1d}.NAXES', 2, 'Number of independent variables in CPDIS function'),
            (f'DP{j:1d}.AXIS.1', 1, 'Axis number of the 1st variable in a CPDIS function'),
            (f'DP{j:1d}.AXIS.2', 2, 'Axis number of the 2nd variable in a CPDIS function'),
        ]

        # Look for HISTORY keywords. If present, insert new keywords before them
        before_key = 'HISTORY' if 'HISTORY' in hdr else None

        for key, value, comment in npol:
            hdr.set(key, value=value, comment=comment, before=before_key)


    addSciExtKw = classmethod(addSciExtKw)

    def getData(cls, nplfile, ccdchip):
        """
        Get the data arrays from the reference NPOL files
        Make sure 'CCDCHIP' in the npolfile matches "CCDCHIP' in the science file.
        """
        npl = fits.open(nplfile)
        for ext in npl:
            nplextname  = ext.header.get('EXTNAME', "")
            nplccdchip  = ext.header.get('CCDCHIP', 1)
            if nplextname == 'DX' and nplccdchip == ccdchip:
                xdata = ext.data.copy()
                continue
            elif nplextname == 'DY' and nplccdchip == ccdchip:
                ydata = ext.data.copy()
                continue
            else:
                continue
        npl.close()
        return xdata, ydata
    getData = classmethod(getData)

    def transformData(cls, dx, dy, coeffs):
        """
        Transform the NPOL data arrays for use with SIP
        """
        ndx, ndy = np.dot(coeffs, [dx.ravel(), dy.ravel()]).astype(np.float32)
        ndx = np.reshape(ndx, dx.shape)
        ndy = np.reshape(ndy, dy.shape)
        return ndx, ndy

    transformData = classmethod(transformData)

    def getIDCCoeffs(cls, header):
        """
        Return a matrix of the scaled first order IDC coefficients.
        """
        try:
            ocx10 = header['OCX10']
            ocx11 = header['OCX11']
            ocy10 = header['OCY10']
            ocy11 = header['OCY11']
            coeffs = np.array([[ocx11, ocx10], [ocy11, ocy10]], dtype=np.float32)
        except KeyError:
            logger.exception(
                "First order IDCTAB coefficients are not available; "
                "cannot convert SIP to IDC coefficients."
            )
            return None
        try:
            idcscale = header['IDCSCALE']
        except KeyError:
            logger.exception("IDCSCALE not found in header - setting it to 1.")
            idcscale = 1

        return np.linalg.inv(coeffs / idcscale)

    getIDCCoeffs = classmethod(getIDCCoeffs)

    def createNpolHDU(cls, sciheader, npolfile=None, wdvarr_ver=1, npl_extname=None,
                      data=None, ccdchip=1):
        """
        Creates an HDU to be added to the file object.
        """
        hdr = cls.createNpolHdr(sciheader, npolfile=npolfile, wdvarr_ver=wdvarr_ver,
                                npl_extname=npl_extname, ccdchip=ccdchip)
        hdu = fits.ImageHDU(header=hdr, data=data)
        return hdu

    createNpolHDU = classmethod(createNpolHDU)

    def createNpolHdr(cls, sciheader, npolfile, wdvarr_ver, npl_extname, ccdchip):
        """
        Creates a header for the WCSDVARR extension based on the NPOL reference file
        and sci extension header. The goal is to always work in image coordinates
        (also for subarrays and binned images. The WCS for the WCSDVARR extension
        i ssuch that a full size npol table is created and then shifted or scaled
        if the science image is a subarray or binned image.
        """
        npl = fits.open(npolfile)
        npol_phdr = npl[0].header
        for ext in npl:
            try:
                nplextname = ext.header['EXTNAME']
                nplextver = ext.header['EXTVER']
            except KeyError:
                continue

            nplccdchip = cls.get_ccdchip(npl, extname=nplextname, extver=nplextver)
            if nplextname == npl_extname and nplccdchip == ccdchip:
                npol_header = ext.header
                break

        npl.close()

        naxis = npl[1].header['NAXIS']
        ccdchip = nplextname  # npol_header['CCDCHIP']

        cdl = [
            ('XTENSION', 'IMAGE', 'Image extension'),
            ('BITPIX', -32, 'number of bits per data pixel'),
            ('NAXIS', naxis, 'Number of data axes'),
            ('EXTNAME', 'WCSDVARR', 'WCS distortion array'),
            ('EXTVER', wdvarr_ver, 'Distortion array version number'),
            ('PCOUNT', 0, 'number of parameters'),
            ('GCOUNT', 1, 'number of groups'),
            ('CCDCHIP', ccdchip),
        ]

        for i in range(1, naxis + 1):
            cdl.append((f'NAXIS{i:d}', npol_header.get(f'NAXIS{i:d}'),
                        f"length of data axis {i:d}"))
            cdl.append((f'CDELT{i:d}', npol_header.get(f'CDELT{i:d}', 1.0) *
                        sciheader.get(f'LTM{i:d}_{i:d}', 1),
                        "Coordinate increment at reference point"))
            cdl.append((f'CRPIX{i:d}', npol_header.get(f'CRPIX{i:d}', 0.0),
                        "Pixel coordinate of reference point"))
            cdl.append((f'CRVAL{i:d}', npol_header.get(f'CRVAL{i:d}', 0.0) -
                        sciheader.get(f'LTV{i:d}', 0),
                        "Coordinate value at reference point"))

        # Now add keywords from NPOLFILE header to document source of calibration
        # include all keywords after and including 'FILENAME' from header
        start_indx = -1
        end_indx = 0
        for i, c in enumerate(npol_phdr):
            if c == 'FILENAME':
                start_indx = i
            if c == '':  # remove blanks from end of header
                end_indx = i + 1
                break
        if start_indx >= 0:
            for card in npol_phdr.cards[start_indx: end_indx]:
                cdl.append(card)

        hdr = fits.Header(cards=cdl)

        return hdr

    createNpolHdr = classmethod(createNpolHdr)

    def get_ccdchip(cls, fobj, extname, extver):
        """
        Given a science file or npol file determine CCDCHIP
        """
        ccdchip = 1
        if fobj[0].header['INSTRUME'] == 'ACS' and fobj[0].header['DETECTOR'] == 'WFC':
            ccdchip = fobj[extname, extver].header['CCDCHIP']
        elif fobj[0].header['INSTRUME'] == 'WFC3' and fobj[0].header['DETECTOR'] == 'UVIS':
            ccdchip = fobj[extname, extver].header['CCDCHIP']
        elif fobj[0].header['INSTRUME'] == 'WFPC2':
            ccdchip = fobj[extname, extver].header['DETECTOR']
        elif fobj[0].header['INSTRUME'] == 'STIS':
            ccdchip = fobj[extname, extver].header['DETECTOR']
        elif fobj[0].header['INSTRUME'] == 'NICMOS':
            ccdchip = fobj[extname, extver].header['CAMERA']
        return ccdchip

    get_ccdchip = classmethod(get_ccdchip)
