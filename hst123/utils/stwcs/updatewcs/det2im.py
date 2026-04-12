from astropy.io import fits
from stsci.tools import fileutil

import logging
import time
logger = logging.getLogger('stwcs.updatewcs.d2im')


class DET2IMCorr:
    """
    Defines a Lookup table prior distortion correction as per WCS paper IV.
    It uses a reference file defined by the D2IMFILE (suffix 'd2im') keyword
    in the primary header.

    Notes
    -----
    - Using extensions in the reference file create a WCSDVARR extensions
      and add them to the science file.
    - Add record-valued keywords to the science extension header to describe
      the lookup tables.
    - Add a keyword 'D2IMEXT' to the science extension header to store
      the name of the reference file used to create the WCSDVARR extensions.

    If WCSDVARR extensions exist and `D2IMFILE` is different from `D2IMEXT`,
    a subsequent update will overwrite the existing extensions.
    If WCSDVARR extensions were not found in the science file, they will be added.

    """

    def updateWCS(cls, fobj):
        """
        Parameters
        ----------
        fobj: `astropy.io.fits.HDUList` object
                Science file, for which a distortion correction in a NPOLFILE is available

        """
        logger.info("Starting DET2IM: {0}".format(time.asctime()))
        try:
            assert isinstance(fobj, fits.HDUList)
        except AssertionError:
            logger.exception('Input must be a fits.HDUList object')
            raise

        cls.applyDet2ImCorr(fobj)
        d2imfile = fobj[0].header['D2IMFILE']

        new_kw = {'D2IMEXT': d2imfile}
        return new_kw

    updateWCS = classmethod(updateWCS)

    def applyDet2ImCorr(cls, fobj):
        """
        For each science extension in a fits file object:
            - create a WCSDVARR extension
            - update science header
            - add/update D2IMEXT keyword
        """
        d2imfile = fileutil.osfn(fobj[0].header['D2IMFILE'])
        # Map D2IMARR EXTVER numbers to FITS extension numbers
        wcsdvarr_ind = cls.getWCSIndex(fobj)
        d2im_num_ext = 1
        for ext in fobj:
            try:
                extname = ext.header['EXTNAME'].lower()
            except KeyError:
                continue
            if extname == 'sci':
                extversion = ext.header['EXTVER']
                ccdchip = cls.get_ccdchip(fobj, extname='SCI', extver=extversion)
                header = ext.header
                # get the data arrays from the reference file
                dx, dy = cls.getData(d2imfile, ccdchip)
                # Determine EXTVER for the D2IMARR extension from the D2I file (EXTNAME, EXTVER) kw.
                # This is used to populate DPj.EXTVER kw
                for ename in zip(['DX', 'DY'], [dx, dy]):
                    if ename[1] is not None:
                        error_val = ename[1].max()
                        cls.addSciExtKw(header, wdvarr_ver=d2im_num_ext, d2im_extname=ename[0], error_val=error_val)
                        hdu = cls.createD2ImHDU(header, d2imfile=d2imfile,
                                                wdvarr_ver=d2im_num_ext,
                                                d2im_extname=ename[0],
                                                data=ename[1], ccdchip=ccdchip)
                        if wcsdvarr_ind and d2im_num_ext in wcsdvarr_ind:
                            fobj[wcsdvarr_ind[d2im_num_ext]] = hdu
                        else:
                            fobj.append(hdu)
                        d2im_num_ext = d2im_num_ext + 1
    applyDet2ImCorr = classmethod(applyDet2ImCorr)

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
            if ename == 'D2IMARR':
                wcsd[fobj[e].header['EXTVER']] = e
        logger.debug("A map of D2IMARR extensions {0}".format(wcsd))
        return wcsd

    getWCSIndex = classmethod(getWCSIndex)

    def addSciExtKw(cls, hdr, wdvarr_ver=None, d2im_extname=None, error_val=0.0):
        """
        Adds kw to sci extension to define WCSDVARR lookup table extensions

        """
        j = 1 if d2im_extname == 'DX' else 2

        d2im = [
            (f'D2IMERR{j:1d}', error_val, f'Maximum error of D2IM correction for axis {j:d}'),
            (f'D2IMDIS{j:1d}', 'Lookup', 'Detector to image correction type'),
            (f'D2IM{j:1d}.EXTVER', wdvarr_ver, 'Version number of WCSDVARR extension'),
            (f'D2IM{j:1d}.NAXES', 2, 'Number of independent variables in D2IM function'),
            (f'D2IM{j:1d}.AXIS.1', 1, 'Axis number of the 1st variable in a D2IM function'),
            (f'D2IM{j:1d}.AXIS.2', 2, 'Axis number of the 2nd variable in a D2IM function'),
        ]

        # Look for HISTORY keywords. If present, insert new keywords before them
        before_key = 'HISTORY' if 'HISTORY' in hdr else None

        for key, value, comment in d2im:
            hdr.set(key, value=value, comment=comment, before=before_key)


    addSciExtKw = classmethod(addSciExtKw)

    def getData(cls, d2imfile, ccdchip):
        """
        Get the data arrays from the reference D2I files
        Make sure 'CCDCHIP' in the npolfile matches "CCDCHIP' in the science file.
        """
        xdata, ydata = (None, None)
        d2im = fits.open(d2imfile)
        for ext in d2im:
            d2imextname  = ext.header.get('EXTNAME', "")
            d2imccdchip  = ext.header.get('CCDCHIP', 1)
            if d2imextname == 'DX' and d2imccdchip == ccdchip:
                xdata = ext.data.copy()
                continue
            elif d2imextname == 'DY' and d2imccdchip == ccdchip:
                ydata = ext.data.copy()
                continue
            else:
                continue
        d2im.close()
        return xdata, ydata
    getData = classmethod(getData)

    def createD2ImHDU(cls, sciheader, d2imfile=None, wdvarr_ver=1,
                      d2im_extname=None, data=None, ccdchip=1):
        """
        Creates an HDU to be added to the file object.
        """
        hdr = cls.createD2ImHdr(sciheader, d2imfile=d2imfile,
                                wdvarr_ver=wdvarr_ver, d2im_extname=d2im_extname,
                                ccdchip=ccdchip)
        hdu = fits.ImageHDU(header=hdr, data=data)
        return hdu

    createD2ImHDU = classmethod(createD2ImHDU)

    def createD2ImHdr(cls, sciheader, d2imfile, wdvarr_ver, d2im_extname, ccdchip):
        """
        Creates a header for the D2IMARR extension based on the D2I reference file
        and sci extension header. The goal is to always work in image coordinates
        (also for subarrays and binned images). The WCS for the D2IMARR extension
        is such that a full size d2im table is created and then shifted or scaled
        if the science image is a subarray or binned image.
        """
        d2im = fits.open(d2imfile)
        d2im_phdr = d2im[0].header
        for ext in d2im:
            try:
                d2imextname = ext.header['EXTNAME']
                d2imextver = ext.header['EXTVER']
            except KeyError:
                continue

            d2imccdchip = cls.get_ccdchip(d2im, extname=d2imextname, extver=d2imextver)
            if d2imextname == d2im_extname and d2imccdchip == ccdchip:
                d2im_header = ext.header
                break

        d2im.close()

        naxis = d2im[1].header['NAXIS']
        ccdchip = d2imextname

        cdl = [
            ('XTENSION', 'IMAGE', 'Image extension'),
            ('BITPIX', -32, 'number of bits per data pixel'),
            ('NAXIS', naxis, 'Number of data axes'),
            ('EXTNAME', 'D2IMARR', 'WCS distortion array'),
            ('EXTVER', wdvarr_ver, 'Distortion array version number'),
            ('PCOUNT', 0, 'number of parameters'),
            ('GCOUNT', 1, 'number of groups'),
            ('CCDCHIP', ccdchip),
        ]

        for i in range(1, naxis + 1):
            cdl.append((f'NAXIS{i:d}', d2im_header.get(f'NAXIS{i:d}'),
                        f"length of data axis {i:d}"))
            cdl.append((f'CDELT{i:d}', d2im_header.get(f'CDELT{i:d}', 1.0) *
                        sciheader.get(f'LTM{i:d}_{i:d}', 1),
                        "Coordinate increment at reference point"))
            cdl.append((f'CRPIX{i:d}', d2im_header.get(f'CRPIX{i:d}', 0.0),
                        "Pixel coordinate of reference point"))
            cdl.append((f'CRVAL{i:d}', d2im_header.get(f'CRVAL{i:d}', 0.0) -
                        sciheader.get(f'LTV{i:d}', 0),
                        "Coordinate value at reference point"))

        # Now add keywords from NPOLFILE header to document source of calibration
        # include all keywords after and including 'FILENAME' from header
        start_indx = -1
        end_indx = 0
        for i, c in enumerate(d2im_phdr):
            if c == 'FILENAME':
                start_indx = i
            if c == '':  # remove blanks from end of header
                end_indx = i + 1
                break

        if start_indx >= 0:
            for card in d2im_phdr.cards[start_indx:end_indx]:
                cdl.append(card)

        hdr = fits.Header(cards=cdl)

        return hdr

    createD2ImHdr = classmethod(createD2ImHdr)

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
