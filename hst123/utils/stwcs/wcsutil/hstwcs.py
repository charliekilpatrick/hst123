import os
import warnings
from astropy.wcs import WCS
from astropy.io import fits
from astropy import log

from ..distortion import models, coeff_converter
import numpy as np
from stsci.tools import fileutil

from . import pc2cd
from . import getinput
from . import instruments
from .mappings import inst_mappings, ins_spec_kw
from ..wcsutil.altwcs import exclude_hst_specific

default_log_level = log.getEffectiveLevel()

__all__ = ['HSTWCS']

warnings.filterwarnings("ignore", message="^Some non-standard WCS keywords were excluded:", module="astropy.wcs.wcs")

def extract_rootname(kwvalue, suffix=""):
    """ Returns the rootname from a full reference filename

        If a non-valid value (any of ['','N/A','NONE','INDEF',None]) is input,
            simply return a string value of 'NONE'

        This function will also replace any 'suffix' specified with a blank.
    """
    # check to see whether a valid kwvalue has been provided as input
    if kwvalue.strip() in ['', 'N/A', 'NONE', 'INDEF', None]:
        return 'NONE'  # no valid value, so return 'NONE'

    # for a valid kwvalue, parse out the rootname
    # strip off any environment variable from input filename, if any are given
    if '$' in kwvalue:
        fullval = kwvalue[kwvalue.find('$') + 1:]
    else:
        fullval = kwvalue
    # Extract filename without path from kwvalue
    fname = os.path.basename(fullval).strip()

    # Now, rip out just the rootname from the full filename
    rootname = fileutil.buildNewRootname(fname)

    # Now, remove any known suffix from rootname
    rootname = rootname.replace(suffix, '')
    return rootname.strip()


def build_default_wcsname(idctab):

    idcname = extract_rootname(idctab, suffix='_idc')
    wcsname = 'IDC_' + idcname
    return wcsname


class NoConvergence(Exception):
    """
    An error class used to report non-convergence and/or divergence of
    numerical methods. It is used to report errors in the iterative solution
    used by the :py:meth:`~stwcs.hstwcs.HSTWCS.all_world2pix`.

    Attributes
    ----------

    best_solution : numpy.array
        Best solution achieved by the method.

    accuracy : float
        Accuracy of the :py:attr:`best_solution`.

    niter : int
        Number of iterations performed by the numerical method to compute
        :py:attr:`best_solution`.

    divergent : None, numpy.array
        Indices of the points in :py:attr:`best_solution` array for which the
        solution appears to be divergent. If the solution does not diverge,
        `divergent` will be set to `None`.

    failed2converge : None, numpy.array
        Indices of the points in :py:attr:`best_solution` array for which the
        solution failed to converge within the specified maximum number
        of iterations. If there are no non-converging poits (i.e., if
        the required accuracy has been achieved for all points) then
        `failed2converge` will be set to `None`.

    """
    def __init__(self, *args, **kwargs):
        super(NoConvergence, self).__init__(*args)

        self.best_solution  = kwargs.pop('best_solution', None)
        self.accuracy       = kwargs.pop('accuracy', None)
        self.niter          = kwargs.pop('niter', None)
        self.divergent      = kwargs.pop('divergent', None)
        self.failed2converge = kwargs.pop('failed2converge', None)


class HSTWCS(WCS):

    def __init__(self, fobj=None, ext=None, minerr=0.0, wcskey=" "):
        """
        Create a WCS object based on the instrument.

        In addition to basic WCS keywords this class provides
        instrument specific information needed in distortion computation.

        Parameters
        ----------
        fobj : str or `astropy.io.fits.HDUList` object or None
            file name, e.g j9irw4b1q_flt.fits
            fully qualified filename[EXTNAME,EXTNUM], e.g. j9irw4b1q_flt.fits[sci,1]
            `astropy.io.fits` file object, e.g fits.open('j9irw4b1q_flt.fits'), in which case the
            user is responsible for closing the file object.
        ext : int, tuple or None
            extension number
            if ext is tuple, it must be ("EXTNAME", EXTNUM), e.g. ("SCI", 2)
            if ext is None, it is assumed the data is in the primary hdu
        minerr : float
            minimum value a distortion correction must have in order to be applied.
            If CPERRja, CQERRja are smaller than minerr, the corersponding
            distortion is not applied.
        wcskey : str
            A one character A-Z or " " used to retrieve and define an
            alternate WCS description.
        """

        self.inst_kw = ins_spec_kw
        self.minerr = minerr
        self.wcskey = wcskey

        if fobj is not None:
            filename, hdr0, ehdr, phdu = getinput.parseSingleInput(f=fobj,
                                                                   ext=ext)
            self.filename = filename
            instrument_name = hdr0.get('INSTRUME', 'DEFAULT')
            if instrument_name == 'DEFAULT' or instrument_name not in list(inst_mappings.keys()):
                self.instrument = 'DEFAULT'
            else:
                self.instrument = instrument_name
            # Set the correct reference frame
            refframe = determine_refframe(hdr0)
            if refframe is not None:
                ehdr['RADESYS'] = refframe

            WCS.__init__(self, ehdr, fobj=phdu, minerr=self.minerr,
                         key=self.wcskey)
            if self.instrument == 'DEFAULT':
                self.pc2cd()
            # If input was a `astropy.io.fits.HDUList` object, it's the user's
            # responsibility to close it, otherwise, it's closed here.
            if not isinstance(fobj, fits.HDUList):
                phdu.close()
            self.setInstrSpecKw(hdr0, ehdr)
            self.readIDCCoeffs(ehdr)
            extname = ehdr.get('EXTNAME', 'PRIMARY' if ehdr is hdr0 else '')
            extnum = ehdr.get('EXTVER', 1)
            self.extname = (extname, extnum)
        else:
            # create a default HSTWCS object
            self.instrument = 'DEFAULT'
            WCS.__init__(self, minerr=self.minerr, key=self.wcskey)
            self.pc2cd()
            self.setInstrSpecKw()
        self.setPscale()
        self.setOrient()

    @property
    def naxis1(self):
        return self.pixel_shape[0]

    @naxis1.setter
    def naxis1(self, value):
        val1 = self.pixel_shape[1] if self.pixel_shape is not None else 0
        self.pixel_shape = (value, val1)

    @property
    def naxis2(self):
        return self.pixel_shape[1]

    @naxis2.setter
    def naxis2(self, value):
        val0 = self.pixel_shape[0] if self.pixel_shape is not None else 0
        self.pixel_shape = (val0, value)

    def readIDCCoeffs(self, header):
        """
        Reads in first order IDCTAB coefficients if present in the header
        """
        coeffs = ['ocx10', 'ocx11', 'ocy10', 'ocy11', 'idcscale',
                  'idcv2ref', 'idcv3ref', 'idctheta']
        for c in coeffs:
            self.__setattr__(c, header.get(c, None))

    def setInstrSpecKw(self, prim_hdr=None, ext_hdr=None):
        """
        Populate the instrument specific attributes:

        These can be in different headers but each instrument class has knowledge
        of where to look for them.

        Parameters
        ----------
        prim_hdr : `astropy.io.fits.Header`
            primary header
        ext_hdr : `astropy.io.fits.Header`
            extension header

        """
        if self.instrument in list(inst_mappings.keys()):
            inst_kl = inst_mappings[self.instrument]
            inst_kl = instruments.__dict__[inst_kl]
            insobj = inst_kl(prim_hdr, ext_hdr)

            for key in self.inst_kw:
                try:
                    self.__setattr__(key, insobj.__getattribute__(key))
                except AttributeError:
                    # Some of the instrument's attributes are recorded in the primary header and
                    # were already set, (e.g. 'DETECTOR'), the code below is a check for that case.
                    if not self.__getattribute__(key):
                        raise
                    else:
                        pass

        else:
            raise KeyError("Unsupported instrument - %s" % self.instrument)

    def setPscale(self):
        """
        Calculates the plate scale from the CD matrix
        """
        try:
            cd11 = self.wcs.cd[0][0]
            cd21 = self.wcs.cd[1][0]
            self.pscale = np.sqrt(np.power(cd11, 2) + np.power(cd21, 2)) * 3600.
        except AttributeError:
            if self.wcs.has_cd():
                print("This file has a PC matrix. You may want to convert it \n \
                to a CD matrix, if reasonable, by running pc2.cd() method.\n \
                The plate scale can be set then by calling setPscale() method.\n")
            self.pscale = None

    def setOrient(self):
        """
        Computes ORIENTAT from the CD matrix
        """
        try:
            cd12 = self.wcs.cd[0][1]
            cd22 = self.wcs.cd[1][1]
            self.orientat = np.rad2deg(np.arctan2(cd12, cd22))
        except AttributeError:
            if self.wcs.has_cd():
                print("This file has a PC matrix. You may want to convert it \n \
                to a CD matrix, if reasonable, by running pc2.cd() method.\n \
                The orientation can be set then by calling setOrient() method.\n")
            self.pscale = None

    def updatePscale(self, scale):
        """
        Updates the CD matrix with a new plate scale
        """
        self.wcs.cd = self.wcs.cd / self.pscale * scale
        self.setPscale()

    def readModel(self, update=False, header=None):
        """
        Reads distortion model from IDCTAB.

        If IDCTAB is not found ('N/A', "", or not found on disk), then
        if SIP coefficients and first order IDCTAB coefficients are present
        in the header, restore the idcmodel from the header.
        If not - assign None to self.idcmodel.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            fits extension header
        update : bool (False)
            if True - record the following IDCTAB quantities as header keywords:
            CX10, CX11, CY10, CY11, IDCSCALE, IDCTHETA, IDCXREF, IDCYREF,
            IDCV2REF, IDCV3REF
        """
        if self.idctab in [None, '', ' ', 'N/A']:
            # Keyword idctab is not present in header - check for sip coefficients
            if header is not None and 'IDCSCALE' in header:
                self._readModelFromHeader(header)
            else:
                print("Distortion model is not available: IDCTAB=None\n")
                self.idcmodel = None
        elif not os.path.exists(fileutil.osfn(self.idctab)):
            if header is not None and 'IDCSCALE' in header:
                self._readModelFromHeader(header)
            else:
                print('Distortion model is not available: IDCTAB file %s not found\n' % self.idctab)
                self.idcmodel = None
        else:
            self.readModelFromIDCTAB(header=header, update=update)

    def _readModelFromHeader(self, header):
        # Recreate idc model from SIP coefficients and header kw
        print('Restoring IDC model from SIP coefficients\n')
        model = models.GeometryModel()
        cx, cy = coeff_converter.sip2idc(self)
        model.cx = cx
        model.cy = cy
        model.name = "sip"
        model.norder = header['A_ORDER']

        refpix = {}
        refpix['XREF'] = header['IDCXREF']
        refpix['YREF'] = header['IDCYREF']
        refpix['PSCALE'] = header['IDCSCALE']
        refpix['V2REF'] = header['IDCV2REF']
        refpix['V3REF'] = header['IDCV3REF']
        refpix['THETA'] = header['IDCTHETA']
        model.refpix = refpix

        self.idcmodel = model

    def readModelFromIDCTAB(self, header=None, update=False):
        """
        Read distortion model from idc table.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            fits extension header
        update : booln (False)
            if True - save teh following as header keywords:
            CX10, CX11, CY10, CY11, IDCSCALE, IDCTHETA, IDCXREF, IDCYREF,
            IDCV2REF, IDCV3REF

        """
        if self.date_obs is None:
            print('date_obs not available\n')
            self.idcmodel = None
            return
        if self.filter1 is None and self.filter2 is None:
            'No filter information available\n'
            self.idcmodel = None
            return

        self.idcmodel = models.IDCModel(self.idctab,
                                        chip=self.chip, direction='forward',
                                        date=self.date_obs,
                                        filter1=self.filter1, filter2=self.filter2,
                                        offtab=self.offtab, binned=self.binned)

        if self.ltv1 != 0. or self.ltv2 != 0.:
            self.resetLTV()

        if update:
            if header is None:
                print('Update header with IDC model kw requested but header was not provided\n.')
            else:
                self._updatehdr(header)

    def resetLTV(self):
        """
        Reset LTV values for polarizer data

        The polarizer field is smaller than the detector field.
        The distortion coefficients are defined for the entire
        polarizer field and the LTV values are set as with subarray
        data. This may also be true for other special filters.
        This is a special case when the observation is considered
        a subarray in terms of detector field but a full frame in
        terms of distortion model.
        To avoid shifting the distortion coefficients the LTV values
        are reset to 0.
        """
        if self.naxis1 == self.idcmodel.refpix['XSIZE'] and  \
           self.naxis2 == self.idcmodel.refpix['YSIZE']:
            self.ltv1 = 0.
            self.ltv2 = 0.


    def wcs2header(self, sip2hdr=False, idc2hdr=True, wcskey=None, relax=False):
        """
        Create a `astropy.io.fits.Header` object from WCS keywords.

        If the original header had a CD matrix, return a CD matrix,
        otherwise return a PC matrix.

        Parameters
        ----------
        sip2hdr : bool
            If True - include SIP coefficients
        """
        warnings.filterwarnings("ignore", message="^Some non-standard WCS keywords were excluded:", module="astropy.wcs")
        h = self.to_header(key=wcskey, relax=relax)
        exclude_hst_specific(h, wcskey=wcskey)

        if not wcskey:
            wcskey = self.wcs.alt
        if self.wcs.has_cd():
            h = pc2cd(h, key=wcskey)

        if 'wcsname' not in h:
            if self.idctab is not None:
                wname = build_default_wcsname(self.idctab)
            else:
                wname = 'DEFAULT'
            h['wcsname{0}'.format(wcskey)] = wname, 'Coordinate system title'

        if idc2hdr:
            for card in self._idc2hdr():
                h[card.keyword + wcskey] = (card.value, card.comment)

        if sip2hdr and self.sip:
            for card in self._sip2hdr('a'):
                h[card.keyword] = (card.value, card.comment)
            for card in self._sip2hdr('b'):
                h[card.keyword] = (card.value, card.comment)

            try:
                ap = self.sip.ap
            except AssertionError:
                ap = None
            try:
                bp = self.sip.bp
            except AssertionError:
                bp = None

            if ap:
                for card in self._sip2hdr('ap'):
                    h[card.keyword] = (card.value, card.comment)
            if bp:
                for card in self._sip2hdr('bp'):
                    h[card.keyword] = (card.value, card.comment)
        return h

    def _sip2hdr(self, k):
        """
        Get a set of SIP coefficients in the form of an array
        and turn them into a `astropy.io.fits.Cardlist`.
        k - one of 'a', 'b', 'ap', 'bp'
        """

        cards = []
        korder = self.sip.__getattribute__(k + '_order')
        cards.append(fits.Card(keyword=k.upper() + '_ORDER', value=korder))
        coeffs = self.sip.__getattribute__(k)
        ind = coeffs.nonzero()
        for i in range(len(ind[0])):
            card = fits.Card(keyword=k.upper() + '_' + str(ind[0][i]) + '_' + str(ind[1][i]),
                             value=coeffs[ind[0][i], ind[1][i]])
            cards.append(card)
        return cards

    def _idc2hdr(self):
        # save some of the idc coefficients
        coeffs = [
            ('ocx10', 'original linear term from IDCTAB'),
            ('ocx11', 'original linear term from IDCTAB'),
            ('ocy10', 'original linear term from IDCTAB'),
            ('ocy11', 'original linear term from IDCTAB'),
            ('idcscale', 'pixel scale from the IDCTAB reference file'),
        ]
        cards = []
        for k, c in coeffs:
            try:
                val = self.__getattribute__(k)
            except AttributeError:
                continue
            if val:
                cards.append(fits.Card(keyword=k, value=val, comment=c))
        return cards

    def pc2cd(self):
        if self.wcs.has_pc():
            self.wcs.cd = self.wcs.pc * self.wcs.cdelt[1]

    def all_world2pix(self, *args, **kwargs):
        """
        all_world2pix(\*arg, tolerance=1.0e-4, maxiter=20, adaptive=False,
        detect_divergence=True, quiet=False)

        Performs full inverse transformation using iterative solution
        on full forward transformation with complete distortion model.

        Parameters
        ----------
        tolerance : float, optional (Default = 1.0e-4)
            Absolute tolerance required the solution. Iteration terminates when
            the correction to the solution found during the previous iteration
            is smaller (in the sence of the L2 norm) than the specified
            ``tolerance``.

        maxiter : int, optional (Default = 20)
            Maximum number of iterations allowed to reach the solution.

        adaptive : bool, optional (Default = False)
            Specifies whether to adaptively select only points that did not
            converge to a solution whithin the required accuracy for the
            next iteration. Default is recommended for HST as well as most
            other instruments.

            .. note::
               The :py:meth:`all_world2pix` uses a vectorized implementation
               of the method of consecutive approximations (see `Notes`
               section below) in which it iterates over *all* input poits
               *regardless* until the required accuracy has been reached for
               *all* input points. In some cases it may be possible that
               *almost all* points have reached the required accuracy but
               there are only a few of input data points left for which
               additional iterations may be needed (this depends mostly on the
               characteristics of the geometric distortions for a given
               instrument). In this situation it may be
               advantageous to set `adaptive` = `True` in which
               case :py:meth:`all_world2pix` will continue iterating *only* over
               the points that have not yet converged to the required
               accuracy. However, for the HST's ACS/WFC detector, which has
               the strongest distortions of all HST instruments, testing has
               shown that enabling this option would lead to a about 10-30 %
               penalty in computational time (depending on specifics of the
               image, geometric distortions, and number of input points to be
               converted). Therefore, for HST instruments,
               it is recommended to set `adaptive` = `False` . The only
               danger in getting this setting wrong will be a performance
               penalty.

            .. note::
               When `detect_divergence` is `True` , :py:meth:`all_world2pix`
               will automatically switch to the adaptive algorithm once
               divergence has been detected.

        detect_divergence : bool, optional (Default = True)
            Specifies whether to perform a more detailed analysis of the
            convergence to a solution. Normally :py:meth:`all_world2pix`
            may not achieve the required accuracy
            if either the `tolerance` or `maxiter` arguments are too low.
            However, it may happen that for some geometric distortions
            the conditions of convergence for the the method of consecutive
            approximations used by :py:meth:`all_world2pix` may not be
            satisfied, in which case consecutive approximations to the
            solution will diverge regardless of the `tolerance` or `maxiter`
            settings.

            When `detect_divergence` is `False` , these divergent points
            will be detected as not having achieved the required accuracy
            (without further details). In addition, if `adaptive` is `False`
            then the algorithm will not know that the solution (for specific
            points) is diverging and will continue iterating and trying to
            "improve" diverging solutions. This may result in NaN or Inf
            values in the return results (in addition to a performance
            penalties). Even when `detect_divergence` is
            `False` , :py:meth:`all_world2pix` , at the end of the iterative
            process, will identify invalid results (NaN or Inf) as "diverging"
            solutions and will raise :py:class:`NoConvergence` unless
            the `quiet` parameter is set to `True` .

            When `detect_divergence` is `True`, :py:meth:`all_world2pix` will
            detect points for
            which current correction to the coordinates is larger than
            the correction applied during the previous iteration **if** the
            requested accuracy **has not yet been achieved**. In this case,
            if `adaptive` is `True`, these points will be excluded from
            further iterations and if `adaptive`
            is `False`, :py:meth:`all_world2pix` will automatically
            switch to the adaptive algorithm.

            .. note::
               When absolute tolerance has been achieved, small increases in
               current corrections may be possible due to rounding errors
               (when `adaptive` is `False` ) and such increases
               will be ignored.

            .. note::
               Setting `detect_divergence` to `True` will incurr about 5-10%
               performance penalty (in our testing on ACS/WFC images).
               Because the benefits of enabling this feature outweigh
               the small performance penalty, it is recommended to set
               `detect_divergence` to `True`, unless extensive testing
               of the distortion models for images from specific
               instruments show a good stability of the numerical method
               for a wide range of coordinates (even outside the image
               itself).

            .. note::
               Indices of the diverging inverse solutions will be reported
               in the `divergent` attribute of the
               raised :py:class:`NoConvergence` object.

        quiet : bool, optional (Default = False)
            Do not throw :py:class:`NoConvergence` exceptions when the method
            does not converge to a solution with the required accuracy
            within a specified number of maximum iterations set by `maxiter`
            parameter. Instead, simply return the found solution.

        Raises
        ------
        NoConvergence
            The method does not converge to a
            solution with the required accuracy within a specified number
            of maximum iterations set by the `maxiter` parameter.

        Notes
        -----
        Inputs can either be (RA, Dec, origin) or (RADec, origin) where RA
        and Dec are 1-D arrays/lists of coordinates and RADec is an
        array/list of pairs of coordinates.

        Using the method of consecutive approximations we iterate starting
        with the initial approximation, which is computed using the
        non-distorion-aware :py:meth:`wcs_world2pix` (or equivalent).

        The :py:meth:`all_world2pix` function uses a vectorized implementation
        of the method of consecutive approximations and therefore it is
        highly efficient (>30x) when *all* data points that need to be
        converted from sky coordinates to image coordinates are passed at
        *once*. Therefore, it is advisable, whenever possible, to pass
        as input a long array of all points that need to be converted
        to :py:meth:`all_world2pix` instead of calling :py:meth:`all_world2pix`
        for each data point. Also see the note to the `adaptive` parameter.

        Examples
        --------
        >>> import stwcs
        >>> from astropy.io import fits
        >>> hdulist = fits.open('j94f05bgq_flt.fits')
        >>> w = stwcs.wcsutil.HSTWCS(hdulist, ext=('sci',1))
        >>> hdulist.close()

        >>> ra, dec = w.all_pix2world([1,2,3],[1,1,1],1); print(ra); print(dec)
        [ 5.52645241  5.52649277  5.52653313]
        [-72.05171776 -72.05171295 -72.05170814]
        >>> radec = w.all_pix2world([[1,1],[2,1],[3,1]],1); print(radec)
        [[  5.52645241 -72.05171776]
         [  5.52649277 -72.05171295]
         [  5.52653313 -72.05170814]]
        >>> x, y = w.all_world2pix(ra,dec,1)
        >>> print(x)
        [ 1.00000233  2.00000232  3.00000233]
        >>> print(y)
        [ 0.99999997  0.99999997  0.99999998]
        >>> xy = w.all_world2pix(radec,1)
        >>> print(xy)
        [[ 1.00000233  0.99999997]
         [ 2.00000232  0.99999997]
         [ 3.00000233  0.99999998]]
        >>> xy = w.all_world2pix(radec,1, maxiter=3, tolerance=1.0e-10, quiet=False)
        NoConvergence: 'HSTWCS.all_world2pix' failed to converge to requested accuracy after 3 iterations.

        >>>
        Now try to use some diverging data:
        >>> divradec = w.all_pix2world([[1.0,1.0],[10000.0,50000.0], [3.0,1.0]],1); print(divradec)
        [[  5.52645241 -72.05171776]
         [  7.15979392 -70.81405561]
         [  5.52653313 -72.05170814]]

        >>> try:
        >>>   xy = w.all_world2pix(divradec,1, maxiter=20, tolerance=1.0e-4, adaptive=False, detect_divergence=True, quiet=False)
        >>> except stwcs.wcsutil.hstwcs.NoConvergence as e:
        >>>   print("Indices of diverging points: {}".format(e.divergent))
        >>>   print("Indices of poorly converging points: {}".format(e.failed2converge))
        >>>   print("Best solution: {}".format(e.best_solution))
        >>>   print("Achieved accuracy: {}".format(e.accuracy))
        >>>   raise e
        Indices of diverging points:
        [1]
        Indices of poorly converging points:
        None
        Best solution:
        [[  1.00006219e+00   9.99999288e-01]
         [ -1.99440907e+06   1.44308548e+06]
         [  3.00006257e+00   9.99999316e-01]]
        Achieved accuracy:
        [[  5.98554253e-05   6.79918148e-07]
         [  8.59514088e+11   6.61703754e+11]
         [  6.02334592e-05   6.59713067e-07]]
        Traceback (innermost last):
          File "<console>", line 8, in <module>
        NoConvergence: 'HSTWCS.all_world2pix' failed to converge to the requested accuracy.
        After 5 iterations, the solution is diverging at least for one input point.

        >>> try:
        >>>   xy = w.all_world2pix(divradec,1, maxiter=20, tolerance=1.0e-4,
              adaptive=False, detect_divergence=False, quiet=False)
        >>> except stwcs.wcsutil.hstwcs.NoConvergence as e:
        >>>   print("Indices of diverging points: {}".format(e.divergent))
        >>>   print("Indices of poorly converging points: {}".format(e.failed2converge))
        >>>   print("Best solution: {}".format(e.best_solution))
        >>>   print("Achieved accuracy: {}".format(e.accuracy))
        >>>   raise e
        Indices of diverging points:
        [1]
        Indices of poorly converging points:
        None
        Best solution:
        [[  1.   1.]
         [ nan  nan]
         [  3.   1.]]
        Achieved accuracy:
        [[  0.   0.]
         [ nan  nan]
         [  0.   0.]]
        Traceback (innermost last):
          File "<console>", line 8, in <module>
        NoConvergence: 'HSTWCS.all_world2pix' failed to converge to the requested accuracy.
        After 20 iterations, the solution is diverging at least for one input point.

        """
        #####################################################################
        ##                     PROCESS ARGUMENTS:                          ##
        #####################################################################
        nargs = len(args)

        if nargs == 3:
            try:
                ra     = np.asarray(args[0], dtype=np.float64)
                dec    = np.asarray(args[1], dtype=np.float64)
                # assert( len(ra.shape) == 1 and len(dec.shape) == 1 )
                origin = int(args[2])
                vect1D = True
            except:
                raise TypeError("When providing three arguments, they must "
                                "be (Ra, Dec, origin) where Ra and Dec are "
                                "Nx1 vectors.")
        elif nargs == 2:
            try:
                rd  = np.asarray(args[0], dtype=np.float64)
                ra  = rd[:, 0]
                dec = rd[:, 1]
                origin = int(args[1])
                vect1D = False
            except:
                raise TypeError("When providing two arguments, they must "
                                "be (RaDec, origin) where RaDec is a Nx2 array.")
        else:
            raise TypeError("Expected 2 or 3 arguments, {:d} given.".format(nargs))

        # process optional arguments:
        tolerance          = kwargs.pop('tolerance', 1.0e-4)
        maxiter           = kwargs.pop('maxiter', 20)
        adaptive          = kwargs.pop('adaptive', False)
        detect_divergence = kwargs.pop('detect_divergence', True)
        quiet             = kwargs.pop('quiet', False)

        #####################################################################
        ##                INITIALIZE ITERATIVE PROCESS:                    ##
        #####################################################################
        x0, y0 = self.wcs_world2pix(ra, dec, origin)  # <-- initial approximation
                                                      #     (WCS based only)

        # see if an iterative solution is required (when any of the
        # non-CD-matrix corrections are present). If not required
        # return initial approximation (x0, y0).
        if self.sip is None and \
           self.cpdis1 is None and self.cpdis2 is None and \
           self.det2im1 is None and self.det2im2 is None:
            # no non-WCS corrections are detected - return
            # initial approximation
            if vect1D:
                return [x0, y0]
            else:
                return np.dstack([x0, y0])[0]

        x  = x0.copy()  # 0-order solution
        y  = y0.copy()  # 0-order solution

        # initial correction:
        dx, dy = self.pix2foc(x, y, origin)
        # If pix2foc does not apply all the required distortion
        # corrections then replace the above line with:
        # r0, d0 = self.all_pix2world(x, y, origin)
        # dx, dy = self.wcs_world2pix(r0, d0, origin )
        dx -= x0
        dy -= y0

        # update initial solution:
        x -= dx
        y -= dy

        # norn (L2) squared of the correction:
        dn2prev = dx ** 2 + dy ** 2
        dn2 = dn2prev

        # prepare for iterative process
        iterlist  = list(range(1, maxiter + 1))
        tolerance2 = tolerance ** 2
        ind = None
        inddiv = None

        npts = x.shape[0]

        # turn off numpy runtime warnings for 'invalid' and 'over':
        old_invalid = np.geterr()['invalid']
        old_over = np.geterr()['over']
        np.seterr(invalid='ignore', over='ignore')

        #####################################################################
        ##                     NON-ADAPTIVE ITERATIONS:                    ##
        #####################################################################
        if not adaptive:
            for k in iterlist:
                # check convergence:
                if np.max(dn2) < tolerance2:
                    break

                # find correction to the previous solution:
                dx, dy = self.pix2foc(x, y, origin)
                # If pix2foc does not apply all the required distortion
                # corrections then replace the above line with:
                # r0, d0 = self.all_pix2world(x, y, origin)
                # dx, dy = self.wcs_world2pix(r0, d0, origin )
                dx -= x0
                dy -= y0

                # update norn (L2) squared of the correction:
                dn2 = dx ** 2 + dy ** 2

                # check for divergence (we do this in two stages
                # to optimize performance for the most common
                # scenario when succesive approximations converge):
                if detect_divergence:
                    ind, = np.where(dn2 <= dn2prev)
                    if ind.shape[0] < npts:
                        inddiv, = np.where(
                            np.logical_and(dn2 > dn2prev, dn2 >= tolerance2))
                        if inddiv.shape[0] > 0:
                            # apply correction only to the converging points:
                            x[ind] -= dx[ind]
                            y[ind] -= dy[ind]
                            # switch to adaptive iterations:
                            ind, = np.where((dn2 >= tolerance2) &
                                            (dn2 <= dn2prev) & np.isfinite(dn2))
                            iterlist = iterlist[k:]
                            adaptive = True
                            break
                    # dn2prev[ind] = dn2[ind]
                    dn2prev = dn2

                # apply correction:
                x -= dx
                y -= dy

        #####################################################################
        ##                      ADAPTIVE ITERATIONS:                       ##
        #####################################################################
        if adaptive:
            if ind is None:
                ind = np.asarray(list(range(npts)), dtype=np.int64)

            for k in iterlist:
                # check convergence:
                if ind.shape[0] == 0:
                    break

                # find correction to the previous solution:
                dx[ind], dy[ind] = self.pix2foc(x[ind], y[ind], origin)
                # If pix2foc does not apply all the required distortion
                # corrections then replace the above line with:
                # r0[ind], d0[ind] = self.all_pix2world(x[ind], y[ind], origin)
                # dx[ind], dy[ind] = self.wcs_world2pix(r0[ind], d0[ind], origin)
                dx[ind] -= x0[ind]
                dy[ind] -= y0[ind]

                # update norn (L2) squared of the correction:
                dn2 = dx ** 2 + dy ** 2

                # update indices of elements that still need correction:
                if detect_divergence:
                    ind, = np.where((dn2 >= tolerance2) & (dn2 <= dn2prev))
                    # ind = ind[np.where((dn2[ind] >= tolerance2) & (dn2[ind] <= dn2prev))]
                    dn2prev[ind] = dn2[ind]
                else:
                    ind, = np.where(dn2 >= tolerance2)
                    # ind = ind[np.where(dn2[ind] >= tolerance2)]

                # apply correction:
                x[ind] -= dx[ind]
                y[ind] -= dy[ind]

        #####################################################################
        ##         FINAL DETECTION OF INVALID, DIVERGING,                  ##
        ##         AND FAILED-TO-CONVERGE POINTS                           ##
        #####################################################################
        # Identify diverging and/or invalid points:
        invalid = (((~np.isfinite(y)) | (~np.isfinite(x)) |
                    (~np.isfinite(dn2))) &
                   (np.isfinite(ra)) & (np.isfinite(dec)))
        # When detect_divergence==False, dn2prev is outdated (it is the
        # norm^2 of the very first correction). Still better than nothing...
        inddiv, = np.where(((dn2 >= tolerance2) & (dn2 > dn2prev)) | invalid)
        if inddiv.shape[0] == 0:
            inddiv = None
        # identify points that did not converge within
        # 'maxiter' iterations:
        if k >= maxiter:
            ind, = np.where((dn2 >= tolerance2) & (dn2 <= dn2prev) & (~invalid))
            if ind.shape[0] == 0:
                ind = None
        else:
            ind = None

        #####################################################################
        ##      RAISE EXCEPTION IF DIVERGING OR TOO SLOWLY CONVERGING      ##
        ##      DATA POINTS HAVE BEEN DETECTED:                            ##
        #####################################################################
        # raise exception if diverging or too slowly converging
        if (ind is not None or inddiv is not None) and not quiet:
            if vect1D:
                sol  = [x, y]
                err  = [np.abs(dx), np.abs(dy)]
            else:
                sol  = np.dstack([x, y] )[0]
                err  = np.dstack([np.abs(dx), np.abs(dy)] )[0]

            # restore previous numpy error settings:
            np.seterr(invalid=old_invalid, over=old_over)

            if inddiv is None:
                raise NoConvergence("'HSTWCS.all_world2pix' failed to "
                                    "converge to the requested accuracy after {:d} "
                                    "iterations.".format(k), best_solution=sol,
                                    accuracy=err, niter=k, failed2converge=ind,
                                    divergent=None)
            else:
                raise NoConvergence("'HSTWCS.all_world2pix' failed to "
                                    "converge to the requested accuracy.{0:s}"
                                    "After {1:d} iterations, the solution is diverging "
                                    "at least for one input point."
                                    .format(os.linesep, k), best_solution=sol,
                                    accuracy=err, niter=k, failed2converge=ind,
                                    divergent=inddiv)

        #####################################################################
        ##             FINALIZE AND FORMAT DATA FOR RETURN:                ##
        #####################################################################
        # restore previous numpy error settings:
        np.seterr(invalid=old_invalid, over=old_over)

        if vect1D:
            return [x, y]
        else:
            return np.dstack([x, y] )[0]

    def _updatehdr(self, ext_hdr):
        # kw2add : OCX10, OCX11, OCY10, OCY11
        # record the model in the header for use by pydrizzle
        ocx_comment = "original linear term from IDCTAB"
        ext_hdr['OCX10'] = self.idcmodel.cx[1, 0], ocx_comment
        ext_hdr['OCX11'] = self.idcmodel.cx[1, 1], ocx_comment
        ext_hdr['OCY10'] = self.idcmodel.cy[1, 0], ocx_comment
        ext_hdr['OCY11'] = self.idcmodel.cy[1, 1], ocx_comment
        ext_hdr['IDCSCALE'] = (self.idcmodel.refpix['PSCALE'],
                               "pixel scale from the IDCTAB reference file")
        ext_hdr['IDCTHETA'] = (self.idcmodel.refpix['THETA'],
                               "orientation of detector's Y-axis w.r.t. V3 axis")
        ext_hdr['IDCXREF'] = (self.idcmodel.refpix['XREF'],
                              "reference pixel location in X")
        ext_hdr['IDCYREF'] = (self.idcmodel.refpix['YREF'],
                              "reference pixel location in Y")
        ext_hdr['IDCV2REF'] = (self.idcmodel.refpix['V2REF'],
                               "reference pixel's V2 position")
        ext_hdr['IDCV3REF'] = (self.idcmodel.refpix['V3REF'],
                               "reference pixel's V3 position")

    def printwcs(self):
        """
        Print the basic WCS keywords.
        """
        print('WCS Keywords\n')
        print('CD_11  CD_12: %r %r' % (self.wcs.cd[0, 0], self.wcs.cd[0, 1]))
        print('CD_21  CD_22: %r %r' % (self.wcs.cd[1, 0], self.wcs.cd[1, 1]))
        print('CRVAL    : %r %r' % (self.wcs.crval[0], self.wcs.crval[1]))
        print('CRPIX    : %r %r' % (self.wcs.crpix[0], self.wcs.crpix[1]))
        print('NAXIS    : %d %d' % (self.naxis1, self.naxis2))
        print('Plate Scale : %r' % self.pscale)
        print('ORIENTAT : %r' % self.orientat)


def determine_refframe(phdr):
    """
    Determine the reference frame in standard FITS WCS.

    This is necessary for two reasons:
    - The reference frame in HST images is stored not in RADESYS (FITS standard) but in REFFRAME.
    - REFFRAME is in the primary header, while the rest of the WCS keywords are in the
      extension header.

    The values of REFFRAME are populated from the APT template where observers are
    given three options: GSC1 (corresponds to FK5), ICRS or OTHER.
    In the case of "OTHER", we leave this to wcslib which has a default of ICRS.

    Parameters
    ----------
    phdr : `astropy.io.fits.Header`
        Primary Header of an HST observation

    Returns
    -------
    refframe : str or None
        One of the FITS WCS standard reference frames.

    """
    try:
        refframe = phdr['REFFRAME'].upper()
    except KeyError:
        refframe = None
    if refframe == "GSC1":
        refframe = "FK5"
    elif refframe != "ICRS":
        refframe = None
    return refframe
