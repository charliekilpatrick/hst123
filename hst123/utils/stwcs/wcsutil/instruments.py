
class InstrWCS:
    """
    A base class for instrument specific keyword definition.
    It prvides a default implementation (modeled by ACS) for
    all set_kw methods.
    """
    def __init__(self, hdr0=None, hdr=None):
        self.exthdr = hdr
        self.primhdr = hdr0
        self.set_ins_spec_kw()

    def set_ins_spec_kw(self):
        """
        This method MUST call all set_kw methods.
        There should be a set_kw method for all kw listed in
        mappings.ins_spec_kw. TypeError handles the case when
        fobj='DEFAULT'.
        """
        self.set_idctab()
        self.set_offtab()
        self.set_date_obs()
        self.set_ra_targ()
        self.set_dec_targ()
        self.set_pav3()
        self.set_detector()
        self.set_filter1()
        self.set_filter2()
        self.set_vafactor()
        self.set_naxis1()
        self.set_naxis2()
        self.set_ltv1()
        self.set_ltv2()
        self.set_binned()
        self.set_chip()
        self.set_parity()

    def set_idctab(self):
        try:
            self.idctab = self.primhdr['IDCTAB']
        except (KeyError, TypeError):
            self.idctab = None

    def set_offtab(self):
        try:
            self.offtab = self.primhdr['OFFTAB']
        except (KeyError, TypeError):
            self.offtab = None

    def set_date_obs(self):
        try:
            self.date_obs = self.primhdr['DATE-OBS']
        except (KeyError, TypeError):
            self.date_obs = None

    def set_ra_targ(self):
        try:
            self.ra_targ = self.primhdr['RA-TARG']
        except (KeyError, TypeError):
            self.ra_targ = None

    def set_dec_targ(self):
        try:
            self.dec_targ = self.primhdr['DEC-TARG']
        except (KeyError, TypeError):
            self.dec_targ = None

    def set_pav3(self):
        try:
            self.pav3 = self.primhdr['PA_V3']
        except (KeyError, TypeError):
            self.pav3 = None

    def set_filter1(self):
        try:
            self.filter1 = self.primhdr['FILTER1']
        except (KeyError, TypeError):
            self.filter1 = None

    def set_filter2(self):
        try:
            self.filter2 = self.primhdr['FILTER2']
        except (KeyError, TypeError):
            self.filter2 = None

    def set_vafactor(self):
        try:
            self.vafactor = self.exthdr['VAFACTOR']
        except (KeyError, TypeError):
            self.vafactor = 1

    def set_naxis1(self):
        try:
            self.naxis1 = self.exthdr['naxis1']
        except (KeyError, TypeError):
            try:
                self.naxis1 = self.exthdr['npix1']
            except (KeyError, TypeError):
                self.naxis1 = None

    def set_naxis2(self):
        try:
            self.naxis2 = self.exthdr['naxis2']
        except (KeyError, TypeError):
            try:
                self.naxis2 = self.exthdr['npix2']
            except (KeyError, TypeError):
                self.naxis2 = None

    def set_ltv1(self):
        try:
            self.ltv1 = self.exthdr['LTV1']
        except (KeyError, TypeError):
            self.ltv1 = 0.0

    def set_ltv2(self):
        try:
            self.ltv2 = self.exthdr['LTV2']
        except (KeyError, TypeError):
            self.ltv2 = 0.0

    def set_binned(self):
        try:
            self.binned = self.exthdr['BINAXIS1']
        except (KeyError, TypeError):
            self.binned = 1

    def set_chip(self):
        try:
            self.chip = self.exthdr['CCDCHIP']
        except (KeyError, TypeError):
            self.chip = 1

    def set_parity(self):
        self.parity = [[1.0, 0.0], [0.0, -1.0]]

    def set_detector(self):
        # each instrument has a different kw for detector and it can be
        # in a different header, so this is to be handled by the instrument classes
        self.detector = 'DEFAULT'


class ACSWCS(InstrWCS):
    """
    get instrument specific kw
    """

    def __init__(self, hdr0, hdr):
        self.primhdr = hdr0
        self.exthdr = hdr
        InstrWCS.__init__(self, hdr0, hdr)
        self.set_ins_spec_kw()

    def set_detector(self):
        try:
            self.detector = self.primhdr['DETECTOR']
        except KeyError:
            print('ERROR: Detector kw not found.\n')
            raise

    def set_parity(self):
        parity = {'WFC': [[1.0, 0.0], [0.0, -1.0]],
                  'HRC': [[-1.0, 0.0], [0.0, 1.0]],
                  'SBC': [[-1.0, 0.0], [0.0, 1.0]]}

        if self.detector not in list(parity.keys()):
            parity = InstrWCS.set_parity(self)
        else:
            self.parity = parity[self.detector]


class WFPC2WCS(InstrWCS):

    def __init__(self, hdr0, hdr):
        self.primhdr = hdr0
        self.exthdr = hdr
        InstrWCS.__init__(self, hdr0, hdr)
        self.set_ins_spec_kw()

    def set_filter1(self):
        self.filter1 = self.primhdr.get('FILTNAM1', None)
        if self.filter1 == " " or self.filter1 is None:
            self.filter1 = 'CLEAR1'

    def set_filter2(self):
        self.filter2 = self.primhdr.get('FILTNAM2', None)
        if self.filter2 == " " or self.filter2 is None:
            self.filter2 = 'CLEAR2'

    def set_binned(self):
        mode = self.primhdr.get('MODE', 1)
        if mode == 'FULL':
            self.binned = 1
        elif mode == 'AREA':
            self.binned = 2

    def set_chip(self):
        self.chip = self.exthdr.get('DETECTOR', 1)

    def set_parity(self):
        self.parity = [[-1.0, 0.], [0., 1.0]]

    def set_detector(self):
        try:
            self.detector = self.exthdr['DETECTOR']
        except KeyError:
            print('ERROR: Detector kw not found.\n')
            raise


class WFC3WCS(InstrWCS):
    """
    Create a WFC3 detector specific class
    """

    def __init__(self, hdr0, hdr):
        self.primhdr = hdr0
        self.exthdr = hdr
        InstrWCS.__init__(self, hdr0, hdr)
        self.set_ins_spec_kw()

    def set_detector(self):
        try:
            self.detector = self.primhdr['DETECTOR']
        except KeyError:
            print('ERROR: Detector kw not found.\n')
            raise

    def set_filter1(self):
        self.filter1 = self.primhdr.get('FILTER', None)
        if self.filter1 == " " or self.filter1 is None:
            self.filter1 = 'CLEAR'

    def set_filter2(self):
        # Nicmos idc tables do not allow 2 filters.
        self.filter2 = 'CLEAR'

    def set_parity(self):
        parity = {'UVIS': [[-1.0, 0.0], [0.0, 1.0]],
                  'IR': [[-1.0, 0.0], [0.0, 1.0]]}

        if self.detector not in list(parity.keys()):
            parity = InstrWCS.set_parity(self)
        else:
            self.parity = parity[self.detector]


class NICMOSWCS(InstrWCS):
    """
    Create a NICMOS specific class
    """

    def __init__(self, hdr0, hdr):
        self.primhdr = hdr0
        self.exthdr = hdr
        InstrWCS.__init__(self, hdr0, hdr)
        self.set_ins_spec_kw()

    def set_parity(self):
        self.parity = [[-1.0, 0.], [0., 1.0]]

    def set_filter1(self):
        self.filter1 = self.primhdr.get('FILTER', None)
        if self.filter1 == " " or self.filter1 is None:
            self.filter1 = 'CLEAR'

    def set_filter2(self):
        # Nicmos idc tables do not allow 2 filters.
        self.filter2 = 'CLEAR'

    def set_chip(self):
        self.chip = self.detector

    def set_detector(self):
        try:
            self.detector = self.primhdr['CAMERA']
        except KeyError:
            print('ERROR: Detector kw not found.\n')
            raise


class STISWCS(InstrWCS):
    """
    A STIS specific class
    """

    def __init__(self, hdr0, hdr):
        self.primhdr = hdr0
        self.exthdr = hdr
        InstrWCS.__init__(self, hdr0, hdr)
        self.set_ins_spec_kw()

    def set_parity(self):
        self.parity = [[-1.0, 0.], [0., 1.0]]

    def set_filter1(self):
        self.filter1 = self.exthdr.get('OPT_ELEM', None)
        if self.filter1 == " " or self.filter1 is None:
            self.filter1 = 'CLEAR1'

    def set_filter2(self):
        self.filter2 = self.exthdr.get('FILTER', None)
        if self.filter2 == " " or self.filter2 is None:
            self.filter2 = 'CLEAR2'

    def set_detector(self):
        try:
            self.detector = self.primhdr['DETECTOR']
        except KeyError:
            print('ERROR: Detector kw not found.\n')
            raise

    def set_date_obs(self):
        try:
            self.date_obs = self.exthdr['DATE-OBS']
        except (KeyError, TypeError):
            self.date_obs = None
