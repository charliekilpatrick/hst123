# This dictionary maps an instrument into an instrument class
# The instrument class handles instrument specific keywords

inst_mappings = {'WFPC2': 'WFPC2WCS',
                 'ACS': 'ACSWCS',
                 'NICMOS': 'NICMOSWCS',
                 'STIS': 'STISWCS',
                 'WFC3': 'WFC3WCS',
                 'DEFAULT': 'InstrWCS'
                 }


# A list of instrument specific keywords
# Every instrument class must have methods which define each of these
# as class attributes.
ins_spec_kw = ['idctab', 'offtab', 'date_obs', 'ra_targ', 'dec_targ',
               'pav3', 'detector', 'ltv1', 'ltv2', 'parity', 'binned',
               'vafactor', 'chip', 'naxis1', 'naxis2', 'filter1', 'filter2']

# A list of keywords defined in the primary header.
# The HSTWCS class sets this as attributes
prim_hdr_kw = ['detector', 'offtab', 'idctab', 'date-obs',
               'pa_v3', 'ra_targ', 'dec_targ']

# These are the keywords which are archived before MakeWCS is run
basic_wcs = ['CD1_', 'CD2_', 'CRVAL', 'CTYPE', 'CRPIX', 'CTYPE',
             'CDELT', 'CUNIT']
