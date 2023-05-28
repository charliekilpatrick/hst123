from astropy import units as u
from astropy.coordinates import SkyCoord

# Utility methods that do not need to be part of hst123 class (i.e., don't 
# reference data inside of class).

# Make sure all standard output is formatted in the same way with banner
# messages for each module so they are clear to see in stdout
def make_banner(message):
    print('\n\n'+message+'\n'+'#'*80+'\n'+'#'*80+'\n\n')

# Check if num is a number
def is_number(num):
    try:
        num = float(num)
    except ValueError:
        return(False)
    return(True)

def parse_coord(ra, dec):
    if (not (is_number(ra) and is_number(dec)) and
        (':' not in str(ra) and ':' not in str(dec))):
        error = 'ERROR: cannot interpret: {ra} {dec}'
        print(error.format(ra=ra, dec=dec))
        return(None)

    if (':' in str(ra) and ':' in str(dec)):
        # Input RA/DEC are sexagesimal
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        coord = SkyCoord(ra, dec, frame='icrs', unit=unit)
        return(coord)
    except ValueError:
        error = f'ERROR: Cannot parse coordinates: {ra} {dec}'
        print(error)
        return(None)
