from astropy.io import fits
from stsci.tools import irafglob, fileutil, parseinput


def parseSingleInput(f=None, ext=None):
    if isinstance(f, str):
        # create an HSTWCS object from a filename
        if ext is not None:
            filename = f
            if isinstance(ext, tuple):
                if ext[0] == '':
                    extnum = ext[1]  # handle ext=('',1)
                else:
                    extnum = ext
            else:
                extnum = int(ext)
        elif ext is None:
            filename, ext = fileutil.parseFilename(f)
            ext = fileutil.parseExtn(ext)
            if ext[0] == '':
                extnum = int(ext[1])  # handle ext=('',extnum)
            else:
                extnum = ext
        phdu = fits.open(filename)
        hdr0 = phdu[0].header
        try:
            ehdr = phdu[extnum].header
        except (IndexError, KeyError) as e:
            raise e.__class__('Unable to get extension %s.' % extnum)

    elif isinstance(f, fits.HDUList):
        phdu = f
        if ext is None:
            extnum = 0
        else:
            extnum = ext
        ehdr = f[extnum].header
        hdr0 = f[0].header
        filename = hdr0.get('FILENAME', "")

    else:
        raise ValueError('Input must be a file name string or a'
                         '`astropy.io.fits.HDUList` object')

    return filename, hdr0, ehdr, phdu


def parseMultipleInput(input):
    if isinstance(input, str):
        if input[0] == '@':
            # input is an @ file
            filelist = irafglob.irafglob(input)
        else:
            try:
                filelist, output = parseinput.parseinput(input)
            except IOError: raise
    elif isinstance(input, list):
        #if isinstance(input[0], HSTWCS):
            ## a list of HSTWCS objects
            #return input
        #else:
        filelist = input[:]
    return filelist
