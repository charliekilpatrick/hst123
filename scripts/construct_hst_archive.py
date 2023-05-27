#!/usr/bin/env python3
# Designed to be run in a directory with hst123 subdirectories
# i.e., directory with */raw/ dirs
import os, glob,numpy as np,shutil
from astropy.io import fits

archivedir = '/data2/ckilpatrick/hst/archive/'

for directory in glob.glob('*'):
    if not os.path.isdir(directory):
        continue

    if not os.path.exists(directory + '/raw/'):
        continue

    for file in glob.glob(directory + '/raw/*.fits'):

        path,basefile = os.path.split(file)

        hdu = fits.open(file)
        inst = hdu[0].header['INSTRUME']
        ra = str(int(np.round(hdu[0].header['RA_TARG'])))

        filefmt = '{inst}/{det}/{ra}/{name}'

        if 'ACS' in inst:
            det = hdu[0].header['DETECTOR']
        elif 'WFC3' in inst:
            det = hdu[0].header['DETECTOR']
        elif 'WFPC2' in inst:
            det = 'WFPC2'
        # Just assume instrument is WFPC2
        else:
            det = 'WFPC2'

        fulloutfile = archivedir + filefmt.format(inst=inst, det=det, ra=ra,
            name=basefile)

        outpath, outfile = os.path.split(fulloutfile)

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if not os.path.exists(fulloutfile):
            print('Copying {0} to {1}'.format(file, fulloutfile))
            shutil.copyfile(file, fulloutfile)
        else:
            print('{0} already exists'.format(fulloutfile))
