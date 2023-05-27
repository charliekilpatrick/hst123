from drizzlepac import tweakreg
import glob

from astropy.io import fits
import stwcs

files = ['wfpc2.f814w.ut000622_0001.drz.fits']

for file in files:
                wcsnames = []
                hdulist = fits.open(file)
                print(file)
                for i,h in enumerate(hdulist):
                    alt = stwcs.wcsutil.altwcs.wcsnames(file,
                        ext=i, include_primary=False)
                    print(alt)
                    for key in alt.keys():
                        if key!='O' and key.strip():
                            print('Deleting {0} key from {1},{2}'.format(key,
                                file, i))
                            stwcs.wcsutil.altwcs.deleteWCS(file,[i],wcskey=key)


                    if ('WCSNAME' in h.header.keys() and
                        h.header['WCSNAME']=='TWEAK'):
                        hdulist[i].header['WCSNAME']='ORIG'

                hdulist.writeto(file, overwrite=True)

tweakreg.TweakReg(files=files, refimage='wfc3.f814w.ut210405_0003.drz.fits',
    interactive=True, updatehdr=True, verbose=True,
    minobj=7,
    imagefindcfg = {'threshold': 10.0},
    refimagefindcfg = {'threshold': 10.0},
    refnbright=10000, nbright=10000,
    xoffset=-4.818, yoffset=28.31,
    fitgeometry='general', shiftfile=True,
    searchrad=10.0, searchunits='arcsec')
