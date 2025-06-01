#!/usr/bin/env python
from drizzlepac import tweakreg,astrodrizzle,catalogs,photeq
from astropy.io import fits

logfile_name = 'test.log'
files = ['u6h90y01m_c0m.drztmp.fits',
             'u6h90z01m_c0m.drztmp.fits',
             'u6h91001m_c0m.drztmp.fits']

output_name='test.drz.fits'
clean=True
skysub=True
wcskey='TWEAK'
rotation=0.0
driz_sep_scale=0.046
nx=5200
ny=5200
ra=344.322167
dec=-41.015961
driz_bits=1032
wht_type='EXP'
combine_type='median'
skymask_cat='skymask_cat'
pixscale=0.1

tmp_input=[]
for i,file in enumerate(files):
    hdu = fits.open(file)
    newhdu = fits.HDUList()

    for h in hdu:
        if h.name!='WCSCORR':
            newhdu.append(h)

    file=f'test{i}.drztmp.fits'
    tmp_input.append(file)
    newhdu.writeto(f'test{i}.drztmp.fits',overwrite=True,output_verify='silentfix')


astrodrizzle.AstroDrizzle(tmp_input, output=output_name,
                runfile=logfile_name,
                wcskey=wcskey, context=True, group='', build=False,
                num_cores=8, preserve=False, clean=clean, skysub=skysub,
                skymethod='globalmin+match', skymask_cat=skymask_cat,
                skystat='mode', skylower=0.0, skyupper=None, updatewcs=True,
                driz_sep_fillval=None, driz_sep_bits=driz_bits,
                driz_sep_wcs=True, driz_sep_rot=rotation,
                driz_sep_scale=driz_sep_scale,
                driz_sep_outnx=nx, driz_sep_outny=ny,
                driz_sep_ra=ra, driz_sep_dec=dec, driz_sep_pixfrac=0.8,
                combine_maskpt=0.2, combine_type=combine_type,
                combine_nlow=0, combine_nhigh=0,
                combine_lthresh=-10000, combine_hthresh=None,
                combine_nsigma='4 3', driz_cr_corr=True,
                driz_cr=True, driz_cr_snr='3.5 3.0', driz_cr_grow=1,
                driz_cr_ctegrow=0, driz_cr_scale='1.2 0.7',
                final_pixfrac=0.8, final_fillval=None,
                final_bits=driz_bits, final_units='counts',
                final_wcs=True, final_refimage=None, final_wht_type=wht_type,
                final_rot=rotation, final_scale=pixscale,
                final_outnx=nx, final_outny=ny,
                final_ra=ra, final_dec=dec,
                coeffs=False)
