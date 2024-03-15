import warnings
warnings.filterwarnings('ignore')
import glob, sys, os, shutil, time, filecmp, astroquery, progressbar, copy
import smtplib, datetime, requests, random
import astropy.wcs as wcs
import numpy as np
from contextlib import contextmanager
from astropy import units as u
from astropy.utils.data import clear_download_cache,download_file
from astropy.io import fits
from astropy.table import Table, Column, unique
from astropy.time import Time
from astroscrappy import detect_cosmics
from dateutil.parser import parse
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Observations

coord = SkyCoord('01:36:41.8','+15:47:01',unit=(u.hour,u.deg))

table = Observations.query_region(coord, radius=5.0*u.arcmin)

print(table)
print(np.unique(table['obs_collection']))

mask = table['obs_collection']=='JWST'
print(table[mask])
obs=table[mask][2]
productList = Observations.get_product_list(obs)
mask = (productList['productType']=='SCIENCE') &\
    np.array(['i2d' in f for f in productList['productFilename']])
print(productList[mask])
print(productList[mask]['productFilename'])

