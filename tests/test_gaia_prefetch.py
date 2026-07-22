"""Unit tests for Gaia prefetch helpers (no network)."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from hst123.utils import gaia_prefetch as gp


def test_gaia_prefetch_cache_path_rounding(tmp_path):
    c = SkyCoord(45.123456789 * u.deg, -12.987654321 * u.deg, frame="icrs")
    p = gp.gaia_prefetch_cache_path(tmp_path, c, 0.36666667 * u.deg)
    assert (
        f"hst123_gaia_dr3_prefetch_{gp.GAIA_PREFETCH_CACHE_VERSION}"
        "_ra45.1235_dec-12.9877_r0.36667deg.txt"
    ) in p.replace("\\", "/")


def test_icrs_field_center_from_fits_primary(tmp_path):
    w = WCS(naxis=2)
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.crval = [120.0, -22.5]
    w.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    data = np.zeros((10, 10), dtype=np.float32)
    phdu = fits.PrimaryHDU(data=data, header=w.to_header())
    path = tmp_path / "wcs.fits"
    phdu.writeto(path, overwrite=True)
    cen = gp.icrs_field_center_from_fits(str(path))
    assert abs(cen.ra.deg - 120.0) < 0.02
    assert abs(cen.dec.deg - (-22.5)) < 0.02


def test_write_jhat_refcat_roundtrip(tmp_path):
    tab = Table()
    tab["ra"] = [10.0, 11.0]
    tab["dec"] = [20.0, 21.0]
    tab["phot_g_mean_mag"] = [18.0, 19.0]
    tab["phot_g_mean_mag_error"] = [0.01, 0.02]
    tab["pmra"] = [1.0, -2.0]
    tab["pmdec"] = [0.5, 0.25]
    tab["parallax"] = [1.2, 0.8]
    tab["ruwe"] = [1.0, 1.1]
    tab["astrometric_excess_noise"] = [0.1, 0.2]
    out = tmp_path / "ref.txt"
    gp._write_jhat_refcat(str(out), tab=tab)
    t2 = Table.read(str(out), format="ascii.basic")
    assert list(t2.colnames)[:4] == ["ra", "dec", "mag", "dmag"]
    assert "pmra" in t2.colnames and "parallax" in t2.colnames
    assert "ruwe" in t2.colnames and "astrometric_excess_noise" in t2.colnames
    assert len(t2) == 2
