"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

import pytest

# Ensure repo root is on path so "common" and "hst123" can be imported when running tests
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def minimal_fits_file(tmp_path):
    """Create a minimal FITS file with PHOTFLAM/PHOTPLAM and INSTRUME for get_zpt/get_instrument tests."""
    from astropy.io import fits
    import numpy as np

    path = tmp_path / "test.fits"
    hdu0 = fits.PrimaryHDU()
    hdu0.header["INSTRUME"] = "WFC3"
    hdu0.header["DETECTOR"] = "UVIS"
    hdu0.header["SUBARRAY"] = False

    # Single SCI extension with photometry keywords (AB mag formula uses these)
    data = np.zeros((10, 10), dtype=np.float32)
    hdu1 = fits.ImageHDU(data=data, name="SCI")
    hdu1.header["CCDCHIP"] = 1
    hdu1.header["PHOTPLAM"] = 5000.0
    hdu1.header["PHOTFLAM"] = 1.0e-19  # arbitrary

    hdul = fits.HDUList([hdu0, hdu1])
    hdul.writeto(path, overwrite=True)
    return str(path)


@pytest.fixture
def hst123_instance():
    """Return an hst123 instance with options['args'] set to a minimal namespace for methods that need it."""
    try:
        import hst123 as _hst
    except Exception as e:
        pytest.skip(f"hst123 not importable (need drizzlepac/stwcs): {e}")
    import argparse
    from common import Options

    parser = argparse.ArgumentParser()
    parser = Options.add_options(parser)
    args = parser.parse_args(["0", "0"])  # ra dec positional
    hst = _hst.hst123()
    hst.options["args"] = args
    return hst
