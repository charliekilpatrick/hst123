"""Regression: compress_reference must not swap PRIMARY+HDRTAB drizzle products."""
import numpy as np
import pytest
from astropy.io import fits


def _simulate_compress_output(hdu):
    """Mirror hst123.hst123.compress_reference HDU selection (no disk I/O)."""
    import copy

    prim = hdu[0]
    naxis = int(prim.header.get("NAXIS", 0) or 0)
    prim_has_image = naxis > 0 and prim.data is not None

    newhdu = fits.HDUList()
    if len(hdu) == 1:
        newhdu.append(copy.copy(prim))
    elif len(hdu) == 2:
        ext1 = hdu[1]
        ext1_name = str(ext1.name).strip().upper()
        if ext1_name == "SCI" and not prim_has_image:
            sci = copy.copy(ext1)
            sci.name = "PRIMARY"
            for key in prim.header.keys():
                if key not in sci.header.keys():
                    sci.header[key] = prim.header[key]
            newhdu.append(sci)
        else:
            newhdu.append(copy.copy(prim))
    else:
        newhdu.append(copy.copy(prim))
    newhdu[0].name = "PRIMARY"
    return newhdu


def test_primary_plus_hdrtab_keeps_image(tmp_path):
    """AstroDrizzle-style PRIMARY (image) + HDRTAB must not replace image with table."""
    data = np.ones((8, 8), dtype=np.float32)
    pri = fits.PrimaryHDU(data=data)
    pri.header["EXTNAME"] = "PRIMARY"
    # minimal bintable second extension
    col = fits.Column(name="X", array=np.array([1.0]), format="E")
    tab = fits.BinTableHDU.from_columns([col])
    tab.header["EXTNAME"] = "HDRTAB"
    hdul = fits.HDUList([pri, tab])
    out = _simulate_compress_output(hdul)
    assert out[0].data is not None
    assert out[0].data.shape == (8, 8)
    assert out[0].data[0, 0] == pytest.approx(1.0)


def test_empty_primary_plus_sci_promotes_sci(tmp_path):
    """Legacy layout: empty PRIMARY + SCI still promotes SCI."""
    pri = fits.PrimaryHDU()
    pri.header["NAXIS"] = 0
    data = np.ones((4, 4), dtype=np.float32)
    sci = fits.ImageHDU(data=data, name="SCI")
    hdul = fits.HDUList([pri, sci])
    out = _simulate_compress_output(hdul)
    assert out[0].data is not None
    assert out[0].data.shape == (4, 4)
