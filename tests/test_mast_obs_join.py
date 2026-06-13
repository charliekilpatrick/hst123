"""Tests for MAST observation ↔ product table joins."""
import pytest
from astropy.table import Table

from hst123._pipeline import hst123


def test_mast_product_obs_key_column_prefers_obsID():
    products = Table({"obsID": [1], "obs_id": ["ABC"], "productFilename": ["a.fits"]})
    assert hst123._mast_product_obs_key_column(products) == "obsID"


def test_mast_lookup_observation_by_obsID():
    obs = Table(
        {
            "obsid": [2003738726],
            "obs_id": ["U9O40504M"],
            "instrument_name": ["WFPC2/WFC"],
            "s_ra": [23.0],
            "s_dec": [60.0],
        }
    )
    by_obsid, by_obs_id = hst123._mast_observation_lookup_maps(obs)
    row = hst123._mast_lookup_observation(by_obsid, by_obs_id, 2003738726, "obsID")
    assert row["obs_id"] == "U9O40504M"


def test_mast_lookup_observation_by_obs_id():
    obs = Table(
        {
            "obsid": [2003738726],
            "obs_id": ["U9O40504M"],
            "instrument_name": ["WFPC2/WFC"],
            "s_ra": [23.0],
            "s_dec": [60.0],
        }
    )
    by_obsid, by_obs_id = hst123._mast_observation_lookup_maps(obs)
    row = hst123._mast_lookup_observation(by_obsid, by_obs_id, "U9O40504M", "obs_id")
    assert str(row["obsid"]) == "2003738726"


def test_mast_product_obs_key_column_missing_raises():
    with pytest.raises(KeyError, match="obsID"):
        hst123._mast_product_obs_key_column(Table({"productFilename": ["a.fits"]}))


def test_mast_resolve_product_observation_uses_parent_obsid():
    """WFC3 FLC rows use exposure-level obsID; parent_obsid matches cone search."""
    obs = Table(
        {
            "obsid": [207202736],
            "obs_id": ["iey902020"],
            "instrument_name": ["WFC3/UVIS"],
            "s_ra": [177.65],
            "s_dec": [55.35],
        }
    )
    products = Table(
        {
            "obsID": [207202771],
            "parent_obsid": [207202736],
            "productFilename": ["iey902shq_flc.fits"],
        }
    )
    by_obsid, by_obs_id = hst123._mast_observation_lookup_maps(obs)
    key_col = hst123._mast_product_obs_key_column(products)
    row = hst123._mast_resolve_product_observation(
        by_obsid, by_obs_id, products[0], key_col,
    )
    assert row["instrument_name"] == "WFC3/UVIS"


@pytest.mark.parametrize(
    "instrument,expected",
    [
        ("WFPC2/PC", True),
        ("WFPC2/WFC", True),
        ("ACS/WFC", True),
        ("ACS/HRC", True),
        ("WFC3/UVIS", True),
        ("WFC3/IR", True),
        ("ACS/SBC", False),
        ("STIS/CCD", False),
    ],
)
def test_mast_observation_instrument_ok(instrument, expected):
    assert hst123._mast_observation_instrument_ok(instrument) is expected
