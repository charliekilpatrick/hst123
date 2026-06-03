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
