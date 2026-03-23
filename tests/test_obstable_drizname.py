"""Obstable drizname column must hold full paths (regression: no U99 truncation)."""
import numpy as np
from astropy.table import Column, Table


def test_drizname_object_column_retains_long_paths():
    long_base = "acs.f555w.ut221110_0001.drz.fits"
    long_path = "/very/long/work/dir/prefix/" * 5 + long_base
    t = Table([[1]], names=["x"])
    t.add_column(Column(np.empty(1, dtype=object), name="drizname"))
    t[0]["drizname"] = long_path
    assert t[0]["drizname"] == long_path
    assert str(t[0]["drizname"]).endswith(".drz.fits")
