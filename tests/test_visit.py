"""Unit tests for utils.visit (add_visit_info)."""
import logging

import pytest
from astropy.table import Table
from astropy.time import Time

from hst123.utils.visit import add_visit_info


def _obstable(datetimes, instruments, filters, images=None):
    if images is None:
        images = [f"im_{i}.fits" for i in range(len(datetimes))]
    return Table({
        "datetime": datetimes,
        "instrument": instruments,
        "filter": filters,
        "image": images,
    })


class TestAddVisitInfo:
    def test_single_row_gets_visit_one(self):
        t = Time("2020-01-01T00:00:00", format="isot")
        obs = _obstable([t], ["wfc3"], ["f475w"])
        out = add_visit_info(obs, visit_tol=1.0)
        assert out is not None
        assert list(out["visit"]) == [1]

    def test_two_same_inst_filter_close_time_same_visit(self):
        t0 = Time("2020-01-01T00:00:00", format="isot")
        t1 = Time("2020-01-01T12:00:00", format="isot")  # 0.5 day
        obs = _obstable(
            [t0, t1],
            ["wfc3", "wfc3"],
            ["f475w", "f475w"],
        )
        out = add_visit_info(obs, visit_tol=1.0)
        assert out is not None
        assert list(out["visit"]) == [1, 1]

    def test_two_far_apart_different_visits(self):
        t0 = Time("2020-01-01T00:00:00", format="isot")
        t1 = Time("2020-01-10T00:00:00", format="isot")
        obs = _obstable(
            [t0, t1],
            ["wfc3", "wfc3"],
            ["f475w", "f475w"],
        )
        out = add_visit_info(obs, visit_tol=1.0)
        assert out is not None
        assert sorted(set(out["visit"])) == [1, 2]

    def test_log_on_error(self, caplog):
        log = logging.getLogger("test_visit")
        # Malformed table could cause issues; at least ensure we don't crash with empty
        obs = _obstable([], [], [])
        out = add_visit_info(obs, visit_tol=1.0, log=log)
        assert out is not None
        assert "visit" in out.colnames
