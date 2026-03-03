"""Unit tests for utils.display (format_instrument_display_name, show_photometry_data, etc.)."""
import io
import logging

import pytest
from astropy.table import Table

from hst123.utils.display import (
    format_instrument_display_name,
    show_photometry_data,
    write_snana_photometry,
    show_photometry,
)


class TestFormatInstrumentDisplayName:
    def test_wfc3_uvis(self):
        assert format_instrument_display_name("wfc3_uvis") == "WFC3/UVIS"
        assert format_instrument_display_name("WFC3") == "WFC3"

    def test_wfc3_ir(self):
        assert format_instrument_display_name("wfc3_ir") == "WFC3/IR"

    def test_acs_wfc(self):
        assert format_instrument_display_name("acs_wfc") == "ACS/WFC"

    def test_wfpc2(self):
        assert format_instrument_display_name("wfpc2") == "WFPC2"

    def test_underscore_fallback(self):
        assert format_instrument_display_name("other_det") == "OTHER/DET"


class TestShowPhotometryData:
    def test_logs_and_writes(self, caplog):
        log = logging.getLogger("test_display")
        phottable = Table({
            "INSTRUMENT": ["wfc3_uvis"],
            "FILTER": ["f475w"],
            "MJD": [58000.0],
            "EXPTIME": [100.0],
            "MAGNITUDE": [20.0],
            "MAGNITUDE_ERROR": [0.05],
        })
        form = "{date: <12} {inst: <10} {filt: <8} {exp: <14} {mag: <9} {err: <11}"
        header = "# MJD Instrument Filter Exposure Magnitude Uncertainty"
        f = io.StringIO()
        show_photometry_data(phottable, form, header, "", log, file=f, avg=False)
        out = f.getvalue()
        assert "WFC3/UVIS" in out
        assert "f475w" in out or "F475W" in out
        assert "20.0" in out

    def test_avg_mode(self, caplog):
        log = logging.getLogger("test_display")
        phottable = Table({
            "INSTRUMENT": ["acs_wfc"],
            "FILTER": ["f606w"],
            "MJD": [99999.0],
            "EXPTIME": [200.0],
            "MAGNITUDE": [19.5],
            "MAGNITUDE_ERROR": [0.1],
        })
        f = io.StringIO()
        form = "{date} {inst} {filt} {exp} {mag} {err}"
        show_photometry_data(
            phottable, form, "", "", log, file=f, avg=True
        )
        assert "Average Photometry" in f.getvalue()


class TestWriteSnanaPhotometry:
    def test_writes_header_and_rows(self):
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        phottable = Table({
            "INSTRUMENT": ["wfc3_uvis"],
            "FILTER": ["f475w"],
            "MJD": [58000.0],
            "MAGNITUDE": [20.0],
            "MAGNITUDE_ERROR": [0.05],
        })
        coord = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)
        f = io.StringIO()
        write_snana_photometry(phottable, f, coord, "test_obj", 25.0)
        out = f.getvalue()
        assert "SNID: test_obj" in out
        assert "RA:" in out
        assert "VARLIST" in out
        assert "OBS:" in out


class TestShowPhotometry:
    def test_key_error_logged(self, caplog):
        log = logging.getLogger("test_display")
        bad_table = Table({"INSTRUMENT": ["a"], "FILTER": ["b"]})  # missing MJD, etc.
        result = show_photometry(
            bad_table, latex=False, show=True, log=log,
            coord=None, options=None,
        )
        assert result is None
        assert "key error" in caplog.text.lower() or "ERROR" in caplog.text

    def test_full_table_show(self, caplog):
        log = logging.getLogger("test_display")
        phottable = Table({
            "INSTRUMENT": ["wfc3_uvis"],
            "FILTER": ["f475w"],
            "MJD": [58000.0],
            "EXPTIME": [100.0],
            "MAGNITUDE": [20.0],
            "MAGNITUDE_ERROR": [0.05],
            "IS_AVG": [0],
        })
        f = io.StringIO()
        show_photometry(
            phottable, latex=False, show=True, f=f,
            snana=False, coord=None, options=None, log=log,
        )
        assert "WFC3" in f.getvalue() or "20.0" in f.getvalue()
