"""Unit tests for common.Settings (structure and expected keys)."""
import pytest

from hst123.common import Settings


class TestGlobalDefaults:
    def test_has_required_keys(self):
        g = Settings.global_defaults
        assert "archive" in g
        assert "astropath" in g
        assert "keys" in g
        assert "cdbs" in g
        assert "mast" in g
        assert "radius" in g
        assert "visit" in g
        assert "search_rad" in g
        assert "dolphot" in g
        assert "fake" in g

    def test_dolphot_subdict(self):
        d = Settings.global_defaults["dolphot"]
        assert "FitSky" in d
        assert "AlignOnly" in d
        assert "SigFind" in d


class TestCatalogPars:
    def test_has_tweakreg_relevant_keys(self):
        c = Settings.catalog_pars
        assert "conv_width" in c
        assert "sharplo" in c
        assert "sharphi" in c
        assert "use_sharp_round" in c


class TestInstrumentDefaults:
    def test_wfc3_acs_wfpc2(self):
        assert "wfc3" in Settings.instrument_defaults
        assert "acs" in Settings.instrument_defaults
        assert "wfpc2" in Settings.instrument_defaults

    def test_crpars_present(self):
        for inst in ("wfc3", "acs", "wfpc2"):
            assert "crpars" in Settings.instrument_defaults[inst]
            assert "rdnoise" in Settings.instrument_defaults[inst]["crpars"]
            assert "gain" in Settings.instrument_defaults[inst]["crpars"]


class TestDetectorDefaults:
    def test_detectors_present(self):
        assert "wfc3_uvis" in Settings.detector_defaults
        assert "wfc3_ir" in Settings.detector_defaults
        assert "acs_wfc" in Settings.detector_defaults
        assert "wfpc2_wfpc2" in Settings.detector_defaults

    def test_detector_has_driz_bits_and_pixel_scale(self):
        for det in ("wfc3_uvis", "acs_wfc"):
            d = Settings.detector_defaults[det]
            assert "driz_bits" in d
            assert "pixel_scale" in d
            assert "input_files" in d


class TestNamesAndLists:
    def test_names_list(self):
        assert "image" in Settings.names
        assert "filter" in Settings.names
        assert "instrument" in Settings.names

    def test_final_names(self):
        assert "MJD" in Settings.final_names
        assert "MAGNITUDE" in Settings.final_names
        assert "MAGNITUDE_ERROR" in Settings.final_names

    def test_acceptable_filters_non_empty(self):
        assert len(Settings.acceptable_filters) > 0
        assert "F814W" in Settings.acceptable_filters

    def test_pipeline_products_and_images(self):
        assert len(Settings.pipeline_products) > 0
        assert len(Settings.pipeline_images) > 0
