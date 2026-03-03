"""Unit tests for hst123.settings (structure and expected keys)."""
import pytest

from hst123 import settings


class TestGlobalDefaults:
    def test_has_required_keys(self):
        g = settings.global_defaults
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

    def test_pipeline_defaults_in_global_defaults(self):
        g = settings.global_defaults
        assert g["rawdir"] == "raw"
        assert g["summary"] == "exposure_summary.out"
        assert g["magsystem"] == "abmag"
        assert g["default_threshold"] == 10.0
        assert g["snr_limit"] == 3.0
        assert g["output_zpt"] == 27.5
        assert g["mask_region_size"] == 200
        assert g["ab_flux_zero_mjy"] == 3631e-3

    def test_dolphot_subdict(self):
        d = settings.global_defaults["dolphot"]
        assert "FitSky" in d
        assert "AlignOnly" in d
        assert "SigFind" in d


class TestDrizzleDefaults:
    def test_has_required_keys(self):
        d = settings.drizzle_defaults
        assert d["num_cores"] == 8
        assert d["driz_sep_pixfrac"] == 0.8
        assert d["final_pixfrac"] == 0.8
        assert d["combine_maskpt"] == 0.2
        assert d["combine_nsigma"] == "4 3"
        assert d["driz_cr_snr"] == "3.5 3.0"
        assert d["driz_cr_grow"] == 1
        assert d["driz_cr_scale"] == "1.2 0.7"
        assert d["driz_cr_ctegrow"] == 0


class TestTweakregDefaults:
    def test_has_required_keys(self):
        t = settings.tweakreg_defaults
        assert t["threshold_min"] == 3.0
        assert t["threshold_max"] == 1000.0
        assert t["separation"] == 0.5
        assert t["minobj_fallback"] == 7
        assert t["conv_width"] == 3.5
        assert t["tolerance"] == 0.25
        assert "detector_overrides" in t
        assert "wfc3_ir" in t["detector_overrides"]
        assert "wfpc2" in t["detector_overrides"]


class TestCatalogPars:
    def test_has_tweakreg_relevant_keys(self):
        c = settings.catalog_pars
        assert "conv_width" in c
        assert "sharplo" in c
        assert "sharphi" in c
        assert "use_sharp_round" in c


class TestInstrumentDefaults:
    def test_wfc3_acs_wfpc2(self):
        assert "wfc3" in settings.instrument_defaults
        assert "acs" in settings.instrument_defaults
        assert "wfpc2" in settings.instrument_defaults

    def test_crpars_present(self):
        for inst in ("wfc3", "acs", "wfpc2"):
            assert "crpars" in settings.instrument_defaults[inst]
            assert "rdnoise" in settings.instrument_defaults[inst]["crpars"]
            assert "gain" in settings.instrument_defaults[inst]["crpars"]


class TestDetectorDefaults:
    def test_detectors_present(self):
        assert "wfc3_uvis" in settings.detector_defaults
        assert "wfc3_ir" in settings.detector_defaults
        assert "acs_wfc" in settings.detector_defaults
        assert "wfpc2_wfpc2" in settings.detector_defaults

    def test_detector_has_driz_bits_and_pixel_scale(self):
        for det in ("wfc3_uvis", "acs_wfc"):
            d = settings.detector_defaults[det]
            assert "driz_bits" in d
            assert "pixel_scale" in d
            assert "input_files" in d


class TestNamesAndLists:
    def test_names_list(self):
        assert "image" in settings.names
        assert "filter" in settings.names
        assert "instrument" in settings.names

    def test_final_names(self):
        assert "MJD" in settings.final_names
        assert "MAGNITUDE" in settings.final_names
        assert "MAGNITUDE_ERROR" in settings.final_names

    def test_acceptable_filters_non_empty(self):
        assert len(settings.acceptable_filters) > 0
        assert "F814W" in settings.acceptable_filters

    def test_pipeline_products_and_images(self):
        assert len(settings.pipeline_products) > 0
        assert len(settings.pipeline_images) > 0
