"""Unit tests for common.Options."""
import pytest

from common import Options


class TestAddOptions:
    def test_returns_parser_with_positional_ra_dec(self):
        parser = Options.add_options()
        args = parser.parse_args(["12:30:00", "-45.0"])
        assert args.ra == "12:30:00"
        assert args.dec == "-45.0"

    def test_optional_flags_exist(self):
        parser = Options.add_options()
        args = parser.parse_args(["0", "0", "--download", "--make-clean"])
        assert args.download is True
        assert args.make_clean is True

    def test_work_dir_and_reference(self):
        parser = Options.add_options()
        args = parser.parse_args(["0", "0", "--work-dir", "/tmp", "--reference", "ref.fits"])
        assert args.work_dir == "/tmp"
        assert args.reference == "ref.fits"

    def test_tweakreg_options(self):
        parser = Options.add_options()
        args = parser.parse_args(
            ["0", "0", "--tweak-search", "2.0", "--tweak-min-obj", "15", "--tweak-thresh", "5.0"]
        )
        assert args.tweak_search == 2.0
        assert args.tweak_min_obj == 15
        assert args.tweak_thresh == 5.0

    def test_drizzle_options(self):
        parser = Options.add_options()
        args = parser.parse_args(["0", "0", "--drizzle-dim", "4000", "--wht-type", "IVM"])
        assert args.drizzle_dim == 4000
        assert args.wht_type == "IVM"

    def test_uses_provided_parser(self):
        import argparse
        p = argparse.ArgumentParser()
        out = Options.add_options(parser=p)
        assert out is p
        args = out.parse_args(["1", "2"])
        assert args.ra == "1"
        assert args.dec == "2"
