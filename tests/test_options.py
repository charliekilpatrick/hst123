"""Unit tests for utils.options."""
import sys
from types import SimpleNamespace

import pytest

from hst123.utils import options


class TestAddOptions:
    def test_returns_parser_with_positional_ra_dec(self):
        parser = options.add_options()
        args = parser.parse_args(["12:30:00", "-45.0"])
        assert args.ra == "12:30:00"
        assert args.dec == "-45.0"

    def test_optional_flags_exist(self):
        parser = options.add_options()
        args = parser.parse_args(["0", "0", "--download", "--cleanup"])
        assert args.download is True
        assert args.cleanup is True

    def test_work_dir_and_reference(self):
        parser = options.add_options()
        args = parser.parse_args(["0", "0", "--work-dir", "/tmp", "--reference", "ref.fits"])
        assert args.work_dir == "/tmp"
        assert args.reference == "ref.fits"

    def test_tweakreg_options(self):
        parser = options.add_options()
        args = parser.parse_args(
            ["0", "0", "--tweak-search", "2.0", "--tweak-min-obj", "15", "--tweak-thresh", "5.0"]
        )
        assert args.tweak_search == 2.0
        assert args.tweak_min_obj == 15
        assert args.tweak_thresh == 5.0

    def test_drizzle_options(self):
        parser = options.add_options()
        args = parser.parse_args(["0", "0", "--drizzle-dim", "4000", "--wht-type", "IVM"])
        assert args.drizzle_dim == 4000
        assert args.wht_type == "IVM"

    def test_max_cores_optional(self):
        parser = options.add_options()
        args = parser.parse_args(["0", "0"])
        assert args.max_cores is None
        args = parser.parse_args(["0", "0", "--max-cores", "8"])
        assert args.max_cores == 8

    def test_uses_provided_parser(self):
        import argparse
        p = argparse.ArgumentParser()
        out = options.add_options(parser=p)
        assert out is p
        args = out.parse_args(["1", "2"])
        assert args.ra == "1"
        assert args.dec == "2"

    def test_redo_flags_parse(self):
        parser = options.add_options()
        args = parser.parse_args(["0", "0", "--redo-astrometry", "--redo-astrodrizzle"])
        assert args.redo_astrometry is True
        assert args.redo_astrodrizzle is True
        assert args.redo is False


def test_want_redo_astrometry():
    assert options.want_redo_astrometry(SimpleNamespace()) is False
    assert options.want_redo_astrometry(SimpleNamespace(clobber=True)) is True
    assert options.want_redo_astrometry(SimpleNamespace(redo=True)) is True
    assert options.want_redo_astrometry(SimpleNamespace(redo_astrometry=True)) is True


def test_want_redo_astrodrizzle():
    assert options.want_redo_astrodrizzle(SimpleNamespace()) is False
    assert options.want_redo_astrodrizzle(SimpleNamespace(clobber=True)) is True
    assert options.want_redo_astrodrizzle(SimpleNamespace(redo=True)) is True
    assert options.want_redo_astrodrizzle(SimpleNamespace(redo_astrodrizzle=True)) is True


def test_want_redo_dolphot():
    assert options.want_redo_dolphot(SimpleNamespace()) is False
    assert options.want_redo_dolphot(SimpleNamespace(redo=True)) is True
    assert options.want_redo_dolphot(SimpleNamespace(redo_dolphot=True)) is True


def test_dolphot_catalog_already_present(tmp_path):
    base = tmp_path / "dp0000"
    col = tmp_path / "dp0000.columns"
    base.write_text("1 2 3\n")
    col.write_text("1. col\n")
    assert options.dolphot_catalog_already_present(
        {"base": str(base), "colfile": str(col)}
    )
    base.write_text("")
    assert not options.dolphot_catalog_already_present(
        {"base": str(base), "colfile": str(col)}
    )


def test_redo_dolphot_flag_parses():
    parser = options.add_options()
    args = parser.parse_args(["0", "0", "--redo-dolphot"])
    assert args.redo_dolphot is True
    assert args.redo is False


def test_handle_args_redo_sets_both_redo_flags(monkeypatch):
    """--redo implies redo_astrometry and redo_astrodrizzle (handle_args)."""
    import hst123 as _hst

    monkeypatch.setattr(sys, "argv", ["hst123", "0", "0", "--redo"])
    hst = _hst.hst123()
    opt = hst.handle_args(hst.add_options())
    assert opt.redo is True
    assert opt.redo_astrometry is True
    assert opt.redo_astrodrizzle is True
