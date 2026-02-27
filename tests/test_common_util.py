"""Unit tests for common.Util."""
import pytest
from astropy.coordinates import SkyCoord

from hst123.common import Util


class TestIsNumber:
    def test_integer_string(self):
        assert Util.is_number("42") is True
        assert Util.is_number("-17") is True

    def test_float_string(self):
        assert Util.is_number("3.14") is True
        assert Util.is_number("1e-5") is True

    def test_non_number_string(self):
        assert Util.is_number("abc") is False
        assert Util.is_number("12:30:45") is False
        assert Util.is_number("") is False

    def test_numeric_types(self):
        assert Util.is_number(42) is True
        assert Util.is_number(3.14) is True


class TestParseCoord:
    def test_degree_input(self):
        coord = Util.parse_coord(180.0, -45.0)
        assert coord is not None
        assert isinstance(coord, SkyCoord)
        assert coord.ra.deg == pytest.approx(180.0)
        assert coord.dec.deg == pytest.approx(-45.0)

    def test_sexagesimal_input(self):
        coord = Util.parse_coord("12:00:00", "+00:00:00")
        assert coord is not None
        assert isinstance(coord, SkyCoord)
        assert coord.ra.hour == pytest.approx(12.0)
        assert coord.dec.deg == pytest.approx(0.0)

    def test_invalid_input_returns_none(self):
        result = Util.parse_coord("not", "valid")
        assert result is None

    def test_string_degrees(self):
        coord = Util.parse_coord("0", "0")
        assert coord is not None
        assert coord.ra.deg == pytest.approx(0.0)
        assert coord.dec.deg == pytest.approx(0.0)


class TestMakeBanner:
    def test_make_banner_prints_and_contains_message(self, capsys):
        Util.make_banner("test message")
        out, _ = capsys.readouterr()
        assert "test message" in out
        assert "#" in out

    def test_make_banner_does_not_raise(self):
        Util.make_banner("")
