"""Unit tests for common.Constants."""
import pytest

from hst123.common import Constants


class TestConstants:
    def test_green_red_end_defined(self):
        assert hasattr(Constants, "green")
        assert hasattr(Constants, "red")
        assert hasattr(Constants, "end")

    def test_are_strings(self):
        assert isinstance(Constants.green, str)
        assert isinstance(Constants.red, str)
        assert isinstance(Constants.end, str)

    def test_ansi_like(self):
        assert "\033" in Constants.green
        assert "\033" in Constants.red
        assert "\033" in Constants.end
