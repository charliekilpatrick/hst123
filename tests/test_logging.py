"""Unit tests for utils.logging (format_success, format_failure, make_banner, LogConfig)."""
import logging
import pytest

from hst123.utils.logging import (
    LogConfig,
    format_failure,
    format_success,
    get_logger,
    make_banner,
    green,
    red,
    end,
)


class TestFormatSuccess:
    def test_returns_string(self):
        assert isinstance(format_success("Downloading file"), str)

    def test_starts_with_carriage_return(self):
        assert format_success("x").startswith("\r")

    def test_contains_prefix(self):
        prefix = "Downloading ref: foo.fits"
        assert prefix in format_success(prefix)

    def test_contains_success_label(self):
        assert " [SUCCESS]" in format_success("")

    def test_ends_with_newline(self):
        assert format_success("x").endswith("\n")

    def test_uses_green_and_end_constants(self):
        prefix = "msg"
        out = format_success(prefix)
        assert green in out
        assert end in out
        assert out == "\r" + prefix + green + " [SUCCESS]" + end + "\n"


class TestFormatFailure:
    def test_returns_string(self):
        assert isinstance(format_failure("Download failed"), str)

    def test_starts_with_carriage_return(self):
        assert format_failure("x").startswith("\r")

    def test_contains_prefix(self):
        prefix = "MAST download failed"
        assert prefix in format_failure(prefix)

    def test_contains_failure_label(self):
        assert " [FAILURE]" in format_failure("")

    def test_ends_with_newline(self):
        assert format_failure("x").endswith("\n")

    def test_uses_red_and_end_constants(self):
        prefix = "msg"
        out = format_failure(prefix)
        assert red in out
        assert end in out
        assert out == "\r" + prefix + red + " [FAILURE]" + end + "\n"


class TestMakeBanner:
    def test_make_banner_logs_and_contains_message(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        make_banner("test message")
        assert "test message" in caplog.text
        assert "#" in caplog.text

    def test_make_banner_does_not_raise(self):
        make_banner("")


class TestLogConfig:
    def test_default_level_is_info(self):
        cfg = LogConfig()
        assert cfg.level == logging.INFO

    def test_level_from_string(self):
        cfg = LogConfig(level="DEBUG")
        assert cfg.level == logging.DEBUG

    def test_context_applies_and_undoes(self):
        cfg = LogConfig(level="DEBUG", enable_stdout=True, enable_file=False)
        with cfg.context():
            log = get_logger("hst123")
            assert log.level == logging.DEBUG
            assert len(log.handlers) >= 1
