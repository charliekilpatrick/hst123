"""Unit tests for utils.logging (format_success, format_failure, make_banner, LogConfig)."""
import logging
import os
import subprocess
import sys
from argparse import Namespace

import pytest

from astropy.io import fits
import numpy as np

from hst123.utils.logging import (
    ROOT_LOGGER,
    LogConfig,
    attach_work_dir_log_file,
    ensure_cli_logging_configured,
    format_failure,
    format_hdu_list_summary,
    format_success,
    get_logger,
    ingest_text_file_to_logger,
    log_pipeline_configuration,
    make_banner,
    run_external_command,
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


def test_format_hdu_list_summary():
    data = np.zeros((2, 2), dtype=np.float32)
    hl = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=data, name="SCI"),
        ]
    )
    s = format_hdu_list_summary(hl)
    assert "2 ext" in s
    assert "PRIMARY:1" in s
    assert "SCI:1" in s


class TestMakeBanner:
    def test_make_banner_logs_and_contains_message(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        make_banner("test message")
        assert "test message" in caplog.text
        assert "—" in caplog.text

    def test_make_banner_does_not_raise(self):
        make_banner("")


def test_log_pipeline_configuration_logs_version_and_flags(tmp_path, caplog, monkeypatch):
    """Startup summary includes version, coordinate, and key booleans."""
    monkeypatch.chdir(tmp_path)
    opt = Namespace(
        work_dir="mywork",
        raw_dir="./r",
        archive=None,
        download=True,
        clobber=False,
        skip_copy=False,
        no_clear_downloads=False,
        token=None,
        before=None,
        after="2020-01-01",
        only_filter=None,
        only_wide=False,
        keep_short=False,
        align_with="tweakreg",
        skip_tweakreg=False,
        hierarchical=False,
        drizzle_all=True,
        redrizzle=False,
        drizzle_dim=5200,
        by_visit=False,
        run_dolphot=True,
        scrape_dolphot=False,
        dolphot="dp",
        dolphot_lim=3.5,
        scrape_radius=None,
        scrape_all=False,
        do_fake=False,
        cleanup=False,
        keep_drizzle_artifacts=False,
        keep_objfile=False,
    )
    caplog.set_level(logging.INFO)
    log = get_logger("hst123")
    log_pipeline_configuration(
        log,
        opt,
        version="9.9.9-test",
        coord_hmsdms="12 00 00.0000 +30 00 00.000",
    )
    text = caplog.text
    assert "9.9.9-test" in text
    assert "12 00 00" in text
    assert "dl=True" in text
    assert "dp run=True" in text
    assert "after=2020-01-01" in text


def test_run_external_command_echo(caplog, tmp_path):
    """External stdout is merged and logged; tee file matches."""
    caplog.set_level(logging.INFO)
    tee = tmp_path / "out.txt"
    run_external_command(
        [sys.executable, "-c", "print('hello world')"],
        tee_path=str(tee),
        log=get_logger("hst123.external"),
    )
    assert "hello world" in caplog.text
    assert tee.read_text(encoding="utf-8").strip() == "hello world"


def test_run_external_command_nonzero_raises():
    with pytest.raises(subprocess.CalledProcessError):
        run_external_command(
            [sys.executable, "-c", "raise SystemExit(7)"],
            check=True,
        )


def test_ensure_cli_logging_configured_idempotent():
    """First call attaches a handler; second call does not duplicate."""
    log = logging.getLogger(ROOT_LOGGER)
    before = list(log.handlers)
    for h in before:
        log.removeHandler(h)
        h.close()
    try:
        ensure_cli_logging_configured(level=logging.INFO)
        n1 = len(log.handlers)
        assert n1 >= 1
        ensure_cli_logging_configured(level=logging.INFO)
        assert len(log.handlers) == n1
    finally:
        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()


def test_attach_work_dir_log_file_creates_logs_and_mirrors(tmp_path):
    """Work dir gets logs/ and a unique file receives hst123 logger records."""
    import hst123.utils.logging as logmod

    logmod._WORK_DIR_LOG_HANDLER = None  # noqa: SLF001
    wd = tmp_path / "w"
    wd.mkdir()
    log = logging.getLogger(ROOT_LOGGER)
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    try:
        ensure_cli_logging_configured(level=logging.INFO)
        path = attach_work_dir_log_file(str(wd), process_name="test_proc")
        assert path is not None
        assert path.endswith(".log")
        assert (wd / "logs").is_dir()
        assert os.path.basename(path) in [p.name for p in (wd / "logs").iterdir()]
        get_logger("hst123.test").info("hello file mirror")
        for h in log.handlers:
            if getattr(h, "_hst123_log_path", None) == path:
                h.flush()
                break
        text = open(path, encoding="utf-8").read()
        assert "hello file mirror" in text
        assert "Session log (copy of console)" in text
        n_handlers = len(log.handlers)
        assert attach_work_dir_log_file(str(wd)) == path
        assert len(log.handlers) == n_handlers
    finally:
        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()
        logmod._WORK_DIR_LOG_HANDLER = None  # noqa: SLF001


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


class TestIngestTextFileToLogger:
    def test_env_summary_compact(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setenv("HST123_REPLAY_SUBLOGS", "0")
        caplog.set_level(logging.INFO)
        p = tmp_path / "t.log"
        p.write_text("line one\n\nline two\n", encoding="utf-8")
        lg = get_logger("hst123.test_ingest_env_sum")
        n = ingest_text_file_to_logger(p, lg, log_tag="testtag")
        assert n == 2
        text = " ".join(r.message for r in caplog.records)
        assert "2 lines" in text and "[testtag]" in text
        assert "REPLAY_SUBLOGS" in text or "compact" in text
        assert "line one" not in text

    def test_default_replay_full_via_env(self, tmp_path, caplog, monkeypatch):
        monkeypatch.delenv("HST123_REPLAY_SUBLOGS", raising=False)
        caplog.set_level(logging.INFO)
        p = tmp_path / "t.log"
        p.write_text("line one\n\nline two\n", encoding="utf-8")
        lg = get_logger("hst123.test_ingest_def")
        n = ingest_text_file_to_logger(p, lg, log_tag="testtag")
        assert n == 2
        text = " ".join(r.message for r in caplog.records)
        assert "line one" in text and "line two" in text
        assert "begin" in text and "end" in text

    def test_replay_full_explicit_false(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        p = tmp_path / "t.log"
        p.write_text("line one\n\nline two\n", encoding="utf-8")
        lg = get_logger("hst123.test_ingest_sum")
        n = ingest_text_file_to_logger(p, lg, log_tag="testtag", replay_full=False)
        assert n == 2
        text = " ".join(r.message for r in caplog.records)
        assert "2 lines" in text and "[testtag]" in text
        assert "line one" not in text

    def test_replay_full_lines_and_markers(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        p = tmp_path / "t.log"
        p.write_text("line one\n\nline two\n", encoding="utf-8")
        lg = get_logger("hst123.test_ingest")
        n = ingest_text_file_to_logger(
            p, lg, log_tag="testtag", replay_full=True
        )
        assert n == 2
        text = " ".join(r.message for r in caplog.records)
        assert "begin" in text and "end" in text
        assert "line one" in text and "line two" in text
        assert "[testtag]" in text

    def test_compact_ws_no_markers(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        p = tmp_path / "t.log"
        p.write_text("  lots   of   space  \n", encoding="utf-8")
        lg = get_logger("hst123.test_ingest_compact")
        n = ingest_text_file_to_logger(
            p,
            lg,
            log_tag="ad",
            replay_full=True,
            begin_end_markers=False,
            compact_ws=True,
        )
        assert n == 1
        assert "begin" not in caplog.text and "end" not in caplog.text
        assert "lots of space" in caplog.text

    def test_missing_ok(self, tmp_path, caplog):
        caplog.set_level(logging.INFO)
        lg = get_logger("hst123.test_ingest2")
        n = ingest_text_file_to_logger(
            tmp_path / "nope.log", lg, missing_ok=True
        )
        assert n == 0
        assert not caplog.records
