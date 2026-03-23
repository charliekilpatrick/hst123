"""Tests for :mod:`hst123.utils.progress_log`."""
import logging
import time

import pytest

from hst123.utils.progress_log import (
    LoggedProgress,
    _bar,
    _format_duration,
    calcsky_progress_enabled,
    progress_log_enabled,
)


def test_format_duration():
    assert _format_duration(45.2) == "45s"
    assert _format_duration(125) == "2m5s"
    assert _format_duration(3700) == "1h1m"
    assert _format_duration(float("nan")) == "…"


def test_bar():
    assert _bar(0, 10) == "----------"
    assert _bar(1.0, 10) == "##########"
    assert len(_bar(0.5, 28)) == 28


def test_logged_progress_emits_start_and_updates(caplog):
    caplog.set_level(logging.INFO)
    log = logging.getLogger("test_prog")
    p = LoggedProgress(log, "step-a", 100, min_interval=0.0, min_fraction_delta=0.0)
    p.start()
    p.update(50)
    p.complete()
    messages = [r.message for r in caplog.records]
    assert any("step-a" in m and "0.0%" in m for m in messages)
    assert any("50.0%" in m for m in messages)
    assert any("100.0%" in m for m in messages)


def test_logged_progress_throttles(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    log = logging.getLogger("test_prog2")
    p = LoggedProgress(log, "step-b", 10, min_interval=10.0, min_fraction_delta=0.5)
    p.start()
    t0 = time.monotonic()

    def fake_monotonic():
        return t0

    monkeypatch.setattr("hst123.utils.progress_log.time.monotonic", fake_monotonic)
    p.update(1)
    p.update(2)
    p.update(3)
    # Same fake time → throttled (no new lines except start)
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) <= 2


def test_context_manager_completes(caplog):
    caplog.set_level(logging.INFO)
    log = logging.getLogger("test_prog3")
    with LoggedProgress(log, "step-c", 5, min_interval=0.0, min_fraction_delta=0.0):
        pass
    assert any("100.0%" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "env,expected",
    [
        ("", True),
        ("0", False),
        ("false", False),
    ],
)
def test_progress_log_enabled(monkeypatch, env, expected):
    monkeypatch.delenv("HST123_PROGRESS_LOG", raising=False)
    if env != "":
        monkeypatch.setenv("HST123_PROGRESS_LOG", env)
    assert progress_log_enabled() is expected


def test_calcsky_progress_respects_disable(monkeypatch):
    monkeypatch.delenv("HST123_CALCSKY_PROGRESS", raising=False)
    monkeypatch.delenv("HST123_PROGRESS_LOG", raising=False)
    assert calcsky_progress_enabled() is True
    monkeypatch.setenv("HST123_CALCSKY_PROGRESS", "0")
    assert calcsky_progress_enabled() is False
