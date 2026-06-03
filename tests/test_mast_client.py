"""Unit tests for MAST client helpers."""
from unittest.mock import patch

import pytest

from hst123.utils.mast_client import (
    MAST_TRANSIENT_ERRORS,
    mast_call_with_retries,
    mast_extended_timeout,
)


def test_mast_transient_errors_includes_builtin_timeout():
    assert TimeoutError in MAST_TRANSIENT_ERRORS


def test_mast_call_with_retries_succeeds_after_transient_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("portal slow")
        return "ok"

    with patch("hst123.utils.mast_client.time.sleep"):
        assert mast_call_with_retries(flaky, retries=3, pause=0.01) == "ok"
    assert calls["n"] == 2


def test_mast_call_with_retries_raises_after_exhausted_retries():
    with patch("hst123.utils.mast_client.time.sleep"):
        with pytest.raises(TimeoutError):
            mast_call_with_retries(
                lambda: (_ for _ in ()).throw(TimeoutError("fail")),
                retries=2,
                pause=0.01,
            )


def test_mast_extended_timeout_restores_portal_timeout():
    from astroquery.mast import Observations
    from astroquery.mast import conf as mast_conf

    portal = Observations._portal_api_connection
    original_conf = mast_conf.timeout
    original_portal = portal.TIMEOUT
    try:
        with mast_extended_timeout(999):
            assert mast_conf.timeout == 999
            assert portal.TIMEOUT == 999
        assert mast_conf.timeout == original_conf
        assert portal.TIMEOUT == original_portal
    finally:
        mast_conf.timeout = original_conf
        portal.TIMEOUT = original_portal
