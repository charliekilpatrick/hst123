"""Tests for hst123.utils.reference_download (URL building, local copy, mocked HTTP)."""
from unittest.mock import MagicMock, patch

import pytest

from hst123.utils.reference_download import (
    build_calibration_reference_urls,
    cdbs_env_to_https_subdir,
    download_calibration_reference,
    fetch_calibration_reference,
    ref_prefix_for_header,
)


@pytest.mark.parametrize(
    "ref_env, expected",
    [
        ("jref.old", "jref"),
        ("iref.old", "iref"),
        ("uref", "uref"),
    ],
)
def test_cdbs_env_to_https_subdir(ref_env, expected):
    assert cdbs_env_to_https_subdir(ref_env) == expected


def test_ref_prefix_matches_header_convention():
    assert ref_prefix_for_header("jref.old") == "jref"
    assert ref_prefix_for_header("jref.old") + "$" == "jref$"


def test_build_calibration_reference_urls_order_and_paths():
    gd = {
        "crds": "https://hst-crds.stsci.edu/unchecked_get/references/hst/",
        "cdbs_https": "https://ssb.stsci.edu/cdbs/",
        "cdbs": "ftp://ftp.stsci.edu/cdbs/",
    }
    urls = build_calibration_reference_urls(gd, "jref.old", "4bb1536mj_idc.fits")
    assert len(urls) == 3
    assert urls[0].endswith("/references/hst/4bb1536mj_idc.fits")
    assert "ssb.stsci.edu/cdbs/jref/4bb1536mj_idc.fits" in urls[1]
    assert urls[2].endswith("/cdbs/jref.old/4bb1536mj_idc.fits")


def test_fetch_uses_local_jref_without_network(monkeypatch, tmp_path):
    jref_dir = tmp_path / "jref"
    jref_dir.mkdir()
    ref_name = "local_idc.fits"
    (jref_dir / ref_name).write_bytes(b"SIMPLE")

    monkeypatch.setenv("jref", str(jref_dir))
    dest = tmp_path / "out.fits"
    gd = {
        "crds": "https://example.invalid/crds/",
        "cdbs_https": "https://example.invalid/cdbs/",
        "cdbs": "ftp://example.invalid/cdbs/",
    }
    ok, err = fetch_calibration_reference(gd, "jref.old", ref_name, str(dest))
    assert ok
    assert err is None
    assert dest.read_bytes() == b"SIMPLE"


def test_download_calibration_reference_http_mock(tmp_path):
    dest = tmp_path / "downloaded.fits"

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_content = MagicMock(return_value=[b"FITS", b"DATA"])

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=None)

    mock_sess = MagicMock()
    mock_sess.get = MagicMock(return_value=mock_cm)

    with patch(
        "hst123.utils.reference_download._requests_session", return_value=mock_sess
    ):
        ok, err = download_calibration_reference(
            ["https://ssb.stsci.edu/cdbs/jref/x.fits"],
            str(dest),
        )

    assert ok
    assert err is None
    assert dest.read_bytes() == b"FITSDATA"
    mock_sess.get.assert_called_once()
