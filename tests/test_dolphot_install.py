"""Tests for DOLPHOT download and install (hst123.dolphot_install)."""
import tarfile
from pathlib import Path

import pytest

from hst123.dolphot_install import (
    DOLPHOT_BASE_URL,
    ONE_PSF_PER_INSTRUMENT,
    PSF_FILES,
    SOURCES_BASE,
    SOURCES_MODULES,
    _url_for,
    download_file,
    extract_tar,
    install_psfs,
)


def test_url_for():
    """URLs are built correctly from base and filename."""
    assert _url_for("dolphot3.0.tar.gz") == DOLPHOT_BASE_URL + "dolphot3.0.tar.gz"
    assert _url_for("ACS_WFC_F435W.tar.gz") == DOLPHOT_BASE_URL + "ACS_WFC_F435W.tar.gz"


def test_sources_constants():
    """Base and module tarball names are set for 3.0."""
    assert SOURCES_BASE == "dolphot3.0.tar.gz"
    assert "dolphot3.0.ACS.tar.gz" in SOURCES_MODULES
    assert "dolphot3.0.WFC3.tar.gz" in SOURCES_MODULES
    assert "dolphot3.0.WFPC2.tar.gz" in SOURCES_MODULES


def test_one_psf_per_instrument():
    """One PSF filename per instrument is defined."""
    assert ONE_PSF_PER_INSTRUMENT["ACS"] == "ACS_WFC_F435W.tar.gz"
    assert ONE_PSF_PER_INSTRUMENT["WFC3"] == "WFC3_UVIS_F555W.tar.gz"
    assert ONE_PSF_PER_INSTRUMENT["WFPC2"] == "WFPC2_F555W.tar.gz"


def test_psf_files_lists():
    """PSF_FILES has non-empty lists for ACS, WFC3, WFPC2."""
    for inv in ("ACS", "WFC3", "WFPC2"):
        assert inv in PSF_FILES
        assert len(PSF_FILES[inv]) > 0
        assert ONE_PSF_PER_INSTRUMENT[inv] in PSF_FILES[inv]


def test_extract_tar_strip_top_level(tmp_path):
    """extract_tar strips a single top-level directory when requested."""
    tar_path = tmp_path / "test.tar.gz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tar:
        # Simulate dolphot3.0/Makefile layout
        import io
        buf = io.BytesIO(b"makefile content")
        info = tarfile.TarInfo(name="dolphot3.0/Makefile")
        info.size = len(buf.getvalue())
        tar.addfile(info, buf)
    extract_tar(tar_path, dest, strip_top_level=True)
    assert (dest / "Makefile").read_text() == "makefile content"


def test_extract_tar_no_strip(tmp_path):
    """extract_tar preserves paths when not stripping."""
    tar_path = tmp_path / "test.tar.gz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tar:
        import io
        buf = io.BytesIO(b"content")
        info = tarfile.TarInfo(name="top/file.txt")
        info.size = len(buf.getvalue())
        tar.addfile(info, buf)
    extract_tar(tar_path, dest, strip_top_level=False)
    assert (dest / "top" / "file.txt").read_text() == "content"


def test_install_psfs_invalid_instrument(tmp_path):
    """install_psfs raises for unknown instrument."""
    with pytest.raises(ValueError, match="Unknown instrument"):
        install_psfs(tmp_path, instruments=["INVALID"], one_per_instrument=True)


# --- Network tests: download one PSF per instrument ---


@pytest.mark.network
def test_download_single_psf_acs(tmp_path):
    """Download a single ACS PSF file and extract into target dir."""
    install_psfs(tmp_path, instruments=["ACS"], one_per_instrument=True, timeout=30)
    # After extract, tarball is removed; we expect some extracted content
    assert list(tmp_path.iterdir())  # directory not empty


@pytest.mark.network
def test_download_single_psf_wfc3(tmp_path):
    """Download a single WFC3 PSF file and extract into target dir."""
    install_psfs(tmp_path, instruments=["WFC3"], one_per_instrument=True, timeout=30)
    assert list(tmp_path.iterdir())


@pytest.mark.network
def test_download_single_psf_wfpc2(tmp_path):
    """Download a single WFPC2 PSF file and extract into target dir."""
    install_psfs(tmp_path, instruments=["WFPC2"], one_per_instrument=True, timeout=30)
    assert list(tmp_path.iterdir())
