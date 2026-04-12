"""Unit tests for pipeline MAST download and archive path helpers."""
from pathlib import Path
from unittest.mock import patch

import pytest
from astropy.table import Table

try:
    import hst123 as hst123_module
except Exception as e:
    hst123_module = None
    _hst123_import_error = e


def _require_hst123():
    if hst123_module is None:
        pytest.skip(f"hst123 package not importable: {_hst123_import_error}")


def test_download_files_empty_productlist_returns_false(hst123_instance):
    _require_hst123()
    hst = hst123_instance
    assert hst.download_files([], work_dir=None) is False


def test_download_files_skips_existing_files_no_mast_call(hst123_instance, tmp_path):
    _require_hst123()
    hst = hst123_instance
    dest = tmp_path / "fits"
    dest.mkdir()
    existing = dest / "already.fits"
    existing.write_bytes(b"FITS")

    t = Table(
        {
            "downloadFilename": ["already.fits"],
            "obsID": ["obs1"],
        }
    )
    with patch("hst123._pipeline.Observations.download_products") as mock_dl:
        ok = hst.download_files(t, dest=str(dest), work_dir=str(tmp_path))
    assert ok is True
    mock_dl.assert_not_called()


def test_mast_download_one_product_row_moves_staged_file(hst123_instance, tmp_path):
    _require_hst123()
    hst = hst123_instance
    staging = tmp_path / "staging.fits"
    staging.write_bytes(b"DATA")
    dest = tmp_path / "out.fits"
    prod = Table(
        {
            "downloadFilename": ["out.fits"],
            "obsID": ["obs1"],
        }
    )[0]
    item = (0, prod, str(dest), 1, str(tmp_path))
    with patch("hst123._pipeline.Observations.download_products") as mock_dl:
        mock_dl.return_value = Table({"Local Path": [str(staging)]})
        hst._mast_download_one_product_row(item)
    mock_dl.assert_called_once()
    assert dest.read_bytes() == b"DATA"
    assert not staging.exists()


def test_mast_download_one_product_row_logs_warning_on_failure(hst123_instance, tmp_path, caplog):
    _require_hst123()
    import logging

    hst = hst123_instance
    prod = Table(
        {
            "downloadFilename": ["missing.fits"],
            "obsID": ["obs1"],
        }
    )[0]
    dest = tmp_path / "missing.fits"
    item = (0, prod, str(dest), 1, str(tmp_path))
    caplog.set_level(logging.WARNING)
    with patch("hst123._pipeline.Observations.download_products", side_effect=RuntimeError("network")):
        hst._mast_download_one_product_row(item)
    assert "MAST fail" in caplog.text


def test_download_files_runs_one_worker_call_per_pending_file(
    hst123_instance, tmp_path
):
    _require_hst123()
    hst = hst123_instance
    setattr(hst.options["args"], "drizzle_num_cores", 6)

    t = Table(
        {
            "downloadFilename": ["a.fits", "b.fits", "c.fits"],
            "obsID": ["1", "2", "3"],
        }
    )
    dest = tmp_path / "out"
    dest.mkdir()

    with patch.object(hst, "_mast_download_one_product_row") as mock_one:
        with patch("hst123._pipeline.Observations.download_products"):
            hst.download_files(t, dest=str(dest), work_dir=str(tmp_path))

    assert mock_one.call_count == 3


def test_archive_path_for_product_acs_wfc(hst123_instance, tmp_path):
    _require_hst123()
    hst = hst123_instance
    product = {
        "productFilename": "prefix_suffix_chip.fits",
        "instrument_name": "ACS/WFC",
        "ra": 123.45,
    }
    root = str(tmp_path / "arch")
    path = hst._archive_path_for_product(product, root)
    assert path == str(Path(root) / "ACS" / "WFC" / "123" / "suffix_chip.fits")


def test_archive_path_for_product_wfpc2(hst123_instance, tmp_path):
    _require_hst123()
    hst = hst123_instance
    product = {
        "productFilename": "a_b_c.fits",
        "instrument_name": "WFPC2/PC",
        "ra": 10.2,
    }
    root = str(tmp_path / "arch")
    path = hst._archive_path_for_product(product, root)
    assert path == str(Path(root) / "WFPC2" / "WFPC2" / "10" / "b_c.fits")


def test_check_archive_true_when_file_present(hst123_instance, tmp_path):
    _require_hst123()
    hst = hst123_instance
    arch_root = tmp_path / "arch"
    product = {
        "productFilename": "foo_bar_name.fits",
        "instrument_name": "ACS/WFC",
        "ra": 200.0,
    }
    full = Path(hst._archive_path_for_product(product, str(arch_root)))
    full.parent.mkdir(parents=True)
    full.write_bytes(b"FITS")
    exists, got_path = hst.check_archive(product, archivedir=str(arch_root))
    assert exists is True
    assert got_path == str(full)


def test_check_archive_false_when_missing(hst123_instance, tmp_path):
    _require_hst123()
    hst = hst123_instance
    arch_root = tmp_path / "arch"
    product = {
        "productFilename": "foo_bar_name.fits",
        "instrument_name": "ACS/WFC",
        "ra": 200.0,
    }
    exists, got_path = hst.check_archive(product, archivedir=str(arch_root))
    assert exists is False
    assert got_path == str(Path(hst._archive_path_for_product(product, str(arch_root))))
