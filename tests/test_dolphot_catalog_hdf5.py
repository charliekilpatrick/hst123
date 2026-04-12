"""Tests for DOLPHOT catalog parsing and HDF5 export (``hst123.utils.dolphot_catalog_hdf5``)."""
import json
from pathlib import Path

import numpy as np
import pytest

from hst123.utils.dolphot_catalog_hdf5 import (
    find_column_index_0based,
    load_dolphot_catalog_array,
    merge_image_metadata,
    parse_column_index_and_description,
    parse_dolphot_columns_file,
    parse_dolphot_data_file,
    parse_dolphot_info_file,
    parse_dolphot_param_file,
    parse_dolphot_warnings_file,
    unique_hdf5_column_names,
    write_dolphot_catalog_hdf5,
)


def test_parse_column_index_and_description():
    assert parse_column_index_and_description("3. Object X position on ref") == (
        3,
        "Object X position on ref",
    )
    assert parse_column_index_and_description("10. Crowding") == (10, "Crowding")
    assert parse_column_index_and_description("bad") is None


def test_find_column_index_matches_scrape_semantics(tmp_path):
    """Match :meth:`get_dolphot_column` behavior for empty image (first Object X)."""
    col = tmp_path / "x.columns"
    col.write_text(
        "1. Extension (zero for base image)\n"
        "2. Chip (for three-dimensional FITS image)\n"
        "3. Object X position on reference image\n"
        "4. Object Y position on reference image\n",
        encoding="utf-8",
    )
    cols = parse_dolphot_columns_file(col)
    assert find_column_index_0based(cols, "Object X", "") == 2
    assert find_column_index_0based(cols, "Object Y", "") == 3


def test_write_dolphot_catalog_hdf5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    base = tmp_path / "dp0000"
    col = tmp_path / "dp0000.columns"
    col.write_text(
        "1. Extension (zero for base image)\n"
        "2. Chip (for three-dimensional FITS image)\n"
        "3. Object X position\n"
        "4. Measured counts, /tmp/foo.chip2 (ACS_F555W, 100.0 sec)\n",
        encoding="utf-8",
    )
    # 2 rows, 4 columns
    cat = np.array(
        [
            [0, 1, 100.0, 50.0],
            [0, 1, 200.0, 60.0],
        ]
    )
    np.savetxt(base, cat)
    (tmp_path / "dp0000.param").write_text(
        "Nimg = 1\n"
        "FitSky = 2\n"
        "img0000_file = /ref.fits\n"
        "img0001_file = /tmp/foo.chip2\n",
        encoding="utf-8",
    )
    (tmp_path / "dp0000.info").write_text(
        "1 sets of output data\n"
        "/tmp/foo.chip2\n"
        "  59893.0\n"
        "EXTENSION 0 CHIP 1\n"
        "Limits\n"
        " 0 100 0 100\n"
        "* image 1: F555W 1 100.000000\n"
        "Alignment\n"
        " 1 2 3 4 5\n"
        "Aperture corrections\n"
        " 0.05\n",
        encoding="utf-8",
    )
    (tmp_path / "dp0000.data").write_text(
        "EXTENSION 0 CHIP 1\n"
        "WCS image 1: 1 2 3 4\n"
        "Align: 10\n"
        "Align image 1: 5 5 1 2 3 4 5 6\n"
        "PSF image 1: 10 0.1\n"
        "Apcor image 1: 1 2 3\n",
        encoding="utf-8",
    )
    (tmp_path / "dp0000.warnings").write_text(
        "Global warning line\n"
        "Scatter for /tmp/foo.chip2 (ACS_F555W)\n",
        encoding="utf-8",
    )

    out = tmp_path / "out.h5"
    write_dolphot_catalog_hdf5(out, base, compression=False)

    from astropy.table import Table

    t = Table.read(str(out), format="hdf5", path="photometry")
    assert len(t) == 2
    assert len(t.colnames) == 4

    with h5py.File(out, "r") as hf:
        assert "metadata" in hf
        assert "raw" in hf["metadata"]
        meta = json.loads(hf["photometry"].attrs["dolphot_column_json"])
        assert len(meta) == 4
        merged = json.loads(hf["photometry"].attrs["dolphot_merged_metadata_json"])
        assert "images" in merged


def test_write_dolphot_catalog_hdf5_embeds_dolphot_directory_manifest(tmp_path):
    """Pipeline writes HDF5 with a manifest of every file under ``<work>/dolphot/``."""
    h5py = pytest.importorskip("h5py")
    base = tmp_path / "dp0000"
    col = tmp_path / "dp0000.columns"
    col.write_text(
        "1. Extension (zero for base image)\n"
        "2. Chip (for three-dimensional FITS image)\n"
        "3. Object X position\n"
        "4. Measured counts, /tmp/foo.chip2 (ACS_F555W, 100.0 sec)\n",
        encoding="utf-8",
    )
    np.savetxt(
        base,
        np.array([[0, 1, 100.0, 50.0], [0, 1, 101.0, 51.0]]),
    )
    (tmp_path / "dp0000.param").write_text("Nimg = 1\n", encoding="utf-8")
    (tmp_path / "dp0000.info").write_text("1 sets of output data\n", encoding="utf-8")
    (tmp_path / "dp0000.data").write_text("WCS image 1: 1\n", encoding="utf-8")
    (tmp_path / "dp0000.warnings").write_text("", encoding="utf-8")
    (tmp_path / "extra_note.txt").write_text("auxiliary file\n", encoding="utf-8")

    out = tmp_path / "bundle.h5"
    write_dolphot_catalog_hdf5(
        out,
        base,
        compression=False,
        dolphot_dir=tmp_path,
    )

    with h5py.File(out, "r") as hf:
        dg = hf["metadata"]["dolphot_directory"]
        man = json.loads(dg["manifest_json"][0])
        rels = {e["relpath"] for e in man}
        assert "extra_note.txt" in rels
        assert "dp0000.columns" in rels
        assert dg.attrs["n_files"] == len(man)
        assert not bool(dg.attrs["text_embed"])
        assert "text_embed" not in dg


def test_vet_against_sample_dp_catalog_if_present(tmp_path):
    """Optional: vet parsers against local DOLPHOT products if ``test_data/dolphot/dp0000`` exists."""
    sample_dp = Path(__file__).resolve().parents[1] / "test_data" / "dolphot" / "dp0000"
    colp = Path(str(sample_dp) + ".columns")
    if not colp.is_file():
        pytest.skip("sample dp0000 not at expected path")
    cols = parse_dolphot_columns_file(colp)
    names = unique_hdf5_column_names(cols)
    assert len(cols) == len(names) == 90
    cat = load_dolphot_catalog_array(sample_dp)
    assert cat.shape == (30365, 90)
    param = parse_dolphot_param_file(Path(str(sample_dp) + ".param"))
    assert param.get("Nimg") == "4"
    info = parse_dolphot_info_file(Path(str(sample_dp) + ".info"))
    assert len(info.get("paths_mjd", [])) == 4
    data = parse_dolphot_data_file(Path(str(sample_dp) + ".data"))
    assert len(data.get("wcs", [])) == 4
    warn = parse_dolphot_warnings_file(Path(str(sample_dp) + ".warnings"))
    assert len(warn["lines"]) >= 1
    merged = merge_image_metadata(
        param,
        info,
        data,
        warn,
    )
    assert "img0001" in merged["images"]
    pytest.importorskip("h5py")
    out = tmp_path / "vet_dp0000.h5"
    write_dolphot_catalog_hdf5(out, sample_dp, compression=True)
    assert out.is_file()


def test_merge_image_metadata_structure():
    param = {
        "Nimg": "2",
        "img0001_file": "/data/a.chip2",
        "img0001_RAper": "2",
        "img0002_file": "/data/b.chip2",
    }
    info = {
        "paths_mjd": [
            {"path": "/data/a.chip2", "mjd": 50000.0},
            {"path": "/data/b.chip2", "mjd": 50001.0},
        ],
        "image_filter_exptime": [
            {"image_index": 1, "filter": "F814W", "chip": "1", "exptime": 390.0},
        ],
    }
    data = {
        "wcs": [{"image_index": 1, "values": [1.0, 2.0]}],
        "align_images": [{"image_index": 1, "n1": 10, "n2": 10, "values": []}],
        "psf": [{"image_index": 1, "values": [1.0, 0.1]}],
        "apcor": [{"image_index": 1, "values": [1.0]}],
    }
    warn = {
        "by_image_path": {"/data/a.chip2": ["warn about a"]},
        "global_lines": ["global"],
    }
    m = merge_image_metadata(param, info, data, warn)
    assert m["images"]["img0001"]["param"]["file"] == "/data/a.chip2"
    assert m["images"]["img0001"]["mjd"] == 50000.0
    assert "warnings" in m["images"]["img0001"]
