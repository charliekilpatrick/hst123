"""Tests for hst123.utils.paths."""
from pathlib import Path

from hst123.utils.paths import (
    normalize_fits_path,
    normalize_work_and_raw_dirs,
    pipeline_chip_output_dir,
    pipeline_workspace_dir,
)


def test_pipeline_workspace_dir_none_or_empty():
    assert pipeline_workspace_dir(None) is None
    assert pipeline_workspace_dir("") is None


def test_pipeline_workspace_dir_joins_workspace_subdir(tmp_path):
    wd = str(tmp_path / "run")
    assert pipeline_workspace_dir(wd) == str(Path(wd).resolve() / "workspace")


def test_pipeline_chip_output_dir_none_or_empty():
    assert pipeline_chip_output_dir(None) is None
    assert pipeline_chip_output_dir("") is None


def test_pipeline_chip_output_dir_is_absolute_base(tmp_path):
    wd = str(tmp_path / "run")
    assert pipeline_chip_output_dir(wd) == str(Path(wd).resolve())


def test_normalize_fits_path_empty_unchanged():
    assert normalize_fits_path("") == ""


def test_normalize_fits_path_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rel = "subdir/foo.fits"
    Path("subdir").mkdir()
    out = normalize_fits_path(rel)
    assert Path(out).is_absolute()
    assert out.endswith("subdir/foo.fits")


def test_normalize_work_and_raw_default_raw_under_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w, r = normalize_work_and_raw_dirs(None, None)
    assert w == str(tmp_path.resolve())
    assert r == str((tmp_path / "raw").resolve())


def test_normalize_work_and_raw_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w, r = normalize_work_and_raw_dirs("mywork", None)
    assert w == str((tmp_path / "mywork").resolve())
    assert r == str((tmp_path / "mywork" / "raw").resolve())


def test_normalize_preserves_absolute_raw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ext = tmp_path / "elsewhere"
    ext.mkdir()
    w, r = normalize_work_and_raw_dirs("wd", str(ext))
    assert "wd" in w
    assert r == str(ext.resolve())
