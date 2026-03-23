"""Tests for hst123.utils.paths."""
from hst123.utils.paths import normalize_work_and_raw_dirs


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
