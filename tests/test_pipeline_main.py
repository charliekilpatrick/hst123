"""Tests for pipeline entry point (main) and module import."""
import os
import sys

import pytest


def test_pipeline_module_import():
    """_pipeline exposes hst123 class and main()."""
    import hst123._pipeline as mod
    assert hasattr(mod, "hst123")
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_main_version_exits_zero():
    """main() with --version exits with code 0 (argparse version action)."""
    try:
        import hst123 as hst123_module
    except Exception as e:
        pytest.skip(f"hst123 not importable: {e}")
    with pytest.raises(SystemExit) as exc_info:
        with _argv(["hst123", "0", "0", "--version"]):
            hst123_module.main()
    assert exc_info.value.code == 0


def test_main_help_exits_zero():
    """main() with --help exits (code 0 or None)."""
    try:
        import hst123 as hst123_module
    except Exception as e:
        pytest.skip(f"hst123 not importable: {e}")
    with pytest.raises(SystemExit) as exc_info:
        with _argv(["hst123", "0", "0", "--help"]):
            hst123_module.main()
    assert exc_info.value.code in (0, None)


class _argv:
    """Temporarily replace sys.argv."""

    def __init__(self, args):
        self.args = args
        self.saved = None

    def __enter__(self):
        self.saved = sys.argv
        sys.argv = self.args
        return self

    def __exit__(self, *exc):
        sys.argv = self.saved
        return False


def test_main_skips_mast_query_without_download_or_archive(tmp_path, monkeypatch):
    """
    ``main`` should not call :meth:`~hst123.hst123.get_productlist` when there is no
    ``--download`` and no archive copy (matches pipeline driver logic).
    """
    try:
        import hst123 as hst123_module
    except Exception as e:
        pytest.skip(f"hst123 not importable: {e}")

    calls = []

    def tracking(self, coord, search_radius):
        calls.append((coord, search_radius))
        return None

    monkeypatch.setattr(hst123_module.hst123, "get_productlist", tracking)

    wd = tmp_path / "work"
    wd.mkdir()
    (wd / "raw").mkdir()

    argv = [
        "hst123",
        "0",
        "0",
        "--work-dir",
        str(wd),
    ]
    with _argv(argv):
        hst123_module.main()

    assert calls == []
    assert os.path.isdir(str(wd / "raw"))


def test_main_calls_get_productlist_when_download_requested(tmp_path, monkeypatch):
    """With ``--download``, MAST product list is required (``get_productlist`` runs)."""
    try:
        import hst123 as hst123_module
    except Exception as e:
        pytest.skip(f"hst123 not importable: {e}")

    calls = []

    def tracking(self, coord, search_radius):
        calls.append(1)
        return None

    monkeypatch.setattr(hst123_module.hst123, "get_productlist", tracking)
    monkeypatch.setattr(
        hst123_module.hst123,
        "download_files",
        lambda self, *a, **k: None,
    )
    monkeypatch.setattr(
        hst123_module.hst123,
        "copy_raw_data",
        lambda self, *a, **k: None,
    )

    wd = tmp_path / "work_dl"
    wd.mkdir()
    (wd / "raw").mkdir()

    argv = [
        "hst123",
        "0",
        "0",
        "--work-dir",
        str(wd),
        "--download",
    ]
    with _argv(argv):
        hst123_module.main()

    assert len(calls) >= 1
