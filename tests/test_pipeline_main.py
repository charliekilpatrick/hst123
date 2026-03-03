"""Tests for pipeline entry point (main) and module import."""
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
