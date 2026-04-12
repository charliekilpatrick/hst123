"""Direct unit tests for :class:`hst123.hst123` methods used by :func:`hst123.main`."""
import pytest
from astropy.table import Table

try:
    import hst123 as _hst
except Exception as e:
    _hst = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _require_hst123():
    if _hst is None:
        pytest.skip(f"hst123 not importable: {_IMPORT_ERR}")


class TestOrganizeReductionTables:
    """``main`` calls :meth:`~hst123.hst123.organize_reduction_tables` after :meth:`input_list`."""

    def test_byvisit_false_returns_single_table(self):
        _require_hst123()
        hst = _hst.hst123()
        t = Table(
            {
                "visit": [1, 1, 2],
                "image": ["a.fits", "b.fits", "c.fits"],
            }
        )
        out = hst.organize_reduction_tables(t, byvisit=False)
        assert len(out) == 1
        assert len(out[0]) == 3

    def test_byvisit_true_splits_by_visit(self):
        _require_hst123()
        hst = _hst.hst123()
        t = Table(
            {
                "visit": [1, 1, 2],
                "image": ["a.fits", "b.fits", "c.fits"],
            }
        )
        out = hst.organize_reduction_tables(t, byvisit=True)
        assert len(out) == 2
        lengths = sorted(len(x) for x in out)
        assert lengths == [1, 2]


class TestHandleReference:
    """``main`` calls :meth:`~hst123.hst123.handle_reference` per visit table."""

    def test_uses_existing_file_path(self, hst123_instance, minimal_fits_file, monkeypatch):
        monkeypatch.setattr(hst123_instance, "sanitize_reference", lambda ref: None)
        obstable = Table({"visit": [1]})
        out = hst123_instance.handle_reference(obstable, minimal_fits_file)
        assert out == minimal_fits_file

    def test_generates_via_pick_reference_when_no_ref(self, hst123_instance, minimal_fits_file, monkeypatch):
        def pick(obst):
            return minimal_fits_file

        monkeypatch.setattr(hst123_instance, "pick_reference", pick)
        monkeypatch.setattr(hst123_instance, "sanitize_reference", lambda ref: None)
        obstable = Table({"visit": [1]})
        out = hst123_instance.handle_reference(obstable, None)
        assert out == minimal_fits_file

    def test_returns_none_when_missing(self, hst123_instance, monkeypatch):
        monkeypatch.setattr(hst123_instance, "pick_reference", lambda obst: None)
        obstable = Table({"visit": [1]})
        assert hst123_instance.handle_reference(obstable, None) is None


class TestInputListEmpty:
    """``main`` calls :meth:`input_list` twice; empty inputs yield no table."""

    def test_empty_image_list_returns_none(self, hst123_instance):
        assert hst123_instance.input_list([], show=False, save=False) is None


class TestRunDolphotDelegate:
    """Pipeline exposes :meth:`run_dolphot` used when executing DOLPHOT."""

    def test_run_dolphot_delegates_to_primitive(self, hst123_instance, monkeypatch):
        called = []

        def fake():
            called.append(True)

        monkeypatch.setattr(hst123_instance._dolphot, "run_dolphot", fake)
        hst123_instance.run_dolphot()
        assert called == [True]


class TestPrepareDolphotDelegate:
    """Pipeline exposes :meth:`prepare_dolphot` for compatibility."""

    def test_prepare_dolphot_delegates(self, hst123_instance, monkeypatch):
        def fake(image):
            return [image.replace(".fits", ".chip1.fits")]

        monkeypatch.setattr(hst123_instance._dolphot, "prepare_dolphot", fake)
        assert hst123_instance.prepare_dolphot("x.fits") == ["x.chip1.fits"]
