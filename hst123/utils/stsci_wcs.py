"""
STScI WCS stack bundled inside hst123 (``hst123.utils.stwcs``).

hst123 does **not** import the PyPI ``stwcs`` distribution. Submodules are loaded
with :func:`importlib.import_module` under ``hst123.utils.stwcs``. DrizzlePac may
still import ``stwcs`` from the environment for its own use.
"""

from __future__ import annotations

import importlib
from typing import Any, Type

_STWCS_PKG = "hst123.utils.stwcs"

_updatewcs_submod: Any | None = None
_altwcs_mod: Any | None = None
_hstwcs_cls: Type[Any] | None = None


def _updatewcs_submodule() -> Any:
    """Return the cached ``updatewcs`` submodule."""
    global _updatewcs_submod
    if _updatewcs_submod is None:
        _updatewcs_submod = importlib.import_module(f"{_STWCS_PKG}.updatewcs")
    return _updatewcs_submod


def run_updatewcs(image: Any, *, use_db: bool = True, **kwargs: Any) -> Any:
    """
    Run STScI ``updatewcs.updatewcs`` on a calibrated FITS path or HDU list.

    Parameters
    ----------
    image : str or `astropy.io.fits.HDUList`
        File to update in place, or an open HDU list.
    use_db : bool, optional
        Whether to query AstrometryDB when available.
    **kwargs
        Extra keyword arguments forwarded to ``updatewcs``.

    Returns
    -------
    object
        Return value from upstream ``updatewcs`` (often file names).
    """
    mod = _updatewcs_submodule()
    return mod.updatewcs(image, use_db=use_db, **kwargs)


def altwcs_module() -> Any:
    """
    Return the bundled ``wcsutil.altwcs`` module (cached singleton).

    Returns
    -------
    module
        Alternate WCS manipulation helpers.
    """
    global _altwcs_mod
    if _altwcs_mod is None:
        _altwcs_mod = importlib.import_module(f"{_STWCS_PKG}.wcsutil.altwcs")
    return _altwcs_mod


def hstwcs_class() -> Type[Any]:
    """
    Return the :class:`~hst123.utils.stwcs.wcsutil.hstwcs.HSTWCS` class (cached).

    Returns
    -------
    type
        HSTWCS constructor from bundled ``wcsutil``.
    """
    global _hstwcs_cls
    if _hstwcs_cls is None:
        wu = importlib.import_module(f"{_STWCS_PKG}.wcsutil")
        _hstwcs_cls = wu.HSTWCS
    return _hstwcs_cls
