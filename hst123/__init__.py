"""
hst123: HST download, alignment, drizzle, and DOLPHOT photometry.

This package provides the :class:`~hst123._pipeline.hst123` pipeline driver
(lazy-imported via :func:`__getattr__` as ``hst123`` and ``main``) and supporting
utilities under :mod:`hst123.utils` and :mod:`hst123.primitives`.

Notes
-----
Version ``__version__`` is written by setuptools-scm at install time.
"""

try:
    from hst123._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "hst123", "main"]


def __getattr__(name):
    """
    Lazy-load pipeline class and main so heavy deps (stwcs, drizzlepac) are not required at import.

    Parameters
    ----------
    name : str
        Attribute name; only "hst123" and "main" are supported.

    Returns
    -------
    type or function
        The hst123 pipeline class or the main entry-point function.

    Raises
    ------
    AttributeError
        If name is not "hst123" or "main".
    """
    if name == "hst123":
        from hst123._pipeline import hst123
        return hst123
    if name == "main":
        from hst123._pipeline import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
