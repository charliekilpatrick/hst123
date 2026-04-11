"""
hst123: HST download, alignment, drizzle, and DOLPHOT photometry.

Public API
----------
``__version__``
    Package version (from setuptools-scm at install time).
``hst123``
    Pipeline class :class:`hst123._pipeline.hst123` (lazy-loaded).
``main``
    CLI entry :func:`hst123._pipeline.main` (lazy-loaded).

Heavy optional dependencies (e.g. DrizzlePac) are imported only when the
pipeline class or ``main`` is first accessed.

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
    Lazy-load the pipeline class and CLI entry so imports stay lightweight.

    Parameters
    ----------
    name : str
        Must be ``"hst123"`` or ``"main"``.

    Returns
    -------
    object
        The pipeline class or the ``main`` function.

    Raises
    ------
    AttributeError
        If name is not ``"hst123"`` or ``"main"``.
    """
    if name == "hst123":
        from hst123._pipeline import hst123
        return hst123
    if name == "main":
        from hst123._pipeline import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
