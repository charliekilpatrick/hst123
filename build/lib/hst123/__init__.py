"""hst123: download, register, drizzle, and run dolphot on HST images."""
__all__ = ["hst123", "main"]


def __getattr__(name):
    # Lazy load so "from hst123.common import X" does not require stwcs/drizzlepac
    if name == "hst123":
        from hst123._pipeline import hst123
        return hst123
    if name == "main":
        from hst123._pipeline import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
