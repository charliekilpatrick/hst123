"""Lightweight HST file handle with path properties and metadata (dict-based)."""
from __future__ import annotations

import os


class HSTModel:
    """
    Base class for associating a FITS path with metadata.

    Subclasses may extend :meth:`on_init` for additional setup.

    Parameters
    ----------
    filepath : str or None
        Path to an HST data file, or None for in-memory-only use.
    meta : dict, optional
        Initial metadata merged into :attr:`meta`.

    Attributes
    ----------
    filepath : str or None
        Path passed at construction.
    meta : dict
        Metadata dictionary (e.g. ``model_type``, ``filename``).
    """

    def __init__(self, filepath, meta: dict | None = None):
        self.filepath = filepath
        self.init_meta(meta=meta)
        self.on_init()

    @property
    def dirname(self) -> str | None:
        """
        Directory containing :attr:`filepath`, or None if no path is set.

        Returns
        -------
        str or None
        """
        return os.path.dirname(self.filepath) if self.filepath else None

    @property
    def basename(self) -> str | None:
        """
        Base filename of :attr:`filepath`, or None if no path is set.

        Returns
        -------
        str or None
        """
        return os.path.basename(self.filepath) if self.filepath else None

    def init_meta(self, meta: dict | None = None) -> None:
        """
        Reset metadata and merge optional user values.

        Parameters
        ----------
        meta : dict, optional
            Keys merged into :attr:`meta`.
        """
        self.meta = {}
        if meta is not None:
            self.meta.update(meta)

    def on_init(self) -> None:
        """
        Hook invoked after :meth:`init_meta`; default sets model type and filename.

        Override in subclasses for custom initialization.
        """
        self.meta["model_type"] = self.__class__.__name__
        if self.filepath is not None:
            self.meta["filename"] = os.path.basename(self.filepath)
