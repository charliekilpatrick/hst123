"""Base FITS data model: multi-extension load/save, metadata, Primary/Image/BinTable HDU."""
from __future__ import annotations

import os
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
import astropy.table

try:
    from hst123 import __version__
except ImportError:
    __version__ = "0.0.0"


_VALID_HDU = fits.ImageHDU | fits.BinTableHDU | fits.PrimaryHDU


class DataModel:
    """
    Base class for data models.

    Load from a FITS file or build from in-memory data and metadata. Extension
    layout is defined by the class attribute EXTENSIONS. Metadata can be read
    from the primary header or set explicitly.

    Parameters
    ----------
    filepath : str, optional
        Path to a FITS file to load.
    data : dict, optional
        Dictionary of extension name -> array or Table. Keys should match
        EXTENSIONS.
    meta : dict, optional
        Metadata key/value pairs (e.g. for primary header).
    meta_only : bool, optional
        If True, only metadata is loaded from the FITS file. Default False.
    extensions : str or list of str, optional
        Extensions to load from file. Default all in EXTENSIONS.
    validate_meta : bool, optional
        Reserved for future metadata validation. Default True.

    Attributes
    ----------
    EXTENSIONS : dict
        Extension names and their properties (hdu_type, dtype, optional columns).
    _ALLOWED_ATTRIBUTES : list
        Attribute names that can be set directly on the model.
    _INHERIT_EXTENSIONS : bool
        Whether subclasses inherit EXTENSIONS from parent. Default True.
    """

    EXTENSIONS = {
        "PRIMARY": {"hdu_type": fits.PrimaryHDU, "dtype": None},
    }

    _ALLOWED_ATTRIBUTES = [
        "meta",
        "_filepath",
        "_validate_meta",
        "_data",
        "_ALLOWED_ATTRIBUTES",
        "EXTENSIONS",
    ]

    _INHERIT_EXTENSIONS = True

    def __init__(
        self,
        filepath: str | None = None,
        data: dict | None = None,
        meta: dict | None = None,
        meta_only: bool = False,
        extensions: str | list[str] | None = None,
        validate_meta: bool = True,
    ):
        self._validate_meta = validate_meta
        self.EXTENSIONS = self._get_extensions()

        if filepath is not None:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            self._filepath = os.path.abspath(filepath)
            self.init_meta(meta=meta)
            if isinstance(self.meta, dict) and self.meta.get("filename") is None:
                self.meta["filename"] = os.path.basename(self._filepath)
            if not meta_only:
                self.read_data(extensions=extensions)
            else:
                self._data = {}
        else:
            self._filepath = None
            self.init_meta(meta=meta)
            if not meta_only:
                self.init_data(data=data)
            else:
                self._data = {}

        self.on_init()

    def _get_extensions(self) -> dict:
        """
        Return the extension schema for this model (subclasses may override).

        Returns
        -------
        dict
            Copy of EXTENSIONS (or subclass override).
        """
        return dict(self.EXTENSIONS)

    def read_data(
        self,
        extensions: str | list[str] | None = None,
        do_not_scale_image_data: bool | None = None,
    ) -> dict[str, Any]:
        """
        Read extension data from the FITS file.

        Parameters
        ----------
        extensions : str or list of str, optional
            Extension names to read. Default all in EXTENSIONS.
        do_not_scale_image_data : bool or None, optional
            Passed to fits.open. See astropy documentation.

        Returns
        -------
        dict
            Extension name -> array or Table.
        """
        extensions = self._resolve_extensions(extensions)
        self._data = {}
        with fits.open(
            self._filepath,
            memmap=False,
            do_not_scale_image_data=do_not_scale_image_data,
        ) as hdulist:
            for extname in extensions:
                if extname in hdulist:
                    self._data[extname] = self.read_extension_data(hdulist[extname])
        for extname in self.EXTENSIONS:
            if extname not in self._data:
                if "columns" in self.EXTENSIONS[extname]:
                    cols = self.EXTENSIONS[extname]["columns"]
                    self._data[extname] = astropy.table.Table({
                        name: np.array([], dtype=dtype)
                        for name, dtype in cols.items()
                    })
                else:
                    self._data[extname] = None
        return self._data

    def read_extension_data(
        self,
        hdu: _VALID_HDU,
    ) -> np.ndarray | astropy.table.Table:
        """
        Read data from a single HDU.

        Parameters
        ----------
        hdu : PrimaryHDU, ImageHDU, or BinTableHDU
            The HDU to read.

        Returns
        -------
        np.ndarray or astropy.table.Table
            Image data as ndarray, table data as Table.
        """
        data = hdu.data
        if type(hdu) is fits.BinTableHDU:
            return astropy.table.Table(data) if data is not None else astropy.table.Table()
        return data

    def init_meta(self, meta: dict | None = None) -> None:
        """
        Initialize metadata from file and/or provided dict.

        If filepath is set, primary header keywords are read into meta.
        Provided meta overrides or adds to those values.

        Parameters
        ----------
        meta : dict, optional
            Metadata key/value pairs to merge with (or override) header-derived meta.

        Raises
        ------
        ValueError
            If meta is not dict or None.
        """
        meta_from_file = {}
        if self._filepath is not None:
            meta_from_file = self.read_meta_from_fits()
        if meta is not None and not isinstance(meta, dict):
            raise ValueError(f"meta must be dict or None, got {type(meta)}")
        merged = {**meta_from_file, **(meta or {})}
        self.meta = merged

    def read_meta_from_fits(self) -> dict:
        """
        Read primary header into a flat dict (keyword -> value).

        Returns
        -------
        dict
            FITS primary header keywords and values (excluding COMMENT, HISTORY, blank).
        """
        meta = {}
        with fits.open(self._filepath, memmap=True) as hdulist:
            for key, value in hdulist["PRIMARY"].header.items():
                if key not in ("COMMENT", "HISTORY") and not key.startswith(" "):
                    meta[key] = value
        return meta

    def init_data(self, data: dict | None = None) -> None:
        """
        Initialize extension data from a dictionary.

        Keys must match EXTENSIONS. Values are numpy arrays or astropy Tables.

        Parameters
        ----------
        data : dict, optional
            Extension name -> array or Table. Keys must be in EXTENSIONS.

        Raises
        ------
        ValueError
            If an extension name is not in EXTENSIONS.
        """
        self._data = {}
        if data is not None:
            for key in data:
                key_upper = key.upper()
                if key_upper in self.EXTENSIONS:
                    if isinstance(data[key], astropy.table.Table):
                        self._data[key_upper] = data[key]
                    elif isinstance(data[key], np.ndarray):
                        self._data[key_upper] = data[key]
                    else:
                        self._data[key_upper] = data[key]
                else:
                    raise ValueError(f"Unknown extension: {key_upper}")
        for extname in self.EXTENSIONS:
            if extname not in self._data:
                if "columns" in self.EXTENSIONS[extname]:
                    cols = self.EXTENSIONS[extname]["columns"]
                    self._data[extname] = astropy.table.Table({
                        name: np.array([], dtype=dtype)
                        for name, dtype in cols.items()
                    })
                else:
                    self._data[extname] = None

    def save(
        self,
        filepath: str | None = None,
        filename: str | None = None,
        output_dir: str | None = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Save the model to a FITS file.

        Parameters
        ----------
        filepath : str, optional
            Full path to write. If None, filename + output_dir can be used.
        filename : str, optional
            Basename; requires output_dir.
        output_dir : str, optional
            Directory; used with filename or to derive path.
        overwrite : bool, optional
            Overwrite existing file. Default True.
        **kwargs
            Passed to HDUList.writeto.

        Returns
        -------
        str
            Path to the saved file.
        """
        if filepath is not None:
            self._filepath = filepath
        elif output_dir is not None and filename is not None:
            self._filepath = os.path.abspath(os.path.join(output_dir, filename))
        elif output_dir is not None and filename is None and isinstance(self.meta, dict) and self.meta.get("filename"):
            self._filepath = os.path.abspath(os.path.join(output_dir, self.meta["filename"]))
        else:
            raise ValueError("Specify filepath or (output_dir and optionally filename).")
        self.on_save()
        hdulist = self.to_hdulist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VerifyWarning)
            hdulist.writeto(self._filepath, overwrite=overwrite, **kwargs)
        return self._filepath

    def to_hdulist(
        self,
        include_empty_extensions: bool = False,
    ) -> fits.HDUList:
        """
        Build an HDUList from current metadata and extension data.

        Parameters
        ----------
        include_empty_extensions : bool, optional
            If True, include extensions that have no data. Default False.

        Returns
        -------
        astropy.io.fits.HDUList
        """
        hdulist = fits.HDUList()
        order = ["PRIMARY"] + [e for e in self.EXTENSIONS if e != "PRIMARY"]
        for extname in order:
            if extname not in self.EXTENSIONS:
                continue
            if not include_empty_extensions and extname != "PRIMARY":
                if extname not in self._data or self._data[extname] is None:
                    continue
            hdu = self.make_hdu(extname)
            hdulist.append(hdu)
        return hdulist

    def make_hdu(self, extname: str) -> _VALID_HDU:
        """
        Create one HDU for the given extension name.

        Parameters
        ----------
        extname : str
            Extension name (must be in EXTENSIONS).

        Returns
        -------
        PrimaryHDU, ImageHDU, or BinTableHDU
            The constructed HDU with header and data from the model.
        """
        hdu_class = self.EXTENSIONS[extname]["hdu_type"]
        header = self._meta_to_header(extname)
        if hdu_class is fits.PrimaryHDU:
            return hdu_class(header=header)
        if hdu_class is fits.ImageHDU:
            data = self._data.get(extname)
            return hdu_class(data=data, name=extname)
        if hdu_class is fits.BinTableHDU:
            data = self._data.get(extname)
            return hdu_class(data=data, name=extname)
        return hdu_class(header=header)

    def _meta_to_header(self, extname: str) -> fits.Header:
        """
        Convert meta dict to FITS header (primary only; others empty).

        Parameters
        ----------
        extname : str
            Extension name; only "PRIMARY" gets meta keys written.

        Returns
        -------
        astropy.io.fits.Header
            Header with meta key/value pairs for PRIMARY; empty for others.
        """
        header = fits.Header()
        if extname == "PRIMARY" and isinstance(self.meta, dict):
            for key, value in self.meta.items():
                if key in ("COMMENT", "HISTORY") or key.startswith(" "):
                    continue
                try:
                    header[key] = value
                except Exception:
                    pass
        return header

    def on_init(self) -> None:
        """
        Hook called after initialization. Subclasses may set meta defaults.

        Sets model_type, filename (if filepath set), and drp_version in meta.
        """
        if isinstance(self.meta, dict):
            self.meta.setdefault("model_type", self.__class__.__name__)
            if self._filepath is not None:
                self.meta.setdefault("filename", os.path.basename(self._filepath))
            self.meta.setdefault("drp_version", __version__)

    def on_save(self) -> None:
        """
        Hook called before save. Subclasses may update meta.

        Raises
        ------
        ValueError
            If _filepath is not set.
        """
        if self._filepath is None:
            raise ValueError("Cannot save: _filepath not set.")
        if isinstance(self.meta, dict):
            self.meta["filename"] = os.path.basename(self._filepath)
            self.meta["model_type"] = self.__class__.__name__

    def _resolve_extensions(self, extensions: str | list[str] | None) -> list[str]:
        """
        Resolve extensions argument to a list of extension names (uppercase).

        Parameters
        ----------
        extensions : str, list of str, or None
            Extension name(s) to load; None means all in EXTENSIONS.

        Returns
        -------
        list of str
            Uppercase extension names.
        """
        if extensions is None:
            return list(self.EXTENSIONS.keys())
        if isinstance(extensions, str):
            return [extensions.upper()]
        return [e.upper() for e in extensions]

    def __getattr__(self, attr: str) -> Any:
        """
        Allow access to extension data by attribute name (e.g. model.SCI).

        Parameters
        ----------
        attr : str
            Extension name (case-insensitive).

        Returns
        -------
        np.ndarray or astropy.table.Table or None
            Extension data.

        Raises
        ------
        AttributeError
            If attr is not in EXTENSIONS.
        """
        attru = attr.upper()
        if attru not in self.EXTENSIONS:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        return self._data.get(attru)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_") or attr in getattr(self, "_ALLOWED_ATTRIBUTES", []):
            object.__setattr__(self, attr, value)
            return
        attru = attr.upper()
        if attru in getattr(self, "EXTENSIONS", {}):
            if not hasattr(self, "_data"):
                object.__setattr__(self, "_data", {})
            self._data[attru] = value
            return
        object.__setattr__(self, attr, value)

    def copy(self) -> DataModel:
        """
        Return a deep copy of the model with _filepath cleared.

        Returns
        -------
        DataModel
            New instance with same data and meta; _filepath set to None.
        """
        model_out = deepcopy(self)
        model_out._filepath = None
        return model_out

    def __repr__(self) -> str:
        fn = None
        if isinstance(self.meta, dict):
            fn = self.meta.get("filename")
        if fn is None and hasattr(self.meta, "filename"):
            fn = self.meta.filename
        if fn is not None:
            return f"{self.__class__.__name__}('{fn}')"
        return f"{self.__class__.__name__}()"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "_INHERIT_EXTENSIONS", True):
            return
        for base in cls.__mro__[1:]:
            if hasattr(base, "EXTENSIONS") and base is not DataModel:
                for extname, spec in base.EXTENSIONS.items():
                    if extname not in cls.EXTENSIONS:
                        cls.EXTENSIONS = dict(cls.EXTENSIONS)
                        cls.EXTENSIONS[extname] = spec
                break
