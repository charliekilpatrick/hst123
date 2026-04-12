"""STWCS

This package provides support for WCS based distortion models and coordinate
transformation. It relies on astropy.wcs (based on WCSLIB). It consists of
two subpackages:

* updatewcs: Performs corrections to the basic WCS and includes other distortion
  infomation in the science files as header keywords or file extensions.
* wcsutil:  Provides an HSTWCS object which extends astropy.wcs.WCS object and
  provides HST instrument specific information as well as methods for coordinate
  transformation. wcsutil also provides functions for manipulating alternate WCS
  descriptions in the headers.

"""
from . import distortion  # noqa
from .version import __version__
