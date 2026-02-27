"""
Pipeline helper primitives: FITS, photometry, astrometry (TweakReg, JHAT), etc.

All helpers inherit from BasePrimitive and are composed by the main hst123
pipeline. AstrometryPrimitive holds TweakReg/alignment logic; run_jhat (JWST
alignment) is optional and requires the jhat package.
"""
from hst123.primitives.base import BasePrimitive
from hst123.primitives.fits import FitsHelper
from hst123.primitives.photometry import PhotometryHelper
from hst123.primitives.astrometry import AstrometryPrimitive, run_jhat

__all__ = ["BasePrimitive", "FitsHelper", "PhotometryHelper", "AstrometryPrimitive", "run_jhat"]
