"""
Astrometry primitive: image alignment and WCS registration.

Contains AstrometryPrimitive (tweakreg and related logic), parse_coord (RA/Dec
parsing), and run_jhat (JWST/JHAT alignment). Requires drizzlepac; run_jhat
requires the optional jhat package.
"""
from hst123.primitives.astrometry.astrometry_primitive import (
    AstrometryPrimitive,
    parse_coord,
)
from hst123.primitives.astrometry.jhat import run_jhat

__all__ = ["AstrometryPrimitive", "parse_coord", "run_jhat"]
