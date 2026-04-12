"""Visit grouping for observation tables (instrument + time + filter)."""
import numpy as np
from astropy.time import Time


def add_visit_info(obstable, visit_tol, log=None):
    """Assign visit numbers by instrument, filter, and time within visit_tol days. In-place; returns obstable or None on error."""
    if log is None:
        import logging
        log = logging.getLogger(__name__)

    obstable["visit"] = [int(0)] * len(obstable)
    obstable.sort("datetime")

    for row in obstable:
        inst = row["instrument"]
        mjd = Time(row["datetime"]).mjd
        filt = row["filter"]

        if all(obs["visit"] == 0 for obs in obstable):
            row["visit"] = int(1)
        else:
            instmask = obstable["instrument"] == inst
            timemask = [
                abs(Time(obs["datetime"]).mjd - mjd) < visit_tol
                for obs in obstable
            ]
            filtmask = [filt == fval for fval in obstable["filter"]]
            nzero = obstable["visit"] != 0
            mask = [all(l) for l in zip(instmask, timemask, filtmask, nzero)]

            if not any(mask):
                row["visit"] = int(np.max(obstable["visit"]) + 1)
            else:
                if len(list(set(obstable[mask]["visit"]))) != 1:
                    log.error("Visit numbers are incorrectly assigned.")
                    return None
                row["visit"] = list(set(obstable[mask]["visit"]))[0]

    return obstable
