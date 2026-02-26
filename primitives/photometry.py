"""
Photometry math and limit-estimation primitives.

PhotometryHelper is used by the main hst123 pipeline for avg_magnitudes
and estimate_mag_limit.
"""
import numpy as np
from scipy.interpolate import interp1d

from primitives.base import BasePrimitive


def weighted_avg_flux_to_mag(flux, fluxerr):
    """Primitive: convert weighted average flux and error to magnitude and error."""
    if len(flux) == 0 or np.any(fluxerr <= 0):
        return float("NaN"), float("NaN")
    weights = 1.0 / fluxerr**2
    avg_flux = np.sum(flux * weights) / np.sum(weights)
    avg_fluxerr = np.sqrt(np.sum(fluxerr**2) / len(fluxerr))
    mag = 27.5 - 2.5 * np.log10(avg_flux)
    magerr = 1.086 * avg_fluxerr / avg_flux
    return mag, magerr


def estimate_limit_from_snr_bins(mags, errs, snr_target=3.0, n_bins=100):
    """
    Primitive: estimate limiting magnitude by binning in mag and extrapolating
    to a target S/N (e.g. 3-sigma).
    """
    try:
        mags = np.array(mags)
        errs = np.array(errs)
        bin_mag = np.linspace(np.min(mags), np.max(mags), n_bins)
        snr = np.zeros(n_bins)
    except ValueError:
        return np.nan
    for i in range(n_bins):
        if i == n_bins - 1:
            snr[i] = snr[i - 1]
        else:
            idx = np.where((mags > bin_mag[i]) & (mags < bin_mag[i + 1]))[0]
            snr[i] = np.median(1.0 / errs[idx]) if len(idx) > 0 else np.nan
    mask = ~np.isnan(snr)
    bin_mag = bin_mag[mask]
    snr = snr[mask]
    if len(snr) <= 10:
        return np.nan
    snr_func = interp1d(snr, bin_mag, fill_value="extrapolate", bounds_error=False)
    return float(snr_func(snr_target))


class PhotometryHelper(BasePrimitive):
    """Photometry math and limits (avg_magnitudes, estimate_mag_limit)."""

    def __init__(self, pipeline):
        super().__init__(pipeline)

    def avg_magnitudes(self, magerrs, counts, exptimes, zpt):
        idx = []
        for i in np.arange(len(magerrs)):
            try:
                if (
                    float(magerrs[i]) < 0.5
                    and float(counts[i]) > 0.0
                    and float(exptimes[i]) > 0.0
                    and float(zpt[i]) > 0.0
                ):
                    idx.append(i)
            except Exception:
                pass
        if not idx:
            return (float("NaN"), float("NaN"))
        magerrs = np.array([float(m) for m in magerrs])[idx]
        counts = np.array([float(c) for c in counts])[idx]
        exptimes = np.array([float(e) for e in exptimes])[idx]
        zpt = np.array([float(z) for z in zpt])[idx]
        flux = counts / exptimes * 10 ** (0.4 * (27.5 - zpt))
        fluxerr = 1.0 / 1.086 * magerrs * flux
        return weighted_avg_flux_to_mag(flux, fluxerr)

    def estimate_mag_limit(self, mags, errs, limit=3.0):
        warning = (
            "WARNING: cannot sample a wide enough range of magnitudes "
            "to estimate a limit"
        )
        try:
            mags = np.array(mags)
            errs = np.array(errs)
        except ValueError:
            print(warning)
            return np.nan
        result = estimate_limit_from_snr_bins(mags, errs, snr_target=limit)
        if np.isnan(result):
            print(warning)
        return result
