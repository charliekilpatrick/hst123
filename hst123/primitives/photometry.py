"""Photometry math and limit estimation (avg_magnitudes, estimate_mag_limit)."""
import logging

import numpy as np
from scipy.interpolate import interp1d

from hst123.primitives.base import BasePrimitive

log = logging.getLogger(__name__)


def weighted_avg_flux_to_mag(flux, fluxerr):
    """
    Convert weighted average flux and flux error to magnitude and magnitude error.

    Parameters
    ----------
    flux : array-like
        Flux values (e.g. count rate).
    fluxerr : array-like
        Flux uncertainties; must be > 0.

    Returns
    -------
    tuple of float
        (mag, magerr). Returns (NaN, NaN) if flux is empty or any fluxerr <= 0.
    """
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
    Estimate limiting magnitude by binning in magnitude and interpolating to target S/N.

    Parameters
    ----------
    mags : array-like
        Magnitudes of sources.
    errs : array-like
        Magnitude errors (same length as mags).
    snr_target : float, optional
        Target signal-to-noise for the limit (e.g. 3 for 3-sigma). Default 3.0.
    n_bins : int, optional
        Number of magnitude bins. Default 100.

    Returns
    -------
    float
        Estimated limiting magnitude at snr_target; np.nan if insufficient data.
    """
    try:
        mags = np.asarray(mags, dtype=np.float64)
        errs = np.asarray(errs, dtype=np.float64)
        lo, hi = np.min(mags), np.max(mags)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.nan
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        inv_err = 1.0 / errs
        dig = np.searchsorted(bin_edges, mags, side="right") - 1
        dig = np.clip(dig, 0, n_bins - 1)
        snr = np.empty(n_bins)
        for i in range(n_bins - 1):
            sel = dig == i
            snr[i] = np.nanmedian(inv_err[sel]) if np.any(sel) else np.nan
        snr[n_bins - 1] = snr[n_bins - 2] if n_bins > 1 else np.nan
    except (ValueError, TypeError):
        return np.nan
    mask = ~np.isnan(snr)
    bin_mag = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_mag = bin_mag[mask]
    snr = snr[mask]
    if len(snr) <= 10:
        return np.nan
    # Ensure strictly increasing snr so scipy interp1d doesn't divide by zero (x_hi - x_lo)
    order = np.argsort(snr)
    snr_s = snr[order]
    bin_mag_s = bin_mag[order]
    keep = np.concatenate([[True], snr_s[1:] > snr_s[:-1]])
    snr_u = snr_s[keep]
    bin_mag_u = bin_mag_s[keep]
    if len(snr_u) < 2:
        return np.nan
    snr_func = interp1d(snr_u, bin_mag_u, fill_value="extrapolate", bounds_error=False)
    return float(snr_func(snr_target))


class PhotometryHelper(BasePrimitive):
    """
    Photometry math and limit estimation for the hst123 pipeline.

    Provides avg_magnitudes (weighted average flux to mag) and estimate_mag_limit
    (limit from S/N bins). Used when scraping dolphot or reporting final photometry.
    """

    def avg_magnitudes(self, magerrs, counts, exptimes, zpt):
        """
        Compute weighted average magnitude and error from multi-epoch counts and zero points.

        Parameters
        ----------
        magerrs : array-like
            Magnitude uncertainties per measurement.
        counts : array-like
            Counts (or count rates) per measurement.
        exptimes : array-like
            Exposure times per measurement.
        zpt : array-like
            Zero points per measurement.

        Returns
        -------
        tuple of float
            (mag, magerr). (NaN, NaN) if no valid measurements (e.g. magerr < 0.5, counts > 0).
        """
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
        mag, magerr = weighted_avg_flux_to_mag(flux, fluxerr)
        return mag, magerr

    def estimate_mag_limit(self, mags, errs, limit=3.0):
        """
        Estimate limiting magnitude at a given S/N (e.g. 3-sigma).

        Parameters
        ----------
        mags : array-like
            Magnitudes of sources.
        errs : array-like
            Magnitude errors.
        limit : float, optional
            Target S/N for the limit. Default 3.0.

        Returns
        -------
        float
            Limiting magnitude; np.nan if insufficient range or data.
        """
        warning = (
            "Cannot sample a wide enough range of magnitudes "
            "to estimate a limit."
        )
        try:
            mags = np.array(mags)
            errs = np.array(errs)
        except ValueError:
            log.warning(warning)
            return np.nan
        result = estimate_limit_from_snr_bins(mags, errs, snr_target=limit)
        if np.isnan(result):
            log.warning(warning)
        return result
