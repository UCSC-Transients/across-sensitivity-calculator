"""Small, dependency-light photon-counting synthetic-photometry helpers."""

from __future__ import annotations

import numpy as np

from .models import Bandpass

AB_ZEROPOINT_NJY = 3631.0e9


def ab_magnitude_to_njy(magnitude: float) -> float:
    """Convert an AB magnitude to flux density in nJy."""
    if not np.isfinite(magnitude):
        raise ValueError("AB magnitude must be finite")
    return float(AB_ZEROPOINT_NJY * 10.0 ** (-0.4 * magnitude))


def njy_to_ab_magnitude(flux_njy: float) -> float:
    """Convert a positive flux density in nJy to AB magnitude."""
    if not np.isfinite(flux_njy) or flux_njy <= 0:
        raise ValueError("Flux density must be finite and positive")
    return float(-2.5 * np.log10(flux_njy / AB_ZEROPOINT_NJY))


def photon_weighted_mean_fnu(
    bandpass: Bandpass,
    wavelength_um: np.ndarray,
    fnu_shape: np.ndarray,
) -> float:
    """Return band-averaged f_nu for a photon-counting response.

    The input SED may have arbitrary normalization. The returned value retains
    those arbitrary units. No extrapolation is allowed beyond the SED grid.
    """
    wave = np.asarray(wavelength_um, dtype=float)
    fnu = np.asarray(fnu_shape, dtype=float)
    if wave.ndim != 1 or fnu.shape != wave.shape or len(wave) < 2:
        raise ValueError("SED wavelength and f_nu arrays must be matching 1-D arrays")
    if np.any(~np.isfinite(wave)) or np.any(~np.isfinite(fnu)):
        raise ValueError("SED contains non-finite values")
    if np.any(np.diff(wave) <= 0):
        raise ValueError("SED wavelengths must be strictly increasing")

    bp_wave = bandpass.wavelength_um
    throughput = bandpass.throughput
    active = throughput > throughput.max() * 1e-6
    if wave[0] > bp_wave[active].min() or wave[-1] < bp_wave[active].max():
        raise ValueError(
            f"SED ({wave[0]:.4g}-{wave[-1]:.4g} um) does not cover the {bandpass.name} "
            f"bandpass ({bp_wave[active].min():.4g}-{bp_wave[active].max():.4g} um)"
        )

    # Integrate on the union of both grids so neither a finely sampled
    # spectrum nor a finely sampled throughput curve is undersampled.
    lo, hi = max(bp_wave[0], wave[0]), min(bp_wave[-1], wave[-1])
    grid = np.union1d(bp_wave, wave)
    grid = grid[(grid >= lo) & (grid <= hi)]
    sampled_fnu = np.interp(grid, wave, fnu)
    weight = np.interp(grid, bp_wave, throughput) / grid
    return float(np.trapezoid(sampled_fnu * weight, grid) / np.trapezoid(weight, grid))
