"""Data models shared by telescope and instrument implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Bandpass:
    name: str
    wavelength_um: np.ndarray
    throughput: np.ndarray
    pivot_um_published: float
    source_url: str
    version: str

    @property
    def pivot_um(self) -> float:
        """Photon-counting pivot wavelength calculated from the stored curve."""
        wave = self.wavelength_um
        transmission = self.throughput
        numerator = np.trapezoid(transmission * wave, wave)
        denominator = np.trapezoid(transmission / wave, wave)
        return float(np.sqrt(numerator / denominator))


@dataclass(frozen=True)
class SensitivityReference:
    filter_name: str
    flux_njy: float
    snr: float
    exposure_s: float
    source_url: str
    version: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class Instrument:
    telescope: str
    name: str
    bandpasses: dict[str, Bandpass]
    sensitivities: dict[str, SensitivityReference]
    official_etc_url: str
    data_root: Path

    @property
    def filters(self) -> tuple[str, ...]:
        return tuple(self.bandpasses)

    @property
    def label(self) -> str:
        """Unique display name, e.g. ``"JWST/NIRCam"`` or ``"HST/WFC3/IR"``."""
        return f"{self.telescope}/{self.name}"


@dataclass(frozen=True)
class CalculationResult:
    quantity: str
    value: float
    unit: str
    filter_name: str
    source_flux_njy: float
    reference: SensitivityReference
    warnings: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"{self.quantity}: {self.value:.4g} {self.unit} | "
            f"{self.filter_name} source flux: {self.source_flux_njy:.4g} nJy | "
            f"reference: {self.reference.snr:g} sigma at "
            f"{self.reference.exposure_s:g} s for {self.reference.flux_njy:g} nJy"
        )
