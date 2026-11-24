"""Reference-scaled exposure-time and signal-to-noise calculations."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np

from .models import Bandpass, CalculationResult, Instrument
from .photometry import njy_to_ab_magnitude


class SensitivityCalculator:
    """Approximate calculator using background-limited square-root scaling."""

    WARNING = (
        "Approximate background-limited scaling; detector noise, source Poisson "
        "noise, saturation, overheads, and configuration changes are not modeled."
    )

    def __init__(self, instrument: Instrument):
        self.instrument = instrument

    def _reference(self, filter_name: str):
        key = filter_name.upper()
        try:
            return key, self.instrument.sensitivities[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported filter {filter_name!r}") from exc

    @staticmethod
    def _positive(value: float, label: str) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
        return float(value)

    def signal_to_noise(self, filter_name: str, source_flux_njy: float, exposure_s: float) -> CalculationResult:
        key, ref = self._reference(filter_name)
        flux = self._positive(source_flux_njy, "Source flux")
        exposure = self._positive(exposure_s, "Exposure time")
        snr = ref.snr * (flux / ref.flux_njy) * np.sqrt(exposure / ref.exposure_s)
        return CalculationResult(
            "Signal-to-noise ratio", float(snr), "", key, flux, ref, (self.WARNING,)
        )

    def exposure_time(self, filter_name: str, source_flux_njy: float, target_snr: float) -> CalculationResult:
        key, ref = self._reference(filter_name)
        flux = self._positive(source_flux_njy, "Source flux")
        snr = self._positive(target_snr, "Target S/N")
        exposure = ref.exposure_s * (snr / ref.snr) ** 2 * (ref.flux_njy / flux) ** 2
        return CalculationResult(
            "Integration time", float(exposure), "s", key, flux, ref, (self.WARNING,)
        )


def sensitivity_table(
    instruments: Mapping[str, Instrument],
    selections: Iterable[tuple[str, str]],
    source_flux_njy: float | Callable[[Bandpass], float],
    exposure_s: float,
    target_snr: float,
) -> list[dict]:
    """S/N and required time for several (instrument label, filter) pairs at once.

    ``source_flux_njy`` is either one flux density used in every filter, or a
    function returning the band-averaged flux for a :class:`Bandpass` (e.g.
    ``lambda bp: sed.flux_njy(bp, normalization_bp, mag)``). A failure in one
    filter (such as a spectrum not covering it) is reported in that row's
    ``error`` entry and does not stop the others.
    """
    exposure_s = SensitivityCalculator._positive(exposure_s, "Exposure time")
    target_snr = SensitivityCalculator._positive(target_snr, "Target S/N")
    rows = []
    for label, filter_name in selections:
        instrument = instruments[label]
        bandpass = instrument.bandpasses[filter_name]
        calculator = SensitivityCalculator(instrument)
        row = {
            "instrument": label,
            "filter": filter_name,
            "pivot_um": bandpass.pivot_um,
            "flux_njy": np.nan,
            "ab_mag": np.nan,
            "snr": np.nan,
            "exposure_time_s": np.nan,
            "reference": instrument.sensitivities[filter_name],
            "error": None,
        }
        try:
            flux = source_flux_njy(bandpass) if callable(source_flux_njy) else source_flux_njy
            row["flux_njy"] = flux
            row["ab_mag"] = njy_to_ab_magnitude(flux)
            row["snr"] = calculator.signal_to_noise(filter_name, flux, exposure_s).value
            row["exposure_time_s"] = calculator.exposure_time(filter_name, flux, target_snr).value
        except ValueError as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows
