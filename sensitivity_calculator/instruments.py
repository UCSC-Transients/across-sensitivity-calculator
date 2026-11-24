"""Load self-contained instrument definitions and their reference data.

Each instrument is described by one JSON file in ``data/instruments/``. Optional
keys (with their defaults) let different observatories publish their data in
their own conventions:

- ``throughput_directory`` (``"throughputs"``): folder, relative to the data
  root, holding the per-filter two-column throughput tables.
- ``throughput_wavelength_unit`` (``"um"``): any astropy length unit.
- per filter, either ``sensitivity_njy`` or ``sensitivity_ab_mag``: the
  reference flux density at ``reference_snr`` in ``reference_exposure_s``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy import units as u

from .models import Bandpass, Instrument, SensitivityReference
from .photometry import ab_magnitude_to_njy

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

#: Display label -> instrument configuration file (without ``.json``).
INSTRUMENT_CONFIGS = {
    "JWST/NIRCam": "jwst_nircam",
    "HST/WFC3/UVIS": "hst_wfc3_uvis",
    "HST/WFC3/IR": "hst_wfc3_ir",
}


def _read_curve(path: Path, wavelength_unit: str) -> tuple[np.ndarray, np.ndarray]:
    # Comment (#) and single non-numeric header lines are both skipped.
    with path.open() as handle:
        first = handle.readline().split()
    try:
        [float(x) for x in first]
        skip = 0
    except ValueError:
        skip = 1
    curve = np.loadtxt(path, skiprows=skip, comments="#")
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError(f"Malformed throughput curve: {path}")
    wave_um = (curve[:, 0] * u.Unit(wavelength_unit)).to_value(u.um)
    return wave_um, curve[:, 1]


def load_instrument(config: str, data_root: str | Path | None = None) -> Instrument:
    """Load an imaging instrument from ``data/instruments/<config>.json``."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    config_path = root / "instruments" / f"{config}.json"
    meta = json.loads(config_path.read_text())
    curve_dir = root / meta.get("throughput_directory", "throughputs")
    wave_unit = meta.get("throughput_wavelength_unit", "um")

    bandpasses: dict[str, Bandpass] = {}
    sensitivities: dict[str, SensitivityReference] = {}
    for item in meta["filters"]:
        name = item["name"]
        wave_um, throughput = _read_curve(curve_dir / item["throughput_file"], wave_unit)
        # Some published tables (e.g. NIRCam F150W2) contain tiny negative
        # numerical artifacts. Preserve the source files verbatim, but enforce
        # the physical lower bound when constructing a bandpass for integration.
        throughput = np.clip(throughput, 0.0, None)
        bandpasses[name] = Bandpass(
            name=name,
            wavelength_um=wave_um,
            throughput=throughput,
            pivot_um_published=item["pivot_um"],
            source_url=meta["throughput_source_url"],
            version=meta["throughput_version"],
        )
        if "sensitivity_njy" in item:
            flux_njy = float(item["sensitivity_njy"])
        else:
            flux_njy = ab_magnitude_to_njy(item["sensitivity_ab_mag"])
        sensitivities[name] = SensitivityReference(
            filter_name=name,
            flux_njy=flux_njy,
            snr=meta["reference_snr"],
            exposure_s=meta["reference_exposure_s"],
            source_url=meta["sensitivity_source_url"],
            version=meta["sensitivity_version"],
            assumptions=tuple(meta["sensitivity_assumptions"]),
        )

    return Instrument(
        telescope=meta["telescope"],
        name=meta["instrument"],
        bandpasses=bandpasses,
        sensitivities=sensitivities,
        official_etc_url=meta["official_etc_url"],
        data_root=root,
    )


def load_jwst_nircam(data_root: str | Path | None = None) -> Instrument:
    """JWST/NIRCam imaging: 29 filters, STScI ETC v5.0 limits."""
    return load_instrument("jwst_nircam", data_root)


def load_hst_wfc3_uvis(data_root: str | Path | None = None) -> Instrument:
    """HST/WFC3 UVIS imaging: filters with published handbook limits."""
    return load_instrument("hst_wfc3_uvis", data_root)


def load_hst_wfc3_ir(data_root: str | Path | None = None) -> Instrument:
    """HST/WFC3 IR imaging: filters with published handbook limits."""
    return load_instrument("hst_wfc3_ir", data_root)


def load_instruments(
    labels: tuple[str, ...] | list[str] | None = None, data_root: str | Path | None = None
) -> dict[str, Instrument]:
    """Load several instruments, keyed by label (``"JWST/NIRCam"``, ...).

    By default every instrument in :data:`INSTRUMENT_CONFIGS` is loaded.
    """
    labels = list(INSTRUMENT_CONFIGS) if labels is None else list(labels)
    unknown = [label for label in labels if label not in INSTRUMENT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown instrument(s) {unknown}; choose from {list(INSTRUMENT_CONFIGS)}")
    return {label: load_instrument(INSTRUMENT_CONFIGS[label], data_root) for label in labels}
