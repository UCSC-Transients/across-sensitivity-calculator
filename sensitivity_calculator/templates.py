"""Libraries of model spectra (stellar populations, stellar atlases, ...).

A library is a directory of spectrum files plus a ``manifest.json``::

    {
      "description": "...",
      "flux_units": "L_sun / Angstrom",        # or "flux_system": "F_lambda"
      "wavelength_unit": "Angstrom",           # optional, default Angstrom
      "templates": [
        {"file": "model_1Gyr.txt", "age_gyr": 1, "logzsol": 0.0, ...},
        ...
      ]
    }

Every key in a template entry other than ``file`` (and bookkeeping keys such as
``flux_units``) is a selectable parameter. Files are read with
:func:`~sensitivity_calculator.seds.read_spectrum_file`, so column 1 is the
spectral axis and column 2 the flux. Rest-frame templates can be redshifted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy import units as u

from .seds import FluxSystem, TabulatedSED, read_spectrum_file

_NON_PARAMETER_KEYS = {"file", "flux_units", "wavelength_unit", "flux_system", "redshift"}

DEFAULT_FSPS_LIBRARY = Path(__file__).resolve().parent.parent / "fsps_template_library"


def flux_system_from_unit(unit: str | u.UnitBase) -> FluxSystem:
    """Infer the flux convention from a flux or luminosity density unit."""
    unit = u.Unit(unit)
    per_wavelength = (u.W / u.m, u.W / u.m**3, u.photon / u.s / u.m**3)
    per_frequency = (u.W / u.Hz, u.W / u.m**2 / u.Hz)
    if any(unit.is_equivalent(x) for x in per_wavelength[:2]):
        return FluxSystem.F_LAMBDA
    if unit.is_equivalent(per_wavelength[2]) or unit.is_equivalent(u.photon / u.s / u.m):
        return FluxSystem.PHOTON_F_LAMBDA
    if any(unit.is_equivalent(x) for x in per_frequency):
        return FluxSystem.F_NU
    if unit.is_equivalent(u.W) or unit.is_equivalent(u.W / u.m**2):
        return FluxSystem.LAMBDA_F_LAMBDA
    raise ValueError(f"Cannot infer a flux convention from unit {unit}")


@dataclass(frozen=True)
class Template:
    path: Path
    parameters: dict
    redshift: float = 0.0


class TemplateLibrary:
    """A directory of model spectra described by ``manifest.json``."""

    def __init__(self, directory: str | Path, name: str | None = None):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {self.directory}")
        manifest = json.loads(manifest_path.read_text())
        self.name = name or manifest.get("name") or self.directory.name
        self.description = manifest.get("description", "")
        self.wavelength_unit = manifest.get("wavelength_unit", "Angstrom")
        if "flux_system" in manifest:
            self.flux_system = FluxSystem(manifest["flux_system"])
        else:
            self.flux_system = flux_system_from_unit(manifest["flux_units"])
        self.templates = [
            Template(
                self.directory / entry["file"],
                {k: v for k, v in entry.items() if k not in _NON_PARAMETER_KEYS},
                float(entry.get("redshift", manifest.get("redshift", 0.0)) or 0.0),
            )
            for entry in manifest["templates"]
        ]
        if not self.templates:
            raise ValueError(f"Template library {self.directory} is empty")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.templates[0].parameters)

    def parameter_values(self, name: str) -> list:
        return sorted({t.parameters[name] for t in self.templates})

    def find(self, **parameters) -> Template:
        def matches(t: Template) -> bool:
            return all(
                np.isclose(t.parameters[k], v) if isinstance(v, (int, float)) else t.parameters[k] == v
                for k, v in parameters.items()
            )

        hits = [t for t in self.templates if matches(t)]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} templates in {self.name} match {parameters}")
        return hits[0]

    def sed(self, redshift: float = 0.0, **parameters) -> TabulatedSED:
        """Return the template as a :class:`TabulatedSED`, shifted to ``redshift``.

        Templates are taken to be at the manifest redshift (normally 0). Only the
        spectral shape matters downstream, so redshifting is a stretch of the
        wavelength axis; the constant (1+z) flux factors cancel on normalization.
        """
        if not np.isfinite(redshift) or redshift < 0:
            raise ValueError("Redshift must be finite and non-negative")
        template = self.find(**parameters)
        wave, flux = _read_template(template.path)
        stretch = (1.0 + redshift) / (1.0 + template.redshift)
        label = ", ".join(_format_parameter(k, v) for k, v in template.parameters.items())
        return TabulatedSED(
            wave * stretch, flux, self.wavelength_unit, self.flux_system,
            name=f"{self.name} ({label}, z={redshift:g})",
        )


def _format_parameter(name: str, value) -> str:
    if name == "age_gyr":
        return f"{value:g} Gyr"
    if name == "logzsol":
        return f"[Z/H]={value:+g}" if value else "[Z/H]=0"
    if name == "teff":
        return f"{value:g} K"
    return f"{name}={value:g}" if isinstance(value, (int, float)) else f"{name}={value}"


@lru_cache(maxsize=128)
def _read_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    spec = read_spectrum_file(path)
    good = np.isfinite(spec.wavelength) & np.isfinite(spec.flux)
    return spec.wavelength[good], spec.flux[good]


def load_fsps_library(directory: str | Path = DEFAULT_FSPS_LIBRARY) -> TemplateLibrary:
    """Load the bundled FSPS simple-stellar-population template library."""
    return TemplateLibrary(directory, name="FSPS SSP")
