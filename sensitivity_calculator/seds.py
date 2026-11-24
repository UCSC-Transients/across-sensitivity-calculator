"""SED models. Add future template families by implementing ``fnu_shape``."""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
from astropy import constants as const
from astropy import units as u

from .models import Bandpass
from .photometry import ab_magnitude_to_njy, photon_weighted_mean_fnu


class SED(Protocol):
    def fnu_shape(self, wavelength_um: np.ndarray) -> np.ndarray:
        """Return f_nu spectral shape with arbitrary normalization."""


def transfer_ab_normalization(
    wavelength_um: np.ndarray,
    fnu_shape: np.ndarray,
    target: Bandpass,
    normalization_bandpass: Bandpass,
    normalization_ab_magnitude: float,
) -> float:
    """Scale an arbitrary-normalization f_nu shape to an AB magnitude in one
    bandpass and return the implied band-averaged flux (nJy) in another."""
    norm_mean = photon_weighted_mean_fnu(normalization_bandpass, wavelength_um, fnu_shape)
    if not norm_mean > 0:
        raise ValueError(
            f"SED has non-positive mean flux in normalization band {normalization_bandpass.name}"
        )
    target_mean = photon_weighted_mean_fnu(target, wavelength_um, fnu_shape)
    if not target_mean > 0:
        raise ValueError(f"SED has non-positive mean flux in observing band {target.name}")
    return ab_magnitude_to_njy(normalization_ab_magnitude) * target_mean / norm_mean


class BlackbodySED:
    """Planck blackbody whose amplitude can be normalized in any bandpass."""

    def __init__(self, temperature_k: float):
        if not np.isfinite(temperature_k) or temperature_k <= 0:
            raise ValueError("Blackbody temperature must be finite and positive")
        self.temperature_k = float(temperature_k)
        self.name = f"{self.temperature_k:g} K blackbody"

    def fnu_shape(self, wavelength_um: np.ndarray) -> np.ndarray:
        wave_m = np.asarray(wavelength_um, dtype=float) * 1e-6
        if np.any(wave_m <= 0):
            raise ValueError("Wavelengths must be positive")
        exponent = const.h.value * const.c.value / (wave_m * const.k_B.value * self.temperature_k)
        # Overall units and solid angle cancel when the SED is normalized.
        return (2.0 * const.h.value * const.c.value / wave_m**3) / np.expm1(exponent)

    def flux_njy(
        self,
        target: Bandpass,
        normalization_bandpass: Bandpass,
        normalization_ab_magnitude: float,
    ) -> float:
        lo = min(target.wavelength_um.min(), normalization_bandpass.wavelength_um.min())
        hi = max(target.wavelength_um.max(), normalization_bandpass.wavelength_um.max())
        wave = np.geomspace(lo, hi, 20_000)
        return transfer_ab_normalization(
            wave, self.fnu_shape(wave), target, normalization_bandpass, normalization_ab_magnitude
        )


class FluxSystem(str, Enum):
    """How the second column of a user spectrum is expressed.

    Only the *shape* matters (the spectrum is renormalized), so constant unit
    factors such as erg/s/cm2/A vs W/m2/um, or Jy vs mJy, are irrelevant. What
    does matter is the convention, because converting to f_nu introduces
    wavelength-dependent factors.
    """

    F_NU = "F_nu"                    # per unit frequency (Jy, erg/s/cm2/Hz, ...)
    F_LAMBDA = "F_lambda"            # per unit wavelength (erg/s/cm2/A, ...)
    LAMBDA_F_LAMBDA = "lambda_F_lambda"  # lambda*F_lambda == nu*F_nu (erg/s/cm2)
    PHOTON_F_LAMBDA = "photon_F_lambda"  # photons/s/cm2/A
    AB_MAG = "AB_mag"                # AB magnitude per wavelength sample
    VEGA_MAG = "Vega_mag"            # Vega magnitude per wavelength sample


_FLUX_SYSTEM_LABELS = {
    FluxSystem.F_NU: "F_ν (per unit frequency, e.g. Jy)",
    FluxSystem.F_LAMBDA: "F_λ (per unit wavelength, e.g. erg/s/cm²/Å)",
    FluxSystem.LAMBDA_F_LAMBDA: "λF_λ = νF_ν (e.g. erg/s/cm²)",
    FluxSystem.PHOTON_F_LAMBDA: "Photon F_λ (photons/s/cm²/Å)",
    FluxSystem.AB_MAG: "AB magnitude",
    FluxSystem.VEGA_MAG: "Vega magnitude",
}


def flux_system_label(system: FluxSystem | str) -> str:
    return _FLUX_SYSTEM_LABELS[FluxSystem(system)]


def _to_fnu_shape(wavelength_um: np.ndarray, flux: np.ndarray, system: FluxSystem) -> np.ndarray:
    if system is FluxSystem.F_NU:
        return flux
    if system is FluxSystem.F_LAMBDA:
        return flux * wavelength_um**2  # f_nu = lambda^2 f_lambda / c
    if system in (FluxSystem.LAMBDA_F_LAMBDA, FluxSystem.PHOTON_F_LAMBDA):
        # f_nu = lambda * (lambda f_lambda) / c ; f_lambda = N_lambda * h c / lambda
        return flux * wavelength_um
    if system is FluxSystem.AB_MAG:
        return 10.0 ** (-0.4 * flux)
    if system is FluxSystem.VEGA_MAG:
        raise NotImplementedError(
            "Vega-magnitude spectra are not yet supported: a reference Vega spectrum "
            "has not been added to the calculator. Convert the spectrum to AB, F_nu, "
            "or F_lambda first."
        )
    raise ValueError(f"Unknown flux system {system!r}")


class TabulatedSED:
    """User-supplied spectrum with arbitrary normalization.

    Parameters
    ----------
    wavelength, flux
        1-D arrays. Order does not matter; rows with non-finite values are
        dropped with a warning.
    wavelength_unit
        Any astropy spectral unit: a wavelength (``"Angstrom"``, ``"nm"``,
        ``"um"``), frequency (``"Hz"``, ``"GHz"``), or energy (``"keV"``).
    flux_system
        A :class:`FluxSystem` value describing the flux column convention.
    """

    def __init__(
        self,
        wavelength,
        flux,
        wavelength_unit: str | u.Unit,
        flux_system: FluxSystem | str,
        name: str = "uploaded spectrum",
    ):
        self.name = name
        self.flux_system = FluxSystem(flux_system)
        self.wavelength_unit = u.Unit(wavelength_unit)
        if self.wavelength_unit.physical_type not in ("length", "frequency", "energy", "wavenumber"):
            raise ValueError(f"{wavelength_unit!r} is not a wavelength, frequency, or energy unit")

        x = np.asarray(wavelength, dtype=float)
        y = np.asarray(flux, dtype=float)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("Spectrum wavelength and flux must be matching 1-D arrays")
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.all():
            warnings.warn(f"Dropped {np.count_nonzero(~finite)} non-finite spectrum rows", stacklevel=2)
            x, y = x[finite], y[finite]
        if np.any(x <= 0):
            raise ValueError("Spectral axis values must be positive")

        wave_um = (x * self.wavelength_unit).to_value(u.um, equivalencies=u.spectral())
        order = np.argsort(wave_um)
        wave_um, y = wave_um[order], y[order]
        if len(wave_um) < 2:
            raise ValueError("Spectrum needs at least two valid samples")
        if np.any(np.diff(wave_um) <= 0):
            raise ValueError("Spectrum contains duplicate wavelengths")

        self.wavelength_um = wave_um
        self.flux = y
        self._fnu = _to_fnu_shape(wave_um, y, self.flux_system)

    @classmethod
    def from_file(cls, source, wavelength_unit, flux_system, columns=(0, 1), name=None):
        """Read a spectrum file; see :func:`read_spectrum_file` for supported
        formats. ``source`` is a path, a string of file contents, or bytes.
        ``columns`` are 0-based indices (or header names) of the spectral-axis
        and flux columns; any other columns are ignored."""
        spec = read_spectrum_file(source, columns=columns)
        return cls(spec.wavelength, spec.flux, wavelength_unit, flux_system, name=name or spec.name)

    def fnu_shape(self, wavelength_um: np.ndarray) -> np.ndarray:
        """f_nu shape interpolated onto ``wavelength_um`` (no extrapolation)."""
        wave = np.asarray(wavelength_um, dtype=float)
        if wave.min() < self.wavelength_um[0] or wave.max() > self.wavelength_um[-1]:
            raise ValueError("Requested wavelengths fall outside the uploaded spectrum")
        return np.interp(wave, self.wavelength_um, self._fnu)

    def flux_njy(
        self,
        target: Bandpass,
        normalization_bandpass: Bandpass,
        normalization_ab_magnitude: float,
    ) -> float:
        return transfer_ab_normalization(
            self.wavelength_um, self._fnu, target, normalization_bandpass, normalization_ab_magnitude
        )


# --- Spectrum file reading ---------------------------------------------------

#: Extensions suggested in upload dialogs. Any plain-text table is accepted
#: regardless of extension; FITS is recognised from the file contents.
SPECTRUM_FILE_EXTENSIONS = (
    ".dat", ".txt", ".flm", ".csv", ".tsv", ".ascii", ".asc", ".spec", ".spc",
    ".tab", ".tbl", ".ipac", ".sed", ".out", ".fits", ".fit", ".fts",
)

_COMMENT_PREFIXES = ("#", "%", "!", "//", ";", "\\", "|")
_SPLIT = re.compile(r"[\s,;|]+")
_FORTRAN_EXP = re.compile(r"(?<=[0-9.])[dD](?=[+-]?[0-9])")


@dataclass
class SpectrumFile:
    """Spectral-axis and flux columns of a spectrum file, as read (no units)."""

    wavelength: np.ndarray
    flux: np.ndarray
    name: str = "uploaded spectrum"
    n_columns: int = 2
    column_names: list[str] = field(default_factory=list)
    n_skipped_lines: int = 0
    format: str = "text"


def _to_float(token: str) -> float:
    try:
        return float(_FORTRAN_EXP.sub("e", token))
    except ValueError:
        return np.nan if token.lower() in ("nan", "--", "null", "none", "inf", "-inf") else None


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _column_index(column, header: list[str]) -> int:
    if isinstance(column, (int, np.integer)):
        return int(column)
    lowered = [h.lower() for h in header]
    if str(column).lower() not in lowered:
        raise ValueError(f"Column {column!r} not found in header {header}")
    return lowered.index(str(column).lower())


def _read_text_table(text: str, columns, name: str) -> SpectrumFile:
    header: list[str] = []
    rows: list[list[float]] = []
    skipped = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_comment = line.startswith(_COMMENT_PREFIXES)
        if not is_comment:
            line = line.split("#", 1)[0]  # inline comment
        tokens = [t for t in _SPLIT.split(line.strip("#%!/;\\|, \t")) if t]
        values = [_to_float(t) for t in tokens]
        if is_comment or len(values) < 2 or values[0] is None or values[1] is None:
            # Comment, header row, or other text. Text just before the data is
            # kept as candidate column names (only needed for name lookups).
            if not rows and len(tokens) >= 2:
                header = tokens
            skipped += 1
            continue
        rows.append([np.nan if v is None else v for v in values])

    if len(rows) < 2:
        raise ValueError(
            "Could not find at least two numeric rows. The file should be a table whose "
            "first two columns are wavelength (or frequency) and flux."
        )
    n_columns = max(len(r) for r in rows)
    iw, iflux = (_column_index(c, header) for c in columns)
    if max(iw, iflux) >= n_columns:
        raise ValueError(f"Spectrum file has only {n_columns} columns")
    wave = np.array([r[iw] if iw < len(r) else np.nan for r in rows], dtype=float)
    flux = np.array([r[iflux] if iflux < len(r) else np.nan for r in rows], dtype=float)
    names = header if len(header) == n_columns else []
    return SpectrumFile(wave, flux, name, n_columns, names, skipped, "text table")


def _read_fits(raw: bytes, columns, name: str) -> SpectrumFile:
    from astropy.io import fits

    with fits.open(io.BytesIO(raw), memmap=False) as hdul:
        for hdu in hdul:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) and hdu.data is not None:
                names = list(hdu.columns.names)
                if len(names) < 2:
                    continue
                wave, flux = (
                    np.asarray(hdu.data[names[_column_index(c, names)]], dtype=float).ravel()
                    for c in columns
                )
                return SpectrumFile(wave, flux, name, len(names), names, 0, "FITS table")
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None or not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                continue
            data = np.asarray(data, dtype=float)
            if data.ndim == 2 and data.shape[0] == 2:  # rows: wavelength, flux
                return SpectrumFile(data[0], data[1], name, 2, [], 0, "FITS image (2 x N)")
            flux = np.squeeze(data)
            if flux.ndim != 1:
                flux = flux.reshape(-1, flux.shape[-1])[0]  # e.g. IRAF multispec: first band
            hdr = hdu.header
            step = hdr.get("CDELT1", hdr.get("CD1_1"))
            if step is None or "CRVAL1" not in hdr:
                raise ValueError("1-D FITS spectrum lacks a CRVAL1/CDELT1 wavelength solution")
            wave = hdr["CRVAL1"] + step * (np.arange(flux.size) + 1.0 - hdr.get("CRPIX1", 1.0))
            if hdr.get("DC-FLAG", 0) == 1 or "LOG" in str(hdr.get("CTYPE1", "")).upper():
                wave = 10.0**wave
            return SpectrumFile(wave, flux, name, 2, [], 0, "FITS 1-D image")
    raise ValueError("No spectrum table or 1-D image found in FITS file")


def read_spectrum_file(source, columns=(0, 1), name: str | None = None) -> SpectrumFile:
    """Read the spectral axis and flux from a spectrum file.

    Supported inputs:

    * Plain-text tables with any extension (``.dat``, ``.txt``, ``.flm``,
      ``.csv``, ``.tsv``, ``.ascii``, ``.spec``, IPAC ``.tbl`` ...), delimited by
      whitespace, tabs, commas, semicolons, or ``|``. Comment lines starting
      with ``#``, ``%``, ``!``, ``//``, ``;``, ``\\`` or ``|``, header rows, and
      other non-numeric lines are skipped. Fortran ``D`` exponents are accepted.
    * FITS (detected from content): the first table HDU, a ``2 x N`` image,
      or a 1-D image with a linear or log-linear ``CRVAL1/CDELT1`` solution.

    By default column 1 is the spectral axis and column 2 the flux; any
    further columns (errors, sky, ...) are ignored. ``source`` is a path, a
    string of file contents, or bytes.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw, default_name = bytes(source), "uploaded spectrum"
    elif isinstance(source, Path) or (isinstance(source, str) and "\n" not in source):
        raw, default_name = Path(source).read_bytes(), Path(source).name
    else:
        raw, default_name = None, "uploaded spectrum"
    name = name or default_name
    if raw is not None and raw[:9] == b"SIMPLE  =":
        return _read_fits(raw, columns, name)
    return _read_text_table(_decode(raw) if raw is not None else source, columns, name)
