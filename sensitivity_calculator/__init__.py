"""Approximate, reference-scaled telescope sensitivity calculations."""

from .calculator import SensitivityCalculator, sensitivity_table
from .instruments import (
    INSTRUMENT_CONFIGS,
    load_hst_wfc3_ir,
    load_hst_wfc3_uvis,
    load_instrument,
    load_instruments,
    load_jwst_nircam,
)
from .models import CalculationResult
from .seds import BlackbodySED, FluxSystem, TabulatedSED, read_spectrum_file
from .templates import TemplateLibrary, load_fsps_library

__all__ = [
    "BlackbodySED",
    "CalculationResult",
    "FluxSystem",
    "INSTRUMENT_CONFIGS",
    "SensitivityCalculator",
    "TabulatedSED",
    "TemplateLibrary",
    "load_fsps_library",
    "load_hst_wfc3_ir",
    "load_hst_wfc3_uvis",
    "load_instrument",
    "load_instruments",
    "load_jwst_nircam",
    "read_spectrum_file",
    "sensitivity_table",
]
