# ACROSS sensitivity-calculator prototype

An interactive imaging sensitivity prototype for JWST/NIRCam and HST/WFC3
(UVIS and IR channels), based on published STScI reference limits and
total-system throughput curves. In the notebook you can select several
instruments and several filters at once; the results are tabulated for each
selected filter and plotted together on the source spectrum.

## Run it

Launch the notebook from the environment in which it was installed:

```bash
conda activate <environment>
jupyter lab notebooks/jwst_nircam_sensitivity_prototype.ipynb
```

Run the tests with:

```bash
conda run -n <environment> pytest -q
```

## Structure

- `notebooks/`: interactive prototype and worked examples.
- `sensitivity_calculator/`: reusable photometry, SED, instrument-loading, and
  sensitivity-scaling modules.
- `data/instruments/`: versioned, declarative instrument metadata and
  sensitivity references, one JSON file per instrument (`jwst_nircam.json`,
  `hst_wfc3_uvis.json`, `hst_wfc3_ir.json`).
- `data/throughputs/`: official STScI total-system throughput tables (NIRCam
  at the top level, WFC3 in `hst_wfc3/`).
- `tests/`: reference reproduction, pivot-wavelength, and round-trip checks.

The current implementation supports direct AB magnitudes, stellar populations, blackbodies, and
user-uploaded spectra normalized by an AB magnitude in any filter of any
instrument. 

`load_instruments()` returns all instruments keyed by label
(`"JWST/NIRCam"`, `"HST/WFC3/UVIS"`, `"HST/WFC3/IR"`), and
`sensitivity_table(instruments, [(label, filter), ...], flux, exposure_s,
target_snr)` evaluates S/N and required time for many filters at once.

Uploaded spectra (`TabulatedSED`, read by `read_spectrum_file`) may be any
text table (`.dat`, `.txt`, `.flm`, `.csv`, `.tsv`, `.ascii`, IPAC `.tbl`, ...;
whitespace/tab/comma/semicolon delimited, header and comment lines skipped) or
a FITS spectrum. Column 1 is the wavelength and column 2 the flux; extra
columns are ignored. The notebook plots the spectrum as soon as it is uploaded.

Stellar-population templates come from `fsps_template_library/` (FSPS SSPs,
0.1–14 Gyr, [Z/H] = −0.5/0/+0.2) via `load_fsps_library()`, and can be
redshifted; any directory of spectra with a `manifest.json` can be loaded as a
`TemplateLibrary`. 

For every spectral source (blackbody, stellar population, upload) the
notebook plots the normalized spectrum with the normalization and observing
filters (`sensitivity_calculator/plotting.py`).
The normalization is arbitrary; the user states the spectral-axis unit (Å, nm, µm, Hz, ...) and
the flux convention (F_nu, F_lambda, lambda*F_lambda, photon F_lambda, or AB
magnitudes). 

Vega-magnitude inputs are not yet supported. 

New SED families can
implement the `SED` protocol in `sensitivity_calculator/seds.py`. Additional
instruments need only a JSON file in `data/instruments/` plus their throughput
tables, and an entry in `INSTRUMENT_CONFIGS`
(`sensitivity_calculator/instruments.py`); references may be given in nJy or
AB magnitudes, and throughput wavelengths in any length unit.

The approximation assumes background-dominated behavior, so S/N scales as
source flux times the square root of integration time. 

**Use the official JWST
or HST ETC for observation planning and configurations outside the published
reference assumptions.**
