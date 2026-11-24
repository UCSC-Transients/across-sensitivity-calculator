# Prototype reference data

## JWST/NIRCam

`throughputs/` contains the 29 detector-averaged JWST + NIRCam total-system
throughput tables linked individually from the STScI NIRCam Filters page. The
curves include the Optical Telescope Element, NIRCam optics, dichroics, filters,
and detector quantum efficiency. For pupil-wheel filters, the required blocking
filter is included. Version: 7.0 (January 2025), downloaded 2026-10-20.

`instruments/jwst_nircam.json` contains the published ETC v5.0 point-source
imaging sensitivity limits and all scaling assumptions used by the prototype.
Flux density in nJy is treated as canonical because the displayed AB magnitudes
in the source table are rounded.

Sources:

- https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
- https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-sensitivity

These files are scientific reference inputs, not outputs of the calculator.
When they are updated, change the version/provenance fields and rerun the tests,
especially the pivot-wavelength validation.

The F150W2 source table contains two tiny negative throughput samples (minimum
about -3.4e-4). The raw table is preserved unchanged; the instrument loader
clips negative samples to zero before synthetic-photometry integration.


## HST/WFC3

`throughputs/hst_wfc3/` contains, unchanged, the STScI WFC3 total-system
throughput tables (wavelength in Angstrom, throughput) for the seven filters
that have a published point-source limit: UVIS F225W, F336W, F438W, F606W,
F814W (`wfc3_uvis1_*.txt`) and IR F110W, F160W (`wfc3_ir_*.txt`). They come
from the `UVIS.zip` and `IR.zip` archives on the STScI WFC3 Throughputs page,
computed with stsynphot at the 2009-06-26 reference epoch with the October
2020 photometric calibration; downloaded 2026-11-24. UVIS chip 1 curves are used
because the handbook's pivot wavelengths (Table 6.2) are for chip 1; the
stored curves reproduce them to 0.1 Angstrom. UVIS2 differs by less than ~1%
above ~4500 Angstrom and is up to ~30% more sensitive in the far-UV.

`instruments/hst_wfc3_uvis.json` and `instruments/hst_wfc3_ir.json` hold the
WFC3 Instrument Handbook (Cycle 34) limiting AB magnitudes for a point source
with optimal extraction at S/N = 10 in 1 hr (Tables 6.10 and 7.10; the
1-hr value is the reference, the 10-hr value is kept for validation). Only
these seven filters have published limits, so only they are offered; more
filters can be added with their own reference values (for example from the
HST ETC).

Sources:

- https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs
- https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-8-uvis-sensitivity
- https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/7-8-ir-sensitivity
- https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-5-uvis-spectral-elements (pivots)
