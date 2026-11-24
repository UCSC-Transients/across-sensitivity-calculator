import numpy as np
import pytest

from sensitivity_calculator import BlackbodySED, SensitivityCalculator, load_jwst_nircam
from sensitivity_calculator.photometry import ab_magnitude_to_njy, njy_to_ab_magnitude


@pytest.fixture(scope="module")
def instrument():
    return load_jwst_nircam()


@pytest.fixture(scope="module")
def calculator(instrument):
    return SensitivityCalculator(instrument)


def test_all_nircam_imaging_filters_have_curves_and_sensitivities(instrument):
    assert len(instrument.filters) == 29
    assert set(instrument.bandpasses) == set(instrument.sensitivities)
    for name, bandpass in instrument.bandpasses.items():
        assert bandpass.wavelength_um.size > 100
        assert np.all(np.diff(bandpass.wavelength_um) > 0)
        assert np.all(bandpass.throughput >= 0)
        assert bandpass.throughput.max() <= 1
        assert instrument.sensitivities[name].flux_njy > 0


def test_stored_curves_reproduce_published_pivot_wavelengths(instrument):
    for bandpass in instrument.bandpasses.values():
        assert bandpass.pivot_um == pytest.approx(bandpass.pivot_um_published, rel=7e-4)


@pytest.mark.parametrize("filter_name", ["F070W", "F150W2", "F323N", "F444W", "F480M"])
def test_published_reference_is_reproduced(calculator, instrument, filter_name):
    ref = instrument.sensitivities[filter_name]
    result = calculator.signal_to_noise(filter_name, ref.flux_njy, ref.exposure_s)
    assert result.value == pytest.approx(ref.snr)


@pytest.mark.parametrize("flux_njy,exposure_s", [(5.0, 100.0), (50.0, 10_000.0), (500.0, 1234.5)])
def test_snr_exposure_round_trip(calculator, flux_njy, exposure_s):
    snr = calculator.signal_to_noise("F200W", flux_njy, exposure_s).value
    recovered = calculator.exposure_time("F200W", flux_njy, snr).value
    assert recovered == pytest.approx(exposure_s)


def test_blackbody_normalization_is_exact_in_same_filter(instrument):
    magnitude = 24.3
    bandpass = instrument.bandpasses["F356W"]
    result = BlackbodySED(6000).flux_njy(bandpass, bandpass, magnitude)
    assert result == pytest.approx(ab_magnitude_to_njy(magnitude), rel=1e-12)


def test_ab_flux_conversion_round_trip():
    assert njy_to_ab_magnitude(ab_magnitude_to_njy(27.25)) == pytest.approx(27.25)


@pytest.mark.parametrize("bad", [0, -1, np.inf, np.nan])
def test_invalid_physical_inputs_are_rejected(calculator, bad):
    with pytest.raises(ValueError):
        calculator.signal_to_noise("F200W", bad, 1000)
    with pytest.raises(ValueError):
        calculator.exposure_time("F200W", 10, bad)



# --- Uploaded spectra -------------------------------------------------------

from astropy import constants as const  # noqa: E402

from sensitivity_calculator import FluxSystem, TabulatedSED  # noqa: E402


def _blackbody_spectrum(temperature_k=5000.0):
    """The same blackbody expressed in every supported flux convention."""
    wave_um = np.geomspace(0.5, 6.0, 4000)
    fnu = BlackbodySED(temperature_k).fnu_shape(wave_um)
    flam = fnu / wave_um**2
    return wave_um, {
        FluxSystem.F_NU: fnu * 1e23,             # arbitrary constant factors
        FluxSystem.F_LAMBDA: flam * 3.7e-11,
        FluxSystem.LAMBDA_F_LAMBDA: flam * wave_um,
        FluxSystem.PHOTON_F_LAMBDA: flam * wave_um / (const.h.value * const.c.value),
        FluxSystem.AB_MAG: -2.5 * np.log10(fnu / fnu.max()) + 20.0,
    }


@pytest.mark.parametrize("system", [s for s in FluxSystem if s is not FluxSystem.VEGA_MAG])
def test_uploaded_spectrum_matches_blackbody_in_every_flux_system(instrument, system):
    wave_um, fluxes = _blackbody_spectrum()
    target, norm = instrument.bandpasses["F444W"], instrument.bandpasses["F090W"]
    expected = BlackbodySED(5000.0).flux_njy(target, norm, 23.0)
    sed = TabulatedSED(wave_um * 1e4, fluxes[system], "Angstrom", system)
    assert sed.flux_njy(target, norm, 23.0) == pytest.approx(expected, rel=1e-4)


def test_uploaded_spectrum_accepts_frequency_axis_in_any_order(instrument):
    wave_um, fluxes = _blackbody_spectrum()
    freq_ghz = (const.c.value / (wave_um * 1e-6)) / 1e9  # descending wavelength order
    target, norm = instrument.bandpasses["F200W"], instrument.bandpasses["F410M"]
    a = TabulatedSED(freq_ghz, fluxes[FluxSystem.F_NU], "GHz", "F_nu").flux_njy(target, norm, 25.0)
    b = TabulatedSED(wave_um, fluxes[FluxSystem.F_NU], "um", "F_nu").flux_njy(target, norm, 25.0)
    assert a == pytest.approx(b, rel=1e-10)


def test_flat_fnu_spectrum_has_zero_ab_color(instrument):
    sed = TabulatedSED([0.3, 6.0], [7.0, 7.0], "um", "F_nu")
    flux = sed.flux_njy(instrument.bandpasses["F070W"], instrument.bandpasses["F480M"], 24.0)
    assert flux == pytest.approx(ab_magnitude_to_njy(24.0), rel=1e-12)


def test_vega_spectrum_is_rejected_clearly():
    with pytest.raises(NotImplementedError, match="Vega"):
        TabulatedSED([1.0, 2.0], [10.0, 10.0], "um", "Vega_mag")


def test_spectrum_must_cover_both_bands(instrument):
    sed = TabulatedSED([1.0, 2.5], [1.0, 1.0], "um", "F_nu")
    with pytest.raises(ValueError, match="does not cover the F444W"):
        sed.flux_njy(instrument.bandpasses["F444W"], instrument.bandpasses["F150W"], 24.0)


@pytest.mark.parametrize("text", [
    "# wave flux\n5000 1.0\n20000 2.0\n60000 3.0\n",
    "wavelength,flux\n5000,1.0\n20000,2.0\n60000,3.0\n",
])
def test_spectrum_file_parsing(tmp_path, text):
    path = tmp_path / "spec.txt"
    path.write_text(text)
    for source in (path, text, text.encode()):
        sed = TabulatedSED.from_file(source, "Angstrom", "F_lambda")
        assert sed.wavelength_um == pytest.approx([0.5, 2.0, 6.0])
        assert sed.flux == pytest.approx([1.0, 2.0, 3.0])


@pytest.mark.parametrize("text", [
    # WISeREP-style .flm: comment header, extra error column
    "# SN 2099zz  z=0.01\n# wave flux err\n5000 1.0 0.1\n20000 2.0 0.1\n60000 3.0 0.2\n",
    # tab-separated with a bare header and trailing extra columns
    "lambda\tflux\terr\tsky\n5000\t1.0\t0.1\t9\n20000\t2.0\t0.1\t9\n60000\t3.0\t0.2\t9\n",
    # semicolon CSV, Fortran exponents, % comments, blank lines
    "% generated\n\n5.0D+03;1.0D0;x\n2.0D+04;2.0D0;x\n\n6.0E+04;3.0;x\n",
    # IPAC table
    "\\fixlen = T\n| wave | flux | err |\n| double | double | double |\n 5000 1.0 0.1\n 20000 2.0 0.1\n 60000 3.0 0.1\n",
    # CSV with quoted-free header, trailing commas, and an inline comment
    "wavelength,flux,flux_err,\n5000,1.0,0.1,\n20000,2.0,0.1,  # peak\n60000,3.0,0.1,\n",
])
def test_spectrum_file_parsing_varied_formats(tmp_path, text):
    for suffix in (".flm", ".dat", ".csv"):
        path = tmp_path / f"spec{suffix}"
        path.write_text(text)
        sed = TabulatedSED.from_file(path, "Angstrom", "F_lambda")
        assert sed.name == path.name
        assert sed.wavelength_um == pytest.approx([0.5, 2.0, 6.0])
        assert sed.flux == pytest.approx([1.0, 2.0, 3.0])


def test_spectrum_file_latin1_and_nan_rows():
    raw = "# Température\n5000 1.0\n10000 nan\n20000 2.0\n60000 3.0\n".encode("latin-1")
    with pytest.warns(UserWarning, match="non-finite"):
        sed = TabulatedSED.from_file(raw, "Angstrom", "F_lambda")
    assert sed.flux == pytest.approx([1.0, 2.0, 3.0])


def test_spectrum_file_rejects_non_tables():
    with pytest.raises(ValueError, match="first two columns"):
        TabulatedSED.from_file(b"just some text\nwith no numbers\n", "um", "F_nu")


def test_fits_spectra(tmp_path):
    from astropy.io import fits

    from sensitivity_calculator import read_spectrum_file

    wave = np.array([5000.0, 20000.0, 60000.0])
    flux = np.array([1.0, 2.0, 3.0])
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="WAVE", format="D", array=wave),
        fits.Column(name="FLUX", format="D", array=flux),
        fits.Column(name="ERR", format="D", array=flux * 0.1),
    ])
    path = tmp_path / "spec_table.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    spec = read_spectrum_file(path)
    assert spec.wavelength == pytest.approx(wave) and spec.flux == pytest.approx(flux)
    assert spec.n_columns == 3

    image = fits.PrimaryHDU(np.array([1.0, 2.0, 3.0, 4.0]))
    image.header.update(CRVAL1=5000.0, CDELT1=1000.0, CRPIX1=1.0)
    spec = read_spectrum_file(_fits_bytes(image))  # bytes, as from an upload widget
    assert spec.wavelength == pytest.approx([5000, 6000, 7000, 8000])


def _fits_bytes(hdu):
    import io

    from astropy.io import fits

    buf = io.BytesIO()
    fits.HDUList([hdu]).writeto(buf)
    return buf.getvalue()


# --- Template libraries and plotting -----------------------------------------

from sensitivity_calculator import load_fsps_library  # noqa: E402
from sensitivity_calculator.templates import DEFAULT_FSPS_LIBRARY, TemplateLibrary, flux_system_from_unit  # noqa: E402

needs_fsps = pytest.mark.skipif(not (DEFAULT_FSPS_LIBRARY / "manifest.json").exists(),
                                reason="FSPS template library not present")


@pytest.mark.parametrize("unit, system", [
    ("L_sun / Angstrom", FluxSystem.F_LAMBDA), ("erg / (s cm2 Angstrom)", FluxSystem.F_LAMBDA),
    ("Jy", FluxSystem.F_NU), ("erg / (s Hz)", FluxSystem.F_NU), ("erg / (s cm2)", FluxSystem.LAMBDA_F_LAMBDA),
])
def test_flux_system_from_unit(unit, system):
    assert flux_system_from_unit(unit) is system


def test_custom_template_library_and_redshift(tmp_path, instrument):
    import json

    wave_um, fluxes = _blackbody_spectrum(4000.0)
    for t in (4000, 8000):
        w, f = _blackbody_spectrum(t)
        np.savetxt(tmp_path / f"bb{t}.txt", np.c_[w * 1e4, f[FluxSystem.F_LAMBDA]])
    (tmp_path / "manifest.json").write_text(json.dumps({
        "flux_units": "erg / (s cm2 Angstrom)",
        "templates": [{"file": "bb4000.txt", "teff": 4000}, {"file": "bb8000.txt", "teff": 8000}],
    }))
    lib = TemplateLibrary(tmp_path)
    assert lib.parameter_names == ("teff",) and lib.parameter_values("teff") == [4000, 8000]
    target, norm = instrument.bandpasses["F200W"], instrument.bandpasses["F150W"]
    assert lib.sed(teff=4000).flux_njy(target, norm, 24.0) == pytest.approx(
        BlackbodySED(4000).flux_njy(target, norm, 24.0), rel=1e-4)
    # A blackbody at T redshifted by z has the shape of one at T/(1+z).
    z = 1.0
    assert lib.sed(redshift=z, teff=8000).flux_njy(target, norm, 24.0) == pytest.approx(
        BlackbodySED(8000 / (1 + z)).flux_njy(target, norm, 24.0), rel=1e-3)
    with pytest.raises(ValueError, match="0 templates"):
        lib.sed(teff=5000)


@needs_fsps
def test_fsps_library_loads_and_reddens_with_age(instrument):
    lib = load_fsps_library()
    assert set(lib.parameter_names) == {"age_gyr", "logzsol"} and len(lib.templates) == 48
    target, norm = instrument.bandpasses["F444W"], instrument.bandpasses["F090W"]
    young = lib.sed(age_gyr=0.1, logzsol=0.0).flux_njy(target, norm, 24.0)
    old = lib.sed(age_gyr=10, logzsol=0.0).flux_njy(target, norm, 24.0)
    assert old > young  # older populations are redder (brighter at 4.4 µm for fixed F090W)


@needs_fsps
@pytest.mark.parametrize("make_sed", [
    lambda: BlackbodySED(6000),
    lambda: TabulatedSED([0.3, 6.0], [1.0, 1.0], "um", "F_nu"),
    lambda: load_fsps_library().sed(redshift=0.5, age_gyr=3, logzsol=0.0),
])
def test_plot_sed_with_bandpasses(instrument, make_sed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensitivity_calculator.plotting import plot_sed_with_bandpasses

    sed = make_sed()
    target, norm = instrument.bandpasses["F356W"], instrument.bandpasses["F115W"]
    flux = sed.flux_njy(target, norm, 25.0)
    fig = plot_sed_with_bandpasses(sed, target, norm, 25.0, flux, wavelength_range_um=(0.6, 5.2))
    ax = fig.axes[0]
    # The red marker sits at the predicted flux, the blue one at the normalization.
    ys = sorted(line.get_ydata()[0] for line in ax.lines if line.get_marker() == "o")
    assert ys == pytest.approx(sorted([flux, ab_magnitude_to_njy(25.0)]))
    plt.close(fig)


# --- HST/WFC3 and multi-instrument support -----------------------------------

import json

from sensitivity_calculator import INSTRUMENT_CONFIGS, load_instruments, sensitivity_table


@pytest.fixture(scope="module")
def instruments():
    return load_instruments()


def test_all_configured_instruments_load(instruments):
    assert list(instruments) == list(INSTRUMENT_CONFIGS)
    assert set(instruments["HST/WFC3/UVIS"].filters) == {"F225W", "F336W", "F438W", "F606W", "F814W"}
    assert set(instruments["HST/WFC3/IR"].filters) == {"F110W", "F160W"}
    for label, inst in instruments.items():
        assert inst.label == label
        assert set(inst.bandpasses) == set(inst.sensitivities)
        for bandpass in inst.bandpasses.values():
            assert np.all(np.diff(bandpass.wavelength_um) > 0)
            assert np.all(bandpass.throughput >= 0) and bandpass.throughput.max() <= 1


@pytest.mark.parametrize("label", ["HST/WFC3/UVIS", "HST/WFC3/IR"])
def test_wfc3_curves_reproduce_handbook_pivots(instruments, label):
    # Curves are in Angstrom on disk; a unit error would be off by 10^4.
    for bandpass in instruments[label].bandpasses.values():
        assert bandpass.pivot_um == pytest.approx(bandpass.pivot_um_published, rel=2e-4)


@pytest.mark.parametrize("label,filter_name,ab_limit", [
    ("HST/WFC3/UVIS", "F606W", 27.9), ("HST/WFC3/UVIS", "F225W", 26.7), ("HST/WFC3/IR", "F160W", 26.6),
])
def test_wfc3_published_one_hour_limit_is_reproduced(instruments, label, filter_name, ab_limit):
    calc = SensitivityCalculator(instruments[label])
    result = calc.signal_to_noise(filter_name, ab_magnitude_to_njy(ab_limit), 3600.0)
    assert result.value == pytest.approx(10.0)


@pytest.mark.parametrize("config", ["hst_wfc3_uvis", "hst_wfc3_ir"])
def test_wfc3_ten_hour_limits_agree_with_sqrt_scaling(instruments, config):
    """Background-limited scaling of the 1-hr limits predicts the published
    10-hr limits to within 0.2 mag (the scaling is slightly conservative)."""
    from sensitivity_calculator.instruments import DEFAULT_DATA_ROOT

    meta = json.loads((DEFAULT_DATA_ROOT / "instruments" / f"{config}.json").read_text())
    for item in meta["filters"]:
        predicted = item["sensitivity_ab_mag"] + 2.5 * np.log10(np.sqrt(10.0))
        assert 0.0 <= item["sensitivity_ab_mag_10hr"] - predicted < 0.2


def test_sensitivity_table_across_instruments(instruments):
    norm = instruments["JWST/NIRCam"].bandpasses["F090W"]
    sed = BlackbodySED(8000)
    selections = [("JWST/NIRCam", "F200W"), ("HST/WFC3/UVIS", "F606W"), ("HST/WFC3/IR", "F160W")]
    rows = sensitivity_table(instruments, selections, lambda bp: sed.flux_njy(bp, norm, 25.0), 3600.0, 10.0)
    assert [(r["instrument"], r["filter"]) for r in rows] == selections
    for row in rows:
        assert row["error"] is None
        calc = SensitivityCalculator(instruments[row["instrument"]])
        assert row["snr"] == pytest.approx(calc.signal_to_noise(row["filter"], row["flux_njy"], 3600.0).value)
        assert row["exposure_time_s"] == pytest.approx(calc.exposure_time(row["filter"], row["flux_njy"], 10.0).value)
    # A hot blackbody normalized at F090W is brighter in F606W than in F200W.
    assert rows[1]["flux_njy"] > rows[0]["flux_njy"]


def test_sensitivity_table_reports_per_filter_errors(instruments):
    # A spectrum covering only the near-IR cannot predict UVIS fluxes.
    sed = TabulatedSED([0.8, 6.0], [1.0, 1.0], "um", "F_nu")
    norm = instruments["JWST/NIRCam"].bandpasses["F200W"]
    rows = sensitivity_table(
        instruments, [("HST/WFC3/UVIS", "F336W"), ("JWST/NIRCam", "F444W")],
        lambda bp: sed.flux_njy(bp, norm, 25.0), 1000.0, 5.0,
    )
    assert "does not cover" in rows[0]["error"] and np.isnan(rows[0]["snr"])
    assert rows[1]["error"] is None and rows[1]["flux_njy"] == pytest.approx(ab_magnitude_to_njy(25.0))


def test_sensitivity_table_constant_flux_and_bad_inputs(instruments):
    rows = sensitivity_table(instruments, [("HST/WFC3/IR", "F110W")], ab_magnitude_to_njy(27.3), 3600.0, 10.0)
    assert rows[0]["snr"] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        sensitivity_table(instruments, [("HST/WFC3/IR", "F110W")], 10.0, -1.0, 10.0)


def test_plot_with_several_observing_filters(instruments):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensitivity_calculator.plotting import plot_sed_with_bandpasses, plot_throughputs

    sed = BlackbodySED(6000)
    norm = instruments["JWST/NIRCam"].bandpasses["F090W"]
    targets = [instruments["HST/WFC3/UVIS"].bandpasses["F336W"],
               instruments["HST/WFC3/IR"].bandpasses["F160W"],
               instruments["JWST/NIRCam"].bandpasses["F444W"]]
    fluxes = [sed.flux_njy(bp, norm, 24.0) for bp in targets]
    fig = plot_sed_with_bandpasses(sed, targets, norm, 24.0, fluxes,
                                   target_colors=["tab:purple", "tab:orange", "tab:red"],
                                   key=[("HST/WFC3", "tab:purple")])
    ys = sorted(line.get_ydata()[0] for line in fig.axes[0].lines if line.get_marker() == "o")
    assert ys == pytest.approx(sorted(fluxes + [ab_magnitude_to_njy(24.0)]))
    lo, hi = fig.axes[0].get_xlim()
    assert lo < 0.3 and hi > 4.5  # default range spans every band
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_sed_with_bandpasses(sed, targets, norm, 24.0, fluxes[:2])
    plt.close(plot_throughputs(instruments))
