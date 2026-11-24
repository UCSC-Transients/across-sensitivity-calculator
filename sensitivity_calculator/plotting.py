"""Plots of source spectra against instrument bandpasses."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from .models import Bandpass
from .photometry import ab_magnitude_to_njy, njy_to_ab_magnitude, photon_weighted_mean_fnu
from .seds import TabulatedSED

#: Scientific-figure style: serif fonts, large labels, inward ticks on all sides.
PLOT_STYLE = {
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 3,
}


def _log_label(value, _pos=None) -> str:
    """0.1, 1, 10 as plain numbers; other powers of ten as 10^n."""
    if value <= 0:
        return ""
    exponent = np.log10(value)
    if not np.isclose(exponent, round(exponent)):
        return ""
    exponent = int(round(exponent))
    return {-1: "0.1", 0: "1", 1: "10"}.get(exponent, rf"$10^{{{exponent}}}$")


def use_log_ticks(axis) -> None:
    """Power-of-ten labels and no minor ticks on a log-scaled axis."""
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(_log_label))
    axis.set_minor_locator(NullLocator())


def sample_sed(sed, lo_um: float, hi_um: float, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength grid (µm) and f_nu shape of any SED between ``lo_um`` and ``hi_um``.

    Tabulated spectra are returned at their native samples (clipped to their
    coverage); analytic SEDs are evaluated on a log-spaced grid.
    """
    if isinstance(sed, TabulatedSED):
        lo_um, hi_um = max(lo_um, sed.wavelength_um[0]), min(hi_um, sed.wavelength_um[-1])
        wave = sed.wavelength_um[(sed.wavelength_um >= lo_um) & (sed.wavelength_um <= hi_um)]
        wave = np.unique(np.concatenate([[lo_um], wave, [hi_um]]))
    else:
        wave = np.geomspace(lo_um, hi_um, n)
    return wave, sed.fnu_shape(wave)


def plot_sed_with_bandpasses(
    sed,
    target: Bandpass | Sequence[Bandpass],
    normalization: Bandpass,
    normalization_ab_magnitude: float,
    source_flux_njy: float | Sequence[float],
    wavelength_range_um: tuple[float, float] | None = None,
    label: str | None = None,
    figsize=(10, 4.2),
    target_colors: Sequence[str] | None = None,
    key: Sequence[tuple[str, str]] | None = None,
):
    """Plot the AB-normalized f_nu of ``sed`` with the normalization (blue) and
    observing bandpasses shaded and their band-averaged fluxes marked.

    ``target`` and ``source_flux_njy`` may be single values or matching
    sequences (one observing filter per entry, e.g. from several instruments);
    ``target_colors`` sets one color per target (default red). ``key`` is an
    optional list of ``(text, color)`` pairs written inside the panel to
    identify the colors (e.g. one entry per instrument), in place of a legend.

    Works for any object with ``fnu_shape`` (blackbody, template, uploaded spectrum).
    ``wavelength_range_um`` defaults to the span of all bandpasses, padded.
    """
    targets = [target] if isinstance(target, Bandpass) else list(target)
    fluxes = np.atleast_1d(np.asarray(source_flux_njy, dtype=float)).tolist()
    if len(fluxes) != len(targets):
        raise ValueError("Give one source flux per observing bandpass")
    colors = list(target_colors) if target_colors is not None else ["tab:red"] * len(targets)
    if len(colors) != len(targets):
        raise ValueError("Give one color per observing bandpass")

    bands = (normalization, *targets)
    if wavelength_range_um is None:
        lo = min(active_range(b)[0] for b in bands) * 0.8
        hi = max(active_range(b)[1] for b in bands) * 1.2
    else:
        lo, hi = wavelength_range_um
    # Normalize on a grid that covers the normalization band.
    nlo, nhi = normalization.wavelength_um.min(), normalization.wavelength_um.max()
    norm_wave, norm_fnu = sample_sed(sed, min(lo, nlo), max(hi, nhi))
    scale = ab_magnitude_to_njy(normalization_ab_magnitude) / photon_weighted_mean_fnu(
        normalization, norm_wave, norm_fnu
    )
    wave, fnu = sample_sed(sed, lo, hi)
    fnu_njy = fnu * scale

    points = [(normalization, ab_magnitude_to_njy(normalization_ab_magnitude), "tab:blue")]
    points += [(bp, f, c) for bp, f, c in zip(targets, fluxes, colors) if np.isfinite(f)]
    # Stagger the filter-name labels in up to three rows when they would collide.
    order = sorted(range(len(points)), key=lambda k: points[k][0].pivot_um)
    rows = {k: 0 for k in order}
    span = hi - lo
    for prev, k in zip(order, order[1:]):
        if points[k][0].pivot_um - points[prev][0].pivot_um < 0.08 * span:
            rows[k] = (rows[prev] + 1) % 3
    n_rows = max(rows.values()) + 1
    annotate_mag = len(points) <= 4

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(wave, fnu_njy, lw=1, color="k")
        for k, (bp, flux, color) in enumerate(points):
            ax.fill_between(bp.wavelength_um, 0, bp.throughput / bp.throughput.max(),
                            transform=ax.get_xaxis_transform(), color=color, alpha=0.15, lw=0)
            ax.text(bp.pivot_um, 0.96 - 0.075 * rows[k], bp.name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=color, fontsize=13 if n_rows == 1 else 11)
            ax.plot(bp.pivot_um, flux, "o", color=color, ms=7, mec="k", mew=0.6, zorder=5)
            if annotate_mag:
                ax.annotate(f"AB {njy_to_ab_magnitude(flux):.2f}", (bp.pivot_um, flux),
                            xytext=(8, -14), textcoords="offset points", color=color, fontsize=12)
        # y-limits from the curve and all markers, with headroom for the filter labels.
        values = np.concatenate([fnu_njy, [p[1] for p in points]])
        positive = values[values > 0]
        headroom = 0.3 + 0.12 * (n_rows - 1)
        if positive.size == values.size and positive.max() / positive.min() > 30:
            ax.set_yscale("log")
            use_log_ticks(ax.yaxis)
            ylo, yhi = positive.min(), positive.max()
            ax.set_ylim(ylo / 1.5, yhi * (yhi / ylo) ** headroom)
        else:
            ylo, yhi = values.min(), values.max()
            bottom = 0.0 if ylo >= 0 else ylo - 0.1 * (yhi - ylo)
            ax.set_ylim(bottom, yhi + headroom * (yhi - bottom))  # starting at 0 keeps near-flat SEDs flat
            ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlim(wave[0], wave[-1])
        ax.set_xlabel("Observed wavelength (µm)")
        ax.set_ylabel(r"$F_\nu$ (nJy)")
        ax.set_title(label or getattr(sed, "name", type(sed).__name__), fontsize=14, loc="right")
        if key:
            # Write the key in whichever lower corner the spectrum leaves emptiest.
            xy = ax.transAxes.inverted().transform(ax.transData.transform(np.column_stack([wave, fnu_njy])))
            height = 0.04 + 0.075 * len(key)
            low = xy[:, 1] < height + 0.03
            left = np.sum(low & (xy[:, 0] < 0.4)) / max(np.sum(xy[:, 0] < 0.4), 1)
            right = np.sum(low & (xy[:, 0] > 0.6)) / max(np.sum(xy[:, 0] > 0.6), 1)
            x, ha = (0.015, "left") if left <= right else (0.985, "right")
            for j, (text, color) in enumerate(reversed(list(key))):
                ax.text(x, 0.04 + 0.075 * j, text, transform=ax.transAxes,
                        ha=ha, va="bottom", color=color, fontsize=12)
        fig.tight_layout()
    return fig


def active_range(bandpass: Bandpass, fraction: float = 0.01) -> tuple[float, float]:
    """Wavelength span (µm) where the throughput exceeds ``fraction`` of its peak."""
    active = bandpass.throughput > fraction * bandpass.throughput.max()
    return float(bandpass.wavelength_um[active].min()), float(bandpass.wavelength_um[active].max())


def plot_throughputs(instruments, colors=None, figsize_per_panel=(11, 3.0)):
    """One panel per instrument with every filter curve, names written at the peaks."""
    instruments = list(instruments.values()) if isinstance(instruments, dict) else list(instruments)
    colors = colors or {}
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(instruments), 1, squeeze=False,
                                 figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(instruments)))
        for ax, inst in zip(axes[:, 0], instruments):
            color = colors.get(inst.label, "k")
            lo = min(active_range(bp)[0] for bp in inst.bandpasses.values())
            hi = max(active_range(bp)[1] for bp in inst.bandpasses.values())
            peak = max(bp.throughput.max() for bp in inst.bandpasses.values())
            # Filter names above each curve, raised in steps where neighbours would overlap.
            level, previous = 0, -np.inf
            for bp in sorted(inst.bandpasses.values(), key=lambda b: b.pivot_um):
                ax.plot(bp.wavelength_um, bp.throughput, lw=1, color=color)
                level = (level + 1) % 3 if bp.pivot_um - previous < 0.025 * (hi - lo) else 0
                previous = bp.pivot_um
                ax.text(bp.pivot_um, bp.throughput.max() + (0.03 + 0.2 * level) * peak, bp.name, rotation=90,
                        ha="center", va="bottom", fontsize=9, color=color)
            ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_ylim(0, peak * 1.9)
            ax.text(0.99, 0.95, inst.label, transform=ax.transAxes, ha="right", va="top",
                    fontsize=14, color=color)
            ax.set_ylabel("Throughput")
        axes[-1, 0].set_xlabel("Wavelength (µm)")
        fig.tight_layout()
    return fig
