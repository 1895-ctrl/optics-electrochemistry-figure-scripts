#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI comparison of iSCAT and dark-field Mie signals.

The calculation places both modalities on a common dimensionless scale: the
detected particle signal is normalized to the reflected reference background
within one effective image area.  Dark-field is q^2, where q is the collected
scattered-to-reference field magnitude.  A single-phase iSCAT camera measures
2 q cos(phi) + q^2, while a phase-optimized or quadrature-recovered envelope is
2 q + q^2.

At the default net phase phi=90 degrees, the single-phase interference term is
zero.  Consequently, the single-phase iSCAT trace overlaps dark-field.  The
separate quadrature/phase-optimized envelope is retained to show the field-
linear small-particle advantage without incorrectly assigning it to a 90-degree
single-channel intensity measurement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex, to_rgb
import numpy as np

try:
    import miepython
except ImportError as exc:
    raise SystemExit(
        "miepython is required. Install it with: python -m pip install miepython"
    ) from exc


MODEL_OPTIONS = (
    "Rayleigh",
    "Dipole Mie (a1, b1)",
    "Full Mie (multipole)",
)
DEFAULT_MODEL = MODEL_OPTIONS[-1]
DEFAULT_WAVELENGTHS_NM = (533.0, 632.0)
DEFAULT_REFERENCE_INTENSITY_RATIO = 0.1
DEFAULT_CAMERA_SENSITIVITY_PERCENT = 0.1
CAMERA_SENSITIVITY_PRESETS = {
    "Kukura 2017: shot-noise SNR=10 benchmark (0.1%)": 0.1,
    "Piliarik 2014: BSA demonstrated contrast (0.03%)": 0.03,
    "Custom value": None,
}
DEFAULT_CAMERA_SENSITIVITY_SOURCE = next(iter(CAMERA_SENSITIVITY_PRESETS))
DEFAULT_DIAMETER_MIN_NM = 10.0
DEFAULT_DIAMETER_MAX_NM = 1000.0
DEFAULT_DIAMETER_POINTS = 1600
OUTPUT_STEM = "iscat_darkfield_mie_comparison"
SERIES_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7A3E9D", "#6B6B6B")

# Review-level wavelength-specific optical constants.  The first three presets
# are homogeneous spheres that map naturally onto Mie theory.  Protein- and
# virus-like entries are explicitly illustrative effective-index spheres.
PARTICLE_PRESETS = {
    "Polystyrene bead": {
        532.0: 1.598 + 0.0j,
        533.0: 1.598 + 0.0j,
        632.0: 1.588 + 0.0j,
        633.0: 1.588 + 0.0j,
    },
    "Silica bead": {
        532.0: 1.461 + 0.0j,
        533.0: 1.461 + 0.0j,
        632.0: 1.457 + 0.0j,
        633.0: 1.457 + 0.0j,
    },
    "Gold nanoparticle": {
        532.0: 0.543 + 2.230j,
        533.0: 0.543 + 2.230j,
        632.0: 0.184 + 3.430j,
        633.0: 0.184 + 3.430j,
    },
    "Protein-like dielectric (illustrative)": {
        532.0: 1.450 + 0.0j,
        533.0: 1.450 + 0.0j,
        632.0: 1.450 + 0.0j,
        633.0: 1.450 + 0.0j,
    },
    "Virus-like dielectric (illustrative)": {
        532.0: 1.480 + 0.0j,
        533.0: 1.480 + 0.0j,
        632.0: 1.480 + 0.0j,
        633.0: 1.480 + 0.0j,
    },
}
DEFAULT_PARTICLES = (
    "Polystyrene bead",
    "Silica bead",
    "Gold nanoparticle",
)


@dataclass(frozen=True)
class ComparisonSystem:
    wavelength_vacuum_nm: float = 533.0
    n_particle: complex = 1.598 + 0.0j
    n_medium: float = 1.333
    n_glass: float = 1.518
    numerical_aperture: float = 1.42
    collection_efficiency: float = 0.30
    reference_intensity_ratio: float = DEFAULT_REFERENCE_INTENSITY_RATIO
    net_phase_deg: float = 90.0


def configure_tk_runtime():
    """Use the Tcl/Tk bundled with the active Python installation if present."""
    import os
    import sys

    base = Path(sys.base_prefix)
    tcl = base / "tcl"
    if not tcl.exists():
        return
    tcl_dirs = sorted(tcl.glob("tcl8.*"))
    tk_dirs = sorted(tcl.glob("tk8.*"))
    if tcl_dirs:
        os.environ.setdefault("TCL_LIBRARY", str(tcl_dirs[-1]))
    if tk_dirs:
        os.environ.setdefault("TK_LIBRARY", str(tk_dirs[-1]))


def validate_system(system: ComparisonSystem):
    values = (
        system.wavelength_vacuum_nm,
        system.n_particle.real,
        system.n_particle.imag,
        system.n_medium,
        system.n_glass,
        system.numerical_aperture,
        system.collection_efficiency,
        system.reference_intensity_ratio,
        system.net_phase_deg,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("All optical inputs must be finite.")
    if system.wavelength_vacuum_nm <= 0:
        raise ValueError("Wavelength must be positive.")
    if system.n_particle.real <= 0 or system.n_particle.imag < 0:
        raise ValueError("Particle n must be positive and k must be non-negative.")
    if system.n_medium <= 0 or system.n_glass <= 0:
        raise ValueError("Medium and glass refractive indices must be positive.")
    if not 0 < system.numerical_aperture <= system.n_glass:
        raise ValueError("Require 0 < NA <= glass refractive index.")
    if not 0 < system.collection_efficiency <= 1:
        raise ValueError("Collection efficiency must lie in (0, 1].")
    if not 0 < system.reference_intensity_ratio <= 1:
        raise ValueError("Reference intensity ratio must lie in (0, 1].")
    if abs(system.n_glass - system.n_medium) < 1e-8:
        raise ValueError("The glass/medium Fresnel reference is nearly zero.")


def parse_wavelengths(text):
    """Parse an editable comma/semicolon-separated wavelength list."""
    values = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Every wavelength must be finite and positive.")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("Enter at least one wavelength, for example: 533, 632")
    if len(values) > 2:
        raise ValueError("Please enter no more than two wavelengths for this overlay.")
    return tuple(values)


def particle_index(particle_name, wavelength_nm):
    """Return the preset complex index at one of the two review wavelengths."""
    try:
        values = PARTICLE_PRESETS[particle_name]
    except KeyError as exc:
        raise ValueError(f"Unknown particle preset: {particle_name}") from exc
    wavelength_nm = float(wavelength_nm)
    if wavelength_nm not in values:
        raise ValueError(
            f"No preset optical constants for {particle_name} at {wavelength_nm:g} nm."
        )
    return values[wavelength_nm]


def size_parameter(diameter_nm, system: ComparisonSystem):
    return (
        np.pi
        * system.n_medium
        * np.asarray(diameter_nm, dtype=float)
        / system.wavelength_vacuum_nm
    )


def rayleigh_qsca(x, relative_index):
    contrast = (relative_index**2 - 1.0) / (relative_index**2 + 2.0)
    return (8.0 / 3.0) * np.asarray(x, dtype=float) ** 4 * abs(contrast) ** 2


def dipole_mie_qsca(x, relative_index):
    """Scattering efficiency from exact electric and magnetic dipoles only."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    solver_index = np.conj(complex(relative_index))
    output = np.empty_like(x)
    for index, value in enumerate(x):
        a_n, b_n = miepython.an_bn(solver_index, float(value))
        output[index] = (
            6.0
            * (abs(a_n[0]) ** 2 + abs(b_n[0]) ** 2)
            / float(value) ** 2
        )
    return output


def full_mie_qsca(x, relative_index):
    """Total homogeneous-sphere Mie scattering efficiency."""
    solver_index = np.conj(complex(relative_index))
    _qext, qsca, _qback, _g = miepython.efficiencies_mx(
        solver_index, np.asarray(x, dtype=float)
    )
    qsca = np.atleast_1d(np.asarray(qsca, dtype=float))
    if np.any(~np.isfinite(qsca)) or np.any(qsca < -1e-12):
        raise ValueError("Mie solver returned a non-finite or negative Qsca.")
    return np.clip(qsca, 0.0, None)


def scattering_efficiency(x, relative_index, model):
    if model == "Rayleigh":
        return rayleigh_qsca(x, relative_index)
    if model == "Dipole Mie (a1, b1)":
        return dipole_mie_qsca(x, relative_index)
    if model == "Full Mie (multipole)":
        return full_mie_qsca(x, relative_index)
    raise ValueError(f"Unknown scattering model: {model}")


def first_upward_crossing(x, y, target=0.0):
    residual = np.asarray(y, dtype=float) - float(target)
    indices = np.flatnonzero((residual[:-1] < 0) & (residual[1:] >= 0))
    if not indices.size:
        return None
    index = int(indices[0])
    fraction = -residual[index] / (residual[index + 1] - residual[index])
    return float(x[index] + fraction * (x[index + 1] - x[index]))


def calculate_comparison(
    system: ComparisonSystem,
    diameter_min_nm=DEFAULT_DIAMETER_MIN_NM,
    diameter_max_nm=DEFAULT_DIAMETER_MAX_NM,
    diameter_points=DEFAULT_DIAMETER_POINTS,
    model=DEFAULT_MODEL,
):
    validate_system(system)
    if not np.isfinite(diameter_min_nm) or not np.isfinite(diameter_max_nm):
        raise ValueError("Diameter limits must be finite.")
    if diameter_min_nm <= 0 or diameter_max_nm <= diameter_min_nm:
        raise ValueError("Require 0 < minimum diameter < maximum diameter.")
    if not 50 <= int(diameter_points) <= 10000:
        raise ValueError("Number of points must be between 50 and 10000.")

    diameter_nm = np.geomspace(
        float(diameter_min_nm), float(diameter_max_nm), int(diameter_points)
    )
    x = size_parameter(diameter_nm, system)
    relative_index = system.n_particle / system.n_medium
    qsca = scattering_efficiency(x, relative_index, model)
    csca_nm2 = qsca * np.pi * (diameter_nm / 2.0) ** 2

    # Reference power is the reflected illumination falling within one
    # diffraction-limited effective area.  The dark-field signal is collected
    # Mie-scattered power on the same reference-normalized scale.
    r_fresnel = (system.n_glass - system.n_medium) / (
        system.n_glass + system.n_medium
    )
    reference_intensity_ratio = (
        abs(r_fresnel) ** 2 * system.reference_intensity_ratio
    )
    effective_area_nm2 = np.pi * (
        0.61 * system.wavelength_vacuum_nm / system.numerical_aperture
    ) ** 2
    darkfield = (
        system.collection_efficiency
        * csca_nm2
        / (effective_area_nm2 * reference_intensity_ratio)
    )
    field_ratio = np.sqrt(np.clip(darkfield, 0.0, None))

    cosine = float(np.cos(np.radians(system.net_phase_deg)))
    if abs(cosine) < 1e-14:
        cosine = 0.0
    interference_signed = 2.0 * field_ratio * cosine
    interference_available = 2.0 * field_ratio
    iscat_single_signed = interference_signed + darkfield
    iscat_single_magnitude = np.abs(iscat_single_signed)
    iscat_envelope = interference_available + darkfield

    crossover_nm = first_upward_crossing(
        diameter_nm, darkfield - interference_available, target=0.0
    )
    return {
        "system": system,
        "model": model,
        "diameter_nm": diameter_nm,
        "size_parameter": x,
        "qsca": qsca,
        "csca_nm2": csca_nm2,
        "effective_area_nm2": effective_area_nm2,
        "reference_intensity_ratio": reference_intensity_ratio,
        "field_ratio": field_ratio,
        "darkfield": darkfield,
        "interference_signed": interference_signed,
        "interference_available": interference_available,
        "iscat_single_signed": iscat_single_signed,
        "iscat_single_magnitude": iscat_single_magnitude,
        "iscat_envelope": iscat_envelope,
        "crossover_nm": crossover_nm,
    }


def calculate_particle_wavelength_overlay(
    base_system: ComparisonSystem,
    particle_names=DEFAULT_PARTICLES,
    wavelengths_nm=DEFAULT_WAVELENGTHS_NM,
    diameter_min_nm=DEFAULT_DIAMETER_MIN_NM,
    diameter_max_nm=DEFAULT_DIAMETER_MAX_NM,
    diameter_points=DEFAULT_DIAMETER_POINTS,
    model=DEFAULT_MODEL,
):
    results = []
    for particle_name in particle_names:
        for wavelength_nm in wavelengths_nm:
            result = calculate_comparison(
                replace(
                    base_system,
                    wavelength_vacuum_nm=float(wavelength_nm),
                    n_particle=particle_index(particle_name, wavelength_nm),
                ),
                diameter_min_nm,
                diameter_max_nm,
                diameter_points,
                model,
            )
            result["particle_name"] = particle_name
            results.append(result)
    return results


def add_camera_sensitivity_limits(
    results, sensitivity_percent, sensitivity_source=DEFAULT_CAMERA_SENSITIVITY_SOURCE
):
    """Attach first diameter crossings for a user-defined contrast floor."""
    sensitivity_percent = float(sensitivity_percent)
    if not np.isfinite(sensitivity_percent) or sensitivity_percent <= 0:
        raise ValueError("Camera sensitivity must be finite and positive.")
    threshold = sensitivity_percent / 100.0
    for result in results:
        d = result["diameter_nm"]
        result["camera_sensitivity_percent"] = sensitivity_percent
        result["camera_sensitivity_source"] = sensitivity_source
        result["iscat_detection_diameter_nm"] = first_upward_crossing(
            d, result["iscat_envelope"], target=threshold
        )
        result["darkfield_detection_diameter_nm"] = first_upward_crossing(
            d, result["darkfield"], target=threshold
        )
    return results


def detection_limit_label(result, signal_key, sensitivity_percent):
    """Format a crossing, including the two useful out-of-range cases."""
    crossing_key = (
        "iscat_detection_diameter_nm"
        if signal_key == "iscat_envelope"
        else "darkfield_detection_diameter_nm"
    )
    crossing = result[crossing_key]
    if crossing is not None:
        return f"{crossing:.1f} nm"
    threshold = float(sensitivity_percent) / 100.0
    values = result[signal_key]
    if values[0] >= threshold:
        return f"≤{result['diameter_nm'][0]:g} nm"
    return f">{result['diameter_nm'][-1]:g} nm"


def build_figure(
    results,
    log_x=True,
    log_y=True,
    show_single_phase=True,
    show_interference=True,
    camera_sensitivity_percent=DEFAULT_CAMERA_SENSITIVITY_PERCENT,
    show_camera_threshold=True,
    camera_sensitivity_source=DEFAULT_CAMERA_SENSITIVITY_SOURCE,
    figure=None,
):
    if figure is None:
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
    else:
        fig = figure
        fig.clear()
        ax = fig.subplots(1, 1)

    if isinstance(results, dict):
        results = [results]
    if not results:
        raise ValueError("No wavelength results were supplied.")
    add_camera_sensitivity_limits(
        results, camera_sensitivity_percent, camera_sensitivity_source
    )

    d = results[0]["diameter_nm"]
    system = results[0]["system"]
    particle_names = list(dict.fromkeys(result["particle_name"] for result in results))
    wavelengths_nm = list(dict.fromkeys(result["system"].wavelength_vacuum_nm for result in results))

    def series_color(particle_index, wavelength_index):
        base = np.asarray(to_rgb(SERIES_COLORS[particle_index % len(SERIES_COLORS)]))
        if wavelength_index == 0:
            return SERIES_COLORS[particle_index % len(SERIES_COLORS)]
        # A lighter tint keeps particle identity while distinguishing the second wavelength.
        return to_hex(0.62 * base + 0.38 * np.ones(3))

    particle_handles = []
    for particle_index, particle_name in enumerate(particle_names):
        for wavelength_index, wavelength_nm in enumerate(wavelengths_nm):
            particle_handles.append(
                Line2D(
                    [], [], color=series_color(particle_index, wavelength_index), lw=3.0,
                    label=f"{particle_name} @ {wavelength_nm:g} nm",
                )
            )

    for result_index, result in enumerate(results):
        particle_index = particle_names.index(result["particle_name"])
        wavelength_index = wavelengths_nm.index(result["system"].wavelength_vacuum_nm)
        color = series_color(particle_index, wavelength_index)
        darkfield_percent = 100.0 * result["darkfield"]
        envelope_percent = 100.0 * result["iscat_envelope"]
        single_percent = 100.0 * result["iscat_single_magnitude"]
        interference_percent = 100.0 * result["interference_available"]

        ax.plot(d, envelope_percent, color=color, lw=2.5, zorder=5)
        ax.plot(d, darkfield_percent, color=color, lw=1.9, ls="--", zorder=3)
        if show_single_phase:
            ax.plot(d, single_percent, color=color, lw=1.15, ls="-.", alpha=0.82, zorder=4)
        if show_interference:
            ax.plot(d, interference_percent, color=color, lw=0.95, ls=":", alpha=0.55, zorder=2)

        crossover_nm = result["crossover_nm"]
        if crossover_nm is not None:
            y_cross = float(np.interp(crossover_nm, d, darkfield_percent))
            ax.plot(crossover_nm, y_cross, "o", ms=4.6, mfc="white", mec=color, mew=1.1, zorder=7)
        if show_camera_threshold:
            for key, marker in (
                ("iscat_detection_diameter_nm", "^"),
                ("darkfield_detection_diameter_nm", "v"),
            ):
                detection_diameter = result[key]
                if detection_diameter is not None:
                    ax.plot(
                        detection_diameter,
                        camera_sensitivity_percent,
                        marker=marker,
                        ms=4.3,
                        mfc="white",
                        mec=color,
                        mew=1.0,
                        linestyle="none",
                        zorder=8,
                    )

    ax.set_xscale("log" if log_x else "linear")
    ax.set_yscale("log" if log_y else "linear")
    ax.set_xlim(d[0], d[-1])
    if not log_y:
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Particle diameter d (nm)")
    ax.set_ylabel("Reference-normalized detected signal (%)")
    wavelength_text = ", ".join(f"{wavelength:g}" for wavelength in wavelengths_nm)
    ax.set_title(
        "Common iSCAT particle examples: two wavelengths\n"
        rf"$\lambda_0={wavelength_text}$ nm, "
        rf"$I_{{ref}}/I_{{ref,0}}={system.reference_intensity_ratio:g}$, "
        rf"$n_m={system.n_medium:.3g}$, NA={system.numerical_aperture:g}; "
        f"{results[0]['model']}"
    )
    ax.grid(True, which="both", color="0.90", lw=0.6)
    particle_legend = ax.legend(
        handles=particle_handles,
        title="Particle and wavelength",
        frameon=False,
        loc="upper left",
        fontsize=6.9,
        title_fontsize=7.4,
        ncol=2,
    )
    ax.add_artist(particle_legend)
    signal_handles = [
        Line2D([], [], color="0.2", lw=2.5, label="iSCAT envelope (solid)"),
        Line2D([], [], color="0.2", lw=1.9, ls="--", label="dark-field Mie (dashed)"),
    ]
    if show_single_phase:
        signal_handles.append(Line2D([], [], color="0.2", lw=1.2, ls="-.", label=f"single phase {system.net_phase_deg:g}°"))
    if show_interference:
        signal_handles.append(Line2D([], [], color="0.2", lw=1.0, ls=":", label="heterodyne 2|u|"))
    if show_camera_threshold:
        ax.axhline(
            camera_sensitivity_percent,
            color="#E07B39",
            lw=1.25,
            ls=(0, (5, 2)),
            alpha=0.9,
            zorder=1,
        )
        signal_handles.append(
            Line2D(
                [], [], color="#E07B39", lw=1.25, ls=(0, (5, 2)),
                label=rf"camera floor {camera_sensitivity_percent:g}% ({camera_sensitivity_source.split(':')[0]})",
            )
        )
    ax.legend(handles=signal_handles, title="Signal type", frameon=False, loc="lower right", fontsize=7.2, title_fontsize=7.6)

    phase_is_quadrature = abs(
        np.cos(np.radians(system.net_phase_deg))
    ) < 1e-12
    footer = (
        "At 90 degrees the single-phase interference term is zero, so that "
        "trace overlaps dark-field; solid envelopes require phase optimization or quadrature recovery. Colors identify particles; lighter tints identify the second wavelength."
        if phase_is_quadrature
        else "Solid curves are maximum recoverable iSCAT envelopes; dash-dot curves are selected single-phase results."
    )
    fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8.1)
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.99))
    return fig


def write_csv(path: Path, results):
    if isinstance(results, dict):
        results = [results]
    headers = [
        "particle_name",
        "wavelength_nm",
        "reference_intensity_ratio",
        "camera_sensitivity_percent",
        "camera_sensitivity_source",
        "iscat_detection_diameter_nm",
        "darkfield_detection_diameter_nm",
        "diameter_nm",
        "size_parameter_x",
        "mie_qsca",
        "mie_csca_um2",
        "field_ratio_abs_u",
        "darkfield_reference_normalized",
        "available_heterodyne_term",
        "selected_phase_interference_term_signed",
        "iscat_single_phase_signed",
        "iscat_single_phase_magnitude",
        "iscat_phase_optimized_envelope",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for result in results:
            columns = [
                result["diameter_nm"],
                result["size_parameter"],
                result["qsca"],
                result["csca_nm2"] / 1.0e6,
                result["field_ratio"],
                result["darkfield"],
                result["interference_available"],
                result["interference_signed"],
                result["iscat_single_signed"],
                result["iscat_single_magnitude"],
                result["iscat_envelope"],
            ]
            wavelength_nm = result["system"].wavelength_vacuum_nm
            particle_name = result["particle_name"]
            for row in zip(*columns):
                writer.writerow(
                    [
                        particle_name,
                        f"{wavelength_nm:.12g}",
                        f"{result['system'].reference_intensity_ratio:.12g}",
                        f"{result.get('camera_sensitivity_percent', float('nan')):.12g}",
                        result.get("camera_sensitivity_source", DEFAULT_CAMERA_SENSITIVITY_SOURCE),
                        "" if result.get("iscat_detection_diameter_nm") is None else f"{result['iscat_detection_diameter_nm']:.12g}",
                        "" if result.get("darkfield_detection_diameter_nm") is None else f"{result['darkfield_detection_diameter_nm']:.12g}",
                    ]
                    + [f"{float(value):.12g}" for value in row]
                )


def save_outputs(output_dir, results, figure):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / OUTPUT_STEM
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    csv_path = stem.with_suffix(".csv")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    write_csv(csv_path, results)
    return png_path, pdf_path, csv_path


def default_system():
    return ComparisonSystem()


def launch_gui():
    configure_tk_runtime()
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    class ComparisonApp:
        def __init__(self, root):
            self.root = root
            self.root.title("iSCAT vs dark-field Mie comparison")
            self.root.geometry("1390x840")
            self.root.minsize(1080, 700)
            self.current = None

            self.wavelengths = tk.StringVar(value="533, 632")
            self.n_medium = tk.StringVar(value="1.333")
            self.n_glass = tk.StringVar(value="1.518")
            self.na = tk.StringVar(value="1.42")
            self.collection = tk.StringVar(value="0.30")
            self.reference_intensity = tk.StringVar(value="0.1")
            self.phase = tk.StringVar(value="90")
            self.model = tk.StringVar(value=DEFAULT_MODEL)
            self.diameter_min = tk.StringVar(value="10")
            self.diameter_max = tk.StringVar(value="1000")
            self.diameter_points = tk.StringVar(value="1600")
            self.camera_sensitivity = tk.StringVar(value="0.1")
            self.camera_sensitivity_source = tk.StringVar(value=DEFAULT_CAMERA_SENSITIVITY_SOURCE)
            self.log_x = tk.BooleanVar(value=True)
            self.log_y = tk.BooleanVar(value=True)
            self.show_single = tk.BooleanVar(value=True)
            self.show_interference = tk.BooleanVar(value=True)
            self.show_camera_threshold = tk.BooleanVar(value=True)
            self.particle_vars = {
                name: tk.BooleanVar(value=name in DEFAULT_PARTICLES)
                for name in PARTICLE_PRESETS
            }
            self.status = tk.StringVar(value="Ready")

            panes = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
            panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            control_shell = ttk.Frame(panes)
            scroll_canvas = tk.Canvas(control_shell, highlightthickness=0, borderwidth=0)
            scroll_bar = ttk.Scrollbar(control_shell, orient="vertical", command=scroll_canvas.yview)
            scrollable_controls = ttk.Frame(scroll_canvas, padding=8)
            scroll_window = scroll_canvas.create_window(
                (0, 0), window=scrollable_controls, anchor="nw"
            )
            scroll_canvas.configure(yscrollcommand=scroll_bar.set)
            scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

            def update_scroll_region(_event=None):
                scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

            def resize_scrollable_frame(event):
                scroll_canvas.itemconfigure(scroll_window, width=event.width)

            def enable_mousewheel(_event=None):
                scroll_canvas.bind_all("<MouseWheel>", scroll_mousewheel)

            def disable_mousewheel(_event=None):
                scroll_canvas.unbind_all("<MouseWheel>")

            def scroll_mousewheel(event):
                scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

            scrollable_controls.bind("<Configure>", update_scroll_region)
            scroll_canvas.bind("<Configure>", resize_scrollable_frame)
            scrollable_controls.bind("<Enter>", enable_mousewheel)
            scrollable_controls.bind("<Leave>", disable_mousewheel)
            controls = scrollable_controls
            plot_frame = ttk.Frame(panes, padding=(6, 0, 0, 0))
            panes.add(control_shell, weight=0)
            panes.add(plot_frame, weight=1)

            optics = ttk.LabelFrame(controls, text="Particle and optical system", padding=9)
            optics.pack(fill=tk.X, pady=(0, 7))
            self._entry_row(optics, 0, "Wavelengths to overlay (nm)", self.wavelengths)
            self._entry_row(optics, 1, "Reference intensity ratio", self.reference_intensity)
            self._entry_row(optics, 2, "Medium refractive index", self.n_medium)
            self._entry_row(optics, 3, "Glass refractive index", self.n_glass)
            self._entry_row(optics, 4, "Objective NA", self.na)
            self._entry_row(optics, 5, "Collection efficiency", self.collection)
            self._entry_row(optics, 6, "Net phase difference (deg)", self.phase)
            ttk.Label(optics, text="Scattering model").grid(
                row=7, column=0, sticky="w", padx=(0, 8), pady=2
            )
            ttk.Combobox(
                optics,
                textvariable=self.model,
                values=MODEL_OPTIONS,
                state="readonly",
                width=25,
            ).grid(row=7, column=1, sticky="ew", pady=2)

            particles = ttk.LabelFrame(controls, text="Particle presets to overlay", padding=9)
            particles.pack(fill=tk.X, pady=(0, 7))
            ttk.Label(
                particles,
                text="Each selected preset is calculated at every wavelength above.",
                foreground="#555555",
                wraplength=350,
                justify="left",
            ).pack(fill=tk.X, pady=(0, 3))
            for name, variable in self.particle_vars.items():
                ttk.Checkbutton(particles, text=name, variable=variable).pack(anchor="w")

            sweep = ttk.LabelFrame(controls, text="Diameter sweep and display", padding=9)
            sweep.pack(fill=tk.X, pady=(0, 7))
            self._entry_row(sweep, 0, "Minimum diameter (nm)", self.diameter_min)
            self._entry_row(sweep, 1, "Maximum diameter (nm)", self.diameter_max)
            self._entry_row(sweep, 2, "Number of points", self.diameter_points)
            self._entry_row(
                sweep,
                3,
                "Camera sensitivity / floor (%)",
                self.camera_sensitivity,
            )
            ttk.Label(sweep, text="Sensitivity source").grid(
                row=4, column=0, sticky="w", padx=(0, 8), pady=2
            )
            sensitivity_source_combo = ttk.Combobox(
                sweep,
                textvariable=self.camera_sensitivity_source,
                values=list(CAMERA_SENSITIVITY_PRESETS),
                state="readonly",
                width=34,
            )
            sensitivity_source_combo.grid(row=4, column=1, sticky="ew", pady=2)
            sensitivity_source_combo.bind(
                "<<ComboboxSelected>>", self.sync_camera_sensitivity
            )
            ttk.Checkbutton(
                sweep, text="Logarithmic diameter axis", variable=self.log_x
            ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))
            ttk.Checkbutton(
                sweep, text="Logarithmic signal axis", variable=self.log_y
            ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
            ttk.Checkbutton(
                sweep,
                text="Show camera sensitivity floor",
                variable=self.show_camera_threshold,
            ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 0))
            ttk.Checkbutton(
                sweep, text="Show selected single-phase curve", variable=self.show_single
            ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(2, 0))
            ttk.Checkbutton(
                sweep, text="Show available heterodyne component", variable=self.show_interference
            ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(2, 0))

            note = ttk.LabelFrame(controls, text="90° interpretation", padding=9)
            note.pack(fill=tk.X, pady=(0, 7))
            ttk.Label(
                note,
                text=(
                    "At a net 90° phase, a single intensity channel has zero "
                    "field-linear interference. Each solid curve is therefore a "
                    "phase-optimized or quadrature-recovered envelope, not the "
                    "90° camera trace. Color identifies the particle; lighter tints identify "
                    "the second wavelength. Line style denotes signal type."
                ),
                wraplength=350,
                foreground="#555555",
                justify="left",
            ).pack(fill=tk.X)

            actions = ttk.LabelFrame(controls, text="Actions", padding=9)
            actions.pack(fill=tk.X, pady=(0, 7))
            ttk.Button(actions, text="Calculate & Plot", command=self.calculate).pack(fill=tk.X, pady=2)
            ttk.Button(actions, text="Save PNG + PDF + CSV", command=self.save_bundle).pack(fill=tk.X, pady=2)
            ttk.Button(actions, text="Reset Defaults", command=self.reset_defaults).pack(fill=tk.X, pady=2)
            ttk.Label(
                controls,
                textvariable=self.status,
                wraplength=360,
                foreground="#204A87",
                justify="left",
            ).pack(fill=tk.X, pady=(5, 0))

            self.figure = Figure(figsize=(9.4, 6.5), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(fill=tk.X)
            self.calculate(silent=True)

        @staticmethod
        def _entry_row(parent, row, text, variable):
            ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", pady=2)
            parent.columnconfigure(1, weight=1)

        def sync_camera_sensitivity(self, _event=None):
            value = CAMERA_SENSITIVITY_PRESETS[self.camera_sensitivity_source.get()]
            if value is not None:
                self.camera_sensitivity.set(f"{value:g}")

        def read_config(self):
            wavelengths_nm = parse_wavelengths(self.wavelengths.get())
            particle_names = [
                name for name, variable in self.particle_vars.items() if variable.get()
            ]
            if not particle_names:
                raise ValueError("Select at least one particle preset to overlay.")
            system = ComparisonSystem(
                wavelength_vacuum_nm=wavelengths_nm[0],
                n_particle=particle_index(particle_names[0], wavelengths_nm[0]),
                n_medium=float(self.n_medium.get()),
                n_glass=float(self.n_glass.get()),
                numerical_aperture=float(self.na.get()),
                collection_efficiency=float(self.collection.get()),
                reference_intensity_ratio=float(self.reference_intensity.get()),
                net_phase_deg=float(self.phase.get()),
            )
            validate_system(system)
            d_min = float(self.diameter_min.get())
            d_max = float(self.diameter_max.get())
            points = int(self.diameter_points.get())
            return {
                "system": system,
                "wavelengths_nm": wavelengths_nm,
                "particle_names": particle_names,
                "diameter_min_nm": d_min,
                "diameter_max_nm": d_max,
                "diameter_points": points,
                "camera_sensitivity_percent": float(self.camera_sensitivity.get()),
                "camera_sensitivity_source": self.camera_sensitivity_source.get(),
                "model": self.model.get(),
                "log_x": bool(self.log_x.get()),
                "log_y": bool(self.log_y.get()),
                "show_camera_threshold": bool(self.show_camera_threshold.get()),
                "show_single_phase": bool(self.show_single.get()),
                "show_interference": bool(self.show_interference.get()),
            }

        def calculate(self, silent=False):
            try:
                self.root.configure(cursor="watch")
                self.root.update_idletasks()
                config = self.read_config()
                results = calculate_particle_wavelength_overlay(
                    config["system"],
                    config["particle_names"],
                    config["wavelengths_nm"],
                    config["diameter_min_nm"],
                    config["diameter_max_nm"],
                    config["diameter_points"],
                    config["model"],
                )
                add_camera_sensitivity_limits(
                    results, config["camera_sensitivity_percent"]
                )
                build_figure(
                    results,
                    log_x=config["log_x"],
                    log_y=config["log_y"],
                    show_single_phase=config["show_single_phase"],
                    show_interference=config["show_interference"],
                    camera_sensitivity_percent=config["camera_sensitivity_percent"],
                    show_camera_threshold=config["show_camera_threshold"],
                    camera_sensitivity_source=config["camera_sensitivity_source"],
                    figure=self.figure,
                )
                self.canvas.draw_idle()
                self.current = {"config": config, "results": results}
                summaries = []
                for result in results:
                    particle_name = result["particle_name"]
                    wavelength_nm = result["system"].wavelength_vacuum_nm
                    crossover = result["crossover_nm"]
                    ratio_end = result["iscat_envelope"][-1] / result["darkfield"][-1]
                    crossover_text = (
                        f"cross {crossover:.1f} nm"
                        if crossover is not None
                        else "no crossover"
                    )
                    iscat_limit = detection_limit_label(
                        result, "iscat_envelope", config["camera_sensitivity_percent"]
                    )
                    darkfield_limit = detection_limit_label(
                        result, "darkfield", config["camera_sensitivity_percent"]
                    )
                    detection_text = f"detect iSCAT/DF={iscat_limit}/{darkfield_limit}"
                    summaries.append(
                        f"{particle_name} {wavelength_nm:g} nm: {crossover_text}, "
                        f"{detection_text}, envelope/DF={ratio_end:.3f}"
                    )
                self.status.set(
                    f"Complete ({config['model']}, d_max={config['diameter_max_nm']:g} nm); "
                    + "; ".join(summaries)
                )
            except Exception as exc:
                self.current = None
                self.status.set("Calculation failed.")
                if silent:
                    raise
                messagebox.showerror("Calculation error", str(exc), parent=self.root)
            finally:
                self.root.configure(cursor="")

        def save_bundle(self):
            if self.current is None:
                self.calculate()
                if self.current is None:
                    return
            selected = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save iSCAT and dark-field comparison",
                defaultextension=".png",
                initialfile=OUTPUT_STEM + ".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            )
            if not selected:
                return
            stem = Path(selected).with_suffix("")
            png_path = stem.with_suffix(".png")
            pdf_path = stem.with_suffix(".pdf")
            csv_path = stem.with_suffix(".csv")
            try:
                self.figure.savefig(png_path, dpi=300, bbox_inches="tight")
                self.figure.savefig(pdf_path, bbox_inches="tight")
                write_csv(csv_path, self.current["results"])
                self.status.set(f"Saved {png_path.name}, PDF, and CSV.")
                messagebox.showinfo(
                    "Saved",
                    f"PNG: {png_path}\nPDF: {pdf_path}\nCSV: {csv_path}",
                    parent=self.root,
                )
            except Exception as exc:
                messagebox.showerror("Save error", str(exc), parent=self.root)

        def reset_defaults(self):
            self.wavelengths.set("533, 632")
            self.n_medium.set("1.333")
            self.n_glass.set("1.518")
            self.na.set("1.42")
            self.collection.set("0.30")
            self.reference_intensity.set("0.1")
            self.phase.set("90")
            self.model.set(DEFAULT_MODEL)
            self.diameter_min.set("10")
            self.diameter_max.set("1000")
            self.diameter_points.set("1600")
            self.camera_sensitivity.set("0.1")
            self.camera_sensitivity_source.set(DEFAULT_CAMERA_SENSITIVITY_SOURCE)
            self.log_x.set(True)
            self.log_y.set(True)
            self.show_camera_threshold.set(True)
            self.show_single.set(True)
            self.show_interference.set(True)
            for name, variable in self.particle_vars.items():
                variable.set(name in DEFAULT_PARTICLES)
            self.calculate()

    root = tk.Tk()
    ComparisonApp(root)
    root.mainloop()


def run_batch(output_dir):
    results = calculate_particle_wavelength_overlay(default_system())
    figure = build_figure(results)
    paths = save_outputs(output_dir, results, figure)
    plt.close(figure)
    print("Wrote:")
    for path in paths:
        print(f"  {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate the default PNG, PDF, and CSV without opening the GUI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Batch output directory (default: script directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch:
        run_batch(args.output_dir)
    else:
        launch_gui()


if __name__ == "__main__":
    main()
