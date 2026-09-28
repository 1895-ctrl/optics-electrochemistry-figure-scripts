#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI for small SPR reflectivity-dip perturbations.

This script models a dilute population of gas domains, dielectric particles,
or metal particles as an effective interfacial layer between metal and liquid:

    prism / metal / effective perturbation layer / liquid

The layer permittivity uses the Maxwell-Garnett model. Reflectivity uses the
multilayer Fresnel engine in ``SPRM_Reflectivity_GUI.py``. This equivalent-layer
model describes an ensemble response, not single-particle SPP scattering.
Its reported height is an *effective optical height*, which need not equal the
diameter of an isolated bubble or particle.

Place this file beside SPRM_Reflectivity_GUI.py.
"""

from __future__ import annotations

import argparse
import csv
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

from SPRM_Reflectivity_GUI import (
    MEDIUM_LIST,
    METALS,
    NK_DATA,
    PRISM_LIST,
    SELLMEIER,
    critical_angle,
    eps_metal,
    reflectivity_p,
    sellmeier_n,
)


PRISM_OPTIONS = [name for name in PRISM_LIST if name in SELLMEIER]
MEDIUM_OPTIONS = [name for name in MEDIUM_LIST if name in SELLMEIER]
DEFAULT_LIQUID = next(
    (name for name in MEDIUM_OPTIONS if name.startswith("Water 20.0")),
    MEDIUM_OPTIONS[0],
)

INCLUSION_PRESETS = {
    "H2 gas domain": (1.00013, 0.0, "gas"),
    "Air gas domain": (1.00027, 0.0, "gas"),
    "O2 gas domain": (1.00027, 0.0, "gas"),
    "Protein particle": (1.45, 0.0, "particle"),
    "Silica particle": (1.46, 0.0, "particle"),
    "Polystyrene particle": (1.59, 0.0, "particle"),
    # n and k are resolved dynamically from NK_DATA at the selected wavelength.
    "Gold nanoparticle (Au)": (np.nan, np.nan, "metal:Au"),
    "Platinum nanoparticle (Pt)": (np.nan, np.nan, "metal:Pt"),
    "Custom inclusion": (1.40, 0.0, "custom"),
}

REPRESENTATION_OBJECT = "Object geometry (intuitive)"
REPRESENTATION_LAYER = "Direct effective layer"
OBJECT_BUBBLE = "Surface spherical-cap bubble"
OBJECT_PARTICLE = "Spherical dielectric particle"
SCAN_OBJECT_SIZE = "Object size (nm)"
SCAN_OBJECT_COUNT = "Object count"
SCAN_THICKNESS = "Effective height (nm)"
SCAN_FRACTION = "Volume fraction (%)"
DEFAULT_FIGURE = "SPR_nano_perturbation_reflectivity_default.png"
DEFAULT_SCAN_FIGURE = "SPR_nano_perturbation_calibration_default.png"
DEFAULT_CSV = "SPR_nano_perturbation_default.csv"
DEFAULT_ANGULAR_NOISE_SIGMA_MDEG = 1.0

# Keep colors stable across the reflectivity and calibration tabs so that the
# same inclusion is easy to follow when several presets are overlaid.
OVERLAY_COLORS = {
    "H2 gas domain": "#0072B2",
    "Air gas domain": "#56B4E9",
    "O2 gas domain": "#009E73",
    "Protein particle": "#D55E00",
    "Silica particle": "#CC79A7",
    "Polystyrene particle": "#7A3E9D",
    "Gold nanoparticle (Au)": "#E6AB02",
    "Platinum nanoparticle (Pt)": "#5F6368",
    "Custom inclusion": "#7A3E9D",
}


@dataclass(frozen=True)
class Parameters:
    wavelength_nm: float = 632.8
    prism: str = "SF10"
    liquid: str = DEFAULT_LIQUID
    metal: str = "Au"
    metal_source: str = ""
    metal_thickness_nm: float = 48.0
    inclusion_name: str = "H2 gas domain"
    inclusion_n: float = 1.00013
    inclusion_k: float = 0.0
    representation: str = REPRESENTATION_OBJECT
    object_geometry: str = OBJECT_BUBBLE
    object_size_nm: float = 100.0
    object_count: int = 100
    roi_radius_um: float = 3.0
    bubble_contact_angle_deg: float = 120.0
    effective_height_nm: float = 10.0
    volume_fraction: float = 0.01
    # Cross-study engineering value rather than the best result from one
    # instrument. Controlled research systems commonly fall within roughly
    # 0.3--3 mdeg; the user's measured blank trace should replace this value.
    angular_noise_sigma_mdeg: float = DEFAULT_ANGULAR_NOISE_SIGMA_MDEG
    lod_sigma_multiplier: float = 3.0
    angle_start_deg: float = 48.0
    angle_stop_deg: float = 65.0
    angle_step_deg: float = 0.005
    tracking_window_deg: float = 2.0
    scan_variable: str = SCAN_OBJECT_SIZE
    scan_start: float = 5.0
    scan_stop: float = 100.0
    scan_step: float = 5.0


def default_source(metal: str) -> str:
    return list(NK_DATA[metal].keys())[0]


def preset_nk(preset_name: str, wavelength_nm: float) -> tuple[float, float, str]:
    """Resolve a preset's n and k, including dispersive metal nanoparticles."""
    n_value, k_value, kind = INCLUSION_PRESETS[preset_name]
    if kind.startswith("metal:"):
        particle_metal = kind.split(":", 1)[1]
        eps_particle = eps_metal(
            particle_metal, default_source(particle_metal), wavelength_nm)
        nk_particle = np.sqrt(eps_particle)
        n_value = float(np.real(nk_particle))
        k_value = abs(float(np.imag(nk_particle)))
    return float(n_value), float(k_value), kind


def inclusive_grid(start: float, stop: float, step: float,
                   maximum: int = 5001) -> np.ndarray:
    if stop <= start:
        raise ValueError("Scan stop must be greater than scan start.")
    if step <= 0.0:
        raise ValueError("Scan step must be positive.")
    count = int(np.floor((stop - start) / step + 1.0e-10)) + 1
    if count > maximum:
        raise ValueError(f"Scan contains {count} points; maximum is {maximum}.")
    values = start + np.arange(count, dtype=float) * step
    if values[-1] < stop - 0.25 * step:
        values = np.append(values, stop)
    return values


def angular_grid(p: Parameters) -> np.ndarray:
    if not (0.0 < p.angle_start_deg < p.angle_stop_deg < 90.0):
        raise ValueError("Angle range must satisfy 0 < start < stop < 90 deg.")
    if p.angle_step_deg <= 0.0:
        raise ValueError("Angle step must be positive.")
    return inclusive_grid(
        p.angle_start_deg, p.angle_stop_deg, p.angle_step_deg, maximum=100001)


def validate(p: Parameters) -> None:
    if not (100.0 <= p.wavelength_nm <= 5000.0):
        raise ValueError("Wavelength must lie between 100 and 5000 nm.")
    if p.metal_thickness_nm <= 0.0:
        raise ValueError("Metal thickness must be positive.")
    if p.inclusion_n <= 0.0 or p.inclusion_k < 0.0:
        raise ValueError("Inclusion requires n > 0 and k >= 0.")
    if p.object_size_nm <= 0.0:
        raise ValueError("Object size must be positive.")
    if p.object_count < 1:
        raise ValueError("Object count must be at least one.")
    if p.roi_radius_um <= 0.0:
        raise ValueError("Sensing-area radius must be positive.")
    if not (0.0 < p.bubble_contact_angle_deg < 175.0):
        raise ValueError("Gas-side contact angle must lie between 0 and 175 deg.")
    if p.effective_height_nm < 0.0:
        raise ValueError("Effective height cannot be negative.")
    if not (0.0 <= p.volume_fraction <= 0.60):
        raise ValueError("Volume fraction must lie between 0 and 60%.")
    if p.tracking_window_deg <= 0.0:
        raise ValueError("Dip tracking window must be positive.")
    if p.angular_noise_sigma_mdeg <= 0.0:
        raise ValueError("Blank angular-noise standard deviation must be positive.")
    if not (1.0 <= p.lod_sigma_multiplier <= 10.0):
        raise ValueError("LOD sigma multiplier must lie between 1 and 10.")
    angular_grid(p)
    scan = inclusive_grid(p.scan_start, p.scan_stop, p.scan_step)
    if p.scan_variable in (SCAN_OBJECT_SIZE, SCAN_OBJECT_COUNT) and np.any(scan <= 0.0):
        raise ValueError("Object-size/count scans must contain positive values.")
    if p.scan_variable == SCAN_THICKNESS and np.any(scan < 0.0):
        raise ValueError("Effective-height scan cannot include negative values.")
    if p.scan_variable == SCAN_FRACTION and (
            np.any(scan < 0.0) or np.any(scan > 60.0)):
        raise ValueError("Volume-fraction scan must stay between 0 and 60%.")


def maxwell_garnett_eps(eps_host: complex, eps_inclusion: complex,
                        fraction: float) -> complex:
    """Maxwell-Garnett permittivity for dilute spherical inclusions."""
    if fraction <= 0.0:
        return complex(eps_host)
    numerator = eps_inclusion + 2.0 * eps_host + 2.0 * fraction * (
        eps_inclusion - eps_host)
    denominator = eps_inclusion + 2.0 * eps_host - fraction * (
        eps_inclusion - eps_host)
    if abs(denominator) < 1.0e-14:
        raise ValueError("Maxwell-Garnett denominator is too close to zero.")
    return complex(eps_host * numerator / denominator)


def resolved_effective_layer(p: Parameters) -> dict:
    """Convert explicit object geometry into an equivalent optical layer.

    The conversion conserves object volume over a circular sensing ROI.  It is
    intentionally reported as an equivalent ensemble response and is not used
    to claim that one lateral nano-object is an infinite planar film.
    """
    if p.representation == REPRESENTATION_LAYER:
        return {
            "height_nm": p.effective_height_nm,
            "fraction": p.volume_fraction,
            "single_volume_nm3": np.nan,
            "total_volume_nm3": np.nan,
            "footprint_diameter_nm": np.nan,
            "projected_coverage": np.nan,
            "areal_density_per_um2": np.nan,
            "description": "direct effective layer",
        }
    if p.representation != REPRESENTATION_OBJECT:
        raise ValueError(f"Unknown representation: {p.representation!r}.")

    radius_nm = 0.5 * p.object_size_nm
    if p.object_geometry == OBJECT_BUBBLE:
        alpha = np.deg2rad(p.bubble_contact_angle_deg)
        height_nm = radius_nm * np.tan(alpha / 2.0)
        single_volume = np.pi * height_nm * (
            3.0 * radius_nm ** 2 + height_nm ** 2) / 6.0
        size_description = (
            f"bubble footprint diameter {p.object_size_nm:g} nm, "
            f"cap height {height_nm:.2f} nm")
    elif p.object_geometry == OBJECT_PARTICLE:
        height_nm = p.object_size_nm
        single_volume = np.pi * p.object_size_nm ** 3 / 6.0
        size_description = f"sphere diameter {p.object_size_nm:g} nm"
    else:
        raise ValueError(f"Unknown object geometry: {p.object_geometry!r}.")

    roi_radius_nm = p.roi_radius_um * 1000.0
    roi_area_nm2 = np.pi * roi_radius_nm ** 2
    total_volume = p.object_count * single_volume
    bounding_volume = roi_area_nm2 * height_nm
    fraction = total_volume / bounding_volume
    projected_coverage = (
        p.object_count * np.pi * radius_nm ** 2 / roi_area_nm2)
    if projected_coverage > 0.15:
        raise ValueError(
            "Projected object coverage exceeds 15%, outside the intended "
            "dilute-particle regime. Reduce object count/size or increase ROI.")
    if fraction > 0.60:
        raise ValueError(
            "The selected objects occupy more than 60% of the equivalent layer. "
            "Increase sensing-area radius, reduce object count/size, or use a "
            "non-dilute structural model.")
    return {
        "height_nm": float(height_nm),
        "fraction": float(fraction),
        "single_volume_nm3": float(single_volume),
        "total_volume_nm3": float(total_volume),
        "footprint_diameter_nm": float(p.object_size_nm),
        "projected_coverage": float(projected_coverage),
        "areal_density_per_um2": float(
            p.object_count / (np.pi * p.roi_radius_um ** 2)),
        "description": (
            f"{p.object_count} x {size_description} in a circular ROI of "
            f"radius {p.roi_radius_um:g} um"),
    }


def refined_minimum(theta: np.ndarray, values: np.ndarray,
                    reference_deg: float | None = None,
                    half_window_deg: float | None = None) -> tuple[float, float]:
    """Quadratic minimum, optionally tracking the branch near a reference."""
    theta = np.asarray(theta, dtype=float)
    values = np.asarray(values, dtype=float)
    if reference_deg is not None and half_window_deg is not None:
        mask = np.abs(theta - reference_deg) <= half_window_deg
        indices = np.flatnonzero(mask)
        if indices.size >= 3:
            i = int(indices[np.argmin(values[indices])])
        else:
            i = int(np.argmin(values))
    else:
        i = int(np.argmin(values))
    if i == 0 or i == theta.size - 1:
        return float(theta[i]), float(values[i])
    x = theta[i - 1:i + 2]
    y = values[i - 1:i + 2]
    coeff = np.polyfit(x, y, 2)
    if coeff[0] <= 0.0:
        return float(theta[i]), float(values[i])
    x_vertex = float(-coeff[1] / (2.0 * coeff[0]))
    if not (x[0] <= x_vertex <= x[-1]):
        return float(theta[i]), float(values[i])
    y_vertex = float(np.polyval(coeff, x_vertex))
    return x_vertex, y_vertex


def optical_system(p: Parameters) -> dict:
    n_prism = sellmeier_n(p.prism, p.wavelength_nm)
    n_liquid = sellmeier_n(p.liquid, p.wavelength_nm)
    # Keep the lossless incident and bulk media real because the shared
    # critical_angle helper intentionally operates on real permittivities.
    eps_prism = float(n_prism ** 2)
    eps_liquid = float(n_liquid ** 2)
    eps_inclusion = complex(p.inclusion_n, p.inclusion_k) ** 2
    eps_m = eps_metal(p.metal, p.metal_source, p.wavelength_nm)
    return {
        "n_prism": n_prism,
        "n_liquid": n_liquid,
        "eps_prism": eps_prism,
        "eps_liquid": eps_liquid,
        "eps_inclusion": eps_inclusion,
        "eps_metal": eps_m,
        "theta_c": critical_angle(eps_prism, eps_liquid),
    }


def parameters_for_preset(p: Parameters, preset_name: str) -> Parameters:
    """Return a copy using a built-in material and its matching geometry."""
    n_value, k_value, kind = preset_nk(preset_name, p.wavelength_nm)
    geometry = p.object_geometry
    if kind == "gas":
        geometry = OBJECT_BUBBLE
    elif kind == "particle" or kind.startswith("metal:"):
        geometry = OBJECT_PARTICLE
    return replace(
        p,
        inclusion_name=preset_name,
        inclusion_n=n_value,
        inclusion_k=k_value,
        object_geometry=geometry,
    )


def series_color(name: str, index: int) -> str:
    fallback = ["#7A3E9D", "#0072B2", "#D55E00", "#009E73",
                "#CC79A7", "#E69F00", "#56B4E9"]
    return OVERLAY_COLORS.get(name, fallback[index % len(fallback)])


def layer_curve(theta: np.ndarray, p: Parameters, system: dict,
                height_nm: float, fraction: float) -> tuple[np.ndarray, complex]:
    eps_eff = maxwell_garnett_eps(
        system["eps_liquid"], system["eps_inclusion"], fraction)
    if height_nm <= 0.0 or fraction <= 0.0:
        curve = reflectivity_p(
            [system["eps_prism"], system["eps_metal"], system["eps_liquid"]],
            [p.metal_thickness_nm], p.wavelength_nm, theta)
    else:
        curve = reflectivity_p(
            [system["eps_prism"], system["eps_metal"], eps_eff,
             system["eps_liquid"]],
            [p.metal_thickness_nm, height_nm], p.wavelength_nm, theta)
    return np.asarray(curve, dtype=float), eps_eff


def calculate(p: Parameters) -> dict:
    validate(p)
    theta = angular_grid(p)
    system = optical_system(p)
    baseline, _ = layer_curve(theta, p, system, 0.0, 0.0)
    baseline_dip = refined_minimum(theta, baseline)
    resolved = resolved_effective_layer(p)
    perturbed, eps_eff = layer_curve(
        theta, p, system, resolved["height_nm"], resolved["fraction"])
    perturbed_dip = refined_minimum(
        theta, perturbed, baseline_dip[0], p.tracking_window_deg)
    delta_r = perturbed - baseline

    slope = np.gradient(baseline, theta)
    slope_mask = np.abs(theta - baseline_dip[0]) <= p.tracking_window_deg
    slope_indices = np.flatnonzero(slope_mask)
    if slope_indices.size == 0:
        slope_indices = np.arange(theta.size)
    optimum_index = int(slope_indices[np.argmax(np.abs(slope[slope_indices]))])
    optimum_angle = float(theta[optimum_index])

    scan_x = inclusive_grid(p.scan_start, p.scan_stop, p.scan_step)
    scan_theta = np.empty_like(scan_x)
    scan_rmin = np.empty_like(scan_x)
    scan_fixed_r = np.empty_like(scan_x)
    scan_max_abs_delta_r = np.empty_like(scan_x)
    scan_optimum_delta_r = np.empty_like(scan_x)
    previous_angle = baseline_dip[0]
    baseline_fixed = float(baseline[optimum_index])
    for i, value in enumerate(scan_x):
        if p.scan_variable == SCAN_OBJECT_SIZE:
            scan_p = replace(p, representation=REPRESENTATION_OBJECT,
                             object_size_nm=float(value))
            scan_resolved = resolved_effective_layer(scan_p)
            height_nm = scan_resolved["height_nm"]
            fraction = scan_resolved["fraction"]
        elif p.scan_variable == SCAN_OBJECT_COUNT:
            scan_p = replace(p, representation=REPRESENTATION_OBJECT,
                             object_count=max(1, int(round(value))))
            scan_resolved = resolved_effective_layer(scan_p)
            height_nm = scan_resolved["height_nm"]
            fraction = scan_resolved["fraction"]
        elif p.scan_variable == SCAN_THICKNESS:
            height_nm = float(value)
            fraction = resolved["fraction"]
        else:
            height_nm = resolved["height_nm"]
            fraction = float(value) / 100.0
        curve, _ = layer_curve(theta, p, system, height_nm, fraction)
        dip = refined_minimum(
            theta, curve, previous_angle, p.tracking_window_deg)
        previous_angle = dip[0]
        scan_theta[i] = dip[0]
        scan_rmin[i] = dip[1]
        scan_fixed_r[i] = curve[optimum_index]
        difference = curve - baseline
        scan_max_abs_delta_r[i] = float(np.max(np.abs(difference)))
        scan_optimum_delta_r[i] = float(difference[optimum_index])

    scan_delta_theta_mdeg = (scan_theta - baseline_dip[0]) * 1000.0
    angle_lod_mdeg = p.angular_noise_sigma_mdeg * p.lod_sigma_multiplier
    crossing_indices = np.flatnonzero(
        np.abs(scan_delta_theta_mdeg) >= angle_lod_mdeg)
    minimum_detectable_x: float | None = None
    if crossing_indices.size:
        j = int(crossing_indices[0])
        if j == 0:
            minimum_detectable_x = float(scan_x[0])
        else:
            x0, x1 = float(scan_x[j - 1]), float(scan_x[j])
            y0 = abs(float(scan_delta_theta_mdeg[j - 1]))
            y1 = abs(float(scan_delta_theta_mdeg[j]))
            if y1 > y0:
                minimum_detectable_x = x0 + (
                    angle_lod_mdeg - y0) * (x1 - x0) / (y1 - y0)
            else:
                minimum_detectable_x = x1

    return {
        "parameters": p,
        "system": system,
        "theta": theta,
        "baseline": baseline,
        "perturbed": perturbed,
        "delta_r": delta_r,
        "baseline_dip": baseline_dip,
        "perturbed_dip": perturbed_dip,
        "delta_theta_mdeg": (perturbed_dip[0] - baseline_dip[0]) * 1000.0,
        "delta_rmin": perturbed_dip[1] - baseline_dip[1],
        "resolved": resolved,
        "angle_lod_mdeg": angle_lod_mdeg,
        "minimum_detectable_x": minimum_detectable_x,
        "detectable_by_angle": abs(
            (perturbed_dip[0] - baseline_dip[0]) * 1000.0
        ) >= angle_lod_mdeg,
        "eps_eff": eps_eff,
        "n_eff": np.sqrt(eps_eff),
        "optimum_index": optimum_index,
        "optimum_angle": optimum_angle,
        "optimum_delta_r": float(delta_r[optimum_index]),
        "max_abs_delta_r": float(np.max(np.abs(delta_r))),
        "scan_x": scan_x,
        "scan_theta": scan_theta,
        "scan_delta_theta_mdeg": scan_delta_theta_mdeg,
        "scan_rmin": scan_rmin,
        "scan_fixed_r": scan_fixed_r,
        "scan_max_abs_delta_r": scan_max_abs_delta_r,
        "scan_optimum_delta_r": scan_optimum_delta_r,
        "baseline_fixed_r": baseline_fixed,
    }


def draw_reflectivity(figure: Figure, result: dict,
                      overlay_results: list[dict] | None = None) -> None:
    figure.clear()
    p: Parameters = result["parameters"]
    resolved = result["resolved"]
    theta = result["theta"]
    series = [result] + list(overlay_results or [])
    ax = figure.add_subplot(2, 1, 1)
    ax_delta = figure.add_subplot(2, 1, 2, sharex=ax)

    ax.plot(theta, result["baseline"], color="#4C2C7A", lw=2.0,
            label="Baseline: metal / liquid")
    b = result["baseline_dip"]
    ax.plot(b[0], b[1], "o", color="#4C2C7A", ms=5)
    for index, item in enumerate(series):
        item_p: Parameters = item["parameters"]
        color = series_color(item_p.inclusion_name, index)
        q_item = item["perturbed_dip"]
        primary_suffix = " (primary)" if index == 0 and len(series) > 1 else ""
        ax.plot(theta, item["perturbed"], color=color,
                lw=2.2 if index == 0 else 1.7,
                label=f"{item_p.inclusion_name}{primary_suffix}")
        ax.plot(q_item[0], q_item[1], "s", color=color,
                ms=5 if index == 0 else 4)
    ax.axvline(result["system"]["theta_c"], color="0.50", ls=":", lw=1.0,
               label=fr"Liquid critical angle {result['system']['theta_c']:.2f}$^\circ$")
    ax.set_ylabel(r"p-polarized reflectivity $R_p$")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="best", framealpha=0.94)
    overlay_note = (
        f"{len(series)} selected inclusions; shared size/count/ROI"
        if len(series) > 1 else f"{p.inclusion_name}: {resolved['description']}")
    ax.set_title(
        f"Small SPR perturbation | {p.prism} / {p.metal} "
        f"{p.metal_thickness_nm:g} nm / {p.liquid}, {p.wavelength_nm:g} nm\n"
        f"{overlay_note}")

    for index, item in enumerate(series):
        item_p: Parameters = item["parameters"]
        color = series_color(item_p.inclusion_name, index)
        ax_delta.plot(theta, 1000.0 * item["delta_r"], color=color,
                      lw=2.0 if index == 0 else 1.6,
                      label=item_p.inclusion_name)
    ax_delta.axhline(0.0, color="0.45", lw=0.8)
    i_opt = result["optimum_index"]
    ax_delta.plot(theta[i_opt], 1000.0 * result["delta_r"][i_opt], "D",
                  color=series_color(p.inclusion_name, 0), ms=5,
                  label=fr"Fixed-angle monitor: {result['optimum_angle']:.3f}$^\circ$")
    ax_delta.set_xlabel(r"Internal angle of incidence $\theta$ (deg)")
    ax_delta.set_ylabel(r"Differential signal $10^3\Delta R$")
    ax_delta.grid(alpha=0.25, lw=0.6)
    ax_delta.legend(loc="best", framealpha=0.94)

    q = result["perturbed_dip"]
    summary = (
        f"Primary: {p.inclusion_name}\n"
        fr"$\theta_0$={b[0]:.4f}$^\circ$" "\n"
        fr"$\theta_1$={q[0]:.4f}$^\circ$" "\n"
        fr"$\Delta\theta$={result['delta_theta_mdeg']:+.2f} mdeg" "\n"
        fr"$\Delta R_{{min}}$={result['delta_rmin']:+.3e}" "\n"
        f"Angle LOD {result['angle_lod_mdeg']:.3g} mdeg "
        f"({p.lod_sigma_multiplier:g}$\\sigma$): "
        f"{'detectable' if result['detectable_by_angle'] else 'below threshold'}")
    ax.text(0.985, 0.04, summary, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.70",
                      alpha=0.94))
    figure.tight_layout()


def draw_calibration(figure: Figure, result: dict,
                     overlay_results: list[dict] | None = None) -> None:
    figure.clear()
    p: Parameters = result["parameters"]
    x = result["scan_x"]
    x_label = p.scan_variable
    series = [result] + list(overlay_results or [])
    ax = figure.add_subplot(2, 1, 1)
    ax_signal = figure.add_subplot(2, 1, 2, sharex=ax)

    for index, item in enumerate(series):
        item_p: Parameters = item["parameters"]
        color = series_color(item_p.inclusion_name, index)
        minimum_x = item["minimum_detectable_x"]
        lod_text = ""
        if minimum_x is not None:
            lod_text = f"; LOD x≈{minimum_x:.1f}"
        primary_suffix = " (primary)" if index == 0 and len(series) > 1 else ""
        ax.plot(x, item["scan_delta_theta_mdeg"], color=color,
                lw=2.2 if index == 0 else 1.7,
                marker="o", ms=3.0, markevery=max(1, len(x) // 18),
                label=f"{item_p.inclusion_name}{primary_suffix}{lod_text}")
        if minimum_x is not None:
            nearest = int(np.argmin(np.abs(x - minimum_x)))
            sign = -1.0 if item["scan_delta_theta_mdeg"][nearest] < 0 else 1.0
            ax.plot(minimum_x, sign * item["angle_lod_mdeg"], marker="D",
                    color=color, ms=4.5, zorder=5)
    ax.axhline(0.0, color="0.50", lw=0.8)
    lod = result["angle_lod_mdeg"]
    ax.axhline(lod, color="#009E73", ls="--", lw=1.0,
               label=(f"±{lod:.3g} mdeg LOD = "
                      f"{p.lod_sigma_multiplier:g}σ"))
    ax.axhline(-lod, color="#009E73", ls="--",
               lw=1.0)
    ax.set_ylabel(r"Tracked dip shift $\Delta\theta$ (mdeg)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="best", framealpha=0.94)
    calibration_title = (
        "Object size/count to SPR signal calibration (spot-averaged estimate)"
        if p.scan_variable in (SCAN_OBJECT_SIZE, SCAN_OBJECT_COUNT)
        else "Equivalent-layer calibration for a small interfacial perturbation")
    ax.set_title(
        calibration_title + "\n"
        "Dip continuation is used to avoid switching to an unrelated minimum")

    for index, item in enumerate(series):
        item_p: Parameters = item["parameters"]
        color = series_color(item_p.inclusion_name, index)
        ax_signal.plot(x, 1000.0 * item["scan_optimum_delta_r"],
                       color=color, lw=2.0 if index == 0 else 1.6,
                       label=item_p.inclusion_name)
        ax_signal.plot(x, 1000.0 * item["scan_max_abs_delta_r"],
                       color=color, lw=1.35, ls="--", label="_nolegend_")
    ax_signal.axhline(0.0, color="0.50", lw=0.8)
    ax_signal.set_xlabel(x_label)
    ax_signal.set_ylabel(r"Differential reflectivity $10^3\Delta R$")
    ax_signal.grid(alpha=0.25, lw=0.6)
    ax_signal.legend(loc="best", framealpha=0.94)
    ax_signal.text(
        0.985, 0.04,
        (f"Solid: $\\Delta R$ at fixed angle "
         f"{result['optimum_angle']:.3f}$^\\circ$\n"
         r"Dashed: maximum $|\Delta R|$ over angle scan"),
        transform=ax_signal.transAxes, ha="right", va="bottom", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75",
                  alpha=0.92))

    if p.scan_variable == SCAN_OBJECT_SIZE:
        fixed_text = (
            f"{p.object_count} object(s), ROI radius {p.roi_radius_um:g} um")
    elif p.scan_variable == SCAN_OBJECT_COUNT:
        fixed_text = (
            f"Object size {p.object_size_nm:g} nm, ROI radius "
            f"{p.roi_radius_um:g} um")
    elif p.scan_variable == SCAN_THICKNESS:
        fixed_text = (
            f"Fixed equivalent fraction = "
            f"{100.0 * result['resolved']['fraction']:.4g}%")
    else:
        fixed_text = (
            f"Fixed effective height = {result['resolved']['height_nm']:.4g} nm")
    has_metal_particle = any(
        INCLUSION_PRESETS.get(
            item["parameters"].inclusion_name,
            (0.0, 0.0, "custom"),
        )[2].startswith("metal:")
        for item in series
    )
    if has_metal_particle:
        fixed_text += (
            "\nMetal NP model: dispersive bulk n+ik + Maxwell-Garnett"
            "\n(no Mie/LSPR scattering)")
    ax.text(0.015, 0.95, fixed_text, transform=ax.transAxes, va="top",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                    ec="0.75", alpha=0.92))
    figure.tight_layout()


def draw_geometry(figure: Figure, result: dict) -> None:
    figure.clear()
    p: Parameters = result["parameters"]
    resolved = result["resolved"]
    ax = figure.add_subplot(111)
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.6, 3.8)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax.add_patch(Rectangle((-5.0, -3.3), 10.0, 2.1,
                           facecolor="#DCE8F2", edgecolor="0.45"))
    ax.add_patch(Rectangle((-5.0, -1.2), 10.0, 0.24,
                           facecolor="#D6A51D", edgecolor="#806000"))
    ax.add_patch(Rectangle((-5.0, -0.96), 10.0, 0.50,
                           facecolor="#CCE8D3", edgecolor="#4B8B60",
                           hatch="..", alpha=0.90))
    ax.add_patch(Rectangle((-5.0, -0.46), 10.0, 3.8,
                           facecolor="#DFF3FB", edgecolor="#6BA7BF",
                           alpha=0.75))

    preset_kind = INCLUSION_PRESETS.get(p.inclusion_name, (0, 0, "custom"))[2]
    if preset_kind == "gas":
        for x, radius in [(-2.7, 0.27), (-1.5, 0.18), (-0.4, 0.23),
                          (0.8, 0.16), (1.9, 0.25), (3.0, 0.19)]:
            ax.add_patch(Circle((x, -0.36 + radius), radius,
                                facecolor="#F7FCFF", edgecolor="#0072B2",
                                lw=1.3))
        object_label = "Dilute gas domains represented by an effective layer"
    else:
        is_metal_particle = preset_kind.startswith("metal:")
        particle_face = "#D9B44A" if "Au" in preset_kind else (
            "#A7A9AC" if "Pt" in preset_kind else "#E9C46A")
        particle_edge = "#806000" if "Au" in preset_kind else (
            "#4F555A" if "Pt" in preset_kind else "#9C6A00")
        for x, radius in [(-2.8, 0.16), (-1.8, 0.22), (-0.7, 0.15),
                          (0.5, 0.20), (1.5, 0.14), (2.6, 0.23)]:
            ax.add_patch(Circle((x, -0.46 + radius), radius,
                                facecolor=particle_face, edgecolor=particle_edge,
                                lw=1.1))
        object_label = (
            "Dilute metal nanoparticles represented by an effective layer"
            if is_metal_particle else
            "Dilute dielectric particles represented by an effective layer")

    angle = np.deg2rad(55.0)
    start = np.array([-4.2, -3.0])
    end = np.array([-1.7, -1.05])
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=2.1))
    reflected = np.array([0.8, -3.0])
    ax.annotate("", xy=reflected, xytext=end,
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=2.1))
    ax.add_patch(Polygon([(-2.25, -0.95), (-1.15, -0.95), (-0.25, 0.30),
                          (-1.1, -0.25)], closed=True,
                         facecolor="#7A3E9D", edgecolor="none", alpha=0.16))

    ax.text(-4.7, -2.35, p.prism, fontsize=10, color="#31536C")
    ax.text(4.65, -1.08, f"{p.metal} {p.metal_thickness_nm:g} nm",
            ha="right", va="center", fontsize=9)
    ax.text(4.65, -0.70,
            f"Equivalent layer: {resolved['height_nm']:.3g} nm, "
            f"f={100.0 * resolved['fraction']:.4g}%",
            ha="right", va="center", fontsize=9)
    ax.text(4.65, 2.75, p.liquid, ha="right", fontsize=10,
            color="#205873")
    ax.text(0.0, 3.35, object_label, ha="center", fontsize=10)

    assumptions = (
        f"Input: {resolved['description']}\n"
        "Interpretation: spot-averaged/equivalent optical perturbation\n"
        "Included: complex refractive-index contrast, finite layer height, "
        "multilayer interference\n"
        "Not included: single-object SPP point-spread function, coherent "
        "particle scattering, lateral geometry"
    )
    ax.text(0.0, -3.48, assumptions, ha="center", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.70",
                      alpha=0.95))
    ax.set_title("Physical meaning of the effective-perturbation model", pad=8)
    figure.tight_layout()


def export_csv(path: Path, result: dict,
               overlay_results: list[dict] | None = None) -> None:
    series = [result] + list(overlay_results or [])
    scan_variable = result["parameters"].scan_variable
    scan_header = {
        SCAN_OBJECT_SIZE: "object_size_nm",
        SCAN_OBJECT_COUNT: "object_count",
        SCAN_THICKNESS: "effective_height_nm",
        SCAN_FRACTION: "volume_fraction_percent",
    }.get(scan_variable, "scan_value")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", scan_header, "delta_theta_mdeg"])
        for item in series:
            name = item["parameters"].inclusion_name
            for x, delta_theta in zip(
                    item["scan_x"], item["scan_delta_theta_mdeg"]):
                writer.writerow([name, x, delta_theta])


class NanoPerturbationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SPRM Nano-Perturbation Explorer")
        self.root.geometry("1500x900")
        self.result: dict | None = None
        self.overlay_results: list[dict] = []
        self.figures: dict[str, Figure] = {}
        self.canvases: dict[str, FigureCanvasTkAgg] = {}
        self._make_variables()
        self._build_layout()
        self.calculate()

    def _make_variables(self) -> None:
        self.v_wavelength = tk.StringVar(value="632.8")
        self.v_prism = tk.StringVar(value="SF10")
        self.v_liquid = tk.StringVar(value=DEFAULT_LIQUID)
        self.v_metal = tk.StringVar(value="Au")
        self.v_source = tk.StringVar(value=default_source("Au"))
        self.v_metal_thickness = tk.StringVar(value="48")
        self.v_inclusion = tk.StringVar(value="H2 gas domain")
        self.v_n = tk.StringVar(value="1.00013")
        self.v_k = tk.StringVar(value="0")
        self.v_representation = tk.StringVar(value=REPRESENTATION_OBJECT)
        self.v_object_geometry = tk.StringVar(value=OBJECT_BUBBLE)
        self.v_object_size = tk.StringVar(value="100")
        self.v_object_count = tk.StringVar(value="100")
        self.v_roi_radius = tk.StringVar(value="3")
        self.v_contact_angle = tk.StringVar(value="120")
        self.v_height = tk.StringVar(value="10")
        self.v_fraction_percent = tk.StringVar(value="1")
        self.v_angle_noise_sigma = tk.StringVar(
            value=f"{DEFAULT_ANGULAR_NOISE_SIGMA_MDEG:g}")
        self.v_lod_sigma_multiplier = tk.StringVar(value="3")
        self.v_angle_start = tk.StringVar(value="48")
        self.v_angle_stop = tk.StringVar(value="65")
        self.v_angle_step = tk.StringVar(value="0.005")
        self.v_tracking = tk.StringVar(value="2")
        self.v_scan_variable = tk.StringVar(value=SCAN_OBJECT_SIZE)
        self.v_scan_start = tk.StringVar(value="5")
        self.v_scan_stop = tk.StringVar(value="100")
        self.v_scan_step = tk.StringVar(value="5")
        self.v_status = tk.StringVar(value="Ready")
        self.v_metrics = tk.StringVar(value="")
        self.overlay_vars = {
            name: tk.BooleanVar(value=name in {
                "Gold nanoparticle (Au)", "Platinum nanoparticle (Pt)"})
            for name in INCLUSION_PRESETS
            if name != "Custom inclusion"
        }

    @staticmethod
    def _entry(parent, row: int, label: str, variable: tk.StringVar,
               width: int = 13) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5,
                                           pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        return entry

    def _combo(self, parent, row: int, label: str, variable: tk.StringVar,
               values, callback=None) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5,
                                           pady=3)
        combo = ttk.Combobox(parent, textvariable=variable, values=list(values),
                             state="readonly", width=27)
        combo.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        if callback is not None:
            combo.bind("<<ComboboxSelected>>", callback)
        return combo

    def _build_layout(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        controls = ttk.Frame(outer, width=350)
        plots = ttk.Frame(outer)
        outer.add(controls, weight=0)
        outer.add(plots, weight=1)

        control_tabs = ttk.Notebook(controls)
        control_tabs.pack(fill=tk.BOTH, expand=True)
        optics = ttk.Frame(control_tabs, padding=8)
        perturb = ttk.Frame(control_tabs, padding=8)
        scan = ttk.Frame(control_tabs, padding=8)
        control_tabs.add(optics, text="Optics")
        control_tabs.add(perturb, text="Perturbation")
        control_tabs.add(scan, text="Scan / export")

        optics.columnconfigure(1, weight=1)
        self._entry(optics, 0, "Wavelength (nm)", self.v_wavelength)
        self._combo(optics, 1, "Prism", self.v_prism, PRISM_OPTIONS)
        self._combo(optics, 2, "Liquid", self.v_liquid, MEDIUM_OPTIONS)
        self._combo(optics, 3, "Metal", self.v_metal, METALS,
                    self._metal_changed)
        self.source_combo = self._combo(
            optics, 4, "Optical constants", self.v_source,
            NK_DATA["Au"].keys())
        self._entry(optics, 5, "Metal thickness (nm)",
                    self.v_metal_thickness)
        ttk.Separator(optics).grid(row=6, column=0, columnspan=2, sticky="ew",
                                   pady=8)
        self._entry(optics, 7, "Angle start (deg)", self.v_angle_start)
        self._entry(optics, 8, "Angle stop (deg)", self.v_angle_stop)
        self._entry(optics, 9, "Angle step (deg)", self.v_angle_step)
        self._entry(optics, 10, "Dip tracking ± (deg)", self.v_tracking)
        self._entry(optics, 11, "Blank noise σθ (mdeg)",
                    self.v_angle_noise_sigma)
        self._entry(optics, 12, "LOD multiplier (σ)",
                    self.v_lod_sigma_multiplier)
        ttk.Label(
            optics,
            text=("Default σθ=1 mdeg is a cross-study engineering value, not "
                  "the best result from one setup. A practical controlled-"
                  "system range is roughly 0.3-3 mdeg; drift-prone systems can "
                  "be worse. Replace it with the SD of your own blank trace."),
            justify=tk.LEFT, wraplength=300,
        ).grid(row=13, column=0, columnspan=2, sticky="nw", padx=5, pady=5)

        perturb.columnconfigure(1, weight=1)
        self._combo(perturb, 0, "Representation", self.v_representation,
                    [REPRESENTATION_OBJECT, REPRESENTATION_LAYER])
        self._combo(perturb, 1, "Object geometry", self.v_object_geometry,
                    [OBJECT_BUBBLE, OBJECT_PARTICLE])
        self._combo(perturb, 2, "Inclusion preset", self.v_inclusion,
                    INCLUSION_PRESETS.keys(), self._preset_changed)
        self._entry(perturb, 3, "Inclusion n", self.v_n)
        self._entry(perturb, 4, "Inclusion k", self.v_k)
        ttk.Separator(perturb).grid(row=5, column=0, columnspan=2, sticky="ew",
                                    pady=6)
        self._entry(perturb, 6, "Object size (nm)", self.v_object_size)
        self._entry(perturb, 7, "Number in sensing ROI", self.v_object_count)
        self._entry(perturb, 8, "Sensing ROI radius (um)", self.v_roi_radius)
        self._entry(perturb, 9, "Bubble contact angle (deg)",
                    self.v_contact_angle)
        ttk.Separator(perturb).grid(row=10, column=0, columnspan=2, sticky="ew",
                                    pady=6)
        self._entry(perturb, 11, "Direct layer height (nm)", self.v_height)
        self._entry(perturb, 12, "Direct volume fraction (%)",
                    self.v_fraction_percent)
        ttk.Separator(perturb).grid(row=13, column=0, columnspan=2, sticky="ew",
                                    pady=8)
        ttk.Label(
            perturb,
            text=("Maxwell-Garnett mixing includes dielectric contrast and the\n"
                  "multilayer calculation includes finite-height interference.\n\n"
                  "For one isolated object, effective height and volume fraction\n"
                  "must be calibrated; they are not automatically its diameter\n"
                  "and projected area."),
            justify=tk.LEFT, wraplength=300,
        ).grid(row=14, column=0, columnspan=2, sticky="nw", padx=5, pady=5)

        scan.columnconfigure(1, weight=1)
        self._combo(scan, 0, "Calibration x-axis", self.v_scan_variable,
                    [SCAN_OBJECT_SIZE, SCAN_OBJECT_COUNT,
                     SCAN_THICKNESS, SCAN_FRACTION], self._scan_mode_changed)
        self._entry(scan, 1, "Scan start", self.v_scan_start)
        self._entry(scan, 2, "Scan stop", self.v_scan_stop)
        self._entry(scan, 3, "Scan step", self.v_scan_step)
        overlay_box = ttk.LabelFrame(
            scan, text="Additional preset overlays", padding=6)
        overlay_box.grid(row=4, column=0, columnspan=2, sticky="ew",
                         padx=5, pady=(10, 5))
        ttk.Label(
            overlay_box,
            text="Current inclusion is always plotted. Select any extras:",
            justify=tk.LEFT, wraplength=285,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        for index, (name, variable) in enumerate(self.overlay_vars.items()):
            ttk.Checkbutton(
                overlay_box, text=name, variable=variable,
            ).grid(row=1 + index // 2, column=index % 2,
                   sticky="w", padx=(0, 8), pady=2)
        button_row = 1 + (len(self.overlay_vars) + 1) // 2
        ttk.Button(
            overlay_box, text="Select all",
            command=lambda: self._set_all_overlays(True),
        ).grid(row=button_row, column=0, sticky="ew", padx=(0, 3), pady=(5, 0))
        ttk.Button(
            overlay_box, text="Clear",
            command=lambda: self._set_all_overlays(False),
        ).grid(row=button_row, column=1, sticky="ew", padx=(3, 0), pady=(5, 0))
        ttk.Label(
            overlay_box,
            text=("Overlays share size, count, ROI and optical settings; "
                  "gas presets use bubble geometry and particle presets use spheres."),
            justify=tk.LEFT, wraplength=285,
        ).grid(row=button_row + 1, column=0, columnspan=2,
               sticky="w", pady=(6, 0))
        ttk.Separator(scan).grid(row=5, column=0, columnspan=2, sticky="ew",
                                 pady=8)
        ttk.Button(scan, text="Calculate / update", command=self.calculate).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        ttk.Button(scan, text="Export active figure", command=self.export_figure).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=4)
        ttk.Button(scan, text="Export tracked-shift scan CSV",
                   command=self.export_data).grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=4)

        ttk.Label(controls, textvariable=self.v_metrics, justify=tk.LEFT,
                  wraplength=330).pack(fill=tk.X, padx=8, pady=(8, 3))
        ttk.Label(controls, textvariable=self.v_status, justify=tk.LEFT,
                  wraplength=330).pack(fill=tk.X, padx=8, pady=(3, 8))

        self.plot_tabs = ttk.Notebook(plots)
        self.plot_tabs.pack(fill=tk.BOTH, expand=True)
        for key, label in [
                ("reflectivity", "Reflectivity dip + ΔR"),
                ("calibration", "Perturbation calibration"),
                ("geometry", "Model geometry")]:
            frame = ttk.Frame(self.plot_tabs)
            self.plot_tabs.add(frame, text=label)
            figure = Figure(figsize=(10.2, 7.5), dpi=100)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figures[key] = figure
            self.canvases[key] = canvas

    def _metal_changed(self, _event=None) -> None:
        metal = self.v_metal.get()
        sources = list(NK_DATA[metal].keys())
        self.source_combo.configure(values=sources)
        self.v_source.set(sources[0])

    def _preset_changed(self, _event=None) -> None:
        try:
            wavelength_nm = float(self.v_wavelength.get())
            n_value, k_value, kind = preset_nk(
                self.v_inclusion.get(), wavelength_nm)
        except Exception as exc:
            self.v_status.set(f"Could not resolve preset optical constants: {exc}")
            return
        self.v_n.set(f"{n_value:g}")
        self.v_k.set(f"{k_value:g}")
        if kind == "gas":
            self.v_object_geometry.set(OBJECT_BUBBLE)
        elif kind == "particle" or kind.startswith("metal:"):
            self.v_object_geometry.set(OBJECT_PARTICLE)

    def _scan_mode_changed(self, _event=None) -> None:
        if self.v_scan_variable.get() == SCAN_OBJECT_SIZE:
            self.v_scan_start.set("5")
            self.v_scan_stop.set("100")
            self.v_scan_step.set("5")
        elif self.v_scan_variable.get() == SCAN_OBJECT_COUNT:
            self.v_scan_start.set("1")
            self.v_scan_stop.set("100")
            self.v_scan_step.set("1")
        elif self.v_scan_variable.get() == SCAN_THICKNESS:
            self.v_scan_start.set("0")
            self.v_scan_stop.set("100")
            self.v_scan_step.set("2")
        else:
            self.v_scan_start.set("0")
            self.v_scan_stop.set("20")
            self.v_scan_step.set("0.5")

    def _set_all_overlays(self, selected: bool) -> None:
        for variable in self.overlay_vars.values():
            variable.set(selected)

    def _selected_overlay_names(self, primary_name: str) -> list[str]:
        return [
            name for name, variable in self.overlay_vars.items()
            if variable.get() and name != primary_name
        ]

    def _parameters(self) -> Parameters:
        wavelength_nm = float(self.v_wavelength.get())
        inclusion_name = self.v_inclusion.get()
        inclusion_n = float(self.v_n.get())
        inclusion_k = float(self.v_k.get())
        preset_kind = INCLUSION_PRESETS.get(
            inclusion_name, (0.0, 0.0, "custom"))[2]
        if preset_kind.startswith("metal:"):
            inclusion_n, inclusion_k, _ = preset_nk(
                inclusion_name, wavelength_nm)
            self.v_n.set(f"{inclusion_n:g}")
            self.v_k.set(f"{inclusion_k:g}")
        return Parameters(
            wavelength_nm=wavelength_nm,
            prism=self.v_prism.get(),
            liquid=self.v_liquid.get(),
            metal=self.v_metal.get(),
            metal_source=self.v_source.get(),
            metal_thickness_nm=float(self.v_metal_thickness.get()),
            inclusion_name=inclusion_name,
            inclusion_n=inclusion_n,
            inclusion_k=inclusion_k,
            representation=self.v_representation.get(),
            object_geometry=self.v_object_geometry.get(),
            object_size_nm=float(self.v_object_size.get()),
            object_count=int(float(self.v_object_count.get())),
            roi_radius_um=float(self.v_roi_radius.get()),
            bubble_contact_angle_deg=float(self.v_contact_angle.get()),
            effective_height_nm=float(self.v_height.get()),
            volume_fraction=float(self.v_fraction_percent.get()) / 100.0,
            angular_noise_sigma_mdeg=float(self.v_angle_noise_sigma.get()),
            lod_sigma_multiplier=float(self.v_lod_sigma_multiplier.get()),
            angle_start_deg=float(self.v_angle_start.get()),
            angle_stop_deg=float(self.v_angle_stop.get()),
            angle_step_deg=float(self.v_angle_step.get()),
            tracking_window_deg=float(self.v_tracking.get()),
            scan_variable=self.v_scan_variable.get(),
            scan_start=float(self.v_scan_start.get()),
            scan_stop=float(self.v_scan_stop.get()),
            scan_step=float(self.v_scan_step.get()),
        )

    def calculate(self) -> None:
        try:
            self.v_status.set("Calculating...")
            self.root.update_idletasks()
            primary_parameters = self._parameters()
            result = calculate(primary_parameters)
            overlay_results = [
                calculate(parameters_for_preset(primary_parameters, name))
                for name in self._selected_overlay_names(
                    primary_parameters.inclusion_name)
            ]
            draw_reflectivity(
                self.figures["reflectivity"], result, overlay_results)
            draw_calibration(
                self.figures["calibration"], result, overlay_results)
            draw_geometry(self.figures["geometry"], result)
            for canvas in self.canvases.values():
                canvas.draw_idle()
            self.result = result
            self.overlay_results = overlay_results
            n_eff = result["n_eff"]
            resolved = result["resolved"]
            if np.isfinite(resolved["projected_coverage"]):
                population_text = (
                    f"Areal density: {resolved['areal_density_per_um2']:.3f} "
                    f"objects/um²; projected coverage: "
                    f"{100.0 * resolved['projected_coverage']:.3f}%\n")
            else:
                population_text = ""
            if result["minimum_detectable_x"] is None:
                minimum_text = "No LOD crossing in selected scan range"
            elif result["parameters"].scan_variable == SCAN_OBJECT_SIZE:
                minimum_text = (
                    f"Estimated minimum detectable diameter: "
                    f"{result['minimum_detectable_x']:.1f} nm")
            else:
                minimum_text = (
                    f"First detectable scan value: "
                    f"{result['minimum_detectable_x']:.2f}")
            self.v_metrics.set(
                f"Plotted inclusions: {1 + len(overlay_results)} "
                f"(primary: {result['parameters'].inclusion_name})\n"
                f"Resolved geometry: {resolved['description']}\n"
                f"{population_text}"
                f"Equivalent layer: {resolved['height_nm']:.3f} nm, "
                f"{100.0 * resolved['fraction']:.5f}%\n"
                f"Baseline dip: {result['baseline_dip'][0]:.4f} deg\n"
                f"Perturbed dip: {result['perturbed_dip'][0]:.4f} deg\n"
                f"Dip shift: {result['delta_theta_mdeg']:+.2f} mdeg\n"
                f"ΔR at {result['optimum_angle']:.3f} deg: "
                f"{result['optimum_delta_r']:+.4e}\n"
                f"Angular LOD: {result['angle_lod_mdeg']:.3f} mdeg "
                f"({result['parameters'].lod_sigma_multiplier:g}σ)\n"
                f"{minimum_text}\n"
                f"Effective n+ik: {np.real(n_eff):.6f} "
                f"{np.imag(n_eff):+.2e}i")
            self.v_status.set(
                "Updated. Angular criterion: "
                f"{'detectable' if result['detectable_by_angle'] else 'below the selected limit'}. "
                "For one nano-object, the local SPRM contrast is more rigorous than this "
                "spot-averaged angle estimate.")
        except Exception as exc:
            self.v_status.set("Calculation failed.")
            messagebox.showerror("Calculation error", str(exc), parent=self.root)

    def _active_key(self) -> str:
        index = self.plot_tabs.index(self.plot_tabs.select())
        return ["reflectivity", "calibration", "geometry"][index]

    def export_figure(self) -> None:
        if self.result is None:
            return
        key = self._active_key()
        initial = {
            "reflectivity": DEFAULT_FIGURE,
            "calibration": DEFAULT_SCAN_FIGURE,
            "geometry": "SPR_nano_perturbation_geometry_default.png",
        }[key]
        filename = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".png", initialfile=initial,
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg")])
        if filename:
            self.figures[key].savefig(filename, dpi=300, bbox_inches="tight")
            self.v_status.set(f"Saved figure: {filename}")

    def export_data(self) -> None:
        if self.result is None:
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".csv", initialfile=DEFAULT_CSV,
            filetypes=[("CSV data", "*.csv")])
        if filename:
            export_csv(Path(filename), self.result, self.overlay_results)
            self.v_status.set(f"Saved data: {filename}")


def default_parameters() -> Parameters:
    return Parameters(metal_source=default_source("Au"))


def export_default(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = calculate(default_parameters())
    figure = Figure(figsize=(9.2, 7.2), dpi=120)
    draw_reflectivity(figure, result)
    figure.savefig(output_dir / DEFAULT_FIGURE, dpi=300, bbox_inches="tight")
    scan_figure = Figure(figsize=(9.2, 7.2), dpi=120)
    draw_calibration(scan_figure, result)
    scan_figure.savefig(output_dir / DEFAULT_SCAN_FIGURE, dpi=300,
                        bbox_inches="tight")
    geometry_figure = Figure(figsize=(9.2, 6.6), dpi=120)
    draw_geometry(geometry_figure, result)
    geometry_figure.savefig(
        output_dir / "SPR_nano_perturbation_geometry_default.png",
        dpi=300, bbox_inches="tight")
    export_csv(output_dir / DEFAULT_CSV, result)
    print(f"Reflectivity figure: {output_dir / DEFAULT_FIGURE}")
    print(f"Calibration figure: {output_dir / DEFAULT_SCAN_FIGURE}")
    print(f"Geometry figure: {output_dir / 'SPR_nano_perturbation_geometry_default.png'}")
    print(f"CSV data: {output_dir / DEFAULT_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-default", type=Path,
        help="Export default figures and CSV without opening the GUI.")
    args = parser.parse_args()
    if args.export_default is not None:
        export_default(args.export_default)
        return
    root = tk.Tk()
    NanoPerturbationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
