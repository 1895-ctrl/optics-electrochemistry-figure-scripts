#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illustrate the TIRF mechanism with a GUI and reproducible data export.

This companion script reuses the validated single-interface physics in
``evanescent_depth_gui.py``.  Curve shape and critical angle are parameterized
by ``R = n1/n2 > 1``.  Physical penetration depth additionally uses one common,
adjustable reference n1 because d scales as 1/n1.  The default vacuum
wavelengths are 532 and 633 nm.  A separate n1*d view is fully ratio-only.

Run
    python TIRF_mechanism_gui.py
    python TIRF_mechanism_gui.py --preview tirf_mechanism_preview.png
    python TIRF_mechanism_gui.py --selftest

The script must remain beside ``evanescent_depth_gui.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

try:
    from evanescent_depth_gui import (
        OKABE,
        critical_angle_deg,
        interface_intensity,
    )
except ImportError as exc:
    raise SystemExit(
        "tirf_mechanism_gui.py must be placed beside evanescent_depth_gui.py "
        "and requires NumPy/Matplotlib.\n" + str(exc)
    ) from exc

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


VERSION = "1.4"
DEFAULTS = {
    "ratios": "1.1, 1.3, 1.5",
    "reference_n1": 1.52,
    "wavelengths": "532, 633",
    "theta_min": 45.0,
    "theta_max": 89.0,
    "theta_points": 900,
    "probe_theta": 70.0,
    "z_max": 3.0,
    "z_points": 900,
    "quantity": "Physical penetration depth vs incidence angle",
    "convention": "intensity",
    "polarization": "s and p",
    "xscale": "linear",
    "yscale": "log",
    "xmin": "",
    "xmax": "",
    "ymin": "",
    "ymax": 180.0,
    "show_critical": True,
    "show_floor": True,
    "show_grid": True,
}

QUANTITIES = (
    "Physical penetration depth vs incidence angle",
    "Index-scaled penetration depth vs incidence angle",
    "Interface intensity vs incidence angle",
    "Normalized axial decay vs distance",
)
POLARIZATIONS = ("s and p", "s only", "p only")
LINESTYLES = ("-", "--", "-.", ":")


def parse_positive_list(value, label, maximum=12):
    """Parse comma/semicolon-separated finite positive values."""
    text = str(value).strip().replace("\uFF0C", ",").replace(";", ",")
    if not text:
        raise ValueError(f"{label} cannot be empty")
    parts = [item.strip() for item in text.split(",")]
    if any(not item for item in parts):
        raise ValueError(f"{label} contains an empty item")
    try:
        values = tuple(float(item) for item in parts)
    except ValueError as exc:
        raise ValueError(f"{label} must contain numbers separated by commas") from exc
    if len(values) > maximum:
        raise ValueError(f"{label} allows at most {maximum} values")
    if not all(np.isfinite(item) and item > 0.0 for item in values):
        raise ValueError(f"{label} must contain finite positive numbers")
    return values


def optional_float(value, label):
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number or blank for Auto") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass
class PlotSettings:
    ratios: tuple[float, ...]
    reference_n1: float
    wavelengths: tuple[float, ...]
    theta_min: float
    theta_max: float
    theta_points: int
    probe_theta: float
    z_max: float
    z_points: int
    quantity: str
    convention: str
    polarization: str
    xscale: str
    yscale: str
    xlim: tuple[float | None, float | None]
    ylim: tuple[float | None, float | None]
    show_critical: bool
    show_floor: bool
    show_grid: bool


def make_settings(raw):
    ratios = parse_positive_list(raw["ratios"], "n1/n2 ratios", maximum=12)
    if not all(ratio > 1.0 for ratio in ratios):
        raise ValueError("every n1/n2 ratio must be > 1 for total internal reflection")
    wavelengths = parse_positive_list(raw["wavelengths"], "wavelengths", maximum=8)
    reference_n1 = float(raw["reference_n1"])
    if not np.isfinite(reference_n1) or reference_n1 <= 0.0:
        raise ValueError("reference n1 must be a finite positive number")
    theta_min = float(raw["theta_min"])
    theta_max = float(raw["theta_max"])
    theta_points = int(float(raw["theta_points"]))
    probe_theta = float(raw["probe_theta"])
    z_max = float(raw["z_max"])
    z_points = int(float(raw["z_points"]))

    if not (0.0 < theta_min < theta_max < 90.0):
        raise ValueError("require 0 < theta min < theta max < 90 deg")
    if not (2 <= theta_points <= 200_000):
        raise ValueError("angle points must lie between 2 and 200000")
    if not (0.0 < probe_theta < 90.0):
        raise ValueError("probe angle must lie between 0 and 90 deg")
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("normalized z max must be > 0")
    if not (2 <= z_points <= 200_000):
        raise ValueError("z points must lie between 2 and 200000")
    if raw["quantity"] not in QUANTITIES:
        raise ValueError("unknown plot quantity")
    if raw["convention"] not in ("intensity", "amplitude"):
        raise ValueError("unknown depth convention")
    if raw["polarization"] not in POLARIZATIONS:
        raise ValueError("unknown polarization selection")
    if raw["xscale"] not in ("linear", "log") or raw["yscale"] not in ("linear", "log"):
        raise ValueError("axis scale must be linear or log")

    xlim = (optional_float(raw.get("xmin", ""), "x min"),
            optional_float(raw.get("xmax", ""), "x max"))
    ylim = (optional_float(raw.get("ymin", ""), "y min"),
            optional_float(raw.get("ymax", ""), "y max"))
    for limits, label in ((xlim, "x"), (ylim, "y")):
        if limits[0] is not None and limits[1] is not None and limits[0] >= limits[1]:
            raise ValueError(f"{label} min must be smaller than {label} max")
    if raw["xscale"] == "log" and any(v is not None and v <= 0 for v in xlim):
        raise ValueError("logarithmic x-axis limits must be > 0")
    if raw["yscale"] == "log" and any(v is not None and v <= 0 for v in ylim):
        raise ValueError("logarithmic y-axis limits must be > 0")

    return PlotSettings(
        ratios=ratios, reference_n1=reference_n1, wavelengths=wavelengths,
        theta_min=theta_min, theta_max=theta_max, theta_points=theta_points,
        probe_theta=probe_theta, z_max=z_max, z_points=z_points,
        quantity=raw["quantity"], convention=raw["convention"],
        polarization=raw["polarization"], xscale=raw["xscale"],
        yscale=raw["yscale"], xlim=xlim, ylim=ylim,
        show_critical=bool(raw["show_critical"]), show_floor=bool(raw["show_floor"]),
        show_grid=bool(raw["show_grid"]),
    )


def _ratio_label(ratio):
    return rf"$R$={ratio:.5g}"


def _depth_symbol(convention):
    return r"$d_I$" if convention == "intensity" else r"$d_E$"


def normalized_depth(ratio, theta_deg, convention="intensity"):
    """Return d/lambda1, where lambda1=lambda0/n1; ratio is R=n1/n2 > 1."""
    theta = np.radians(np.asarray(theta_deg, dtype=float))
    radicand = np.sin(theta) ** 2 - (1.0 / float(ratio)) ** 2
    factor = 4.0 if convention == "intensity" else 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 / (factor * np.pi * np.sqrt(np.where(radicand > 0.0, radicand, np.nan)))


def normalized_depth_floor(ratio, convention="intensity"):
    """Grazing-incidence limit of d/lambda1."""
    factor = 4.0 if convention == "intensity" else 2.0
    return 1.0 / (factor * np.pi * np.sqrt(1.0 - (1.0 / float(ratio)) ** 2))


def index_scaled_depth_nm(wavelength_nm, ratio, theta_deg, convention="intensity"):
    """Return n1*d in nm; this depends only on ratio and vacuum wavelength."""
    return float(wavelength_nm) * normalized_depth(ratio, theta_deg, convention)


def physical_depth_nm(wavelength_nm, reference_n1, ratio, theta_deg,
                      convention="intensity"):
    """Return physical d in nm using one common reference n1 for all ratio curves."""
    return index_scaled_depth_nm(wavelength_nm, ratio, theta_deg, convention) / float(reference_n1)


def compute_rows(settings):
    """Return CSV-ready dictionaries for the selected plot."""
    rows = []
    theta = np.linspace(settings.theta_min, settings.theta_max, settings.theta_points)
    if settings.quantity in QUANTITIES[:2]:
        physical = settings.quantity == QUANTITIES[0]
        for curve_id, ratio in enumerate(settings.ratios, 1):
            thc = float(critical_angle_deg(ratio, 1.0))
            for lam in settings.wavelengths:
                scaled = index_scaled_depth_nm(lam, ratio, theta, settings.convention)
                depth = scaled / settings.reference_n1 if physical else scaled
                floor_scaled = float(index_scaled_depth_nm(
                    lam, ratio, 90.0, settings.convention))
                floor = floor_scaled / settings.reference_n1 if physical else floor_scaled
                for xv, yv in zip(theta, depth):
                    rows.append(dict(curve_id=curve_id, ratio_n1_over_n2=float(ratio),
                                     reference_n1=(settings.reference_n1 if physical else np.nan),
                                     wavelength_nm=float(lam), theta1_deg=float(xv),
                                     theta_c_deg=thc,
                                     depth_nm=float(yv) if physical else np.nan,
                                     n1_times_depth_nm=float(yv) if not physical else float(yv * settings.reference_n1),
                                     grazing_floor_nm=float(floor),
                                     is_TIR=int(np.isfinite(yv))))
    elif settings.quantity == QUANTITIES[2]:
        for curve_id, ratio in enumerate(settings.ratios, 1):
            thc = float(critical_angle_deg(ratio, 1.0))
            Is, Ip, Ix, Iz = interface_intensity(ratio, 1.0, theta)
            for lam in settings.wavelengths:
                for xv, ys, yp, yx, yz in zip(theta, Is, Ip, Ix, Iz):
                    rows.append(dict(curve_id=curve_id, ratio_n1_over_n2=float(ratio),
                                     wavelength_nm=float(lam), theta1_deg=float(xv),
                                     theta_c_deg=thc, I_s=float(ys), I_p=float(yp),
                                     I_p_tangential=float(yx), I_p_normal=float(yz),
                                     is_TIR=int(np.isfinite(ys))))
    else:
        u = np.linspace(0.0, settings.z_max, settings.z_points)
        for curve_id, ratio in enumerate(settings.ratios, 1):
            thc = float(critical_angle_deg(ratio, 1.0))
            for lam in settings.wavelengths:
                depth = float(normalized_depth(ratio, settings.probe_theta, settings.convention))
                y = np.exp(-u / depth) if np.isfinite(depth) else np.full_like(u, np.nan)
                for uv, yv in zip(u, y):
                    rows.append(dict(curve_id=curve_id, ratio_n1_over_n2=float(ratio),
                                     wavelength_nm=float(lam),
                                     theta1_deg=settings.probe_theta,
                                     theta_c_deg=thc, depth_over_lambda1=depth,
                                     z_over_lambda1=float(uv), normalized_signal=float(yv),
                                     is_TIR=int(np.isfinite(depth))))
    return rows


def _apply_axes(ax, settings):
    ax.set_xscale(settings.xscale)
    ax.set_yscale(settings.yscale)
    if settings.xlim[0] is not None or settings.xlim[1] is not None:
        ax.set_xlim(left=settings.xlim[0], right=settings.xlim[1])
    if settings.ylim[0] is not None or settings.ylim[1] is not None:
        ax.set_ylim(bottom=settings.ylim[0], top=settings.ylim[1])
    ax.grid(settings.show_grid, which="both", alpha=0.28)


def _add_legends(ax, settings, include_wavelength=True, include_pol=False,
                 combine_wavelengths=False, wavelengths_coincident=True):
    ratio_handles = [Line2D([0], [0], color=OKABE[k % len(OKABE)], lw=2.2,
                            label=_ratio_label(ratio))
                     for k, ratio in enumerate(settings.ratios)]
    first = ax.legend(handles=ratio_handles, title=r"Index ratio $R=n_1/n_2$",
                      loc="upper right", fontsize=8, title_fontsize=8,
                      framealpha=0.92)
    ax.add_artist(first)

    if include_wavelength:
        if combine_wavelengths:
            wave_text = " = ".join(f"{lam:g} nm" for lam in settings.wavelengths)
            wave_handles = [Line2D([0], [0], color="0.2", lw=2.0,
                                   label=wave_text + " (coincident)")]
        else:
            suffix = " (coincident)" if wavelengths_coincident else ""
            wave_handles = [Line2D([0], [0], color="0.2", lw=2.0,
                                   ls=LINESTYLES[k % len(LINESTYLES)],
                                   label=f"{lam:g} nm{suffix}")
                            for k, lam in enumerate(settings.wavelengths)]
        second = ax.legend(handles=wave_handles, title="Vacuum wavelength",
                           loc="upper center", fontsize=8, title_fontsize=8,
                           framealpha=0.92)
        ax.add_artist(second)
    if include_pol:
        pol_handles = []
        if settings.polarization in ("s and p", "s only"):
            pol_handles.append(Line2D([0], [0], color="0.2", lw=2.0, ls="-", label="s polarization"))
        if settings.polarization in ("s and p", "p only"):
            pol_handles.append(Line2D([0], [0], color="0.2", lw=2.0, ls="--", label="p polarization"))
        ax.legend(handles=pol_handles, title="Field", loc="upper left",
                  fontsize=8, title_fontsize=8, framealpha=0.92)


def draw_figure(fig, settings):
    fig.clear()
    ax = fig.subplots(1, 1)
    status = []

    if settings.quantity in QUANTITIES[:2]:  # physical or index-scaled depth
        theta = np.linspace(settings.theta_min, settings.theta_max, settings.theta_points)
        physical = settings.quantity == QUANTITIES[0]
        for k, ratio in enumerate(settings.ratios):
            colour = OKABE[k % len(OKABE)]
            thc = float(critical_angle_deg(ratio, 1.0))
            for j, lam in enumerate(settings.wavelengths):
                scaled = index_scaled_depth_nm(lam, ratio, theta, settings.convention)
                depth = scaled / settings.reference_n1 if physical else scaled
                ax.plot(theta, depth, color=colour,
                        ls=LINESTYLES[j % len(LINESTYLES)], lw=2.0)
                if settings.show_floor:
                    floor = float(index_scaled_depth_nm(
                        lam, ratio, 90.0, settings.convention))
                    if physical:
                        floor /= settings.reference_n1
                    ax.axhline(floor, color=colour,
                               ls=LINESTYLES[j % len(LINESTYLES)],
                               lw=0.8, alpha=0.35)
            if settings.show_critical:
                ax.axvline(thc, color=colour, ls=":", lw=1.0, alpha=0.65)
        ax.set_xlabel(r"Incidence angle $\theta_1$ (deg, inside medium 1)")
        if physical:
            ax.set_ylabel(f"Physical {_depth_symbol(settings.convention)} penetration depth (nm)")
            ax.set_title(
                rf"TIRF penetration depth versus incidence angle, reference $n_1$={settings.reference_n1:g}" + "\n"
                r"colour = $n_1/n_2$ ratio; line style = vacuum wavelength")
            status.append("all ratio curves use one common reference n1")
        else:
            ax.set_ylabel(rf"Index-scaled depth $n_1${_depth_symbol(settings.convention)} (nm)")
            ax.set_title("TIRF index-scaled penetration depth: ratio-only representation\n"
                         r"physical depth is obtained by dividing the y value by $n_1$")
            status.append("ratio-only y axis is n1*d; divide by the actual n1 for physical nm")
        _add_legends(ax, settings, include_wavelength=True,
                     wavelengths_coincident=False)

    elif settings.quantity == QUANTITIES[2]:  # interface intensity
        theta = np.linspace(settings.theta_min, settings.theta_max, settings.theta_points)
        for k, ratio in enumerate(settings.ratios):
            colour = OKABE[k % len(OKABE)]
            Is, Ip, _Ix, _Iz = interface_intensity(ratio, 1.0, theta)
            if settings.polarization in ("s and p", "s only"):
                ax.plot(theta, Is, color=colour, ls="-", lw=2.0)
            if settings.polarization in ("s and p", "p only"):
                ax.plot(theta, Ip, color=colour, ls="--", lw=2.0)
            if settings.show_critical:
                ax.axvline(float(critical_angle_deg(ratio, 1.0)), color=colour,
                           ls=":", lw=1.0, alpha=0.65)
        ax.set_xlabel(r"Incidence angle $\theta_1$ (deg, inside medium 1)")
        ax.set_ylabel(r"Normalized interface intensity $|E(0)|^2/|E_{inc}|^2$")
        ax.set_title(r"TIRF interface intensity versus incidence angle: ratio-only model" + "\n"
                     r"same $n_1/n_2$ ratio -> identical curve at 532 and 633 nm")
        _add_legends(ax, settings, include_wavelength=True, include_pol=True,
                     combine_wavelengths=True)
        status.append("532 and 633 nm coincide exactly because r is held fixed")

    else:
        u = np.linspace(0.0, settings.z_max, settings.z_points)
        valid = 0
        for k, ratio in enumerate(settings.ratios):
            colour = OKABE[k % len(OKABE)]
            thc = float(critical_angle_deg(ratio, 1.0))
            for j, _lam in enumerate(settings.wavelengths):
                depth = float(normalized_depth(ratio, settings.probe_theta, settings.convention))
                if np.isfinite(depth):
                    ax.plot(u, np.exp(-u / depth), color=colour,
                            ls=LINESTYLES[j % len(LINESTYLES)],
                            lw=2.8 if j == 0 else 1.7, alpha=0.85)
                    valid += 1
                else:
                    status.append(f"R={ratio:g}: probe angle <= theta_c={thc:.3f} deg")
        if not valid:
            ax.text(0.5, 0.5, "Probe angle is below every critical angle:\nno total internal reflection",
                    ha="center", va="center", transform=ax.transAxes, color=OKABE[1], fontsize=12)
        ax.set_xlabel(r"Normalized distance $z/\lambda_1$,  $\lambda_1=\lambda_0/n_1$")
        signal = "intensity" if settings.convention == "intensity" else "field amplitude"
        ax.set_ylabel(rf"Normalized {signal}")
        ax.set_title(rf"TIRF normalized axial decay at $\theta_1$={settings.probe_theta:g}$^\circ$" + "\n"
                     "same ratio gives the same 532/633 nm curve")
        _add_legends(ax, settings, include_wavelength=True)
        status.append("physical z in nm additionally requires an absolute n1")

    _apply_axes(ax, settings)
    if settings.quantity == QUANTITIES[0]:
        footer = (f"R=n1/n2; common reference n1={settings.reference_n1:g}; "
                  "532/633 nm are separate physical-depth curves.")
    elif settings.quantity == QUANTITIES[1]:
        footer = "Ratio-only representation n1*d; 532/633 nm are separate curves."
    else:
        footer = "Ratio-only model: 532/633 nm overlap when R is held fixed."
    footer += "\nLossless dielectric interface; not for metals, absorption, or multilayers."
    fig.text(0.5, 0.014, footer, ha="center", va="bottom",
             fontsize=7.4, color="0.3")
    # Fixed margins keep the two-line title and two-line footer inside a 4:3
    # canvas on Windows, where tight_layout can underestimate math-text height.
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.84)
    return "  |  ".join(dict.fromkeys(status))


def export_csv(path, settings):
    rows = compute_rows(settings)
    if not rows:
        raise ValueError("no data to export")
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        handle.write(f"# tirf_mechanism_gui.py v{VERSION}\n")
        handle.write(f"# quantity: {settings.quantity}\n")
        handle.write(f"# curve parameter: R = n1/n2 > 1; reference n1 = {settings.reference_n1:.9g}\n")
        handle.write("# physical depth uses the common reference n1; index-scaled depth n1*d is ratio-only\n")
        handle.write(f"# convention: {settings.convention}\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_selftests(verbose=True):
    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))

    p = make_settings(dict(DEFAULTS))
    check("default wavelengths are 532 and 633 nm", p.wavelengths == (532.0, 633.0))
    check("three rounded ratio curves are configured", p.ratios == (1.1, 1.3, 1.5))
    check("default physical-depth y maximum is 180 nm", p.ylim[1] == 180.0)
    check("all ratios satisfy n1/n2 > 1", all(ratio > 1.0 for ratio in p.ratios))
    th = np.linspace(65.0, 85.0, 50)
    Is_a, Ip_a, *_ = interface_intensity(1.5, 1.2, th)
    Is_b, Ip_b, *_ = interface_intensity(2.0, 1.6, th)
    check("same ratio gives identical s-interface intensity",
          np.allclose(Is_a, Is_b, equal_nan=True))
    check("same ratio gives identical p-interface intensity",
          np.allclose(Ip_a, Ip_b, equal_nan=True))
    dnorm = normalized_depth(1.25, th, "intensity")
    check("normalized depth is finite above critical angle", np.all(np.isfinite(dnorm)))
    check("normalized depth is NaN below critical angle",
          np.isnan(float(normalized_depth(1.25, 40.0, "intensity"))))
    d532 = float(physical_depth_nm(532.0, p.reference_n1, 1.25, 70.0))
    d633 = float(physical_depth_nm(633.0, p.reference_n1, 1.25, 70.0))
    check("physical depth scales with vacuum wavelength",
          abs(d633 / d532 - 633.0 / 532.0) < 1e-12)
    d_n15 = float(physical_depth_nm(532.0, 1.5, 1.25, 70.0))
    d_n20 = float(physical_depth_nm(532.0, 2.0, 1.25, 70.0))
    check("same ratio physical depth scales inversely with reference n1",
          abs(d_n15 / d_n20 - 2.0 / 1.5) < 1e-12)
    rows = compute_rows(p)
    check("export contains ratio x wavelength x angle rows",
          len(rows) == len(p.ratios) * len(p.wavelengths) * DEFAULTS["theta_points"])
    other_views_ok = True
    for quantity in QUANTITIES[1:]:
        view = make_settings(dict(DEFAULTS, quantity=quantity))
        view_rows = compute_rows(view)
        expected_points = (DEFAULTS["z_points"] if quantity == QUANTITIES[-1]
                           else DEFAULTS["theta_points"])
        other_views_ok &= (len(view_rows)
                           == len(view.ratios) * len(view.wavelengths) * expected_points)
        test_fig = Figure(figsize=(4, 3), dpi=60)
        draw_figure(test_fig, view)
    check("scaled depth, interface, and axial views compute and render", other_views_ok)

    if verbose:
        for name, passed in checks:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print(f"{sum(passed for _, passed in checks)}/{len(checks)} passed")
    return all(passed for _, passed in checks)


def preview(path):
    settings = make_settings(dict(DEFAULTS))
    fig = Figure(figsize=(8, 6), dpi=130)
    draw_figure(fig, settings)
    fig.savefig(path, dpi=180)
    return settings


def launch_gui():  # pragma: no cover - interactive UI
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title(f"TIRF mechanism: n1/n2 ratio comparison  v{VERSION}")
    root.geometry("1540x900")
    root.minsize(1120, 700)

    control_host = ttk.Frame(root, width=390)
    control_host.pack(side="left", fill="y")
    control_host.pack_propagate(False)
    control_canvas = tk.Canvas(control_host, borderwidth=0, highlightthickness=0, width=370)
    scroll = ttk.Scrollbar(control_host, orient="vertical", command=control_canvas.yview)
    control_canvas.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    control_canvas.pack(side="left", fill="both", expand=True)
    controls = ttk.Frame(control_canvas)
    window_id = control_canvas.create_window((0, 0), window=controls, anchor="nw")
    controls.bind("<Configure>", lambda _e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
    control_canvas.bind("<Configure>", lambda e: control_canvas.itemconfigure(window_id, width=e.width))

    vars_ = {}

    def section(title):
        ttk.Separator(controls).pack(fill="x", pady=(9, 3))
        ttk.Label(controls, text=title, font=("", 9, "bold")).pack(anchor="w", padx=9)

    def field(key, label, initial, width=18):
        frame = ttk.Frame(controls)
        frame.pack(fill="x", padx=9, pady=2)
        ttk.Label(frame, text=label, width=23).pack(side="left")
        var = tk.StringVar(value=str(initial))
        ttk.Entry(frame, textvariable=var, width=width).pack(side="left", fill="x", expand=True)
        vars_[key] = var
        return var

    section("Curves")
    field("ratios", "R = n1/n2 > 1", DEFAULTS["ratios"], 25)
    field("reference_n1", "reference n1 (physical d)", DEFAULTS["reference_n1"], 25)
    field("wavelengths", "lambda0 values (nm)", DEFAULTS["wavelengths"], 25)
    ttk.Label(controls,
              text="Curve identity is set by R=n1/n2 > 1.\n"
                   "One common reference n1 converts the ratio curves to nm.\n"
                   "Use the index-scaled view when no absolute n1 is wanted.",
              foreground="#555", font=("", 7), justify="left").pack(anchor="w", padx=9)

    section("Plot")
    vars_["quantity"] = tk.StringVar(value=DEFAULTS["quantity"])
    ttk.Combobox(controls, textvariable=vars_["quantity"], values=QUANTITIES,
                 state="readonly", width=43).pack(anchor="w", padx=9, pady=2)
    field("theta_min", "theta1 min (deg)", DEFAULTS["theta_min"])
    field("theta_max", "theta1 max (deg)", DEFAULTS["theta_max"])
    field("theta_points", "theta1 points", DEFAULTS["theta_points"])
    field("probe_theta", "profile theta1 (deg)", DEFAULTS["probe_theta"])
    field("z_max", "profile z/lambda1 max", DEFAULTS["z_max"])
    field("z_points", "profile z points", DEFAULTS["z_points"])

    vars_["convention"] = tk.StringVar(value=DEFAULTS["convention"])
    ttk.Radiobutton(controls, text="intensity 1/e depth  dI = lambda0/(4 pi Q)",
                    variable=vars_["convention"], value="intensity").pack(anchor="w", padx=9)
    ttk.Radiobutton(controls, text="field 1/e depth  dE = lambda0/(2 pi Q)",
                    variable=vars_["convention"], value="amplitude").pack(anchor="w", padx=9)
    vars_["polarization"] = tk.StringVar(value=DEFAULTS["polarization"])
    pol_frame = ttk.Frame(controls)
    pol_frame.pack(fill="x", padx=9, pady=2)
    ttk.Label(pol_frame, text="interface polarization", width=23).pack(side="left")
    ttk.Combobox(pol_frame, textvariable=vars_["polarization"], values=POLARIZATIONS,
                 state="readonly", width=16).pack(side="left")

    section("Axes")
    for key, label, default in (("xscale", "x scale", DEFAULTS["xscale"]),
                                ("yscale", "y scale", DEFAULTS["yscale"])):
        frame = ttk.Frame(controls)
        frame.pack(fill="x", padx=9, pady=2)
        ttk.Label(frame, text=label, width=23).pack(side="left")
        vars_[key] = tk.StringVar(value=default)
        ttk.Combobox(frame, textvariable=vars_[key], values=("linear", "log"),
                     state="readonly", width=10).pack(side="left")
    field("xmin", "x min (blank = Auto)", DEFAULTS["xmin"])
    field("xmax", "x max (blank = Auto)", DEFAULTS["xmax"])
    field("ymin", "y min (blank = Auto)", DEFAULTS["ymin"])
    field("ymax", "y max (blank = Auto)", DEFAULTS["ymax"])
    for key, label in (("show_critical", "Show critical angles"),
                       ("show_floor", "Show grazing depth floors"),
                       ("show_grid", "Show grid")):
        vars_[key] = tk.BooleanVar(value=DEFAULTS[key])
        ttk.Checkbutton(controls, text=label, variable=vars_[key]).pack(anchor="w", padx=9)

    right = ttk.Frame(root)
    right.pack(side="right", fill="both", expand=True)
    fig = Figure(figsize=(8, 6), dpi=110)
    canvas = FigureCanvasTkAgg(fig, master=right)
    toolbar_host = ttk.Frame(right)
    toolbar_host.pack(side="bottom", fill="x")
    NavigationToolbar2Tk(canvas, toolbar_host).update()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    status = tk.Label(root, text="", anchor="w", padx=8, pady=5)
    status.pack(side="bottom", fill="x")
    state = {"settings": None}

    def set_status(kind, message):
        colours = {"ok": ("#0b6b3a", "#e3f5ea"),
                   "warn": ("#8a5a00", "#fdf3dd"),
                   "err": ("#8a1c1c", "#fbe4e4")}
        fg, bg = colours[kind]
        status.configure(text=message, fg=fg, bg=bg)

    def raw_values():
        return {key: var.get() for key, var in vars_.items()}

    def update_plot(*_args):
        try:
            settings = make_settings(raw_values())
            note = draw_figure(fig, settings)
            canvas.draw_idle()
            state["settings"] = settings
            wave_note = ("wavelength curve(s)" if settings.quantity in QUANTITIES[:2]
                         else "coincident wavelength label(s)")
            msg = (f"Ready: {len(settings.ratios)} ratio curve(s), "
                   f"{len(settings.wavelengths)} {wave_note}")
            set_status("warn" if note else "ok", msg + (("  |  " + note) if note else ""))
        except Exception as exc:
            set_status("err", f"{type(exc).__name__}: {exc}")

    def reset():
        for key, value in DEFAULTS.items():
            vars_[key].set(value)
        update_plot()

    def save_figure():
        if state["settings"] is None:
            update_plot()
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG image", "*.png"),
                                                       ("PDF vector", "*.pdf"),
                                                       ("SVG vector", "*.svg")],
                                            initialfile="tirf_mechanism.png")
        if path:
            try:
                export_fig = Figure(figsize=(8, 6), dpi=110)
                draw_figure(export_fig, state["settings"])
                export_fig.savefig(path, dpi=300)
                set_status("ok", f"Figure written: {path}")
            except Exception as exc:
                set_status("err", f"{type(exc).__name__}: {exc}")

    def save_csv():
        settings = state["settings"]
        if settings is None:
            update_plot()
            settings = state["settings"]
        if settings is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV data", "*.csv")],
                                            initialfile="tirf_mechanism_data.csv")
        if path:
            try:
                export_csv(path, settings)
                set_status("ok", f"Data written: {path}")
            except Exception as exc:
                set_status("err", f"{type(exc).__name__}: {exc}")

    section("Actions")
    for text, command in (("Update plot", update_plot),
                          ("Reset defaults", reset),
                          ("Save figure (PNG/PDF/SVG)", save_figure),
                          ("Export plotted data (CSV)", save_csv)):
        ttk.Button(controls, text=text, command=command).pack(fill="x", padx=9, pady=2)
    ttk.Label(controls, text="Press Enter to update.", foreground="#555").pack(anchor="w", padx=9, pady=(2, 12))

    root.bind("<Return>", update_plot)
    update_plot()
    root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run numerical and data-shape tests")
    parser.add_argument("--preview", metavar="PNG", help="write a headless default preview")
    args = parser.parse_args(argv)
    if args.selftest:
        raise SystemExit(0 if run_selftests() else 1)
    if args.preview:
        preview(os.path.abspath(args.preview))
        print(f"preview -> {os.path.abspath(args.preview)}")
        return
    launch_gui()


if __name__ == "__main__":
    main()
