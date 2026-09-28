"""Minimal polarization/orientation SFG simulation.

The script intentionally produces only two figures:

1. resonant SFG intensity versus IR wavenumber for PPP/SSP/SPS/PSS;
2. resonant peak intensity versus molecular tilt angle for the same combinations.

The optical configuration is the CO/Pt(111) literature benchmark of Li et al.,
Topics in Catalysis 61, 751-762 (2018), doi:10.1007/s11244-018-0949-7.
With an upright CO molecule and R = beta_aac / beta_ccc = 0.49, the model gives
I_PPP/I_SSP = 26.83, compared with the reported experimental value of 27.

Only numpy and matplotlib are required.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


# =============================================================================
# USER SETTINGS -- these are the only physics values normally worth changing.
# =============================================================================
RESONANCE_CM1 = 2090.0       # center of the vibrational resonance (cm^-1)
HWHM_CM1 = 8.0               # Lorentzian half width at half maximum (cm^-1)
SPECTRUM_TILT_DEG = 25.0     # fixed tilt used in the frequency plot
HYPERPOLARIZABILITY_RATIO = 0.49  # R = beta_aac / beta_ccc

POLARIZATIONS = ("ppp", "ssp", "sps", "pss")
FREQUENCY_RANGE_CM1 = (1800.0, 2160.0)
TILT_RANGE_DEG = (0.0, 60.0)


# =============================================================================
# FIXED LITERATURE CONFIGURATION -- CO/Pt(111), Li et al. (2018), Table 1.
# Order of each tuple is (SFG, visible, IR).
# =============================================================================
VISIBLE_WAVELENGTH_NM = 532.0
VISIBLE_INCIDENCE_DEG = 58.5
IR_INCIDENCE_DEG = 55.0

# Complex bulk refractive indices n + ik at 0.479 um, 0.532 um, and 4.785 um.
PT_BULK_INDEX = (1.91 + 3.30j, 2.04 + 3.60j, 3.89 + 18.90j)

# Literature values used only for a transparent internal consistency check.
PT_INTERFACE_INDEX_TABLE = (
    1.14 + 1.40j,
    1.18 + 1.60j,
    1.96 + 9.40j,
)
LITERATURE_PPP_SSP_RATIO = 27.0
LITERATURE_PEAK_CM1 = 2092.0

BEAM_SFG, BEAM_VIS, BEAM_IR = 0, 1, 2
ALLOWED_POLARIZATIONS = {"ppp", "ssp", "sps", "pss"}

PLOT_STYLE = {
    "ppp": ("#D55E00", "-"),
    "ssp": ("#0072B2", "--"),
    "sps": ("#009E73", "-."),
    "pss": ("#CC79A7", ":"),
}

# Publication-style typography, matched to the 6.4 x 4.8 inch ATR reference
# figure while making the SFG labels slightly larger for screen use.
PLOT_TITLE_SIZE = 12.0
PLOT_AXIS_LABEL_SIZE = 11.0
PLOT_TICK_SIZE = 9.5
PLOT_LEGEND_SIZE = 9.0
PLOT_ANNOTATION_SIZE = 9.0
PLOT_LINE_WIDTH = 1.8


def apply_publication_axis_style(ax) -> None:
    """Apply the shared ATR-reference typography and axis treatment."""

    ax.grid(True, which="both", color="0.82", alpha=0.55, linewidth=0.6)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_TICK_SIZE,
        width=0.9,
        length=4.0,
    )
    ax.tick_params(axis="both", which="minor", width=0.7, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def modified_lorentz_index(n_bulk: complex | np.ndarray) -> complex | np.ndarray:
    """Interfacial index used by Zhuang et al. and Li et al.

    For air (n1 = 1):
        n' = n2 * sqrt((n2^2 + 5) / (4*n2^2 + 2)).
    """

    n2 = np.asarray(n_bulk, dtype=complex)
    result = n2 * np.sqrt((n2**2 + 5.0) / (4.0 * n2**2 + 2.0))
    return complex(result) if result.ndim == 0 else result


def fresnel_factors(
    n_bulk: complex,
    n_interface: complex,
    incidence_rad: float | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the local-field factors (Lxx, Lyy, Lzz).

    These are the two-medium SFG Fresnel factors in Zhuang et al.,
    Phys. Rev. B 59, 12632 (1999), with the incident medium set to air.
    """

    beta = np.asarray(incidence_rad, dtype=float)
    n1 = 1.0 + 0.0j
    n2 = complex(n_bulk)
    n_prime = complex(n_interface)
    cos_gamma = np.sqrt(1.0 - (n1 * np.sin(beta) / n2) ** 2)
    cos_beta = np.cos(beta)

    lxx = 2.0 * n1 * cos_gamma / (n1 * cos_gamma + n2 * cos_beta)
    lyy = 2.0 * n1 * cos_beta / (n1 * cos_beta + n2 * cos_gamma)
    lzz = (
        2.0
        * n2
        * cos_beta
        / (n1 * cos_gamma + n2 * cos_beta)
        * (n1 / n_prime) ** 2
    )
    return lxx, lyy, lzz


def sfg_exit_angle_rad(ir_wavenumber_cm1: np.ndarray) -> np.ndarray:
    """Copropagating SFG exit angle from in-plane momentum conservation."""

    ir = np.asarray(ir_wavenumber_cm1, dtype=float)
    visible = 1.0e7 / VISIBLE_WAVELENGTH_NM
    numerator = (
        visible * math.sin(math.radians(VISIBLE_INCIDENCE_DEG))
        + ir * math.sin(math.radians(IR_INCIDENCE_DEG))
    )
    return np.arcsin(np.clip(numerator / (visible + ir), -1.0, 1.0))


def orientation_factors(
    tilt_deg: float | np.ndarray,
    ratio: float | None = None,
) -> Dict[str, np.ndarray]:
    """Macroscopic tensor factors for a delta-function tilt distribution.

    The molecular c axis is tilted by theta from the surface normal.  Azimuthal
    orientations are isotropic, and the interface is non-chiral.
    """

    if ratio is None:
        # Resolve at call time so the minimal GUI can adjust R dynamically.
        ratio = HYPERPOLARIZABILITY_RATIO
    c1 = np.cos(np.radians(np.asarray(tilt_deg, dtype=float)))
    c3 = c1**3
    return {
        "zzz": ratio * c1 + (1.0 - ratio) * c3,
        "xxz": 0.5 * ((1.0 + ratio) * c1 - (1.0 - ratio) * c3),
        "zxx": 0.5 * (1.0 - ratio) * (c1 - c3),
        "xzx": 0.5 * (1.0 - ratio) * (c1 - c3),
    }


def geometric_factors(
    ir_wavenumber_cm1: float | np.ndarray,
    polarization: str,
) -> Dict[str, np.ndarray]:
    """Fresnel/polarization coefficient multiplying each chi tensor element.

    Polarization order is (SFG, visible, IR), so ``ssp`` means
    s(SFG), s(visible), p(IR).  The closed forms follow Hirose et al.,
    Appl. Spectrosc. 46, 1051 (1992).
    """

    pol = polarization.lower()
    if pol not in ALLOWED_POLARIZATIONS:
        allowed = ", ".join(sorted(ALLOWED_POLARIZATIONS))
        raise ValueError(f"Unsupported polarization '{polarization}'. Use: {allowed}")

    ir = np.asarray(ir_wavenumber_cm1, dtype=float)
    beta_sfg = sfg_exit_angle_rad(ir)
    beta_vis = math.radians(VISIBLE_INCIDENCE_DEG)
    beta_ir = math.radians(IR_INCIDENCE_DEG)

    n_interface = tuple(modified_lorentz_index(n) for n in PT_BULK_INDEX)
    l_sfg = fresnel_factors(
        PT_BULK_INDEX[BEAM_SFG], n_interface[BEAM_SFG], beta_sfg
    )
    l_vis = fresnel_factors(
        PT_BULK_INDEX[BEAM_VIS], n_interface[BEAM_VIS], beta_vis
    )
    l_ir = fresnel_factors(
        PT_BULK_INDEX[BEAM_IR], n_interface[BEAM_IR], beta_ir
    )

    lsx, lsy, lsz = l_sfg
    lvx, lvy, lvz = l_vis
    lix, liy, liz = l_ir
    zero = np.zeros_like(ir, dtype=complex)

    factors = {"zzz": zero.copy(), "xxz": zero.copy(),
               "zxx": zero.copy(), "xzx": zero.copy()}
    if pol == "ssp":
        factors["xxz"] = lsy * lvy * liz * math.sin(beta_ir)
    elif pol == "sps":
        factors["xzx"] = lsy * lvz * liy * math.sin(beta_vis)
    elif pol == "pss":
        factors["zxx"] = lsz * lvy * liy * np.sin(beta_sfg)
    else:  # ppp
        factors["xxz"] = (
            -lsx * lvx * liz
            * np.cos(beta_sfg) * math.cos(beta_vis) * math.sin(beta_ir)
        )
        factors["xzx"] = (
            -lsx * lvz * lix
            * np.cos(beta_sfg) * math.sin(beta_vis) * math.cos(beta_ir)
        )
        factors["zxx"] = (
            lsz * lvx * lix
            * np.sin(beta_sfg) * math.cos(beta_vis) * math.cos(beta_ir)
        )
        factors["zzz"] = (
            lsz * lvz * liz
            * np.sin(beta_sfg) * math.sin(beta_vis) * math.sin(beta_ir)
        )
    return factors


def chi_eff(
    ir_wavenumber_cm1: float | np.ndarray,
    tilt_deg: float | np.ndarray,
    polarization: str,
) -> np.ndarray:
    """Effective resonant susceptibility for one CO stretching mode."""

    ir = np.asarray(ir_wavenumber_cm1, dtype=float)
    orientation = orientation_factors(tilt_deg)
    geometry = geometric_factors(ir, polarization)
    projected = sum(geometry[key] * orientation[key] for key in geometry)
    lorentzian = 1.0 / (ir - RESONANCE_CM1 + 1j * HWHM_CM1)
    return projected * lorentzian


def intensity(
    ir_wavenumber_cm1: float | np.ndarray,
    tilt_deg: float | np.ndarray,
    polarization: str,
) -> np.ndarray:
    """SFG intensity in arbitrary units: I is proportional to |chi_eff|^2."""

    return np.abs(chi_eff(ir_wavenumber_cm1, tilt_deg, polarization)) ** 2


def literature_benchmark() -> Dict[str, float]:
    """Return compact checks against Li et al. (2018), Table 1 and Fig. 2."""

    n_interface = tuple(modified_lorentz_index(n) for n in PT_BULK_INDEX)
    index_error = max(
        abs(calculated - tabulated)
        for calculated, tabulated in zip(n_interface, PT_INTERFACE_INDEX_TABLE)
    )

    i_ppp = float(intensity(RESONANCE_CM1, 0.0, "ppp"))
    i_ssp = float(intensity(RESONANCE_CM1, 0.0, "ssp"))
    ratio = i_ppp / i_ssp
    relative_error = abs(ratio - LITERATURE_PPP_SSP_RATIO) / LITERATURE_PPP_SSP_RATIO
    return {
        "model_ratio": ratio,
        "literature_ratio": LITERATURE_PPP_SSP_RATIO,
        "relative_error": relative_error,
        "max_interface_index_error": index_error,
    }


def _validate_settings(polarizations: Iterable[str]) -> Tuple[str, ...]:
    selected = tuple(pol.lower() for pol in polarizations)
    if not selected:
        raise ValueError("Select at least one polarization combination.")
    unsupported = set(selected) - ALLOWED_POLARIZATIONS
    if unsupported:
        raise ValueError(f"Unsupported polarizations: {sorted(unsupported)}")
    if HWHM_CM1 <= 0.0:
        raise ValueError("HWHM_CM1 must be positive.")
    if not (0.0 <= SPECTRUM_TILT_DEG <= 90.0):
        raise ValueError("SPECTRUM_TILT_DEG must lie between 0 and 90 degrees.")
    return selected


def plot_frequency(output_dir: Path, polarizations: Tuple[str, ...]) -> Path:
    """Create the polarization-resolved frequency plot."""

    import matplotlib.pyplot as plt

    wavenumber = np.linspace(*FREQUENCY_RANGE_CM1, 721)
    curves = {
        pol: intensity(wavenumber, SPECTRUM_TILT_DEG, pol)
        for pol in polarizations
    }
    common_scale = max(float(np.max(values)) for values in curves.values())

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)
    for pol, values in curves.items():
        color, linestyle = PLOT_STYLE[pol]
        normalized = values / common_scale
        ax.plot(
            wavenumber,
            np.maximum(normalized, 1.0e-9),
            color=color,
            linestyle=linestyle,
            linewidth=PLOT_LINE_WIDTH,
            label=pol.upper(),
        )

    ax.axvline(
        LITERATURE_PEAK_CM1,
        color="0.35",
        linestyle=(0, (2, 3)),
        linewidth=1.2,
        label=r"Li et al. peak: 2092 cm$^{-1}$",
    )
    ax.set_yscale("log")
    ax.set_ylim(1.0e-7, 1.5)
    ax.set_xlabel(
        r"IR wavenumber (cm$^{-1}$)",
        fontsize=PLOT_AXIS_LABEL_SIZE,
        labelpad=7,
    )
    ax.set_ylabel(
        "Normalized resonant SFG intensity",
        fontsize=PLOT_AXIS_LABEL_SIZE,
        labelpad=7,
    )
    ax.set_title(
        f"Polarization-resolved SFG spectrum  (tilt = {SPECTRUM_TILT_DEG:g} deg)",
        fontsize=PLOT_TITLE_SIZE,
        pad=10,
    )
    apply_publication_axis_style(ax)
    ax.legend(
        fontsize=PLOT_LEGEND_SIZE,
        ncol=2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        handlelength=2.7,
    )
    fig.tight_layout(pad=1.1)

    output_path = output_dir / "sfg_intensity_vs_frequency.png"
    fig.savefig(output_path, dpi=300, facecolor="white")
    return output_path


def plot_tilt(output_dir: Path, polarizations: Tuple[str, ...]) -> Path:
    """Create the peak-intensity versus molecular-tilt plot."""

    import matplotlib.pyplot as plt

    tilt = np.linspace(*TILT_RANGE_DEG, 301)
    benchmark = literature_benchmark()

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=120)
    for pol in polarizations:
        values = intensity(RESONANCE_CM1, tilt, pol)
        own_maximum = max(float(np.max(values)), np.finfo(float).tiny)
        color, linestyle = PLOT_STYLE[pol]
        ax.plot(
            tilt,
            values / own_maximum,
            color=color,
            linestyle=linestyle,
            linewidth=PLOT_LINE_WIDTH,
            label=pol.upper(),
        )

    comparison = (
        r"CO/Pt(111), upright:  "
        f"model $I_{{PPP}}/I_{{SSP}}$ = {benchmark['model_ratio']:.2f}\n"
        f"Li et al. experiment = {benchmark['literature_ratio']:.0f}"
    )
    ax.text(
        0.98,
        0.97,
        comparison,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=PLOT_ANNOTATION_SIZE,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.set_xlim(*TILT_RANGE_DEG)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(
        "Tilt angle from the surface normal (deg)",
        fontsize=PLOT_AXIS_LABEL_SIZE,
        labelpad=7,
    )
    ax.set_ylabel(
        "Peak intensity / maximum of each polarization",
        fontsize=PLOT_AXIS_LABEL_SIZE,
        labelpad=7,
    )
    ax.set_title(
        "SFG resonant peak intensity versus molecular tilt",
        fontsize=PLOT_TITLE_SIZE,
        pad=10,
    )
    apply_publication_axis_style(ax)
    ax.legend(
        fontsize=PLOT_LEGEND_SIZE,
        ncol=2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        handlelength=2.7,
        loc="lower right",
    )
    fig.tight_layout(pad=1.1)

    output_path = output_dir / "sfg_intensity_vs_tilt.png"
    fig.savefig(output_path, dpi=300, facecolor="white")
    return output_path


def _print_benchmark(benchmark: Dict[str, float]) -> None:
    print("Literature benchmark: CO/Pt(111), Li et al. (2018)")
    print(f"  model I_PPP/I_SSP      = {benchmark['model_ratio']:.4f}")
    print(f"  reported experiment    = {benchmark['literature_ratio']:.1f}")
    print(f"  relative difference    = {100.0 * benchmark['relative_error']:.2f}%")
    print(
        "  max |calculated n' - table n'| = "
        f"{benchmark['max_interface_index_error']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate only the two requested SFG plots."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sfg_two_plots_output",
        help="directory for the two PNG files",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save figures without opening an interactive window",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the literature benchmark without creating figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    polarizations = _validate_settings(POLARIZATIONS)
    benchmark = literature_benchmark()
    _print_benchmark(benchmark)

    benchmark_ok = (
        benchmark["relative_error"] < 0.02
        # Li et al. tabulate n' to only two decimals, so allow for rounding.
        and benchmark["max_interface_index_error"] < 0.05
    )
    if not benchmark_ok:
        print("Benchmark check FAILED; figures were not generated.")
        return 1
    if args.check:
        print("Benchmark check PASSED.")
        return 0

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frequency_path = plot_frequency(args.output_dir, polarizations)
    tilt_path = plot_tilt(args.output_dir, polarizations)
    print(f"Saved: {frequency_path}")
    print(f"Saved: {tilt_path}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
