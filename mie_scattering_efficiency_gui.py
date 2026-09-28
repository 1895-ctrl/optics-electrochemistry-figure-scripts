#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mie-scattering cross-section GUI for electrochemical particle materials.

Model: one homogeneous ideal sphere in a homogeneous, nonabsorbing medium,
illuminated by an unpolarized plane wave. Wavelengths are vacuum wavelengths.
The displayed output is total scattered power normalized by illumination
irradiance, integrated over 4*pi:
C_sca = P_sca / I0 = Q_sca * pi * (D/2)^2.
No particle-area normalization or objective-NA collection model is applied.
Consequently, the calculated total scattering is not a measured camera signal.

Dependencies:
    pip install numpy matplotlib miepython

Usage:
    python mie_scattering_efficiency_gui.py
    python mie_scattering_efficiency_gui.py --selftest
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _configure_tcltk_environment() -> None:
    """Help selected Windows/Python installations locate the Tcl/Tk runtime."""
    if os.environ.get("TCL_LIBRARY") and os.environ.get("TK_LIBRARY"):
        return
    roots = {
        Path(sys.base_prefix),
        Path(sys.prefix),
        Path(sys.executable).resolve().parent,
        Path(sys.executable).resolve().parent.parent,
    }
    for root in roots:
        for base in (root / "tcl", root / "Lib", root):
            tcl = base / "tcl8.6"
            tk = base / "tk8.6"
            if tcl.exists():
                os.environ.setdefault("TCL_LIBRARY", str(tcl))
            if tk.exists():
                os.environ.setdefault("TK_LIBRARY", str(tk))
        if os.environ.get("TCL_LIBRARY") and os.environ.get("TK_LIBRARY"):
            return


_configure_tcltk_environment()

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - clear message for end users
    raise SystemExit("Missing numpy. Run: pip install numpy matplotlib miepython") from exc

try:
    import miepython
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing miepython. Run: pip install numpy matplotlib miepython") from exc

try:
    import matplotlib

    matplotlib.use("Agg" if "--selftest" in sys.argv else "TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing matplotlib. Run: pip install numpy matplotlib miepython") from exc


# Sellmeier: n^2 - 1 = sum_i B_i * lambda^2 / (lambda^2 - C_i), lambda in um.
# SiO2: I. H. Malitson, JOSA 55, 1205-1209 (1965).
# Al2O3: M. J. Dodge, Handbook of Laser Science and Technology, Vol. IV (1986).
#         The data are commonly also cited to Malitson & Dodge, JOSA 62, 1405 (1972).
SELLMEIER = {
    "SiO2": {
        "B": (0.6961663, 0.4079426, 0.8974794),
        "C": (0.0684043**2, 0.1162414**2, 9.896161**2),
        "range_nm": (210.0, 6700.0),
    },
    "Al2O3-o": {
        "B": (1.4313493, 0.65054713, 5.3414021),
        "C": (0.0726631**2, 0.1193242**2, 18.028251**2),
        "range_nm": (200.0, 5000.0),
    },
    "Al2O3-e": {
        "B": (1.5039759, 0.55069141, 6.5927379),
        "C": (0.0740288**2, 0.1216529**2, 20.072248**2),
        "range_nm": (200.0, 5000.0),
    },
}

# Bulk optical constants n + i*k from P. B. Johnson and R. W. Christy,
# Physical Review B 6, 4370-4379 (1972), DOI: 10.1103/PhysRevB.6.4370.
# The source reports room-temperature evaporated films. Interpolation is
# performed in the complex dielectric function to preserve passive absorption.
METAL_NK_TABLES = {
    "Au": """\
191.6 1.32 1.203
195.3 1.34 1.226
199.3 1.33 1.251
203.3 1.33 1.277
207.3 1.30 1.304
211.9 1.30 1.350
216.4 1.30 1.387
221.4 1.30 1.427
226.2 1.31 1.460
231.3 1.30 1.497
237.1 1.32 1.536
242.6 1.32 1.577
249.0 1.33 1.631
255.1 1.33 1.688
261.6 1.35 1.749
268.9 1.38 1.803
276.1 1.43 1.847
284.4 1.47 1.869
292.4 1.49 1.878
300.9 1.53 1.889
310.7 1.53 1.893
320.4 1.54 1.898
331.5 1.48 1.883
342.5 1.48 1.871
354.2 1.50 1.866
367.9 1.48 1.895
381.5 1.46 1.933
397.4 1.47 1.952
413.3 1.46 1.958
430.5 1.45 1.948
450.9 1.38 1.914
471.4 1.31 1.849
495.9 1.04 1.833
520.9 0.62 2.081
548.6 0.43 2.455
582.1 0.29 2.863
616.8 0.21 3.272
659.5 0.14 3.697
704.5 0.13 4.103
756.0 0.14 4.542
821.1 0.16 5.083
892.0 0.17 5.663
984.0 0.22 6.350
1088.0 0.27 7.150""",
    "Ag": """\
191.6 1.10 1.232
195.3 1.12 1.255
199.3 1.14 1.277
203.3 1.15 1.296
207.3 1.18 1.312
211.9 1.20 1.325
216.4 1.22 1.336
221.4 1.25 1.342
226.2 1.26 1.344
231.3 1.28 1.357
237.1 1.28 1.367
242.6 1.30 1.378
249.0 1.31 1.389
255.1 1.33 1.393
261.6 1.35 1.387
268.9 1.38 1.372
276.1 1.41 1.331
284.4 1.41 1.264
292.4 1.39 1.161
300.9 1.34 0.964
310.7 1.13 0.616
320.4 0.81 0.392
331.5 0.17 0.829
342.5 0.14 1.142
354.2 0.10 1.419
367.9 0.07 1.657
381.5 0.05 1.864
397.4 0.05 2.070
413.3 0.05 2.275
430.5 0.04 2.462
450.9 0.04 2.657
471.4 0.05 2.869
495.9 0.05 3.093
520.9 0.05 3.324
548.6 0.06 3.586
582.1 0.05 3.858
616.8 0.06 4.152
659.5 0.05 4.483
704.5 0.04 4.838
756.0 0.03 5.242
821.1 0.04 5.727
892.0 0.04 6.312
984.0 0.04 6.992
1088.0 0.04 7.795""",
    "Cu": """\
191.6 0.95 1.388
195.3 0.97 1.440
199.3 0.98 1.493
203.3 0.99 1.550
207.3 1.01 1.599
211.9 1.04 1.651
216.4 1.08 1.699
221.4 1.13 1.737
226.2 1.18 1.768
231.3 1.23 1.792
237.1 1.28 1.802
242.6 1.34 1.799
249.0 1.37 1.783
255.1 1.41 1.741
261.6 1.41 1.691
268.9 1.45 1.668
276.1 1.46 1.646
284.4 1.45 1.633
292.4 1.42 1.633
300.9 1.40 1.679
310.7 1.38 1.729
320.4 1.38 1.783
331.5 1.34 1.821
342.5 1.36 1.864
354.2 1.37 1.916
367.9 1.36 1.975
381.5 1.33 2.045
397.4 1.32 2.116
413.3 1.28 2.207
430.5 1.25 2.305
450.9 1.24 2.397
471.4 1.25 2.483
495.9 1.22 2.564
520.9 1.18 2.608
548.6 1.02 2.577
582.1 0.70 2.704
616.8 0.30 3.205
659.5 0.22 3.747
704.5 0.21 4.205
756.0 0.24 4.665
821.1 0.26 5.180
892.0 0.30 5.768
984.0 0.32 6.421
1088.0 0.36 7.217""",
}

# Additional electrochemical/catalyst materials.  These compact tables are
# sampled from the cited full datasets and interpolated in complex dielectric
# permittivity, matching the treatment of the Johnson-Christy tables above.
#
# Pt: W. S. M. Werner et al., J. Phys. Chem. Ref. Data 38, 1013-1092 (2009),
#     DOI 10.1063/1.3243762; experimental REELS-derived optical constants.
# Ni: P. B. Johnson and R. W. Christy, Phys. Rev. B 9, 5056-5070 (1974),
#     DOI 10.1103/PhysRevB.9.5056; room-temperature bulk metal.
METAL_NK_TABLES.update({
    "Pt": """\
354.241 1.5367 3.0624
381.490 1.2882 2.9080
413.281 0.8675 3.2129
450.852 0.6273 3.7605
495.937 0.5124 4.3965
551.041 0.4643 5.1210
619.921 0.4611 5.9757
708.481 0.5013 7.0273
826.561 0.5979 8.3824
991.874 0.7870 10.2266""",
    "Ni": """\
354 1.74 2.32
368 1.70 2.40
381 1.72 2.48
397 1.72 2.57
413 1.70 2.69
431 1.71 2.82
451 1.73 2.95
471 1.78 3.09
496 1.82 3.25
521 1.85 3.42
549 1.92 3.61
582 1.96 3.80
617 1.99 4.02
659 1.99 4.26
704 2.06 4.50
756 2.13 4.73
821 2.26 4.97
892 2.40 5.23
984 2.48 5.55""",
})

# Representative electrochemical oxide films.  TiO2 is an anatase ALD film
# deposited at 300 degC (A. Jolivet et al., Appl. Surf. Sci. 608, 155214,
# 2023, DOI 10.1016/j.apsusc.2022.155214).  ITO is a 72 nm commercial film on
# BK7, 15-25 ohm/sq (T. A. F. Koenig et al., ACS Nano 8, 6182-6192, 2014,
# DOI 10.1021/nn501601e).  Film preparation changes n and k, so these are
# representative models rather than universal constants.
OXIDE_NK_TABLES = {
    "TiO2": """\
350.24 3.19404 0.06830
358.34 3.07935 0.01750
366.82 2.97647 0.000052
375.71 2.90167 0
385.04 2.84344 0
396.12 2.78945 0
406.51 2.74866 0
417.46 2.71300 0
429.01 2.68142 0
441.22 2.65320 0
454.15 2.62778 0
467.86 2.60474 0
482.43 2.58376 0
499.94 2.56229 0
516.60 2.54487 0
534.41 2.52883 0
553.50 2.51403 0
574.00 2.50034 0
596.08 2.48767 0
619.92 2.47593 0
645.75 2.46502 0
677.51 2.45369 0
708.48 2.44437 0
742.42 2.43572 0
779.77 2.42769 0
821.09 2.42023 0""",
    "ITO": """\
350.26 2.1334588 0.02331379
372.54 2.0832101 0.01625075
394.83 2.0436584 0.01167263
418.70 2.0082375 0.00849415
440.97 1.9793559 0.00655029
463.24 1.9531623 0.00524630
485.51 1.9287944 0.00437210
507.77 1.9056515 0.00379424
530.02 1.8832990 0.00342657
553.84 1.8598601 0.00320207
576.07 1.8381989 0.00311137
598.29 1.8165479 0.00311010
620.50 1.7947523 0.00318168
642.69 1.7726842 0.00331496
664.88 1.7502354 0.00350254
688.62 1.7256540 0.00375841
710.77 1.7021293 0.00404557
732.91 1.6779645 0.00437810
755.02 1.6530884 0.00475594
777.12 1.6274321 0.00517995
799.20 1.6009281 0.00565185
822.83 1.5715127 0.00621345
844.87 1.5430336 0.00679338
866.88 1.5134898 0.00743143
888.87 1.4828025 0.00813268
910.83 1.4508870 0.00890328
932.78 1.4176498 0.00975064
956.25 1.3804536 0.01075390
978.14 1.3441334 0.01179087
1000.0 1.3061268 0.01293873""",
}

# Intrinsic crystalline Si optical constants n+i*k at 300 K from
# M. A. Green, Solar Energy Materials and Solar Cells 92, 1305-1310 (2008),
# DOI: 10.1016/j.solmat.2008.06.009. The source tabulates 10 nm increments.
SILICON_NK_TABLE = """\
400 5.613 0.296
410 5.330 0.227
420 5.119 0.176
430 4.949 0.138
440 4.812 0.107
450 4.691 0.086
460 4.587 0.071
470 4.497 0.062
480 4.419 0.055
490 4.350 0.049
500 4.294 0.044
510 4.241 0.039
520 4.193 0.036
530 4.151 0.033
540 4.112 0.030
550 4.077 0.028
560 4.045 0.026
570 4.015 0.024
580 3.988 0.023
590 3.963 0.021
600 3.940 0.020
610 3.918 0.018
620 3.898 0.017
630 3.879 0.016
640 3.861 0.015
650 3.844 0.014
660 3.828 0.013
670 3.813 0.013
680 3.798 0.012
690 3.784 0.011
700 3.772 0.011
710 3.759 0.010
720 3.748 0.010
730 3.737 0.009
740 3.727 0.008
750 3.717 0.008
760 3.708 0.007
770 3.699 0.007
780 3.691 0.006
790 3.683 0.006
800 3.675 0.005"""

# Water: M. Daimon & A. Masumura, Applied Optics 46, 3811-3820 (2007), 20 degC.
WATER = {
    "B": (5.684027565e-1, 1.726177391e-1, 2.086189578e-2, 1.130748688e-1),
    "C": (5.101829712e-3, 1.821153936e-2, 2.620722293e-2, 1.069792721e1),
    "range_nm": (182.0, 1129.0),
}

MATERIAL_LABELS = {
    "Au": "Au",
    "Ag": "Ag",
    "Cu": "Cu",
    "Pt": "Pt",
    "Ni": "Ni",
    "TiO2": "TiO2 (anatase film)",
    "ITO": "ITO film",
    "SiO2": "SiO2",
    "Si": "Si",
    "Polystyrene": "Polystyrene",
    "Al2O3": "Al2O3",
}

ALUMINA_MODELS = {
    "Ordinary index, o (default)": "o",
    "Extraordinary index, e": "e",
    "Mean of o/e indices": "mean",
}

MEDIUM_LABELS = {
    "Air (n = 1.000)": "air",
    "Water (20 °C, dispersive)": "water",
    "Custom refractive index": "custom",
}

# Shared publication style, matched to the previous 6.4 x 4.8 inch figures.
PLOT_TITLE_SIZE = 12.0
PLOT_AXIS_LABEL_SIZE = 11.0
PLOT_TICK_SIZE = 9.5
PLOT_LEGEND_SIZE = 9.0
PLOT_ANNOTATION_SIZE = 9.0
PLOT_LINE_WIDTH = 1.8
EXPORT_FIGURE_SIZE = (6.4, 4.8)
EXPORT_DPI = 300
LEGEND_POSITIONS = {
    "Lower right (default)": "lower right",
    "Upper right": "upper right",
    "Upper left": "upper left",
    "Lower left": "lower left",
    "Center right": "center right",
    "Center left": "center left",
    "Upper center": "upper center",
    "Lower center": "lower center",
    "Center": "center",
    "Best (automatic)": "best",
    "Custom center (X, Y)": "custom",
}
SERIES_COLORS = {
    "Au": "#c89400",
    "Ag": "#7b8794",
    "Cu": "#b65c32",
    "Pt": "#4c566a",
    "Ni": "#2e8b57",
    "TiO2": "#1f77b4",
    "ITO": "#17becf",
    "SiO2": "#1f77b4",
    "Si": "#3f4a54",
    "Polystyrene": "#159b83",
    "Al2O3": "#8e44ad",
}
LINE_STYLES = ("-", "--", "-.", ":")
PLANCK_CONSTANT = 6.62607015e-34
LIGHT_SPEED = 299792458.0
# Manufacturer presets are deliberately explicit so the lower references are
# reproducible rather than arbitrary electron-count thresholds.
# Hamamatsu: https://www.hamamatsu.com/us/en/product/cameras/cmos-cameras/C13440-20CU.html
# Teledyne Photometrics: https://www.teledynevisionsolutions.com/en-in/products/prime-bsi/
CAMERA_MODELS = {
    "orca_flash_v3": {
        "label": "ORCA-Flash4.0 V3 (standard sCMOS)",
        "short_label": "ORCA-Flash4.0 V3",
        "qe": 0.82,
        "read_noise_rms_e": 1.6,
        "dark_current_e_pixel_s": 0.06,
        "full_well_e": 30000.0,
        "color": "#8c564b",
        "linestyle": ":",
        "source": "Hamamatsu C13440-20CU specifications",
    },
    "prime_bsi": {
        "label": "Prime BSI (high-QE sCMOS)",
        "short_label": "Prime BSI",
        "qe": 0.95,
        "read_noise_rms_e": 1.1,
        "dark_current_e_pixel_s": 0.5,
        "full_well_e": 45000.0,
        "color": "#9467bd",
        "linestyle": "--",
        "source": "Teledyne Photometrics Prime BSI specifications",
    },
}
CAMERA_LABEL_TO_KEY = {
    camera["label"]: key for key, camera in CAMERA_MODELS.items()
}
BACKGROUND_LIMIT_COLOR = "#e07b39"
BACKGROUND_FIELD_COLOR = "#c23b22"


@dataclass(frozen=True)
class Series:
    material_key: str
    material_label: str
    wavelength_nm: float
    particle_index: complex
    medium_index: float
    diameter_nm: np.ndarray
    qsca: np.ndarray
    csca_um2: np.ndarray
    incident_irradiance_w_cm2: float
    psca_pw: np.ndarray


@dataclass(frozen=True)
class ReferenceSettings:
    exposure_s: float
    roi_pixels: int
    camera_keys: tuple[str, ...]
    show_measured_background: bool
    measured_background_e_roi: float
    measured_background_camera_key: str
    show_background_field: bool
    background_irradiance_w_cm2: float
    label_intersections: bool


@dataclass(frozen=True)
class Intersection:
    reference_kind: str
    reference_label: str
    camera_key: str
    material_key: str
    material_label: str
    wavelength_nm: float
    diameter_nm: float
    normalized_signal_um2: float


def sellmeier_n(wavelength_nm: np.ndarray | float, model: str) -> np.ndarray:
    """Return a nonabsorbing Sellmeier index; reject extrapolation."""
    data = SELLMEIER[model]
    wavelength = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    lo, hi = data["range_nm"]
    if not np.all(np.isfinite(wavelength)) or wavelength.min() < lo or wavelength.max() > hi:
        raise ValueError(f"Wavelength for {model} must be within {lo:g}-{hi:g} nm.")
    lambda2 = (wavelength / 1000.0) ** 2
    b = np.asarray(data["B"], dtype=float)[:, None]
    c = np.asarray(data["C"], dtype=float)[:, None]
    n2 = 1.0 + np.sum(b * lambda2[None, :] / (lambda2[None, :] - c), axis=0)
    if np.any(n2 <= 0):
        raise ValueError(f"The {model} Sellmeier model is nonphysical at the selected wavelength.")
    return np.sqrt(n2)


def water_n(wavelength_nm: np.ndarray | float) -> np.ndarray:
    """Daimon-Masumura refractive index of water at 20 degC."""
    wavelength = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    lo, hi = WATER["range_nm"]
    if not np.all(np.isfinite(wavelength)) or wavelength.min() < lo or wavelength.max() > hi:
        raise ValueError(f"The water dispersion model is valid only from {lo:g} to {hi:g} nm.")
    lambda2 = (wavelength / 1000.0) ** 2
    b = np.asarray(WATER["B"], dtype=float)[:, None]
    c = np.asarray(WATER["C"], dtype=float)[:, None]
    n2 = 1.0 + np.sum(b * lambda2[None, :] / (lambda2[None, :] - c), axis=0)
    return np.sqrt(n2)


def _tabulated_nk(
    wavelength_nm: np.ndarray | float,
    material_key: str,
    tables: dict[str, str],
    dataset_label: str,
) -> np.ndarray:
    """Return tabulated n+i*k, linearly interpolated in complex permittivity."""
    if material_key not in tables:
        raise ValueError(f"Unknown tabulated material: {material_key}")
    wavelength = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    rows = np.asarray(
        [[float(value) for value in line.split()]
         for line in tables[material_key].strip().splitlines()],
        dtype=float,
    )
    lo, hi = float(rows[0, 0]), float(rows[-1, 0])
    if not np.all(np.isfinite(wavelength)) or wavelength.min() < lo or wavelength.max() > hi:
        raise ValueError(
            f"The {dataset_label} {material_key} data are valid only from {lo:g} to {hi:g} nm."
        )
    tabulated_eps = (rows[:, 1] + 1j * rows[:, 2]) ** 2
    eps_real = np.interp(wavelength, rows[:, 0], tabulated_eps.real)
    eps_imag = np.clip(np.interp(wavelength, rows[:, 0], tabulated_eps.imag), 0.0, None)
    nk = np.sqrt(eps_real + 1j * eps_imag)
    if np.any(nk.real <= 0) or np.any(nk.imag < -1e-12):
        raise RuntimeError(f"Interpolated {material_key} optical constants are nonphysical.")
    return nk


def metal_nk(wavelength_nm: np.ndarray | float, material_key: str) -> np.ndarray:
    """Return bulk-metal n+i*k from the selected experimental table."""
    source = "Werner" if material_key == "Pt" else "Johnson-Christy"
    return _tabulated_nk(wavelength_nm, material_key, METAL_NK_TABLES, source)


def oxide_nk(wavelength_nm: np.ndarray | float, material_key: str) -> np.ndarray:
    """Return thin-film oxide n+i*k; values depend on film preparation."""
    source = "Jolivet-anatase" if material_key == "TiO2" else "Konig-film"
    return _tabulated_nk(wavelength_nm, material_key, OXIDE_NK_TABLES, source)


def silicon_nk(wavelength_nm: np.ndarray | float) -> np.ndarray:
    """Return intrinsic crystalline-Si n+i*k using Green's 300 K table."""
    wavelength = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    rows = np.asarray(
        [[float(value) for value in line.split()]
         for line in SILICON_NK_TABLE.strip().splitlines()],
        dtype=float,
    )
    lo, hi = float(rows[0, 0]), float(rows[-1, 0])
    if not np.all(np.isfinite(wavelength)) or wavelength.min() < lo or wavelength.max() > hi:
        raise ValueError(f"The Green Si data are valid only from {lo:g} to {hi:g} nm.")
    tabulated_eps = (rows[:, 1] + 1j * rows[:, 2]) ** 2
    eps_real = np.interp(wavelength, rows[:, 0], tabulated_eps.real)
    eps_imag = np.clip(np.interp(wavelength, rows[:, 0], tabulated_eps.imag), 0.0, None)
    nk = np.sqrt(eps_real + 1j * eps_imag)
    if np.any(nk.real <= 0) or np.any(nk.imag < -1e-12):
        raise RuntimeError("Interpolated Si optical constants are nonphysical.")
    return nk


def polystyrene_n(wavelength_nm: np.ndarray | float) -> np.ndarray:
    """Return visible-range polystyrene index from a one-pole Sellmeier fit.

    The parameters are from K. Clays and J. S. Schildkraut, JOSA B 9,
    2274-2282 (1992), DOI: 10.1364/JOSAB.9.002274.
    """
    wavelength = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    if (
        not np.all(np.isfinite(wavelength))
        or wavelength.min() < 400.0
        or wavelength.max() > 830.0
    ):
        raise ValueError("The polystyrene model is restricted to 400-830 nm.")
    wavelength_um2 = (wavelength / 1000.0) ** 2
    resonance_um2 = 0.1385**2
    n2 = 1.0 + 1.4306 * wavelength_um2 / (wavelength_um2 - resonance_um2)
    if np.any(n2 <= 0):
        raise RuntimeError("The polystyrene Sellmeier model is nonphysical.")
    return np.sqrt(n2)


def particle_n(material_key: str, wavelength_nm: float, alumina_model: str) -> complex:
    if material_key in METAL_NK_TABLES:
        return complex(metal_nk(wavelength_nm, material_key)[0])
    if material_key in OXIDE_NK_TABLES:
        return complex(oxide_nk(wavelength_nm, material_key)[0])
    if material_key == "SiO2":
        return complex(float(sellmeier_n(wavelength_nm, "SiO2")[0]), 0.0)
    if material_key == "Si":
        return complex(silicon_nk(wavelength_nm)[0])
    if material_key == "Polystyrene":
        return complex(float(polystyrene_n(wavelength_nm)[0]), 0.0)
    if material_key != "Al2O3":
        raise ValueError(f"Unknown material: {material_key}")
    if alumina_model == "o":
        return complex(float(sellmeier_n(wavelength_nm, "Al2O3-o")[0]), 0.0)
    if alumina_model == "e":
        return complex(float(sellmeier_n(wavelength_nm, "Al2O3-e")[0]), 0.0)
    if alumina_model == "mean":
        no = float(sellmeier_n(wavelength_nm, "Al2O3-o")[0])
        ne = float(sellmeier_n(wavelength_nm, "Al2O3-e")[0])
        return complex(0.5 * (no + ne), 0.0)
    raise ValueError(f"Unknown Al2O3 model: {alumina_model}")


def medium_n(wavelength_nm: float, medium_key: str, custom_n: float) -> float:
    if medium_key == "air":
        return 1.0
    if medium_key == "water":
        return float(water_n(wavelength_nm)[0])
    if medium_key == "custom":
        if not math.isfinite(custom_n) or custom_n <= 0:
            raise ValueError("The custom medium index must be finite and positive.")
        return float(custom_n)
    raise ValueError(f"Unknown medium: {medium_key}")


def mie_qsca(
    diameter_nm: np.ndarray,
    wavelength_nm: float,
    particle_index: complex,
    medium_index: float,
) -> np.ndarray:
    """Calculate Q_sca using vacuum wavelength and x = pi*n_medium*D/lambda0."""
    diameter = np.asarray(diameter_nm, dtype=float)
    if np.any(~np.isfinite(diameter)) or np.any(diameter <= 0):
        raise ValueError("Diameters must be finite and positive.")
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("Wavelength must be finite and positive.")
    particle_index = complex(particle_index)
    if (
        not math.isfinite(particle_index.real)
        or not math.isfinite(particle_index.imag)
        or particle_index.real <= 0
        or particle_index.imag < 0
        or medium_index <= 0
    ):
        raise ValueError("Use a passive particle index n+i*k with n>0, k>=0, and medium n>0.")
    x = np.pi * medium_index * diameter / wavelength_nm
    relative_index = particle_index / medium_index
    # This program stores passive materials as n+i*k; miepython uses n-i*k.
    solver_index = np.conj(relative_index)
    qext, qsca, _qback, _g = miepython.efficiencies_mx(solver_index, x)
    qext = np.atleast_1d(np.asarray(qext, dtype=float))
    qsca = np.atleast_1d(np.asarray(qsca, dtype=float))
    qabs = qext - qsca
    if np.any(~np.isfinite(qsca)) or np.any(qsca < -1e-12):
        raise RuntimeError("The Mie solver returned nonfinite or negative Q_sca values.")
    tolerance = 1e-8 * np.maximum(1.0, np.abs(qext))
    if np.any(qabs < -tolerance):
        raise RuntimeError("The Mie solver returned negative absorption; check index sign conventions.")
    return np.clip(qsca, 0.0, None)


def compute_series(
    material_keys: list[str],
    wavelengths_nm: list[float],
    diameter_min_nm: float,
    diameter_max_nm: float,
    point_count: int,
    log_diameter_grid: bool,
    medium_key: str,
    custom_medium_n: float,
    alumina_model: str,
    incident_irradiance_w_cm2: float = 1.0,
) -> list[Series]:
    if not material_keys:
        raise ValueError("Select at least one material.")
    if not wavelengths_nm:
        raise ValueError("Enter at least one wavelength.")
    if len(wavelengths_nm) > 4:
        raise ValueError("Compare at most four wavelengths to keep line styles unambiguous.")
    if any((not math.isfinite(v) or v <= 0) for v in wavelengths_nm):
        raise ValueError("All wavelengths must be finite and positive.")
    if not (math.isfinite(diameter_min_nm) and math.isfinite(diameter_max_nm)):
        raise ValueError("Diameter limits must be finite.")
    if diameter_min_nm <= 0 or diameter_max_nm <= diameter_min_nm:
        raise ValueError("The diameter range must satisfy 0 < D_min < D_max.")
    if not 20 <= point_count <= 5000:
        raise ValueError("The number of samples must be between 20 and 5000.")
    if not math.isfinite(incident_irradiance_w_cm2) or incident_irradiance_w_cm2 <= 0:
        raise ValueError("Incident irradiance must be finite and positive.")

    diameter = (
        np.geomspace(diameter_min_nm, diameter_max_nm, point_count)
        if log_diameter_grid
        else np.linspace(diameter_min_nm, diameter_max_nm, point_count)
    )
    output: list[Series] = []
    for material_key in material_keys:
        for wavelength_nm in wavelengths_nm:
            nm = medium_n(wavelength_nm, medium_key, custom_medium_n)
            nparticle = particle_n(material_key, wavelength_nm, alumina_model)
            qsca = mie_qsca(diameter, wavelength_nm, nparticle, nm)
            # C_sca is the ideal total scattered power divided by incident
            # irradiance. The factor 1e6 converts nm^2 to um^2.
            csca_um2 = qsca * np.pi * (diameter / 2.0) ** 2 / 1.0e6
            # W/cm^2 * um^2 = 1e4 pW.
            psca_pw = incident_irradiance_w_cm2 * csca_um2 * 1.0e4
            output.append(
                Series(
                    material_key=material_key,
                    material_label=MATERIAL_LABELS[material_key],
                    wavelength_nm=wavelength_nm,
                    particle_index=nparticle,
                    medium_index=nm,
                    diameter_nm=diameter.copy(),
                    qsca=qsca,
                    csca_um2=csca_um2,
                    incident_irradiance_w_cm2=incident_irradiance_w_cm2,
                    psca_pw=psca_pw,
                )
            )
    return output


def parse_wavelengths(text: str) -> list[float]:
    cleaned = text.replace(";", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Enter comma-separated wavelengths, for example: 405, 633.") from exc
    # Preserve order while removing duplicate, fully overlapping curves.
    unique: list[float] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def validate_reference_settings(settings: ReferenceSettings) -> None:
    if not math.isfinite(settings.exposure_s) or settings.exposure_s <= 0:
        raise ValueError("Exposure time must be finite and positive.")
    if not 1 <= settings.roi_pixels <= 1_000_000:
        raise ValueError("ROI pixel count must be between 1 and 1,000,000.")
    if any(key not in CAMERA_MODELS for key in settings.camera_keys):
        raise ValueError("Unknown camera model selected for a camera-only reference.")
    if settings.measured_background_camera_key not in CAMERA_MODELS:
        raise ValueError("Unknown camera model selected for the measured-background reference.")
    if (
        not math.isfinite(settings.measured_background_e_roi)
        or settings.measured_background_e_roi < 0
    ):
        raise ValueError("Measured background must be a finite, non-negative electron count.")
    if settings.show_measured_background and settings.measured_background_e_roi <= 0:
        raise ValueError("Enter a positive measured background before enabling its reference line.")
    if (
        not math.isfinite(settings.background_irradiance_w_cm2)
        or settings.background_irradiance_w_cm2 <= 0
    ):
        raise ValueError("Background irradiance must be finite and positive.")


def photon_energy_j(wavelength_nm: float) -> float:
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("Wavelength must be finite and positive.")
    return PLANCK_CONSTANT * LIGHT_SPEED / (wavelength_nm * 1.0e-9)


def camera_noise_equivalent_power_pw(
    wavelength_nm: float,
    camera_key: str,
    exposure_s: float,
    roi_pixels: int,
    background_e_roi: float = 0.0,
) -> float:
    """Return the SNR=1 camera-input power for one exposure and one ROI.

    The manufacturer peak QE is used as an idealized camera reference. The
    returned power is later divided by I0 for display as an equivalent
    scattering cross-section. No collection-efficiency model is included.
    """
    if camera_key not in CAMERA_MODELS:
        raise ValueError(f"Unknown camera model: {camera_key}")
    camera = CAMERA_MODELS[camera_key]
    if exposure_s <= 0 or roi_pixels < 1 or background_e_roi < 0:
        raise ValueError("Camera-reference inputs are outside their physical range.")
    variance_e2 = roi_pixels * (
        camera["read_noise_rms_e"] ** 2
        + camera["dark_current_e_pixel_s"] * exposure_s
    ) + background_e_roi
    noise_e = math.sqrt(variance_e2)
    input_energy_j = noise_e * photon_energy_j(wavelength_nm) / camera["qe"]
    return input_energy_j / exposure_s * 1.0e12


def background_field_power_pw(
    diameter_nm: np.ndarray | float,
    irradiance_w_cm2: float,
) -> np.ndarray:
    """Power in the background field over the particle's projected area."""
    diameter = np.asarray(diameter_nm, dtype=float)
    return irradiance_w_cm2 * np.pi * (diameter / 2.0) ** 2 * 1.0e-2


def power_pw_to_equivalent_csca_um2(
    power_pw: np.ndarray | float,
    incident_irradiance_w_cm2: float,
) -> np.ndarray:
    """Normalize power by I0 and return its equivalent area in um^2."""
    if not math.isfinite(incident_irradiance_w_cm2) or incident_irradiance_w_cm2 <= 0:
        raise ValueError("Incident irradiance must be finite and positive.")
    return np.asarray(power_pw, dtype=float) / (incident_irradiance_w_cm2 * 1.0e4)


def background_field_equivalent_csca_um2(
    diameter_nm: np.ndarray | float,
    background_irradiance_w_cm2: float,
    incident_irradiance_w_cm2: float,
) -> np.ndarray:
    """Background power over the projected area, normalized by illumination I0."""
    return power_pw_to_equivalent_csca_um2(
        background_field_power_pw(diameter_nm, background_irradiance_w_cm2),
        incident_irradiance_w_cm2,
    )


def first_upward_crossing(
    diameter_nm: np.ndarray,
    signal: np.ndarray,
    reference: np.ndarray | float,
) -> tuple[float, float] | None:
    """Return the first below-to-above crossing using local linear interpolation."""
    diameter = np.asarray(diameter_nm, dtype=float)
    signal_values = np.asarray(signal, dtype=float)
    reference_values = np.broadcast_to(np.asarray(reference, dtype=float), signal_values.shape)
    delta = signal_values - reference_values
    indices = np.flatnonzero((delta[:-1] < 0.0) & (delta[1:] >= 0.0))
    if indices.size == 0:
        return None
    index = int(indices[0])
    denominator = delta[index + 1] - delta[index]
    fraction = 0.0 if denominator == 0 else -delta[index] / denominator
    crossing_diameter = diameter[index] + fraction * (diameter[index + 1] - diameter[index])
    crossing_signal = signal_values[index] + fraction * (
        signal_values[index + 1] - signal_values[index]
    )
    return float(crossing_diameter), float(crossing_signal)


def compute_reference_intersections(
    series_list: list[Series],
    settings: ReferenceSettings,
) -> list[Intersection]:
    validate_reference_settings(settings)
    output: list[Intersection] = []
    for item in series_list:
        for camera_key in settings.camera_keys:
            limit_pw = camera_noise_equivalent_power_pw(
                item.wavelength_nm,
                camera_key,
                settings.exposure_s,
                settings.roi_pixels,
            )
            limit_csca_um2 = power_pw_to_equivalent_csca_um2(
                limit_pw, item.incident_irradiance_w_cm2
            )
            crossing = first_upward_crossing(
                item.diameter_nm, item.csca_um2, limit_csca_um2
            )
            if crossing is not None:
                output.append(
                    Intersection(
                        reference_kind="camera_only_nep",
                        reference_label=f"{CAMERA_MODELS[camera_key]['short_label']} camera-only NEP",
                        camera_key=camera_key,
                        material_key=item.material_key,
                        material_label=item.material_label,
                        wavelength_nm=item.wavelength_nm,
                        diameter_nm=crossing[0],
                        normalized_signal_um2=crossing[1],
                    )
                )
        if settings.show_measured_background:
            camera_key = settings.measured_background_camera_key
            limit_pw = camera_noise_equivalent_power_pw(
                item.wavelength_nm,
                camera_key,
                settings.exposure_s,
                settings.roi_pixels,
                settings.measured_background_e_roi,
            )
            limit_csca_um2 = power_pw_to_equivalent_csca_um2(
                limit_pw, item.incident_irradiance_w_cm2
            )
            crossing = first_upward_crossing(
                item.diameter_nm, item.csca_um2, limit_csca_um2
            )
            if crossing is not None:
                output.append(
                    Intersection(
                        reference_kind="measured_background_nep",
                        reference_label="Measured-background NEP",
                        camera_key=camera_key,
                        material_key=item.material_key,
                        material_label=item.material_label,
                        wavelength_nm=item.wavelength_nm,
                        diameter_nm=crossing[0],
                        normalized_signal_um2=crossing[1],
                    )
                )
        if settings.show_background_field:
            reference_curve = background_field_equivalent_csca_um2(
                item.diameter_nm,
                settings.background_irradiance_w_cm2,
                item.incident_irradiance_w_cm2,
            )
            crossing = first_upward_crossing(item.diameter_nm, item.csca_um2, reference_curve)
            if crossing is not None:
                output.append(
                    Intersection(
                        reference_kind="background_field_equality",
                        reference_label="Background-field equality",
                        camera_key="",
                        material_key=item.material_key,
                        material_label=item.material_label,
                        wavelength_nm=item.wavelength_nm,
                        diameter_nm=crossing[0],
                        normalized_signal_um2=crossing[1],
                    )
                )
    return output


def rayleigh_qsca(size_parameter: np.ndarray, relative_index: complex) -> np.ndarray:
    """Rayleigh limit for tests: Q_sca = 8/3*x^4*|(m^2-1)/(m^2+2)|^2."""
    x = np.asarray(size_parameter, dtype=float)
    m2 = complex(relative_index) ** 2
    return (8.0 / 3.0) * x**4 * abs((m2 - 1.0) / (m2 + 2.0)) ** 2


def run_selftest() -> int:
    """Quick checks for electrochemical optical data and normalized scattering."""
    checks: list[tuple[str, bool, str]] = []

    reference_nodes = {
        ("Pt", 551.041): complex(0.4643, 5.1210),
        ("Ni", 549.0): complex(1.92, 3.61),
        ("TiO2", 534.41): complex(2.52883, 0.0),
        ("ITO", 530.02): complex(1.883299, 0.00342657),
    }
    node_errors = []
    for (key, wavelength), expected in reference_nodes.items():
        measured = particle_n(key, wavelength, "o")
        node_errors.append(abs(measured - expected))
    maximum_node_error = max(node_errors)
    checks.append((
        "electrochemical-material optical tables",
        maximum_node_error < 1e-10,
        f"maximum exact-node error={maximum_node_error:.3e}",
    ))

    passivity_ok = True
    passivity_details = []
    particle_x = np.pi * 80.0 / 532.0
    for key in ("Au", "Pt", "Ni", "TiO2", "ITO"):
        nk = particle_n(key, 532.0, "o")
        qext, qsca_value, _qback, _g = miepython.efficiencies_mx(np.conj(nk), particle_x)
        qabs = float(qext - qsca_value)
        passivity_ok = passivity_ok and qabs >= -1e-10 and qsca_value >= 0
        passivity_details.append(f"{key}:{qabs:.3g}")
    checks.append((
        "passive-index sign convention",
        passivity_ok,
        "Qabs=" + ", ".join(passivity_details),
    ))

    n_tio2 = particle_n("TiO2", 532.0, "o")
    diameter = np.array([1.0, 2.0, 5.0])
    exact = mie_qsca(diameter, 532.0, n_tio2, 1.0)
    x = np.pi * diameter / 532.0
    approx = rayleigh_qsca(x, n_tio2)
    rel_error = float(np.max(np.abs(exact - approx) / np.maximum(approx, 1e-300)))
    checks.append(("Rayleigh limit", rel_error < 5e-3, f"max relative error={rel_error:.3e}"))

    default_materials = ["Au", "Pt", "Ni", "TiO2", "ITO"]
    default_series = compute_series(
        material_keys=default_materials,
        wavelengths_nm=[405.0, 633.0],
        diameter_min_nm=5.0,
        diameter_max_nm=5000.0,
        point_count=240,
        log_diameter_grid=True,
        medium_key="air",
        custom_medium_n=1.333,
        alumina_model="o",
        incident_irradiance_w_cm2=1.0,
    )
    finite_default = all(
        np.all(np.isfinite(item.csca_um2)) and np.all(item.csca_um2 >= 0)
        for item in default_series
    )
    endpoint_growth = all(item.csca_um2[-1] > item.csca_um2[0] for item in default_series)
    checks.append((
        "default 10-curve, 5-5000 nm calculation",
        len(default_series) == 10 and finite_default,
        f"curves={len(default_series)}, finite_nonnegative={finite_default}",
    ))
    checks.append((
        "overall normalized size-dependent growth",
        endpoint_growth,
        f"all Csca(5000 nm) > Csca(5 nm): {endpoint_growth}",
    ))

    brighter_series = compute_series(
        material_keys=["Pt"],
        wavelengths_nm=[633.0],
        diameter_min_nm=5.0,
        diameter_max_nm=5000.0,
        point_count=80,
        log_diameter_grid=True,
        medium_key="air",
        custom_medium_n=1.333,
        alumina_model="o",
        incident_irradiance_w_cm2=7.0,
    )[0]
    base_series = next(
        item for item in default_series
        if item.material_key == "Pt" and item.wavelength_nm == 633.0
    )
    comparison_series = compute_series(
        material_keys=["Pt"],
        wavelengths_nm=[633.0],
        diameter_min_nm=5.0,
        diameter_max_nm=5000.0,
        point_count=80,
        log_diameter_grid=True,
        medium_key="air",
        custom_medium_n=1.333,
        alumina_model="o",
        incident_irradiance_w_cm2=1.0,
    )[0]
    normalized_invariance = np.allclose(
        brighter_series.csca_um2, comparison_series.csca_um2, rtol=1e-13, atol=0.0
    )
    power_ratio_ok = np.allclose(
        brighter_series.psca_pw, 7.0 * comparison_series.psca_pw, rtol=1e-13, atol=0.0
    )
    checks.append((
        "illumination normalization",
        bool(normalized_invariance and power_ratio_ok and base_series.csca_um2[-1] > 0),
        f"Csca invariant={normalized_invariance}, Psca ratio correct={power_ratio_ok}",
    ))

    background_equivalent = float(background_field_equivalent_csca_um2(100.0, 1.0, 2.0))
    expected_background_equivalent = 0.5 * np.pi * (0.05**2)
    checks.append((
        "normalized background-area conversion",
        abs(background_equivalent - expected_background_equivalent) < 1e-14,
        f"equivalent Csca={background_equivalent:.9g} um^2",
    ))

    reference_settings = ReferenceSettings(
        exposure_s=0.1,
        roi_pixels=9,
        camera_keys=("orca_flash_v3", "prime_bsi"),
        show_measured_background=False,
        measured_background_e_roi=0.0,
        measured_background_camera_key="orca_flash_v3",
        show_background_field=True,
        background_irradiance_w_cm2=1.0,
        label_intersections=False,
    )
    intersections = compute_reference_intersections(default_series, reference_settings)
    intersections_ok = bool(intersections) and all(
        math.isfinite(item.normalized_signal_um2) and item.normalized_signal_um2 > 0
        for item in intersections
    )
    checks.append((
        "normalized reference intersections",
        intersections_ok,
        f"first upward crossings={len(intersections)}",
    ))

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    smoke_figure = Figure(figsize=EXPORT_FIGURE_SIZE, dpi=100, facecolor="white")
    smoke_axis = smoke_figure.add_subplot(111)
    plot_series_on_axis(
        smoke_axis,
        default_series,
        log_x=True,
        log_y=True,
        reference_settings=reference_settings,
        intersections=intersections,
        legend_location="custom",
        legend_anchor=(0.70, 0.22),
    )
    FigureCanvasAgg(smoke_figure).draw()
    checks.append((
        "log-log single-axis plot render",
        len(smoke_figure.axes) == 1
        and smoke_axis.get_xscale() == "log"
        and smoke_axis.get_yscale() == "log",
        f"axes={len(smoke_figure.axes)}, scales={smoke_axis.get_xscale()}/{smoke_axis.get_yscale()}",
    ))
    checks.append(("camera lower-limit bands", len(smoke_axis.patches) == 2,
                   f"horizontal bands={len(smoke_axis.patches)}"))
    checks.append(("custom legend render", smoke_axis.get_legend() is not None,
                   "custom center=(0.70, 0.22)"))
    preview_width, preview_height, preview_dpi = wysiwyg_preview_geometry(1000, 800)
    preview_aspect = preview_width / preview_height
    export_aspect = EXPORT_FIGURE_SIZE[0] / EXPORT_FIGURE_SIZE[1]
    typography_matches = (
        smoke_axis.title.get_fontsize() == PLOT_TITLE_SIZE
        and smoke_axis.xaxis.label.get_fontsize() == PLOT_AXIS_LABEL_SIZE
        and all(text.get_fontsize() == PLOT_LEGEND_SIZE for text in smoke_axis.get_legend().texts)
    )
    checks.append(("WYSIWYG preview geometry", abs(preview_aspect - export_aspect) < 0.002,
                   f"preview={preview_width}x{preview_height}, dpi={preview_dpi:.3f}"))
    checks.append(("shared preview/export typography", typography_matches,
                   f"title={PLOT_TITLE_SIZE:g}, axes={PLOT_AXIS_LABEL_SIZE:g}, legend={PLOT_LEGEND_SIZE:g} pt"))

    failed = 0
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
        failed += int(not passed)
    return failed


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": PLOT_TITLE_SIZE,
            "axes.labelsize": PLOT_AXIS_LABEL_SIZE,
            "xtick.labelsize": PLOT_TICK_SIZE,
            "ytick.labelsize": PLOT_TICK_SIZE,
            "legend.fontsize": PLOT_LEGEND_SIZE,
            "figure.dpi": 100,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def apply_publication_axis_style(axis) -> None:
    """Apply the shared publication typography and axis treatment."""
    axis.grid(True, which="both", color="0.82", alpha=0.55, linewidth=0.6)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_TICK_SIZE,
        width=0.9,
        length=4.0,
    )
    axis.tick_params(axis="both", which="minor", width=0.7, length=2.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)


def wysiwyg_preview_geometry(
    container_width_px: int,
    container_height_px: int,
    padding_px: int = 8,
) -> tuple[int, int, float]:
    """Fit the export aspect ratio and return width, height, and display DPI."""
    available_width = max(1, int(container_width_px) - padding_px)
    available_height = max(1, int(container_height_px) - padding_px)
    aspect = EXPORT_FIGURE_SIZE[0] / EXPORT_FIGURE_SIZE[1]
    if available_width / available_height >= aspect:
        target_height = available_height
        target_width = int(round(target_height * aspect))
    else:
        target_width = available_width
        target_height = int(round(target_width / aspect))
    target_width = max(1, target_width)
    target_height = max(1, target_height)
    display_dpi = target_width / EXPORT_FIGURE_SIZE[0]
    return target_width, target_height, display_dpi


def plot_series_on_axis(
    axis,
    series_list: list[Series],
    *,
    log_x: bool,
    log_y: bool,
    reference_settings: ReferenceSettings,
    intersections: list[Intersection],
    legend_location: str = "lower right",
    legend_anchor: tuple[float, float] | None = None,
) -> None:
    """Draw illumination-normalized scattering and normalized references."""
    validate_reference_settings(reference_settings)
    if not series_list:
        raise ValueError("At least one scattering series is required for plotting.")
    axis.clear()
    incident_irradiance = series_list[0].incident_irradiance_w_cm2
    wavelengths: list[float] = []
    for item in series_list:
        if item.wavelength_nm not in wavelengths:
            wavelengths.append(item.wavelength_nm)
    wavelength_style = {
        wavelength: LINE_STYLES[index % len(LINE_STYLES)]
        for index, wavelength in enumerate(wavelengths)
    }
    for item in series_list:
        axis.plot(
            item.diameter_nm,
            item.csca_um2,
            color=SERIES_COLORS[item.material_key],
            linestyle=wavelength_style[item.wavelength_nm],
            linewidth=PLOT_LINE_WIDTH,
            zorder=3,
            label=f"{item.material_label}, {item.wavelength_nm:g} nm",
        )

    axis.set_xlabel(
        "Particle diameter, D (nm)",
        fontsize=PLOT_AXIS_LABEL_SIZE,
        labelpad=7,
    )
    axis.set_ylabel(
        r"Normalized scattering, $P_{\mathrm{sca}}/I_0=C_{\mathrm{sca}}$ ($\mu$m$^2$)",
        fontsize=PLOT_AXIS_LABEL_SIZE,
        labelpad=7,
    )
    axis.set_title(
        "Illumination-Normalized Mie Scattering with Dark-Field References",
        fontsize=PLOT_TITLE_SIZE,
        pad=10,
    )
    axis.set_xscale("log" if log_x else "linear")
    axis.set_yscale("log" if log_y else "linear")

    camera_reference_ranges: dict[str, tuple[float, float]] = {}
    for camera_key in reference_settings.camera_keys:
        camera = CAMERA_MODELS[camera_key]
        limits_um2 = [
            float(
                power_pw_to_equivalent_csca_um2(
                    camera_noise_equivalent_power_pw(
                        wavelength,
                        camera_key,
                        reference_settings.exposure_s,
                        reference_settings.roi_pixels,
                    ),
                    incident_irradiance,
                )
            )
            for wavelength in wavelengths
        ]
        lower_um2 = min(limits_um2)
        upper_um2 = max(limits_um2)
        camera_reference_ranges[camera_key] = (lower_um2, upper_um2)
        if upper_um2 > lower_um2:
            axis.axhspan(
                lower_um2,
                upper_um2,
                facecolor=camera["color"],
                edgecolor=camera["color"],
                linewidth=0.8,
                alpha=0.18,
                zorder=0,
            )
        else:
            axis.axhline(
                lower_um2,
                color=camera["color"],
                linestyle=camera["linestyle"],
                linewidth=1.2,
                alpha=0.9,
                zorder=1,
            )

    measured_background_range: tuple[float, float] | None = None
    if reference_settings.show_measured_background:
        camera_key = reference_settings.measured_background_camera_key
        limits_um2 = [
            float(
                power_pw_to_equivalent_csca_um2(
                    camera_noise_equivalent_power_pw(
                        wavelength,
                        camera_key,
                        reference_settings.exposure_s,
                        reference_settings.roi_pixels,
                        reference_settings.measured_background_e_roi,
                    ),
                    incident_irradiance,
                )
            )
            for wavelength in wavelengths
        ]
        lower_um2 = min(limits_um2)
        upper_um2 = max(limits_um2)
        measured_background_range = (lower_um2, upper_um2)
        if upper_um2 > lower_um2:
            axis.axhspan(
                lower_um2,
                upper_um2,
                facecolor=BACKGROUND_LIMIT_COLOR,
                edgecolor=BACKGROUND_LIMIT_COLOR,
                linewidth=0.8,
                alpha=0.18,
                zorder=0,
            )
        else:
            axis.axhline(
                lower_um2,
                color=BACKGROUND_LIMIT_COLOR,
                linestyle="-.",
                linewidth=1.2,
                alpha=0.9,
                zorder=1,
            )

    if reference_settings.show_background_field:
        diameter = series_list[0].diameter_nm
        background_csca = background_field_equivalent_csca_um2(
            diameter,
            reference_settings.background_irradiance_w_cm2,
            incident_irradiance,
        )
        axis.plot(
            diameter,
            background_csca,
            color=BACKGROUND_FIELD_COLOR,
            linestyle=(0, (8, 2, 1.5, 2)),
            linewidth=2.25,
            alpha=0.98,
            zorder=4,
            label="_nolegend_",
        )

    marker_by_kind = {
        "camera_only_nep": "v",
        "measured_background_nep": "D",
        "background_field_equality": "s",
    }
    for index, crossing in enumerate(intersections):
        marker = marker_by_kind[crossing.reference_kind]
        marker_edge_color = (
            BACKGROUND_FIELD_COLOR
            if crossing.reference_kind == "background_field_equality"
            else SERIES_COLORS[crossing.material_key]
        )
        axis.plot(
            crossing.diameter_nm,
            crossing.normalized_signal_um2,
            marker=marker,
            markersize=5.0 if crossing.reference_kind == "background_field_equality" else 4.0,
            markerfacecolor="white",
            markeredgecolor=marker_edge_color,
            markeredgewidth=1.25,
            linestyle="none",
            zorder=6,
            label="_nolegend_",
        )
        if reference_settings.label_intersections:
            vertical_offset = 4 if index % 2 == 0 else -10
            axis.annotate(
                f"{crossing.diameter_nm:.1f} nm",
                xy=(crossing.diameter_nm, crossing.normalized_signal_um2),
                xytext=(3, vertical_offset),
                textcoords="offset points",
                fontsize=7.0,
                color=SERIES_COLORS[crossing.material_key],
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.3},
            )

    axis.text(
        0.015,
        0.975,
        (
            rf"$I_0={incident_irradiance:g}\;\mathrm{{W/cm^2}}$; "
            rf"$t={reference_settings.exposure_s * 1000:g}\;\mathrm{{ms}}$; "
            rf"ROI={reference_settings.roi_pixels} px; ideal $4\pi$ total"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=PLOT_ANNOTATION_SIZE,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )

    apply_publication_axis_style(axis)
    material_order: list[str] = []
    for item in series_list:
        if item.material_key not in material_order:
            material_order.append(item.material_key)
    handles = [
        Line2D([0], [0], color=SERIES_COLORS[key], linewidth=PLOT_LINE_WIDTH)
        for key in material_order
    ]
    labels = [MATERIAL_LABELS[key] for key in material_order]
    handles.extend(
        Line2D([0], [0], color="0.25", linestyle=wavelength_style[wavelength],
               linewidth=PLOT_LINE_WIDTH)
        for wavelength in wavelengths
    )
    labels.extend(f"{wavelength:g} nm" for wavelength in wavelengths)
    for camera_key in reference_settings.camera_keys:
        camera = CAMERA_MODELS[camera_key]
        lower_um2, upper_um2 = camera_reference_ranges[camera_key]
        if upper_um2 > lower_um2:
            handles.append(
                Patch(
                    facecolor=camera["color"],
                    edgecolor=camera["color"],
                    linewidth=0.8,
                    alpha=0.18,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0], [0], color=camera["color"], linestyle=camera["linestyle"],
                    linewidth=1.2, marker="v", markerfacecolor="white", markersize=4.0
                )
            )
        labels.append(
            ("ORCA camera NEP range" if upper_um2 > lower_um2 else "ORCA camera NEP")
            if camera_key == "orca_flash_v3"
            else (
                "High-QE camera NEP range"
                if upper_um2 > lower_um2
                else "High-QE camera NEP"
            )
        )
    if reference_settings.show_measured_background:
        assert measured_background_range is not None
        lower_um2, upper_um2 = measured_background_range
        if upper_um2 > lower_um2:
            handles.append(
                Patch(
                    facecolor=BACKGROUND_LIMIT_COLOR,
                    edgecolor=BACKGROUND_LIMIT_COLOR,
                    linewidth=0.8,
                    alpha=0.18,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0], [0], color=BACKGROUND_LIMIT_COLOR, linestyle="-.",
                    linewidth=1.2, marker="D", markerfacecolor="white", markersize=4.0
                )
            )
        labels.append(
            "Measured-background NEP range"
            if upper_um2 > lower_um2
            else "Measured-background NEP"
        )
    if reference_settings.show_background_field:
        handles.append(
            Line2D(
                [0], [0], color=BACKGROUND_FIELD_COLOR, linestyle=(0, (8, 2, 1.5, 2)),
                linewidth=2.25, marker="s", markerfacecolor="white", markersize=5.0
            )
        )
        labels.append(r"$C_{sca}=P_{bg}/I_0$")
    valid_legend_locations = set(LEGEND_POSITIONS.values())
    if legend_location not in valid_legend_locations:
        raise ValueError(f"Unknown legend location: {legend_location}")
    legend_kwargs: dict[str, object] = {"loc": legend_location}
    if legend_location == "custom":
        if legend_anchor is None:
            raise ValueError("Custom legend placement requires X and Y axes coordinates.")
        legend_x, legend_y = legend_anchor
        if not (
            math.isfinite(legend_x)
            and math.isfinite(legend_y)
            and 0.0 <= legend_x <= 1.0
            and 0.0 <= legend_y <= 1.0
        ):
            raise ValueError("Custom legend X and Y must both be between 0 and 1.")
        legend_kwargs = {
            "loc": "center",
            "bbox_to_anchor": (legend_x, legend_y),
            "bbox_transform": axis.transAxes,
        }
    axis.legend(
        handles,
        labels,
        ncols=3 if len(labels) > 6 else (2 if len(labels) > 3 else 1),
        fontsize=PLOT_LEGEND_SIZE,
        frameon=True,
        framealpha=0.94,
        borderpad=0.45,
        handlelength=2.7,
        columnspacing=1.0,
        handletextpad=0.55,
        **legend_kwargs,
    )


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    _configure_matplotlib()

    root = tk.Tk()
    root.title("Dark-Field Mie Scattering: Electrochemical Materials")
    root.geometry("1450x930")
    root.minsize(1180, 780)

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    outer = ttk.Frame(root, padding=10)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(1, weight=1)
    outer.rowconfigure(0, weight=1)

    control_host = ttk.Frame(outer, padding=(0, 0, 12, 0))
    control_host.grid(row=0, column=0, sticky="nsew")
    control_host.columnconfigure(0, weight=1)
    control_host.rowconfigure(0, weight=1)

    control_canvas = tk.Canvas(
        control_host,
        width=355,
        highlightthickness=0,
        borderwidth=0,
    )
    control_scrollbar = ttk.Scrollbar(
        control_host,
        orient="vertical",
        command=control_canvas.yview,
    )
    controls = ttk.Frame(control_canvas, padding=(0, 0, 4, 0))
    controls_window = control_canvas.create_window(
        (0, 0),
        window=controls,
        anchor="nw",
    )
    control_canvas.configure(yscrollcommand=control_scrollbar.set)
    control_canvas.grid(row=0, column=0, sticky="nsew")
    control_scrollbar.grid(row=0, column=1, sticky="ns")

    def update_control_scrollregion(_event: object | None = None) -> None:
        control_canvas.configure(scrollregion=control_canvas.bbox("all"))

    def fit_controls_to_canvas(event: object) -> None:
        canvas_width = getattr(event, "width", control_canvas.winfo_width())
        control_canvas.itemconfigure(controls_window, width=canvas_width)

    def scroll_controls(event: object) -> str:
        delta = getattr(event, "delta", 0)
        if delta:
            control_canvas.yview_scroll(int(-delta / 120), "units")
        return "break"

    controls.bind("<Configure>", update_control_scrollregion)
    control_canvas.bind("<Configure>", fit_controls_to_canvas)
    control_canvas.bind("<Enter>", lambda _event: control_canvas.bind_all("<MouseWheel>", scroll_controls))
    control_canvas.bind("<Leave>", lambda _event: control_canvas.unbind_all("<MouseWheel>"))

    plot_frame = ttk.Frame(outer)
    plot_frame.grid(row=0, column=1, sticky="nsew")

    def section(title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(controls, text=title, padding=8)
        frame.pack(fill="x", pady=(0, 8))
        frame.columnconfigure(1, weight=1)
        return frame

    material_frame = section("Materials")
    use_au = tk.BooleanVar(value=True)
    use_pt = tk.BooleanVar(value=True)
    use_ni = tk.BooleanVar(value=True)
    use_tio2 = tk.BooleanVar(value=True)
    use_ito = tk.BooleanVar(value=True)
    use_ag = tk.BooleanVar(value=False)
    use_cu = tk.BooleanVar(value=False)
    ttk.Checkbutton(material_frame, text="Au (gold)", variable=use_au).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Checkbutton(material_frame, text="Pt (platinum)", variable=use_pt).grid(
        row=0, column=1, sticky="w", padx=(8, 0)
    )
    ttk.Checkbutton(material_frame, text="Ni (nickel)", variable=use_ni).grid(
        row=1, column=0, sticky="w"
    )
    ttk.Checkbutton(material_frame, text="TiO2 (anatase film)", variable=use_tio2).grid(
        row=1, column=1, sticky="w", padx=(8, 0)
    )
    ttk.Checkbutton(material_frame, text="ITO film", variable=use_ito).grid(
        row=2, column=0, sticky="w"
    )
    ttk.Checkbutton(material_frame, text="Ag (silver)", variable=use_ag).grid(
        row=2, column=1, sticky="w", padx=(8, 0)
    )
    ttk.Checkbutton(material_frame, text="Cu (copper)", variable=use_cu).grid(
        row=3, column=0, sticky="w"
    )

    optical_frame = section("Optical parameters")
    ttk.Label(optical_frame, text="Vacuum wavelength (nm)").grid(row=0, column=0, sticky="w")
    wavelength_text = tk.StringVar(value="405, 633")
    ttk.Entry(optical_frame, textvariable=wavelength_text, width=22).grid(
        row=0, column=1, sticky="ew", padx=(6, 0)
    )
    ttk.Label(optical_frame, text="Incident irradiance (W/cm^2)").grid(
        row=1, column=0, sticky="w", pady=(5, 0)
    )
    incident_irradiance_text = tk.StringVar(value="1.0")
    ttk.Entry(optical_frame, textvariable=incident_irradiance_text, width=10).grid(
        row=1, column=1, sticky="ew", padx=(6, 0), pady=(5, 0)
    )
    ttk.Label(optical_frame, text="Surrounding medium").grid(row=2, column=0, sticky="w", pady=(5, 0))
    medium_choice = tk.StringVar(value="Air (n = 1.000)")
    medium_combo = ttk.Combobox(
        optical_frame,
        textvariable=medium_choice,
        values=list(MEDIUM_LABELS),
        state="readonly",
        width=19,
    )
    medium_combo.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=(5, 0))
    ttk.Label(optical_frame, text="Custom medium n").grid(row=3, column=0, sticky="w", pady=(5, 0))
    custom_n_text = tk.StringVar(value="1.333")
    custom_n_entry = ttk.Entry(optical_frame, textvariable=custom_n_text, width=10, state="disabled")
    custom_n_entry.grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=(5, 0))

    size_frame = section("Particle diameter")
    diameter_min_text = tk.StringVar(value="5")
    diameter_max_text = tk.StringVar(value="5000")
    point_count_text = tk.StringVar(value="1200")
    for row, (label, variable) in enumerate(
        (
            ("Minimum D (nm)", diameter_min_text),
            ("Maximum D (nm)", diameter_max_text),
            ("Number of samples", point_count_text),
        )
    ):
        ttk.Label(size_frame, text=label).grid(row=row, column=0, sticky="w", pady=(3 if row else 0, 0))
        ttk.Entry(size_frame, textvariable=variable, width=10).grid(
            row=row, column=1, sticky="ew", padx=(6, 0), pady=(3 if row else 0, 0)
        )

    axes_frame = section("Display")
    log_x = tk.BooleanVar(value=True)
    log_y = tk.BooleanVar(value=True)
    ttk.Checkbutton(axes_frame, text="Logarithmic diameter axis", variable=log_x).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Checkbutton(axes_frame, text="Logarithmic normalized axis", variable=log_y).grid(
        row=1, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(axes_frame, text="Legend position").grid(
        row=2, column=0, sticky="w", pady=(5, 0)
    )
    legend_choice = tk.StringVar(value="Lower right (default)")
    legend_combo = ttk.Combobox(
        axes_frame,
        textvariable=legend_choice,
        values=list(LEGEND_POSITIONS),
        state="readonly",
        width=21,
    )
    legend_combo.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=(5, 0))
    ttk.Label(axes_frame, text="Custom center X (0-1)").grid(
        row=3, column=0, sticky="w", pady=(4, 0)
    )
    legend_x_text = tk.StringVar(value="0.75")
    legend_x_entry = ttk.Entry(
        axes_frame,
        textvariable=legend_x_text,
        width=10,
        state="disabled",
    )
    legend_x_entry.grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
    ttk.Label(axes_frame, text="Custom center Y (0-1)").grid(
        row=4, column=0, sticky="w", pady=(4, 0)
    )
    legend_y_text = tk.StringVar(value="0.25")
    legend_y_entry = ttk.Entry(
        axes_frame,
        textvariable=legend_y_text,
        width=10,
        state="disabled",
    )
    legend_y_entry.grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))

    reference_frame = section("Camera and background references")
    exposure_ms_text = tk.StringVar(value="100")
    roi_pixels_text = tk.StringVar(value="9")
    show_orca_nep = tk.BooleanVar(value=True)
    show_prime_nep = tk.BooleanVar(value=True)
    show_measured_background = tk.BooleanVar(value=False)
    measured_background_e_text = tk.StringVar(value="0")
    measured_background_camera = tk.StringVar(value=CAMERA_MODELS["orca_flash_v3"]["label"])
    show_background_field = tk.BooleanVar(value=True)
    background_irradiance_text = tk.StringVar(value="1.0")
    label_intersections = tk.BooleanVar(value=False)

    ttk.Label(reference_frame, text="Exposure time (ms)").grid(row=0, column=0, sticky="w")
    ttk.Entry(reference_frame, textvariable=exposure_ms_text, width=10).grid(
        row=0, column=1, sticky="ew", padx=(6, 0)
    )
    ttk.Label(reference_frame, text="Particle ROI (pixels)").grid(
        row=1, column=0, sticky="w", pady=(4, 0)
    )
    ttk.Entry(reference_frame, textvariable=roi_pixels_text, width=10).grid(
        row=1, column=1, sticky="ew", padx=(6, 0), pady=(4, 0)
    )
    ttk.Checkbutton(
        reference_frame,
        text="ORCA-Flash4.0 V3 camera-only NEP band",
        variable=show_orca_nep,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
    ttk.Checkbutton(
        reference_frame,
        text="Prime BSI high-QE camera-only NEP band",
        variable=show_prime_nep,
    ).grid(row=3, column=0, columnspan=2, sticky="w")
    ttk.Checkbutton(
        reference_frame,
        text="Measured-background NEP band",
        variable=show_measured_background,
    ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
    ttk.Label(reference_frame, text="Background camera").grid(
        row=5, column=0, sticky="w", pady=(3, 0)
    )
    measured_background_camera_combo = ttk.Combobox(
        reference_frame,
        textvariable=measured_background_camera,
        values=list(CAMERA_LABEL_TO_KEY),
        state="disabled",
        width=22,
    )
    measured_background_camera_combo.grid(
        row=5, column=1, sticky="ew", padx=(6, 0), pady=(3, 0)
    )
    ttk.Label(reference_frame, text="Background (e-/ROI/exposure)").grid(
        row=6, column=0, sticky="w", pady=(3, 0)
    )
    measured_background_entry = ttk.Entry(
        reference_frame,
        textvariable=measured_background_e_text,
        width=10,
        state="disabled",
    )
    measured_background_entry.grid(
        row=6, column=1, sticky="ew", padx=(6, 0), pady=(3, 0)
    )
    ttk.Checkbutton(
        reference_frame,
        text="Background-field equality curve",
        variable=show_background_field,
    ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))
    ttk.Label(reference_frame, text="Background irradiance (W/cm^2)").grid(
        row=8, column=0, sticky="w", pady=(3, 0)
    )
    background_irradiance_entry = ttk.Entry(
        reference_frame,
        textvariable=background_irradiance_text,
        width=10,
    )
    background_irradiance_entry.grid(
        row=8, column=1, sticky="ew", padx=(6, 0), pady=(3, 0)
    )
    ttk.Checkbutton(
        reference_frame,
        text="Label intersection diameters",
        variable=label_intersections,
    ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 0))

    button_frame = ttk.Frame(controls)
    button_frame.pack(fill="x", pady=(2, 8))
    button_frame.columnconfigure((0, 1), weight=1)
    run_button = ttk.Button(button_frame, text="Calculate / Update")
    run_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
    data_button = ttk.Button(button_frame, text="Export CSV", state="disabled")
    data_button.grid(row=1, column=0, sticky="ew", padx=(0, 3))
    figure_button = ttk.Button(button_frame, text="Export Figure", state="disabled")
    figure_button.grid(row=1, column=1, sticky="ew", padx=(3, 0))
    intersection_button = ttk.Button(
        button_frame, text="Export Intersections", state="disabled"
    )
    intersection_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

    status_text = tk.StringVar(value="Ready to calculate illumination-normalized scattering.")
    status_label = ttk.Label(controls, textvariable=status_text, wraplength=325, justify="left")
    status_label.pack(fill="x", pady=(0, 8))
    ttk.Separator(controls).pack(fill="x", pady=(0, 8))
    ttk.Label(
        controls,
        text=(
            "P_sca/I0 = C_sca = Q_sca*pi(D/2)^2.\n"
            "Ideal illumination-normalized scattering over 4*pi; no collection model.\n"
            "Camera bands: SNR=1 NEP divided by illumination irradiance.\n"
            "Measured background is entered as electrons per particle ROI.\n"
            "Red reference: C_sca = P_bg/I0 over projected particle area.\n"
            "Preview/export share a fixed 4:3 layout and publication typography.\n"
            "Au/Ag/Cu/Ni: Johnson-Christy; Pt: Werner.\n"
            "TiO2: Jolivet anatase; ITO: Konig film (film-dependent)."
        ),
        wraplength=325,
        justify="left",
        foreground="#555555",
    ).pack(fill="x")

    preview_host = ttk.Frame(plot_frame)
    figure = Figure(
        figsize=EXPORT_FIGURE_SIZE,
        dpi=100,
        constrained_layout=True,
        facecolor="white",
    )
    axis = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, master=preview_host)
    toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")
    preview_host.pack(side="top", fill="both", expand=True)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.place(
        relx=0.5,
        rely=0.5,
        anchor="center",
        width=int(EXPORT_FIGURE_SIZE[0] * 100),
        height=int(EXPORT_FIGURE_SIZE[1] * 100),
    )

    def resize_wysiwyg_preview(event: object) -> None:
        """Fit a 4:3 export-sized figure while preserving relative typography."""
        target_width, target_height, preview_dpi = wysiwyg_preview_geometry(
            int(getattr(event, "width", 1)),
            int(getattr(event, "height", 1)),
        )
        figure.set_dpi(preview_dpi)
        figure.set_size_inches(*EXPORT_FIGURE_SIZE, forward=False)
        canvas_widget.place_configure(width=target_width, height=target_height)
        canvas.draw_idle()

    preview_host.bind("<Configure>", resize_wysiwyg_preview)

    state: dict[str, object] = {
        "series": None,
        "intersections": None,
        "reference_settings": None,
        "fresh": False,
    }

    def draw(
        series_list: list[Series],
        reference_settings: ReferenceSettings,
        intersections: list[Intersection],
    ) -> None:
        legend_location, legend_anchor = read_legend_settings()
        plot_series_on_axis(
            axis,
            series_list,
            log_x=log_x.get(),
            log_y=log_y.get(),
            reference_settings=reference_settings,
            intersections=intersections,
            legend_location=legend_location,
            legend_anchor=legend_anchor,
        )
        canvas.draw_idle()

    def mark_stale(*_args) -> None:
        if state["series"] is None:
            return
        state["fresh"] = False
        data_button.configure(state="disabled")
        figure_button.configure(state="disabled")
        intersection_button.configure(state="disabled")
        status_text.set("Parameters changed. The plot is stale; calculate again before exporting.")

    def sync_medium(*_args) -> None:
        custom_n_entry.configure(
            state="normal" if MEDIUM_LABELS[medium_choice.get()] == "custom" else "disabled"
        )

    def sync_reference_controls(*_args) -> None:
        measured_state = "readonly" if show_measured_background.get() else "disabled"
        measured_background_camera_combo.configure(state=measured_state)
        measured_background_entry.configure(
            state="normal" if show_measured_background.get() else "disabled"
        )
        background_irradiance_entry.configure(
            state="normal" if show_background_field.get() else "disabled"
        )

    def sync_legend_controls(*_args) -> None:
        custom_state = (
            "normal"
            if LEGEND_POSITIONS[legend_choice.get()] == "custom"
            else "disabled"
        )
        legend_x_entry.configure(state=custom_state)
        legend_y_entry.configure(state=custom_state)

    def read_legend_settings() -> tuple[str, tuple[float, float] | None]:
        if legend_choice.get() not in LEGEND_POSITIONS:
            raise ValueError("Select a valid legend position.")
        legend_location = LEGEND_POSITIONS[legend_choice.get()]
        if legend_location != "custom":
            return legend_location, None
        legend_x = float(legend_x_text.get())
        legend_y = float(legend_y_text.get())
        if not (
            math.isfinite(legend_x)
            and math.isfinite(legend_y)
            and 0.0 <= legend_x <= 1.0
            and 0.0 <= legend_y <= 1.0
        ):
            raise ValueError("Custom legend X and Y must both be between 0 and 1.")
        return legend_location, (legend_x, legend_y)

    def read_reference_settings() -> ReferenceSettings:
        camera_keys = []
        if show_orca_nep.get():
            camera_keys.append("orca_flash_v3")
        if show_prime_nep.get():
            camera_keys.append("prime_bsi")
        settings = ReferenceSettings(
            exposure_s=float(exposure_ms_text.get()) / 1000.0,
            roi_pixels=int(roi_pixels_text.get()),
            camera_keys=tuple(camera_keys),
            show_measured_background=show_measured_background.get(),
            measured_background_e_roi=float(measured_background_e_text.get()),
            measured_background_camera_key=CAMERA_LABEL_TO_KEY[measured_background_camera.get()],
            show_background_field=show_background_field.get(),
            background_irradiance_w_cm2=float(background_irradiance_text.get()),
            label_intersections=label_intersections.get(),
        )
        validate_reference_settings(settings)
        return settings

    def gather_and_compute() -> tuple[list[Series], ReferenceSettings, list[Intersection]]:
        materials = []
        if use_au.get():
            materials.append("Au")
        if use_pt.get():
            materials.append("Pt")
        if use_ni.get():
            materials.append("Ni")
        if use_tio2.get():
            materials.append("TiO2")
        if use_ito.get():
            materials.append("ITO")
        if use_ag.get():
            materials.append("Ag")
        if use_cu.get():
            materials.append("Cu")
        medium_key = MEDIUM_LABELS[medium_choice.get()]
        custom_medium_index = float(custom_n_text.get()) if medium_key == "custom" else 1.333
        incident_irradiance = float(incident_irradiance_text.get())
        reference_settings = read_reference_settings()
        series_list = compute_series(
            material_keys=materials,
            wavelengths_nm=parse_wavelengths(wavelength_text.get()),
            diameter_min_nm=float(diameter_min_text.get()),
            diameter_max_nm=float(diameter_max_text.get()),
            point_count=int(point_count_text.get()),
            log_diameter_grid=log_x.get(),
            medium_key=medium_key,
            custom_medium_n=custom_medium_index,
            alumina_model="o",
            incident_irradiance_w_cm2=incident_irradiance,
        )
        intersections = compute_reference_intersections(series_list, reference_settings)
        return series_list, reference_settings, intersections

    def calculate() -> None:
        run_button.configure(state="disabled")
        root.configure(cursor="watch")
        status_text.set("Calculating illumination-normalized scattering...")
        root.update_idletasks()
        try:
            series_list, reference_settings, intersections = gather_and_compute()
            draw(series_list, reference_settings, intersections)
            state["series"] = series_list
            state["reference_settings"] = reference_settings
            state["intersections"] = intersections
            legend_location, legend_anchor = read_legend_settings()
            state["plot_options"] = {
                "log_x": log_x.get(),
                "log_y": log_y.get(),
                "reference_settings": reference_settings,
                "intersections": intersections,
                "legend_location": legend_location,
                "legend_anchor": legend_anchor,
            }
            state["fresh"] = True
            data_button.configure(state="normal")
            figure_button.configure(state="normal")
            intersection_button.configure(state="normal")
            cmin = min(float(np.min(item.csca_um2)) for item in series_list)
            cmax = max(float(np.max(item.csca_um2)) for item in series_list)
            status_text.set(
                f"Done: {len(series_list)} curves; C_sca range {cmin:.3g}-{cmax:.3g} um^2; "
                f"{len(intersections)} first upward intersections."
            )
        except Exception as exc:
            state["fresh"] = False
            data_button.configure(state="disabled")
            figure_button.configure(state="disabled")
            intersection_button.configure(state="disabled")
            status_text.set(f"Calculation failed: {exc}")
            messagebox.showerror("Calculation failed", str(exc))
        finally:
            root.configure(cursor="")
            run_button.configure(state="normal")

    def export_csv() -> None:
        if not state["fresh"] or state["series"] is None:
            return
        series_list = state["series"]
        assert isinstance(series_list, list)
        output = filedialog.asksaveasfilename(
            title="Export illumination-normalized Mie-scattering data",
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")],
            initialfile="darkfield_mie_normalized_scattering.csv",
        )
        if not output:
            return
        diameter = series_list[0].diameter_nm
        incident_irradiance = series_list[0].incident_irradiance_w_cm2
        headers = ["diameter_nm", "incident_irradiance_W_cm2"]
        for item in series_list:
            suffix = f"{item.material_key}_{item.wavelength_nm:g}nm"
            headers.append(f"Csca_um2_{suffix}")
        with open(output, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for index, value in enumerate(diameter):
                row: list[str] = [f"{value:.12g}", f"{incident_irradiance:.12g}"]
                for item in series_list:
                    row.append(f"{item.csca_um2[index]:.12g}")
                writer.writerow(row)
        status_text.set(f"CSV saved: {output}")

    def export_intersections() -> None:
        if not state["fresh"] or state["intersections"] is None:
            return
        intersections = state["intersections"]
        reference_settings = state["reference_settings"]
        series_list = state["series"]
        assert isinstance(intersections, list)
        assert isinstance(reference_settings, ReferenceSettings)
        assert isinstance(series_list, list)
        output = filedialog.asksaveasfilename(
            title="Export reference-curve intersections",
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv")],
            initialfile="darkfield_mie_reference_intersections.csv",
        )
        if not output:
            return
        headers = [
            "reference_kind",
            "reference_label",
            "camera_model",
            "material",
            "wavelength_nm",
            "intersection_diameter_nm",
            "intersection_normalized_signal_um2",
            "incident_irradiance_W_cm2",
            "background_irradiance_W_cm2",
            "exposure_ms",
            "roi_pixels",
            "measured_background_e_roi_exposure",
        ]
        with open(output, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for crossing in intersections:
                camera_label = (
                    CAMERA_MODELS[crossing.camera_key]["short_label"]
                    if crossing.camera_key
                    else ""
                )
                writer.writerow(
                    [
                        crossing.reference_kind,
                        crossing.reference_label,
                        camera_label,
                        crossing.material_label,
                        f"{crossing.wavelength_nm:.12g}",
                        f"{crossing.diameter_nm:.12g}",
                        f"{crossing.normalized_signal_um2:.12g}",
                        f"{series_list[0].incident_irradiance_w_cm2:.12g}",
                        f"{reference_settings.background_irradiance_w_cm2:.12g}",
                        f"{reference_settings.exposure_s * 1000.0:.12g}",
                        str(reference_settings.roi_pixels),
                        f"{reference_settings.measured_background_e_roi:.12g}",
                    ]
                )
        status_text.set(f"Intersection CSV saved: {output}")

    def export_figure() -> None:
        if not state["fresh"] or state["series"] is None:
            return
        output = filedialog.asksaveasfilename(
            title="Export normalized dark-field Mie figure",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF vector figure", "*.pdf"),
                       ("SVG vector figure", "*.svg")],
            initialfile="darkfield_mie_normalized_references.png",
        )
        if not output:
            return
        series_list = state["series"]
        plot_options = state["plot_options"]
        assert isinstance(series_list, list)
        assert isinstance(plot_options, dict)
        export_canvas = Figure(
            figsize=EXPORT_FIGURE_SIZE,
            dpi=EXPORT_DPI,
            constrained_layout=True,
            facecolor="white",
        )
        export_axis = export_canvas.add_subplot(111)
        plot_series_on_axis(export_axis, series_list, **plot_options)
        export_canvas.savefig(
            output,
            dpi=EXPORT_DPI,
            facecolor="white",
        )
        status_text.set(f"Figure saved: {output}")

    run_button.configure(command=calculate)
    data_button.configure(command=export_csv)
    figure_button.configure(command=export_figure)
    intersection_button.configure(command=export_intersections)
    medium_combo.bind("<<ComboboxSelected>>", sync_medium)
    sync_medium()
    sync_reference_controls()
    sync_legend_controls()

    tracked_variables = (
        use_au,
        use_pt,
        use_ni,
        use_tio2,
        use_ito,
        use_ag,
        use_cu,
        wavelength_text,
        incident_irradiance_text,
        medium_choice,
        custom_n_text,
        diameter_min_text,
        diameter_max_text,
        point_count_text,
        log_x,
        log_y,
        legend_choice,
        legend_x_text,
        legend_y_text,
        exposure_ms_text,
        roi_pixels_text,
        show_orca_nep,
        show_prime_nep,
        show_measured_background,
        measured_background_e_text,
        measured_background_camera,
        show_background_field,
        background_irradiance_text,
        label_intersections,
    )
    for variable in tracked_variables:
        variable.trace_add("write", mark_stale)
    show_measured_background.trace_add("write", sync_reference_controls)
    show_background_field.trace_add("write", sync_reference_controls)
    legend_choice.trace_add("write", sync_legend_controls)

    root.after(120, calculate)
    root.mainloop()


def main() -> None:
    if "--selftest" in sys.argv:
        raise SystemExit(1 if run_selftest() else 0)
    launch_gui()


if __name__ == "__main__":
    main()
