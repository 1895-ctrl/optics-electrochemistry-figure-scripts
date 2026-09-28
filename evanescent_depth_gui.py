#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evanescent_depth_gui.py   --   v2.1
=========================================================================
TIRF evanescent-wave penetration depth  d(theta, lambda)
Dielectric / dielectric single interface.  Medium 2 = pure water.

WHAT IT DOES
    - d(theta1, lambda0) over a 2-D grid, with the dispersion of BOTH media.
    - Two depth conventions (they differ by exactly a factor 2):
        intensity  1/e :  d_I = lam0 / (4 pi Q)      <- TIRF convention (Axelrod)
        amplitude  1/e :  d_E = lam0 / (2 pi Q) = 2 d_I   <- ATR convention (Harrick)
      with  Q = sqrt(n1^2 sin^2(theta1) - n2^2),  lam0 = VACUUM wavelength.
      d is polarization-INDEPENDENT.  Only the interface intensity I(0) is not.
    - Grazing-incidence floor  d_floor = lam0 / (4 pi sqrt(n1^2 - n2^2)):
      the smallest depth a given substrate can EVER give, at any angle.
      This is usually the binding constraint, not the angle.
    - Angle entry in three ways:  direct theta1  |  objective NA  |  prism geometry.
    - Far-field ("non-evanescent") contamination diagnostic, after
      Mattheyses & Axelrod, J. Biomed. Opt. 11, 014006 (2006).

WHAT IT DOES *NOT* DO  (model boundaries -- read before trusting a number)
    - No absorbing media (real n only).  No metal film, no multilayer:
      this is NOT valid for SPR / SPRM / SEIRAS geometries.
    - Strict plane wave, semi-infinite media.  A real focused/finite beam has an
      angular spread; within ~1 deg of theta_c the single-exponential picture and
      d(theta) itself become unreliable.
    - No Goos-Haenchen shift, no surface roughness, no fluorophore-surface
      photophysics (quenching, near-field dipole coupling, SAF emission).

PROVENANCE TAGS   [LIT] literature   [DER] derived from a spec sheet   [SPEC] product spec
Every optical constant below carries its source.

CLI
    python evanescent_depth_gui.py                 # GUI
    python evanescent_depth_gui.py --selftest      # run self-tests only
    python evanescent_depth_gui.py --preview f.png # headless figure
=========================================================================
"""

from __future__ import annotations

import sys
import os
import csv
import json
import math
import datetime


# On Windows, running a Conda interpreter directly (as some PyCharm run
# configurations do) may omit the DLL directories normally injected by
# ``conda activate``.  NumPy can then terminate the process inside its BLAS
# startup with Windows exception 0xc06d007f, before Python can raise a normal
# ImportError.  Bootstrap the current interpreter's own Conda directories
# before importing NumPy.  This is a no-op for non-Conda Python installations.
_CONDA_DLL_HANDLES = []


def _bootstrap_windows_conda_dlls():
    if sys.platform != "win32":
        return
    prefix = os.path.realpath(sys.prefix)
    if not os.path.isdir(os.path.join(prefix, "conda-meta")):
        return

    candidates = [
        prefix,
        os.path.join(prefix, "Library", "mingw-w64", "bin"),
        os.path.join(prefix, "Library", "usr", "bin"),
        os.path.join(prefix, "Library", "bin"),
        os.path.join(prefix, "Scripts"),
        os.path.join(prefix, "bin"),
    ]
    candidates = [path for path in candidates if os.path.isdir(path)]

    current = [path for path in os.environ.get("PATH", "").split(os.pathsep) if path]
    known = {os.path.normcase(os.path.normpath(path)) for path in current}
    prepend = [path for path in candidates
               if os.path.normcase(os.path.normpath(path)) not in known]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + current)

    if hasattr(os, "add_dll_directory"):
        for path in candidates:
            try:
                # Keep each handle alive for the lifetime of the process.
                _CONDA_DLL_HANDLES.append(os.add_dll_directory(path))
            except OSError:
                pass


_bootstrap_windows_conda_dlls()

import numpy as np
import matplotlib

if "--preview" in sys.argv or "--selftest" in sys.argv:
    matplotlib.use("Agg")
else:                                                   # pragma: no cover
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.figure import Figure                                     # noqa: E402

# Okabe-Ito colourblind-safe palette (Okabe & Ito 2008)
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
         "#E69F00", "#56B4E9", "#F0E442", "#000000"]
matplotlib.rcParams.update({
    "axes.unicode_minus": False,        # ASCII minus -> no tofu boxes on CJK Windows
    "font.size": 9,
    "axes.prop_cycle": matplotlib.cycler(color=OKABE),
    "figure.dpi": 110,
    "savefig.dpi": 300,
})

VERSION = "2.1"

# =========================================================================
#  1.  DISPERSION MODELS  --  MEDIUM 1 (the dense side)
# =========================================================================
# Generalised Sellmeier:   n^2 = A0 + sum_i  B_i lam^2 / (lam^2 - C_i),  lam in um
#   A0 = 1 for the standard 3-term form.

_SELLMEIER = {
    # ---- SCHOTT optical glasses. Coefficients from the SCHOTT optical glass
    #      data sheets / catalogue (Sellmeier 1, lam in um).           [LIT]
    "N-BK7 (Schott)": dict(
        A0=1.0,
        BC=[(1.03961212, 0.00600069867),
            (0.231792344, 0.0200179144),
            (1.01046945, 103.560653)],
        nd=1.5168, tag="LIT", rng=(0.30, 2.50),
        ref="SCHOTT optical glass data sheet (N-BK7)"),

    "F2 (Schott)": dict(
        A0=1.0,
        BC=[(1.34533359, 0.00997743871),
            (0.209073176, 0.0470450767),
            (0.937357162, 111.886764)],
        nd=1.6200, tag="LIT", rng=(0.32, 2.50),
        ref="SCHOTT optical glass data sheet (F2)"),

    "N-SF10 (Schott)": dict(
        A0=1.0,
        BC=[(1.62153902, 0.0122241457),
            (0.256287842, 0.0595736775),
            (1.64447552, 147.468793)],
        nd=1.7283, tag="LIT", rng=(0.38, 2.50),
        ref="SCHOTT optical glass data sheet (N-SF10)"),

    "N-SF11 (Schott)": dict(
        A0=1.0,
        BC=[(1.73759695, 0.013188707),
            (0.313747346, 0.0623068142),
            (1.89878101, 155.23629)],
        nd=1.7847, tag="LIT", rng=(0.37, 2.50),
        ref="SCHOTT optical glass data sheet (N-SF11)"),

    "N-LASF9 (Schott)": dict(
        A0=1.0,
        BC=[(2.00029547, 0.0121426017),
            (0.298926886, 0.0538736236),
            (1.80691843, 156.530829)],
        nd=1.8503, tag="LIT", rng=(0.37, 2.50),
        ref="SCHOTT optical glass data sheet (N-LASF9)"),

    # ---- Fused silica.  The "quartz slide" of prism-TIRF / smFRET is this. [LIT]
    "Fused silica (Malitson 1965)": dict(
        A0=1.0,
        BC=[(0.6961663, 0.0684043 ** 2),
            (0.4079426, 0.1162414 ** 2),
            (0.8974794, 9.896161 ** 2)],
        nd=1.4585, tag="LIT",
        rng=(0.21, 3.71),
        ref="I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965). "
            "DOI 10.1364/JOSA.55.001205"),

    # ---- Crystalline quartz, ordinary ray.                              [LIT]
    "Crystalline quartz, o-ray (Ghosh 1999)": dict(
        A0=1.28604141,
        BC=[(1.07044083, 1.00585997e-2),
            (1.10202242, 100.0)],
        nd=1.5442, tag="LIT",
        rng=(0.198, 2.05),
        ref="G. Ghosh, Opt. Commun. 163, 95 (1999). "
            "DOI 10.1016/S0030-4018(99)00091-7"),

    # ---- Sapphire (uniaxial: give both rays; c-cut windows use the o-ray). [LIT]
    "Sapphire, o-ray (Malitson 1962)": dict(
        A0=1.0,
        BC=[(1.4313493, 0.0726631 ** 2),
            (0.65054713, 0.1193242 ** 2),
            (5.3414021, 18.028251 ** 2)],
        nd=1.7682, tag="LIT",
        rng=(0.20, 5.00),
        ref="I. H. Malitson, J. Opt. Soc. Am. 52, 1377 (1962). "
            "DOI 10.1364/JOSA.52.001377"),

    "Sapphire, e-ray (Malitson 1962)": dict(
        A0=1.0,
        BC=[(1.5039759, 0.0740288 ** 2),
            (0.55069141, 0.1216529 ** 2),
            (6.5927379, 20.072248 ** 2)],
        nd=1.7601, tag="LIT",
        rng=(0.20, 5.00),
        ref="I. H. Malitson, J. Opt. Soc. Am. 52, 1377 (1962). "
            "DOI 10.1364/JOSA.52.001377"),

    # ---- Polystyrene: TIRF calibration beads, some polymer-bottom dishes. [LIT]
    "Polystyrene (Sultanova 2009)": dict(
        A0=1.0,
        BC=[(1.4435, 0.020216)],
        nd=1.5916, tag="LIT",
        rng=(0.4358, 1.052),
        ref="N. Sultanova, S. Kasarova, I. Nikolov, Acta Phys. Pol. A 116, 585 "
            "(2009). DOI 10.12693/APhysPolA.116.585"),
}


def _cauchy_from_abbe(n_ref, lam_ref_um, nu, lam_F_um, lam_C_um):
    """
    Two-term Cauchy  n(lam) = A + B/lam^2  reconstructed from a data-sheet pair
    (index at a reference line, Abbe number).  This is a DERIVED model:
    it reproduces the two catalogue numbers exactly and interpolates smoothly
    between them, but it is not a measured dispersion curve.  Accuracy over
    400-800 nm is ~ +-0.002, i.e. comparable to the manufacturing tolerance
    on n itself (+-0.0015 for D 263).
    """
    dn = (n_ref - 1.0) / nu                       # = n_F - n_C  (or n_F' - n_C')
    B = dn / (1.0 / lam_F_um ** 2 - 1.0 / lam_C_um ** 2)
    A = n_ref - B / lam_ref_um ** 2
    return A, B


# Spectral lines (nm)
LINE_e, LINE_Fp, LINE_Cp = 546.07, 479.99, 643.85      # Hg e, Cd F', Cd C'
LINE_d, LINE_F, LINE_C = 587.56, 486.13, 656.27        # He d, H F, H C

_CAUCHY = {
    # ---- Standard microscope cover glass.  SCHOTT D 263 M data sheet:
    #      n_e = 1.5255 +- 0.0015, nu_e = 55; complies with ISO 8255-1.  [DER]
    "Cover glass, D 263 M / ISO 8255-1 (Schott)": dict(
        coef=_cauchy_from_abbe(1.5255, LINE_e / 1000.0, 55.0,
                               LINE_Fp / 1000.0, LINE_Cp / 1000.0),
        nd=1.5232, tag="DER",
        rng=(0.40, 0.80),
        ref="SCHOTT D 263 M data sheet: n_e = 1.5255 +- 0.0015, nu_e = 55 "
            "(ISO 8255-1). Cauchy A + B/lam^2 derived from (n_e, nu_e)."),

    # ---- Immersion oil.  Cargille Type A: n_D = 1.5150, nu_D = 43
    #      (-> n_e = 1.518, the value engraved on TIRF objectives).        [DER]
    "Immersion oil, n_d = 1.515 (Cargille Type A)": dict(
        coef=_cauchy_from_abbe(1.5150, LINE_d / 1000.0, 43.0,
                               LINE_F / 1000.0, LINE_C / 1000.0),
        nd=1.5150, tag="DER",
        rng=(0.40, 0.80),
        ref="Cargille Type A immersion oil: n_D = 1.5150, nu_D = 43. "
            "Cauchy A + B/lam^2 derived from (n_D, nu_D). "
            "NOT a TIR substrate - listed to check the NA chain."),
}

_CONST_MEDIA = {
    # ---- High-index cover glass required by NA 1.65 / 1.7 TIRF objectives.
    #      Olympus: cover glass n_d = 1.788, immersion liquid n_d = 1.78
    #      (Cargille, diiodomethane-based; volatile).  Dispersion not
    #      published -> modelled as CONSTANT n.  Wavelength scans with this
    #      entry ignore dispersion.                                       [SPEC]
    "High-index cover glass, n_d = 1.788 (NA 1.65/1.7 obj.)":
        dict(n=1.788, tag="SPEC", rng=(0.40, 0.80),
             ref="Olympus/Evident TIRF application note: high-index cover glass "
                 "n_d = 1.788 + immersion liquid n_d = 1.78 (Cargille). "
                 "CONSTANT n - no dispersion model available."),
}

MEDIUM1_LIB = (list(_SELLMEIER.keys()) + list(_CAUCHY.keys())
               + list(_CONST_MEDIA.keys()) + ["Custom (constant n)"])

# Substrates worth comparing head-to-head in the "which glass?" panel
COMPARE_SET = [
    "Fused silica (Malitson 1965)",
    "Cover glass, D 263 M / ISO 8255-1 (Schott)",
    "N-BK7 (Schott)",
    "Polystyrene (Sultanova 2009)",
    "F2 (Schott)",
    "N-SF10 (Schott)",
    "N-SF11 (Schott)",
    "Sapphire, o-ray (Malitson 1962)",
    "N-LASF9 (Schott)",
]


def medium1_n(name, lam_nm, const_n=1.515):
    """Refractive index of medium 1. lam_nm: scalar or array (vacuum, nm)."""
    lam = np.atleast_1d(np.asarray(lam_nm, dtype=float)) / 1000.0      # -> um
    if name in _SELLMEIER:
        e = _SELLMEIER[name]
        n2 = np.full_like(lam, e["A0"])
        for B, C in e["BC"]:
            n2 = n2 + B * lam ** 2 / (lam ** 2 - C)
        out = np.sqrt(n2)
    elif name in _CAUCHY:
        A, B = _CAUCHY[name]["coef"]
        out = A + B / lam ** 2
    elif name in _CONST_MEDIA:
        out = np.full_like(lam, _CONST_MEDIA[name]["n"])
    elif name == "Custom (constant n)":
        out = np.full_like(lam, float(const_n))
    else:
        raise KeyError(f"unknown medium 1: {name!r}")
    return out if np.ndim(lam_nm) else float(out[0])


def medium1_ref(name):
    for lib in (_SELLMEIER, _CAUCHY, _CONST_MEDIA):
        if name in lib:
            return f"[{lib[name]['tag']}] {lib[name]['ref']}"
    return "[ASSUMED] user-supplied constant index"


def medium1_range(name):
    """Validity range of the dispersion model, in nm.  None = user constant."""
    for lib in (_SELLMEIER, _CAUCHY, _CONST_MEDIA):
        if name in lib:
            lo, hi = lib[name]["rng"]
            return 1000.0 * lo, 1000.0 * hi
    return None


def medium1_tag(name):
    for lib in (_SELLMEIER, _CAUCHY, _CONST_MEDIA):
        if name in lib:
            return lib[name]["tag"]
    return "ASSUMED"


# Validity of the medium-2 (water) model:
#   IAPWS R9-97 : 200-1100 nm, -12..500 C, rho <= 1060 kg/m^3
#   Tanaka 2001 : 0-40 C  (this is the tighter of the two -> it sets the limit)
WATER_LAM_RANGE = (200.0, 1100.0)      # nm
WATER_T_RANGE = (0.0, 40.0)            # deg C


# =========================================================================
#  2.  DISPERSION MODEL  --  MEDIUM 2 (pure water)
# =========================================================================
# IAPWS R9-97 formulation for the refractive index of ordinary water:
#   A. H. Harvey, J. S. Gallagher, J. M. H. Levelt Sengers,
#   J. Phys. Chem. Ref. Data 27, 761 (1998).  DOI 10.1063/1.556029       [LIT]
# Density from:
#   G. S. Kell, J. Chem. Eng. Data 20, 97 (1975). DOI 10.1021/je60064a005 [LIT]

_IAPWS = dict(a0=0.244257733, a1=9.74634476e-3, a2=-3.73234996e-3,
              a3=2.68678472e-4, a4=1.58920570e-3, a5=2.45934259e-3,
              a6=0.900704920, a7=-1.66626219e-2,
              LUV=0.2292020, LIR=5.432937)
_RHO_STAR, _T_STAR, _LAM_STAR = 1000.0, 273.15, 0.589   # kg/m^3, K, um

WATER_REF = ("[LIT] IAPWS R9-97 / A. H. Harvey, J. S. Gallagher, "
             "J. M. H. Levelt Sengers, J. Phys. Chem. Ref. Data 27, 761 (1998), "
             "DOI 10.1063/1.556029; density: M. Tanaka et al., Metrologia 38, 301 "
             "(2001), DOI 10.1088/0026-1394/38/4/3 (CIPM-recommended, 0-40 C)")

# Tanaka, Girard, Davis, Peuto & Bignell, Metrologia 38, 301 (2001):
#   rho(t) = a5 * [ 1 - (t + a1)^2 (t + a2) / (a3 (t + a4)) ]      0-40 C, SMOW
# CIPM-recommended; reproduces IAPWS-95 to better than 0.001 kg/m^3 in 0-40 C.
# (Kell 1975 is ~0.003 kg/m^3 lower -- kept below as an independent cross-check.)
_TANAKA = (-3.983035, 301.797, 522528.9, 69.34881, 999.974950)


def water_density(T_C):
    """Tanaka et al. (2001), 1 atm, 0-40 C (SMOW), kg/m^3."""
    t = np.asarray(T_C, dtype=float)
    a1, a2, a3, a4, a5 = _TANAKA
    return a5 * (1.0 - (t + a1) ** 2 * (t + a2) / (a3 * (t + a4)))


def water_density_kell(T_C):
    """Kell (1975), 1 atm, 0-150 C, kg/m^3.  Independent cross-check only."""
    t = np.asarray(T_C, dtype=float)
    num = (999.83952 + 16.945176 * t - 7.9870401e-3 * t ** 2
           - 46.170461e-6 * t ** 3 + 105.56302e-9 * t ** 4
           - 280.54253e-12 * t ** 5)
    return num / (1.0 + 16.879850e-3 * t)


def water_n(lam_nm, T_C=25.0, dn=0.0):
    """IAPWS R9-97 refractive index of pure water. dn: additive offset."""
    lam = np.atleast_1d(np.asarray(lam_nm, dtype=float)) / 1000.0
    rho = water_density(T_C) / _RHO_STAR
    T = (np.asarray(T_C, dtype=float) + 273.15) / _T_STAR
    L = lam / _LAM_STAR
    c = _IAPWS
    f = (c["a0"] + c["a1"] * rho + c["a2"] * T + c["a3"] * L ** 2 * T
         + c["a4"] / L ** 2 + c["a5"] / (L ** 2 - c["LUV"] ** 2)
         + c["a6"] / (L ** 2 - c["LIR"] ** 2) + c["a7"] * rho ** 2)
    fr = f * rho
    n = np.sqrt((1.0 + 2.0 * fr) / (1.0 - fr)) + dn
    return n if np.ndim(lam_nm) else float(n[0])


MAX_INDEX_SCAN_CURVES = 12


def parse_index_values(value, label="refractive-index values"):
    """Parse a scalar, sequence, or comma-separated GUI value into real indices."""
    if isinstance(value, str):
        text = value.strip().replace("\uFF0C", ",").replace(";", ",")
        if not text:
            raise ValueError(f"{label} cannot be empty")
        tokens = [item.strip() for item in text.split(",")]
        if any(not item for item in tokens):
            raise ValueError(f"{label} contains an empty item")
    elif np.isscalar(value):
        tokens = [value]
    else:
        tokens = list(value)
        if not tokens:
            raise ValueError(f"{label} cannot be empty")

    try:
        values = tuple(float(item) for item in tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only numbers separated by commas") from exc
    if not all(np.isfinite(item) and item > 0.0 for item in values):
        raise ValueError(f"{label} must contain finite positive real numbers")
    return values


def resolve_index_pairs(n1_values, n2_values, max_curves=MAX_INDEX_SCAN_CURVES):
    """
    Resolve fixed-wavelength scan inputs into equal-length (n1, n2) arrays.

    One value on either side is broadcast across the other side.  If both sides
    contain multiple values, they are paired element by element; a Cartesian
    product is deliberately not generated because it quickly makes the plot and
    legend unreadable.
    """
    n1 = np.asarray(parse_index_values(n1_values, "n1 values"), dtype=float)
    n2 = np.asarray(parse_index_values(n2_values, "n2 values"), dtype=float)

    if n1.size == n2.size:
        mode = "single pair" if n1.size == 1 else "paired n1/n2"
    elif n2.size == 1:
        n2 = np.full(n1.shape, float(n2[0]))
        mode = "vary n1 (fixed n2)"
    elif n1.size == 1:
        n1 = np.full(n2.shape, float(n1[0]))
        mode = "vary n2 (fixed n1)"
    else:
        raise ValueError("n1/n2 lists must have equal lengths, or one list must contain one value")

    if n1.size > int(max_curves):
        raise ValueError(f"refractive-index scan allows at most {int(max_curves)} curves")
    invalid = np.where(n1 <= n2)[0]
    if invalid.size:
        pairs = ", ".join(str(int(i) + 1) for i in invalid[:5])
        suffix = "..." if invalid.size > 5 else ""
        raise ValueError(f"TIR requires n1 > n2 for every pair; invalid pair(s): {pairs}{suffix}")
    return n1, n2, mode


# =========================================================================
#  3.  CORE PHYSICS
# =========================================================================

def critical_angle_deg(n1, n2):
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arcsin(np.where(n2 < n1, n2 / n1, np.nan)))


def _Q(n1, n2, th_deg):
    """Q = sqrt(n1^2 sin^2 th - n2^2).  NaN below the critical angle."""
    s = np.sin(np.radians(np.asarray(th_deg, float)))
    a = np.asarray(n1, float) ** 2 * s ** 2 - np.asarray(n2, float) ** 2
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.where(a > 0.0, a, np.nan))


def penetration_depth_nm(lam0_nm, n1, n2, th_deg, convention="intensity"):
    """
    convention = "intensity" : d_I = lam0 / (4 pi Q)   [Axelrod / TIRF]
    convention = "amplitude" : d_E = lam0 / (2 pi Q)   [Harrick / ATR] = 2 d_I
    lam0 = VACUUM wavelength.  Returns NaN below theta_c.
    """
    k = 4.0 if convention == "intensity" else 2.0
    return np.asarray(lam0_nm, float) / (k * np.pi * _Q(n1, n2, th_deg))


def depth_floor_nm(lam0_nm, n1, n2, convention="intensity"):
    """
    Grazing-incidence limit  theta1 -> 90 deg :
        d_floor = lam0 / (k pi sqrt(n1^2 - n2^2)).
    The SMALLEST depth a given substrate/medium pair can ever produce.
    No incidence angle, and no objective, can beat this.
    """
    k = 4.0 if convention == "intensity" else 2.0
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    with np.errstate(invalid="ignore"):
        r = np.sqrt(np.where(n1 > n2, n1 ** 2 - n2 ** 2, np.nan))
    return np.asarray(lam0_nm, float) / (k * np.pi * r)


def kz2_complex(lam0_nm, n1, n2, th_deg):
    """Independent route: kz2 = (2 pi / lam0) sqrt(n2^2 - n1^2 sin^2 th)."""
    s = np.sin(np.radians(np.asarray(th_deg, float)))
    arg = np.asarray(n2, float) ** 2 - np.asarray(n1, float) ** 2 * s ** 2
    return (2.0 * np.pi / np.asarray(lam0_nm, float)) * np.sqrt(arg.astype(complex))


def fresnel_tir(n1, n2, th_deg):
    """Amplitude reflection coefficients (complex) for s and p."""
    th = np.radians(np.asarray(th_deg, float))
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    c1 = np.cos(th)
    c2 = np.sqrt((1.0 - (n1 / n2) ** 2 * np.sin(th) ** 2).astype(complex))
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    return rs, rp


def interface_intensity(n1, n2, th_deg):
    """
    |E(0)|^2 / |E_inc|^2 at z = 0+.  D. Axelrod, Traffic 2, 764 (2001);
    Methods Enzymol. 361, 1 (2003).  Derivation (nb = n2/n1, q^2 = s2 - nb^2):

        t_s = 2 c1 / (c1 + i q)          -> I_s = |t_s|^2 = 4 c1^2 / (1 - nb^2)
        t_p = 2 nb c1 / (nb^2 c1 + i q)  -> |t_p|^2 = 4 nb^2 c1^2 / D,
                                            D = nb^4 c1^2 + s2 - nb^2
        E_2x = t_p cos(th2),  E_2z = t_p sin(th2),
        |cos th2|^2 = q^2/nb^2,  |sin th2|^2 = s2/nb^2
        -> I_x = 4 c1^2 (s2 - nb^2) / D      (the nb^2 cancels: NO 1/(1-nb^2) here)
           I_z = 4 c1^2 s2 / D

    Returns (I_s, I_p, I_x, I_z) with I_p = I_x + I_z.
    Analytic limits used in the self-test:
        th -> th_c+ :  I_s -> 4,  I_p -> 4 / nb^2
        th -> 90    :  I_s, I_p -> 0
    """
    th = np.radians(np.asarray(th_deg, float))
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    nb = n2 / n1
    s2, c2 = np.sin(th) ** 2, np.cos(th) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        Is = 4.0 * c2 / (1.0 - nb ** 2)
        D = nb ** 4 * c2 + s2 - nb ** 2
        Ix = 4.0 * c2 * (s2 - nb ** 2) / D
        Iz = 4.0 * c2 * s2 / D
        bad = s2 <= nb ** 2
        Is, Ix, Iz = (np.where(bad, np.nan, v) for v in (Is, Ix, Iz))
    return Is, Ix + Iz, Ix, Iz


# ---- angle entry: objective-type ----------------------------------------

def NA_eff(n1, th_deg):
    """
    NA_eff = n1 sin(theta1).  THE quantity that unifies both TIRF geometries.

    Because n sin(theta) is conserved through the (index-matched) front lens /
    oil / cover-glass chain, the objective's NA IS the n1 sin(theta1) delivered
    at the TIR interface.  Hence  Q = sqrt(n1^2 sin^2 th - n2^2) = sqrt(NA^2 - n2^2)
    and, for objective-type TIRF,

        d_I = lam0 / (4 pi sqrt(NA^2 - n2^2))          <- n1 CANCELS OUT.

    So a higher-index cover glass does NOT buy you a thinner evanescent layer at
    fixed NA; it only raises the ceiling on the NA you are allowed to have
    (NA < n_coverglass AND NA < n_immersion).  A prism, by contrast, lets you
    push theta1 to grazing, so there the substrate index IS the binding limit:
    NA_eff -> n1 as theta1 -> 90 deg.
    """
    return np.asarray(n1, float) * np.sin(np.radians(np.asarray(th_deg, float)))


def theta_from_NA(NA, n1):
    """theta1 = arcsin(NA / n1).  NA = n sin(theta) is conserved through the
    (index-matched) oil / cover-glass chain, so n1 here is the COVER GLASS."""
    NA, n1 = np.asarray(NA, float), np.asarray(n1, float)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arcsin(np.where(NA < n1, NA / n1, np.nan)))


def supercritical_annulus_fraction(NA, n2):
    """
    Fraction of the back-focal-plane AREA that lies beyond the critical angle,
    (NA^2 - n2^2) / NA^2.  This is the usable illumination annulus in
    objective-type TIRF -- and the reason NA 1.40 is a poor TIRF lens.
    """
    NA, n2 = np.asarray(NA, float), np.asarray(n2, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(NA > n2, (NA ** 2 - n2 ** 2) / NA ** 2, np.nan)


# ---- angle entry: prism-type --------------------------------------------
# Geometry:  A = angle between the ENTRANCE-FACE normal and the TIR-SURFACE
# normal.  Equilateral prism sitting on its base: A = 60 deg.  Right-angle
# prism coupled through its hypotenuse: A = 45 deg.
#   phi  = signed external incidence angle on the entrance face
#   phi' = arcsin(sin phi / n1)                     (Snell at the entrance face)
#   theta1 = A - phi'                               (angle at the TIR surface)
# The prism must be index-matched (oil/glycerol) to the slide, otherwise there
# is an extra refraction that this formula does not contain.

def prism_theta1_deg(A_deg, phi_ext_deg, n1):
    phi = np.radians(np.asarray(phi_ext_deg, float))
    with np.errstate(invalid="ignore"):
        phip = np.arcsin(np.sin(phi) / np.asarray(n1, float))
    return np.asarray(A_deg, float) - np.degrees(phip)


def prism_phi_ext_deg(A_deg, theta1_deg, n1):
    """Inverse: external angle needed to hit a target theta1. NaN if impossible."""
    x = np.asarray(n1, float) * np.sin(np.radians(np.asarray(A_deg, float)
                                                  - np.asarray(theta1_deg, float)))
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arcsin(np.where(np.abs(x) <= 1.0, x, np.nan)))


def prism_dtheta_dphi(A_deg, phi_ext_deg, n1):
    """d(theta1)/d(phi_ext): how much internal angle you get per external degree."""
    phi = np.radians(np.asarray(phi_ext_deg, float))
    n1 = np.asarray(n1, float)
    with np.errstate(invalid="ignore"):
        phip = np.arcsin(np.sin(phi) / n1)
        return -np.cos(phi) / (n1 * np.cos(phip))


# ---- far-field ("non-evanescent") contamination --------------------------
# Mattheyses & Axelrod, J. Biomed. Opt. 11, 014006 (2006), DOI 10.1117/1.2161018:
# with a 1.45 or 1.65 NA objective the measured axial profile fits a DOUBLE
# exponential; at the coverslip/sample interface ~90 % of the field is the true
# evanescent wave and the remaining ~10 % is a much more slowly decaying
# component, identified as scattering in the illumination optics.
# See also Brunstein, Teremetz & Oheim, Biophys. J. 106, 1020 (2014) (Part I,
# sources of non-evanescent excitation) and 106, 1044 (2014) (Part II).
#
#   I(z) = (1-f) exp(-z/d) + f exp(-z/d_stray),   I(0) = 1
#
#   f       ~ 0.10   [LIT]      objective-type; set f = 0 for prism-type
#   d_stray = 1000 nm [ASSUMED] the long decay constant is geometry- and
#             sample-dependent (it is NOT a universal number); 1000 nm is only
#             an illustrative value. Calibrate it for the experimental setup,
#             for example using the index-matched fluorescent-bead method of
#             Mattheyses & Axelrod,
#             or a step-height calibration slide.

def far_field_metrics(d_nm, f, d_stray_nm):
    """
    Returns dict with
      z_1e      : depth where the COMPOSITE profile falls to 1/e  (apparent depth)
      z_cross   : depth beyond which the stray component dominates
      E_stray   : fraction of the TOTAL axial excitation integral that is stray
    """
    d, ds = float(d_nm), float(d_stray_nm)
    f = float(np.clip(f, 0.0, 0.999))
    if f <= 0.0:
        return dict(z_1e=d, z_cross=np.inf, E_stray=0.0)

    def I(z):
        return (1.0 - f) * math.exp(-z / d) + f * math.exp(-z / ds)

    target = 1.0 / math.e
    lo, hi = 0.0, max(10.0 * ds, 10.0 * d)
    for _ in range(200):                                   # bisection
        mid = 0.5 * (lo + hi)
        if I(mid) > target:
            lo = mid
        else:
            hi = mid
    z_1e = 0.5 * (lo + hi)

    if ds > d:
        z_cross = math.log((1.0 - f) / f) / (1.0 / d - 1.0 / ds)
    else:
        z_cross = np.inf

    E_ev, E_st = (1.0 - f) * d, f * ds
    return dict(z_1e=z_1e, z_cross=z_cross, E_stray=E_st / (E_ev + E_st))


# =========================================================================
#  4.  SELF-TESTS   (must pass before any number is believed)
# =========================================================================

def run_selftests(verbose=True):
    R = []

    def chk(name, cond):
        R.append((bool(cond), name, ""))

    def close(name, a, b, tol):
        ok = np.isfinite(a) and abs(float(a) - float(b)) <= tol
        R.append((bool(ok), name, f"got {float(a):.6f}, want {float(b):.6f} +-{tol:g}"))

    def rejects_value_error(func):
        try:
            func()
        except ValueError:
            return True
        return False

    # ---- water: official IAPWS R9-97 Table 3 check values (0 C, 1 atm) ----
    close("IAPWS water n(226.50 nm, 0 C)", water_n(226.50, 0.0), 1.394527, 2e-6)
    close("IAPWS water n(589.00 nm, 0 C)", water_n(589.00, 0.0), 1.334344, 2e-6)
    close("IAPWS water n(1013.98 nm, 0 C)", water_n(1013.98, 0.0), 1.326135, 2e-6)
    # density: Tanaka 2001 vs the standard tabulated (IAPWS-95) values
    close("Tanaka rho(20 C) = 998.2071", water_density(20.0), 998.2071, 1e-3)
    close("Tanaka rho(25 C) = 997.0479", water_density(25.0), 997.0479, 1e-3)
    close("Tanaka rho_max = 999.9750", water_density(-_TANAKA[0]), 999.9750, 1e-3)
    t = np.linspace(0, 10, 10001)
    chk("density maximum at 3.98 C", abs(t[np.argmax(water_density(t))] - 3.983) < 0.01)
    # independent correlation: Kell 1975 must agree with Tanaka to ~0.005 kg/m^3
    tt = np.linspace(0.0, 40.0, 401)
    chk("Tanaka vs Kell agree to < 0.006 kg/m3 over 0-40 C",
        np.max(np.abs(water_density(tt) - water_density_kell(tt))) < 0.006)

    # ---- medium-1 library: every Sellmeier entry against its catalogue n_d ----
    for nm, e in _SELLMEIER.items():
        lam = 589.3 if ("Sapphire" in nm or "quartz, o-ray" in nm) else LINE_d
        close(f"n_d  {nm}", medium1_n(nm, lam), e["nd"], 6e-4)

    # ---- derived Cauchy models must reproduce the data-sheet numbers ----
    CG = "Cover glass, D 263 M / ISO 8255-1 (Schott)"
    OIL = "Immersion oil, n_d = 1.515 (Cargille Type A)"
    close("D 263: n_e = 1.5255", medium1_n(CG, LINE_e), 1.5255, 1e-6)
    nFp, nCp = medium1_n(CG, LINE_Fp), medium1_n(CG, LINE_Cp)
    close("D 263: nu_e = 55", (1.5255 - 1.0) / (nFp - nCp), 55.0, 1e-6)
    close("D 263: n_d ~ 1.523", medium1_n(CG, LINE_d), 1.5232, 1e-3)
    close("Oil: n_d = 1.515", medium1_n(OIL, LINE_d), 1.5150, 1e-6)
    close("Oil: n_e ~ 1.518 (engraved value)", medium1_n(OIL, LINE_e), 1.518, 5e-4)

    # ---- every library entry must carry a usable validity range ----
    for nm in MEDIUM1_LIB:
        if nm == "Custom (constant n)":
            continue
        r = medium1_range(nm)
        chk(f"validity range present & sane  {nm}",
            r is not None and 100.0 <= r[0] < r[1] <= 6000.0)
    chk("d-line lies inside every model's validity range",
        all(medium1_range(nm)[0] <= LINE_d <= medium1_range(nm)[1]
            for nm in MEDIUM1_LIB if nm != "Custom (constant n)"))
    chk("water model ranges are sane",
        WATER_LAM_RANGE[0] < WATER_LAM_RANGE[1] and WATER_T_RANGE[0] < WATER_T_RANGE[1])

    # ---- geometry / core physics ----
    close("theta_c(1.5, 1.0)", critical_angle_deg(1.5, 1.0), 41.8103, 1e-3)
    chk("no TIR when n2 > n1 -> NaN", np.isnan(critical_angle_deg(1.33, 1.52)))

    lam, n1, n2, th = 488.0, 1.5224, 1.3369, 68.0
    dI = penetration_depth_nm(lam, n1, n2, th)
    kz = kz2_complex(lam, n1, n2, th)
    close("d_I: closed form == complex-kz route", dI, 1.0 / (2.0 * abs(kz.imag)), 1e-9)
    close("d_amplitude == 2 * d_intensity",
          penetration_depth_nm(lam, n1, n2, th, "amplitude"), 2.0 * dI, 1e-9)

    close("d(89.999 deg) == d_floor", penetration_depth_nm(lam, n1, n2, 89.999),
          depth_floor_nm(lam, n1, n2), 1e-4)
    close("d_floor closed form", depth_floor_nm(lam, n1, n2),
          lam / (4 * np.pi * np.sqrt(n1 ** 2 - n2 ** 2)), 1e-9)
    chk("d > d_floor for every finite angle",
        np.all(penetration_depth_nm(lam, n1, n2, np.linspace(62, 89, 400))
               > depth_floor_nm(lam, n1, n2)))

    chk("d proportional to lambda0",
        abs(penetration_depth_nm(2 * lam, n1, n2, th) - 2 * dI) < 1e-9)
    dd = penetration_depth_nm(lam, n1, n2, np.linspace(62, 89, 300))
    chk("d monotonically decreasing in theta", np.all(np.diff(dd) < 0))
    chk("d = NaN below theta_c", np.isnan(penetration_depth_nm(lam, n1, n2, 55.0)))

    rs, rp = fresnel_tir(n1, n2, th)
    close("|r_s| == 1 in TIR", abs(rs), 1.0, 1e-12)
    close("|r_p| == 1 in TIR", abs(rp), 1.0, 1e-12)

    thc = float(critical_angle_deg(n1, n2))
    Is, Ip, Ix, Iz = interface_intensity(n1, n2, thc + 1e-7)
    close("I_s(theta_c+) -> 4", Is, 4.0, 1e-3)
    close("I_p(theta_c+) -> 4 / nb^2", Ip, 4.0 / (n2 / n1) ** 2, 1e-2)
    Is9, Ip9, _, _ = interface_intensity(n1, n2, 89.99999)
    close("I_s(90 deg) -> 0", Is9, 0.0, 1e-6)
    close("I_p(90 deg) -> 0", Ip9, 0.0, 1e-6)
    Is2, Ip2, Ix2, Iz2 = interface_intensity(n1, n2, 70.0)
    close("I_p == I_x + I_z", Ip2, Ix2 + Iz2, 1e-12)

    # ---- objective-type angle entry ----
    close("theta_from_NA(1.49, 1.5255)", theta_from_NA(1.49, 1.5255),
          math.degrees(math.asin(1.49 / 1.5255)), 1e-9)
    chk("NA >= n1 -> NaN", np.isnan(theta_from_NA(1.60, 1.5255)))

    # ---- THE NA_eff identity (the central result) ----
    close("NA_eff(n1, arcsin(NA/n1)) == NA",
          NA_eff(1.5255, theta_from_NA(1.49, 1.5255)), 1.49, 1e-12)
    # d = lam0 / (4 pi sqrt(NA^2 - n2^2)) : n1 must cancel out completely
    n2t = 1.3349
    d_direct = 532.0 / (4 * np.pi * np.sqrt(1.49 ** 2 - n2t ** 2))
    for n1t in (1.5210, 1.5264, 1.5983, 1.7367, 1.8590):
        close(f"objective identity: d(NA=1.49) independent of n1={n1t}",
              penetration_depth_nm(532.0, n1t, n2t, theta_from_NA(1.49, n1t)),
              d_direct, 1e-9)
    # a prism at grazing recovers NA_eff -> n1
    close("prism grazing: NA_eff -> n1", NA_eff(1.4607, 89.999), 1.4607, 1e-6)

    close("supercritical annulus (NA 1.49, n2 1.333)",
          supercritical_annulus_fraction(1.49, 1.333),
          (1.49 ** 2 - 1.333 ** 2) / 1.49 ** 2, 1e-12)
    chk("NA <= n2 -> no TIR annulus", np.isnan(supercritical_annulus_fraction(1.30, 1.333)))

    # ---- prism-type angle entry ----
    npz = 1.4607
    close("prism: phi = 0 -> theta1 = A", prism_theta1_deg(60.0, 0.0, npz), 60.0, 1e-12)
    phi_need = prism_phi_ext_deg(60.0, 68.0, npz)
    close("prism: forward/inverse round-trip",
          prism_theta1_deg(60.0, phi_need, npz), 68.0, 1e-9)
    chk("prism: theta1 > A requires phi < 0", phi_need < 0.0)
    num = ((prism_theta1_deg(60.0, phi_need + 1e-5, npz)
            - prism_theta1_deg(60.0, phi_need - 1e-5, npz)) / 2e-5)
    close("prism: analytic dtheta/dphi == numerical",
          prism_dtheta_dphi(60.0, phi_need, npz), num, 1e-6)
    chk("prism: unreachable target -> NaN", np.isnan(prism_phi_ext_deg(60.0, 5.0, 1.46)))

    # ---- far-field contamination ----
    m = far_field_metrics(100.0, 0.10, 1000.0)
    close("stray: analytic energy fraction", m["E_stray"],
          0.10 * 1000.0 / (0.90 * 100.0 + 0.10 * 1000.0), 1e-12)
    close("stray: analytic crossover depth", m["z_cross"],
          math.log(0.9 / 0.1) / (1 / 100.0 - 1 / 1000.0), 1e-9)
    chk("stray: apparent 1/e depth > nominal", m["z_1e"] > 100.0)
    chk("stray: f = 0 recovers the ideal depth",
        abs(far_field_metrics(100.0, 0.0, 1000.0)["z_1e"] - 100.0) < 1e-9)

    # ---- fixed-wavelength refractive-index scan ----
    chk("index scan: comma/semicolon/full-width-comma parser",
        parse_index_values("1.46; 1.52\uFF0C1.60") == (1.46, 1.52, 1.60))
    sn1, sn2, smode = resolve_index_pairs("1.46, 1.52, 1.60", "1.333")
    chk("index scan: one n2 broadcasts across n1 list",
        smode == "vary n1 (fixed n2)" and sn1.size == 3 and np.all(sn2 == 1.333))
    sn1, sn2, smode = resolve_index_pairs("1.525", "1.333, 1.35, 1.37")
    chk("index scan: one n1 broadcasts across n2 list",
        smode == "vary n2 (fixed n1)" and sn2.size == 3 and np.all(sn1 == 1.525))
    sn1, sn2, smode = resolve_index_pairs("1.52, 1.60", "1.333, 1.38")
    chk("index scan: equal lists are paired element by element",
        smode == "paired n1/n2" and np.allclose(sn1, [1.52, 1.60])
        and np.allclose(sn2, [1.333, 1.38]))
    chk("index scan: unequal multi-value lists are rejected",
        rejects_value_error(lambda: resolve_index_pairs("1.5,1.6", "1.2,1.3,1.4")))
    chk("index scan: n1 <= n2 is rejected",
        rejects_value_error(lambda: resolve_index_pairs("1.33", "1.34")))
    chk("index scan: more than 12 curves is rejected",
        rejects_value_error(lambda: resolve_index_pairs(np.linspace(1.4, 1.8, 13), [1.3])))
    d_low_n1 = penetration_depth_nm(488.0, 1.52, 1.333, 75.0)
    d_high_n1 = penetration_depth_nm(488.0, 1.70, 1.333, 75.0)
    chk("index scan physics: depth decreases as n1 increases at fixed n2/theta",
        d_high_n1 < d_low_n1)
    d_low_n2 = penetration_depth_nm(488.0, 1.52, 1.333, 75.0)
    d_high_n2 = penetration_depth_nm(488.0, 1.52, 1.40, 75.0)
    chk("index scan physics: depth increases as n2 approaches n1 at fixed theta",
        d_high_n2 > d_low_n2)

    p_scan = dict(DEFAULTS)
    p_scan.update(n_th=11, n_lam=3, index_n1_values=[1.52, 1.60],
                  index_n2_values=[1.333], index_lam=532.0)
    model_scan = Model(p_scan)
    probe_scan = model_scan.probe(p_scan["p_th"], p_scan["p_lam"])
    scan_columns, scan_rows, _scan_meta = model_scan.panel_table("index_scan", probe_scan)
    scan_rows = list(scan_rows)
    chk("index scan model/export table uses the plotted angle x pair grid",
        model_scan.index_d.shape == (11, 2) and len(scan_rows) == 22
        and scan_columns[:6] == ["curve_id", "theta1_deg", "lambda0_nm",
                                 "n1", "n2", "theta_c_deg"])

    # ---- end-to-end sanity ----
    n1b, n2b = medium1_n("N-BK7 (Schott)", 488.0), water_n(488.0, 25.0)
    dchk = penetration_depth_nm(488.0, n1b, n2b, 65.0)
    chk("sanity: d_I(BK7/water, 488 nm, 65 deg) in 60-200 nm", 60.0 < dchk < 200.0)
    chk("sanity: fused-silica floor > BK7 floor (low index costs you depth)",
        depth_floor_nm(532.0, medium1_n("Fused silica (Malitson 1965)", 532.0),
                       water_n(532.0, 25.0))
        > depth_floor_nm(532.0, medium1_n("N-BK7 (Schott)", 532.0), water_n(532.0, 25.0)))

    n_ok = sum(1 for o, _, _ in R if o)
    if verbose:
        print("=" * 78)
        print(f" SELF-TEST  evanescent_depth_gui.py  v{VERSION}")
        print("=" * 78)
        for o, nm, info in R:
            print(f"  [{'PASS' if o else 'FAIL'}] {nm}" + (f"   -- {info}" if not o else ""))
        print("-" * 78)
        print(f"  {n_ok}/{len(R)} passed")
        print("=" * 78)
    return n_ok == len(R), R


# =========================================================================
#  5.  MODEL
# =========================================================================

class Model:
    def __init__(self, p):
        self.p = dict(p)
        q = self.p
        q["index_lam"] = float(q.get("index_lam", q.get("p_lam", 488.0)))
        if not np.isfinite(q["index_lam"]) or q["index_lam"] <= 0.0:
            raise ValueError("fixed refractive-index scan wavelength must be > 0")
        if "index_n1_values" not in q:
            q["index_n1_values"] = [medium1_n(q["m1"], q["index_lam"], q["m1_n"])]
        if "index_n2_values" not in q:
            n2_value = (water_n(q["index_lam"], q["T"], q["dn2"])
                        if q["m2"] == "Pure water (IAPWS)" else float(q["m2_n"]))
            q["index_n2_values"] = [n2_value]
        q["index_n1_values"] = list(parse_index_values(q["index_n1_values"], "n1 values"))
        q["index_n2_values"] = list(parse_index_values(q["index_n2_values"], "n2 values"))
        self.compute()

    def compute(self):
        p = self.p
        self.th = np.linspace(p["th_min"], p["th_max"], p["n_th"])
        self.lam = np.linspace(p["lam_min"], p["lam_max"], p["n_lam"])
        self.n1 = medium1_n(p["m1"], self.lam, p["m1_n"])
        self.n2 = (water_n(self.lam, p["T"], p["dn2"]) if p["m2"] == "Pure water (IAPWS)"
                   else np.full_like(self.lam, float(p["m2_n"])))

        TH, _ = np.meshgrid(self.th, self.lam, indexing="ij")
        N1 = np.broadcast_to(self.n1, TH.shape)
        N2 = np.broadcast_to(self.n2, TH.shape)
        LAM = np.broadcast_to(self.lam, TH.shape)

        self.thc = critical_angle_deg(self.n1, self.n2)
        self.d = penetration_depth_nm(LAM, N1, N2, TH, p["conv"])
        self.dfloor = depth_floor_nm(self.lam, self.n1, self.n2, p["conv"])
        self.Is, self.Ip, self.Ix, self.Iz = interface_intensity(N1, N2, TH)

        self.index_n1, self.index_n2, self.index_scan_mode = resolve_index_pairs(
            p["index_n1_values"], p["index_n2_values"])
        self.index_thc = critical_angle_deg(self.index_n1, self.index_n2)
        self.index_dfloor = depth_floor_nm(
            p["index_lam"], self.index_n1, self.index_n2, p["conv"])
        self.index_d = penetration_depth_nm(
            p["index_lam"], self.index_n1[np.newaxis, :], self.index_n2[np.newaxis, :],
            self.th[:, np.newaxis], p["conv"])

    def probe(self, th, lam):
        p = self.p
        n1 = medium1_n(p["m1"], lam, p["m1_n"])
        n2 = (water_n(lam, p["T"], p["dn2"]) if p["m2"] == "Pure water (IAPWS)"
              else float(p["m2_n"]))
        thc = float(critical_angle_deg(n1, n2))
        out = dict(
            lam=lam, th=th, n1=float(n1), n2=float(n2), thc=thc,
            dI=float(penetration_depth_nm(lam, n1, n2, th, "intensity")),
            dE=float(penetration_depth_nm(lam, n1, n2, th, "amplitude")),
            floor=float(depth_floor_nm(lam, n1, n2, p["conv"])),
            d=float(penetration_depth_nm(lam, n1, n2, th, p["conv"])),
        )
        Is, Ip, Ix, Iz = interface_intensity(n1, n2, th)
        out.update(Is=float(Is), Ip=float(Ip), Ix=float(Ix), Iz=float(Iz))
        out["NAeff"] = float(NA_eff(n1, th))

        NA = p["NA"]
        out["th_NA"] = float(theta_from_NA(NA, n1))
        out["d_NA"] = float(penetration_depth_nm(lam, n1, n2, out["th_NA"], p["conv"]))
        out["annulus"] = float(supercritical_annulus_fraction(NA, n2))
        with np.errstate(invalid="ignore"):
            out["dthdNA"] = float(np.degrees(1.0 / (n1 * np.cos(np.radians(out["th_NA"])))))

        A = p["prismA"]
        out["phi_ext"] = float(prism_phi_ext_deg(A, th, n1))
        out["dthdphi"] = float(prism_dtheta_dphi(A, out["phi_ext"], n1))
        dth = 0.1 * abs(out["dthdphi"])
        d2 = float(penetration_depth_nm(lam, n1, n2, th + dth, p["conv"]))
        out["dd_rel_0p1deg"] = (abs(d2 - out["d"]) / out["d"]
                                if np.isfinite(out["d"]) else np.nan)

        out.update(far_field_metrics(out["d"], p["stray_f"], p["stray_d"]))
        return out

    def angle_cut_indices(self):
        """Wavelength indices used by panel (b)."""
        return np.unique(np.linspace(0, len(self.lam) - 1,
                                     self.p["ncut"]).astype(int))

    def wavelength_cut_indices(self):
        """Angle indices used by panel (c); every cut must be TIR at all wavelengths."""
        ok_th = np.where(self.th > np.nanmax(self.thc))[0]
        return (np.unique(np.linspace(ok_th[0], ok_th[-1],
                                     self.p["ncut"]).astype(int))
                if ok_th.size else np.array([], dtype=int))

    def _header(self, panel_id, panel_meta=()):
        p = self.p
        r1 = medium1_range(p["m1"])
        r1s = f"{r1[0]:.0f}-{r1[1]:.0f} nm" if r1 else "n/a (user constant)"
        panel_title = next(title for key, title, _description in PANEL_SPECS
                           if key == panel_id)
        header = [
            f"# evanescent_depth_gui.py v{VERSION}   {datetime.datetime.now():%Y-%m-%d %H:%M}",
            "# TIRF evanescent-wave penetration depth, dielectric/dielectric interface",
            f"# exported GUI panel  : {panel_title} [{panel_id}]",
            f"# convention          : {p['conv']}  "
            f"({'d = lam0/(4 pi Q)' if p['conv'] == 'intensity' else 'd = lam0/(2 pi Q)'}), "
            "Q = sqrt(n1^2 sin^2 th - n2^2), lam0 = VACUUM wavelength",
        ]
        if panel_id == "index_scan":
            n1_text = ", ".join(f"{item:.9g}" for item in p["index_n1_values"])
            n2_text = ", ".join(f"{item:.9g}" for item in p["index_n2_values"])
            header.extend([
                "# refractive indices  : user-supplied real constants at fixed wavelength [ASSUMED]",
                f"# input n1 values     : {n1_text}",
                f"# input n2 values     : {n2_text}",
                f"# fixed lambda0       : {p['index_lam']:.9g} nm (VACUUM wavelength)",
            ])
        else:
            header.extend([
                f"# medium 1            : {p['m1']}   [{medium1_tag(p['m1'])}]",
                f"#   source            : {medium1_ref(p['m1'])}",
                f"#   model valid over  : {r1s}",
                f"# medium 2            : {p['m2']}   T = {p['T']} C   dn = {p['dn2']:+.4f}",
            ])
            if p["m2"] == "Pure water (IAPWS)":
                header.extend([
                    f"#   source            : {WATER_REF}",
                    f"#   model valid over  : {WATER_LAM_RANGE[0]:.0f}-{WATER_LAM_RANGE[1]:.0f} nm, "
                    f"{WATER_T_RANGE[0]:.0f}-{WATER_T_RANGE[1]:.0f} C",
                ])
            else:
                header.extend([
                    "#   source            : [ASSUMED] user-supplied constant index",
                    "#   model valid over  : n/a (user constant)",
                ])
            header.append(
                f"# lambda0             : {p['lam_min']}..{p['lam_max']} nm, {p['n_lam']} pts")
        header.extend([
            f"# theta1              : {p['th_min']}..{p['th_max']} deg, {p['n_th']} pts "
            "(angle INSIDE medium 1)",
            "# NOT valid for: absorbing media, metal films, multilayers (SPR/SPRM/SEIRAS),",
            "#   or within ~1 deg of theta_c with a finite-aperture beam.",
        ])
        header.extend(f"# panel metadata      : {item}" for item in panel_meta)
        header.append("#")
        return header

    def rows(self):
        """Legacy full-grid rows retained for API compatibility."""
        for i, th in enumerate(self.th):
            for j, lam in enumerate(self.lam):
                yield [f"{th:.5f}", f"{lam:.4f}", f"{self.n1[j]:.6f}", f"{self.n2[j]:.6f}",
                       f"{self.thc[j]:.5f}", f"{self.d[i, j]:.5f}", f"{self.dfloor[j]:.5f}",
                       f"{self.Is[i, j]:.6f}", f"{self.Ip[i, j]:.6f}"]

    def panel_table(self, panel_id, probe):
        """Return (columns, rows, metadata) for exactly one plotted GUI panel."""
        p = self.p

        if panel_id == "depth_map":
            th_na = float(probe["th_NA"])
            columns = ["theta1_deg", "lambda0_nm", "n1", "n2", "theta_c_deg",
                       "depth_nm", "depth_floor_nm", "objective_NA",
                       "objective_theta_limit_deg"]

            def rows():
                for i, th in enumerate(self.th):
                    for j, lam in enumerate(self.lam):
                        yield [float(th), float(lam), float(self.n1[j]), float(self.n2[j]),
                               float(self.thc[j]), float(self.d[i, j]), float(self.dfloor[j]),
                               float(p["NA"]), th_na]

            meta = ["full theta1 x lambda0 grid used by the colour map and contours",
                    f"plotted objective-NA vertical limit: {th_na:.9g} deg at the probe wavelength"]

        elif panel_id == "angle_scan":
            idx = self.angle_cut_indices()
            columns = ["theta1_deg", "lambda0_nm", "n1", "n2", "theta_c_deg",
                       "depth_nm", "depth_floor_nm"]

            def rows():
                for j in idx:
                    for i, th in enumerate(self.th):
                        yield [float(th), float(self.lam[j]), float(self.n1[j]),
                               float(self.n2[j]), float(self.thc[j]),
                               float(self.d[i, j]), float(self.dfloor[j])]

            meta = ["plotted wavelength cuts: "
                    + ", ".join(f"{self.lam[j]:.4g} nm" for j in idx)]

        elif panel_id == "index_scan":
            columns = ["curve_id", "theta1_deg", "lambda0_nm", "n1", "n2",
                       "theta_c_deg", "depth_nm", "depth_floor_nm", "is_TIR"]

            def rows():
                for k, (n1k, n2k) in enumerate(zip(self.index_n1, self.index_n2), 1):
                    for i, th in enumerate(self.th):
                        depth = float(self.index_d[i, k - 1])
                        yield [k, float(th), float(p["index_lam"]), float(n1k),
                               float(n2k), float(self.index_thc[k - 1]), depth,
                               float(self.index_dfloor[k - 1]), int(np.isfinite(depth))]

            pair_text = "; ".join(
                f"{k}: n1={n1k:.9g}, n2={n2k:.9g}, theta_c={thc:.9g} deg"
                for k, (n1k, n2k, thc) in enumerate(
                    zip(self.index_n1, self.index_n2, self.index_thc), 1))
            meta = [f"fixed vacuum wavelength: {p['index_lam']:.9g} nm",
                    f"scan mode: {self.index_scan_mode}",
                    "refractive-index values are real, wavelength-fixed numeric inputs [ASSUMED]",
                    f"resolved pairs: {pair_text}"]

        elif panel_id == "wavelength_scan":
            idx = self.wavelength_cut_indices()
            columns = ["lambda0_nm", "n1", "n2", "theta_c_deg", "depth_floor_nm"]
            columns.extend(f"depth_at_theta1_{self.th[i]:.5f}_deg_nm" for i in idx)

            def rows():
                for j, lam in enumerate(self.lam):
                    yield ([float(lam), float(self.n1[j]), float(self.n2[j]),
                            float(self.thc[j]), float(self.dfloor[j])]
                           + [float(self.d[i, j]) for i in idx])

            cuts = ", ".join(f"{self.th[i]:.5g} deg" for i in idx) or "none"
            meta = [f"plotted internal-angle cuts: {cuts}"]

        elif panel_id == "substrates":
            lam0, n2p = float(probe["lam"]), float(probe["n2"])
            columns = ["substrate", "lambda0_nm", "n1", "n2", "theta_c_deg",
                       "depth_floor_nm", "theta_c_plus_2deg", "depth_at_theta_c_plus_2deg_nm",
                       "prism_theta1_deg", "prism_depth_nm", "objective_NA",
                       "objective_theta1_deg", "objective_depth_nm"]

            def rows():
                for name in COMPARE_SET:
                    n1k = float(medium1_n(name, lam0))
                    if n1k <= n2p:
                        continue
                    thc_k = float(critical_angle_deg(n1k, n2p))
                    th_hi = thc_k + 2.0
                    d_hi = float(penetration_depth_nm(lam0, n1k, n2p, th_hi, p["conv"]))
                    d_lo = float(depth_floor_nm(lam0, n1k, n2p, p["conv"]))
                    d_75 = float(penetration_depth_nm(lam0, n1k, n2p, 75.0, p["conv"]))
                    th_na = float(theta_from_NA(p["NA"], n1k))
                    d_na = (float(penetration_depth_nm(lam0, n1k, n2p, th_na, p["conv"]))
                            if np.isfinite(th_na) and th_na > thc_k else np.nan)
                    yield [name, lam0, n1k, n2p, thc_k, d_lo, th_hi, d_hi,
                           75.0, d_75, float(p["NA"]), th_na, d_na]

            meta = [f"probe wavelength: {lam0:.6g} nm",
                    f"objective NA: {p['NA']:.6g}; prism marker angle: 75 deg"]

        elif panel_id == "interface":
            j = len(self.lam) // 2
            columns = ["theta1_deg", "lambda0_nm", "n1", "n2", "theta_c_deg",
                       "I_s(0)", "I_p(0)", "I_x", "I_z"]

            def rows():
                for i, th in enumerate(self.th):
                    yield [float(th), float(self.lam[j]), float(self.n1[j]),
                           float(self.n2[j]), float(self.thc[j]),
                           float(self.Is[i, j]), float(self.Ip[i, j]),
                           float(self.Ix[i, j]), float(self.Iz[i, j])]

            meta = [f"interface-intensity wavelength: {self.lam[j]:.6g} nm"]

        elif panel_id == "stray":
            d = float(probe["d"])
            if not np.isfinite(d):
                raise ValueError("probe angle is below theta_c: panel (f) has no profile data")
            f_stray, ds = float(p["stray_f"]), float(p["stray_d"])
            z = np.linspace(0.0, min(6.0 * d, 4.0 * ds), 800)
            ideal = np.exp(-z / d)
            columns = ["z_nm", "ideal_evanescent_intensity"]
            if f_stray > 0.0:
                composite = ((1.0 - f_stray) * ideal + f_stray * np.exp(-z / ds))
                stray = f_stray * np.exp(-z / ds)
                columns += ["composite_intensity", "stray_component_intensity",
                            "stray_dominates"]

                def rows():
                    for k in range(len(z)):
                        yield [float(z[k]), float(ideal[k]), float(composite[k]),
                               float(stray[k]), int(stray[k] > (1.0 - f_stray) * ideal[k])]
            else:
                def rows():
                    for k in range(len(z)):
                        yield [float(z[k]), float(ideal[k])]

            meta = [f"nominal penetration depth: {d:.9g} nm",
                    f"stray fraction at z=0: {f_stray:.9g}",
                    f"stray decay length: {ds:.9g} nm",
                    f"apparent composite 1/e depth: {probe['z_1e']:.9g} nm",
                    f"stray crossover depth: {probe['z_cross']:.9g} nm",
                    f"integrated non-evanescent fraction: {probe['E_stray']:.9g}"]

        else:
            raise KeyError(f"unknown plot panel: {panel_id!r}")

        return columns, rows(), meta

    @staticmethod
    def _json_safe(value):
        """Convert NumPy values and non-finite floats into strict JSON values."""
        if isinstance(value, dict):
            return {str(key): Model._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [Model._json_safe(item) for item in value]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            value = float(value)
            return value if np.isfinite(value) else None
        return value

    def export_settings_json(self, data_path, panel_id, probe, panel_meta=()):
        """Write a same-stem JSON sidecar containing all settings needed to reproduce a run."""
        p = self.p
        panel_title, panel_description = next(
            (title, description) for key, title, description in PANEL_SPECS
            if key == panel_id)
        settings = dict(p)
        settings.update(p_th=float(probe["th"]), p_lam=float(probe["lam"]))
        r1 = medium1_range(p["m1"])
        medium2 = (dict(name=p["m2"], tag="LIT", source=WATER_REF,
                        valid_lambda_nm=list(WATER_LAM_RANGE),
                        valid_temperature_C=list(WATER_T_RANGE))
                   if p["m2"] == "Pure water (IAPWS)"
                   else dict(name=p["m2"], tag="ASSUMED",
                             source="user-supplied constant index",
                             valid_lambda_nm=None, valid_temperature_C=None))
        data_path = os.fspath(data_path)
        json_path = os.path.splitext(data_path)[0] + ".json"
        payload = {
            "schema_version": 1,
            "application": {
                "script": "evanescent_depth_gui.py",
                "version": VERSION,
            },
            "exported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "data_file": {
                "name": os.path.basename(data_path),
                "format": os.path.splitext(data_path)[1].lstrip(".").lower(),
            },
            "panel": {
                "id": panel_id,
                "title": panel_title,
                "description": panel_description.replace("\n", " "),
                "metadata": list(panel_meta),
            },
            # Keys intentionally match DEFAULTS / GUI parameter names so a future
            # import function can pass this mapping back into Model unchanged.
            "settings": settings,
            "probe_results": dict(probe),
            "provenance": {
                "medium1": {
                    "name": p["m1"],
                    "tag": medium1_tag(p["m1"]),
                    "source": medium1_ref(p["m1"]),
                    "valid_lambda_nm": list(r1) if r1 else None,
                },
                "medium2": medium2,
            },
        }
        if panel_id == "index_scan":
            payload["refractive_index_scan"] = {
                "tag": "ASSUMED",
                "source": "user-supplied real refractive indices at one fixed vacuum wavelength",
                "fixed_lambda0_nm": p["index_lam"],
                "mode": self.index_scan_mode,
                "input_n1_values": list(p["index_n1_values"]),
                "input_n2_values": list(p["index_n2_values"]),
                "resolved_pairs": [
                    {"curve_id": k, "n1": n1k, "n2": n2k, "theta_c_deg": thc}
                    for k, (n1k, n2k, thc) in enumerate(
                        zip(self.index_n1, self.index_n2, self.index_thc), 1)
                ],
            }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2,
                      allow_nan=False)
            f.write("\n")
        return json_path

    def export_csv(self, path, panel_id, probe):
        columns, rows, meta = self.panel_table(panel_id, probe)
        with open(path, "w", newline="", encoding="utf-8") as f:
            for line in self._header(panel_id, meta):
                f.write(line + "\n")
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows(rows)
        return self.export_settings_json(path, panel_id, probe, meta)

    def export_xlsx(self, path, panel_id, probe):
        try:
            from openpyxl import Workbook
        except ImportError:
            raise RuntimeError("openpyxl is not installed:  pip install openpyxl")
        columns, rows, meta = self.panel_table(panel_id, probe)
        wb = Workbook()
        ws = wb.active
        ws.title = panel_id[:31]
        for line in self._header(panel_id, meta):
            ws.append([line])
        ws.append(columns)
        for row in rows:
            clean = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    value = float(value)
                    clean.append(value if np.isfinite(value) else None)
                elif isinstance(value, np.integer):
                    clean.append(int(value))
                else:
                    clean.append(value)
            ws.append(clean)
        wb.save(path)
        return self.export_settings_json(path, panel_id, probe, meta)


# =========================================================================
#  6.  FIGURE
# =========================================================================

PANEL_SPECS = (
    ("depth_map", "(a) Depth map",
     "x: incidence angle inside medium 1.  y: vacuum wavelength.  "
     "Colour and white contours: selected 1/e penetration depth.\n"
     "Dashed curve: critical-angle boundary.  Dotted line: objective-NA limit."),
    ("angle_scan", "(b) Angle scan",
     "x: incidence angle inside medium 1.  y: selected 1/e penetration depth (log scale).  "
     "Each colour is a vacuum wavelength; its dotted line is the grazing-incidence floor."),
    ("wavelength_scan", "(c) Wavelength scan",
     "x: vacuum wavelength.  y: selected 1/e penetration depth (log scale).  "
     "Each colour is an internal incidence angle; the dashed curve is the grazing floor."),
    ("substrates", "(d) Substrate comparison",
     "x: penetration depth at the probe wavelength (log scale).  y: substrate.  "
     "Bar: floor to the value at two degrees above critical; tick: floor; circle: prism at 75 deg; square: objective."),
    ("interface", "(e) Interface intensity",
     "x: incidence angle inside medium 1.  y: interface field intensity normalized by incident-field intensity.  "
     "Polarization changes I(0), but not the penetration depth."),
    ("stray", "(f) Far-field contamination",
     "x: distance from the interface.  y: normalized excitation intensity (log scale).  "
     "The ideal evanescent exponential is compared with the double-exponential profile and its stray component."),
    ("index_scan", "(g) Refractive-index scan",
     "x: incidence angle inside medium 1.  y: selected 1/e penetration depth (log scale).  "
     "Each colour is one real (n1, n2) pair at a fixed vacuum wavelength; its dotted line is the grazing floor.\n"
     "Numeric refractive-index inputs are wavelength-fixed [ASSUMED] values."),
)


def _depth_label(p):
    return ("Intensity 1/e depth $d_I$" if p["conv"] == "intensity"
            else "Field 1/e depth $d_E$")


def _context_title(M):
    p = M.p
    m2lab = ("Pure water (IAPWS R9-97)" if p["m2"] == "Pure water (IAPWS)"
             else f"n2 = {p['m2_n']}")
    return (f"{p['m1']}  |  {m2lab} @ {p['T']:g} $^\\circ$C   "
             f"($\\theta_c$ = {np.nanmin(M.thc):.2f}-{np.nanmax(M.thc):.2f}$^\\circ$)")


def _index_context_title(M):
    p = M.p
    convention = "intensity 1/e" if p["conv"] == "intensity" else "field 1/e"
    return (f"Fixed $\\lambda_0$ = {p['index_lam']:.6g} nm  |  "
            f"{M.index_scan_mode}  |  {convention}  |  "
            "real numeric indices [ASSUMED]")


def _plot_depth_map(a, fig, M, probe, compact=False):
    p = M.p
    fs, lfs = ((9, 7) if compact else (12, 9))
    try:
        cmap = matplotlib.colormaps["viridis"].with_extremes(bad="0.90")
    except (AttributeError, KeyError):
        cmap = matplotlib.cm.get_cmap("viridis").copy()
        cmap.set_bad("0.90")
    dclip = np.clip(M.d, None, p["dmax"])
    im = a.pcolormesh(M.th, M.lam, dclip.T, cmap=cmap, shading="auto",
                      norm=matplotlib.colors.LogNorm(
                          vmin=max(np.nanmin(M.d), 1.0), vmax=p["dmax"]))
    cb = fig.colorbar(im, ax=a, pad=0.025)
    cb.set_label(f"{_depth_label(p)} (nm)")
    a.plot(M.thc, M.lam, ls="--", lw=1.8, color=OKABE[1],
           label=r"critical angle $\theta_c(\lambda_0)$")
    cs = a.contour(M.th, M.lam, dclip.T, levels=[50, 75, 100, 150, 200, 300, 500],
                   colors="white", linewidths=0.8)
    a.clabel(cs, inline=True, fontsize=lfs, fmt="%d nm")
    if np.isfinite(probe["th_NA"]):
        a.axvline(probe["th_NA"], color=OKABE[3], ls=":", lw=1.8,
                  label=f"objective NA={p['NA']:.2f} limit")
    a.set_xlabel(r"Incidence angle $\theta_1$ (deg, measured inside medium 1)")
    a.set_ylabel(r"Vacuum wavelength $\lambda_0$ (nm)")
    a.set_title(f"(a) {_depth_label(p)} over angle and wavelength (log colour scale)", fontsize=fs)
    a.legend(fontsize=lfs, loc="best", framealpha=0.9)


def _plot_angle_scan(a, _fig, M, _probe, compact=False):
    p = M.p
    fs, lfs = ((9, 7) if compact else (12, 9))
    idx = M.angle_cut_indices()
    for k, j in enumerate(idx):
        c = OKABE[k % len(OKABE)]
        a.plot(M.th, M.d[:, j], color=c, lw=1.7, label=f"{M.lam[j]:.0f} nm")
        a.axhline(M.dfloor[j], color=c, ls=":", lw=1.1)
    a.plot([], [], color="0.35", ls=":", lw=1.1, label="matching-colour floor")
    a.set_yscale("log")
    a.set_xlabel(r"Incidence angle $\theta_1$ (deg, measured inside medium 1)")
    a.set_ylabel(f"{_depth_label(p)} (nm, log scale)")
    a.set_title(r"(b) Penetration depth versus incidence angle", fontsize=fs)
    a.legend(fontsize=lfs, title=r"Vacuum wavelength $\lambda_0$", title_fontsize=lfs)
    a.grid(alpha=0.3, which="both")


def _plot_index_scan(a, _fig, M, _probe, compact=False):
    p = M.p
    fs, lfs = ((9, 7) if compact else (12, 8))
    for k, (n1k, n2k, thc, floor) in enumerate(
            zip(M.index_n1, M.index_n2, M.index_thc, M.index_dfloor)):
        colour = OKABE[k % len(OKABE)]
        label = (rf"$n_1$={n1k:.5g}, $n_2$={n2k:.5g}, "
                 rf"$\theta_c$={thc:.3f}$^\circ$")
        a.plot(M.th, M.index_d[:, k], color=colour, lw=1.8, label=label)
        a.axhline(floor, color=colour, ls=":", lw=1.0)
    a.plot([], [], color="0.35", ls=":", lw=1.0, label="matching-colour grazing floor")
    a.set_yscale("log")
    a.set_xlabel(r"Incidence angle $\theta_1$ (deg, measured inside medium 1)")
    a.set_ylabel(f"{_depth_label(p)} (nm, log scale)")
    a.set_title(
        f"(g) Refractive-index sensitivity at fixed vacuum wavelength "
        f"$\\lambda_0$={p['index_lam']:.6g} nm\n"
        r"Each curve uses one wavelength-fixed, real $(n_1,n_2)$ pair",
        fontsize=fs)
    ncol = 2 if not compact and len(M.index_n1) > 6 else 1
    a.legend(fontsize=lfs, title=M.index_scan_mode, title_fontsize=lfs,
             ncol=ncol, loc="best", framealpha=0.9)
    a.grid(alpha=0.3, which="both")


def _plot_wavelength_scan(a, _fig, M, _probe, compact=False):
    p = M.p
    fs, lfs = ((9, 7) if compact else (12, 9))
    jdx = M.wavelength_cut_indices()
    for k, i in enumerate(jdx):
        a.plot(M.lam, M.d[i, :], color=OKABE[k % len(OKABE)], lw=1.7,
               label=f"{M.th[i]:.2f}$^\\circ$")
    a.plot(M.lam, M.dfloor, color="0.3", ls="--", lw=1.5,
           label=r"grazing floor ($\theta_1\to90^\circ$)")
    a.set_yscale("log")
    a.set_xlabel(r"Vacuum wavelength $\lambda_0$ (nm)")
    a.set_ylabel(f"{_depth_label(p)} (nm, log scale)")
    a.set_title(r"(c) Wavelength dependence including dispersion of both $n_1$ and $n_2$",
                fontsize=fs)
    a.legend(fontsize=lfs, title=r"Internal angle $\theta_1$", title_fontsize=lfs)
    a.grid(alpha=0.3, which="both")


def _plot_substrates(a, _fig, M, probe, compact=False):
    p = M.p
    fs, lfs = ((9, 7) if compact else (12, 9))
    lam0, n2p = probe["lam"], probe["n2"]
    names, y = [], []
    lab_p, lab_o = False, False
    for k, nm in enumerate(COMPARE_SET):
        n1k = float(medium1_n(nm, lam0))
        if n1k <= n2p:
            continue
        thc_k = float(critical_angle_deg(n1k, n2p))
        d_hi = float(penetration_depth_nm(lam0, n1k, n2p, thc_k + 2.0, p["conv"]))
        d_lo = float(depth_floor_nm(lam0, n1k, n2p, p["conv"]))
        d_75 = float(penetration_depth_nm(lam0, n1k, n2p, 75.0, p["conv"]))
        yy = len(names)
        col = OKABE[k % len(OKABE)]
        names.append(f"{nm.split('(')[0].strip()} (n={n1k:.3f})")
        a.plot([d_lo, d_hi], [yy, yy], lw=7, color=col, alpha=0.35,
               solid_capstyle="butt")
        a.plot([d_lo], [yy], "|", ms=14, color=col)
        a.plot([d_75], [yy], "o", ms=6, color=col,
               label=None if lab_p else "prism at 75$^\\circ$")
        lab_p = True
        thN = float(theta_from_NA(p["NA"], n1k))
        if np.isfinite(thN) and thN > thc_k:
            dN = float(penetration_depth_nm(lam0, n1k, n2p, thN, p["conv"]))
            a.plot([dN], [yy], "s", ms=7, mfc="none", mew=1.7, color=col,
                   label=None if lab_o else f"objective NA={p['NA']:.2f}")
            lab_o = True
        y.append(yy)
    a.set_yticks(y)
    a.set_yticklabels(names, fontsize=lfs)
    a.set_xscale("log")
    a.set_xlabel(rf"{_depth_label(p)} at $\lambda_0$={lam0:.0f} nm (log scale)")
    a.set_title("(d) Substrate comparison: bar spans grazing floor to $\\theta_c+2^\\circ$\n"
                r"Objective squares coincide because $d=\lambda_0/(4\pi\sqrt{NA^2-n_2^2})$",
                fontsize=fs)
    a.grid(alpha=0.3, axis="x", which="both")
    a.legend(fontsize=lfs, loc="best")
    a.invert_yaxis()


def _plot_interface(a, _fig, M, _probe, compact=False):
    fs, lfs = ((9, 7) if compact else (12, 9))
    jm = len(M.lam) // 2
    n1m, n2m = M.n1[jm], M.n2[jm]
    Is, Ip, Ix, Iz = interface_intensity(n1m, n2m, M.th)
    a.plot(M.th, Is, color=OKABE[0], lw=1.8, label=r"s polarization: $I_s(0)$")
    a.plot(M.th, Ip, color=OKABE[1], lw=1.8, label=r"p polarization: $I_p(0)$")
    a.plot(M.th, Ix, color=OKABE[2], lw=1.3, ls="-.", label=r"p tangential: $I_x$")
    a.plot(M.th, Iz, color=OKABE[3], lw=1.3, ls=":", label=r"p normal: $I_z$")
    a.axvline(M.thc[jm], color="0.4", ls="--", lw=1.1,
              label=r"critical angle $\theta_c$")
    a.set_xlabel(r"Incidence angle $\theta_1$ (deg, measured inside medium 1)")
    a.set_ylabel(r"Normalized interface intensity $|E(0)|^2/|E_{inc}|^2$")
    a.set_title(rf"(e) Interface field intensity at $\lambda_0$={M.lam[jm]:.0f} nm" "\n"
                "Polarization changes the interface intensity, not the penetration depth",
                fontsize=fs)
    a.legend(fontsize=lfs, ncol=2 if not compact else 1)
    a.grid(alpha=0.3)


def _plot_stray(a, _fig, M, probe, compact=False):
    p = M.p
    fs, lfs = ((9, 7) if compact else (12, 9))
    dprobe = probe["d"]
    f_, ds = p["stray_f"], p["stray_d"]
    if not np.isfinite(dprobe):
        a.text(0.5, 0.5, "Probe angle is below $\\theta_c$:\nno total internal reflection",
               ha="center", va="center", transform=a.transAxes,
               fontsize=fs, color=OKABE[1])
        a.set_axis_off()
        return
    z = np.linspace(0, min(6 * dprobe, 4 * ds), 800)
    a.plot(z, np.exp(-z / dprobe), color="0.3", ls="--", lw=1.7,
           label=f"ideal evanescent exp. ($d$={dprobe:.1f} nm)")
    if f_ > 0:
        a.plot(z, (1 - f_) * np.exp(-z / dprobe) + f_ * np.exp(-z / ds),
               color=OKABE[1], lw=2.0, label=f"composite profile ({100 * f_:.0f}% stray at z=0)")
        a.plot(z, f_ * np.exp(-z / ds), color=OKABE[3], lw=1.4, ls=":",
               label=f"stray component ($d_{{stray}}$={ds:.0f} nm)")
    if np.isfinite(probe["z_cross"]) and probe["z_cross"] < z[-1]:
        a.axvline(probe["z_cross"], color=OKABE[2], lw=1.3,
                  label=f"stray crossover: {probe['z_cross']:.0f} nm")
    a.set_yscale("log")
    a.set_ylim(1e-3, 1.5)
    a.set_xlabel(r"Distance from interface $z$ (nm, into medium 2)")
    a.set_ylabel(r"Normalized excitation intensity $I(z)/I(0)$ (log scale)")
    a.set_title("(f) Far-field contamination (Mattheyses & Axelrod 2006)\n"
                f"Non-evanescent share of the integrated axial excitation: "
                f"{100 * probe['E_stray']:.0f}%", fontsize=fs)
    a.legend(fontsize=lfs, loc="best")
    a.grid(alpha=0.3, which="both")


PANEL_DRAWERS = {
    "depth_map": _plot_depth_map,
    "angle_scan": _plot_angle_scan,
    "wavelength_scan": _plot_wavelength_scan,
    "substrates": _plot_substrates,
    "interface": _plot_interface,
    "stray": _plot_stray,
    "index_scan": _plot_index_scan,
}


def draw_panel_figure(fig, panel_id, M, probe):
    """Draw one spacious GUI tab, including a concise physical-axis caption."""
    if panel_id not in PANEL_DRAWERS:
        raise KeyError(f"unknown plot panel: {panel_id!r}")
    fig.clear()
    a = fig.subplots(1, 1)
    PANEL_DRAWERS[panel_id](a, fig, M, probe, compact=False)
    context_title = _index_context_title(M) if panel_id == "index_scan" else _context_title(M)
    fig.suptitle(context_title, fontsize=11, y=0.985)
    description = next(d for key, _tab, d in PANEL_SPECS if key == panel_id)
    fig.text(0.5, 0.012, description, ha="center", va="bottom", fontsize=9, color="0.25")
    fig.tight_layout(rect=(0.035, 0.075, 0.985, 0.935))


def draw_figure(fig, M, probe):
    """Draw the legacy six-panel overview used by the headless --preview CLI."""
    fig.clear()
    ax = fig.subplots(2, 3)
    for a, (panel_id, _tab, _description) in zip(ax.flat, PANEL_SPECS[:6]):
        PANEL_DRAWERS[panel_id](a, fig, M, probe, compact=True)
    fig.suptitle(_context_title(M), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


# =========================================================================
#  7.  GUI
# =========================================================================

DEFAULTS = dict(
    m1="Cover glass, D 263 M / ISO 8255-1 (Schott)", m1_n=1.5255,
    m2="Pure water (IAPWS)", m2_n=1.333, T=25.0, dn2=0.0,
    th_min=60.0, th_max=80.0, n_th=401,
    lam_min=400.0, lam_max=800.0, n_lam=201,
    conv="intensity", ncut=5, dmax=600.0,
    NA=1.49, prismA=60.0,
    stray_f=0.10, stray_d=1000.0,
    p_th=68.0, p_lam=488.0,
    index_lam=488.0,
    index_n1_values=(1.46, 1.52, 1.60, 1.78),
    index_n2_values=(1.333,),
)


def launch_gui():                                            # pragma: no cover
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)

    ok, results = run_selftests(verbose=True)

    root = tk.Tk()
    root.title(f"TIRF evanescent penetration depth  v{VERSION}")
    root.geometry("1580x940")

    left = ttk.Frame(root, width=395)
    left.pack(side="left", fill="y")
    left.pack_propagate(False)
    lcanvas = tk.Canvas(left, borderwidth=0, highlightthickness=0, width=377)
    vsb = ttk.Scrollbar(left, orient="vertical", command=lcanvas.yview)
    lcanvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    lcanvas.pack(side="left", fill="both", expand=True)
    ctl = ttk.Frame(lcanvas)
    win = lcanvas.create_window((0, 0), window=ctl, anchor="nw")
    ctl.bind("<Configure>", lambda e: lcanvas.configure(scrollregion=lcanvas.bbox("all")))
    lcanvas.bind("<Configure>", lambda e: lcanvas.itemconfigure(win, width=e.width))

    def _wheel(event):
        delta = -(event.delta // 120) if event.delta else (1 if event.num == 5 else -1)
        lcanvas.yview_scroll(int(delta), "units")

    def _wheel_on(_):
        for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            lcanvas.bind_all(s, _wheel)

    def _wheel_off(_):
        for s in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            lcanvas.unbind_all(s)

    lcanvas.bind("<Enter>", _wheel_on)
    lcanvas.bind("<Leave>", _wheel_off)

    def sec(title):
        ttk.Separator(ctl).pack(fill="x", pady=(8, 2))
        ttk.Label(ctl, text=title, font=("", 9, "bold")).pack(anchor="w", padx=8)

    def field(label, init, width=11):
        fr = ttk.Frame(ctl)
        fr.pack(fill="x", padx=8, pady=1)
        ttk.Label(fr, text=label, width=23).pack(side="left")
        v = tk.StringVar(value=str(init))
        ttk.Entry(fr, textvariable=v, width=width).pack(side="left")
        return v

    sec("Medium 1  (dense side)")
    v_m1 = tk.StringVar(value=DEFAULTS["m1"])
    ttk.Combobox(ctl, textvariable=v_m1, values=MEDIUM1_LIB, state="readonly",
                 width=45).pack(padx=8, pady=2, anchor="w")
    lbl_ref = ttk.Label(ctl, text="", wraplength=355, foreground="#555", font=("", 7))
    lbl_ref.pack(anchor="w", padx=8)
    v_m1n = field("custom n1 (const)", DEFAULTS["m1_n"])

    sec("Medium 2  (rare side)")
    v_m2 = tk.StringVar(value=DEFAULTS["m2"])
    ttk.Combobox(ctl, textvariable=v_m2,
                 values=["Pure water (IAPWS)", "Custom (constant n)"],
                 state="readonly", width=45).pack(padx=8, pady=2, anchor="w")
    v_m2n = field("custom n2 (const)", DEFAULTS["m2_n"])
    v_T = field("temperature (C)", DEFAULTS["T"])
    v_dn2 = field("dn2 offset", DEFAULTS["dn2"])

    sec("Scan grid")
    v_thmin = field("theta1 min (deg)", DEFAULTS["th_min"])
    v_thmax = field("theta1 max (deg)", DEFAULTS["th_max"])
    v_nth = field("theta1 points", DEFAULTS["n_th"])
    v_lmin = field("lambda0 min (nm)", DEFAULTS["lam_min"])
    v_lmax = field("lambda0 max (nm)", DEFAULTS["lam_max"])
    v_nlam = field("lambda0 points", DEFAULTS["n_lam"])

    sec("Fixed-lambda refractive-index scan")
    v_ilam = field("fixed lambda0 (nm)", DEFAULTS["index_lam"])
    v_in1 = field("n1 values (comma list)",
                  ", ".join(f"{v:g}" for v in DEFAULTS["index_n1_values"]), width=21)
    v_in2 = field("n2 values (comma list)",
                  ", ".join(f"{v:g}" for v in DEFAULTS["index_n2_values"]), width=21)
    ttk.Button(ctl, text="Use current media at fixed lambda",
               command=lambda: use_current_pair()).pack(fill="x", padx=8, pady=2)
    ttk.Label(ctl, text="One value is held fixed while the other list is scanned.\n"
                        "If both lists contain >1 value, their lengths must match\n"
                        "and entries are paired by position (maximum 12 curves).",
              foreground="#555", font=("", 7), justify="left").pack(anchor="w", padx=8)

    sec("Depth convention")
    v_conv = tk.StringVar(value=DEFAULTS["conv"])
    ttk.Radiobutton(ctl, text="intensity 1/e:  lam0/(4 pi Q)   [TIRF, Axelrod]",
                    variable=v_conv, value="intensity").pack(anchor="w", padx=8)
    ttk.Radiobutton(ctl, text="field 1/e:  lam0/(2 pi Q) = 2 d_I   [ATR, Harrick]",
                    variable=v_conv, value="amplitude").pack(anchor="w", padx=8)

    sec("Configuration")
    v_NA = field("objective NA", DEFAULTS["NA"])
    v_A = field("prism face angle A (deg)", DEFAULTS["prismA"])
    ttk.Label(ctl, text="A = angle between the entrance-face normal and the\n"
                        "TIR-surface normal.  Equilateral prism on its base: 60.\n"
                        "Right-angle prism coupled via hypotenuse: 45.\n"
                        "(prism must be index-matched to the slide)",
              foreground="#555", font=("", 7), justify="left").pack(anchor="w", padx=8)

    sec("Far-field contamination")
    v_sf = field("stray fraction f", DEFAULTS["stray_f"])
    v_sd = field("stray decay (nm)", DEFAULTS["stray_d"])
    ttk.Label(ctl, text="f ~ 0.10 for objective-type TIRF (Mattheyses &\n"
                        "Axelrod, J. Biomed. Opt. 11, 014006, 2006).\n"
                        "Set f = 0 for the ideal / prism-type case.",
              foreground="#555", font=("", 7), justify="left").pack(anchor="w", padx=8)

    sec("Probe point / display")
    v_pth = field("probe theta1 (deg)", DEFAULTS["p_th"])
    v_plam = field("probe lambda0 (nm)", DEFAULTS["p_lam"])
    v_ncut = field("line cuts", DEFAULTS["ncut"])
    v_dmax = field("depth clip (nm)", DEFAULTS["dmax"])

    status = tk.Label(root, text="", anchor="w", font=("", 9), padx=8, pady=4)
    status.pack(side="bottom", fill="x")

    def set_status(kind, msg):
        if not ok:
            kind, msg = "err", "SELF-TEST FAILED - output NOT trustworthy.  | " + msg
        col = {"ok": ("#0b6b3a", "#e3f5ea"), "warn": ("#8a5a00", "#fdf3dd"),
               "err": ("#8a1c1c", "#fbe4e4")}[kind]
        status.configure(
            text={"ok": "[OK] ", "warn": "[WARN] ", "err": "[ERROR] "}[kind] + msg,
            fg=col[0], bg=col[1])

    right = ttk.Frame(root)
    right.pack(side="right", fill="both", expand=True)
    notebook = ttk.Notebook(right)
    notebook.pack(fill="both", expand=True)

    plot_views = {}
    for panel_id, tab_title, _description in PANEL_SPECS:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=tab_title)
        fig = Figure(figsize=(11.5, 7.5))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        toolbar_host = ttk.Frame(tab)
        toolbar_host.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(canvas, toolbar_host).update()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        plot_views[panel_id] = dict(fig=fig, canvas=canvas, tab=tab)

    readout_tab = ttk.Frame(notebook)
    notebook.add(readout_tab, text="Probe readout")
    readout_scroll = ttk.Scrollbar(readout_tab, orient="vertical")
    readout = tk.Text(readout_tab, font=("Consolas", 10), bg="#f7f7f7",
                      relief="flat", padx=12, pady=10,
                      yscrollcommand=readout_scroll.set)
    readout_scroll.configure(command=readout.yview)
    readout_scroll.pack(side="right", fill="y")
    readout.pack(side="left", fill="both", expand=True)

    state = {"M": None, "probe": None}

    def params():
        p = dict(
            m1=v_m1.get(), m1_n=float(v_m1n.get()),
            m2=v_m2.get(), m2_n=float(v_m2n.get()),
            T=float(v_T.get()), dn2=float(v_dn2.get()),
            th_min=float(v_thmin.get()), th_max=float(v_thmax.get()),
            n_th=int(float(v_nth.get())),
            lam_min=float(v_lmin.get()), lam_max=float(v_lmax.get()),
            n_lam=int(float(v_nlam.get())),
            conv=v_conv.get(), ncut=int(float(v_ncut.get())),
            dmax=float(v_dmax.get()), NA=float(v_NA.get()), prismA=float(v_A.get()),
            stray_f=float(v_sf.get()), stray_d=float(v_sd.get()),
            index_lam=float(v_ilam.get()),
            index_n1_values=list(parse_index_values(v_in1.get(), "n1 values")),
            index_n2_values=list(parse_index_values(v_in2.get(), "n2 values")),
        )
        if not (0.0 < p["th_min"] < p["th_max"] < 90.0):
            raise ValueError("require 0 < theta_min < theta_max < 90 deg")
        if not (0.0 < p["lam_min"] < p["lam_max"]):
            raise ValueError("require 0 < lambda_min < lambda_max")
        if p["n_th"] < 2 or p["n_lam"] < 2:
            raise ValueError("need at least 2 grid points per axis")
        if p["n_th"] * p["n_lam"] > 4_000_000:
            raise ValueError("grid too large (> 4e6 points)")
        if not (0.0 <= p["stray_f"] < 1.0):
            raise ValueError("stray fraction must lie in [0, 1)")
        if p["stray_d"] <= 0.0:
            raise ValueError("stray decay must be > 0")
        if not np.isfinite(p["index_lam"]) or p["index_lam"] <= 0.0:
            raise ValueError("fixed refractive-index scan wavelength must be > 0")
        resolve_index_pairs(p["index_n1_values"], p["index_n2_values"])
        return p

    def use_current_pair():
        """Resolve the selected media at the fixed scan wavelength into one numeric pair."""
        try:
            lam = float(v_ilam.get())
            if not np.isfinite(lam) or lam <= 0.0:
                raise ValueError("fixed refractive-index scan wavelength must be > 0")
            n1 = medium1_n(v_m1.get(), lam, float(v_m1n.get()))
            n2 = (water_n(lam, float(v_T.get()), float(v_dn2.get()))
                  if v_m2.get() == "Pure water (IAPWS)" else float(v_m2n.get()))
            if not (np.isfinite(n1) and np.isfinite(n2) and n1 > n2 > 0.0):
                raise ValueError("selected media do not give finite indices with n1 > n2")
            v_in1.set(f"{n1:.9g}")
            v_in2.set(f"{n2:.9g}")
            compute()
        except Exception as e:
            set_status("err", f"{type(e).__name__}: {e}")

    def compute(*_):
        lbl_ref.configure(text=medium1_ref(v_m1.get()))
        try:
            p = params()
            M = Model(p)
            pr = M.probe(float(v_pth.get()), float(v_plam.get()))
            state["M"], state["probe"] = M, pr
            for panel_id, view in plot_views.items():
                draw_panel_figure(view["fig"], panel_id, M, pr)
                view["canvas"].draw_idle()

            u = "d_I" if p["conv"] == "intensity" else "d_E"
            L = [f"medium 1 : {p['m1']}",
                 f"           {medium1_ref(p['m1'])[:92]}",
                 f"probe    : theta1 = {pr['th']:.3f} deg   lambda0 = {pr['lam']:.1f} nm"
                 f"   n1 = {pr['n1']:.5f}   n2 = {pr['n2']:.5f}",
                 f"           theta_c = {pr['thc']:.3f} deg    {u} = {pr['d']:.2f} nm"
                 f"    (d_I = {pr['dI']:.2f} nm,  d_E = {pr['dE']:.2f} nm)",
                 f"           FLOOR (theta1 -> 90 deg) = {pr['floor']:.2f} nm"
                 f"   <- no angle, no objective can beat this on this substrate",
                 f"           NA_eff = n1 sin(theta1) = {pr['NAeff']:.4f}"
                 f"   <- the ONE number that compares prism vs objective",
                 f"           interface intensity  I_s(0) = {pr['Is']:.3f}"
                 f"   I_p(0) = {pr['Ip']:.3f}",
                 ""]
            if np.isfinite(pr["th_NA"]):
                L += [f"objective: NA = {p['NA']:.3f}  ->  theta_max = {pr['th_NA']:.3f} deg"
                      f"   {u}(theta_max) = {pr['d_NA']:.2f} nm",
                      f"           supercritical BFP annulus = {100 * pr['annulus']:.1f}% "
                      f"of area   |   dtheta/dNA = {pr['dthdNA']:.0f} deg per unit NA"]
            else:
                L += [f"objective: NA = {p['NA']:.3f} >= n1 = {pr['n1']:.4f}  ->  impossible"]
            if np.isfinite(pr["phi_ext"]):
                L += [f"prism    : A = {p['prismA']:.1f} deg  ->  external angle for "
                      f"theta1 = {pr['th']:.2f} deg  is  phi = {pr['phi_ext']:+.3f} deg",
                      f"           dtheta1/dphi = {pr['dthdphi']:+.3f}   |   0.1 deg error "
                      f"in phi  ->  {100 * pr['dd_rel_0p1deg']:.2f}% error in {u}"]
            else:
                L += [f"prism    : theta1 = {pr['th']:.2f} deg unreachable with A = "
                      f"{p['prismA']:.1f} deg (would need |phi| > 90 deg)"]
            L += ["",
                  f"stray    : f = {p['stray_f']:.2f}, decay {p['stray_d']:.0f} nm  ->  "
                  f"apparent 1/e depth = {pr['z_1e']:.1f} nm (nominal {pr['d']:.1f} nm)",
                  f"           stray dominates beyond z = {pr['z_cross']:.0f} nm;  "
                  f"{100 * pr['E_stray']:.0f}% of the TOTAL axial excitation is "
                  f"non-evanescent"]
            readout.delete("1.0", "end")
            readout.insert("1.0", "\n".join(L))

            if pr["th"] < pr["thc"]:
                set_status("err", f"theta1 = {pr['th']:.2f} deg is BELOW theta_c = "
                                  f"{pr['thc']:.2f} deg: no total internal reflection.")
                return
            msgs = []
            if pr["th"] - pr["thc"] < 1.0:
                msgs.append("probe angle within 1 deg of theta_c: with a real "
                            "finite-aperture beam, d is not reliable there")
            if p["th_min"] < np.nanmax(M.thc):
                msgs.append("theta range extends below theta_c(lambda) (grey region)")
            r1 = medium1_range(p["m1"])
            if r1 and (p["lam_min"] < r1[0] - 1e-9 or p["lam_max"] > r1[1] + 1e-9):
                msgs.append(f"lambda outside the validity of the n1 model "
                            f"({r1[0]:.0f}-{r1[1]:.0f} nm) -> EXTRAPOLATING")
            if p["m2"] == "Pure water (IAPWS)":
                if (p["lam_min"] < WATER_LAM_RANGE[0]
                        or p["lam_max"] > WATER_LAM_RANGE[1]):
                    msgs.append(f"lambda outside IAPWS R9-97 validity "
                                f"({WATER_LAM_RANGE[0]:.0f}-{WATER_LAM_RANGE[1]:.0f} nm)")
                if not (WATER_T_RANGE[0] <= p["T"] <= WATER_T_RANGE[1]):
                    msgs.append(f"T outside the Tanaka density range "
                                f"({WATER_T_RANGE[0]:.0f}-{WATER_T_RANGE[1]:.0f} C) "
                                f"-> density EXTRAPOLATED")
            if p["m1"] in _CONST_MEDIA:
                msgs.append("this medium has NO dispersion model (constant n): "
                            "wavelength scans are approximate")
            if p["m1"] in _CAUCHY:
                msgs.append("n1 is a 2-term Cauchy DERIVED from (n, Abbe): ~+-0.002")
            if msgs:
                set_status("warn", "  |  ".join(msgs))
            else:
                set_status("ok", f"{u}({pr['th']:.2f} deg, {pr['lam']:.0f} nm) = "
                                 f"{pr['d']:.2f} nm   floor = {pr['floor']:.2f} nm   "
                                 f"theta_c = {pr['thc']:.2f} deg")
        except Exception as e:
            set_status("err", f"{type(e).__name__}: {e}")

    def auto_theta():
        """Set the theta window from theta_c up to the objective's NA limit."""
        try:
            p = params()
            lam = np.linspace(p["lam_min"], p["lam_max"], 51)
            n1 = medium1_n(p["m1"], lam, p["m1_n"])
            n2 = (water_n(lam, p["T"], p["dn2"]) if p["m2"] == "Pure water (IAPWS)"
                  else np.full_like(lam, p["m2_n"]))
            thc = float(np.nanmax(critical_angle_deg(n1, n2)))
            thN = float(np.nanmin(theta_from_NA(p["NA"], n1)))
            hi = thN if np.isfinite(thN) else 85.0
            if hi <= thc + 0.5:
                raise ValueError(f"NA = {p['NA']} gives theta_max = {hi:.2f} deg, barely "
                                 f"above theta_c = {thc:.2f} deg: not a TIRF lens")
            v_thmin.set(f"{thc + 0.5:.3f}")
            v_thmax.set(f"{hi:.3f}")
            v_pth.set(f"{0.5 * (thc + 0.5 + hi):.3f}")
            compute()
        except Exception as e:
            set_status("err", f"{type(e).__name__}: {e}")

    def save(kind):
        M = state["M"]
        if M is None:
            set_status("err", "compute first")
            return
        tab_index = notebook.index(notebook.select())
        if tab_index >= len(PANEL_SPECS):
            set_status("warn", f"select one of the {len(PANEL_SPECS)} figure tabs before exporting")
            return
        active_panel = PANEL_SPECS[tab_index]
        active_fig = plot_views[active_panel[0]]["fig"] if kind == "fig" else None
        ext = {"csv": ".csv", "xlsx": ".xlsx", "fig": ".png"}[kind]
        path = filedialog.asksaveasfilename(
            defaultextension=ext, initialfile=active_panel[0] + ext)
        if not path:
            return
        try:
            json_path = None
            if kind == "csv":
                json_path = M.export_csv(path, active_panel[0], state["probe"])
            elif kind == "xlsx":
                json_path = M.export_xlsx(path, active_panel[0], state["probe"])
            else:
                active_fig.savefig(path, dpi=300, bbox_inches="tight")
            what = active_panel[1] if active_panel else kind.upper()
            msg = f"written {what}: {path}"
            if json_path:
                msg += f"  |  settings: {json_path}"
            set_status("ok", msg)
        except Exception as e:
            set_status("err", f"{type(e).__name__}: {e}")

    def show_tests():
        messagebox.showinfo(
            "Self-test",
            "\n".join(f"[{'PASS' if o else 'FAIL'}] {n}" for o, n, _ in results)
            + f"\n\n{sum(1 for o, _, _ in results if o)}/{len(results)} passed")

    sec("Actions")
    for txt, cmd in (("Compute", compute),
                     ("Auto theta range (theta_c -> NA limit)", auto_theta),
                     ("Export current tab CSV", lambda: save("csv")),
                     ("Export current tab XLSX", lambda: save("xlsx")),
                     ("Save current figure tab (300 dpi)", lambda: save("fig")),
                     ("Self-test report", show_tests)):
        ttk.Button(ctl, text=txt, command=cmd).pack(fill="x", padx=8, pady=1)

    n_pass = sum(1 for o, _, _ in results if o)
    ttk.Label(ctl, text=f"self-test: {n_pass}/{len(results)} passed"
                        + ("" if ok else "   <-- DO NOT TRUST OUTPUT"),
              foreground="#0b6b3a" if ok else "#8a1c1c").pack(anchor="w", padx=8, pady=(4, 12))

    for w in (v_m1, v_m2, v_conv):
        w.trace_add("write", lambda *_: compute())
    root.bind("<Return>", compute)

    compute()
    root.mainloop()


# =========================================================================
#  8.  CLI
# =========================================================================

def main():
    if "--selftest" in sys.argv:
        ok, _ = run_selftests()
        sys.exit(0 if ok else 1)

    if "--preview" in sys.argv:
        ok, _ = run_selftests()
        out = sys.argv[sys.argv.index("--preview") + 1]
        p = dict(DEFAULTS)
        M = Model(p)
        pr = M.probe(p["p_th"], p["p_lam"])
        fig = plt.figure(figsize=(15.0, 8.6))
        draw_figure(fig, M, pr)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        u = "d_I" if p["conv"] == "intensity" else "d_E"
        print(f"\nprobe({pr['th']:.0f} deg, {pr['lam']:.0f} nm): n1={pr['n1']:.5f} "
              f"n2={pr['n2']:.5f} theta_c={pr['thc']:.3f} deg")
        print(f"  {u} = {pr['d']:.2f} nm   FLOOR = {pr['floor']:.2f} nm")
        print(f"  objective NA={p['NA']}: theta_max={pr['th_NA']:.2f} deg, "
              f"d={pr['d_NA']:.2f} nm, BFP annulus={100 * pr['annulus']:.1f}%")
        print(f"  prism A={p['prismA']} deg: external phi = {pr['phi_ext']:+.3f} deg, "
              f"0.1 deg -> {100 * pr['dd_rel_0p1deg']:.2f}% depth error")
        print(f"  stray f={p['stray_f']}: {100 * pr['E_stray']:.0f}% of the axial excitation "
              f"is non-evanescent; dominates beyond {pr['z_cross']:.0f} nm")
        print(f"preview -> {out}")
        sys.exit(0 if ok else 1)

    launch_gui()


if __name__ == "__main__":
    main()
