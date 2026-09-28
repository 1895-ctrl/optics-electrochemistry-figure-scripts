#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CW-ODMR fluorescence simulator for negatively charged NV centers.

The program calculates fluorescence as a function of microwave frequency and
static magnetic field. It displays a two-dimensional PL(B, f) map and a line
cut and can export the numerical results to XLSX and JSON.

Physical model
--------------
In the local NV frame, with z parallel to the NV axis, the ground-state
spin Hamiltonian (in Hz) is

    H_gs/h = D[Sz^2 - S(S+1)/3]
           + Ex(Sx^2 - Sy^2) + Ey(Sx Sy + Sy Sx)
           + gamma_e(B . S)
           + A_par Sz Iz + A_perp(Sx Ix + Sy Iy)
           + P[Iz^2 - I(I+1)/3] - gamma_n(B . I).

Here gamma_e = g_e mu_B/h (28.03 GHz/T for the default g_e = 2.0028),
and (Ex, Ey) = E(cos(2 phi_E), sin(2 phi_E)). The room-temperature,
orbitally averaged excited state is approximated as S = 1 with D_es about
1.42 GHz. Excited-state hyperfine structure is omitted.

The two fluorescence models use a dimensionless microwave drive weight Wd,
normalized so an ideal transverse Delta-m_s = +/-1 transition has Wd = 1:

    single linear polarization: Wd_ij = |<i|n_mw . S|j>|^2 / 0.5
    transverse average:         Wd_ij = (|<i|Sx|j>|^2 + |<i|Sy|j>|^2)
                                    / (2 * 0.5)
    Rabi frequency:              Omega_ij = 2 pi gamma_e B1 sqrt(Wd_ij/2).

At zero field, strain splits the |X> and |Y> combinations of m_s = +/-1.
A single linear polarization drives only one combination; the default
transverse average gives the two lines equal weight.

Model A is a fast, weak-drive phenomenological Lorentzian model. It exactly
diagonalizes H_gs at each B. If p_i = <i|P_0|i> is the m_s = 0 projection,

    PL(f) = 1 - sum_(i<j) A_ij L(f; f_ij, Gamma),
    A_ij = C_max (p_i - p_j)^2 Wd_ij,
    f_ij = E_j - E_i,

where L has FWHM Gamma. Optical-pumping populations and brightness both
scale with p_i in the weak-drive limit, so the squared projection difference
captures contrast loss caused by ground-state mixing. C_max is the contrast
of one isolated, ideally driven line. At B -> 0 and E -> 0, two overlapping
lines can reach about 2 C_max; Model A does not impose power saturation.

Model B is a slower seven-level rate-equation model (multiplied by the number
of nuclear-spin states). It includes ground and excited triplet eigenstates
and a spin-conserving singlet shelf. Optical pumping and radiative decay use
spin overlap; intersystem crossing and shelf decay are spin dependent.
Microwave transition rates use Omega_ij and G2 = pi Gamma. The steady-state
population solves M P = 0 with sum(P) = 1. Fluorescence is proportional to
k_r times total excited-state population and is normalized to the
microwave-off baseline. Rate sources: Robledo et al., New J. Phys. 13,
025013 (2011); Tetienne et al., New J. Phys. 14, 103033 (2012).

For geometry='single', B and microwave polarization are specified in one
NV-local frame, with theta measured from the NV axis. For 'ensemble', they
are specified in the crystal frame and averaged equally over the four <111>
orientations, typically producing eight resonance lines.

Limitations and calibration
---------------------------
* Model A is valid in the weak-microwave limit. Its linear addition of
  overlapping lines can overestimate contrast at high drive; use Model B
  when saturation or power broadening matters.
* Excited-state hyperfine structure is omitted, so nuclear polarization
  near the ESLAC (about 50 mT) and line weights around 51.2 mT are not
  described correctly. 13C hyperfine structure and NV-/NV0 photoionization
  are also omitted.
* Model B defaults were inferred from NV J at 300 K in Robledo et al.
  Individual centers may differ by roughly 30%; beta_pump and B1 are
  instrument dependent and must be calibrated for quantitative comparison.
* The same local strain-axis angle phi_E is used for every orientation.
  The temperature correction to D is linearized near 300 K using
  dD/dT = -74.2 kHz/K.

Dependencies: NumPy, Matplotlib, pandas, openpyxl, and Tkinter. Figure
labels use ASCII and Unicode minus is disabled to avoid missing-glyph
warnings in environments without suitable fonts.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from dataclasses import dataclass, asdict, fields, replace
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["axes.unicode_minus"] = False   # Avoid missing U+2212 glyphs
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # Figure labels use ASCII
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# =============================================================================
#  Physical constants
# =============================================================================
MU_B_OVER_H = 13.996_244_936e9      # Hz/T   (Bohr magneton / h)
G_E_DEFAULT = 2.0028                # NV- ground-state g factor
D_GS_300K   = 2.8700e9              # Hz     zero-field splitting at 300 K
DD_DT       = -74.2e3               # Hz/K   dD/dT (linear approximation near 300 K)
D_ES_DEF    = 1.420e9               # Hz     excited-state zero-field splitting (room-temperature orbital average)

# Nuclear-spin parameters (Hz). Defaults from Felton et al., PRB 79, 075203 (2009), Table II (300 K).
#   14N: A_par = -2.14(7) MHz, A_perp = -2.70(7) MHz, P = -5.01(6) MHz
#   15N: A_par = +3.03(3) MHz, A_perp = +3.65(3) MHz, P = 0 (I = 1/2)
#   gamma_n = g_N * mu_N/h ; g_N(14N) = 0.4038, g_N(15N) = -0.5664 (CRC Handbook),
#            mu_N/h = 7.6225932 MHz/T  ->  3.078 / -4.317 MHz/T
# Alternative literature values, consistent with the values above within experimental uncertainty:
#   A_par(14N) = -2.162 MHz  (matches the observed 2.16 MHz ODMR triplet spacing)
#   P(14N)     = -4.945 MHz  (Gali, Nanophotonics 8, 1907 (2019) value used by)
# To reproduce the measured 2.16 MHz spacing, set A_par below to -2.162e6.
NUC_TABLE = {
    "none": dict(I=0.0, A_par=0.0,      A_perp=0.0,     P=0.0,       gamma_n=0.0),
    "14N":  dict(I=1.0, A_par=-2.14e6,  A_perp=-2.70e6, P=-5.01e6,   gamma_n=3.078e6),
    "15N":  dict(I=0.5, A_par=+3.03e6,  A_perp=+3.65e6, P=0.0,       gamma_n=-4.317e6),
}

# Four <111> NV orientations (unit vectors in the crystal frame)
NV_AXES = np.array([[1., 1., 1.],
                    [1., -1., -1.],
                    [-1., 1., -1.],
                    [-1., -1., 1.]]) / np.sqrt(3.0)


# =============================================================================
#  Spin operators and geometry
# =============================================================================
def spin_matrices(s: float):
    """Return (Sx, Sy, Sz) for spin s with basis ordered m = s, s-1, ..., -s."""
    dim = int(round(2 * s + 1))
    m = np.array([s - k for k in range(dim)], float)
    sz = np.diag(m).astype(complex)
    sp = np.zeros((dim, dim), complex)
    for k in range(1, dim):
        mk = m[k]
        sp[k - 1, k] = np.sqrt(s * (s + 1) - mk * (mk + 1))
    sm = sp.conj().T
    sx = 0.5 * (sp + sm)
    sy = (sp - sm) / (2j)
    return sx, sy, sz


def sph_to_vec(theta_deg: float, phi_deg: float) -> np.ndarray:
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def nv_local_frames():
    """Return four local NV orthonormal frames as 3x3 matrices with rows (x_loc, y_loc, z_loc).
    Use B_local = R @ B_crystal."""
    frames = []
    for z in NV_AXES:
        ref = np.array([0., 0., 1.])
        if abs(float(np.dot(ref, z))) > 0.9:
            ref = np.array([1., 0., 0.])
        x = np.cross(ref, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        frames.append(np.stack([x, y, z]))
    return frames


class SpinSystem:
    """Electron S=1 operators, optionally tensor-producted with nuclear spin I. Basis: |m_s> x |m_I>,
    m_s = +1, 0, -1; m_I = I, ..., -I. Index = m_s_idx * d_n + m_I_idx."""

    def __init__(self, nuc: str = "none"):
        p = NUC_TABLE[nuc]
        self.nuc = nuc
        self.I = p["I"]
        self.A_par, self.A_perp = p["A_par"], p["A_perp"]
        self.P, self.gamma_n = p["P"], p["gamma_n"]

        self.dn = int(round(2 * self.I + 1))
        self.dim = 3 * self.dn

        Sx, Sy, Sz = spin_matrices(1.0)
        Ix, Iy, Iz = spin_matrices(self.I)
        In, I3 = np.eye(self.dn), np.eye(3)

        self.SX, self.SY, self.SZ = (np.kron(Sx, In), np.kron(Sy, In), np.kron(Sz, In))
        self.IX, self.IY, self.IZ = (np.kron(I3, Ix), np.kron(I3, Iy), np.kron(I3, Iz))
        self.EYE = np.eye(self.dim)

        # m_s projection operators (indices 0, 1, 2 correspond to m_s = +1, 0, -1)
        self.Pms = [np.kron(np.diag([1., 0., 0.]), In),
                    np.kron(np.diag([0., 1., 0.]), In),
                    np.kron(np.diag([0., 0., 1.]), In)]


def _h_spin1(ss: SpinSystem, D, Ex, Ey, gamma_e, B):
    """Electron S=1 terms (D, strain, Zeeman) extended to the full spin space."""
    H = D * (ss.SZ @ ss.SZ - (2.0 / 3.0) * ss.EYE)
    if Ex != 0.0 or Ey != 0.0:
        H = H + Ex * (ss.SX @ ss.SX - ss.SY @ ss.SY) + Ey * (ss.SX @ ss.SY + ss.SY @ ss.SX)
    H = H + gamma_e * (B[0] * ss.SX + B[1] * ss.SY + B[2] * ss.SZ)
    return H


def hamiltonian_gs(ss: SpinSystem, D, Ex, Ey, gamma_e, B):
    """Ground-state Hamiltonian (Hz); B is in the local frame in tesla."""
    H = _h_spin1(ss, D, Ex, Ey, gamma_e, B)
    if ss.dn > 1:
        H = H + ss.A_par * (ss.SZ @ ss.IZ) + ss.A_perp * (ss.SX @ ss.IX + ss.SY @ ss.IY)
        if abs(ss.I - 1.0) < 1e-9:
            H = H + ss.P * (ss.IZ @ ss.IZ - (2.0 / 3.0) * ss.EYE)
        H = H - ss.gamma_n * (B[0] * ss.IX + B[1] * ss.IY + B[2] * ss.IZ)
    return H


def hamiltonian_es(ss: SpinSystem, D_es, Ex, Ey, gamma_e, B):
    """Excited-state Hamiltonian (Hz), omitting hyperfine terms as noted above."""
    return _h_spin1(ss, D_es, Ex, Ey, gamma_e, B)


# =============================================================================
#  Parameters
# =============================================================================
@dataclass
class Params:
    # --- Magnetic-field sweep ---
    B_min: float = 0.0          # mT
    B_max: float = 10.0         # mT
    n_B: int = 121
    theta_B: float = 0.0        # deg
    phi_B: float = 0.0          # deg
    geometry: str = "single"    # "single" | "ensemble"

    # --- Microwave-frequency sweep ---
    auto_f: bool = True
    f_min: float = 2.60         # GHz
    f_max: float = 3.14         # GHz
    n_f: int = 601

    # --- Spin Hamiltonian ---
    D_gs: float = 2.8700        # GHz
    E_gs: float = 0.0           # MHz
    phi_E: float = 0.0          # deg
    g_e: float = G_E_DEFAULT
    use_T: bool = False
    temperature: float = 300.0  # K
    nuc: str = "none"           # "none" | "14N" | "15N"

    # --- Line shape and microwave drive ---
    gamma_fwhm: float = 5.0     # MHz
    mw_ideal_perp: bool = True
    theta_mw: float = 90.0      # deg
    phi_mw: float = 0.0         # deg

    # --- Model ---
    model: str = "A"            # "A" | "B"
    contrast: float = 20.0      # %   (Model A)

    # --- Model B photophysical rates (MHz) ---
    # Inferred from Robledo et al., New J. Phys. 13, 025013 (2011), Table 1 (NV J, 300 K):
    #   T1|3> = 13.26 ns, T1|4> = 6.89 ns, T1|5> = 178 ns, p35 = 0.14, p45 = 0.55,
    #   p51/p52 = 1.15   ->   inferred from the cited rate data
    D_es: float = 1.420         # GHz  excited-state ZFS (Fuchs 2008 / Neumann 2009: 1.42-1.43)
    E_es: float = 0.0           # MHz  excited-state transverse strain
    B1: float = 0.05            # mT   microwave-field amplitude (instrument dependent; calibrate)
    beta_pump: float = 5.0      # MHz  optical-pumping rate (laser-power dependent; calibrate)
    k_r: float = 64.9           # MHz  excited-state radiative decay = (1 - p35)/T1|3>
    k_isc0: float = 10.6        # MHz  ES(m_s=0) to singlet = p35/T1|3>
    k_isc1: float = 79.8        # MHz  ES(m_s=+/-1) to singlet = p45/T1|4>
    k_s0: float = 3.00          # MHz  singlet to GS(m_s=0)
    k_s1: float = 1.31          # MHz  singlet to each GS(m_s=+/-1) sublevel


# =============================================================================
#  Microwave drive weights shared by both models
# =============================================================================
def drive_weights(ss: SpinSystem, U, n_mw, avg: bool):
    """Return dimensionless drive weights Wd[i,j], normalized so an aligned field with ideal transverse drive has
    Delta-m_s = +/-1 transition weight equal to one.

    avg=True : average over transverse linear-polarization azimuth -> Wd = (|Mx|^2 + |My|^2)/(2 * 0.5)
               (represents uncontrolled transverse microwave polarization and gives the zero-field strain-split
                |X> and |Y> lines equal weight; one linear polarization drives only one)
    avg=False: single linear polarization n_mw    -> Wd = |<i|n.S|j>|^2 / 0.5

    In either case, the Rabi frequency is Omega = 2*pi*gamma_e*B1*sqrt(Wd/2).
    """
    if avg:
        Mx = U.conj().T @ ss.SX @ U
        My = U.conj().T @ ss.SY @ U
        return (np.abs(Mx) ** 2 + np.abs(My) ** 2) / 1.0
    S = n_mw[0] * ss.SX + n_mw[1] * ss.SY + n_mw[2] * ss.SZ
    M = U.conj().T @ S @ U
    return np.abs(M) ** 2 / 0.5


# =============================================================================
#  Model A: phenomenological Lorentzians
# =============================================================================
def spectrum_A(ss, ev, U, n_mw, mw_avg, f_grid, gamma_fwhm, contrast, tab_thresh=0.02):
    P0 = ss.Pms[1]                                   # m_s = 0 projection
    pop = np.real(np.einsum('ji,jk,ki->i', U.conj(), P0, U))          # p_i

    Wme = drive_weights(ss, U, n_mw, mw_avg)

    dp = pop[:, None] - pop[None, :]
    amp2d = contrast * (dp ** 2) * Wme               # contrast is a fraction, not a percentage
    f2d = ev[None, :] - ev[:, None]                  # f_ij = E_j - E_i (i<j -> >0)

    iu = np.triu_indices(ss.dim, 1)
    fr, amp = f2d[iu], amp2d[iu]

    G = gamma_fwhm
    keep = (amp > 1e-6) & (fr > f_grid[0] - 20 * G) & (fr < f_grid[-1] + 20 * G)
    fr, amp = fr[keep], amp[keep]

    pl = np.ones_like(f_grid)
    if fr.size:
        L = (0.5 * G) ** 2 / ((f_grid[:, None] - fr[None, :]) ** 2 + (0.5 * G) ** 2)
        pl = pl - (L * amp[None, :]).sum(axis=1)
    np.clip(pl, 0.0, None, out=pl)

    if fr.size:
        sel = amp > tab_thresh * max(amp.max(), 1e-12)
        table = list(zip(fr[sel], amp[sel]))
    else:
        table = []
    return pl, table


# =============================================================================
#  Model B: seven-level rate equations per nuclear-spin state
# =============================================================================
def spectrum_B(ss, eg, Ug, ee, Ue, n_mw, mw_avg, f_grid, p: Params, gamma_e,
               tab_thresh=0.02):
    dn, ng = ss.dn, ss.dim
    ne, nsg = ss.dim, dn
    N = ng + ne + nsg

    # --- State overlap (orbital transitions conserve spin) ---
    O = np.abs(Ue.conj().T @ Ug) ** 2                       # O[a,i] = |<e_a|g_i>|^2

    # --- |<m,mu|.>|^2 decomposition ---
    Wg3 = (np.abs(Ug) ** 2).reshape(3, dn, ng)              # [m, mu, i]
    We3 = (np.abs(Ue) ** 2).reshape(3, dn, ne)              # [m, mu, a]

    kisc = np.array([p.k_isc1, p.k_isc0, p.k_isc1])         # m_s = +1, 0, -1
    ks   = np.array([p.k_s1,  p.k_s0,  p.k_s1])

    rate_es = np.einsum('mua,m->au', We3, kisc)             # [a, mu]  ES -> singlet
    rate_sg = np.einsum('mui,m->ui', Wg3, ks)               # [mu, i]  singlet -> GS

    # --- Rate matrix without microwave drive (MHz) ---
    M0 = np.zeros((N, N))
    M0[ng:ng + ne, 0:ng] = p.beta_pump * O                  # pumping
    M0[0:ng, ng:ng + ne] = p.k_r * O.T                      # radiative decay
    M0[ng + ne:, ng:ng + ne] = rate_es.T                    # ISC
    M0[0:ng, ng + ne:] = rate_sg.T                          # singlet relaxation
    np.fill_diagonal(M0, -M0.sum(axis=0))                   # column sums are zero (population conservation)

    # --- Microwave transition pairs ---
    Wd = drive_weights(ss, Ug, n_mw, mw_avg)                # dimensionless
    om2_2d = (2 * np.pi * gamma_e * (p.B1 * 1e-3)) ** 2 * Wd / 2.0   # (rad/s)^2
    f2d = eg[None, :] - eg[:, None]                         # Hz
    iu = np.triu_indices(ng, 1)
    fr, om2, wme = f2d[iu], om2_2d[iu], Wd[iu]
    G = p.gamma_fwhm * 1e6                                  # Hz (FWHM)
    G2 = np.pi * G                                          # rad/s (dephasing rate)
    keep = (om2 > 0) & (fr > f_grid[0] - 30 * G) & (fr < f_grid[-1] + 30 * G)
    ii, jj = iu[0][keep], iu[1][keep]
    fr, om2, wme = fr[keep], om2[keep], wme[keep]

    nf = f_grid.size
    Mb = np.repeat(M0[None, :, :], nf, axis=0)
    for k in range(fr.size):
        dw = 2 * np.pi * (f_grid - fr[k])
        Wf = 0.5 * om2[k] * G2 / (G2 ** 2 + dw ** 2) / 1e6  # s^-1 -> MHz
        i, j = int(ii[k]), int(jj[k])
        Mb[:, i, j] += Wf
        Mb[:, j, i] += Wf
        Mb[:, i, i] -= Wf
        Mb[:, j, j] -= Wf

    # --- Steady state: replace the final row with normalization ---
    Mb[:, -1, :] = 1.0
    b = np.zeros((nf, N, 1))
    b[:, -1, 0] = 1.0
    Pss = np.linalg.solve(Mb, b)[:, :, 0]
    pl_raw = p.k_r * Pss[:, ng:ng + ne].sum(axis=1)

    M0n = M0.copy()
    M0n[-1, :] = 1.0
    b0 = np.zeros(N)
    b0[-1] = 1.0
    P0ss = np.linalg.solve(M0n, b0)
    base = p.k_r * float(P0ss[ng:ng + ne].sum())

    if wme.size:
        sel = wme > tab_thresh * max(wme.max(), 1e-12)
        table = list(zip(fr[sel], wme[sel]))
    else:
        table = []
    return pl_raw, base, table


# =============================================================================
#  Main calculation
# =============================================================================
def auto_frange(p: Params) -> tuple[float, float]:
    gamma_e = p.g_e * MU_B_OVER_H
    Bmax = max(abs(p.B_min), abs(p.B_max)) * 1e-3          # T
    D = p.D_gs * 1e9
    span = gamma_e * Bmax + 2 * abs(p.E_gs) * 1e6 + 8 * p.gamma_fwhm * 1e6 + 30e6
    return max(0.0, D - span) / 1e9, (D + span) / 1e9


def simulate(p: Params, progress=None, cancel=None) -> dict:
    gamma_e = p.g_e * MU_B_OVER_H                          # Hz/T
    D = p.D_gs * 1e9 + (DD_DT * (p.temperature - 300.0) if p.use_T else 0.0)
    E = p.E_gs * 1e6
    c2, s2 = np.cos(2 * np.deg2rad(p.phi_E)), np.sin(2 * np.deg2rad(p.phi_E))
    Ex, Ey = E * c2, E * s2
    Ees = p.E_es * 1e6
    Exe, Eye = Ees * c2, Ees * s2
    D_es = p.D_es * 1e9

    ss = SpinSystem(p.nuc)
    B_mT = np.linspace(p.B_min, p.B_max, p.n_B)
    B_T = B_mT * 1e-3

    f_lo, f_hi = auto_frange(p) if p.auto_f else (p.f_min, p.f_max)
    f_GHz = np.linspace(f_lo, f_hi, p.n_f)
    f_Hz = f_GHz * 1e9

    frames = [np.eye(3)] if p.geometry == "single" else nv_local_frames()
    nfam = len(frames)

    b_hat = sph_to_vec(p.theta_B, p.phi_B)
    mw_hat = sph_to_vec(p.theta_mw, p.phi_mw)

    PL = np.zeros((p.n_B, p.n_f))
    PL_raw = np.zeros((p.n_B, p.n_f)) if p.model == "B" else None
    base = np.zeros(p.n_B) if p.model == "B" else None
    rows = []

    total = nfam * p.n_B
    done = 0
    for k, R in enumerate(frames):
        if p.mw_ideal_perp:
            n_mw = np.array([1.0, 0.0, 0.0])               # local frame, perpendicular to the NV axis
        else:
            v = R @ mw_hat
            nrm = np.linalg.norm(v)
            n_mw = v / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0])

        for ib in range(p.n_B):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cancelled by user")
            Bloc = R @ (B_T[ib] * b_hat)

            Hg = hamiltonian_gs(ss, D, Ex, Ey, gamma_e, Bloc)
            eg, Ug = np.linalg.eigh(Hg)

            if p.model == "A":
                pl, tab = spectrum_A(ss, eg, Ug, n_mw, p.mw_ideal_perp, f_Hz,
                                     p.gamma_fwhm * 1e6, p.contrast / 100.0)
                PL[ib] += pl / nfam
            else:
                He = hamiltonian_es(ss, D_es, Exe, Eye, gamma_e, Bloc)
                ee, Ue = np.linalg.eigh(He)
                raw, b0, tab = spectrum_B(ss, eg, Ug, ee, Ue, n_mw, p.mw_ideal_perp,
                                          f_Hz, p, gamma_e)
                PL_raw[ib] += raw / nfam
                base[ib] += b0 / nfam

            for fr, w in tab:
                rows.append((k + 1, B_mT[ib], fr / 1e9, float(w)))

            done += 1
            if progress is not None and (done % 5 == 0 or done == total):
                progress(done / total)

    if p.model == "B":
        PL = PL_raw / base[:, None]

    return dict(B_mT=B_mT, f_GHz=f_GHz, PL=PL, PL_raw=PL_raw, baseline=base,
                res=np.array(rows, float) if rows else np.zeros((0, 4)),
                model=p.model, n_family=nfam)


def simulate_linecut(p: Params, B_mT: float, f_GHz: np.ndarray) -> dict:
    """Recompute a line spectrum at an exact field on the existing frequency grid."""
    f_GHz = np.asarray(f_GHz, float)
    if f_GHz.ndim != 1 or f_GHz.size < 2:
        raise ValueError("The line-spectrum frequency grid must be one-dimensional with at least two points.")
    p_cut = replace(p, B_min=float(B_mT), B_max=float(B_mT), n_B=1,
                    auto_f=False, f_min=float(f_GHz[0]), f_max=float(f_GHz[-1]),
                    n_f=int(f_GHz.size))
    r = simulate(p_cut)
    return {
        "B_mT": float(B_mT),
        "PL_norm": r["PL"][0].copy(),
        "PL_raw": None if r["PL_raw"] is None else r["PL_raw"][0].copy(),
    }


# =============================================================================
#  XLSX export
# =============================================================================
XL_MAX_COL, XL_MAX_ROW = 16384, 1_048_576


def _selected_linecut(res: dict, cut_idx: int, linecut: dict | None = None) -> dict:
    """Use one export structure for slider-selected and exact manually entered line spectra."""
    if linecut is not None:
        return {
            "B_mT": float(linecut["B_mT"]),
            "PL_norm": np.asarray(linecut["PL_norm"], float),
            "PL_raw": None if linecut.get("PL_raw") is None
            else np.asarray(linecut["PL_raw"], float),
        }
    i = int(np.clip(cut_idx, 0, len(res["B_mT"]) - 1))
    return {
        "B_mT": float(res["B_mT"][i]),
        "PL_norm": np.asarray(res["PL"][i], float),
        "PL_raw": None if res["PL_raw"] is None
        else np.asarray(res["PL_raw"][i], float),
    }


def write_xlsx(path: str, res: dict, p: Params, cut_idx: int,
               linecut: dict | None = None, generated: str | None = None):
    import pandas as pd

    B, f, PL = res["B_mT"], res["f_GHz"], res["PL"]
    nB, nf = len(B), len(f)
    cut = _selected_linecut(res, cut_idx, linecut)
    generated = generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info = [("generated", generated),
            ("script", "nv_odmr_gui.py"),
            ("model", "A: phenomenological Lorentzian" if p.model == "A"
                      else "B: 7-level rate equations"),
            ("gamma_e_GHz_per_T", p.g_e * MU_B_OVER_H / 1e9),
            ("n_NV_families", res["n_family"]),
            ("PL_normalization", "off-resonance baseline = 1")]
    info += [(fl.name, getattr(p, fl.name)) for fl in fields(p)]
    df_info = pd.DataFrame(info, columns=["parameter", "value"])
    df_info["value"] = df_info["value"].astype(str)

    df_cut = pd.DataFrame({"f_GHz": f, "PL_norm": cut["PL_norm"]})
    if cut["PL_raw"] is not None:
        df_cut["PL_raw_arb"] = cut["PL_raw"]
    df_cut.insert(0, "B_mT", cut["B_mT"])

    R = res["res"]
    df_res = pd.DataFrame(R, columns=["NV_family", "B_mT", "f_res_GHz", "weight"])

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df_info.to_excel(xw, sheet_name="Info", index=False)

        if nf + 1 <= XL_MAX_COL:
            cols = [f"{x:.9g}" for x in f]
            pd.DataFrame(PL, index=pd.Index(B, name="B_mT / f_GHz->"),
                         columns=cols).to_excel(xw, sheet_name="PL_map")
            if res["PL_raw"] is not None:
                pd.DataFrame(res["PL_raw"], index=pd.Index(B, name="B_mT / f_GHz->"),
                             columns=cols).to_excel(xw, sheet_name="PL_raw_map")
        else:  # too many frequency points: use a long table
            BB, FF = np.meshgrid(B, f, indexing="ij")
            long = pd.DataFrame({"B_mT": BB.ravel(), "f_GHz": FF.ravel(),
                                 "PL_norm": PL.ravel()})
            if len(long) > XL_MAX_ROW - 2:
                long = long.iloc[: XL_MAX_ROW - 2]
            long.to_excel(xw, sheet_name="PL_long", index=False)

        df_cut.to_excel(xw, sheet_name="Linecut", index=False)
        if res["baseline"] is not None:
            pd.DataFrame({"B_mT": B, "PL_baseline_arb": res["baseline"]}
                         ).to_excel(xw, sheet_name="Baseline", index=False)
        if len(df_res) <= XL_MAX_ROW - 2:
            df_res.to_excel(xw, sheet_name="Resonances", index=False)


def write_json(path: str, res: dict, p: Params, cut_idx: int,
               linecut: dict | None = None, generated: str | None = None):
    """Export parameters, the 2D map, current line spectrum, and resonance data together."""
    cut = _selected_linecut(res, cut_idx, linecut)
    payload = {
        "schema_version": 1,
        "generated": generated or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "nv_odmr_gui.py",
        "model": res["model"],
        "n_NV_families": int(res["n_family"]),
        "PL_normalization": "off-resonance baseline = 1",
        "parameters": asdict(p),
        "result": {
            "B_mT": np.asarray(res["B_mT"]).tolist(),
            "f_GHz": np.asarray(res["f_GHz"]).tolist(),
            "PL_norm": np.asarray(res["PL"]).tolist(),
            "PL_raw": None if res["PL_raw"] is None
            else np.asarray(res["PL_raw"]).tolist(),
            "baseline": None if res["baseline"] is None
            else np.asarray(res["baseline"]).tolist(),
            "resonances": np.asarray(res["res"]).tolist(),
        },
        "linecut": {
            "B_mT": cut["B_mT"],
            "f_GHz": np.asarray(res["f_GHz"]).tolist(),
            "PL_norm": cut["PL_norm"].tolist(),
            "PL_raw": None if cut["PL_raw"] is None else cut["PL_raw"].tolist(),
        },
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))


# =============================================================================
#  GUI
# =============================================================================
class ScrollFrame(ttk.Frame):
    def __init__(self, master, width=380, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(ev)

    def _wheel(self, e):
        d = -1 if getattr(e, "num", 0) == 4 else (1 if getattr(e, "num", 0) == 5
                                                  else int(-e.delta / 120))
        self.canvas.yview_scroll(d, "units")


class NVApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.result = None
        self.params_used = None
        self.manual_linecut = None
        self.cancel = threading.Event()
        self._q: queue.Queue | None = None

        self.vars: dict[str, tk.Variable] = {}
        self._build_vars()
        self._build_layout()
        self._trace_all()
        self._toggle_model()
        self._toggle_manual_B(redraw=False)
        self.set_status("Not calculated", "stale")

    # ---------------- Variables ----------------
    def _build_vars(self):
        d = Params()
        for fl in fields(d):
            v = getattr(d, fl.name)
            if isinstance(v, bool):
                self.vars[fl.name] = tk.BooleanVar(value=v)
            elif isinstance(v, int) and not isinstance(v, bool):
                self.vars[fl.name] = tk.IntVar(value=v)
            elif isinstance(v, float):
                self.vars[fl.name] = tk.DoubleVar(value=v)
            else:
                self.vars[fl.name] = tk.StringVar(value=v)
        self.v_cmap = tk.StringVar(value="viridis")
        self.v_overlay = tk.BooleanVar(value=False)
        self.v_cut = tk.IntVar(value=0)
        self.v_manual_B = tk.BooleanVar(value=False)
        self.v_manual_B_value = tk.StringVar(value="0.0")

    def _trace_all(self):
        for name, v in self.vars.items():
            v.trace_add("write", lambda *_a, n=name: self._on_param_change(n))

    def _on_param_change(self, name):
        if name in ("model",):
            self._toggle_model()
        if self.result is not None:
            self.set_status("Parameters changed; recalculate to update the result", "stale")

    # ---------------- Layout ----------------
    def _build_layout(self):
        left = ScrollFrame(self, width=390)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        P = left.inner
        self._sec_sweep(P)
        self._sec_mw(P)
        self._sec_ham(P)
        self._sec_line(P)
        self._sec_model(P)
        self._sec_actions(P)

        # ---- Right side: status bar and plots ----
        self.lbl_status = tk.Label(right, text="", anchor="w", padx=8, pady=4)
        self.lbl_status.pack(side="top", fill="x")

        self.fig = Figure(figsize=(7.6, 6.4), dpi=100)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.46,
                                   left=0.11, right=0.98, top=0.95, bottom=0.09)
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_cut = self.fig.add_subplot(gs[1])
        # A figure-level dashed line separates the 2D map from the slider-controlled line spectrum.
        map_box = self.ax_map.get_position()
        cut_box = self.ax_cut.get_position()
        sep_y = 0.5 * (map_box.y0 + cut_box.y1)
        self.panel_separator = Line2D(
            [min(map_box.x0, cut_box.x0), max(map_box.x1, cut_box.x1)],
            [sep_y, sep_y],
            transform=self.fig.transFigure,
            color="0.45",
            linewidth=1.0,
            linestyle=(0, (5, 4)),
            alpha=0.85,
            clip_on=False,
        )
        self.fig.add_artist(self.panel_separator)
        self.cbar = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

        bar = ttk.Frame(right)
        bar.pack(side="top", fill="x", padx=8, pady=(0, 6))
        ttk.Label(bar, text="Line spectrum at B index:").pack(side="left")
        self.sld = ttk.Scale(bar, from_=0, to=0, orient="horizontal",
                             command=self._on_slider)
        self.sld.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_cut = ttk.Label(bar, text="--", width=16)
        self.lbl_cut.pack(side="left")

        manual = ttk.Frame(right)
        manual.pack(side="top", fill="x", padx=8, pady=(0, 6))
        ttk.Checkbutton(manual, text="Enter an exact B field", variable=self.v_manual_B,
                        command=self._toggle_manual_B).pack(side="left")
        self.ent_manual_B = ttk.Entry(manual, textvariable=self.v_manual_B_value,
                                      width=12)
        self.ent_manual_B.pack(side="left", padx=(8, 4))
        self.ent_manual_B.bind("<Return>", lambda _e: self._apply_manual_B())
        ttk.Label(manual, text="mT").pack(side="left")
        self.btn_manual_B = ttk.Button(manual, text="Apply",
                                       command=self._apply_manual_B, width=7)
        self.btn_manual_B.pack(side="left", padx=(8, 8))
        self.lbl_manual_range = ttk.Label(
            manual, text="Disables the slider; the range follows the current B sweep", foreground="#666")
        self.lbl_manual_range.pack(side="left")

        self._blank_axes()

    def _grid(self, parent, title):
        lf = ttk.LabelFrame(parent, text=title)
        lf.pack(fill="x", padx=8, pady=5)
        lf.columnconfigure(1, weight=1)
        return lf

    def _row(self, lf, r, label, key, unit=""):
        ttk.Label(lf, text=label).grid(row=r, column=0, sticky="w", padx=(6, 4), pady=2)
        ttk.Entry(lf, textvariable=self.vars[key], width=11).grid(row=r, column=1,
                                                                  sticky="ew", pady=2)
        ttk.Label(lf, text=unit, width=7).grid(row=r, column=2, sticky="w", padx=(4, 6))

    def _sec_sweep(self, P):
        lf = self._grid(P, "1. Static magnetic-field sweep")
        self._row(lf, 0, "B start", "B_min", "mT")
        self._row(lf, 1, "B end", "B_max", "mT")
        self._row(lf, 2, "Points N_B", "n_B", "")
        self._row(lf, 3, "Polar angle theta_B", "theta_B", "deg")
        self._row(lf, 4, "Azimuth phi_B", "phi_B", "deg")
        ttk.Label(lf, text="Geometry").grid(row=5, column=0, sticky="w", padx=(6, 4))
        cb = ttk.Combobox(lf, textvariable=self.vars["geometry"], state="readonly",
                          width=9, values=["single", "ensemble"])
        cb.grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Label(lf, text="(4x<111>)", width=9).grid(row=5, column=2, sticky="w")
        ttk.Label(lf, text="single: angles relative to the NV axis\nensemble: angles relative to crystal [001]/[100]",
                  foreground="#666").grid(row=6, column=0, columnspan=3,
                                          sticky="w", padx=6, pady=(2, 4))

    def _sec_mw(self, P):
        lf = self._grid(P, "2. Microwave-frequency sweep")
        ttk.Checkbutton(lf, text="Determine frequency range automatically", variable=self.vars["auto_f"]
                        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6)
        self._row(lf, 1, "f start", "f_min", "GHz")
        self._row(lf, 2, "f end", "f_max", "GHz")
        self._row(lf, 3, "Points N_f", "n_f", "")

    def _sec_ham(self, P):
        lf = self._grid(P, "3. Spin Hamiltonian")
        self._row(lf, 0, "D (ground state)", "D_gs", "GHz")
        self._row(lf, 1, "E (strain)", "E_gs", "MHz")
        self._row(lf, 2, "Strain azimuth phi_E", "phi_E", "deg")
        self._row(lf, 3, "g factor", "g_e", "")
        ttk.Checkbutton(lf, text="Correct D for temperature (dD/dT = -74.2 kHz/K)",
                        variable=self.vars["use_T"]
                        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=6)
        self._row(lf, 5, "Temperature T", "temperature", "K")
        ttk.Label(lf, text="Hyperfine").grid(row=6, column=0, sticky="w", padx=(6, 4))
        ttk.Combobox(lf, textvariable=self.vars["nuc"], state="readonly", width=9,
                     values=["none", "14N", "15N"]).grid(row=6, column=1,
                                                         sticky="ew", pady=2)

    def _sec_line(self, P):
        lf = self._grid(P, "4. Line shape and microwave direction")
        self._row(lf, 0, "Linewidth FWHM", "gamma_fwhm", "MHz")
        ttk.Checkbutton(lf, text="Ideal transverse drive (B1 perpendicular to NV; azimuth averaged)",
                        variable=self.vars["mw_ideal_perp"]
                        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=6)
        ttk.Label(lf, text="Uncheck to specify one linear-polarization direction:",
                  foreground="#666").grid(row=2, column=0, columnspan=3,
                                          sticky="w", padx=6)
        self._row(lf, 3, "theta_mw", "theta_mw", "deg")
        self._row(lf, 4, "phi_mw", "phi_mw", "deg")

    def _sec_model(self, P):
        lf = self._grid(P, "5. Fluorescence model")
        ttk.Radiobutton(lf, text="A  phenomenological Lorentzians (fast)", value="A",
                        variable=self.vars["model"]).grid(row=0, column=0, columnspan=3,
                                                          sticky="w", padx=6)
        ttk.Radiobutton(lf, text="B  seven-level rate equations (slower, includes baseline quenching)", value="B",
                        variable=self.vars["model"]).grid(row=1, column=0, columnspan=3,
                                                          sticky="w", padx=6)
        self._row(lf, 2, "Maximum contrast C", "contrast", "%")
        self.frm_B = ttk.LabelFrame(P, text="5b. Model B photophysical rates (Robledo NJP 2011, NV J)")
        self.frm_B.pack(fill="x", padx=8, pady=5)
        self.frm_B.columnconfigure(1, weight=1)
        b = self.frm_B
        self._row(b, 0, "D_es", "D_es", "GHz")
        self._row(b, 1, "E_es", "E_es", "MHz")
        self._row(b, 2, "Microwave amplitude B1", "B1", "mT")
        self._row(b, 3, "Pumping rate beta", "beta_pump", "MHz")
        self._row(b, 4, "Radiative rate k_r", "k_r", "MHz")
        self._row(b, 5, "ISC  k(ms=0)", "k_isc0", "MHz")
        self._row(b, 6, "ISC  k(ms=+-1)", "k_isc1", "MHz")
        self._row(b, 7, "Singlet k->ms=0", "k_s0", "MHz")
        self._row(b, 8, "Singlet k->ms=+/-1", "k_s1", "MHz")

    def _sec_actions(self, P):
        lf = self._grid(P, "6. Plotting and actions")
        ttk.Label(lf, text="colormap").grid(row=0, column=0, sticky="w", padx=(6, 4))
        ttk.Combobox(lf, textvariable=self.v_cmap, state="readonly", width=11,
                     values=["viridis", "magma", "inferno", "plasma", "cividis",
                             "gray", "bone", "coolwarm"]).grid(row=0, column=1,
                                                               sticky="ew", pady=2)
        ttk.Checkbutton(lf, text="Overlay calculated resonance frequencies", variable=self.v_overlay,
                        command=self._redraw).grid(row=1, column=0, columnspan=3,
                                                   sticky="w", padx=6)
        self.pb = ttk.Progressbar(lf, mode="determinate", maximum=100)
        self.pb.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 2))
        self.btn_calc = ttk.Button(lf, text="Calculate", command=self.on_compute)
        self.btn_calc.grid(row=3, column=0, sticky="ew", padx=6, pady=3)
        ttk.Button(lf, text="Export XLSX + JSON", command=self.on_export
                   ).grid(row=3, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(lf, text="Export image", command=self.on_savefig
                   ).grid(row=3, column=2, sticky="ew", padx=6, pady=3)
        ttk.Button(lf, text="Restore default parameters", command=self.on_reset
                   ).grid(row=4, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))

    def _toggle_model(self):
        state = "normal" if self.vars["model"].get() == "B" else "disabled"
        for ch in self.frm_B.winfo_children():
            try:
                ch.configure(state=state)
            except tk.TclError:
                pass

    # ---------------- Status bar ----------------
    def set_status(self, txt, kind):
        col = {"fresh": ("#1b5e20", "#e8f5e9"),
               "stale": ("#8a6d00", "#fff8e1"),
               "busy":  ("#0d47a1", "#e3f2fd"),
               "err":   ("#b71c1c", "#ffebee")}[kind]
        self.lbl_status.config(text=txt, fg=col[0], bg=col[1])

    # ---------------- Read parameters ----------------
    def read_params(self) -> Params:
        p = Params()
        for fl in fields(p):
            dflt = getattr(p, fl.name)
            try:
                raw = self.vars[fl.name].get()
                if isinstance(dflt, bool):
                    val = bool(raw)
                elif isinstance(dflt, float):
                    val = float(raw)
                elif isinstance(dflt, int):
                    val = int(raw)
                else:
                    val = str(raw)
            except (tk.TclError, TypeError, ValueError):
                raise ValueError(f"Field '{fl.name}' must be numeric.")
            setattr(p, fl.name, val)

        if p.n_B < 1 or p.n_f < 2:
            raise ValueError("N_B >= 1 and N_f >= 2")
        if p.B_max < p.B_min:
            raise ValueError("B end must be at least B start")
        if not p.auto_f and p.f_max <= p.f_min:
            raise ValueError("f end must exceed f start")
        if p.gamma_fwhm <= 0:
            raise ValueError("Linewidth must be positive")
        if p.model == "B" and min(p.k_r, p.beta_pump, p.k_s0) <= 0:
            raise ValueError("Model B rate constants must be positive")
        if p.model == "B":
            cells = p.n_B * p.n_f * (4 if p.geometry == "ensemble" else 1)
            if cells > 4e6:
                if not messagebox.askokcancel(
                        "Calculation size",
                        f"Model B will solve about {cells:.1e} steady-state systems and may take several minutes.\n"
                        "Reduce N_B or N_f if needed. Continue?"):
                    raise ValueError("Cancelled")
        return p

    # ---------------- Calculate ----------------
    def on_compute(self):
        try:
            p = self.read_params()
        except ValueError as ex:
            messagebox.showerror("Parameter error", str(ex))
            return
        self.btn_calc.config(state="disabled")
        self.pb["value"] = 0
        self.set_status("Calculating...", "busy")
        self.cancel.clear()
        self._q = queue.Queue()

        def worker():
            try:
                r = simulate(p, progress=lambda x: self._q.put(("p", x)),
                             cancel=self.cancel)
                self._q.put(("ok", (r, p)))
            except Exception:
                self._q.put(("err", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()
        self.after(60, self._poll)

    def _poll(self):
        try:
            while True:
                tag, payload = self._q.get_nowait()
                if tag == "p":
                    self.pb["value"] = 100 * payload
                elif tag == "ok":
                    self.result, self.params_used = payload
                    self.pb["value"] = 100
                    self.btn_calc.config(state="normal")
                    n = len(self.result["B_mT"])
                    self.v_manual_B.set(False)
                    self.manual_linecut = None
                    self.sld.config(from_=0, to=max(n - 1, 0))
                    self.sld.set(0)
                    self.v_manual_B_value.set(f"{self.result['B_mT'][0]:.9g}")
                    self.lbl_manual_range.config(
                        text=f"Allowed range: {self.result['B_mT'][0]:.9g} – "
                             f"{self.result['B_mT'][-1]:.9g} mT")
                    self._toggle_manual_B(redraw=False)
                    self._redraw()
                    self.set_status(
                        f"Result current  |  Model {self.result['model']}  |  "
                        f"{n} x {len(self.result['f_GHz'])} points  |  "
                        f"{self.result['n_family']} NV orientations", "fresh")
                    return
                else:
                    self.btn_calc.config(state="normal")
                    self.set_status("Calculation failed", "err")
                    messagebox.showerror("Calculation failed", payload[-1500:])
                    return
        except queue.Empty:
            pass
        self.after(60, self._poll)

    # ---------------- Plotting ----------------
    def _blank_axes(self):
        for ax in (self.ax_map, self.ax_cut):
            ax.clear()
        self.ax_map.set_xlabel("MW frequency (GHz)")
        self.ax_map.set_ylabel("B (mT)")
        self.ax_cut.set_xlabel("MW frequency (GHz)")
        self.ax_cut.set_ylabel("PL (norm.)")
        self.ax_map.set_title("2D ODMR Map: PL(B, f)", fontsize=10)
        self.ax_cut.set_title("ODMR Linecut (controlled by slider below)", fontsize=9)
        self.ax_map.text(0.5, 0.5, "press  [ Compute ]", ha="center", va="center",
                         transform=self.ax_map.transAxes, color="0.6")
        self.canvas.draw_idle()

    def _redraw(self):
        if self.result is None:
            return
        r = self.result
        B, f, PL = r["B_mT"], r["f_GHz"], r["PL"]

        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        self.ax_map.clear()

        if len(B) > 1:
            dB = B[1] - B[0]
        else:
            dB = max(abs(B[0]) * 0.02, 0.1)
        ext = [f[0], f[-1], B[0] - dB / 2, B[-1] + dB / 2]
        im = self.ax_map.imshow(PL, origin="lower", aspect="auto", extent=ext,
                                cmap=self.v_cmap.get(), interpolation="nearest")
        self.cbar = self.fig.colorbar(im, ax=self.ax_map, pad=0.015)
        self.cbar.set_label("PL (normalized)")

        if self.v_overlay.get() and r["res"].size:
            R = r["res"]
            w = R[:, 3] / max(R[:, 3].max(), 1e-12)
            self.ax_map.scatter(R[:, 2], R[:, 1], s=2.0, c="w", alpha=0.35 * w + 0.05,
                                linewidths=0, zorder=3)

        self.ax_map.set_xlabel("MW frequency (GHz)")
        self.ax_map.set_ylabel("B (mT)")
        self.ax_map.set_title(f"2D ODMR Map: PL(B, f)  |  Model {r['model']}  |  "
                              f"{r['n_family']} NV orientation(s)", fontsize=10)
        self._draw_cut()

    def _draw_cut(self):
        r = self.result
        if r is None:
            return
        manual = self.v_manual_B.get() and self.manual_linecut is not None
        if manual:
            B_value = float(self.manual_linecut["B_mT"])
            pl = self.manual_linecut["PL_norm"]
            control = "exact manual B"
        else:
            i = int(np.clip(self.v_cut.get(), 0, len(r["B_mT"]) - 1))
            B_value = float(r["B_mT"][i])
            pl = r["PL"][i]
            control = "slider-controlled"
        self.ax_cut.clear()
        self.ax_cut.plot(r["f_GHz"], pl, lw=1.3, color="#c62828")
        self.ax_cut.set_xlim(r["f_GHz"][0], r["f_GHz"][-1])
        self.ax_cut.set_xlabel("MW frequency (GHz)")
        self.ax_cut.set_ylabel("PL (norm.)")
        self.ax_cut.grid(alpha=0.25)
        self.ax_cut.set_title(
            f"ODMR Linecut ({control})  |  B = {B_value:.9g} mT",
            fontsize=9,
        )
        self.lbl_cut.config(text=f"B = {B_value:.9g} mT")
        self.canvas.draw_idle()

    def _on_slider(self, val):
        if self.result is None or self.v_manual_B.get():
            return
        self.v_cut.set(int(float(val)))
        self._draw_cut()

    def _toggle_manual_B(self, redraw=True):
        active = bool(self.v_manual_B.get())
        state = "normal" if active else "disabled"
        self.ent_manual_B.configure(state=state)
        self.btn_manual_B.configure(state=state)
        self.sld.configure(state="disabled" if active else "normal")

        if active and self.result is not None:
            i = int(np.clip(self.v_cut.get(), 0, len(self.result["B_mT"]) - 1))
            self.v_manual_B_value.set(f"{self.result['B_mT'][i]:.9g}")
            self._apply_manual_B()
            self.ent_manual_B.focus_set()
            self.ent_manual_B.selection_range(0, tk.END)
        else:
            self.manual_linecut = None
            if redraw:
                self._draw_cut()

    def _apply_manual_B(self):
        if not self.v_manual_B.get():
            return
        if self.result is None or self.params_used is None:
            messagebox.showwarning("No data", "Calculate the 2D map first.")
            return
        try:
            B_value = float(self.v_manual_B_value.get())
        except ValueError:
            messagebox.showerror("Input error", "The manual magnetic field must be numeric.")
            return
        if not np.isfinite(B_value):
            messagebox.showerror("Input error", "The manual magnetic field must be finite.")
            return
        B_lo, B_hi = float(self.result["B_mT"][0]), float(self.result["B_mT"][-1])
        if B_value < B_lo or B_value > B_hi:
            messagebox.showerror(
                "Out of range",
                f"The manual magnetic field must be between {B_lo:.9g}–{B_hi:.9g} mT .",
            )
            return
        try:
            self.manual_linecut = simulate_linecut(
                self.params_used, B_value, self.result["f_GHz"])
        except Exception:
            messagebox.showerror("Line-spectrum calculation failed", traceback.format_exc()[-1500:])
            return
        self.v_manual_B_value.set(f"{B_value:.9g}")
        self._draw_cut()

    # ---------------- Export ----------------
    def on_export(self):
        if self.result is None:
            messagebox.showwarning("No data", "Calculate first.")
            return
        default = f"NV_ODMR_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        path = filedialog.asksaveasfilename(
            title="Save XLSX and matching JSON", defaultextension=".xlsx",
            initialfile=default, filetypes=[("Excel workbook", "*.xlsx"),
                                            ("All files", "*.*")])
        if not path:
            return
        cut_idx = int(np.clip(self.v_cut.get(), 0, len(self.result["B_mT"]) - 1))
        linecut = self.manual_linecut if self.v_manual_B.get() else None
        json_path = os.path.splitext(path)[0] + ".json"
        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            write_xlsx(path, self.result, self.params_used, cut_idx,
                       linecut=linecut, generated=generated)
            write_json(json_path, self.result, self.params_used, cut_idx,
                       linecut=linecut, generated=generated)
        except Exception:
            messagebox.showerror("Export failed", traceback.format_exc()[-1500:])
            return
        messagebox.showinfo(
            "Export complete",
            f"Saved XLSX:\n{os.path.abspath(path)}\n\n"
            f"Matching JSON:\n{os.path.abspath(json_path)}",
        )

    def on_savefig(self):
        if self.result is None:
            messagebox.showwarning("No data", "Calculate first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save image", defaultextension=".png",
            initialfile=f"NV_ODMR_{datetime.now():%Y%m%d_%H%M%S}.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        self.fig.savefig(path, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Saved", os.path.abspath(path))

    def on_reset(self):
        d = Params()
        for fl in fields(d):
            self.vars[fl.name].set(getattr(d, fl.name))
        self.v_manual_B.set(False)
        self.manual_linecut = None
        self.v_manual_B_value.set("0.0")
        self._toggle_manual_B()
        self.set_status("Default parameters restored", "stale")


def main():
    root = tk.Tk()
    root.title("NV Center CW-ODMR Simulator  --  fluorescence vs microwave frequency vs magnetic field")
    root.geometry("1240x820")
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    NVApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
