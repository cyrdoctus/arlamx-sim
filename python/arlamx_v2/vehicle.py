"""Six-beam hex sail: mass properties and torque-rod layout (config/plant/vehicle.yaml)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arlamx_v2.config import load
from arlamx_v2.geometry import load_geom
from arlamx_v2.paths import DATA


@dataclass
class Vehicle:
    mass: float
    I: np.ndarray
    com: np.ndarray
    rod_axis: np.ndarray
    rod_pos: np.ndarray
    rod_dmax: np.ndarray
    rod_pmax: np.ndarray
    tau_max: np.ndarray
    cp_offset: np.ndarray
    beam_share: float

    @property
    def I_diag(self):
        return np.diag(self.I).copy()


# Thin rod: I = m(|p|^2 E - p p^T) + m L^2/12 (E - t t^T) (parallel-axis theorem).
def _rod_I(m, p, t, L):
    p, t = np.asarray(p, float), np.asarray(t, float)
    E = np.eye(3)
    return m * (p @ p * E - np.outer(p, p)) + m * L * L / 12.0 * (E - np.outer(t, t))


# Point mass: I = m(|p|^2 E - p p^T).
def _point_I(m, p):
    p = np.asarray(p, float)
    return m * (p @ p * np.eye(3) - np.outer(p, p))


def beam_area_share(geom_path):
    n, A, c = load_geom(str(geom_path))
    beam = (np.abs(c[:, 2]) > 1e-4) | (np.abs(n[:, 2]) < 0.9)
    return float(A[beam].sum() / A.sum())


def build(cfg=None):
    cfg = cfg or load("vehicle")
    ms = cfg["mass_kg"]
    rc = cfg["rods"]
    n_rod = int(rc["count"])
    th = np.radians(float(cfg.get("beam_angle0_deg", 0.0))) + np.arange(n_rod) * 2.0 * np.pi / n_rod
    a = float(cfg["beam_apothem_m"])
    L = float(cfg["beam_length_m"])
    R = float(cfg["membrane_radius_m"])
    pos = np.c_[a * np.cos(th), a * np.sin(th), np.zeros(n_rod)]
    axis = np.c_[-np.sin(th), np.cos(th), np.zeros(n_rod)]

    share = cfg.get("beam_share", "area")
    share = beam_area_share(DATA / cfg["geom"]) if share == "area" else float(share)
    m_beam = float(ms["structure"]) * share
    m_mem = float(ms["structure"]) - m_beam
    m_rod = float(ms["rods"]) / n_rod

    # Regular-hexagon lamina: I_z = 5/12 m R^2, I_x = I_y = 5/24 m R^2
    # (Young & Budynas, Roark's Formulas for Stress and Strain, 7th ed., Table A.1).
    I = np.diag([5.0 / 24.0, 5.0 / 24.0, 5.0 / 12.0]) * m_mem * R * R
    p_pcb = np.asarray(cfg.get("pcb_pos_m", [0.0, 0.0, 0.0]), float)
    trim = cfg.get("trim") or {}
    parts = [(m_mem, np.zeros(3)), (float(ms["pcb_battery"]), p_pcb)]
    I += _point_I(float(ms["pcb_battery"]), p_pcb)
    if trim:
        parts.append((float(trim["mass_kg"]), np.asarray(trim["pos_m"], float)))
        I += _point_I(float(trim["mass_kg"]), np.asarray(trim["pos_m"], float))
    for p, t in zip(pos, axis):
        I += _rod_I(m_beam / n_rod, p, t, L) + _point_I(m_rod, p)
        parts += [(m_beam / n_rod, p), (m_rod, p)]
    mass = sum(m for m, _ in parts)
    com = sum(m * p for m, p in parts) / mass
    I -= _point_I(mass, com)

    dmax = np.full(n_rod, float(rc["dipole_max_Am2"]))
    pmax = np.full(n_rod, float(rc["power_peak_W"]))
    nc = cfg.get("normal_coil") or {}
    if float(nc.get("dipole_max_Am2", 0.0)) > 0.0:
        axis = np.r_[axis, [[0.0, 0.0, 1.0]]]
        pos = np.r_[pos, [[0.0, 0.0, 0.0]]]
        dmax = np.r_[dmax, float(nc["dipole_max_Am2"])]
        pmax = np.r_[pmax, float(nc["power_peak_W"])]
    m_cap = np.abs(axis).T @ dmax
    b = float(cfg["b_ref_T"])
    # tau = m x B: tau_x needs m_y, tau_y needs m_x, tau_z either in-plane component.
    tau_max = b * np.array([max(m_cap[1], m_cap[2]), max(m_cap[0], m_cap[2]), max(m_cap[0], m_cap[1])])
    return Vehicle(mass, I, com, axis, pos, dmax, pmax, tau_max,
                   np.asarray(cfg.get("cp_offset_m", [0, 0, 0]), float), share)


def axis_authority(on, veh):
    on = np.asarray(on, bool)
    w = np.abs(veh.rod_axis) * veh.rod_dmax[:, None]
    full = np.maximum(w.sum(axis=0), 1e-30)
    frac = (w[on].sum(axis=0) if on.any() else np.zeros(3)) / full
    return np.array([frac[1], frac[0], 0.5 * (frac[0] + frac[1])])
