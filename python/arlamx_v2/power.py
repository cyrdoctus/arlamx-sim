"""Magnetorquer electrical power: minimum-norm dipole, I^2R duty, achievable torque."""
from __future__ import annotations

import numpy as np

from arlamx_v2.config import load

EPS = 1e-18


# Minimum-norm dipole m = (B x tau) / |B|^2; tau parallel to B is unproducible (Stickler & Alfriend 1976).
def dipole_for_torque(tau_cmd, b_body):
    tau = np.asarray(tau_cmd, float).reshape(3)
    b = np.asarray(b_body, float).reshape(3)
    b2 = float(np.dot(b, b))
    tn = float(np.linalg.norm(tau))
    if b2 < EPS or tn < EPS:
        return np.zeros(3), 0.0
    m = np.cross(b, tau) / b2
    bhat = b / np.sqrt(b2)
    par = abs(float(np.dot(tau, bhat)))
    return m, float(np.clip(par / tn, 0.0, 1.0))


# Coil I^2R: m = N I A and P = I^2 R, so P = sum_i P_peak,i (m_i / m_max,i)^2.
def mtq_power_w(tau_cmd, b_body, cfg=None):
    cfg = cfg or load("power_mtq")
    mt = cfg["magnetorquers"]
    m_max = np.asarray(mt["dipole_max_Am2"], float)
    p_peak = np.asarray(mt["power_peak_W"], float)

    if str(mt.get("model", "quadratic")).lower() == "legacy_linear":
        tau_max = np.asarray(mt["torque_max_Nm"], float)
        eff = float(np.clip(np.mean(np.abs(np.asarray(tau_cmd, float)) / tau_max), 0.0, 1.0))
        return float(mt["legacy_peak_W"]) * eff, {"effort": eff, "model": "legacy_linear"}

    m, unach = dipole_for_torque(tau_cmd, b_body)
    m_sat = np.clip(m, -m_max, m_max)
    duty = np.abs(m_sat) / np.maximum(m_max, EPS)
    p = float(np.sum(p_peak * duty ** 2))
    return p, {
        "dipole_Am2": m_sat,
        "dipole_unsat_Am2": m,
        "duty": duty,
        "saturated": bool(np.any(np.abs(m) > m_max * 1.000001)),
        "unachievable_frac": unach,
        "model": "quadratic",
    }


# tau = m_sat x B (Stickler & Alfriend 1976).
def achievable_torque(tau_cmd, b_body, cfg=None):
    cfg = cfg or load("power_mtq")
    m_max = np.asarray(cfg["magnetorquers"]["dipole_max_Am2"], float)
    tau = np.asarray(tau_cmd, float).reshape(3)
    b = np.asarray(b_body, float).reshape(3)
    m, _ = dipole_for_torque(tau, b)
    m_sat = np.clip(m, -m_max, m_max)
    tau_ach = np.cross(m_sat, b)
    tn = float(np.linalg.norm(tau))
    if tn < EPS:
        return tau_ach, 0.0
    return tau_ach, float(np.clip(1.0 - np.linalg.norm(tau_ach) / tn, 0.0, 1.0))


def mtq_energy_J(tau_cmd, b_body, dt_s, cfg=None):
    p, diag = mtq_power_w(tau_cmd, b_body, cfg)
    return p * float(dt_s), diag


def effort_from_dipole(m, cfg=None):
    cfg = cfg or load("power_mtq")
    mt = cfg["magnetorquers"]
    m_max = np.asarray(mt["dipole_max_Am2"], float)
    p_peak = np.asarray(mt["power_peak_W"], float)
    m = np.asarray(m, float).reshape(3)
    duty = np.abs(m) / np.maximum(m_max, EPS)
    p = float(np.sum(p_peak * duty ** 2))
    p_full = float(np.sum(p_peak))
    return float(np.clip(p / max(p_full, EPS), 0.0, 1.0)), {
        "dipole_Am2": m,
        "duty": duty,
        "model": "quadratic_plant_m",
    }


def effort_equivalent(tau_cmd, b_body, cfg=None):
    cfg = cfg or load("power_mtq")
    p, diag = mtq_power_w(tau_cmd, b_body, cfg)
    p_full = float(np.sum(np.asarray(cfg["magnetorquers"]["power_peak_W"], float)))
    return float(np.clip(p / max(p_full, EPS), 0.0, 1.0)), diag
