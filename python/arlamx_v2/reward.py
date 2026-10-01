"""SC_v3-SC_v7 reward composites."""
from __future__ import annotations

import math

import numpy as np


# SC_v3-v7 shaping terms: docs/handoff/06_reward_and_delegation.md (SC_v3 core, v7 composite).
def dE_vs_baseline(dE_a, dE_b, alt_km, weight=0.75, k=3.0, lo=300.0, hi=500.0):
    x = np.clip((hi - alt_km) / (hi - lo), 0.0, 1.0)
    w_long = (math.expm1(k * x) / math.expm1(k)) if k != 0 else x
    den = max(abs(dE_b), 1e-6)
    core = float(np.clip((dE_a - dE_b) / den, -1.0, 1.0))
    return weight * w_long * core


def power_budget(soc, gen_norm, depleted, weight=1.0):
    if depleted:
        return -50.0
    m = 0.8 if soc >= 0.6 else 2.0
    return weight * m * (min(soc, 0.8) + 0.5 * gen_norm)


def gs_alignment(soc, Cd, z_dot_g, visible, weight=1.0, cd_ref=1.05, lo=0.10, hi=0.20):
    if soc <= 0.5 or visible < 0.5:
        return 0.0
    rho = Cd / max(cd_ref, 1e-9) - 1.0
    if rho <= lo:
        f = 1.0
    elif rho >= hi:
        f = 0.0
    else:
        f = 1.0 - (rho - lo) / (hi - lo)
    return weight * f * max(0.0, z_dot_g)


def shade_conservation(effort_dark, weight=0.2):
    return -weight * effort_dark


def action_smoothness(q_new, q_prev, weight=0.02):
    return -weight * (1.0 - float(np.dot(q_new, q_prev)))


def omega_penalty(omega, weight=0.002):
    return -weight * float(np.linalg.norm(omega))


def altitude_cliff(alt_km, weight=1000.0, floor_km=250.0):
    return -weight if alt_km < floor_km else 0.0


def power_band(soc, depleted, weight=1.0, soc_lo=0.4, soc_hi=0.6, decay_k=6.0,
               low_weight=4.0, depletion_penalty=50.0):
    if depleted or soc <= 1e-6:
        return -depletion_penalty
    if soc < soc_lo:
        return -low_weight * (soc_lo - soc) / soc_lo
    if soc <= soc_hi:
        return weight
    return weight * float(np.exp(-decay_k * (soc - soc_hi)))


def gs_tiered_pointing(err_axes_deg, visible, eclipse, weight=2.0,
                       tier_deg=(2.0, 5.0, 10.0), tier_scale=(1.0, 0.5, 0.25),
                       penalty=0.5, daylight_only=True):
    if visible < 0.5:
        return 0.0
    if daylight_only and eclipse <= 0.5:
        return 0.0
    err = np.asarray(err_axes_deg, float).reshape(-1)
    if err.size != 3:
        return 0.0
    total = 0.0
    for e in np.abs(err):
        for edge, scale in zip(tier_deg, tier_scale):
            if e <= edge:
                total += weight * scale
                break
        else:
            total -= penalty
    return total


def action_feasibility(cmd_angle_deg, invalid, weight=1.0, max_cmd_deg=40.0,
                       invalid_penalty=5.0, in_band_reward=0.1):
    if invalid:
        return -invalid_penalty
    if cmd_angle_deg <= max_cmd_deg:
        return in_band_reward
    return -weight * (cmd_angle_deg - max_cmd_deg) / max_cmd_deg


def brownout_recovery(recovered, active, gen_norm, weight=100.0, gen_shaping=1.0):
    r = 0.0
    if recovered:
        r += weight
    if active and gen_shaping > 0.0:
        r += gen_shaping * float(gen_norm)
    return r


def momentum_penalty(omega, inertia_diag=(0.0125, 0.0125, 0.025), weight=0.001):
    h = np.asarray(inertia_diag, float) * np.asarray(omega, float)
    return -weight * float(np.linalg.norm(h))


def compose_sc_v3(info):
    parts = {
        "dE_vs_baseline": dE_vs_baseline(info["dE_actual"], info["dE_baseline"], info["altitude_km"]),
        "power_budget": power_budget(info["battery_soc"], info["power_gen_norm"], info["battery_depleted"]),
        "gs_alignment": gs_alignment(
            info["battery_soc"], info["Cd"], info.get("gs_point_cos", 0.0), info.get("gs_visible", 0.0)
        ),
        "shade_conservation": shade_conservation(info.get("shade_draw", 0.0)),
        "action_smoothness": action_smoothness(info["q_new"], info["q_prev"]),
        "omega_penalty": omega_penalty(info["omega"]),
        "altitude_cliff": altitude_cliff(info["altitude_km"]),
    }
    return float(sum(parts.values())), parts


def lift_work(dE_lift, alt_km, weight=1.5, lo=280.0, hi=450.0):
    x = np.clip((hi - alt_km) / max(hi - lo, 1.0), 0.0, 1.0)
    return weight * x * float(np.clip(dE_lift / 5.0, -1.0, 1.0))


def srp_work(dE_srp, alt_km, srp_scale, weight=0.8, lo=420.0, hi=550.0):
    x = np.clip((alt_km - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    flash = 1.0 + 0.5 * max(0.0, srp_scale - 1.0)
    return weight * x * flash * float(np.clip(dE_srp / 8.0, -1.0, 1.0))


def torque_thrift(effort, weight=0.6):
    return -weight * float(np.clip(effort, 0.0, 1.0))


def gps_dropout_skill(used_prop, alt_ok, soc_ok, weight=0.4):
    if not used_prop:
        return 0.0
    return weight if (alt_ok and soc_ok) else -0.25 * weight


def compose_sc_v4(info, brownout=False):
    parts = {
        "power_band": power_band(
            info["battery_soc"], info["battery_depleted"], weight=1.0
        ),
        "shade_conservation": shade_conservation(info.get("shade_draw", 0.0)),
        "dE_vs_baseline": dE_vs_baseline(
            info["dE_actual"], info["dE_baseline"], info["altitude_km"], weight=2.0
        ),
        "gs_tiered_pointing": gs_tiered_pointing(
            info.get("gs_err_axes_deg", np.zeros(3)),
            info.get("gs_visible", 0.0),
            info.get("eclipse", 1.0),
        ),
        "action_feasibility": action_feasibility(
            info.get("cmd_angle_deg", 0.0), info.get("action_invalid", False)
        ),
        "action_smoothness": action_smoothness(info["q_new"], info["q_prev"]),
        "omega_penalty": omega_penalty(info["omega"]),
        "momentum_penalty": momentum_penalty(info["omega"]),
        "altitude_cliff": altitude_cliff(info["altitude_km"]),
    }
    if brownout:
        parts["brownout_recovery"] = brownout_recovery(
            info.get("brownout_recovered", False),
            info.get("brownout_active", False),
            info.get("power_gen_norm", 0.0),
        )
    return float(sum(parts.values())), parts


def compose_sc_v5(info, brownout=False):
    _rew, parts = compose_sc_v4(info, brownout=brownout)
    parts["dE_vs_baseline"] = dE_vs_baseline(
        info["dE_actual"], info["dE_baseline"], info["altitude_km"], weight=3.5
    )
    parts["lift_work"] = lift_work(info.get("dE_lift", 0.0), info["altitude_km"], weight=1.5)
    parts["srp_work"] = srp_work(
        info.get("dE_srp", 0.0), info["altitude_km"], info.get("srp_scale", 1.0), weight=0.8
    )
    if brownout:
        parts["brownout_recovery"] = brownout_recovery(
            info.get("brownout_recovered", False),
            info.get("brownout_active", False),
            info.get("power_gen_norm", 0.0),
            weight=180.0,
            gen_shaping=2.0,
        )
        if info.get("brownout_recovered", False):
            steps = float(info.get("recover_steps", 8.0))
            parts["recover_speed"] = 40.0 * float(np.exp(-0.15 * max(0.0, steps - 1.0)))
    parts["torque_thrift"] = torque_thrift(info.get("torque_effort", 0.1), weight=0.25)
    return float(sum(parts.values())), parts


def compose_sc_v6(info, brownout=False):
    _rew, parts = compose_sc_v5(info, brownout=brownout)
    parts["torque_thrift"] = torque_thrift(info.get("torque_effort", 0.1), weight=0.9)
    parts["gps_dropout_skill"] = gps_dropout_skill(
        info.get("used_propagator", False),
        info["altitude_km"] >= 250.0,
        info["battery_soc"] > 0.2,
        weight=0.5,
    )
    if info.get("held_inference", False):
        parts["inference_thrift"] = 0.15
    return float(sum(parts.values())), parts


V7_DEFAULT_W = {
    "band_weight": 1.0, "band_lo": 0.4, "band_hi": 0.6,
    "band_above_mode": "exp",
    "band_decay_k": 6.0,
    "band_above_slope": 2.0,
    "band_low_weight": 4.0, "band_depletion": 50.0,
    "dE_weight": 2.0,
    "gs_weight": 2.0, "tier_deg": (2.0, 5.0, 10.0),
    "tier_scale": (1.0, 0.5, 0.25), "gs_penalty": 0.5,
    "feas_weight": 1.0, "smooth_weight": 0.02,
    "omega_weight": 0.002, "momentum_weight": 0.001,
    "thrift_weight": 0.25,
    "brownout_weight": 100.0, "brownout_gen": 1.0,
}


def power_band_v7(soc, depleted, w):
    if depleted or soc <= 1e-6:
        return -float(w["band_depletion"])
    lo, hi = float(w["band_lo"]), float(w["band_hi"])
    weight = float(w["band_weight"])
    if soc < lo:
        return -float(w["band_low_weight"]) * (lo - soc) / lo
    if soc <= hi:
        return weight
    if w["band_above_mode"] == "linear_penalty":
        return weight - (weight + float(w["band_above_slope"])) * (soc - hi) / (1.0 - hi)
    return weight * float(np.exp(-float(w["band_decay_k"]) * (soc - hi)))


def compose_sc_v7(info, w=None):
    cfg = dict(V7_DEFAULT_W)
    if w:
        cfg.update(w)
    parts = {
        "power_band": power_band_v7(
            info["battery_soc"], info["battery_depleted"], cfg
        ),
        "shade_conservation": shade_conservation(info.get("shade_draw", 0.0)),
        "dE_vs_baseline": dE_vs_baseline(
            info["dE_actual"], info["dE_baseline"], info["altitude_km"],
            weight=float(cfg["dE_weight"]),
        ),
        "gs_tiered_pointing": gs_tiered_pointing(
            info.get("gs_err_axes_deg", np.zeros(3)),
            info.get("gs_visible", 0.0),
            info.get("eclipse", 1.0),
            weight=float(cfg["gs_weight"]),
            tier_deg=tuple(cfg["tier_deg"]),
            tier_scale=tuple(cfg["tier_scale"]),
            penalty=float(cfg["gs_penalty"]),
        ),
        "action_feasibility": action_feasibility(
            info.get("cmd_angle_deg", 0.0), info.get("action_invalid", False),
            weight=float(cfg["feas_weight"]),
        ),
        "action_smoothness": action_smoothness(
            info["q_new"], info["q_prev"], weight=float(cfg["smooth_weight"])
        ),
        "omega_penalty": omega_penalty(info["omega"], weight=float(cfg["omega_weight"])),
        "momentum_penalty": momentum_penalty(
            info["omega"], weight=float(cfg["momentum_weight"])
        ),
        "altitude_cliff": altitude_cliff(info["altitude_km"]),
        "torque_thrift": torque_thrift(
            info.get("torque_effort", 0.0), weight=float(cfg["thrift_weight"])
        ),
        "brownout_recovery": brownout_recovery(
            info.get("brownout_recovered", False),
            info.get("brownout_active", False),
            info.get("power_gen_norm", 0.0),
            weight=float(cfg["brownout_weight"]),
            gen_shaping=float(cfg["brownout_gen"]),
        ),
    }
    return float(sum(parts.values())), parts
