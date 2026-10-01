"""SC_v8+ reward: stability, delegation, trend, adapt and Duo terms (docs/modules/14_reward_v8.md)."""
from __future__ import annotations

import numpy as np

from arlamx_v2.reward import compose_sc_v7

EPS = 1e-12


# gate = 1 - s (1 - gate_floor), eq. (1), docs/modules/14_reward_v8.md §2.1.
def split_to_gates(split, gate_floor):
    s = np.clip(np.asarray(split, float).reshape(3), 0.0, 1.0)
    return 1.0 - s * (1.0 - float(gate_floor))


# A_i = sign(tau_env,i tau_want,i) min(1, |tau_env,i| / tau_ref), eq. (2), §2.2.
def delegation_alignment(tau_env, tau_want, tau_env_ref=1.5e-8, eps=1e-11):
    te = np.asarray(tau_env, float).reshape(3)
    tw = np.asarray(tau_want, float).reshape(3)
    out = np.zeros(3)
    live = np.abs(tw) > eps
    if not np.any(live):
        return out
    mag = np.minimum(1.0, np.abs(te[live]) / max(float(tau_env_ref), EPS))
    out[live] = np.sign(te[live] * tw[live]) * mag
    return out


# B = sum s_i A_i / sum s_i, eq. (3), §2.3.
def delegation_benefit(split, alignment):
    s = np.clip(np.asarray(split, float).reshape(3), 0.0, 1.0)
    a = np.asarray(alignment, float).reshape(3)
    tot = float(np.sum(s))
    if tot < EPS:
        return 0.0
    return float(np.clip(np.sum(s * a) / tot, -1.0, 1.0))


# P = sum (E_full - E_gate) / sum E_full, eq. (4), §2.3.
def power_saved_fraction(tau_ctrl, tau_want, tau_max, gates):
    del tau_ctrl  # measured delivery no longer enters eq. (4)
    tw = np.abs(np.asarray(tau_want, float).reshape(3))
    tm = np.maximum(np.asarray(tau_max, float).reshape(3), EPS)
    g = np.clip(np.asarray(gates, float).reshape(3), EPS, 1.0)
    e_full = np.minimum(tw, tm) / tm
    e_gate = np.minimum(tw, g * tm) / tm
    saved = np.maximum(0.0, e_full - e_gate)
    den = float(np.sum(e_full))
    if den < EPS:
        return 0.0
    return float(np.clip(np.sum(saved) / den, 0.0, 1.0))


# S = clip((P - P_min)/(P_sig - P_min), 0, 1), eq. (5), §2.4.
def significance(p_saved, p_min, p_sig):
    if p_sig <= p_min:
        return 1.0 if p_saved >= p_sig else 0.0
    return float(np.clip((float(p_saved) - p_min) / (p_sig - p_min), 0.0, 1.0))


# v11 adapt terms, docs/handoff/06_reward_and_delegation.md (v11 adaptability).
def power_drop(soc, soc_prev, weight, thresh, scale, cap=3.0):
    if soc_prev is None:
        return 0.0
    drop = max(0.0, float(soc_prev) - float(soc))
    excess = max(0.0, drop - float(thresh))
    return -float(weight) * min(excess / max(float(scale), EPS), float(cap))


def adapt_consistency(track_err, track_ref, weight, scale, cap=3.0):
    excess = max(0.0, float(track_err) - 2.0 * float(track_ref))
    return -float(weight) * min(excess / max(float(scale), EPS), float(cap))


def hold_term(B, damage_active, weight):
    if not damage_active:
        return 0.0
    return float(weight) * float(np.clip(B, 0.0, 1.0))


# r = -w max(0, (h_soft - h)/(h_soft - h_floor))^2, eq. (9), §3.
def altitude_margin(alt_km, weight, h_soft, h_floor):
    if alt_km >= h_soft:
        return 0.0
    span = max(h_soft - h_floor, 1e-6)
    x = min(1.0, (h_soft - float(alt_km)) / span)
    return -float(weight) * x * x


# r = -w |dE - dE_prev| / dE_scale, eq. (10), §3.
def decay_stability(dE, dE_prev, weight, scale, cap=3.0):
    if dE_prev is None:
        return 0.0
    adverse = max(0.0, float(dE_prev) - float(dE))
    return -float(weight) * min(adverse / max(float(scale), EPS), float(cap))


BOOSTABLE = ("dE_vs_baseline", "power_band", "gs_tiered_pointing")


def compose_sc_v8(info, cfg):
    base_w = dict(cfg.get("v7", {}))
    stab = cfg.get("stability", {})
    deleg = cfg.get("delegation", {})

    # v9 trend adaptation, eqs. (11)-(12), docs/modules/14_reward_v8.md §8; absent for v8
    trend = info.get("trend")
    if trend and cfg.get("trend", {}).get("enabled", False):
        tc = cfg["trend"]
        d_alt = float(trend.get("d_alt_norm", 0.0))
        d_soc = float(trend.get("d_soc", 0.0))
        base_w["dE_weight"] = float(base_w.get("dE_weight", 2.0)) * (
            1.0 + float(tc["gain_alt"]) * float(np.clip(-d_alt, 0.0, 1.0)))
        base_w["band_weight"] = float(base_w.get("band_weight", 1.0)) * (
            1.0 + float(tc["gain_soc"]) * float(
                np.clip(-d_soc / max(float(tc["soc_ref"]), EPS), 0.0, 1.0)))

    total, parts = compose_sc_v7(info, w=base_w)

    # v8 stability terms, eqs. (9)-(10), docs/modules/14_reward_v8.md §3
    parts["altitude_margin"] = altitude_margin(
        info["altitude_km"], stab.get("margin_weight", 0.0),
        stab.get("h_soft_km", 320.0), stab.get("h_floor_km", 250.0))
    parts["decay_stability"] = decay_stability(
        info.get("dE_actual", 0.0), info.get("dE_prev"),
        stab.get("decay_stab_weight", 0.0), stab.get("decay_stab_scale", 100.0),
        cap=stab.get("decay_stab_cap", 3.0))
    parts["power_drop"] = power_drop(
        info.get("battery_soc", 0.5), info.get("soc_prev"),
        stab.get("power_drop_weight", 0.0),
        stab.get("power_drop_thresh", 0.02),
        stab.get("power_drop_scale", 0.05),
        cap=stab.get("power_drop_cap", 3.0))

    adapt = cfg.get("adapt", {})
    if adapt.get("enabled", False):
        parts["adapt_consistency"] = adapt_consistency(
            info.get("tracking_err_rad", 0.0), info.get("track_ref_rad", 0.0),
            adapt.get("w_consistency", 0.0), adapt.get("track_scale_rad", 0.1),
            cap=adapt.get("track_cap", 3.0))
    else:
        parts["adapt_consistency"] = 0.0

    # v8b delegation, eqs. (2)-(8), docs/modules/14_reward_v8.md §2
    if deleg.get("enabled", False):
        split = np.asarray(info.get("deleg_split", np.zeros(3)), float)
        align = delegation_alignment(
            info.get("tau_env_est", np.zeros(3)),
            info.get("tau_want", np.zeros(3)),
            tau_env_ref=float(deleg.get("tau_env_ref_Nm", 1.5e-8)),
            eps=float(deleg.get("torque_floor_Nm", 1e-11)))
        B = delegation_benefit(split, align)
        P = power_saved_fraction(
            info.get("tau_ctrl_mean", np.zeros(3)),
            info.get("tau_want_abs", np.abs(info.get("tau_want", np.zeros(3)))),
            info.get("tau_max", np.ones(3)),
            info.get("gates", np.ones(3)))
        S = significance(P, float(deleg["p_min"]), float(deleg["p_sig"]))
        pos_B = float(np.clip(B, 0.0, 1.0))

        parts["deleg_power"] = float(deleg["w_power"]) * P * pos_B
        parts["deleg_accuracy"] = float(deleg["w_accuracy"]) * B
        parts["adapt_deleg_hold"] = hold_term(
            B, info.get("damage_active", False),
            cfg.get("adapt", {}).get("w_deleg_hold", 0.0))

        boost = 1.0 + float(deleg["kappa"]) * S * pos_B
        for k in BOOSTABLE:
            if k in parts and parts[k] > 0.0:
                parts[k] *= boost
        info.setdefault("_diag", {}).update(
            {"deleg_B": B, "deleg_P": P, "deleg_S": S, "deleg_boost": boost,
             "deleg_align": align.tolist()})
    else:
        parts["deleg_power"] = 0.0
        parts["deleg_accuracy"] = 0.0

    total = float(sum(parts.values()))
    return total, parts


# Duo secondary advisor reward, docs/modules/14_reward_v8.md §9.
def duo_reward(info, cfg):
    w = cfg.get("duo_secondary", {})
    dec_ref = max(float(w.get("decay_ref_km_d", 10.0)), EPS)
    soc_ref = max(float(w.get("soc_ref", 0.5)), EPS)

    decay_prop = float(info.get("decay_prop_km_d", 0.0))
    decay_main = float(info.get("decay_main_km_d", 0.0))
    soc_end = float(info.get("soc_end_prop", soc_ref))
    dq = float(np.clip(abs(float(info.get("q_dot", 1.0))), 0.0, 1.0))

    parts = {
        "horizon_decay": -float(w.get("w_decay", 3.0)) * float(
            np.clip(decay_prop / dec_ref, 0.0, 3.0)),
        "horizon_power": -float(w.get("w_power", 2.0)) * float(
            np.clip((soc_ref - soc_end) / soc_ref, 0.0, 1.0)),
        "horizon_improvement": float(w.get("w_improve", 6.0)) * float(
            np.clip((decay_main - decay_prop) / dec_ref, -1.0, 1.0)),
        "horizon_slew": -float(w.get("w_slew", 0.5)) * (1.0 - dq),
    }
    return float(sum(parts.values())), parts
