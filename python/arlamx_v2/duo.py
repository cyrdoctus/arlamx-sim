"""Duo arbiter: picks between two advisors by FP32 forward propagation of each candidate."""
from __future__ import annotations

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2 import propagator as prop
from arlamx_v2.config import load

EPS = 1e-12


def _unit_q(q):
    q = np.asarray(q, float).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return q if q[0] >= 0.0 else -q


def quat_angle_deg(qa, qb):
    d = abs(float(np.dot(_unit_q(qa), _unit_q(qb))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def horizon_observation(state, q_main, cfg):
    h = cfg["horizon"]
    r = np.asarray(state["r"], float)
    v = np.asarray(state["v"], float)
    rn = max(float(np.linalg.norm(r)), 1.0)
    vn = max(float(np.linalg.norm(v)), 1e-6)

    stride = int(h.get("eclipse_stride", 5))
    n_ecl = max(2, int(h["horizon_s"] / (h["prop_dt_s"] * stride)))
    offs = np.linspace(0.0, float(h["horizon_s"]), n_ecl)
    samples = prop.propagate_samples(
        r, v, offs, dt_s=h["prop_dt_s"],
        bc_inv=state.get("bc_inv", 0.0), rho0=state.get("rho", 3e-12),
        alt0_m=state.get("alt_km", 400.0) * 1e3)
    r1, v1 = samples[-1]
    sma0 = prop.sma_m(r, v)
    sma1 = prop.sma_m(r1, v1)
    d_sma_km = (sma0 - sma1) / 1e3 if np.isfinite(sma0) and np.isfinite(sma1) else 0.0
    decay_km_d = d_sma_km / max(h["horizon_s"] / 86400.0, EPS)

    ecl_frac = _ecl_frac(samples, state.get("sun_N"))
    p_gen_sunlit = float(state.get("p_gen_sunlit_W", 0.0))
    cap_J = max(float(state.get("batt_cap_J", 0.53 * 3600.0)), EPS)
    p_avg = ((1.0 - ecl_frac) * (p_gen_sunlit - 0.205)
             - ecl_frac * 0.0) - 0.007
    soc_end = float(np.clip(state.get("soc", 0.5)
                            + p_avg * h["horizon_s"] / cap_J, 0.0, 1.0))

    q = _unit_q(q_main)
    obs = np.concatenate([
        r / rn,
        v / vn,
        [float(state.get("alt_km", 400.0)) / 7000.0],
        [d_sma_km / 10.0],
        [decay_km_d / 10.0],
        [float(state.get("soc", 0.5))],
        [soc_end],
        [ecl_frac],
        q,
        [np.log10(max(float(state.get("rho", 1e-16)), 1e-16))],
        [float(state.get("f107", 150.0)) / 250.0],
    ])
    return obs.astype(np.float32), {"decay_main_km_d": decay_km_d,
                                    "soc_end_prop": soc_end,
                                    "ecl_frac": ecl_frac}


def _ecl_frac(samples, sun_n):
    if sun_n is None:
        return 0.35
    s = np.asarray(sun_n, float)
    dark = 0
    for rr, _vv in samples:
        if cpp.eclipse_cylindrical(np.asarray(rr, float), s, cpp.RE_WGS) < 0.5:
            dark += 1
    return float(dark) / max(len(samples), 1)


def attitude_bc_scale(panels_n, panels_A, q_att, q_ref, v_body_ref):
    def _proj(q):
        q = _unit_q(q)
        sig = q[1:4] / max(1.0 + q[0], 1e-9)
        c_bn = np.asarray(cpp.mrp_to_dcm(sig), float)
        v_b = c_bn @ np.asarray(v_body_ref, float)
        vh = v_b / max(np.linalg.norm(v_b), 1e-9)
        cosin = -(np.asarray(panels_n, float) @ vh)
        return float(np.sum(np.asarray(panels_A, float) * np.clip(cosin, 0.0, None)))
    ref = _proj(q_ref)
    if ref < 1e-9:
        return 1.0
    return float(np.clip(_proj(q_att) / ref, 0.05, 20.0))


def _score_candidate(q, state, cfg, panels=None):
    a = cfg["arbiter"]
    r = np.asarray(state["r"], float)
    v = np.asarray(state["v"], float)
    bc = float(state.get("bc_inv", 0.0)) * float(state.get("bc_scale", 1.0))
    r1, v1 = prop.propagate(r, v, a["arbiter_horizon_s"], dt_s=a["prop_dt_s"],
                            bc_inv=bc, rho0=state.get("rho", 3e-12),
                            alt0_m=state.get("alt_km", 400.0) * 1e3)
    sma0, sma1 = prop.sma_m(r, v), prop.sma_m(r1, v1)
    decay_km_d = ((sma0 - sma1) / 1e3 / max(a["arbiter_horizon_s"] / 86400.0, EPS)
                  if np.isfinite(sma0) and np.isfinite(sma1) else 0.0)
    return {
        "q": _unit_q(q),
        "decay_km_d": float(decay_km_d),
        "power_norm": float(state.get("power_norm", 0.0)),
        "downlink_norm": float(state.get("downlink_norm", 0.0)),
        "r_end": r1, "v_end": v1,
    }


# Duo arbitration, docs/modules/14_reward_v8.md §9.
def arbitrate(q_main, q_horizon, state_main, state_horizon, q_current, cfg=None):
    cfg = cfg or load("duo")
    a = cfg["arbiter"]

    cand_m = _score_candidate(q_main, state_main, cfg)
    cand_h = _score_candidate(q_horizon, state_horizon, cfg)

    feasible = {}
    for tag, cand in (("main", cand_m), ("horizon", cand_h)):
        ang = quat_angle_deg(cand["q"], q_current)
        cand["slew_deg"] = ang
        if ang <= float(a["max_slew_deg"]) + 1e-6:
            feasible[tag] = cand
    if not feasible:
        return _unit_q(q_current), {"choice": "hold", "reason": "both infeasible",
                                    "main": cand_m, "horizon": cand_h}
    if len(feasible) == 1:
        tag, cand = next(iter(feasible.items()))
        return cand["q"], {"choice": tag, "reason": "only feasible candidate",
                           "main": cand_m, "horizon": cand_h}

    soc = float(state_main.get("soc", 0.5))
    dl_soon = bool(state_main.get("downlink_soon", False))

    if soc < float(a["soc_critical"]):
        tag = max(feasible, key=lambda t: feasible[t]["power_norm"])
        return feasible[tag]["q"], {"choice": tag, "reason": "soc_critical",
                                    "main": cand_m, "horizon": cand_h}

    d_main, d_hor = cand_m["decay_km_d"], cand_h["decay_km_d"]
    if abs(d_main - d_hor) > float(a["decay_override_km_d"]):
        tag = "main" if d_main < d_hor else "horizon"
        if tag in feasible:
            return feasible[tag]["q"], {"choice": tag, "reason": "decay_override",
                                        "main": cand_m, "horizon": cand_h}

    if soc > float(a["soc_comfortable"]) and dl_soon:
        tag = max(feasible, key=lambda t: feasible[t]["downlink_norm"])
        return feasible[tag]["q"], {"choice": tag, "reason": "downlink_priority",
                                    "main": cand_m, "horizon": cand_h}

    def norm(lo, hi, x, lower_better):
        if abs(hi - lo) < EPS:
            return 0.5
        u = (x - lo) / (hi - lo)
        return 1.0 - u if lower_better else u

    dl_lo, dl_hi = min(d_main, d_hor), max(d_main, d_hor)
    scores = {}
    for tag, cand in feasible.items():
        scores[tag] = (float(a["w_decay"]) * norm(dl_lo, dl_hi, cand["decay_km_d"], True)
                       + float(a["w_power"]) * cand["power_norm"]
                       + float(a["w_downlink"]) * cand["downlink_norm"])
    best = max(scores, key=scores.get)
    if best == "horizon" and "main" in scores:
        if scores["horizon"] - scores["main"] < float(a["switch_margin"]):
            best = "main"
    return feasible[best]["q"], {"choice": best, "reason": "weighted",
                                 "scores": scores, "main": cand_m,
                                 "horizon": cand_h}


def pick_duo_models(ledger_rows, cfg=None):
    cfg = cfg or load("duo")
    sel = cfg["selection"]
    pool = [r for r in ledger_rows if str(r.get("variant", "")).startswith("v8b")]
    if not pool:
        raise ValueError("no v8b runs in ledger — train the size sweep first")

    main_pool = pool
    if sel.get("main_require_zero_brownouts", True):
        zero = [r for r in pool if float(r.get("brownouts", 0)) <= 0]
        if zero:
            main_pool = zero
    main = min(main_pool, key=lambda r: float(r[sel["main_metric"]]))
    secondary = max(pool, key=lambda r: float(r.get(sel["secondary_metric"], -1e9)))
    return {"main": main, "secondary": secondary}
