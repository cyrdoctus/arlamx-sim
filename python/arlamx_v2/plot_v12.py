"""SC_v12 conference eval, figures, and ADCS-loop animation.

Does not edit the plant. Reads the best s2 checkpoint, runs it and MPC_v2
on the same v10 protocol, writes plots under outputs/v12/plots/.

    PYTHONPATH=python python -m arlamx_v2.plot_v12 --all
    PYTHONPATH=python python -m arlamx_v2.plot_v12 --eval
    PYTHONPATH=python python -m arlamx_v2.plot_v12 --plot
    PYTHONPATH=python python -m arlamx_v2.plot_v12 --anim

Level: advanced.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from arlamx_v2 import cpp
from arlamx_v2.bench_v8 import mpc_fn
from arlamx_v2.config import load as load_cfg
from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts
from arlamx_v2.train_v12 import (
    EVAL_OPT, N_ORBITS_EVAL, REF_A, REF_D, VARIANT, V12Wrapper,
    _eval_wrapped, min_score,
)

V12 = OUTPUTS / "v12"
S2_CKPT = (V12 / "runs" / "s2_cmp_slew0.5_4x18_300k_s43" / "ckpts" / "step_300096")
V13_CKPT = OUTPUTS / "v13" / "ckpt"
V14_CKPT = OUTPUTS / "v13" / "ckpt_v14b"
BEST = S2_CKPT
PLOTS = V12 / "plots"
TRACES = V12 / "traces"
ANIM = V12 / "anim"
SLUG = "sc_v12"
LABEL = "SC_v12"
THREEWAY = False
FOURWAY = False
WRAP = dict(w_slew=0.5, w_env=0.1, use_cmp=True, clip_wrong_way=False,
            use_pwr_est=False, w_clip=0.0)
V13_WRAP = dict(w_slew=0.5, w_env=0.35, use_cmp=True, clip_wrong_way=True,
                use_pwr_est=True, w_clip=0.08)
WRAP_V14 = dict(w_slew=0.5, w_env=0.35, use_cmp=True, clip_wrong_way=True,
                use_pwr_est=True, w_clip=0.08, adapt_mrp=True)

C = {"MPC": "#2a78d6", "SC_v12": "#0f766e", "SC_v12+clip": "#7c3aed",
     "SC_v13": "#b45309", "SC_v14": "#6d28d9", "ref": "#8a8880"}
POLICY_ORDER = ("MPC", "SC_v12", "SC_v13", "SC_v14", "SC_v12+clip")


def _policy_names(rows):
    return [n for n in POLICY_ORDER if n in rows and n != "meta"]


def configure(v13=False, v14=False, ckpt=None, out_group=None, fourway=False):
    """Point eval/plots at SC_v12, SC_v13, or SC_v14.

    v14 figures write under outputs/v14/plots. Traces stay in outputs/v13/traces
    next to the v14 checkpoints.
    """
    global BEST, PLOTS, TRACES, ANIM, SLUG, LABEL, THREEWAY, FOURWAY, WRAP, V14_CKPT
    FOURWAY = bool(fourway or v14)
    if v14:
        BEST = Path(ckpt) if ckpt else V14_CKPT
        V14_CKPT = BEST
        group = OUTPUTS / (out_group or "v14")
        SLUG, LABEL, THREEWAY = "sc_v14", "SC_v14", True
        WRAP = dict(WRAP_V14)
        PLOTS = group / "plots"
        TRACES = OUTPUTS / "v13" / "traces"
        ANIM = group / "anim"
    elif v13:
        BEST = Path(ckpt) if ckpt else V13_CKPT
        group = OUTPUTS / (out_group or "v13")
        SLUG, LABEL, THREEWAY = "sc_v13", "SC_v13", True
        WRAP = dict(V13_WRAP)
        PLOTS = group / "plots"
        TRACES = group / "traces"
        ANIM = group / "anim"
    else:
        BEST = Path(ckpt) if ckpt else S2_CKPT
        group = OUTPUTS / (out_group or "v12")
        SLUG, LABEL, THREEWAY = "sc_v12", "SC_v12", False
        WRAP = dict(w_slew=0.5, w_env=0.1, use_cmp=True, clip_wrong_way=False,
                    use_pwr_est=False, w_clip=0.0)
        PLOTS = group / "plots"
        TRACES = group / "traces"
        ANIM = group / "anim"
    PLOTS.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)
    ANIM.mkdir(parents=True, exist_ok=True)
    print(f"[plot] {LABEL}  ckpt={BEST}  fourway={FOURWAY}  plots={PLOTS}",
          flush=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de", "#fcfcfb"

SCENARIOS = {
    "eval10": dict(opt=dict(EVAL_OPT), days=None, orbits=N_ORBITS_EVAL, jumps=[]),
    "nominal2d": dict(opt=dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001,
                               f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                               nu_deg=0.0, mass_kg=0.625, soc=0.5,
                               omega_dps=[1.0, 1.0, 0.5]),
                      days=2.0, orbits=None, jumps=[]),
    "storm1d": dict(opt=dict(altitude_km=350.0, inc_deg=23.0, ecc=0.001,
                             f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                             nu_deg=0.0, mass_kg=0.625, soc=0.5,
                             omega_dps=[1.0, 1.0, 0.5]),
                    days=1.0, orbits=None,
                    jumps=[(0.30, 130.0, 76.0), (0.70, -130.0, -76.0)]),
}

TRACE_KEYS = [
    "t_h", "sigma", "r", "v", "sun_N", "eclipse", "gates", "split",
    "tau_ctrl", "tau_dist", "tau_aero", "tau_want", "tau_env_filt",
    "tau_env_pwr", "F_est", "effort", "mtq_W", "mtq_quad_W", "soc", "gen",
    "gs_vis", "gs_cos", "downlink", "cmd_track_deg", "q_cmd", "cd",
    "sma_km", "alt_km", "f107", "ap", "omega_dps", "brownout",
    "deleg_B", "deleg_P", "n_clip", "kept", "dE_drag", "dE_lift", "dE_srp",
    "eta", "kd_eff", "shock", "margin_eff", "gate_floor_eff",
]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 12, "axes.titlesize": 13.5,
        "legend.fontsize": 11, "figure.dpi": 120,
    })


def _header(fig, title, subtitle, top=0.82):
    fig.subplots_adjust(top=top)
    fig.text(0.006, 0.985, title, fontsize=18, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.006, 0.928, subtitle, fontsize=11.5, color=INK2, ha="left", va="top")


def _save(fig, stem, note=None):
    if note:
        fig.text(0.006, 0.004, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf", flush=True)


def load_v12_predict(ckpt=BEST):
    from stable_baselines3 import PPO
    model = PPO.load(str(Path(ckpt) / "model.zip"))

    def predict(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return a
    return predict


def load_mpc_predict():
    cfg = load_cfg("mpc_v2", use_cache=False)
    return mpc_fn(cfg=cfg, noisy=True, seed=0)


def record_trace(predict, wrap_kw, scen_name, slug):
    scen = SCENARIOS[scen_name]
    base = ArlamxV2Env(seed=0, variant=VARIANT)
    env = V12Wrapper(base, **wrap_kw)
    period = 2 * np.pi * np.sqrt((6371e3 + float(scen["opt"]["altitude_km"]) * 1e3) ** 3
                                 / 3.986004418e14)
    if scen.get("days"):
        env.env._max_steps = int(scen["days"] * 86400.0 / env.env._advisor_s)
    else:
        env.env._max_steps = int(scen["orbits"] * period / env.env._advisor_s)
    obs, _ = env.reset(seed=1001, options=dict(scen["opt"]))
    env.env._weather_jumps = [
        (f * env.env._max_steps, df, da) for f, df, da in scen["jumps"]]
    env.env._srp_flashes = []
    H = {k: [] for k in TRACE_KEYS}
    infer_s, n_inf = 0.0, 0
    done, n = False, 0
    while not done:
        t0 = time.perf_counter()
        a = predict(obs, env.env)
        infer_s += time.perf_counter() - t0
        n_inf += 1
        a = np.asarray(a, np.float32)
        obs, _r, term, trunc, info = env.step(a)
        done = term or trunc
        n += 1
        st = env.env.sim.get_state()
        sigma = np.asarray(st["sigma"], float)
        q_cmd = a[:4] / max(float(np.linalg.norm(a[:4])), 1e-9)
        c_bn = cpp.mrp_to_dcm(sigma)
        c_cmd = cpp.mrp_to_dcm(np.asarray(
            [q_cmd[1], q_cmd[2], q_cmd[3]], float) / max(1.0 + q_cmd[0], 1e-9))
        r_err = c_cmd @ c_bn.T
        ang = float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(r_err) - 1.0),
                                                 -1.0, 1.0))))
        vis = float(info.get("gs_visible", 0.0))
        lit = float(info.get("eclipse", 1.0))
        cosg = float(info.get("gs_point_cos", 0.0))
        H["t_h"].append(n * env.env._advisor_s / 3600.0)
        H["sigma"].append(sigma)
        H["r"].append(np.asarray(st["r"], float))
        H["v"].append(np.asarray(st["v"], float))
        sun = st.get("sun_N", np.zeros(3))
        H["sun_N"].append(np.asarray(sun, float))
        H["eclipse"].append(lit)
        H["gates"].append(np.asarray(info.get("gates", np.ones(3)), float))
        H["split"].append(np.asarray(info.get("deleg_split", np.zeros(3)), float))
        H["tau_ctrl"].append(np.asarray(info.get("tau_ctrl_mean", np.zeros(3)), float))
        H["tau_dist"].append(np.asarray(info.get("tau_dist_est", np.zeros(3)), float))
        H["tau_aero"].append(np.asarray(info.get("tau_aero_mean", np.zeros(3)), float))
        H["tau_want"].append(np.asarray(info.get("tau_want", np.zeros(3)), float))
        H["tau_env_filt"].append(np.asarray(info.get("tau_env_filt", np.zeros(3)), float))
        H["tau_env_pwr"].append(np.asarray(info.get("tau_env_pwr", np.zeros(3)), float))
        H["F_est"].append(np.asarray(info.get("F_est", np.zeros(3)), float))
        H["effort"].append(float(info.get("torque_effort", 0.0)))
        H["mtq_W"].append(float(info.get("mtq_power_W", 0.0)))
        H["mtq_quad_W"].append(float(info.get("mtq_power_quad_W", 0.0)))
        H["soc"].append(float(info["battery_soc"]))
        H["gen"].append(float(info.get("power_gen_norm", 0.0)))
        H["gs_vis"].append(vis)
        H["gs_cos"].append(cosg)
        H["downlink"].append(1.0 if (vis > 0.5 and lit > 0.5 and cosg > 0.7) else 0.0)
        H["cmd_track_deg"].append(ang)
        H["q_cmd"].append(q_cmd)
        H["cd"].append(float(info.get("Cd", np.nan)))
        H["sma_km"].append(float(info["sma_m"]) / 1e3)
        H["alt_km"].append(float(info["altitude_km"]))
        H["f107"].append(float(env.env._f107))
        H["ap"].append(float(env.env._ap))
        H["omega_dps"].append(float(info.get("omega_dps", 0.0)))
        H["brownout"].append(1.0 if info.get("brownout_active") else 0.0)
        H["deleg_B"].append(float(info.get("deleg_B", 0.0)))
        H["deleg_P"].append(float(info.get("deleg_P", 0.0)))
        H["n_clip"].append(float(info.get("n_clip", 0.0)))
        H["kept"].append(float(info.get("kept_prev", 0.0)))
        H["dE_drag"].append(float(info.get("dE_drag", 0.0)))
        H["dE_lift"].append(float(info.get("dE_lift", 0.0)))
        H["dE_srp"].append(float(info.get("dE_srp", 0.0)))
        H["eta"].append(float(info.get("eta", 0.0)))
        H["kd_eff"].append(float(info.get("kd_eff", getattr(env.env, "_kd", 0.0))))
        H["shock"].append(float(info.get("shock", 0.0)))
        H["margin_eff"].append(float(info.get("margin_eff", 0.05)))
        H["gate_floor_eff"].append(float(info.get("gate_floor_eff",
                                                 getattr(env.env, "_gate_floor", 0.3))))
    env.close()
    out = {k: np.asarray(v) for k, v in H.items()}
    out["infer_ws_ms"] = np.array([infer_s / max(n_inf, 1) * 1e3])
    TRACES.mkdir(parents=True, exist_ok=True)
    dest = TRACES / f"{slug}__{scen_name}.npz"
    np.savez_compressed(dest, **out)
    print(f"  trace {slug}/{scen_name}: {n} steps  infer={out['infer_ws_ms'][0]:.3f} ms  "
          f"-> {dest.name}", flush=True)
    return dest


def run_eval():
    TRACES.mkdir(parents=True, exist_ok=True)
    infer = json.loads((V12 / "infer.json").read_text()) if (V12 / "infer.json").exists() else {}
    pol = load_v12_predict(BEST)
    mpc = load_mpc_predict()
    mpc_wrap = dict(w_slew=0.0, w_env=0.0, use_cmp=False,
                    clip_wrong_way=False, use_pwr_est=True, w_clip=0.0)
    s2_wrap = dict(w_slew=0.5, w_env=0.1, use_cmp=True,
                   clip_wrong_way=False, use_pwr_est=False, w_clip=0.0)

    rows = {}
    t0 = time.time()
    print("[eval] 10-orbit quiet  MPC_v2 (sensors on)", flush=True)
    rows["MPC"] = _eval_wrapped(mpc, **mpc_wrap)
    if FOURWAY:
        print("[eval] 10-orbit quiet  SC_v12 (s2)", flush=True)
        rows["SC_v12"] = _eval_wrapped(load_v12_predict(S2_CKPT), **s2_wrap)
        print("[eval] 10-orbit quiet  SC_v13 freeze", flush=True)
        rows["SC_v13"] = _eval_wrapped(load_v12_predict(V13_CKPT), **V13_WRAP)
        print("[eval] 10-orbit quiet  SC_v14 (dual-gated v14b)", flush=True)
        rows["SC_v14"] = _eval_wrapped(load_v12_predict(V14_CKPT), **WRAP_V14)
    elif THREEWAY:
        print("[eval] 10-orbit quiet  SC_v12 (s2 talk ckpt)", flush=True)
        rows["SC_v12"] = _eval_wrapped(load_v12_predict(S2_CKPT), **s2_wrap)
        print(f"[eval] 10-orbit quiet  {LABEL} (clip in the loop)", flush=True)
        rows[LABEL] = _eval_wrapped(pol, **WRAP)
    else:
        print("[eval] 10-orbit quiet  SC_v12 as trained", flush=True)
        rows["SC_v12"] = _eval_wrapped(pol, **WRAP)
        print("[eval] 10-orbit quiet  SC_v12 + wrong-way clip (zero-shot)", flush=True)
        rows["SC_v12+clip"] = _eval_wrapped(pol, **dict(V13_WRAP, w_clip=0.0))

    for k, agg in rows.items():
        sA, sD = min_score(agg, REF_A), min_score(agg, REF_D)
        agg["score_A"] = sA
        agg["score_D"] = sD
        print(f"  {k:14s}  A={sA}  D={sD}  dec={agg['decay_km_d']:.2f}  "
              f"band={agg['band_pct']:.1f}  dl={agg['downlink_min_d']:.1f}  "
              f"B={agg['deleg_B']:.3f}  clip={agg.get('clip_frac', 0):.2f}  "
              f"infer={agg['infer_ws_ms']:.3f}+{agg.get('infer_cmp_ms', 0):.3f} ms  "
              f"U575 pol={infer.get('v12_policy_u575_ms', 0.172):.3f}  "
              f"MPC U575={infer.get('mpc_u575_ms_published', 6.81):.2f}",
              flush=True)

    rows["meta"] = {
        "ckpt": str(BEST),
        "label": LABEL, "slug": SLUG,
        "protocol": "10-orbit quiet, EVAL_OPT 400 km / i=23 / F10.7=150 / Ap=4, v10 sensors on",
        "ref_A": REF_A, "ref_D": REF_D, "infer_u575": infer,
        "wall_s": round(time.time() - t0, 1),
        "note": (f"{LABEL} ckpt {BEST}. Clip and power residual are in the "
                 "training wrapper for SC_v13; s2 had neither."),
    }
    (TRACES / "eval10.json").write_text(json.dumps(rows, indent=2, default=str))

    print("[eval] traces nominal2d + storm1d", flush=True)
    record_trace(mpc, mpc_wrap, "nominal2d", "mpc")
    record_trace(mpc, mpc_wrap, "storm1d", "mpc")
    record_trace(pol, WRAP, "nominal2d", SLUG)
    record_trace(pol, WRAP, "storm1d", SLUG)
    record_trace(pol, WRAP, "eval10", SLUG)
    if FOURWAY:
        record_trace(load_v12_predict(S2_CKPT), s2_wrap, "storm1d", "sc_v12")
        record_trace(load_v12_predict(V13_CKPT), V13_WRAP, "storm1d", "sc_v13")
        record_trace(load_v12_predict(V14_CKPT), WRAP_V14, "storm1d", "sc_v14")
    print(f"[eval] done in {time.time() - t0:.0f}s", flush=True)
    return rows


def _load_eval():
    p = TRACES / "eval10.json"
    if not p.exists():
        raise SystemExit("run --eval first")
    return json.loads(p.read_text())


def _load_npz(slug, scen):
    f = TRACES / f"{slug}__{scen}.npz"
    return np.load(f) if f.exists() else None


def fig1_scorecard(rows):
    """Talk scorecard: frozen MPC_v2 column D vs best SC_v12 checkpoint.

    Do not mix the live 10-orbit MPC sample (one seed, quiet only, undersampled
    passes) with the freeze. Column D is the sensors-on reference the min-score
    is trained against.
    """
    _style()
    infer = rows.get("meta", {}).get("infer_u575", {})
    primary = rows.get(LABEL) or rows["SC_v12"]
    mc_path = TRACES / "mc_lifetime.json"
    mpc_days, rl_days = 17.9, float(primary["lifetime_d_500_300"])
    mc_note = f"MPC 17.9 d published MC; {LABEL} implied 200 km / 10-orbit decay"
    if mc_path.exists():
        mcj = json.loads(mc_path.read_text()).get("summary", {})
        if mcj.get("mpc", {}).get("days_500_to_300_mean"):
            mpc_days = float(mcj["mpc"]["days_500_to_300_mean"])
        key = "sc_v13" if "sc_v13" in mcj else "sc_v12"
        if mcj.get(key, {}).get("days_500_to_300_mean"):
            rl_days = float(mcj[key]["days_500_to_300_mean"])
        mc_note = (f"Days: this-package MC 24 runs, 28-day cap, 500→300 km "
                   f"(MPC n={mcj.get('mpc', {}).get('days_n')}, "
                   f"{key} n={mcj.get(key, {}).get('days_n')})")
    def pack(src, days=None):
        return {
            "decay_km_d": float(src["decay_km_d"]),
            "band_pct": float(src["band_pct"]),
            "downlink_min_d": float(src["downlink_min_d"]),
            "lifetime_d_500_300": float(days if days is not None else src["lifetime_d_500_300"]),
            "mtq_energy_J": float(src["mtq_energy_J"]),
            "infer_u575_ms": float(infer.get("v12_policy_plus_cmp_u575_ms", 1.00)),
        }
    mpc = {
        "decay_km_d": REF_D["decay_nominal"],
        "band_pct": REF_D["band_pct"],
        "downlink_min_d": REF_D["downlink_min_d"],
        "lifetime_d_500_300": mpc_days,
        "mtq_energy_J": REF_D["mtq_energy_J"],
        "infer_u575_ms": float(infer.get("mpc_u575_ms_published", 6.81)),
    }
    rl = pack(primary, rl_days)
    entries = [("MPC_v2", mpc, C["MPC"])]
    if FOURWAY:
        mc4p = OUTPUTS / "v13" / "traces" / "mc_four.json"
        mc4 = json.loads(mc4p.read_text()).get("summary", {}) if mc4p.exists() else {}
        alias = {"SC_v13": "SC_v13"}
        for cand in ("SC_v14e", "SC_v14d", "SC_v14b", "SC_v14"):
            if cand in mc4:
                alias["SC_v14"] = cand
                break
        for name in ("SC_v12", "SC_v13", "SC_v14"):
            if name not in rows:
                continue
            days = None
            lab = alias.get(name)
            if lab and lab in mc4 and (mc4[lab].get("days_500_to_300") or {}).get("mean"):
                days = float(mc4[lab]["days_500_to_300"]["mean"])
            entries.append((name, pack(rows[name], days), C[name]))
    else:
        if THREEWAY and "SC_v12" in rows and LABEL != "SC_v12":
            entries.append(("SC_v12", pack(rows["SC_v12"]), C["SC_v12"]))
        entries.append((LABEL, rl, C.get(LABEL, C["SC_v12"])))
    panels = [
        ("decay_km_d", "Orbit decay (km/day)\nlower better", "%.2f", True),
        ("band_pct", "Battery in 40–60 % band\n% of time, higher better", "%.1f", False),
        ("downlink_min_d", "Downlink contact (min/day)\nhigher better", "%.1f", False),
        ("lifetime_d_500_300", "Days in orbit 500→300 km\nthis-package Monte-Carlo", "%.1f", False),
        ("mtq_energy_J", "Coil energy (J)\nlower better", "%.2f", True),
        ("infer_u575_ms", "STM32U575 decision (ms)\nanalytic, 0.35 FLOP/cycle", "%.2f", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.4))
    fig.subplots_adjust(top=0.80, hspace=0.48, wspace=0.28,
                        left=0.06, right=0.98, bottom=0.10)
    x = np.arange(len(entries))
    for ax, (key, title, fmt, lower) in zip(axes.ravel(), panels):
        vals = [float(e[1][key]) for e in entries]
        ax.bar(x, vals, color=[e[2] for e in entries], width=0.62,
               edgecolor=SURF, linewidth=2, zorder=3)
        best = (min if lower else max)(vals)
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(vals) * 0.045, fmt % v, ha="center", va="bottom",
                    fontsize=12, color=INK,
                    fontweight="bold" if abs(v - best) < 1e-9 else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in entries], fontsize=11)
        ax.set_title(title, pad=8, fontsize=12.0, color=INK)
        ax.set_ylim(0, max(vals) * 1.32 if max(vals) > 0 else 1)
        ax.grid(axis="x", visible=False)
    fig.text(0.006, 0.985, f"{LABEL} vs sampling MPC — mission targets and onboard compute",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.006, 0.928,
             "MPC_v2 = frozen sensors-on column D (9.19 km/d, 84.1 % band, 18.6 min/d, 3.73 J, 6.81 ms).  "
             f"{LABEL} = {BEST.name} on the 10-orbit quiet eval.  "
             "U575: policy 0.17 ms, policy+comparator 1.00 ms.",
             fontsize=11, color=INK2, ha="left", va="top")
    _save(fig, "fig1_scorecard",
          mc_note + ". Coil energy: freeze is 3-probe; RL bars are 10-orbit quiet. Always quote U575.")


RADAR_RMAX = 2.0


def _clip_radar(vals):
    """Keep every spoke on or inside the outer ring (linewidth sits on the rim)."""
    return [float(np.clip(v, 0.0, RADAR_RMAX)) for v in vals]


def _radar_axes(ax):
    ax.set_ylim(0, RADAR_RMAX)
    try:
        ax.set_rmax(RADAR_RMAX)
    except Exception:
        pass
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5", "1.0 match", "1.5", "2.0"], color=INK2, fontsize=9)


def fig2_radar(rows):
    _style()
    # Oriented ratios vs column D (sensors-on freeze), 1.0 = match MPC_v2.
    ref = REF_D
    labels = ["decay\n(lower)", "band", "downlink", "B\nalign", "coil J\n(lower)",
              "compute\n(lower)"]
    def orient(agg):
        dec = (1.1 * ref["decay_nominal"]) / max(float(agg["decay_km_d"]), 0.05)
        band = float(agg["band_pct"]) / max(0.9 * ref["band_pct"], 1.0)
        dl = float(agg["downlink_min_d"]) / max(ref["downlink_min_d"], 0.1)
        # MPC B = 0 maps to 1.0 on the freeze ring; +0.50 → 2.0.
        b = (float(agg.get("deleg_B", 0.0)) + 0.5) / 0.5
        mtq = ref["mtq_energy_J"] / max(float(agg["mtq_energy_J"]), 0.01)
        inf = max(float(agg.get("infer_total_ms", agg.get("infer_ws_ms", 0.15))), 1e-6)
        cmpu = 6.81 / inf
        return _clip_radar([dec, band, dl, b, mtq, cmpu / 10.0]), cmpu
    fig, ax = plt.subplots(figsize=(8.6, 8.2), subplot_kw=dict(polar=True))
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    # 1.0 ring is the sensors-on freeze, not the live 10-orbit MPC sample
    # (that sample undersamples GS and is not column D).
    freeze = [1.0] * (len(labels) + 1)
    ax.plot(ang, freeze, color=C["MPC"], lw=1.6, ls="--", label="MPC freeze (1.0)",
            clip_on=True, zorder=2)
    for name in [n for n in ("SC_v12", "SC_v13", "SC_v14") if n in rows]:
        vals, xcomp = orient(rows[name])
        vals = vals + vals[:1]
        ax.plot(ang, vals, color=C[name], lw=2.2, label=f"{name}", clip_on=True)
        ax.fill(ang, vals, color=C[name], alpha=0.12, clip_on=True)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    _radar_axes(ax)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.28, 1.12))
    prim = rows.get(LABEL) or rows.get("SC_v12") or rows["MPC"]
    x12 = 6.81 / max(float(prim.get("infer_total_ms", 0.15)), 1e-6)
    fig.text(0.5, 0.04,
             f"Compute spoke is (MPC 6.81 ms) / (10 × t_decision), clipped at 2.  "
             f"True speedup {LABEL} vs MPC on the workstation: {x12:.0f}×.  "
             f"U575: 6.81 ms vs 0.17 ms policy / 1.00 ms with comparator.",
             ha="center", fontsize=9, color=INK2)
    fig.suptitle("Match to the sensors-on MPC freeze (column D)", fontsize=16,
                 fontweight="bold", y=0.98)
    _save(fig, "fig2_radar",
          "Spokes are oriented ratios vs MPC_v2 freeze (9.19 km/d, 84.1 % band, 18.6 min/d, "
          "B = 0, 3.73 J). A spoke at 1.0 matches the freeze; above 1.0 beats it. "
          "B spoke: (B + 0.5) / 0.5 so alignment is visible on the same ring.")


def fig3_power(rows):
    _style()
    names = _policy_names(rows)
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.8))
    keys = [
        ("deleg_P", "Delegation power saved P\n(vs full-authority PD)", "%.2f", False),
        ("coil_idle_frac", "Coil-idle fraction\n(steps with ~0 MTQ power)", "%.2f", False),
        ("mtq_energy_J", "Coil energy, 10-orbit quiet\n(J, lower better)", "%.2f", True),
        ("deleg_B", "Delegation benefit B\n(+ env helping, − fighting)", "%+.2f", False),
    ]
    x = np.arange(len(names))
    for ax, (key, title, fmt, _) in zip(axes, keys):
        vals = [float(rows[n][key]) for n in names]
        ax.bar(x, vals, color=[C[n] for n in names], width=0.66, edgecolor=SURF, lw=2, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ymax = max(abs(v) for v in vals)
        for xi, v in enumerate(vals):
            off = 0.04 * max(ymax, 0.05) * (1 if v >= 0 else -1)
            ax.text(xi, v + off, fmt % v, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=11, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_title(title, pad=10, fontsize=11.5)
        ax.grid(axis="x", visible=False)
        if key == "deleg_B":
            ax.set_ylim(-max(0.5, ymax * 1.4), max(0.5, ymax * 1.4))
        else:
            lo = min(0, min(vals))
            ax.set_ylim(lo * 1.2 if lo < 0 else 0, max(vals) * 1.35 + 0.05)
    _header(fig, "Environment actuation — B is the quantity s3/s4 train on",
            "P is power the authority gate withheld vs a full PD. B is sign(τ_env · τ_want) weighted by split. "
            "SC_v12 (s2) has B < 0 (wrong-way). SC_v13/v14 train with the clip so B stays positive.",
            top=0.80)
    _save(fig, "fig3_env_actuation",
          "MPC pads split to s=0, so P and B are ~0 by construction. PEI vs full PD is ~0.998 for every policy "
          "on this quiet probe (aero ≪ PD demand) and is not shown. +clip is zero-shot, not a new network.")


def fig4_timeline():
    _style()
    mpc = _load_npz("mpc", "nominal2d")
    rl = _load_npz(SLUG, "nominal2d")
    if mpc is None or rl is None:
        print("  [skip] fig4: missing nominal2d traces")
        return
    fig, axes = plt.subplots(4, 1, figsize=(13.6, 9.0), sharex=True)
    for d, name, lw in ((mpc, "MPC", 1.8), (rl, LABEL, 1.9)):
        t = d["t_h"]
        axes[0].plot(t, d["soc"] * 100, color=C[name], lw=lw, label=name)
        axes[1].fill_between(t, 0, d["downlink"] * 100, color=C[name], alpha=0.35, step="mid")
        axes[2].plot(t, d["sma_km"] - d["sma_km"][0], color=C[name], lw=lw)
        axes[3].plot(t, np.maximum(np.asarray(d["mtq_quad_W"]) * 1e3, 1e-4),
                     color=C[name], lw=1.3)
    axes[0].axhspan(40, 60, color="#1baf7a", alpha=0.10)
    axes[0].set_ylabel("SoC (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    axes[1].set_ylabel("downlink")
    axes[1].set_yticks([0, 100])
    axes[1].set_yticklabels(["off", "on"])
    axes[2].set_ylabel("Δ SMA (km)")
    axes[3].set_ylabel("MTQ power (mW)")
    axes[3].set_yscale("log")
    axes[3].set_ylim(1e-4, 8)
    axes[3].set_xlabel("mission time (h)")
    _header(fig, "Two-day quiet mission — power, contact, decay, coils",
            "Identical initial orbit (400 km, i = 23°, F10.7 = 150). Shaded band is the 40–60 % supercap target. "
            "Downlink requires a sunlit pass with boresight cosine > 0.7.",
            top=0.90)
    _save(fig, "fig4_mission_timeline",
          f"Source: outputs/{PLOTS.parent.name}/traces/{{mpc,{SLUG}}}__nominal2d.npz. "
          "Quadratic I²R coil power. Quiet F10.7=150 — not the MC.")


def fig5_adcs_loop():
    _style()
    d = _load_npz(SLUG, "storm1d")
    if d is None:
        d = _load_npz(SLUG, "nominal2d")
    if d is None:
        print(f"  [skip] fig5: no {SLUG} trace")
        return
    t = d["t_h"]
    storm = np.asarray(d["ap"]) > 40.0
    has_kd = "kd_eff" in d.files if hasattr(d, "files") else "kd_eff" in d
    n_ax = 6 if has_kd else 5
    fig, axes = plt.subplots(n_ax, 1, figsize=(13.8, 2.1 * n_ax + 1.2), sharex=True)
    for ax in axes:
        if np.any(storm):
            ax.fill_between(t, 0, 1, where=storm, color="#e34948", alpha=0.08,
                            transform=ax.get_xaxis_transform())
    axes[0].plot(t, np.linalg.norm(d["tau_aero"], axis=1) * 1e6, color="0.45",
                 lw=1.1, label="plant aero torque")
    axes[0].plot(t, np.linalg.norm(d["tau_env_filt"], axis=1) * 1e6, color="#8b5cf6",
                 lw=1.6, ls=(0, (3, 2)), label="KF τ_env")
    axes[0].plot(t, np.linalg.norm(d["tau_env_pwr"], axis=1) * 1e6, color="#0f766e",
                 lw=1.3, label="power/geometry τ_env")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("torque (µN·m)")
    axes[0].legend(frameon=False, ncol=3, loc="lower right", fontsize=9.5)
    for i, (al, col) in enumerate(zip("XYZ", ("#2a78d6", "#eb6834", "#1baf7a"))):
        axes[1].step(t, d["split"][:, i], where="mid", color=col, lw=1.3, label=f"s {al}")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel("delegation split")
    axes[1].legend(frameon=False, ncol=3, loc="upper right", fontsize=9.5)
    B = np.asarray(d["deleg_B"], float)
    axes[2].axhline(0, color=MUTED, lw=0.8)
    axes[2].fill_between(t, 0, B, where=B >= 0, color="#0f766e", alpha=0.55, step="mid",
                         label="B > 0  environment helping")
    axes[2].fill_between(t, 0, B, where=B < 0, color="#e34948", alpha=0.55, step="mid",
                         label="B < 0  environment fighting")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel("benefit B")
    axes[2].legend(frameon=False, ncol=2, loc="upper right", fontsize=9.5)
    axes[3].plot(t, np.asarray(d["mtq_quad_W"]) * 1e3, color="#4a3aa7", lw=1.3)
    axes[3].set_ylabel("MTQ (mW)")
    axes[4].axhspan(40, 60, color="#1baf7a", alpha=0.10)
    axes[4].plot(t, d["soc"] * 100, color="#eda100", lw=1.5)
    axes[4].set_ylabel("SoC (%)")
    axes[4].set_ylim(0, 105)
    if has_kd:
        axes[5].plot(t, np.asarray(d["kd_eff"]) * 1e3, color="#6d28d9", lw=1.5,
                     label="kd ×10³")
        if "eta" in (d.files if hasattr(d, "files") else d):
            axr = axes[5].twinx()
            axr.plot(t, d["eta"], color="#ea580c", lw=1.1, alpha=0.85, label="η")
            axr.set_ylabel("η")
            axr.set_ylim(0, 1.05)
            axr.spines["top"].set_visible(False)
        axes[5].set_ylabel("kd ×10³")
        axes[5].legend(frameon=False, loc="upper right", fontsize=9)
        axes[5].set_xlabel("mission time (h)")
    else:
        axes[4].set_xlabel("mission time (h)")
    _header(fig, f"ADCS loop on {LABEL} — sense, estimate, delegate, act",
            "Observer is the datasheet-R torque Kalman filter; the power/geometry residual "
            "adds back the along-B torque the coils cannot produce. "
            + ("Bottom: SC_v14 scheduled kd and η." if has_kd else
               "Split s=1 is gate floor 0.3."),
            top=0.92)
    _save(fig, "fig5_adcs_loop",
          f"Storm segment if present (Ap>40 shaded). Source: outputs/{PLOTS.parent.name}/traces/"
          f"{SLUG}__storm1d.npz (falls back to nominal2d).")


def fig6_infer(rows):
    _style()
    infer = rows.get("meta", {}).get("infer_u575", {})
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    labels = ["MPC\n(U575 analytic)", f"{LABEL} policy\n(U575 analytic)",
              f"{LABEL} + comparator\n(U575 analytic)",
              f"{LABEL} workstation\n(measured)"]
    vals = [
        float(infer.get("mpc_u575_ms_published", 6.81)),
        float(infer.get("v12_policy_u575_ms", 0.172)),
        float(infer.get("v12_policy_plus_cmp_u575_ms", 1.00)),
        float((rows.get(LABEL) or rows["SC_v12"]).get(
            "infer_total_ms", (rows.get(LABEL) or rows["SC_v12"])["infer_ws_ms"])),
    ]
    cols = [C["MPC"], C["SC_v12"], "#4a3aa7", "#eda100"]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cols, width=0.62, edgecolor=SURF, lw=2, zorder=3)
    ax.axhline(15.0, color="#e34948", ls="--", lw=1.2, label="15 ms flight budget")
    for xi, v in enumerate(vals):
        ax.text(xi, v + max(vals) * 0.03, f" {v:.3f} ms", ha="center", va="bottom",
                fontsize=12, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("milliseconds per 300 s decision")
    ax.set_ylim(0, max(16.5, max(vals) * 1.35))
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    speed = vals[0] / max(vals[1], 1e-9)
    _header(fig, "Onboard compute — STM32U575 analytic vs workstation sample",
            f"U575 at 160 MHz, 0.35 FLOP/cycle. Policy is 4×18 MLP, 83-D obs, 7-D action. "
            f"Analytic speedup vs MPC: {speed:.0f}× (policy only).",
            top=0.80)
    _save(fig, "fig6_inference",
          "Analytic budget is the flight number. Workstation times are Threadripper samples of the "
          "Python policy forward (+ comparator when enabled); they are not the U575 number.")


def fig7_targets(rows):
    """Compact 2-slide option: one chart of every mission target vs MPC."""
    _style()
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    metrics = [
        ("decay_km_d", "Decay\n(km/d)", True),
        ("band_pct", "Band\n(%)", False),
        ("downlink_min_d", "Downlink\n(min/d)", False),
        ("lifetime_d_500_300", "Days\n500→300", False),
        ("mtq_energy_J", "Coil\nenergy (J)", True),
        ("infer_total_ms", "Decision\n(ms)", True),
    ]
    names = [n for n in ("MPC", "SC_v12", "SC_v13", "SC_v14") if n in rows]
    x = np.arange(len(metrics))
    width = 0.78 / max(len(names), 1)
    for i, name in enumerate(names):
        vals = []
        for key, _lab, lower in metrics:
            raw = float(rows[name].get(key, rows[name].get("infer_ws_ms", 0)))
            vals.append(raw)
        off = (i - (len(names) - 1) / 2.0) * width
        ax.bar(x + off, vals, width, color=C[name], label=name,
               edgecolor=SURF, lw=1.5, zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02 * max(vals), f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.0, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    _header(fig, f"Mission targets on one axis — {LABEL} vs MPC",
            "Same 10-orbit quiet probe. Units differ per cluster; read the number on the bar, not the height across clusters.",
            top=0.82)
    _save(fig, "fig7_targets_overlay",
          "Prefer fig1_scorecard for the talk: that one has a separate panel per unit. This overlay is a backup slide option.")


def write_slide_architecture_png():
    """Onboard ADCS flowchart. Boxes are opaque; figure background is transparent."""
    _style()
    fig, ax = plt.subplots(figsize=(16.8, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    def box(x, y, w, h, title, lines, fc, tc="#ffffff", title_size=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                    fc=fc, ec="#334155", lw=1.4, alpha=1.0))
        ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=tc)
        for i, ln in enumerate(lines):
            ax.text(x + 0.012, y + h - 0.062 - 0.028 * i, ln, ha="left", va="top",
                    fontsize=9.5, color="#0f172a")

    ax.text(0.012, 0.97, "Onboard ADCS — sensors, filter, regime, Deep RL, clips, act",
            fontsize=16, fontweight="bold", va="top", color=INK)
    ax.text(0.012, 0.915,
            "Decision every 300 s. Inner MRP-PD at 2 s. No LSTM. REGIME is classical "
            "(η / shock → kd, gate floor, clip κ, comparator margin). Fly FP32.",
            fontsize=11, va="top", color=INK2)

    box(0.012, 0.58, 0.14, 0.28, "SENSE",
        ["ICM-42688-P  100 Hz", "MMC5983MA ×2", "Orion B16-01 GNSS", "EPS  SoC, P_gen"],
        "#3b82f6")
    box(0.168, 0.58, 0.15, 0.28, "FILTER",
        ["MRP σ, ω", "Torque KF  (CV)", "R from gyro spec", "τ_env = 0.7 KF+0.3 P"],
        "#8b5cf6")
    box(0.334, 0.58, 0.175, 0.28, "REGIME  (v14)",
        ["η = |τ_env|/5.0 µN·m", "η_slow  ~1 orbit EWMA", "shock on jump",
         "schedules kd, floor, κ"],
        "#6d28d9")
    box(0.525, 0.58, 0.175, 0.28, "DECIDE  Deep RL",
        ["stacked hist, not LSTM", "PPO  4×18  FP32", "83-D → q + split",
         "0.17 ms policy / 1.00 ms +cmp"],
        "#10b981")
    box(0.716, 0.58, 0.27, 0.28, "SAFETY CLIPS",
        ["slew ≤ 40° / step", "comparator H=2  0.05(1+2η)",
         "wrong-way: s ← s(1−κ)", "κ 0.70 quiet → 1.0 in storm"],
        "#f59e0b", tc="#0f172a")
    box(0.168, 0.16, 0.30, 0.30, "ALLOCATE",
        ["MRP-PD  τ = −kp σ_e − kd ω + ω×Iω",
         "kd 1.0e-3 quiet → 0.60e-3 storm",
         "gate_floor 0.30 → 0.15",
         "kp 6e-5  (yaml 1.6e-3 stays on disk)"],
        "#0ea5e9")
    box(0.490, 0.16, 0.24, 0.30, "ACT",
        ["MTQ  1.473 / 1.473 / 0.442 A·m²", "38.24 / 38.24 / 11.48 µN·m",
         "I²R  249 mW  3-axis full", "1 cm cp–cm; env turns delegated axes"],
        "#ef4444")
    box(0.755, 0.16, 0.23, 0.30, "PLANT + MODE",
        ["Sentman aero · panel SRP", "GGM03S · IGRF/WMM",
         "B-dot on brownout", "advisor resumes SoC ≥ 15 %"],
        "#64748b")
    ax.annotate("", xy=(0.168, 0.72), xytext=(0.152, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8))
    ax.annotate("", xy=(0.334, 0.72), xytext=(0.318, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8))
    ax.annotate("", xy=(0.525, 0.72), xytext=(0.509, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8))
    ax.annotate("", xy=(0.716, 0.72), xytext=(0.700, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8))
    ax.annotate("", xy=(0.318, 0.46), xytext=(0.850, 0.58),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.6))
    ax.annotate("", xy=(0.490, 0.31), xytext=(0.468, 0.31),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8))
    ax.annotate("", xy=(0.755, 0.31), xytext=(0.730, 0.31),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8))
    ax.text(0.015, 0.05,
            "Closed loop: plant measurements feed SENSE. τ_env is gyro + commanded dipole (I²R), "
            "never plant-truth aero. REGIME is classical (few dozen FLOPs), not a second network. "
            "U575: 6.81 ms MPC vs 1.00 ms policy+comparator (6.8×), budget 15 ms.",
            fontsize=10, color=INK2)
    PLOTS.mkdir(parents=True, exist_ok=True)
    kw = dict(dpi=220, bbox_inches="tight", transparent=True,
              facecolor="none", edgecolor="none")
    for stem in ("fig0_architecture", "key_architecture"):
        fig.savefig(PLOTS / f"{stem}.png", **kw)
        fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight", transparent=True,
                    facecolor="none", edgecolor="none")
    plt.close(fig)
    print("  wrote fig0_architecture.png/.pdf (transparent) and key_architecture",
          flush=True)


def fig_B_compare():
    """Where the environment helps: B(t) on the same storm1d for v12/v13/v14."""
    _style()
    series = []
    for slug, name in (("sc_v12", "SC_v12"), ("sc_v13", "SC_v13"),
                       ("sc_v14", "SC_v14")):
        d = _load_npz(slug, "storm1d")
        if d is None:
            continue
        series.append((name, d))
    if len(series) < 2:
        print("  [skip] fig_B_compare: need ≥2 storm1d traces")
        return
    fig, axes = plt.subplots(len(series) + 1, 1, figsize=(13.8, 2.2 * (len(series) + 1)),
                             sharex=True)
    t0 = series[0][1]["t_h"]
    storm = np.asarray(series[0][1]["ap"]) > 40.0
    ax0 = axes[0]
    ax0.plot(t0, series[0][1]["ap"], color="#e34948", lw=1.6)
    ax0.set_ylabel("Ap")
    ax0.set_ylim(0, max(200, float(np.max(series[0][1]["ap"])) * 1.1))
    for ax in axes:
        if np.any(storm):
            ax.fill_between(t0, 0, 1, where=storm, color="#e34948", alpha=0.08,
                            transform=ax.get_xaxis_transform())
    for ax, (name, d) in zip(axes[1:], series):
        t = d["t_h"]
        B = np.asarray(d["deleg_B"], float)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.fill_between(t, 0, B, where=B >= 0, color=C[name], alpha=0.55, step="mid")
        ax.fill_between(t, 0, B, where=B < 0, color="#e34948", alpha=0.45, step="mid")
        mean = float(np.mean(B))
        ax.set_ylabel(f"{name}\nB")
        ax.set_ylim(-1.05, 1.05)
        ax.text(0.99, 0.88, f"mean {mean:+.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=INK)
    axes[-1].set_xlabel("mission time (h)")
    _header(fig, "Where the environment helps — B on the same storm day",
            "Green/colour = τ_env aligned with the turn (environment as actuator). "
            "Red = fighting. Shaded = Ap > 40. SC_v12 is wrong-way; v13/v14 stay positive.",
            top=0.90)
    _save(fig, "fig_B_storm_compare",
          "Same storm1d protocol (350 km, two F10.7/Ap jumps). Companion to the radar.")


def fig_ipc_stills():
    """PDF stand-in for the animation: 3D spacecraft at quiet / onset / storm."""
    from arlamx_v2 import ipc_animation as IA
    src = TRACES / f"{SLUG}__storm1d.npz"
    if not src.exists():
        print("  [skip] fig_ipc_stills: no storm1d trace")
        return
    d = dict(np.load(src))
    cache = OUTPUTS / "analysis" / "showcase" / "geometry_simplified.npz"
    if cache.exists():
        g = np.load(cache)
        tris, tri_n = g["tris"], g["tri_normals"]
    else:
        tris, tri_n = IA.simplified_model()
    segs = IA.find_segments(d)
    if "storm_onset" in segs:
        i0, i1 = segs["storm_onset"]
    else:
        i0, i1 = 0, len(d["t_h"])
    picks = [i0, (i0 + i1) // 2, max(i0, i1 - 1)]
    labels = ["before storm", "storm onset", "in storm"]
    ext = np.abs(tris).max() * 1.30
    s = ext * 1.15
    fig = plt.figure(figsize=(14.8, 5.4))
    for col, (k, lab) in enumerate(zip(picks, labels)):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        sig = np.asarray(d["sigma"][k], float)
        q = IA._mrp_to_quat(sig)
        c_bn = IA._quat_dcm_bn(q)
        c_nb = c_bn.T
        world = tris @ c_nb.T
        lit = float(d["eclipse"][k]) > 0.5
        sun_b = c_bn @ d["sun_N"][k]
        base_rgb = np.array([0.95, 0.76, 0.31])
        dark_rgb = np.array([0.36, 0.39, 0.44])
        if lit:
            shade = 0.35 + 0.65 * np.abs(tri_n @ sun_b)
            colors = np.clip(shade[:, None] * base_rgb[None, :], 0, 1)
        else:
            colors = np.tile(dark_rgb, (len(tris), 1))
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        coll = Poly3DCollection(list(world), linewidths=0.12,
                                edgecolors=(0, 0, 0, 0.25))
        coll.set_facecolor(np.c_[colors, np.full(len(colors), 0.95)])
        ax.add_collection3d(coll)
        sun = d["sun_N"][k]
        v = d["v"][k] / max(np.linalg.norm(d["v"][k]), 1e-9)
        z_b = c_nb @ np.array([0.0, 0.0, 1.0])
        tau = d["tau_dist"][k]
        tau_nrm = c_nb @ (tau / max(np.linalg.norm(tau), 1e-12))
        ax.quiver(0, 0, 0, *(sun * s), color="#eda100", lw=1.8, arrow_length_ratio=0.12)
        ax.quiver(0, 0, 0, *(v * s * 0.92), color="0.45", lw=1.4, arrow_length_ratio=0.12)
        ax.quiver(0, 0, 0, *(z_b * s * 0.98), color="#2a78d6", lw=1.8, arrow_length_ratio=0.14)
        ax.quiver(0, 0, 0, *(tau_nrm * s * 1.06), color="#e34948", lw=2.2,
                  arrow_length_ratio=0.16)
        ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_zlim(-ext, ext)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=35)
        state = "storm" if float(d["ap"][k]) > 40 else "quiet"
        ax.set_title(f"{lab}\nt = {d['t_h'][k]:.1f} h · Ap {d['ap'][k]:.0f} ({state})",
                     fontsize=11, color=INK, pad=4)
    fig.suptitle(f"IPC stills — {LABEL}  (PDF stand-in for the animation)",
                 fontsize=15, fontweight="bold", y=0.98, color=INK)
    fig.text(0.5, 0.02,
             "Gold = Sun, grey = velocity, blue = +Z antenna, red = onboard disturbance estimate. "
             "Same 3D view as the mp4; these three frames sit in the PDF.",
             ha="center", fontsize=9, color=INK2)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for extn in ("png", "pdf"):
        fig.savefig(PLOTS / f"fig_ipc_stills.{extn}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_ipc_stills.png/.pdf", flush=True)


TALK_U575_MPC = 6.81
TALK_U575_RL = 1.00
TALK_COLS = {
    "MPC": "#2a78d6", "SC_v12": "#0f766e", "SC_v13": "#b45309",
    "SC_v14b": "#6d28d9", "SC_v14d": "#155e75", "SC_v14e": "#be185d",
    "SC_v14f": "#9f1239", "SC_v14": "#6d28d9",
}


def _talk_v14_ckpts():
    out = {}
    for tag, folder in (("SC_v14b", "ckpt_v14b"), ("SC_v14d", "ckpt_v14d"),
                        ("SC_v14e", "ckpt_v14e"), ("SC_v14f", "ckpt_v14f")):
        p = OUTPUTS / "v13" / folder
        if (p / "model.zip").exists():
            out[tag] = p
    return out


def pick_best_v14(n=2):
    """Two v14 zips for the talk radar: fewest MC brownouts, then days, then J."""
    path = OUTPUTS / "v13" / "traces" / "mc_four.json"
    ck = _talk_v14_ckpts()
    if not path.exists():
        names = [k for k in ("SC_v14d", "SC_v14b", "SC_v14f", "SC_v14e") if k in ck]
        return names[:n]
    s = json.loads(path.read_text()).get("summary") or {}
    scored = []
    for name in ck:
        if name not in s:
            scored.append((1e9, -1e9, 1e9, name))
            continue
        br = float(s[name].get("brownouts_any") or 99)
        days = ((s[name].get("days_500_to_300") or {}).get("mean") or 0.0)
        j = ((s[name].get("mtq_J_per_day") or {}).get("mean") or 99.0)
        scored.append((br, -days, j, name))
    scored.sort()
    return [t[-1] for t in scored[:n]]


def run_talk_eval(v14_names=None):
    """Quiet 10-orbit + storm1d traces for the two-plot talk set."""
    s2_wrap = dict(w_slew=0.5, w_env=0.1, use_cmp=True,
                   clip_wrong_way=False, use_pwr_est=False, w_clip=0.0)
    specs = {
        "SC_v12": (S2_CKPT, s2_wrap),
        "SC_v13": (V13_CKPT, V13_WRAP),
    }
    for tag, folder in _talk_v14_ckpts().items():
        specs[tag] = (folder, dict(WRAP_V14))
    if v14_names:
        keep = {"SC_v12", "SC_v13", *v14_names}
        specs = {k: v for k, v in specs.items() if k in keep}
    rows = {}
    t0 = time.time()
    infer = json.loads((V12 / "infer.json").read_text()) if (V12 / "infer.json").exists() else {}
    print("[talk] quiet 10-orbit evals", flush=True)
    for name, (ckpt, wrap) in specs.items():
        pred = load_v12_predict(ckpt)
        agg = _eval_wrapped(pred, **wrap)
        agg["u575_ms"] = TALK_U575_RL
        rows[name] = agg
        print(f"  {name:8s}  dec={agg['decay_km_d']:.2f}  band={agg['band_pct']:.1f}  "
              f"dl={agg['downlink_min_d']:.1f}  B={agg['deleg_B']:+.2f}  "
              f"J={agg['mtq_energy_J']:.2f}", flush=True)
    rows["meta"] = {"infer_u575": infer, "u575_mpc": TALK_U575_MPC,
                    "u575_rl": TALK_U575_RL, "wall_s": round(time.time() - t0, 1)}
    dest = (OUTPUTS / "v13" / "traces")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "talk_eval.json").write_text(json.dumps(rows, indent=2, default=str))
    print("[talk] storm1d traces", flush=True)
    slugs = {"SC_v12": "sc_v12", "SC_v13": "sc_v13",
             "SC_v14b": "sc_v14b", "SC_v14d": "sc_v14d",
             "SC_v14e": "sc_v14e", "SC_v14f": "sc_v14f"}
    for name, (ckpt, wrap) in specs.items():
        slug = slugs.get(name, name.lower())
        tr = dest / f"{slug}__storm1d.npz"
        if tr.exists() and name not in (v14_names or ()):
            continue
        record_trace(load_v12_predict(ckpt), wrap, "storm1d", slug)
    print(f"[talk] eval done in {time.time() - t0:.0f}s", flush=True)
    return rows


def fig_talk_radar(rows=None, v14_names=None):
    """Talk radar: MPC freeze + v12 + v13 + best two v14. Compute is U575."""
    if rows is None:
        p = OUTPUTS / "v13" / "traces" / "talk_eval.json"
        if not p.exists():
            rows = run_talk_eval(v14_names)
        else:
            rows = json.loads(p.read_text())
    if v14_names is None:
        v14_names = pick_best_v14(2)
    names = [n for n in ("SC_v12", "SC_v13", *v14_names) if n in rows]
    _style()
    ref = REF_D
    labels = ["decay\n(lower)", "band", "downlink", "B\nalign",
              "coil J\n(lower)", "compute\n(U575)"]

    def orient(name, agg):
        dec = (1.1 * ref["decay_nominal"]) / max(float(agg["decay_km_d"]), 0.05)
        band = float(agg["band_pct"]) / max(0.9 * ref["band_pct"], 1.0)
        dl = float(agg["downlink_min_d"]) / max(ref["downlink_min_d"], 0.1)
        b = (float(agg.get("deleg_B", 0.0)) + 0.5) / 0.5
        mtq = ref["mtq_energy_J"] / max(float(agg["mtq_energy_J"]), 0.01)
        # Flight number: 6.81 ms MPC vs 1.00 ms policy+comparator → 6.8×.
        # Cap the spoke at 2.0 so 6.8× fills the chart; annotate the true ×.
        spd = TALK_U575_MPC / TALK_U575_RL
        return _clip_radar([dec, band, dl, b, mtq, spd / 3.4])

    fig, ax = plt.subplots(figsize=(10.2, 8.8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.78)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    ax.plot(ang, [1.0] * (len(labels) + 1), color=TALK_COLS["MPC"], lw=2.0,
            ls="--", label="MPC freeze (1.0)", clip_on=True, zorder=2)
    ax.fill(ang, [1.0] * (len(labels) + 1), color=TALK_COLS["MPC"], alpha=0.04,
            clip_on=True)
    for name in names:
        vals = orient(name, rows[name]) + orient(name, rows[name])[:1]
        ax.plot(ang, vals, color=TALK_COLS.get(name, "#0f766e"), lw=2.3,
                label=name, clip_on=True)
        ax.fill(ang, vals, color=TALK_COLS.get(name, "#0f766e"), alpha=0.10,
                clip_on=True)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    _radar_axes(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.12, 0.55),
              fontsize=11)
    fig.suptitle("Deep RL vs sampling MPC — mission targets and onboard compute",
                 fontsize=16, fontweight="bold", y=0.97, x=0.42)
    fig.text(0.5, 0.035,
             f"1.0 ring = sensors-on MPC freeze (9.19 km/d, 84.1 % band, 18.6 min/d, "
             f"B = 0, 3.73 J, {TALK_U575_MPC:.2f} ms).  "
             f"Compute spoke is U575: {TALK_U575_MPC:.2f} ms / {TALK_U575_RL:.2f} ms "
             f"= {TALK_U575_MPC / TALK_U575_RL:.1f}× with the flight comparator "
             f"(spoke capped at 2). Policy-only 0.17 ms is 40× and is not flown.",
             ha="center", fontsize=9, color=INK2)
    _save(fig, "fig_talk_radar",
          "Quiet 10-orbit for decay/band/downlink/B/coil J. Compute is analytic U575, "
          "not the workstation forward. Do not mix with the 24-draw MC radar.")
    return names


def _draw_sat_still(ax, d, k, tris, tri_n, title):
    from arlamx_v2 import ipc_animation as IA
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sig = np.asarray(d["sigma"][k], float)
    q = IA._mrp_to_quat(sig)
    c_bn = IA._quat_dcm_bn(q)
    c_nb = c_bn.T
    world = tris @ c_nb.T
    lit = float(d["eclipse"][k]) > 0.5
    sun_b = c_bn @ d["sun_N"][k]
    base_rgb = np.array([0.95, 0.76, 0.31])
    dark_rgb = np.array([0.36, 0.39, 0.44])
    if lit:
        shade = 0.35 + 0.65 * np.abs(tri_n @ sun_b)
        colors = np.clip(shade[:, None] * base_rgb[None, :], 0, 1)
    else:
        colors = np.tile(dark_rgb, (len(tris), 1))
    coll = Poly3DCollection(list(world), linewidths=0.12,
                            edgecolors=(0, 0, 0, 0.25))
    coll.set_facecolor(np.c_[colors, np.full(len(colors), 0.95)])
    ax.add_collection3d(coll)
    ext = np.abs(tris).max() * 1.30
    s = ext * 1.15
    sun = d["sun_N"][k]
    v = d["v"][k] / max(np.linalg.norm(d["v"][k]), 1e-9)
    z_b = c_nb @ np.array([0.0, 0.0, 1.0])
    tau = d["tau_dist"][k]
    tau_nrm = c_nb @ (tau / max(np.linalg.norm(tau), 1e-12))
    ax.quiver(0, 0, 0, *(sun * s), color="#eda100", lw=1.8, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, *(v * s * 0.92), color="0.45", lw=1.4, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, *(z_b * s * 0.98), color="#2a78d6", lw=1.8, arrow_length_ratio=0.14)
    ax.quiver(0, 0, 0, *(tau_nrm * s * 1.06), color="#e34948", lw=2.2,
              arrow_length_ratio=0.16)
    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext); ax.set_zlim(-ext, ext)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=11, color=INK, pad=2)


def _z_sun(d):
    from arlamx_v2 import ipc_animation as IA
    out = []
    for k in range(len(d["t_h"])):
        q = IA._mrp_to_quat(np.asarray(d["sigma"][k], float))
        c_nb = IA._quat_dcm_bn(q).T
        z = c_nb @ np.array([0.0, 0.0, 1.0])
        sun = np.asarray(d["sun_N"][k], float)
        sn = np.linalg.norm(sun)
        out.append(float(np.dot(z, sun / max(sn, 1e-9))))
    return np.asarray(out)


def fig_talk_storm(v14_name=None):
    """Storm adaptability + environment as actuator, with simplified-model stills.

    Left/top: SolarCat simplified CAD at quiet / onset / in-storm, same arrows
    as the animation (Sun, velocity, +Z antenna, disturbance estimate).
    Bottom: time series; vertical lines mark the three 3D frames.
    """
    from arlamx_v2 import ipc_animation as IA
    if v14_name is None:
        best = pick_best_v14(2)
        v14_name = best[0] if best else "SC_v14d"
    slug = {"SC_v14b": "sc_v14b", "SC_v14d": "sc_v14d",
            "SC_v14e": "sc_v14e", "SC_v14f": "sc_v14f"}.get(v14_name, "sc_v14")
    primary = _load_npz(slug, "storm1d")
    if primary is None:
        primary = _load_npz("sc_v14", "storm1d")
        slug = "sc_v14"
    if primary is None:
        print("  [skip] fig_talk_storm: no storm1d trace", flush=True)
        return
    series = []
    slug_of = {"SC_v12": "sc_v12", "SC_v13": "sc_v13",
               "SC_v14b": "sc_v14b", "SC_v14d": "sc_v14d",
               "SC_v14e": "sc_v14e", "SC_v14f": "sc_v14f"}
    want = ["SC_v12", "SC_v13"] + [n for n in pick_best_v14(2) if n not in
                                   ("SC_v12", "SC_v13")]
    if v14_name not in want:
        want.append(v14_name)
    for lab in want:
        d = _load_npz(slug_of.get(lab, lab.lower()), "storm1d")
        if d is not None:
            series.append((lab, d))
    cache = OUTPUTS / "analysis" / "showcase" / "geometry_simplified.npz"
    if cache.exists():
        g = np.load(cache)
        tris, tri_n = g["tris"], g["tri_normals"]
    else:
        tris, tri_n = IA.simplified_model()
    t = np.asarray(primary["t_h"])
    ap = np.asarray(primary["ap"])
    storm = ap > 40.0
    segs = IA.find_segments(dict(primary) if not hasattr(primary, "files")
                            else {k: primary[k] for k in primary.files})
    if "storm_onset" in segs:
        i0, i1 = segs["storm_onset"]
    else:
        i0, i1 = 0, len(t)
    k_on = int(np.argmax(storm)) if np.any(storm) else (i0 + i1) // 2
    picks = [
        max(0, k_on - max(2, int(0.8 / max(float(t[1] - t[0]), 1e-3)))),
        k_on,
        min(len(t) - 1, k_on + max(2, int(1.6 / max(float(t[1] - t[0]), 1e-3)))),
    ]
    mark_cols = ("#0f766e", "#ea580c", "#e34948")
    mark_labs = ("quiet", "storm onset", "in storm")

    _style()
    fig = plt.figure(figsize=(16.8, 10.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1.55],
                          hspace=0.28, wspace=0.12,
                          left=0.02, right=0.99, top=0.86, bottom=0.07)
    for row, (k, lab, mc) in enumerate(zip(picks, mark_labs, mark_cols)):
        ax = fig.add_subplot(gs[row, 0], projection="3d")
        state = "storm" if float(ap[k]) > 40 else "quiet"
        _draw_sat_still(ax, primary, k, tris, tri_n,
                        f"{lab}   t = {t[k]:.1f} h   Ap {ap[k]:.0f}")

    axleg = fig.add_subplot(gs[3, 0])
    axleg.axis("off")
    for lab, c in (("Sun", "#eda100"), ("velocity", "0.45"),
                   ("+Z antenna", "#2a78d6"), ("disturbance est.", "#e34948")):
        axleg.plot([], [], color=c, lw=2.4, label=lab)
    axleg.legend(frameon=False, loc="center", fontsize=10)

    axes = [fig.add_subplot(gs[r, 1]) for r in range(4)]
    t1, t2 = float(t[0]), float(t[-1])
    for ax in axes:
        if np.any(storm):
            ax.fill_between(t, 0, 1, where=storm, color="#e34948", alpha=0.08,
                            transform=ax.get_xaxis_transform(), zorder=0)
        for k, mc in zip(picks, mark_cols):
            ax.axvline(t[k], color=mc, lw=1.6, ls="--", zorder=3)
        ax.set_xlim(t1, t2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.grid(axis="x", visible=False)

    axes[0].plot(t, ap, color="#e34948", lw=1.8)
    axes[0].set_ylabel("Ap  (storm)")
    axes[0].set_ylim(0, max(120, float(np.max(ap)) * 1.15))
    axes[0].set_xticklabels([])

    for lab, d in series:
        B = np.asarray(d["deleg_B"], float)
        tt = np.asarray(d["t_h"], float)
        col = TALK_COLS.get(lab, "#0f766e")
        axes[1].plot(tt, B, color=col, lw=0.7, alpha=0.28)
        if len(B) >= 5:
            ker = np.ones(5) / 5.0
            sm = np.convolve(B, ker, mode="same")
            axes[1].plot(tt, sm, color=col, lw=1.8, label=lab)
        else:
            axes[1].plot(tt, B, color=col, lw=1.8, label=lab)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_ylabel("B  (env as\nactuator)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    axes[1].set_xticklabels([])

    zsun = _z_sun(primary)
    axes[2].plot(t, zsun, color="#2a78d6", lw=1.7, label="+Z · Sun")
    if "gates" in (primary.files if hasattr(primary, "files") else primary):
        g = np.asarray(primary["gates"])
        axes[2].plot(t, g[:, 0], color="#2a78d6", lw=0.9, alpha=0.45, label="gate X")
        axes[2].plot(t, g[:, 1], color="#eb6834", lw=0.9, alpha=0.45, label="gate Y")
        axes[2].plot(t, g[:, 2], color="#1baf7a", lw=0.9, alpha=0.45, label="gate Z")
    axes[2].set_ylabel("pointing")
    axes[2].set_ylim(-1.05, 1.15)
    axes[2].legend(frameon=False, ncol=4, loc="lower right", fontsize=8)
    axes[2].set_xticklabels([])

    axes[3].axhspan(40, 60, color="#1baf7a", alpha=0.10)
    for lab, d in series:
        axes[3].plot(np.asarray(d["t_h"]), np.asarray(d["soc"]) * 100,
                     color=TALK_COLS.get(lab, "#0f766e"), lw=1.6, label=lab)
    axes[3].set_ylabel("SoC (%)")
    axes[3].set_ylim(0, 105)
    axes[3].set_xlabel("mission time (h)")
    axes[3].legend(frameon=False, ncol=3, loc="lower right", fontsize=9)

    fig.text(0.02, 0.975, "Storm day — environment as actuator",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.925,
             f"Left: {v14_name} simplified SolarCat (same CAD as the animation). "
             f"Gold = Sun, grey = velocity, blue = +Z antenna, red = disturbance estimate.  "
             f"Dashed lines mark the three frames. Shaded = Ap > 40.  "
             f"B > 0: τ_env turns the spacecraft the commanded way.",
             fontsize=10.5, color=INK2, ha="left", va="top")
    fig.text(0.02, 0.012,
             "storm1d protocol, 350 km, two F10.7/Ap jumps. No animation. "
             "SC_v12 B goes negative (wrong-way); v13/v14 stay ≥ 0. Thick B is a 5-step mean.",
             fontsize=8.5, color=MUTED, ha="left", va="bottom")
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"fig_talk_storm.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_talk_storm.png/.pdf", flush=True)


def run_talk():
    """The three presentation files: architecture + radar + storm."""
    import shutil
    configure(v14=True)
    v14_names = pick_best_v14(2)
    print(f"[talk] best two v14 for radar: {v14_names}", flush=True)
    write_slide_architecture_png()
    rows = run_talk_eval(v14_names)
    fig_talk_radar(rows, v14_names)
    fig_talk_storm(v14_names[0] if v14_names else None)
    mapping = (
        ("fig0_architecture", "key_architecture"),
        ("fig_talk_radar", "key_radar"),
        ("fig_talk_storm", "key_storm"),
    )
    for src, dst in mapping:
        for ext in ("png", "pdf"):
            p = PLOTS / f"{src}.{ext}"
            if p.exists():
                shutil.copy2(p, PLOTS / f"{dst}.{ext}")
    (PLOTS / "KEY_PRESENTATION.md").write_text(
        "# Talk figures — only these three\n\n"
        "Folder: `outputs/v14/plots/`\n\n"
        "1. `key_architecture`  onboard ADCS (sensors → filter → REGIME → Deep RL → clips → act)\n"
        "2. `key_radar`  circular: MPC freeze vs SC_v12, SC_v13, and the two best v14 zips. "
        "Compute spoke is U575 6.81 ms vs 1.00 ms (6.8×).\n"
        "3. `key_storm`  storm-day stills of the simplified SolarCat model + B / pointing / SoC, "
        "with vertical lines at the three frames. No animation.\n"
    )
    print("[talk] wrote key_architecture, key_radar, key_storm", flush=True)


def _copy_us(stems):
    import shutil
    for stem in stems:
        for ext in ("png", "pdf"):
            src = PLOTS / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, PLOTS / f"{stem}us.{ext}")


KEY_MAP = (
    ("fig0_architecture", "key_architecture"),
    ("fig1_scorecard", "key_scorecard"),
    ("fig2_radar", "key_radar"),
    ("fig3_env_actuation", "key_env_actuation"),
    ("fig5_adcs_loop", "key_adcs"),
    ("fig6_inference", "key_inference"),
    ("fig_B_storm_compare", "key_B_storm"),
    ("fig9_mc_v14", "key_mc_tasks"),
    ("fig9_mc_table", "key_mc_table"),
    ("fig_soc_ternary", "key_soc"),
    ("fig_radar_mc", "key_radar_mc"),
)


def _copy_key():
    """Presentation set: filenames start with key_ so they are easy to pick."""
    import shutil
    copied = []
    for src_stem, dst_stem in KEY_MAP:
        for ext in ("png", "pdf"):
            src = PLOTS / f"{src_stem}.{ext}"
            if src.exists():
                shutil.copy2(src, PLOTS / f"{dst_stem}.{ext}")
                copied.append(f"{dst_stem}.{ext}")
    print(f"  key_ copies: {len(copied)} files", flush=True)
    return copied


def fig_soc_ternary(soc=None):
    """Stacked SoC: <40 (supercap-critical) / 40–60 / >60.

    >60 is the less important side of 'out of band' for a supercapacitor.
    """
    path = OUTPUTS / "v13" / "SOC_TERNARY.json"
    if soc is None:
        if not path.exists():
            print("  [skip] fig_soc_ternary: no SOC_TERNARY.json", flush=True)
            return
        soc = json.loads(path.read_text())
    _style()
    protocols = [p for p in ("quiet", "storm1d", "mc") if p in soc]
    if not protocols:
        print("  [skip] fig_soc_ternary: empty", flush=True)
        return
    fig, axes = plt.subplots(1, len(protocols), figsize=(5.2 * len(protocols), 5.6),
                             sharey=True)
    if len(protocols) == 1:
        axes = [axes]
    titles = {"quiet": "Quiet 10-orbit", "storm1d": "Storm 1-day",
              "mc": "MC 24 draws (mean)"}
    lo_c, bd_c, hi_c = "#e34948", "#0f766e", "#2a78d6"
    for ax, prot in zip(axes, protocols):
        block = soc[prot]
        names = list(block.keys())
        x = np.arange(len(names))
        lo = np.array([float(block[n]["soc_lo_pct"]) for n in names])
        bd = np.array([float(block[n]["band_pct"]) for n in names])
        hi = np.array([float(block[n]["soc_hi_pct"]) for n in names])
        ax.bar(x, lo, color=lo_c, width=0.62, label="< 40 %  (under)", zorder=3)
        ax.bar(x, bd, bottom=lo, color=bd_c, width=0.62, label="40–60 %  (band)", zorder=3)
        ax.bar(x, hi, bottom=lo + bd, color=hi_c, width=0.62, label="> 60 %  (over)", zorder=3)
        for xi, a, b, c in zip(x, lo, bd, hi):
            if a > 4:
                ax.text(xi, a / 2, f"{a:.0f}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            ax.text(xi, a + b / 2, f"{b:.0f}", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
            if c > 4:
                ax.text(xi, a + b + c / 2, f"{c:.0f}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylim(0, 108)
        ax.set_title(titles.get(prot, prot), fontsize=13, color=INK)
        ax.grid(axis="x", visible=False)
        if ax is axes[0]:
            ax.set_ylabel("% of time")
            ax.legend(frameon=False, loc="upper right", fontsize=9)
    _header(fig, "Battery time split — under 40 % is the supercap-critical side",
            "Band = 40–60 %. Time above 60 % is also 'out of band' but is not a brownout risk. "
            "MC is the 24-draw mean, not the 10-orbit quiet probe.",
            top=0.80)
    _save(fig, "fig_soc_ternary",
          "Under-40 is what empties a supercap. Quote under and over separately; do not lump them as 'out of band'.")


def fig_radar_mc():
    """Circular plot of the 24-draw MC objectives vs MPC mean."""
    path = OUTPUTS / "v13" / "traces" / "mc_four.json"
    if not path.exists():
        print("  [skip] fig_radar_mc: no mc_four.json", flush=True)
        return
    payload = json.loads(path.read_text())
    s = payload.get("summary") or {}
    if "MPC" not in s:
        print("  [skip] fig_radar_mc: no MPC in MC summary", flush=True)
        return
    _style()
    labels = ["days", "band", "downlink", "B\nalign", "coil J\n(lower)",
              "no brownout"]
    mpc = s["MPC"]

    def mean(block, key, default=0.0):
        v = (block.get(key) or {})
        if isinstance(v, dict):
            return float(v.get("mean", default) or default)
        return float(v or default)

    def orient(name):
        b = s[name]
        n = max(int(b.get("n") or 1), 1)
        days = mean(b, "days_500_to_300") / max(mean(mpc, "days_500_to_300"), 0.05)
        band = mean(b, "band_pct") / max(mean(mpc, "band_pct"), 1.0)
        dl = mean(b, "downlink_min_d") / max(mean(mpc, "downlink_min_d"), 0.1)
        bb = (mean(b, "deleg_B") + 0.5) / 0.5
        j = mean(mpc, "mtq_J_per_day") / max(mean(b, "mtq_J_per_day"), 0.01)
        nb = 1.0 - float(b.get("brownouts_any") or 0) / n
        return _clip_radar([days, band, dl, bb, j, nb])

    names = [n for n in ("SC_v13", "SC_v14b", "SC_v14d", "SC_v14e", "SC_v14")
             if n in s]
    cols = {"SC_v13": "#b45309", "SC_v14": "#6d28d9", "SC_v14b": "#6d28d9",
            "SC_v14d": "#155e75", "SC_v14e": "#be185d"}
    fig, ax = plt.subplots(figsize=(8.8, 8.4), subplot_kw=dict(polar=True))
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    ax.plot(ang, [1.0] * (len(labels) + 1), color=C["MPC"], lw=1.6, ls="--",
            label="MPC MC mean (1.0)", clip_on=True, zorder=2)
    for name in names:
        vals = orient(name) + orient(name)[:1]
        ax.plot(ang, vals, color=cols.get(name, "#0f766e"), lw=2.2, label=name,
                clip_on=True)
        ax.fill(ang, vals, color=cols.get(name, "#0f766e"), alpha=0.10,
                clip_on=True)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    _radar_axes(ax)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.30, 1.12))
    fig.suptitle("Monte-Carlo match to sampling MPC (24 draws)", fontsize=16,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.03,
             "Same 24 weather draws. 1.0 = MPC mean. No-brownout spoke is 1 − (runs with a brownout)/24. "
             "B spoke: (B + 0.5)/0.5 so MPC at B = 0 sits on the freeze ring.",
             ha="center", fontsize=9, color=INK2)
    _save(fig, "fig_radar_mc",
          "MC radar, not the 10-orbit quiet radar. Do not mix the two rings.")


def run_plots():
    rows = _load_eval()
    write_slide_architecture_png()
    fig1_scorecard(rows)
    fig2_radar(rows)
    fig3_power(rows)
    fig4_timeline()
    fig5_adcs_loop()
    fig6_infer(rows)
    fig7_targets(rows)
    fig_B_compare()
    fig_ipc_stills()
    fig_soc_ternary()
    fig_radar_mc()
    us = ("fig0_architecture", "fig1_scorecard", "fig2_radar",
          "fig3_env_actuation", "fig5_adcs_loop", "fig6_inference",
          "fig_B_storm_compare", "fig_ipc_stills", "fig_soc_ternary",
          "fig_radar_mc")
    _copy_us(us)
    _copy_key()
    (PLOTS / "README.md").write_text(
        f"# {LABEL} plots — presentation set starts with `key_`\n\n"
        "No animations in this set. Use PNG/PDF.\n\n"
        "## Presentation (`key_`)\n"
        "- `key_radar`  circular quiet 10-orbit vs MPC freeze (decay, band, downlink, B, coil J, compute)\n"
        "- `key_radar_mc`  circular 24-draw MC vs MPC mean (days, band, downlink, B, J, no-brownout)\n"
        "- `key_scorecard`  decay, band, downlink, days, coil J, U575\n"
        "- `key_architecture`  onboard flowchart with REGIME\n"
        "- `key_env_actuation`  B / P / idle / coil J\n"
        "- `key_B_storm`  B(t) on the same storm day\n"
        "- `key_adcs`  storm ADCS strip\n"
        "- `key_inference`  6.81 vs 0.17 vs 1.00 vs 15 ms\n"
        "- `key_mc_tasks`  24-draw MC four tasks\n"
        "- `key_mc_table`  MC numbers table\n"
        "- `key_soc`  SoC <40 / 40–60 / >60 (under is supercap-critical)\n\n"
        "`us` copies still exist for older slides. Animation stays in `../anim/` "
        "and is not part of the presentation set.\n"
    )


def run_anim():
    from arlamx_v2 import ipc_animation as IA
    src = TRACES / f"{SLUG}__storm1d.npz"
    if not src.exists():
        src = TRACES / f"{SLUG}__nominal2d.npz"
    if not src.exists():
        print("  [skip] anim: no trace")
        return
    ANIM.mkdir(parents=True, exist_ok=True)
    dest = ANIM / src.name
    d = dict(np.load(src))
    if "mtq_power_mW" not in d:
        d["mtq_power_mW"] = np.asarray(d["mtq_quad_W"]) * 1e3
    if "effort" not in d:
        d["effort"] = np.asarray(d["mtq_W"]) / 0.270
    np.savez_compressed(dest, **d)
    IA.SHOW = ANIM
    cache = OUTPUTS / "analysis" / "showcase" / "geometry_simplified.npz"
    if cache.exists():
        g = np.load(cache)
        tris, tri_n = g["tris"], g["tri_normals"]
    else:
        tris, tri_n = IA.simplified_model()
    segs = IA.find_segments(d)
    want = ["storm_onset", "downlink_pass", "quiet_cruise"]
    if "storm_onset" not in segs:
        want = ["full"]
    for name in want:
        if name not in segs:
            continue
        i0, i1 = segs[name]
        if i1 - i0 < 4:
            continue
        IA.render(d, tris, tri_n, SLUG, src.stem.split("__")[1], name, i0, i1)
    print("[anim] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--anim", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--v13", action="store_true",
                    help="SC_v13 freeze, writes outputs/v13/")
    ap.add_argument("--v14", action="store_true",
                    help="SC_v14 four-way (v12/v13/v14 vs MPC), writes outputs/v14/plots")
    ap.add_argument("--fourway", action="store_true",
                    help="Eval MPC + v12 + v13 + v14 on the quiet 10-orbit")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--talk", action="store_true",
                    help="Only the 3 presentation files: architecture, radar, storm")
    a = ap.parse_args()
    configure(v13=a.v13, v14=a.v14 or a.talk, ckpt=a.ckpt, fourway=a.fourway or a.v14)
    if a.talk:
        run_talk()
        return
    if not (a.eval or a.plot or a.anim or a.all):
        a.all = True
    if a.all or a.eval:
        run_eval()
    if a.all or a.plot:
        run_plots()
    if a.all or a.anim:
        run_anim()


if __name__ == "__main__":
    main()
