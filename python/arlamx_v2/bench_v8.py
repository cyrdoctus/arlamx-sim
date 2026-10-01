"""SC_v8 / v8Duo / v9 campaign: burn-in gate, size sweep, Duo, v9.

Stages (run in this order; each refuses to start without its inputs):

  burnin   one instrumented v8b episode + KF acceptance. PRINTS the grok gate:
           mean deleg_P, deleg_B, tau_want vs tau_aero, brownouts, gate_floor,
           mass. If deleg_P is ~0 the stage FAILS and nothing else may run.
  ref      MPC + heuristic scored on the V8 PLANT (v8a env, gates pinned).
           v7 bake-off numbers are not comparable — different coils, cp offset,
           power law, brownout wiring — so the baselines are re-run here.
  sweep    sizes x algos x budgets x arms -> outputs/v8/runs/, ledger.csv
  duo      pick slots by rule (duo.pick_duo_models), train the horizon
           advisor on the wrapper env, evaluate the assembled Duo.
  v9       top-2 sizes per algo from the v8b ledger -> v9a / v9b.
  trace    per-step traces of representatives for the figures.

Artifacts: outputs/v8/{runs,data,ledger.csv}. Every training run freezes its
resolved YAMLs to snapshot.json (train.train_one).

    PYTHONPATH=python python -m arlamx_v2.bench_v8 --stage burnin
Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arlamx_v2 import cpp
from arlamx_v2 import duo as duolib
from arlamx_v2 import propagator as fastprop
from arlamx_v2.advisors.stations import nearest_visible
from arlamx_v2.campaign import EPOCH_JD, PROBES
from arlamx_v2.config import load as load_cfg
from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts
from arlamx_v2.power import mtq_power_w
from arlamx_v2.train import train_one

OUT = OUTPUTS / "v8"
RUNS = OUT / "runs"
DATA = OUT / "data"
LEDGER = OUT / "ledger.csv"

ARCHS = ("4x14", "4x16", "4x18", "4x20", "4x24", "4x32")
ALGOS = ("ppo", "sac", "td3")
BUDGETS = (50_000, 300_000)
ARMS = ("v8a", "v8b")
SEED = 42
N_ENVS = 32

LEDGER_FIELDS = [
    "name", "variant", "algo", "arch", "timesteps", "seed", "wall_s",
    "decay_nominal", "decay_spike", "band_pct", "brownouts", "downlink_min_d",
    "act_energy_J", "mtq_energy_J", "deleg_benefit", "deleg_P", "deleg_S",
    "kf_r", "raw_r", "gap_index",
]


def arch_list(tag):
    layers, width = (int(x) for x in tag.lower().split("x"))
    return [width] * layers


# ---------------------------------------------------------------------------
# Probe rollouts (v8-aware; campaign.py stays v7-only)
# ---------------------------------------------------------------------------

SLEW_DEG = 2.0          # a command change above this counts as a slew
SOC_SIGMA = 0.005       # [ASSUME] supercap voltage-divider, 1-sigma SoC


def _mk_env(variant, probe, n_orbits=20.0):
    env = ArlamxV2Env(seed=0, variant=variant)
    alt = float(PROBES[probe]["opt"]["altitude_km"])
    period = 2 * np.pi * np.sqrt((6371e3 + alt * 1e3) ** 3 / 3.986004418e14)
    env._max_steps = int(float(n_orbits) * period / env._advisor_s)
    return env


def _rollout(env, predict_fn, probe, seed, opt=None, jumps=None):
    if opt is None:
        spec = PROBES[probe]
        opt, jumps = dict(spec["opt"]), spec["jumps"]
    if jumps is None:
        jumps = []
    obs, _ = env.reset(seed=seed, options=dict(opt))
    env._weather_jumps = [(f * env._max_steps, df, da) for f, df, da in jumps]
    env._srp_flashes = []
    H = {k: [] for k in ("sma", "soc", "dep", "dl", "eff", "P", "B", "S",
                         "mtqW", "p_pd", "slew", "kfpair")}
    done = False
    while not done:
        a = predict_fn(obs, env)
        obs, _r, term, trunc, info = env.step(np.asarray(a, np.float32))
        done = term or trunc
        H["sma"].append(float(info["sma_m"]))
        H["soc"].append(float(info["battery_soc"]))
        H["dep"].append(1.0 if info.get("battery_depleted") else 0.0)
        vis = float(info.get("gs_visible", 0)) > 0.5
        lit = float(info.get("eclipse", 1)) > 0.5
        H["dl"].append(1.0 if (vis and lit and float(info.get("gs_point_cos", 0)) > 0.7) else 0.0)
        H["eff"].append(float(info.get("torque_effort", 0.0)))
        H["P"].append(float(info.get("deleg_P", 0.0)))
        H["B"].append(float(info.get("deleg_B", 0.0)))
        H["S"].append(float(info.get("deleg_S", 0.0)))
        H["mtqW"].append(float(info.get("mtq_power_W", 0.0)))
        H["slew"].append(1.0 if float(info.get("cmd_angle_deg", 0.0)) > SLEW_DEG else 0.0)
        tau_w = np.asarray(info.get("tau_want", np.zeros(3)), float)
        bb = np.asarray(info.get("B_B", getattr(env, "_B_B", np.zeros(3))), float)
        pcfg = getattr(env, "_pcfg", None)
        if pcfg is not None and float(np.linalg.norm(bb)) > 1e-12:
            p_pd, _ = mtq_power_w(tau_w, bb, pcfg)
        else:
            p_pd = float(info.get("mtq_power_W", 0.0))
        H["p_pd"].append(float(p_pd))
        H["kfpair"].append((np.asarray(info.get("tau_aero_mean", np.zeros(3)), float),
                            np.asarray(info.get("tau_env_filt", np.zeros(3)), float),
                            np.asarray(info.get("tau_env_raw", np.zeros(3)), float)))
    n = max(1, len(H["soc"]))
    days = n * env._advisor_s / 86400.0
    sma = np.asarray(H["sma"])
    dep = np.asarray(H["dep"]) > 0.5
    soc = np.asarray(H["soc"])
    ta = np.array([np.linalg.norm(a) for a, _, _ in H["kfpair"]])
    tf = np.array([np.linalg.norm(f) for _, f, _ in H["kfpair"]])
    tr_ = np.array([np.linalg.norm(rw) for _, _, rw in H["kfpair"]])
    keep = np.arange(n) >= max(2, int(3600.0 / env._advisor_s))   # drop detumble hour
    def _r(x, y):
        if keep.sum() < 4 or np.std(x[keep]) < 1e-15 or np.std(y[keep]) < 1e-15:
            return 0.0
        return float(np.corrcoef(x[keep], y[keep])[0, 1])
    e_act = float(np.sum(H["mtqW"]) * env._advisor_s)
    e_pd = float(np.sum(H["p_pd"]) * env._advisor_s)
    return {
        "decay_km_d": float((sma[0] - sma[-1]) / 1e3 / max(days, 1e-9)),
        "band_pct": float(np.mean((soc >= 0.4) & (soc <= 0.6)) * 100.0),
        "brownouts": int(np.sum(np.diff(dep.astype(int)) > 0) + (1 if dep[:1].any() else 0)),
        "downlink_min_d": float(np.sum(H["dl"]) * env._advisor_s / 60.0 / max(days, 1e-9)),
        "act_energy_J": float(np.sum(H["eff"]) * 0.270 * env._advisor_s),
        "mtq_energy_J": e_act,
        "mtq_energy_J_d": float(e_act / max(days, 1e-9)),
        "slews_per_day": float(np.sum(H["slew"]) / max(days, 1e-9)),
        "pei": float(e_act / max(e_pd, 1e-18)),
        "deleg_benefit": float(np.mean(H["B"])),
        "deleg_P": float(np.mean(H["P"])),
        "deleg_S": float(np.mean(H["S"])),
        "kf_r": _r(ta, tf),
        "raw_r": _r(ta, tr_),
        "days": float(days),
        "steps": n,
    }


def probe_suite(predict_fn, variant):
    per = {}
    for probe in PROBES:
        env = _mk_env(variant, probe)
        rows = [_rollout(env, predict_fn, probe, s) for s in (1001, 1002)]
        env.close()
        per[probe] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    agg = {
        "decay_nominal": per["nominal"]["decay_km_d"],
        "decay_spike": per["spike"]["decay_km_d"],
        "band_pct": float(np.mean([per[p]["band_pct"] for p in per])),
        "brownouts": float(np.sum([per[p]["brownouts"] for p in per])),
        "downlink_min_d": float(np.mean([per[p]["downlink_min_d"] for p in per])),
        "act_energy_J": float(np.sum([per[p]["act_energy_J"] for p in per])),
        "mtq_energy_J": float(np.sum([per[p]["mtq_energy_J"] for p in per])),
        "deleg_benefit": float(np.mean([per[p]["deleg_benefit"] for p in per])),
        "deleg_P": float(np.mean([per[p]["deleg_P"] for p in per])),
        "deleg_S": float(np.mean([per[p]["deleg_S"] for p in per])),
        "kf_r": float(np.mean([per[p]["kf_r"] for p in per])),
        "raw_r": float(np.mean([per[p]["raw_r"] for p in per])),
        "slews_per_day": float(np.mean([per[p]["slews_per_day"] for p in per])),
        "mtq_energy_J_d": float(np.mean([per[p]["mtq_energy_J_d"] for p in per])),
        "pei": float(np.mean([per[p]["pei"] for p in per])),
    }
    return agg, per


GAP_METRICS = [("decay_nominal", True, 1.0), ("band_pct", False, 10.0),
               ("brownouts", True, 1.0), ("downlink_min_d", False, 10.0),
               ("decay_spike", True, 1.0)]


def gap_index(agg, ref):
    gaps = []
    for key, lower, floor in GAP_METRICS:
        pol, mpc = float(agg[key]), float(ref[key])
        den = max(abs(mpc), floor)
        gaps.append((pol - mpc) / den if lower else (mpc - pol) / den)
    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# Advisor baselines on the v8 plant
# ---------------------------------------------------------------------------

def _pad7(q):
    return np.concatenate([np.asarray(q, float).reshape(4), [-1.0, -1.0, -1.0]]).astype(np.float32)


def _advisor_state(env, noisy=False, last_rv=None):
    from arlamx_v2.atmosphere import jd_to_datetime, query_msis
    st = env.sim.get_state()
    r = np.asarray(st["r"], float)
    v = np.asarray(st["v"], float)
    jd = EPOCH_JD + float(st["t"]) / 86400.0
    gmst = cpp.gmst_rad(jd)
    lat, lon, alt_m = cpp.ecef_to_geodetic(r)
    lon_e = (lon - gmst + np.pi) % (2 * np.pi) - np.pi
    try:
        rho, T, mb = query_msis(alt_m / 1e3, np.degrees(lat), np.degrees(lon_e),
                                jd_to_datetime(jd), env._f107, env._ap)
    except Exception:
        rho, T, mb = None, 900.0, 2.656e-26
    gs_n, gs_vis, _el = nearest_visible(r, gmst)
    c_bn = np.asarray(st["C_BN"], float)
    truth = {
        "r": r, "v": v,
        "sun_N": np.asarray(st.get("sun_N", np.zeros(3)), float),
        "C_BN": c_bn, "battery_soc": env._batt_E / env._batt_cap_J,
        "rho": rho, "T": T, "m_bar": mb,
        "gs_dir_N": gs_n if gs_vis > 0.5 else None,
        "gs_dir_B": (c_bn @ gs_n) if gs_vis > 0.5 else None,
        "gs_visible": gs_vis,
        "power_gen_norm": 0.0,
        "B_B": np.asarray(getattr(env, "_B_B", np.zeros(3)), float),
        "t": float(st["t"]),
    }
    if not noisy or getattr(env, "sensors", None) is None:
        return truth
    return _apply_sensors(env, truth, last_rv)


def _apply_sensors(env, st, last_rv=None):
    """Feed the advisor the SensorSuite, not plant truth.

    GNSS (Orion B16-01) on r,v; dual MMC5983MA on B, averaged unless they
    disagree by > 5 uT; SoC 0.5 % [ASSUME] white. Sun stays ephemeris
    (the MPC never had a sun sensor). MSIS and GS visibility are recomputed
    from the noisy position so density and downlink follow the same fix.
    """
    from arlamx_v2.atmosphere import jd_to_datetime, query_msis
    t = float(st["t"])
    gnss = env.sensors.read_gnss(st["r"], st["v"], t, powered=True)
    if gnss.get("valid"):
        r, v = np.asarray(gnss["r"], float), np.asarray(gnss["v"], float)
    elif last_rv is not None:
        r, v = last_rv
    else:
        r, v = st["r"], st["v"]          # TTFF: no fix yet, hold truth this step
    bb = np.asarray(st.get("B_B", np.zeros(3)), float)
    ba, bb2 = env.sensors.read_mag(bb, env._advisor_s)
    if np.all(np.isfinite(ba)) and np.all(np.isfinite(bb2)):
        if float(np.linalg.norm(ba - bb2)) < 5e-6:
            b_meas = 0.5 * (ba + bb2)
        else:
            def _dev(x):
                return abs(float(np.linalg.norm(x)) - 3.0e-5)
            b_meas = ba if _dev(ba) <= _dev(bb2) else bb2
    else:
        b_meas = ba if np.all(np.isfinite(ba)) else bb
    rng = np.random.default_rng(int(env.sensors._seed) ^ (int(t) & 0xFFFFFFFF))
    soc = float(np.clip(st["battery_soc"] + rng.normal(0.0, SOC_SIGMA), 0.0, 1.0))
    jd = EPOCH_JD + t / 86400.0
    gmst = cpp.gmst_rad(jd)
    gs_n, gs_vis, _el = nearest_visible(r, gmst)
    lat, lon, alt_m = cpp.ecef_to_geodetic(r)
    lon_e = (lon - gmst + np.pi) % (2 * np.pi) - np.pi
    try:
        rho, T, mb = query_msis(alt_m / 1e3, np.degrees(lat), np.degrees(lon_e),
                                jd_to_datetime(jd), env._f107, env._ap)
    except Exception:
        rho, T, mb = st.get("rho"), st.get("T", 900.0), st.get("m_bar", 2.656e-26)
    c_bn = st["C_BN"]
    out = dict(st)
    out.update(
        r=r, v=v, battery_soc=soc, rho=rho, T=T, m_bar=mb,
        gs_dir_N=gs_n if gs_vis > 0.5 else None,
        gs_dir_B=(c_bn @ gs_n) if gs_vis > 0.5 else None,
        gs_visible=gs_vis, B_B=b_meas,
    )
    return out


def mpc_fn(cfg=None, noisy=False, seed=0):
    from arlamx_v2.advisors.mpc import SamplingMpcPolicy
    pol = (SamplingMpcPolicy.from_config(cfg, seed=seed)
           if cfg is not None else SamplingMpcPolicy(seed=seed))
    holder = {"env": None, "rv": None}

    def f(obs, env):
        if holder["env"] is not env:
            pol.reset()
            pol.panels = (env._n, env._A, env._c)
            pol.advisor_step_s = env._advisor_s
            holder["env"] = env
            holder["rv"] = None
        st = _advisor_state(env, noisy=noisy, last_rv=holder["rv"])
        if noisy:
            holder["rv"] = (np.asarray(st["r"], float).copy(),
                            np.asarray(st["v"], float).copy())
        q, _ = pol.predict(obs, st)
        return _pad7(q)
    return f


def heuristic_fn():
    from arlamx_v2.advisors.heuristic import HeuristicPowerPolicy
    pol = HeuristicPowerPolicy()
    holder = {"env": None}

    def f(obs, env):
        if holder["env"] is not env:
            pol.reset()
            holder["env"] = env
        st = _advisor_state(env)
        st["power_gen_norm"] = float(obs[21])
        st["battery_soc"] = float(obs[20])
        q, _ = pol.predict(obs, st)
        return _pad7(q)
    return f


def sb3_fn(model):
    def f(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return a
    return f


def load_model(algo, name):
    from stable_baselines3 import PPO, SAC, TD3
    cls = {"sac": SAC, "td3": TD3}.get(algo, PPO)
    return cls.load(str(RUNS / name / "models" / f"{algo}_{name}.zip"))


# ---------------------------------------------------------------------------
# Stage: burn-in gate
# ---------------------------------------------------------------------------

def stage_burnin():
    """The grok gate. Prints, judges, and refuses to pass silently."""
    print("[burnin] one v8b episode, slew-heavy driver", flush=True)
    env = _mk_env("v8b", "nominal")
    obs, _ = env.reset(seed=7, options=dict(PROBES["nominal"]["opt"]))
    rng = np.random.default_rng(7)
    Ps, Bs, want, aero, dep = [], [], [], [], 0
    kf_pairs = []
    for k in range(120):
        if k % 4 == 0:      # a fresh slew every 20 min keeps the PD demand alive
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            ang = np.radians(rng.uniform(15, 40))
            q = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])
        # Full delegation: the burn-in measures the MECHANISM's ceiling — how
        # often the gate can bind at all — not a random policy's average.
        s = np.ones(3)
        obs, r, te, tr, info = env.step(np.concatenate([q, s]).astype(np.float32))
        Ps.append(info["deleg_P"]); Bs.append(info["deleg_B"])
        want.append(np.linalg.norm(info["tau_want"]))
        aero.append(np.linalg.norm(info["tau_aero_mean"]))
        kf_pairs.append((np.linalg.norm(info["tau_aero_mean"]),
                         np.linalg.norm(info["tau_env_filt"]),
                         np.linalg.norm(info["tau_env_raw"])))
        dep += int(bool(info.get("battery_depleted")))
        if te or tr:
            break
    ta, tf, trw = (np.array(x) for x in zip(*kf_pairs))
    keep = np.arange(len(ta)) >= 12
    r_f = float(np.corrcoef(ta[keep], tf[keep])[0, 1])
    r_raw = float(np.corrcoef(ta[keep], trw[keep])[0, 1])
    report = {
        "mean_deleg_P": float(np.mean(Ps)),
        "frac_steps_P_positive": float(np.mean(np.asarray(Ps) > 1e-6)),
        "mean_deleg_B": float(np.mean(Bs)),
        "mean_abs_B": float(np.mean(np.abs(Bs))),
        "tau_want_mean_uNm": float(np.mean(want)) * 1e6,
        "tau_aero_mean_uNm": float(np.mean(aero)) * 1e6,
        "want_over_aero": float(np.mean(want) / max(np.mean(aero), 1e-30)),
        "kf_r_filtered": r_f, "kf_r_raw": r_raw,
        "kf_improvement": r_f - r_raw,
        "depleted_steps": dep,
        "gate_floor": float(env._gate_floor),
        "mass_kg": float(env._mass),
        "tau_max_uNm": (env._tau_max * 1e6).tolist(),
    }
    env.close()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "burnin.json").write_text(json.dumps(report, indent=2))
    for k, v in report.items():
        print(f"  {k:26s} {v}", flush=True)
    ok = (report["mean_deleg_P"] > 0.01
          and report["frac_steps_P_positive"] >= 0.04
          and report["mean_abs_B"] > 0.05
          and abs(report["gate_floor"] - 0.3) < 1e-9
          and abs(report["mass_kg"] - 0.625) < 1e-9)
    print(f"[burnin] GATE {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("burn-in gate failed — do not train (grok fix order)")
    acc = load_cfg("estimator_kf")["acceptance"]
    kf_ok = report["kf_r_filtered"] >= report["kf_r_raw"] - float(acc["max_r_degradation"])
    print(f"[burnin] KF acceptance (no degradation beyond "
          f"{acc['max_r_degradation']}): {'PASS' if kf_ok else 'FAIL'}", flush=True)
    if not kf_ok:
        raise SystemExit("KF degrades the raw observer — retune before training")
    return report


# ---------------------------------------------------------------------------
# Stage: baselines on the v8 plant
# ---------------------------------------------------------------------------

def stage_ref(force=False):
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "ref.json"
    if out.exists() and not force:
        print("[ref] exists, skipping", flush=True)
        return json.loads(out.read_text())
    res = {}
    for label, fn in (("MPC", mpc_fn()), ("Heuristic", heuristic_fn())):
        print(f"[ref] {label} on the v8 plant (v8a env, gates pinned)", flush=True)
        agg, per = probe_suite(fn, "v8a")
        res[label] = {"agg": agg, "per_probe": per}
        print(f"   decay={agg['decay_nominal']:.2f} band={agg['band_pct']:.1f}% "
              f"brn={agg['brownouts']:.0f} dl={agg['downlink_min_d']:.1f} "
              f"mtqJ={agg['mtq_energy_J']:.2f}", flush=True)
    out.write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# Stage: size sweep
# ---------------------------------------------------------------------------

def _ledger_rows():
    if not LEDGER.exists():
        return []
    with open(LEDGER) as f:
        return list(csv.DictReader(f))


def _ledger_append(row):
    OUT.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def run_one(variant, algo, arch_tag, steps, ref_agg, seed=SEED):
    name = f"{variant}_{arch_tag}_{algo}_{steps // 1000}k_s{seed}"
    done = {r["name"] for r in _ledger_rows()}
    if name in done:
        print(f"[skip] {name}", flush=True)
        return
    t0 = time.time()
    train_one(variant, algo, steps, N_ENVS, seed, name, RUNS,
              arch=arch_list(arch_tag))
    model = load_model(algo, name)
    agg, per = probe_suite(sb3_fn(model), variant)
    (RUNS / name / "probe_metrics.json").write_text(
        json.dumps({"agg": agg, "per_probe": per}, indent=2))
    row = {"name": name, "variant": variant, "algo": algo, "arch": arch_tag,
           "timesteps": steps, "seed": seed, "wall_s": round(time.time() - t0, 1),
           **{k: agg[k] for k in ("decay_nominal", "decay_spike", "band_pct",
                                  "brownouts", "downlink_min_d", "act_energy_J",
                                  "mtq_energy_J", "deleg_benefit", "deleg_P",
                                  "deleg_S", "kf_r", "raw_r")},
           "gap_index": gap_index(agg, ref_agg)}
    _ledger_append(row)
    print(f"[done] {name} gap={row['gap_index']:.3f} "
          f"decay={agg['decay_nominal']:.2f} B={agg['deleg_benefit']:.3f} "
          f"P={agg['deleg_P']:.4f}", flush=True)


def stage_sweep(budgets=BUDGETS, archs=ARCHS, algos=ALGOS, arms=ARMS):
    ref = stage_ref()["MPC"]["agg"]
    for steps in budgets:
        for arm in arms:
            for algo in algos:
                for arch_tag in archs:
                    run_one(arm, algo, arch_tag, steps, ref)


# ---------------------------------------------------------------------------
# Stage: v8Duo
# ---------------------------------------------------------------------------

class HorizonEnv(gym.Env):
    """Training env for the Duo horizon advisor.

    Wraps a v8b base env plus a FROZEN main advisor. The agent's action is
    q_horizon; the applied command comes from the deterministic arbiter; the
    reward is duo_reward — the 6 h counterfactual objective.
    """

    def __init__(self, main_model, seed=None, duo_cfg=None, base_variant="v8b"):
        super().__init__()
        self.cfg = duo_cfg or load_cfg("duo")
        self.base = ArlamxV2Env(seed=seed, variant=base_variant)
        self.main = main_model
        h = self.cfg["horizon"]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(int(h["obs_dim"]),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._obs_main = None
        self._q_main = np.array([1.0, 0.0, 0.0, 0.0])
        self._split_main = np.zeros(3)
        self._hobs = None
        self._meta = None
        self._state = None

    def _flight_state(self):
        st = self.base.sim.get_state()
        r = np.asarray(st["r"], float)
        v = np.asarray(st["v"], float)
        alt_km = float(st.get("altitude_km", (np.linalg.norm(r) - cpp.RE_WGS) / 1e3))
        rho = 3e-12
        drag = 5e-6
        sun_b = None
        c_bn = np.asarray(st["C_BN"], float)
        sun_n = np.asarray(st.get("sun_N", np.array([1.0, 0, 0])), float)
        illum = abs(float((c_bn @ sun_n)[2]))
        return {
            "r": r, "v": v, "alt_km": alt_km,
            "soc": self.base._batt_E / self.base._batt_cap_J,
            "rho": rho, "sun_N": sun_n, "f107": self.base._f107,
            "bc_inv": fastprop.bc_from_drag(
                drag, rho, float(np.linalg.norm(v)), self.base._mass),
            "p_gen_sunlit_W": self.base._panel_peak * illum,
            "batt_cap_J": self.base._batt_cap_J,
            "C_BN": c_bn,
        }

    def _advance_main(self):
        a, _ = self.main.predict(self._obs_main, deterministic=True)
        a = np.asarray(a, float)
        qn = a[:4] / max(np.linalg.norm(a[:4]), 1e-9)
        self._q_main = qn if qn[0] >= 0 else -qn
        self._split_main = np.clip(0.5 * (a[4:7] + 1.0), 0, 1) if a.size >= 7 else np.zeros(3)
        self._state = self._flight_state()
        self._hobs, self._meta = duolib.horizon_observation(
            self._state, self._q_main, self.cfg)

    def reset(self, *, seed=None, options=None):
        self._obs_main, info = self.base.reset(seed=seed, options=options)
        self._advance_main()
        return self._hobs, info

    def step(self, action):
        q_h = np.asarray(action, float).reshape(4)
        qn = q_h / max(np.linalg.norm(q_h), 1e-9)
        q_h = qn if qn[0] >= 0 else -qn
        st = self._state
        q_cur = self._q_main            # attitude target currently in force

        # Per-candidate drag for the arbiter and the counterfactual decay.
        v_n = st["v"] / max(np.linalg.norm(st["v"]), 1e-9)
        bcs_h = duolib.attitude_bc_scale(self.base._n, self.base._A,
                                         q_h, q_cur, v_n)
        s_main = dict(st, bc_scale=1.0, power_norm=0.0, downlink_norm=0.0)
        s_hor = dict(st, bc_scale=bcs_h, power_norm=0.0, downlink_norm=0.0)
        for s_, qq in ((s_main, self._q_main), (s_hor, q_h)):
            sig = qq[1:4] / max(1.0 + qq[0], 1e-9)
            c_b = np.asarray(cpp.mrp_to_dcm(sig), float)
            s_["power_norm"] = abs(float((c_b @ st["sun_N"])[2]))
            gs_n, gs_vis, _ = nearest_visible(
                st["r"], cpp.gmst_rad(EPOCH_JD + float(self.base.sim.get_state()["t"]) / 86400.0))
            z_n = c_b.T @ np.array([0.0, 0.0, 1.0])
            s_["downlink_norm"] = max(0.0, float(np.dot(z_n, gs_n))) if gs_vis > 0.5 else 0.0
        q_cmd, arb = duolib.arbitrate(self._q_main, q_h, s_main, s_hor,
                                      self._q_main, self.cfg)

        # 6 h counterfactual decay of the candidate (its own bc).
        h = self.cfg["horizon"]
        decay_prop = fastprop.decay_rate(
            st["r"], st["v"], h["horizon_s"], dt_s=h["prop_dt_s"],
            bc_inv=st["bc_inv"] * bcs_h, rho0=st["rho"], alt0_m=st["alt_km"] * 1e3)
        from arlamx_v2.reward_v8 import duo_reward
        rew, parts = duo_reward({
            "decay_prop_km_d": decay_prop,
            "decay_main_km_d": self._meta["decay_main_km_d"],
            "soc_end_prop": self._meta["soc_end_prop"],
            "q_dot": float(np.dot(q_h, self._q_main)),
        }, self.cfg)

        act = np.concatenate([q_cmd, 2.0 * self._split_main - 1.0]).astype(np.float32)
        self._obs_main, _r, term, trunc, info = self.base.step(act)
        self._advance_main()
        info = dict(info)
        info["duo_choice"] = arb["choice"]
        info["duo_reason"] = arb["reason"]
        info["duo_parts"] = parts
        return self._hobs, float(rew), term, trunc, info

    def close(self):
        self.base.close()


def stage_duo(steps=50_000, main_name=None, sec_pool=("ppo", "sac", "td3"),
              base_variant="v8b", tag="duo"):
    rows = _ledger_rows()
    if main_name:
        main_r = next(r for r in rows if r["name"] == main_name)
        # Secondary: the most ACCURATE delegator among the allowed algos —
        # highest median-free deleg_benefit across every delegating family.
        cand = [r for r in rows if r["algo"] in sec_pool
                and r["variant"] in ("v8b", "v9a", "v9b", "v9c", "v9d")]
        sec_r = max(cand, key=lambda r: float(r["deleg_benefit"]))
        pick = {"main": main_r, "secondary": sec_r}
    else:
        pick = duolib.pick_duo_models(rows)
        main_r, sec_r = pick["main"], pick["secondary"]
    print(f"[duo] main     = {main_r['name']} (decay {main_r['decay_nominal']})", flush=True)
    print(f"[duo] secondary policy class = {sec_r['name']} "
          f"(B {sec_r['deleg_benefit']})", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{tag}_pick.json").write_text(json.dumps(pick, indent=2))

    main_model = load_model(main_r["algo"], main_r["name"])
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    cls = {"sac": SAC, "td3": TD3}.get(sec_r["algo"], PPO)
    arch = arch_list(sec_r["arch"])
    from arlamx_v2.train import torch_threads
    torch_threads()

    def mk(rank):
        def _init():
            return Monitor(HorizonEnv(main_model, seed=SEED + rank,
                                      base_variant=base_variant))
        return _init

    n_envs = 16
    venv = VecNormalize(SubprocVecEnv([mk(i) for i in range(n_envs)]),
                        norm_obs=False, norm_reward=True, clip_reward=10.0)
    name = f"{tag}_horizon_{sec_r['algo']}_{sec_r['arch']}_{steps // 1000}k"
    out = RUNS / name
    (out / "models").mkdir(parents=True, exist_ok=True)
    if sec_r["algo"] == "ppo":
        model = cls("MlpPolicy", venv, learning_rate=3e-4, n_steps=128,
                    batch_size=64, n_epochs=10, gamma=0.99, seed=SEED, verbose=0,
                    policy_kwargs={"net_arch": {"pi": arch, "vf": arch}})
    else:
        model = cls("MlpPolicy", venv, learning_rate=3e-4, buffer_size=100_000,
                    batch_size=256, gamma=0.99, tau=0.005, seed=SEED, verbose=0,
                    policy_kwargs={"net_arch": arch})
    t0 = time.time()
    model.learn(total_timesteps=steps)
    model.save(str(out / "models" / f"{sec_r['algo']}_{name}"))
    print(f"[duo] horizon advisor trained in {time.time()-t0:.0f}s", flush=True)

    # Evaluate the assembled Duo on the probe suite.
    hor_model = model
    duo_cfg = load_cfg("duo")

    def duo_predict(obs, env):
        henv = duo_predict._h
        if henv is None or henv.base is not env:
            # lightweight adapter: reuse the pipeline pieces directly
            duo_predict._h = h = HorizonEnv.__new__(HorizonEnv)
            h.cfg = duo_cfg
            h.base = env
            h.main = main_model
            duo_predict._h = h
        h = duo_predict._h
        h._obs_main = obs
        h._advance_main()
        a, _ = hor_model.predict(h._hobs, deterministic=True)
        st = h._state
        q_h = np.asarray(a, float).reshape(4)
        q_h = q_h / max(np.linalg.norm(q_h), 1e-9)
        v_n = st["v"] / max(np.linalg.norm(st["v"]), 1e-9)
        bcs = duolib.attitude_bc_scale(env._n, env._A, q_h, h._q_main, v_n)
        s_m = dict(st, bc_scale=1.0, power_norm=0.0, downlink_norm=0.0)
        s_h = dict(st, bc_scale=bcs, power_norm=0.0, downlink_norm=0.0)
        q_cmd, _arb = duolib.arbitrate(h._q_main, q_h, s_m, s_h, h._q_main, duo_cfg)
        return np.concatenate([q_cmd, 2.0 * h._split_main - 1.0]).astype(np.float32)
    duo_predict._h = None

    agg, per = probe_suite(duo_predict, base_variant)
    ref = stage_ref()["MPC"]["agg"]
    res = {"agg": agg, "per_probe": per, "gap_index": gap_index(agg, ref),
           "main": main_r["name"], "horizon": name}
    (DATA / f"{tag}_eval.json").write_text(json.dumps(res, indent=2))
    print(f"[duo] gap={res['gap_index']:.3f} decay={agg['decay_nominal']:.2f} "
          f"band={agg['band_pct']:.1f}%", flush=True)
    return res


# ---------------------------------------------------------------------------
# Stage: v9
# ---------------------------------------------------------------------------

def stage_v9(budgets=BUDGETS, arms=("v9a", "v9b"), source="v8b"):
    rows = [r for r in _ledger_rows() if r["variant"] == source]
    if not rows:
        raise SystemExit(f"v9 needs {source} rows in the ledger first")
    ref = stage_ref()["MPC"]["agg"]
    for algo in ALGOS:
        pool = sorted((r for r in rows if r["algo"] == algo),
                      key=lambda r: float(r["gap_index"]))
        best_archs = []
        for r in pool:
            if r["arch"] not in best_archs:
                best_archs.append(r["arch"])
            if len(best_archs) == 2:
                break
        print(f"[v9] {algo}: sizes {best_archs} (from {source})", flush=True)
        for arch_tag in best_archs:
            for steps in budgets:
                for arm in arms:
                    run_one(arm, algo, arch_tag, steps, ref)


# ---------------------------------------------------------------------------
# Stage: traces for the figures
# ---------------------------------------------------------------------------

TRACE_KEYS = ("t_h", "alt_km", "sma_km", "soc", "eclipse", "gates", "split",
              "deleg_B", "deleg_P", "tau_want", "tau_env_filt", "tau_env_raw",
              "tau_aero", "tau_ctrl", "mtq_W", "shortfall", "omega_dps",
              "duo_choice")


def trace_one(predict_fn, variant, probe, seed=1001):
    env = _mk_env(variant, probe)
    obs, _ = env.reset(seed=seed, options=dict(PROBES[probe]["opt"]))
    env._weather_jumps = [(f * env._max_steps, df, da)
                          for f, df, da in PROBES[probe]["jumps"]]
    tr = {k: [] for k in TRACE_KEYS}
    done, k = False, 0
    while not done:
        a = predict_fn(obs, env)
        obs, _r, te, tc, info = env.step(np.asarray(a, np.float32))
        done = te or tc
        k += 1
        tr["t_h"].append(k * env._advisor_s / 3600.0)
        tr["alt_km"].append(float(info["altitude_km"]))
        tr["sma_km"].append(float(info["sma_m"]) / 1e3)
        tr["soc"].append(float(info["battery_soc"]))
        tr["eclipse"].append(float(info.get("eclipse", 1.0)))
        tr["gates"].append(np.asarray(info.get("gates", np.ones(3)), float).tolist())
        tr["split"].append(np.asarray(info.get("deleg_split", np.zeros(3)), float).tolist())
        tr["deleg_B"].append(float(info.get("deleg_B", 0.0)))
        tr["deleg_P"].append(float(info.get("deleg_P", 0.0)))
        for kk, key in (("tau_want", "tau_want"), ("tau_env_filt", "tau_env_filt"),
                        ("tau_env_raw", "tau_env_raw"), ("tau_aero", "tau_aero_mean"),
                        ("tau_ctrl", "tau_ctrl_mean")):
            tr[kk].append(float(np.linalg.norm(
                np.asarray(info.get(key, np.zeros(3)), float))))
        tr["mtq_W"].append(float(info.get("mtq_power_W", 0.0)))
        tr["shortfall"].append(float(info.get("tau_shortfall_frac", 0.0)))
        tr["omega_dps"].append(float(info.get("omega_dps", 0.0)))
        tr["duo_choice"].append(info.get("duo_choice", ""))
    env.close()
    return tr


def stage_trace():
    rows = _ledger_rows()
    if not rows:
        raise SystemExit("trace needs the ledger")
    picks = {}
    for variant in ("v8a", "v8b", "v9a", "v9b", "v9c", "v9d", "v10"):
        pool = sorted((r for r in rows if r["variant"] == variant),
                      key=lambda r: float(r["gap_index"]))
        if pool:
            picks[variant] = pool[0]
    traces = {}
    for lab, fn in (("MPC", mpc_fn()), ("Heuristic", heuristic_fn())):
        traces[lab] = {p: trace_one(fn, "v8a", p) for p in ("nominal", "spike")}
    for variant, row in picks.items():
        model = load_model(row["algo"], row["name"])
        traces[f"{variant} {row['algo'].upper()} {row['arch']}"] = {
            p: trace_one(sb3_fn(model), variant, p) for p in ("nominal", "spike")}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "traces.json").write_text(json.dumps(traces))
    (DATA / "trace_picks.json").write_text(json.dumps(picks, indent=2))
    print(f"[trace] {len(traces)} policies traced", flush=True)


V10_BUDGETS = (50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000)


def stage_v10(budgets=V10_BUDGETS, extra_arch="4x40", winner_seeds=(43, 44)):
    """The v10 matrix: per algo, the two best sizes seen across v9b/c/d, plus
    the new 4x40, at six budgets. Afterwards the single best cell is replicated
    with two extra seeds so the headline number is not a lottery ticket."""
    rows = [r for r in _ledger_rows() if r["variant"] in ("v9b", "v9c", "v9d")]
    if not rows:
        raise SystemExit("v10 needs v9 rows to pick sizes from")
    ref = stage_ref()["MPC"]["agg"]
    for algo in ALGOS:
        pool = sorted((r for r in rows if r["algo"] == algo),
                      key=lambda r: float(r["gap_index"]))
        archs = []
        for r in pool:
            if r["arch"] not in archs:
                archs.append(r["arch"])
            if len(archs) == 2:
                break
        archs.append(extra_arch)
        print(f"[v10] {algo}: sizes {archs}", flush=True)
        for arch_tag in archs:
            for steps in budgets:
                run_one("v10", algo, arch_tag, steps, ref)
    # seed replication of the winner
    v10 = [r for r in _ledger_rows() if r["variant"] == "v10"]
    if v10 and winner_seeds:
        b = min(v10, key=lambda r: float(r["gap_index"]))
        print(f"[v10] replicating winner {b['name']} with seeds {winner_seeds}",
              flush=True)
        for sd in winner_seeds:
            run_one("v10", b["algo"], b["arch"], int(float(b["timesteps"])),
                    ref, seed=sd)


V11_BUDGETS = (50_000, 500_000, 1_000_000, 4_000_000)


def stage_v11(budgets=V11_BUDGETS):
    """v11 matrix: the BEST THREE (algo, size) combos of the v10 ledger, at
    50k / 500k / 1M / 4M, on the damage-training env."""
    rows = [r for r in _ledger_rows() if r["variant"] == "v10"]
    if not rows:
        raise SystemExit("v11 needs v10 rows to pick its classes from")
    ref = stage_ref()["MPC"]["agg"]
    combos, seen = [], set()
    for r in sorted(rows, key=lambda r: float(r["gap_index"])):
        key = (r["algo"], r["arch"])
        if key not in seen:
            seen.add(key)
            combos.append(key)
        if len(combos) == 3:
            break
    print(f"[v11] classes: {combos}", flush=True)
    for algo, arch_tag in combos:
        for steps in budgets:
            run_one("v11", algo, arch_tag, steps, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("burnin", "ref", "sweep", "duo", "v9duo", "v9",
                             "v10", "v11", "trace"))
    ap.add_argument("--budgets", type=str, default="")
    ap.add_argument("--archs", type=str, default="")
    ap.add_argument("--arms", type=str, default="",
                    help="v9 stage: comma list, e.g. v9c,v9d")
    ap.add_argument("--source", type=str, default="v8b",
                    help="v9 stage: which family picks the sizes")
    a = ap.parse_args()
    budgets = tuple(int(x) for x in a.budgets.split(",")) if a.budgets else BUDGETS
    archs = tuple(a.archs.split(",")) if a.archs else ARCHS
    if a.stage == "burnin":
        stage_burnin()
    elif a.stage == "ref":
        stage_ref(force=True)
    elif a.stage == "sweep":
        stage_sweep(budgets=budgets, archs=archs)
    elif a.stage == "duo":
        stage_duo()
    elif a.stage == "v9duo":
        rows = _ledger_rows()
        best = min((r for r in rows if r["variant"] == "v9b"),
                   key=lambda r: float(r["gap_index"]))
        stage_duo(main_name=best["name"], sec_pool=("sac", "td3"),
                  base_variant="v9b", tag="v9duo")
    elif a.stage == "v10":
        stage_v10(budgets=budgets if a.budgets else V10_BUDGETS)
    elif a.stage == "v11":
        stage_v11(budgets=budgets if a.budgets else V11_BUDGETS)
    elif a.stage == "v9":
        arms = tuple(a.arms.split(",")) if a.arms else ("v9a", "v9b")
        stage_v9(budgets=budgets, arms=arms, source=a.source)
    elif a.stage == "trace":
        stage_trace()


if __name__ == "__main__":
    main()
