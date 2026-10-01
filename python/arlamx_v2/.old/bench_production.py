"""Production bake-off: PPO / SAC / TD3 vs the MPC and heuristic advisors.

Retrains the current SC_v7 recipe at two budgets (50k and 250k steps) on the
post-fix plant, scores every policy on the same probe suite the tuning campaign
used, and records per-step traces for the presentation figures.

Reuses campaign.probe_suite / gap_index / mpc_predict_fn so the numbers stay
comparable with outputs/campaign/LEADERBOARD.md. Writes only under
outputs/presentation/data/ and outputs/presentation/runs/.

    PYTHONPATH=python python -m arlamx_v2.bench_production --stage all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.campaign import (EPOCH_JD, PROBES, _install_probe_jumps, _probe_env,
                                gap_index, mpc_predict_fn, probe_suite,
                                sb3_predict_fn)
from arlamx_v2.env import MTQ_TAU_NM, ArlamxV2Env
from arlamx_v2.paths import OUTPUTS
from arlamx_v2.train import train_one

OUT = OUTPUTS / "presentation"
DATA = OUT / "data"
RUNS = OUT / "runs"

# The stage-S4 winner configuration from the SC_v7 campaign (best gap_index of
# every arm that was swept). Held fixed across algorithms so the comparison
# isolates the learner, not the reward shaping.
V7_ENV_KW = {
    "kp": 1.0e-4,
    "kd": 2.0e-3,
    "obs_extra": "observer",
    "gate_floor": 0.3,
    "reward_w": {"band_above_mode": "exp", "dE_weight": 3.5, "thrift_weight": 0.9},
}

ALGOS = ("ppo", "sac", "td3")
BUDGETS = (50_000, 250_000)
# Three seeds per cell. The published campaign leaderboard reported a best-of-53
# trial, so a single fresh seed cannot be compared against it honestly; the
# spread across seeds is itself one of the results.
SEEDS = (42, 43, 44)
N_ENVS = 32

# Observation indices (see env._obs).
OBS_SOC = 20
OBS_GEN = 21


def run_name(algo, steps, seed):
    return f"prod_{algo}_{steps // 1000}k_s{seed}"


def cell_label(algo, steps):
    return f"{algo.upper()} {steps // 1000}k"


# ---------------------------------------------------------------------------
# Heuristic advisor driven through the same probe env as the MPC
# ---------------------------------------------------------------------------

def heuristic_predict_fn():
    """Drive HeuristicPowerPolicy with the inputs a flight computer would have:
    ephemeris from the propagator, SoC and panel current from the EPS, and the
    ground-station direction from the stored catalogue."""
    from arlamx_v2.advisors.heuristic import HeuristicPowerPolicy
    from arlamx_v2.advisors.stations import nearest_visible

    pol = HeuristicPowerPolicy()
    holder = {"env": None}

    def f(obs, env):
        if holder["env"] is not env:
            pol.reset()
            holder["env"] = env
        st = env.sim.get_state()
        r = np.asarray(st["r"], float)
        jd = EPOCH_JD + float(st["t"]) / 86400.0
        gmst = cpp.gmst_rad(jd)
        gs_n, gs_vis, _el = nearest_visible(r, gmst)
        c_bn = np.asarray(st["C_BN"], float)
        state = {
            "r": r,
            "v": np.asarray(st["v"], float),
            "C_BN": c_bn,
            "battery_soc": float(obs[OBS_SOC]),
            "power_gen_norm": float(obs[OBS_GEN]),
            "gs_dir_N": gs_n if gs_vis > 0.5 else None,
            "gs_dir_B": (c_bn @ gs_n) if gs_vis > 0.5 else None,
            "gs_visible": gs_vis,
        }
        q, _meta = pol.predict(obs, state)
        return np.concatenate([q, [1.0, 1.0, 1.0]]).astype(np.float32)

    return f


# ---------------------------------------------------------------------------
# Traced rollout (per-step series for the figures)
# ---------------------------------------------------------------------------

def _rot_z(v, ang):
    """Inertial -> Earth-fixed: the same R3(GMST) the plant applies."""
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


TRACE_KEYS = ("t_h", "alt_km", "sma_km", "soc", "cd", "eclipse", "effort",
              "gs_vis", "dl", "tau_ctrl", "tau_dist", "tau_aero", "gate",
              "omega_dps", "bfield_uT", "dE_lift", "dE_actual", "dE_baseline")


def trace_rollout(predict_fn, probe="nominal", seed=1001, variant="v7",
                  env_kw=None, orbits=20.0):
    """One deterministic episode, recording the series the figures need."""
    env = _probe_env(variant, env_kw if env_kw is not None else V7_ENV_KW, probe)
    period = 2 * np.pi * np.sqrt((6371e3 + PROBES[probe]["opt"]["altitude_km"] * 1e3) ** 3
                                 / 3.986004418e14)
    env._max_steps = int(orbits * period / env._advisor_s)
    obs, _ = env.reset(seed=seed, options=dict(PROBES[probe]["opt"]))
    _install_probe_jumps(env, probe)

    tr = {k: [] for k in TRACE_KEYS}
    done, k = False, 0
    while not done:
        a = predict_fn(obs, env)
        obs, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        done = term or trunc
        k += 1
        st = env.sim.get_state()
        # B_B is not published in info; rebuild it from the state the same way
        # the plant does, so the trace shows the field the ADCS actually sees.
        r = np.asarray(st["r"], float)
        jd = EPOCH_JD + float(st["t"]) / 86400.0
        b_e = np.asarray(cpp.dipole_field_ecef(_rot_z(r, cpp.gmst_rad(jd))), float)
        tau_c = np.asarray(info.get("tau_ctrl_mean", np.zeros(3)), float)
        tau_d = np.asarray(info.get("tau_dist_est", np.zeros(3)), float)
        tau_a = np.asarray(info.get("tau_aero_mean", np.zeros(3)), float)
        vis = float(info.get("gs_visible", 0.0)) > 0.5
        lit = float(info.get("eclipse", 1.0)) > 0.5
        cosg = float(info.get("gs_point_cos", 0.0))
        tr["t_h"].append(k * env._advisor_s / 3600.0)
        tr["alt_km"].append(float(info["altitude_km"]))
        tr["sma_km"].append(float(info["sma_m"]) / 1e3)
        tr["soc"].append(float(info["battery_soc"]))
        tr["cd"].append(float(info.get("Cd", np.nan)))
        tr["eclipse"].append(float(info.get("eclipse", 1.0)))
        tr["effort"].append(float(info.get("torque_effort", 0.0)))
        tr["gs_vis"].append(1.0 if vis else 0.0)
        tr["dl"].append(1.0 if (vis and lit and cosg > 0.7) else 0.0)
        tr["tau_ctrl"].append(np.abs(tau_c / MTQ_TAU_NM).tolist())
        tr["tau_dist"].append(np.abs(tau_d / MTQ_TAU_NM).tolist())
        tr["tau_aero"].append(np.abs(tau_a / MTQ_TAU_NM).tolist())
        tr["gate"].append(float(np.mean(np.asarray(info.get("gates", np.ones(3)), float))))
        tr["omega_dps"].append(float(info.get("omega_dps", 0.0)))
        tr["bfield_uT"].append(float(np.linalg.norm(b_e)) * 1e6)
        tr["dE_lift"].append(float(info.get("dE_lift", 0.0)))
        tr["dE_actual"].append(float(info.get("dE_actual", 0.0)))
        tr["dE_baseline"].append(float(info.get("dE_baseline", 0.0)))
    env.close()
    return {k: np.asarray(v).tolist() for k, v in tr.items()}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_train(force=False):
    """Train 3 algorithms x 2 budgets x 3 seeds on the fixed V7 recipe."""
    RUNS.mkdir(parents=True, exist_ok=True)
    out = {}
    for steps in BUDGETS:
        for algo in ALGOS:
            for seed in SEEDS:
                name = run_name(algo, steps, seed)
                mpath = RUNS / name / "models" / f"{algo}_{name}.zip"
                if mpath.exists() and not force:
                    print(f"[skip] {name} already trained", flush=True)
                    out[name] = json.loads((RUNS / name / "metrics.json").read_text())
                    continue
                print(f"[train] {name}", flush=True)
                t0 = time.time()
                m = train_one("v7", algo, steps, N_ENVS, seed, name, RUNS,
                              env_kw=dict(V7_ENV_KW))
                m["wall_s"] = round(time.time() - t0, 1)
                out[name] = m
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "train_metrics.json").write_text(json.dumps(out, indent=2))
    return out


def _load(algo, name):
    from stable_baselines3 import PPO, SAC, TD3
    cls = {"sac": SAC, "td3": TD3}.get(algo, PPO)
    return cls.load(str(RUNS / name / "models" / f"{algo}_{name}.zip"))


def policy_table(seeds=SEEDS):
    """(label, algo, predict_fn factory) for every policy in the bake-off."""
    rows = [("MPC", "mpc", mpc_predict_fn),
            ("Heuristic", "heuristic", heuristic_predict_fn)]
    for steps in BUDGETS:
        for algo in ALGOS:
            for seed in seeds:
                name = run_name(algo, steps, seed)
                rows.append((f"{cell_label(algo, steps)} s{seed}", algo,
                             (lambda a=algo, n=name: sb3_predict_fn(_load(a, n)))))
    return rows


def repr_seed(results, algo, steps):
    """Median-gap seed of a cell — the honest single representative for traces."""
    cands = [(v["gap_index"], s) for s in SEEDS
             for k, v in results.items()
             if k == f"{cell_label(algo, steps)} s{s}"]
    if not cands:
        return SEEDS[0]
    cands.sort()
    return cands[len(cands) // 2][1]


def stage_eval():
    """Score every policy on the probe suite; MPC is the reference."""
    DATA.mkdir(parents=True, exist_ok=True)
    results = {}
    ref = None
    for label, algo, mk in policy_table():
        # The scripted advisors run with the IPC block masked off: they do not
        # consume the observer features, and this matches the campaign
        # reference exactly.
        env_kw = {"obs_extra": "none"} if algo in ("mpc", "heuristic") else dict(V7_ENV_KW)
        print(f"[eval] {label}", flush=True)
        t0 = time.time()
        agg, per = probe_suite(mk(), "v7", env_kw)
        if label == "MPC":
            ref = agg
        results[label] = {"algo": algo, "agg": agg, "per_probe": per,
                          "eval_s": round(time.time() - t0, 1)}
        print(f"   decay={agg['decay_nominal']:.2f} km/d  band={agg['band_pct']:.1f}%  "
              f"brn={agg['brownouts']:.0f}  dl={agg['downlink_min_d']:.0f} min/d  "
              f"actJ={agg['act_energy_J']:.0f}  ({results[label]['eval_s']}s)", flush=True)
    for label, row in results.items():
        row["gap_index"] = gap_index(row["agg"], ref)
    (DATA / "bakeoff.json").write_text(json.dumps(results, indent=2))
    return results


def _representative_table():
    """MPC, heuristic, and one median-gap seed per (algo, budget) cell."""
    results = json.loads((DATA / "bakeoff.json").read_text())
    rows = [("MPC", "mpc", mpc_predict_fn),
            ("Heuristic", "heuristic", heuristic_predict_fn)]
    for steps in BUDGETS:
        for algo in ALGOS:
            seed = repr_seed(results, algo, steps)
            name = run_name(algo, steps, seed)
            rows.append((cell_label(algo, steps), algo,
                         (lambda a=algo, n=name: sb3_predict_fn(_load(a, n)))))
    return rows


def stage_trace():
    """Per-step traces for the figures: nominal and storm-spike probes."""
    DATA.mkdir(parents=True, exist_ok=True)
    traces = {}
    for label, algo, mk in _representative_table():
        env_kw = {"obs_extra": "none"} if algo in ("mpc", "heuristic") else dict(V7_ENV_KW)
        traces[label] = {}
        for probe in ("nominal", "spike"):
            print(f"[trace] {label} / {probe}", flush=True)
            traces[label][probe] = trace_rollout(mk(), probe=probe, env_kw=env_kw)
    (DATA / "traces.json").write_text(json.dumps(traces))
    return traces


def stage_inference_cost():
    """Wall-clock cost of one control decision — the deployability argument."""
    import timeit
    rows = {}
    for label, algo, mk in _representative_table():
        env_kw = {"obs_extra": "none"} if algo in ("mpc", "heuristic") else dict(V7_ENV_KW)
        env = _probe_env("v7", env_kw, "nominal")
        obs, _ = env.reset(seed=1001, options=dict(PROBES["nominal"]["opt"]))
        fn = mk()
        for _ in range(5):
            fn(obs, env)
        n = 20 if algo == "mpc" else 200
        dt = timeit.timeit(lambda: fn(obs, env), number=n) / n
        rows[label] = {"algo": algo, "ms_per_decision": dt * 1e3}
        print(f"[infer] {label}: {dt*1e3:.3f} ms/decision", flush=True)
        env.close()
    (DATA / "inference_cost.json").write_text(json.dumps(rows, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "train", "eval", "trace", "infer"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage in ("all", "train"):
        stage_train(force=a.force)
    if a.stage in ("all", "eval"):
        stage_eval()
    if a.stage in ("all", "trace"):
        stage_trace()
    if a.stage in ("all", "infer"):
        stage_inference_cost()
    print("[bench_production] done", flush=True)


if __name__ == "__main__":
    main()
