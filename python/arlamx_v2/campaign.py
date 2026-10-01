"""SC_v7 IPC tuning campaign: many small PPO/SAC runs, probe evals, a ledger.

Stages (run: PYTHONPATH=python python -m arlamx_v2.campaign --stage s1|s2|s3|s4|mpc):
  mpc  compute the MPC reference row on the probe suite (once)
  s1   controller-gain grid, NO training (scripted attitude tracking probes)
  s2   reward-weight sweep, gates pinned open (isolates the reward)
  s3   observation ablation (none / ctrl / observer / observer+foresight)
  s4   IPC gates on: gate floor x thrift weight sweep
Every trial trains via train.train_one, is evaluated on the standardized probe
suite, and appends one row to outputs/campaign/ledger.csv; LEADERBOARD.md is
regenerated after each row, sorted by the MPC-gap index (negative = beats MPC).
Completed trials (ledger rows) are skipped, so stages are resumable.
Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "python"))

from arlamx_v2 import cpp                      # noqa: E402
from arlamx_v2.env import ArlamxV2Env, EPOCH_JD, MTQ_KP, MTQ_KD  # noqa: E402
from arlamx_v2.train import train_one          # noqa: E402

OUT = ROOT / "outputs" / ".old" / "campaign"
RUNS = OUT / "runs"
LEDGER = OUT / "ledger.csv"
LEADER = OUT / "LEADERBOARD.md"
MPC_REF = OUT / "mpc_reference.json"

PROBE_ORBITS = 20.0
PROBE_SEEDS = (1001, 1002)

# Probe scenarios: fixed reset options; "jumps" are deterministic
# (step_fraction, dF107, dAp) storm steps installed after reset (the random
# per-episode jump plan is cleared so probes are identical across policies).
PROBES = {
    "nominal": dict(opt=dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001,
                             f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                             nu_deg=0.0, mass_kg=0.625, soc=0.5,
                             omega_dps=[1.0, 1.0, 0.5]),
                    jumps=[]),
    "spike": dict(opt=dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001,
                           f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                           nu_deg=0.0, mass_kg=0.625, soc=0.5,
                           omega_dps=[1.0, 1.0, 0.5]),
                  jumps=[(0.25, 100.0, 36.0), (0.60, -100.0, -36.0)]),
    "offnominal": dict(opt=dict(altitude_km=550.0, inc_deg=51.6, ecc=0.001,
                                f107=150.0, ap=4.0, raan_deg=40.0, argp_deg=0.0,
                                nu_deg=0.0, mass_kg=0.625, soc=0.5,
                                omega_dps=[1.0, 1.0, 0.5]),
                       jumps=[]),
}

GAP_METRICS = [
    # (ledger key, lower_is_better, denominator floor — keeps the index sane
    #  when the MPC reference value is ~0, e.g. zero brownouts)
    ("decay_nominal", True, 1.0),
    ("band_pct", False, 10.0),
    ("brownouts", True, 1.0),
    ("downlink_min_d", False, 10.0),
    ("decay_spike", True, 1.0),
]


# ---------------------------------------------------------------------------
# Probe rollouts
# ---------------------------------------------------------------------------

def _probe_env(variant, env_kw, probe):
    env = ArlamxV2Env(seed=0, variant=variant, **(env_kw or {}))
    alt = float(PROBES[probe]["opt"]["altitude_km"])
    period = 2 * np.pi * np.sqrt((6371e3 + alt * 1e3) ** 3 / 3.986004418e14)
    env._max_steps = int(PROBE_ORBITS * period / env._advisor_s)
    return env


def _install_probe_jumps(env, probe):
    env._weather_jumps = [
        (frac * env._max_steps, df, da) for frac, df, da in PROBES[probe]["jumps"]
    ]
    env._srp_flashes = []


def _rollout(env, predict_fn, probe, seed):
    obs, _ = env.reset(seed=seed, options=dict(PROBES[probe]["opt"]))
    _install_probe_jumps(env, probe)
    hist = {k: [] for k in ("sma_m", "soc", "dep", "dl", "gs_err", "cd",
                            "effort", "day_pass")}
    done = False
    while not done:
        a = predict_fn(obs, env)
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        done = term or trunc
        hist["sma_m"].append(float(info["sma_m"]))
        hist["soc"].append(float(info["battery_soc"]))
        hist["dep"].append(1.0 if info.get("battery_depleted") else 0.0)
        vis = float(info.get("gs_visible", 0.0)) > 0.5
        lit = float(info.get("eclipse", 1.0)) > 0.5
        cosg = float(info.get("gs_point_cos", 0.0))
        hist["dl"].append(1.0 if (vis and lit and cosg > 0.7) else 0.0)
        err = np.asarray(info.get("gs_err_axes_deg", np.zeros(3)), float)
        hist["gs_err"].append(float(np.mean(np.abs(err))) if (vis and lit) else np.nan)
        hist["day_pass"].append(1.0 if (vis and lit) else 0.0)
        hist["cd"].append(float(info.get("Cd", np.nan)))
        hist["effort"].append(float(info.get("torque_effort", 0.0)))
    n = max(1, len(hist["soc"]))
    days = n * env._advisor_s / 86400.0
    soc = np.asarray(hist["soc"])
    dep = np.asarray(hist["dep"]) > 0.5
    sma = np.asarray(hist["sma_m"])
    return {
        "decay_km_d": float((sma[0] - sma[-1]) / 1e3 / max(days, 1e-9)),
        "band_pct": float(np.mean((soc >= 0.4) & (soc <= 0.6)) * 100.0),
        "brownouts": int(np.sum(np.diff(dep.astype(int)) > 0) + (1 if dep[:1].any() else 0)),
        "downlink_min_d": float(np.sum(hist["dl"]) * env._advisor_s / 60.0 / max(days, 1e-9)),
        "gs_err_deg": (float(np.nanmean(hist["gs_err"]))
                       if np.any(np.isfinite(hist["gs_err"])) else float("nan")),
        "cd_mean": float(np.nanmean(hist["cd"])),
        "effort_mean": float(np.mean(hist["effort"])),
        "act_energy_J": float(np.sum(hist["effort"]) * 0.270 * env._advisor_s),
        "steps": int(n),
    }


def probe_suite(predict_fn, variant, env_kw):
    """Run all probes x seeds; return per-probe means + headline aggregates."""
    per = {}
    for probe in PROBES:
        rows = []
        env = _probe_env(variant, env_kw, probe)
        for seed in PROBE_SEEDS:
            rows.append(_rollout(env, predict_fn, probe, seed))
        env.close()
        per[probe] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    agg = {
        "decay_nominal": per["nominal"]["decay_km_d"],
        "decay_spike": per["spike"]["decay_km_d"],
        "decay_offnominal": per["offnominal"]["decay_km_d"],
        "band_pct": float(np.mean([per[p]["band_pct"] for p in per])),
        "brownouts": float(np.sum([per[p]["brownouts"] for p in per])),
        "downlink_min_d": float(np.mean([per[p]["downlink_min_d"] for p in per])),
        "gs_err_deg": float(np.nanmean([per[p]["gs_err_deg"] for p in per])),
        "cd_mean": float(np.mean([per[p]["cd_mean"] for p in per])),
        "act_energy_J": float(np.sum([per[p]["act_energy_J"] for p in per])),
    }
    return agg, per


def gap_index(agg, ref):
    """Signed normalized distance to the MPC reference (negative = beats it)."""
    gaps = []
    for key, lower_better, floor in GAP_METRICS:
        pol, mpc = float(agg[key]), float(ref[key])
        den = max(abs(mpc), floor)
        gaps.append((pol - mpc) / den if lower_better else (mpc - pol) / den)
    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# Policies for probing
# ---------------------------------------------------------------------------

def sb3_predict_fn(model):
    def f(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return a
    return f


def load_trial_model(algo, name):
    from stable_baselines3 import PPO, SAC, TD3
    path = RUNS / name / "models" / f"{algo}_{name}.zip"
    cls = {"sac": SAC, "td3": TD3}.get(algo, PPO)
    return cls.load(str(path))


def mpc_predict_fn():
    """Drive the SamplingMpcPolicy through the v7 probe env (gates full open),
    building its state dict from the sim + an MSIS query, exactly the inputs
    an onboard planner would have."""
    from arlamx_v2.advisors.mpc import SamplingMpcPolicy
    from arlamx_v2.advisors.stations import nearest_visible
    from arlamx_v2.atmosphere import jd_to_datetime, query_msis

    pol = SamplingMpcPolicy(seed=0)
    holder = {"soc": 0.5, "env": None}

    def f(obs, env):
        if holder["env"] is not env:      # new env/episode
            pol.reset()
            pol.panels = (env._n, env._A, env._c)
            pol.advisor_step_s = env._advisor_s
            holder["env"] = env
            holder["soc"] = env._batt_E / env._batt_cap_J
        st = env.sim.get_state()
        r = np.asarray(st["r"], float)
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
        state = {
            "r": r, "v": np.asarray(st["v"], float),
            "sun_N": np.asarray(st.get("sun_N", np.zeros(3)), float),
            "battery_soc": env._batt_E / env._batt_cap_J,
            "rho": rho, "T": T, "m_bar": mb,
            "gs_dir_N": gs_n if gs_vis > 0.5 else None, "gs_visible": gs_vis,
        }
        q, _meta = pol.predict(obs, state)
        return np.concatenate([q, [1.0, 1.0, 1.0]]).astype(np.float32)
    return f


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

LEDGER_FIELDS = [
    "name", "stage", "variant", "algo", "timesteps", "seed", "wall_s",
    "env_kw", "reward_w", "ppo_kw",
    "decay_nominal", "decay_spike", "decay_offnominal", "band_pct",
    "brownouts", "downlink_min_d", "gs_err_deg", "cd_mean", "act_energy_J",
    "gap_index",
]


def ledger_rows():
    if not LEDGER.exists():
        return []
    with open(LEDGER) as f:
        return list(csv.DictReader(f))


def ledger_append(row):
    OUT.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            wr.writeheader()
        wr.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})
    regen_leaderboard()


def regen_leaderboard():
    rows = ledger_rows()
    if not rows:
        return
    def _g(r):
        try:
            return float(r["gap_index"])
        except (ValueError, TypeError):
            return float("inf")
    rows = sorted(rows, key=_g)
    lines = ["# SC_v7 tuning campaign — leaderboard",
             "",
             "gap_index: mean normalized distance to the MPC reference over "
             "{decay_nominal, band%, brownouts, downlink, decay_spike}. "
             "**Negative beats the MPC.**", "",
             "| # | name | stage | algo | gap | decay_nom | decay_spike | band% "
             "| brownouts | dl min/d | gs err | Cd | act J |",
             "|---|------|-------|------|-----|-----------|-------------|------"
             "|-----------|----------|--------|----|-------|"]
    for i, r in enumerate(rows, 1):
        def fv(k, nd=2):
            try:
                return f"{float(r[k]):.{nd}f}"
            except (ValueError, TypeError):
                return "-"
        lines.append(
            f"| {i} | {r['name']} | {r['stage']} | {r['algo']} | {fv('gap_index',3)} "
            f"| {fv('decay_nominal')} | {fv('decay_spike')} | {fv('band_pct',1)} "
            f"| {fv('brownouts',0)} | {fv('downlink_min_d',0)} | {fv('gs_err_deg',0)} "
            f"| {fv('cd_mean')} | {fv('act_energy_J',0)} |")
    LEADER.write_text("\n".join(lines) + "\n")


def mpc_reference(force=False):
    if MPC_REF.exists() and not force:
        return json.loads(MPC_REF.read_text())
    print("[campaign] computing MPC reference on the probe suite ...", flush=True)
    agg, per = probe_suite(mpc_predict_fn(), "v7", {"obs_extra": "none"})
    OUT.mkdir(parents=True, exist_ok=True)
    MPC_REF.write_text(json.dumps({"agg": agg, "per_probe": per}, indent=2))
    print(json.dumps(agg, indent=2), flush=True)
    ledger_append({"name": "MPC_reference", "stage": "ref", "variant": "-",
                   "algo": "mpc", "timesteps": 0, "seed": 0, "wall_s": 0,
                   "env_kw": "", "reward_w": "", "ppo_kw": "",
                   **{k: agg[k] for k in ("decay_nominal", "decay_spike",
                                          "decay_offnominal", "band_pct",
                                          "brownouts", "downlink_min_d",
                                          "gs_err_deg", "cd_mean",
                                          "act_energy_J")},
                   "gap_index": 0.0})
    return {"agg": agg, "per_probe": per}


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(trial, ref_agg):
    name = trial["name"]
    done = {r["name"] for r in ledger_rows()}
    if name in done:
        print(f"[skip] {name} already in ledger", flush=True)
        return
    print(f"[trial] {name} :: {trial}", flush=True)
    t0 = time.time()
    train_one(trial.get("variant", "v7"), trial["algo"], trial["timesteps"],
              trial.get("n_envs", 32), trial.get("seed", 42), name, RUNS,
              ppo_kw=trial.get("ppo_kw"), env_kw=trial.get("env_kw"))
    model = load_trial_model(trial["algo"], name)
    agg, per = probe_suite(sb3_predict_fn(model),
                           trial.get("variant", "v7"), trial.get("env_kw"))
    (RUNS / name / "probe_metrics.json").write_text(
        json.dumps({"agg": agg, "per_probe": per}, indent=2))
    row = {
        "name": name, "stage": trial.get("stage", "-"),
        "variant": trial.get("variant", "v7"), "algo": trial["algo"],
        "timesteps": trial["timesteps"], "seed": trial.get("seed", 42),
        "wall_s": round(time.time() - t0, 1),
        "env_kw": json.dumps({k: v for k, v in (trial.get("env_kw") or {}).items()
                              if k != "reward_w"}),
        "reward_w": json.dumps((trial.get("env_kw") or {}).get("reward_w") or {}),
        "ppo_kw": json.dumps(trial.get("ppo_kw") or {}),
        **agg,
        "gap_index": gap_index(agg, ref_agg),
    }
    ledger_append(row)
    print(f"[trial done] {name} gap={row['gap_index']:.3f} "
          f"decay={agg['decay_nominal']:.2f} band={agg['band_pct']:.1f}% "
          f"brn={agg['brownouts']:.0f} dl={agg['downlink_min_d']:.0f}", flush=True)


# ---------------------------------------------------------------------------
# Stage S1: controller gains, no RL (scripted tracking probes)
# ---------------------------------------------------------------------------

def stage_s1():
    """Grid kp x kd: track a rotating z-along-velocity target for 3 orbits at
    400 km; report tracking error, realized Cd, effort. Writes s1_gains.json
    and prints the winner (lowest err with effort not worse than 2x best)."""
    from arlamx_v2.advisors.attitudes import quat_body_z_along

    kps = (1e-4, 4e-4, 1.6e-3, 6.4e-3)
    kds = (2e-3, 8e-3, 3.2e-2)
    results = []
    for kp in kps:
        for kd in kds:
            env = ArlamxV2Env(seed=0, variant="v7",
                              kp=kp, kd=kd, obs_extra="none")
            period = 2 * np.pi * np.sqrt((6371e3 + 400e3) ** 3 / 3.986004418e14)
            env._max_steps = int(3 * period / env._advisor_s)
            obs, _ = env.reset(seed=7, options=dict(PROBES["nominal"]["opt"]))
            env._weather_jumps = []
            errs, efforts, cds = [], [], []
            done = False
            while not done:
                st = env.sim.get_state()
                r = np.asarray(st["r"], float)
                v = np.asarray(st["v"], float)
                q = quat_body_z_along(v / np.linalg.norm(v), helper_n=np.cross(r, v))
                a = np.concatenate([q, [1.0, 1.0, 1.0]]).astype(np.float32)
                obs, _rw, term, trunc, info = env.step(a)
                done = term or trunc
                errs.append(np.degrees(info.get("tracking_err_rad", 0.0)))
                efforts.append(info.get("torque_effort", 0.0))
                cds.append(info.get("Cd", np.nan))
            env.close()
            row = {"kp": kp, "kd": kd,
                   "track_err_deg": float(np.mean(errs)),
                   "effort": float(np.mean(efforts)),
                   "cd_mean": float(np.nanmean(cds))}
            results.append(row)
            print(f"[s1] kp={kp:g} kd={kd:g} err={row['track_err_deg']:.2f} deg "
                  f"effort={row['effort']:.4f} Cd={row['cd_mean']:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s1_gains.json").write_text(json.dumps(results, indent=2))
    best = min(results, key=lambda r: r["track_err_deg"])
    print(f"[s1 winner] kp={best['kp']:g} kd={best['kd']:g} "
          f"err={best['track_err_deg']:.2f} deg", flush=True)
    return best


# ---------------------------------------------------------------------------
# Stage trial lists (S2-S4). Gains are read from s1_gains.json's winner when
# present, else the env defaults.
# ---------------------------------------------------------------------------

def _s1_gains():
    p = OUT / "s1_gains.json"
    if p.exists():
        rows = json.loads(p.read_text())
        best = min(rows, key=lambda r: r["track_err_deg"])
        return {"kp": best["kp"], "kd": best["kd"]}
    return {}


def stage_s2_trials():
    g = _s1_gains()
    base = dict(stage="s2", variant="v7", timesteps=50_000, n_envs=32, seed=42)
    trials = []
    grid = []
    for above in ("exp", "linear_penalty"):
        for dE in (1.0, 2.0, 3.5):
            grid.append({"band_above_mode": above, "dE_weight": dE})
    for gcfg in grid:
        for tiers, tag in ((None, "t2"), (((10.0, 25.0, 45.0),
                                           (1.0, 0.5, 0.25)), "t10")):
            w = dict(gcfg)
            if tiers:
                w["tier_deg"], w["tier_scale"] = tiers
            nm = (f"s2_{gcfg['band_above_mode'][:3]}_dE{gcfg['dE_weight']:g}"
                  f"_{tag}").replace(".", "p")
            trials.append({**base, "algo": "ppo", "name": nm,
                           "env_kw": {**g, "obs_extra": "none",
                                      "gate_floor": 1.0,   # gates pinned open
                                      "reward_w": w}})
    return trials


def stage_s3_trials(best_w):
    g = _s1_gains()
    base = dict(stage="s3", variant="v7", timesteps=50_000, n_envs=32, seed=42)
    return [{**base, "algo": "ppo", "name": f"s3_obs_{m}",
             "env_kw": {**g, "obs_extra": m, "gate_floor": 1.0,
                        "reward_w": best_w}}
            for m in ("none", "ctrl", "observer", "observer_fore")]


def stage_s4_trials(best_w, best_obs):
    g = _s1_gains()
    base = dict(stage="s4", variant="v7", timesteps=50_000, n_envs=32, seed=42)
    # The gates only have a *reason* when the policy can sense the disturbance
    # torque, so S4 tests every (floor, thrift) cell under BOTH the ledger-best
    # obs mode and the observer mode when they differ — the IPC hypothesis is
    # specifically the gates+observer pairing.
    obs_modes = [best_obs] if best_obs == "observer" else [best_obs, "observer"]
    trials = []
    for floor in (0.05, 0.15, 0.3):
        for thrift in (0.25, 0.6, 0.9):
            for obs in obs_modes:
                w = dict(best_w or {})
                w["thrift_weight"] = thrift
                tag = "" if obs == best_obs else "_obs"
                nm = f"s4_f{floor:g}_th{thrift:g}{tag}".replace(".", "p")
                trials.append({**base, "algo": "ppo", "name": nm,
                               "env_kw": {**g, "obs_extra": obs,
                                          "gate_floor": floor, "reward_w": w}})
    return trials


def _best_row(stage):
    rows = [r for r in ledger_rows() if r.get("stage") == stage]
    if not rows:
        return None
    return min(rows, key=lambda r: float(r["gap_index"]))


def run_stage(stage, sac_top=3):
    ref = mpc_reference()["agg"]
    if stage == "s1":
        stage_s1()
        return
    if stage == "s2":
        trials = stage_s2_trials()
    elif stage == "s3":
        best = _best_row("s2")
        best_w = json.loads(best["reward_w"]) if best else {}
        trials = stage_s3_trials(best_w)
    elif stage == "s4":
        best2 = _best_row("s2")
        best3 = _best_row("s3")
        best_w = json.loads(best2["reward_w"]) if best2 else {}
        best_obs = (json.loads(best3["env_kw"]).get("obs_extra", "observer")
                    if best3 else "observer")
        trials = stage_s4_trials(best_w, best_obs)
    else:
        raise ValueError(stage)
    for t in trials:
        run_trial(t, ref)
    # SAC on the stage's top PPO configs.
    rows = sorted((r for r in ledger_rows()
                   if r.get("stage") == stage and r["algo"] == "ppo"),
                  key=lambda r: float(r["gap_index"]))[:sac_top]
    for r in rows:
        env_kw = json.loads(r["env_kw"])
        env_kw["reward_w"] = json.loads(r["reward_w"])
        run_trial({"name": r["name"] + "_sac", "stage": stage, "variant": "v7",
                   "algo": "sac", "timesteps": int(r["timesteps"]),
                   "n_envs": 32, "seed": 42, "env_kw": env_kw}, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("mpc", "s1", "s2", "s3", "s4"))
    ap.add_argument("--force-ref", action="store_true")
    args = ap.parse_args()
    if args.stage == "mpc":
        mpc_reference(force=args.force_ref)
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
