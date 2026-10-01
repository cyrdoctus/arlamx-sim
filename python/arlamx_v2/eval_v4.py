"""SC_v4 bake-off: heuristic, MPC, min-drag, and trained checkpoints.

Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from arlamx_v2.advisors import HEURISTIC_BANK, MinDragPolicy, SamplingMpcPolicy
from arlamx_v2.env import ArlamxV2Env

ROOT = Path(__file__).resolve().parents[2]
TRAINED = (
    "SC_v4a_4x16_PPO_300k",
    "SC_v4b_4x16_PPO_300k",
    "SC_v4a_4x16_SAC_300k",
    "SC_v4b_4x16_SAC_300k",
    "SC_v4a_4x16_PPO_50k",
    "SC_v4b_4x16_PPO_50k",
)


def _load_sb3(name, out_root):
    from stable_baselines3 import PPO, SAC

    folder = Path(out_root) / name / "models"
    algo = "sac" if "SAC" in name else "ppo"
    stem = folder / f"{algo}_{name}.zip"
    if not stem.exists():
        alt = folder / f"{algo}_{name}"
        path = str(alt)
        if not (folder / f"{algo}_{name}.zip").exists() and not Path(path + ".zip").exists():
            return None
        path = str(stem if stem.exists() else alt)
    else:
        path = str(stem)
    cls = SAC if algo == "sac" else PPO
    return cls.load(path)


def _roll(env, policy_fn, n_ep, seeds):
    rows = []
    for ep, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        ret = 0.0
        soc_min = 1.0
        brown = 0
        cds = []
        alt0 = None
        alt = None
        gs_lock = 0
        steps = 0
        while True:
            act = policy_fn(obs, env)
            obs, rew, term, trunc, info = env.step(act)
            ret += float(rew)
            soc = float(info.get("battery_soc", 1.0))
            soc_min = min(soc_min, soc)
            if soc <= 1e-6:
                brown += 1
            cds.append(float(info.get("Cd", 0.0)))
            alt = float(info.get("altitude_km", 0.0))
            if alt0 is None:
                alt0 = alt
            if float(info.get("gs_visible", 0.0)) > 0.5:
                gs_lock += 1
            steps += 1
            if term or trunc:
                break
        rows.append(
            {
                "ep": ep,
                "seed": int(seed),
                "return": ret,
                "soc_min": soc_min,
                "brownouts": brown,
                "cd_mean": float(np.mean(cds)) if cds else 0.0,
                "dalt_km": (alt - alt0) if alt0 is not None else 0.0,
                "gs_steps": gs_lock,
                "steps": steps,
                "survived": soc_min > 1e-6 and (alt is None or alt >= 250.0),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / ".old" / "analysis" / "sc_v4_eval")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + i for i in range(args.episodes)]
    summary = []

    def run_named(name, variant, fn):
        env = ArlamxV2Env(seed=args.seed, variant=variant)
        recs = _roll(env, fn, args.episodes, seeds)
        env.close()
        agg = {
            "name": name,
            "variant": variant,
            "return_mean": float(np.mean([r["return"] for r in recs])),
            "soc_min": float(np.min([r["soc_min"] for r in recs])),
            "brownouts": int(np.sum([r["brownouts"] for r in recs])),
            "cd_mean": float(np.mean([r["cd_mean"] for r in recs])),
            "dalt_km": float(np.mean([r["dalt_km"] for r in recs])),
            "survived": all(r["survived"] for r in recs),
        }
        summary.append(agg)
        with open(args.out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print(agg, flush=True)

    heur = HEURISTIC_BANK["mission_v17"]
    heur.reset()

    def heur_fn(obs, env):
        st = env.sim.get_state()
        st["battery_soc"] = env._batt_E / env._batt_cap_J
        st["power_gen_norm"] = 0.0
        q, _ = heur.predict(obs, st)
        return q

    def md_fn(obs, env):
        st = env.sim.get_state()
        q, _ = MinDragPolicy().predict(obs, st)
        return q

    mpc = SamplingMpcPolicy(horizon_steps=6, n_random=8, seed=0)
    mpc.reset()

    def mpc_fn(obs, env):
        st = env.sim.get_state()
        st["battery_soc"] = env._batt_E / env._batt_cap_J
        st["rho"] = 1e-12
        st["T"] = 900.0
        st["m_bar"] = 2.656e-26
        q, _ = mpc.predict(obs, st)
        return q

    run_named("min_drag", "v4a", md_fn)
    heur.reset()
    run_named("heuristic_mission", "v4a", heur_fn)
    mpc.reset()
    run_named("mpc", "v4a", mpc_fn)

    for name in TRAINED:
        from arlamx_v2.paths import run_dir as _run_dir
        model = _load_sb3(name, _run_dir(name).parent)
        if model is None:
            print("skip missing", name, flush=True)
            continue
        variant = "v4b" if "v4b" in name else "v4a"

        def rl_fn(obs, env, m=model):
            act, _ = m.predict(obs, deterministic=True)
            return act

        run_named(name, variant, rl_fn)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        keys = [k for k in summary[0] if k != "name"]
        with open(args.out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name"] + keys)
            w.writeheader()
            w.writerows(summary)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
