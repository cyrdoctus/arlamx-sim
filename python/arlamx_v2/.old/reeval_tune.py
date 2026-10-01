"""Robust re-eval of the 32 PPO burn-in trials, then extra consistency seeds.

The original board used 1 episode x 16 steps, which made +31 look like a
winner. This script uses 4 episodes x 64 steps and writes robust_best.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.train import train_one
from arlamx_v2.paths import run_dir as _run_dir

ROOT = Path(__file__).resolve().parents[2]


def eval_ckpt(name, n_ep=4, n_steps=64, variant="v4a"):
    from stable_baselines3 import PPO

    z = _run_dir(name) / "models" / f"ppo_{name}.zip"
    if not z.exists():
        return {"return_mean": -1e9, "return_std": 0.0, "soc_min": 0.0, "n_ep": 0}
    model = PPO.load(str(z))
    env = ArlamxV2Env(seed=99, variant=variant)
    rets, socs = [], []
    for ep in range(n_ep):
        obs, _ = env.reset(
            seed=200 + ep,
            options={"altitude_km": 400.0, "inc_deg": 23.0, "ecc": 0.001, "nu_deg": 90.0, "f107": 150.0, "ap": 4.0, "soc": 0.5, "mass_kg": 0.625},
        )
        ret = 0.0
        sm = 1.0
        for _ in range(n_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ret += float(r)
            sm = min(sm, float(info["battery_soc"]))
            if term or trunc:
                break
        rets.append(ret)
        socs.append(sm)
    env.close()
    return {
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "soc_min": float(np.min(socs)),
        "n_ep": n_ep,
        "n_steps": n_steps,
    }


def main():
    board_p = ROOT / "outputs" / "tuning" / "tune_ppo" / "board.json"
    best_p = ROOT / "outputs" / "tuning" / "tune_ppo" / "best.json"
    board = json.loads(board_p.read_text())
    rows = []
    for row in board:
        ev = eval_ckpt(row["name"])
        rec = {**row, "robust": ev}
        rows.append(rec)
        print("reeval", row["i"], row["name"], ev, flush=True)
    rows.sort(key=lambda r: r["robust"]["return_mean"], reverse=True)
    winner = rows[0]
    keys = ("learning_rate", "n_steps", "batch_size", "ent_coef", "n_epochs", "clip_range", "gae_lambda")
    kw = {k: winner[k] for k in keys}
    cons = []
    for seed in (45, 46, 47):
        name = f"tune_robust_seed{seed}"
        z = _run_dir(name) / "models" / f"ppo_{name}.zip"
        if not z.exists():
            train_one("v4a", "ppo", 100_000, 32, seed, name, ROOT / "outputs" / "tuning", ppo_kw=kw)
        ev = eval_ckpt(name)
        cons.append({"seed": seed, "name": name, **ev})
        print("consistency", seed, ev, flush=True)
    out = {
        "best": json.loads(best_p.read_text()).get("best") if best_p.exists() else winner,
        "short_eval_consistency": json.loads(best_p.read_text()).get("consistency", []) if best_p.exists() else [],
        "robust_best": {
            "i": winner["i"],
            "name": winner["name"],
            **kw,
            "return_mean": winner["robust"]["return_mean"],
            "return_std": winner["robust"]["return_std"],
            "soc_min": winner["robust"]["soc_min"],
        },
        "robust_board": [
            {
                "i": r["i"],
                "name": r["name"],
                **{k: r[k] for k in keys},
                "eval_return_mean": r["robust"]["return_mean"],
                "eval_return_std": r["robust"]["return_std"],
                "soc_min": r["robust"]["soc_min"],
                "n_ep": r["robust"]["n_ep"],
                "eval_steps": r["robust"]["n_steps"],
            }
            for r in rows
        ],
        "robust_consistency": cons,
    }
    best_p.write_text(json.dumps(out, indent=2))
    print("ROBUST BEST", out["robust_best"], flush=True)
    print("wrote", best_p)


if __name__ == "__main__":
    main()
