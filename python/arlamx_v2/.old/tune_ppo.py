"""PPO burn-in: 32 hyperparameter trials at 100k, then consistency reruns.

Same v4a plant/reward. Level: advanced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.train import train_one

ROOT = Path(__file__).resolve().parents[2]

TRIALS = [
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 1e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 2e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 1e-3, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 64, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 128, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 32, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 128, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 256, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.0001, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.001, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.01, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 5, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 20, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.1, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.3, "gae_lambda": 0.95},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.90},
    {"learning_rate": 3e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.003, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.98},
    {"learning_rate": 1e-4, "n_steps": 256, "batch_size": 128, "ent_coef": 0.001, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 2e-4, "n_steps": 256, "batch_size": 64, "ent_coef": 0.001, "n_epochs": 15, "clip_range": 0.15, "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "n_steps": 64, "batch_size": 64, "ent_coef": 0.01, "n_epochs": 5, "clip_range": 0.2, "gae_lambda": 0.92},
    {"learning_rate": 1e-4, "n_steps": 512, "batch_size": 256, "ent_coef": 0.0005, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.97},
    {"learning_rate": 3e-4, "n_steps": 256, "batch_size": 128, "ent_coef": 0.003, "n_epochs": 8, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 2e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.002, "n_epochs": 12, "clip_range": 0.25, "gae_lambda": 0.94},
    {"learning_rate": 4e-4, "n_steps": 192, "batch_size": 96, "ent_coef": 0.004, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 1.5e-4, "n_steps": 256, "batch_size": 128, "ent_coef": 0.001, "n_epochs": 10, "clip_range": 0.18, "gae_lambda": 0.96},
    {"learning_rate": 3e-4, "n_steps": 96, "batch_size": 48, "ent_coef": 0.005, "n_epochs": 8, "clip_range": 0.22, "gae_lambda": 0.93},
    {"learning_rate": 2.5e-4, "n_steps": 160, "batch_size": 80, "ent_coef": 0.002, "n_epochs": 10, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 8e-5, "n_steps": 256, "batch_size": 64, "ent_coef": 0.001, "n_epochs": 15, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 6e-4, "n_steps": 128, "batch_size": 64, "ent_coef": 0.008, "n_epochs": 6, "clip_range": 0.25, "gae_lambda": 0.90},
]


def eval_ckpt(name, n_ep=1):
    from stable_baselines3 import PPO

    from arlamx_v2.paths import run_dir as _run_dir
    z = _run_dir(name) / "models" / f"ppo_{name}.zip"
    if not z.exists():
        return {"return_mean": -1e9, "soc_min": 0.0}
    model = PPO.load(str(z))
    env = ArlamxV2Env(seed=99, variant="v4a")
    rets, socs = [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=200 + ep)
        ret = 0.0
        sm = 1.0
        for _ in range(16):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ret += float(r)
            sm = min(sm, float(info["battery_soc"]))
            if term or trunc:
                break
        rets.append(ret)
        socs.append(sm)
    env.close()
    return {"return_mean": float(np.mean(rets)), "soc_min": float(np.min(socs))}


def main():
    out = ROOT / "outputs" / "tuning" / "tune_ppo"
    out.mkdir(parents=True, exist_ok=True)
    board = []
    if (out / "board.json").exists():
        board = json.loads((out / "board.json").read_text())
    done = {int(r["i"]) for r in board}
    for i, kw in enumerate(TRIALS):
        if i in done:
            print("skip done", i, flush=True)
            continue
        name = f"tune_v4a_{i:02d}"
        print("trial", i, kw, flush=True)
        train_one("v4a", "ppo", 100_000, 32, 42, name, ROOT / "outputs" / "tuning", ppo_kw=kw)
        ev = eval_ckpt(name)
        row = {"i": i, "name": name, **kw, **ev}
        board.append(row)
        (out / "board.json").write_text(json.dumps(board, indent=2))
        print("  eval", ev, flush=True)
    board.sort(key=lambda r: r["return_mean"], reverse=True)
    best = board[0]
    print("BEST", best, flush=True)
    cons = []
    for seed in (42, 43, 44):
        name = f"tune_best_seed{seed}"
        train_one("v4a", "ppo", 100_000, 32, seed, name, ROOT / "outputs" / "tuning", ppo_kw={
            k: best[k] for k in ("learning_rate", "n_steps", "batch_size", "ent_coef", "n_epochs", "clip_range", "gae_lambda")
        })
        ev = eval_ckpt(name)
        cons.append({"seed": seed, **ev})
        print("consistency", seed, ev, flush=True)
    (out / "best.json").write_text(json.dumps({"best": best, "consistency": cons}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
