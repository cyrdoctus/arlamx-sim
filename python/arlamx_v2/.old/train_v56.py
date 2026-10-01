"""Train SC_v5a/v5b with tuned PPO, then SC_v6 at 100k/250k/500k/4M."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arlamx_v2.train import train_one

ROOT = Path(__file__).resolve().parents[2]

V5_JOBS = [
    ("v5a", 100_000, "SC_v5a_4x16_PPO_tuned"),
    ("v5b", 100_000, "SC_v5b_4x16_PPO_tuned"),
]
V6_JOBS = [
    ("v6", 100_000, "SC_v6_4x16_PPO_100k"),
    ("v6", 250_000, "SC_v6_4x16_PPO_250k"),
    ("v6", 500_000, "SC_v6_4x16_PPO_500k"),
    ("v6", 4_000_000, "SC_v6_4x16_PPO_4000k"),
]


def best_kw():
    p = ROOT / "outputs" / "tuning" / "tune_ppo" / "best.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    best = data.get("robust_best") or data.get("best") or {}
    keys = ("learning_rate", "n_steps", "batch_size", "ent_coef", "n_epochs", "clip_range", "gae_lambda")
    if all(k in best for k in keys):
        return {k: best[k] for k in keys}
    return {
        "learning_rate": 1e-4,
        "n_steps": 256,
        "batch_size": 128,
        "ent_coef": 0.001,
        "n_epochs": 10,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("v5", "v6", "all"), default="all")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    kw = best_kw()
    print("hparams", kw, flush=True)
    jobs = []
    if args.phase in ("v5", "all"):
        jobs.extend(V5_JOBS)
    if args.phase in ("v6", "all"):
        jobs.extend(V6_JOBS)
    done_path = ROOT / "outputs" / "training" / "sc_v56_train.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else []
    done_names = {d.get("name") for d in done}
    for var, steps, name in jobs:
        from arlamx_v2.paths import run_dir as _run_dir
        zip_p = _run_dir(name) / "models" / f"ppo_{name}.zip"
        if args.skip_done and (name in done_names or zip_p.exists()):
            print("skip done", name, flush=True)
            continue
        print("====", name, flush=True)
        rec = train_one(var, "ppo", steps, 32, 42, name, ROOT / "outputs" / "training", ppo_kw=kw)
        done.append(rec)
        done_path.write_text(json.dumps(done, indent=2))
    print("V5/V6 TRAINING DONE phase", args.phase, flush=True)


if __name__ == "__main__":
    main()
