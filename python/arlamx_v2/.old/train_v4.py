"""Sequential SC_v4a/v4b matrix matching V1.7 train_v4ab.sh.

Order: PPO 300k a/b, SAC 300k a/b, PPO 50k a/b. Seed 42, 32 envs.
Level: advanced.
"""
from __future__ import annotations

import json
from pathlib import Path

from arlamx_v2.train import train_one

ROOT = Path(__file__).resolve().parents[2]
JOBS = (
    ("v4a", "ppo", 300_000, "SC_v4a_4x16_PPO_300k"),
    ("v4b", "ppo", 300_000, "SC_v4b_4x16_PPO_300k"),
    ("v4a", "sac", 300_000, "SC_v4a_4x16_SAC_300k"),
    ("v4b", "sac", 300_000, "SC_v4b_4x16_SAC_300k"),
    ("v4a", "ppo", 50_000, "SC_v4a_4x16_PPO_50k"),
    ("v4b", "ppo", 50_000, "SC_v4b_4x16_PPO_50k"),
)


def main():
    out = ROOT / "outputs" / "training"
    summary = []
    for variant, algo, steps, name in JOBS:
        print("====", name, flush=True)
        summary.append(train_one(variant, algo, steps, 32, 42, name, out))
        (out / "sc_v4_train_summary.json").write_text(json.dumps(summary, indent=2))
    print("ALL SC_v4 TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
