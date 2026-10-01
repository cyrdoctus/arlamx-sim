"""OOD analysis, PPO tune (32x100k), v5/v6 train, then OOD again."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main():
    print("==== PHASE OOD v4", flush=True)
    sys.argv = ["analyze_ood", "--steps", "12", "--out", str(ROOT / "outputs" / "analysis" / "ood")]
    runpy.run_module("arlamx_v2.analyze_ood", run_name="__main__")
    print("==== PHASE TUNE", flush=True)
    sys.argv = ["tune_ppo"]
    runpy.run_module("arlamx_v2.tune_ppo", run_name="__main__")
    print("==== PHASE V5 V6 TRAIN", flush=True)
    sys.argv = ["train_v56"]
    runpy.run_module("arlamx_v2.train_v56", run_name="__main__")
    print("==== PHASE OOD v5v6", flush=True)
    sys.argv = ["analyze_ood", "--steps", "12", "--out", str(ROOT / "outputs" / "analysis" / "ood_v56")]
    runpy.run_module("arlamx_v2.analyze_ood", run_name="__main__")
    print("PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
