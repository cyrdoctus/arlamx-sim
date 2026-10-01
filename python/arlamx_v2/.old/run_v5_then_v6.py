"""Train SC_v5, OOD-validate, then SC_v6 at 100k/250k/500k/4M, then OOD again."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main():
    print("==== PHASE V5 TRAIN", flush=True)
    sys.argv = ["train_v56", "--phase", "v5", "--skip-done"]
    runpy.run_module("arlamx_v2.train_v56", run_name="__main__")
    print("==== PHASE OOD v5", flush=True)
    sys.argv = [
        "analyze_ood",
        "--steps",
        "48",
        "--recover-steps",
        "80",
        "--out",
        str(ROOT / "outputs" / "analysis" / "ood_v5"),
    ]
    runpy.run_module("arlamx_v2.analyze_ood", run_name="__main__")
    print("==== PHASE V6 TRAIN", flush=True)
    sys.argv = ["train_v56", "--phase", "v6", "--skip-done"]
    runpy.run_module("arlamx_v2.train_v56", run_name="__main__")
    print("==== PHASE OOD v5v6", flush=True)
    sys.argv = [
        "analyze_ood",
        "--steps",
        "48",
        "--recover-steps",
        "80",
        "--out",
        str(ROOT / "outputs" / "analysis" / "ood_v56"),
    ]
    runpy.run_module("arlamx_v2.analyze_ood", run_name="__main__")
    print("V5 THEN V6 PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
