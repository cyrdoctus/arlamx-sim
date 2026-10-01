"""SC_v10 burn-in tuner — iterate the reward toward the MPC, with seeds.

One round = a set of named override candidates, each trained SEEDS x BURN_STEPS
on the reference policy class, then scored on the fixed probe suite. Results
append to outputs/v8/v10_tune.csv so rounds accumulate into one ledger.

The targets (from the v10 brief):
    decay_nominal <= 1.10 x MPC          "within 10 % of the MPC"
    brownouts == 0 and band not collapsing   "not lose too much power"
    downlink as high as possible             "best ground coverage"

Usage:
    PYTHONPATH=python python -m arlamx_v2.v10_tune --round r0
Rounds are code, not config: each round's candidate list is explicit below so
the tuning history is readable in git, and the chosen adjustment is written
back into reward_v10.yaml BY HAND between rounds (with a [TUNE-n] tag).
Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from arlamx_v2.bench_v8 import (DATA, RUNS, gap_index, probe_suite, sb3_fn,
                                stage_ref)
from arlamx_v2.train import train_one

SEEDS = (42, 43, 44)
BURN_STEPS = 40_000

FIELDS = ["round", "cand", "seed", "gap", "decay_nominal", "decay_ratio_mpc",
          "band_pct", "brownouts", "downlink_min_d", "deleg_B", "deleg_P",
          "kf_r", "wall_s"]

# ---------------------------------------------------------------------------
# Rounds. Each candidate is a reward_w override merged onto reward_v10.yaml.
# ---------------------------------------------------------------------------

ROUNDS = {
    # r0: does the TUNE-0 shaping already move toward the MPC, and which KF
    # sigma_drive tracks storms best? (kf.sigma_drive rides in the override so
    # it lands in each run's snapshot.)
    "r0": {
        "base": {},
        "dE_hi": {"dE_weight": 4.0},
        "dE_lo": {"dE_weight": 2.6},
        "kf_slow": {"kf": {"sigma_drive": 1.0e-10}},
        "kf_fast": {"kf": {"sigma_drive": 1.0e-9}},
    },
    # r1: dE 2.6 is folded into the YAML ([TUNE-1]). r0 also showed
    # kf_fast (sigma_drive 1e-9) with the best torque tracking (r 0.773) and a
    # striking downlink gain (25.8 min/d) but a collapsed band — test whether
    # power-side terms can keep the band while keeping that KF.
    "r1": {
        "kf9": {"kf": {"sigma_drive": 1.0e-9}},
        "kf9_band": {"kf": {"sigma_drive": 1.0e-9},
                     "band_weight": 2.0},
        "kf9_drop": {"kf": {"sigma_drive": 1.0e-9},
                     "stability": {"power_drop_weight": 4.0}},
        "kf9_soft": {"kf": {"sigma_drive": 1.0e-9},
                     "smooth_weight": 0.2, "thrift_weight": 2.0},
    },
    # r2: r1 killed the fast KF (every kf9 combo lost to r0's dE_lo on the
    # default 3e-10, and band_weight 2.0 caused 12 brownouts). Final polish on
    # the current YAML base: can power_drop / softness / coverage push the gap
    # below dE_lo's 0.921 without breaking the band?
    "r2": {
        "confirm": {},
        "drop4": {"stability": {"power_drop_weight": 4.0}},
        "soft": {"smooth_weight": 0.2, "thrift_weight": 2.0},
        "gs28": {"gs_weight": 2.8},
    },
    # ---- v11 rounds (run with --variant v11): adaptability weights --------
    # base = frozen v10 reward + TUNE-0 adapt (2.0/2.0) + damage prob 0.6
    "a0": {
        "base": {},
        "adapt_hi": {"adapt": {"w_consistency": 4.0, "w_deleg_hold": 4.0}},
        "adapt_lo": {"adapt": {"w_consistency": 1.0, "w_deleg_hold": 1.0}},
        "dmg_hi": {"damage": {"prob": 0.85}},
    },
    "a1": {},
}


def run_candidate(rnd, name, override, variant, algo, arch, tune_csv):
    ref = stage_ref()["MPC"]["agg"]
    for seed in SEEDS:
        run = f"{variant}tune_{rnd}_{name}_s{seed}"
        if (RUNS / run / "probe_metrics.json").exists():
            print(f"[skip] {run}", flush=True)
            continue
        t0 = time.time()
        train_one(variant, algo, BURN_STEPS, 32, seed, run, RUNS,
                  env_kw={"reward_w": override} if override else None,
                  arch=arch)
        from arlamx_v2.bench_v8 import load_model
        model = load_model(algo, run)
        agg, per = probe_suite(sb3_fn(model), variant)
        (RUNS / run / "probe_metrics.json").write_text(
            json.dumps({"agg": agg, "per_probe": per}, indent=2))
        row = {"round": rnd, "cand": name, "seed": seed,
               "gap": round(gap_index(agg, ref), 4),
               "decay_nominal": round(agg["decay_nominal"], 3),
               "decay_ratio_mpc": round(agg["decay_nominal"] / ref["decay_nominal"], 3),
               "band_pct": round(agg["band_pct"], 2),
               "brownouts": int(agg["brownouts"]),
               "downlink_min_d": round(agg["downlink_min_d"], 2),
               "deleg_B": round(agg["deleg_benefit"], 4),
               "deleg_P": round(agg["deleg_P"], 4),
               "kf_r": round(agg["kf_r"], 4),
               "wall_s": round(time.time() - t0, 1)}
        new = not tune_csv.exists()
        with open(tune_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[tune] {run}: gap={row['gap']} decay_ratio={row['decay_ratio_mpc']} "
              f"band={row['band_pct']} B={row['deleg_B']}", flush=True)


def summarize(rnd, tune_csv):
    if not tune_csv.exists():
        return
    rows = [r for r in csv.DictReader(open(tune_csv)) if r["round"] == rnd]
    if not rows:
        return
    print(f"\n[tune] round {rnd} — median over {len(SEEDS)} seeds "
          f"(target: decay_ratio <= 1.10, brownouts 0)")
    print(f"{'cand':10s} {'gap':>7s} {'decay_ratio':>12s} {'band %':>7s} "
          f"{'brn':>4s} {'dl':>6s} {'B':>7s} {'P':>7s} {'kf_r':>6s}")
    by = {}
    for r in rows:
        by.setdefault(r["cand"], []).append(r)
    for cand, rs in by.items():
        def m(k):
            return float(np.median([float(x[k]) for x in rs]))
        print(f"{cand:10s} {m('gap'):7.3f} {m('decay_ratio_mpc'):12.3f} "
              f"{m('band_pct'):7.1f} {m('brownouts'):4.0f} "
              f"{m('downlink_min_d'):6.2f} {m('deleg_B'):7.3f} "
              f"{m('deleg_P'):7.3f} {m('kf_r'):6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, choices=tuple(ROUNDS))
    ap.add_argument("--variant", default="v10")
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--arch", default="4x18",
                    help="the v10 winner's class by default")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    layers, width = (int(x) for x in a.arch.lower().split("x"))
    arch = [width] * layers
    tune_csv = DATA.parent / f"{a.variant}_tune.csv"
    if not a.summary_only:
        cands = ROUNDS[a.round]
        if not cands:
            raise SystemExit(f"round {a.round} has no candidates defined yet")
        for name, override in cands.items():
            run_candidate(a.round, name, override, a.variant, a.algo, arch,
                          tune_csv)
    summarize(a.round, tune_csv)


if __name__ == "__main__":
    main()
