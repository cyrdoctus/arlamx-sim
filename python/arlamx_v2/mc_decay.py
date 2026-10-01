"""Monte-Carlo decay study: 512 orbits from 500 km down through 300 km.

Each run: a randomized orbit (inclination, node, phase, epoch weather, seed)
started at 500 km and flown until it falls through 295 km or the step budget
runs out. The DECAY RATE is recorded as the vehicle crosses each checkpoint
altitude (500, 400, 300 km): a centred window of SMA slope around the crossing,
so every policy is measured at the same physical altitudes rather than at the
same mission times.

Policies: MPC, heuristic, and the best v10 models (from the ledger). All on the
identical v10-class plant. Fully process-parallel — each worker owns its env.

    PYTHONPATH=python python -m arlamx_v2.mc_decay --runs 512 --policies mpc,heuristic,v10:1,v10:2,v10:3
Outputs: outputs/v8/data/mc_decay.json (+ per-policy summary printed).
Level: advanced.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

V8 = OUTPUTS / "v8"
CHECKPOINTS_KM = (500.0, 400.0, 300.0)
WINDOW_STEPS = 12                 # +-1 h of SMA slope around each crossing
FLOOR_KM = 295.0
MAX_STEPS = 8000                  # ~28 days of 300 s steps — v1 default
# 20000 ≈ 69 days: enough for a 5.4 km/d 500→300 km crossing (the v1 MPC
# only reached 300 km on 252/512 orbits because 8000 steps ran out).


def _draw_options(rng):
    """One Monte-Carlo orbit: geometry and epoch weather randomized; the
    in-episode storm schedule then evolves it (v10 plant)."""
    return dict(
        altitude_km=500.0,
        inc_deg=float(rng.uniform(20.0, 40.0)),
        ecc=float(rng.uniform(0.0, 0.01)),
        raan_deg=float(rng.uniform(0.0, 360.0)),
        argp_deg=float(rng.uniform(0.0, 360.0)),
        nu_deg=float(rng.uniform(0.0, 360.0)),
        f107=float(rng.uniform(65.0, 250.0)),
        ap=float(rng.uniform(2.0, 80.0)),
        soc=0.5,
        omega_dps=[0.5, 0.5, 0.2],
    )


def _make_policy(spec):
    """spec: 'mpc' | 'mpc:<config name>' | 'mpcjson:<path>' | 'heuristic' |
    'run:<ledger name>'. Imported lazily so the worker builds its own callable.
    Bare 'mpc' is the historical v1 (truth state, DEFAULT_W) so existing MC
    numbers stay reproducible."""
    from arlamx_v2 import bench_v8 as B
    if spec == "mpc":
        return B.mpc_fn()
    if spec.startswith("mpcjson:"):
        cfg = json.loads(open(spec[8:]).read())
        return B.mpc_fn(cfg=cfg, noisy=bool(cfg.get("sensors", {}).get("enabled", True)))
    if spec.startswith("mpc:"):
        from arlamx_v2.config import load as load_cfg
        cfg = load_cfg(spec[4:], use_cache=False)
        noisy = bool((cfg.get("sensors") or {}).get("enabled", True))
        return B.mpc_fn(cfg=cfg, noisy=noisy)
    if spec == "heuristic":
        return B.heuristic_fn()
    assert spec.startswith("run:")
    name = spec[4:]
    algo = name.split("_")[2] if name.startswith("v10_") else name.split("_")[1]
    model = B.load_model(algo, name)
    return B.sb3_fn(model)


def _one_run(args):
    spec, variant, run_idx, max_steps, floor_km = args
    import warnings
    warnings.filterwarnings("ignore")
    from arlamx_v2.env import ArlamxV2Env
    rng = np.random.default_rng(10_000 + run_idx)
    opts = _draw_options(rng)
    env = ArlamxV2Env(seed=run_idx, variant=variant)
    env._max_steps = int(max_steps)
    predict = _make_policy(spec)
    obs, _ = env.reset(seed=run_idx, options=opts)
    sma, alt, t_h = [], [], []
    slews, mtqJ = 0, 0.0
    done, k = False, 0
    while not done and k < max_steps:
        a = predict(obs, env)
        obs, _r, term, trunc, info = env.step(np.asarray(a, np.float32))
        k += 1
        sma.append(float(info["sma_m"]))
        alt.append(float(info["altitude_km"]))
        t_h.append(k * env._advisor_s / 3600.0)
        if float(info.get("cmd_angle_deg", 0.0)) > 2.0:
            slews += 1
        mtqJ += float(info.get("mtq_power_W", 0.0)) * env._advisor_s
        done = term or (alt[-1] < floor_km)
    env.close()

    sma = np.asarray(sma)
    alt = np.asarray(alt)
    t_d = np.asarray(t_h) / 24.0
    days = float(t_d[-1]) if len(t_d) else 0.0
    out = {"idx": run_idx, "steps": k,
           "reached_floor": bool(alt[-1] < floor_km + 5),
           "slews_per_day": float(slews / max(days, 1e-9)),
           "mtq_energy_J": float(mtqJ),
           "mtq_energy_J_d": float(mtqJ / max(days, 1e-9))}
    for cp in CHECKPOINTS_KM:
        below = np.where(alt <= cp)[0]
        if len(below) == 0:
            out[f"decay_{int(cp)}"] = None
            continue
        i = int(below[0])
        lo, hi = max(0, i - WINDOW_STEPS), min(len(sma) - 1, i + WINDOW_STEPS)
        if hi <= lo:
            out[f"decay_{int(cp)}"] = None
            continue
        rate = (sma[lo] - sma[hi]) / 1e3 / max(t_d[hi] - t_d[lo], 1e-9)
        out[f"decay_{int(cp)}"] = float(rate)
    out["days_500_to_300"] = (float(t_d[int(np.where(alt <= 300.0)[0][0])])
                              if np.any(alt <= 300.0) else None)
    return spec, out


def run_study(policies, n_runs, variant="v10", workers=None,
              max_steps=None, floor_km=None, outfile=None):
    workers = workers or max(2, (os.cpu_count() or 8) - 2)
    max_steps = int(max_steps or MAX_STEPS)
    floor_km = float(FLOOR_KM if floor_km is None else floor_km)
    jobs = [(spec, variant, i, max_steps, floor_km)
            for spec in policies for i in range(n_runs)]
    print(f"[mc] {len(jobs)} runs on {workers} workers "
          f"(max_steps={max_steps}, floor={floor_km} km)", flush=True)
    t0 = time.time()
    results = {spec: [] for spec in policies}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for spec, out in ex.map(_one_run, jobs, chunksize=4):
            results[spec].append(out)
            n_done = sum(len(v) for v in results.values())
            if n_done % 64 == 0:
                print(f"[mc] {n_done}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    summary = {}
    for spec, rows in results.items():
        s = {"n": len(rows),
             "reached_floor": int(sum(r["reached_floor"] for r in rows))}
        for cp in CHECKPOINTS_KM:
            vals = [r[f"decay_{int(cp)}"] for r in rows
                    if r[f"decay_{int(cp)}"] is not None]
            s[f"decay_{int(cp)}_mean"] = float(np.mean(vals)) if vals else None
            s[f"decay_{int(cp)}_std"] = float(np.std(vals)) if vals else None
            s[f"decay_{int(cp)}_n"] = len(vals)
        days = [r["days_500_to_300"] for r in rows if r["days_500_to_300"]]
        s["days_500_to_300_mean"] = float(np.mean(days)) if days else None
        slews = [r.get("slews_per_day") for r in rows if r.get("slews_per_day") is not None]
        mtqd = [r.get("mtq_energy_J_d") for r in rows if r.get("mtq_energy_J_d") is not None]
        s["slews_per_day_mean"] = float(np.mean(slews)) if slews else None
        s["mtq_energy_J_d_mean"] = float(np.mean(mtqd)) if mtqd else None
        summary[spec] = s
    out = {"summary": summary, "runs": results,
           "config": {"n_runs": n_runs, "checkpoints_km": CHECKPOINTS_KM,
                      "window_steps": WINDOW_STEPS, "variant": variant,
                      "max_steps": max_steps, "floor_km": floor_km},
           "wall_s": round(time.time() - t0, 1)}
    dest = Path(outfile) if outfile else (V8 / "data" / "mc_decay.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"[mc] wrote {dest}", flush=True)
    print(f"\n[mc] done in {out['wall_s']}s")
    print(f"{'policy':34s} {'dec@500':>9s} {'dec@400':>9s} {'dec@300':>9s} {'days 500->300':>14s}")
    for spec, s in summary.items():
        f = lambda k: (f"{s[k+'_mean']:.2f}±{s[k+'_std']:.2f}"
                       if s[k + "_mean"] is not None else "-")
        d = (f"{s['days_500_to_300_mean']:.1f}"
             if s["days_500_to_300_mean"] else "-")
        print(f"{spec:34s} {f('decay_500'):>9s} {f('decay_400'):>9s} "
              f"{f('decay_300'):>9s} {d:>14s}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=512)
    ap.add_argument("--policies", type=str, required=True,
                    help="comma list: mpc,heuristic,run:<name>,... or v10:K for "
                         "the K-th best v10 row")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="advisor-step ceiling (default 8000 ≈ 28 d)")
    ap.add_argument("--floor-km", type=float, default=FLOOR_KM)
    ap.add_argument("--outfile", type=str, default="")
    a = ap.parse_args()
    specs = []
    import csv
    rows = sorted((r for r in csv.DictReader(open(V8 / "ledger.csv"))
                   if r["variant"] == "v10"),
                  key=lambda r: float(r["gap_index"]))
    for tok in a.policies.split(","):
        if tok.startswith("v10:"):
            specs.append("run:" + rows[int(tok[4:]) - 1]["name"])
        else:
            specs.append(tok)
    print("[mc] policies:", specs, flush=True)
    run_study(specs, a.runs, workers=a.workers or None,
              max_steps=a.max_steps or None, floor_km=a.floor_km,
              outfile=a.outfile or None)


if __name__ == "__main__":
    main()
