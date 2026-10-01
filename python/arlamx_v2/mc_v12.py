"""Modest Monte-Carlo lifetime for SC_v12 vs MPC_v2. Wrapper only.

    PYTHONPATH=python python -m arlamx_v2.mc_v12 --runs 24 --workers 12
Writes outputs/v12/traces/mc_lifetime.json. Level: advanced.
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

V12 = OUTPUTS / "v12"
BEST = (V12 / "runs" / "s2_cmp_slew0.5_4x18_300k_s43" / "ckpts" / "step_300096")
FLOOR_KM = 295.0
MAX_STEPS = 8000          # ~28 days of 300 s steps
CHECKPOINTS_KM = (500.0, 400.0, 300.0)
WINDOW = 12


def _draw(rng):
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


def _one(args):
    spec, idx, max_steps, floor_km, clip = args
    import warnings
    warnings.filterwarnings("ignore")
    from arlamx_v2.env import ArlamxV2Env
    from arlamx_v2.plot_v12 import load_mpc_predict, load_v12_predict
    from arlamx_v2.train_v12 import V12Wrapper, VARIANT

    rng = np.random.default_rng(10_000 + idx)
    opts = _draw(rng)
    base = ArlamxV2Env(seed=idx, variant=VARIANT)
    if spec == "mpc":
        predict = load_mpc_predict()
        env = V12Wrapper(base, w_slew=0.0, use_cmp=False, clip_wrong_way=False,
                         use_pwr_est=True)
    else:
        predict = load_v12_predict(Path(spec))
        env = V12Wrapper(base, w_slew=0.5, w_env=0.1, use_cmp=True,
                         clip_wrong_way=bool(clip), use_pwr_est=True)
    env.env._max_steps = int(max_steps)
    obs, _ = env.reset(seed=idx, options=opts)
    sma, alt, t_h = [], [], []
    done, k = False, 0
    while not done and k < max_steps:
        a = predict(obs, env.env)
        obs, _r, term, trunc, info = env.step(np.asarray(a, np.float32))
        k += 1
        sma.append(float(info["sma_m"]))
        alt.append(float(info["altitude_km"]))
        t_h.append(k * env.env._advisor_s / 3600.0)
        done = term or (alt[-1] < floor_km)
    env.close()
    sma = np.asarray(sma)
    alt = np.asarray(alt)
    t_d = np.asarray(t_h) / 24.0
    days = float(t_d[-1]) if len(t_d) else 0.0
    out = {"idx": idx, "steps": k, "days": days,
           "reached_floor": bool(alt[-1] < floor_km + 5)}
    for cp in CHECKPOINTS_KM:
        below = np.where(alt <= cp)[0]
        if len(below) == 0:
            out[f"decay_{int(cp)}"] = None
            continue
        i = int(below[0])
        lo, hi = max(0, i - WINDOW), min(len(sma) - 1, i + WINDOW)
        if hi <= lo:
            out[f"decay_{int(cp)}"] = None
            continue
        out[f"decay_{int(cp)}"] = float(
            (sma[lo] - sma[hi]) / 1e3 / max(t_d[hi] - t_d[lo], 1e-9))
    out["days_500_to_300"] = (
        float(t_d[int(np.where(alt <= 300.0)[0][0])])
        if np.any(alt <= 300.0) else None)
    return spec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 4))
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--ckpt", default=str(BEST))
    ap.add_argument("--label", default="sc_v12")
    ap.add_argument("--out", default=None,
                    help="json path; default outputs/v12/traces/mc_lifetime.json")
    a = ap.parse_args()
    jobs = ([("mpc", i, MAX_STEPS, FLOOR_KM, False) for i in range(a.runs)]
            + [(a.ckpt, i, MAX_STEPS, FLOOR_KM, a.clip) for i in range(a.runs)])
    print(f"[mc_v12] {len(jobs)} jobs, {a.workers} workers, ckpt={a.ckpt}", flush=True)
    t0 = time.time()
    results = {"mpc": [], a.ckpt: []}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        n_done = 0
        for spec, out in ex.map(_one, jobs, chunksize=1):
            results[spec].append(out)
            n_done += 1
            if n_done % 8 == 0:
                print(f"[mc_v12] {n_done}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    summary = {}
    for spec, rows in results.items():
        days = [r["days_500_to_300"] for r in rows if r["days_500_to_300"]]
        lab = "mpc" if spec == "mpc" else a.label
        s = {
            "n": len(rows),
            "reached_floor": int(sum(r["reached_floor"] for r in rows)),
            "days_500_to_300_mean": float(np.mean(days)) if days else None,
            "days_500_to_300_median": float(np.median(days)) if days else None,
            "days_n": len(days),
        }
        for cp in CHECKPOINTS_KM:
            vals = [r[f"decay_{int(cp)}"] for r in rows
                    if r.get(f"decay_{int(cp)}") is not None]
            s[f"decay_{int(cp)}_mean"] = float(np.mean(vals)) if vals else None
            s[f"decay_{int(cp)}_n"] = len(vals)
        summary[lab] = s
        print(f"  {lab:8s}  days500→300 mean={s['days_500_to_300_mean']}  "
              f"n={s['days_n']}/{s['n']}", flush=True)
    dest = Path(a.out) if a.out else (V12 / "traces" / "mc_lifetime.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"summary": summary, "wall_s": round(time.time() - t0, 1),
         "runs": a.runs, "ckpt": a.ckpt, "clip": a.clip},
        indent=2, default=str))
    print(f"[mc_v12] wrote {dest} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
