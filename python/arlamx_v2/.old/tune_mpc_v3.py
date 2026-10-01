"""MPC_v3 continuation sweep. Writes ONLY under outputs/mpc/ (not outputs/v8).

Does not change the plant, SamplingMpcPolicy physics, or v2 freeze files.
Reuses bench_v8.mpc_fn / _rollout / probe_suite and mc_decay.run_study.

    PYTHONPATH=python python -m arlamx_v2.tune_mpc_v3 --stage all --workers 44

Wall clock hard cap: 3 hours, then freeze + plots with whatever is done.
Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from arlamx_v2.advisors.mpc import SamplingMpcPolicy
from arlamx_v2.bench_v8 import mpc_fn, probe_suite
from arlamx_v2.config import CONFIG_DIR
from arlamx_v2.inference_budget import mpc_u575_ms
from arlamx_v2.paths import OUTPUTS
from arlamx_v2.tune_mpc import (
    BUDGET_MS,
    DECAY_FLOOR,
    N_ORBITS,
    PUBLISHED_V1,
    SEEDS,
    SLEW_FLOOR,
    TUNE_PROBES,
    VARIANT,
    _mk_env,
    score_vs,
)

OUT = OUTPUTS / "mpc"
V2_SRC = OUTPUTS / "v8"
RUNS = OUT / "runs"
LEDGER = OUT / "ledger.csv"
PLOTS = OUT / "plots"
STATE = OUT / "state.json"
WALL_S = 3 * 3600
T0 = None  # set in main

V2_ANCHOR = {
    "decay_km_d": 17.807317585064425,
    "band_pct": 86.5036231884058,
    "downlink_min_d": 18.260869565217394,
    "slews_per_day": 166.43478260869566,
    "brownouts": 0.0,
}
V1_NOISY = {
    "decay_km_d": 18.11527618290063,
    "band_pct": 85.41666666666667,
    "downlink_min_d": 15.652173913043478,
    "slews_per_day": 188.6086956521739,
    "brownouts": 0.0,
}

LEDGER_FIELDS = [
    "id", "round", "horizon_steps", "n_random", "cd", "slew", "gs", "gs_align",
    "power_low_pen", "soc_k", "power_band", "gs_pen", "max_cmd_deg",
    "decay_km_d", "band_pct", "downlink_min_d", "brownouts", "slews_per_day",
    "mtq_energy_J_d", "pei", "score_v2", "score_v1", "u575_ms", "u575_ms_actual",
    "in_budget", "wall_s", "disqualified",
]


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def remaining():
    return WALL_S - (time.time() - T0)


def v2_cfg():
    return SamplingMpcPolicy.from_config(yaml.safe_load(
        (CONFIG_DIR / "mpc_v2.yaml").read_text())).to_config()


def _set(cfg, key, value):
    cfg = deepcopy(cfg)
    if key in ("horizon_steps", "n_random"):
        cfg[key] = int(value)
    elif key in ("soc_k", "max_cmd_deg", "soc_lo", "soc_hi"):
        cfg[key] = float(value)
    else:
        cfg.setdefault("weights", {})[key] = float(value)
    return cfg


def _get(cfg, key):
    if key in ("horizon_steps", "n_random", "soc_k", "max_cmd_deg", "soc_lo", "soc_hi"):
        return cfg[key]
    return cfg["weights"][key]


def cfg_id(rnd, cfg):
    w = cfg["weights"]
    return (f"{rnd}_H{int(cfg['horizon_steps'])}_R{int(cfg['n_random'])}"
            f"_cd{w['cd']:g}_sl{w['slew']:g}_gs{w['gs']:g}_ga{w['gs_align']:g}"
            f"_pl{w['power_low_pen']:g}_k{cfg['soc_k']:g}"
            f"_pb{w['power_band']:g}_gp{w['gs_pen']:g}"
            f"_mc{cfg.get('max_cmd_deg', 40):g}")


def cost_ms(cfg, scale="published"):
    return mpc_u575_ms(int(cfg["horizon_steps"]), int(cfg["n_random"]),
                       scale=scale)


def _eval_one(job):
    import warnings
    warnings.filterwarnings("ignore")
    from arlamx_v2.bench_v8 import _rollout, mpc_fn as _mpc

    cid, cfg = job["id"], job["cfg"]
    dest = Path(job["dest"])
    if dest.exists():
        return json.loads(dest.read_text())
    t0 = time.time()
    predict = _mpc(cfg=cfg, noisy=job.get("noisy", True), seed=0)
    rows = []
    for seed in SEEDS:
        for name, opt, jumps in TUNE_PROBES:
            env = _mk_env(opt, job.get("n_orbits", N_ORBITS))
            m = _rollout(env, predict, name, seed, opt=opt, jumps=jumps)
            m["seed"] = int(seed)
            m["probe"] = name
            rows.append(m)
            env.close()
    keys = ("decay_km_d", "band_pct", "downlink_min_d", "slews_per_day",
            "mtq_energy_J_d", "pei", "mtq_energy_J")
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    agg["brownouts"] = float(np.sum([r["brownouts"] for r in rows]))
    out = {
        "id": cid, "round": job["round"], "cfg": cfg,
        "agg": agg, "per": rows,
        "u575_ms": cost_ms(cfg, "published"),
        "u575_ms_actual": cost_ms(cfg, "actual"),
        "in_budget": bool(cost_ms(cfg, "published") <= BUDGET_MS),
        "wall_s": round(time.time() - t0, 1),
        "when": _now(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    return out


def _row(r):
    cfg, agg, w = r["cfg"], r["agg"], r["cfg"]["weights"]
    sv2 = score_vs(agg, V2_ANCHOR)
    sv1 = score_vs(agg, V1_NOISY)
    return {
        "id": r["id"], "round": r["round"],
        "horizon_steps": cfg["horizon_steps"], "n_random": cfg["n_random"],
        "cd": w["cd"], "slew": w["slew"], "gs": w["gs"], "gs_align": w["gs_align"],
        "power_low_pen": w["power_low_pen"], "soc_k": cfg["soc_k"],
        "power_band": w["power_band"], "gs_pen": w["gs_pen"],
        "max_cmd_deg": cfg.get("max_cmd_deg", 40.0),
        "decay_km_d": round(agg["decay_km_d"], 4),
        "band_pct": round(agg["band_pct"], 3),
        "downlink_min_d": round(agg["downlink_min_d"], 3),
        "brownouts": int(agg["brownouts"]),
        "slews_per_day": round(agg["slews_per_day"], 3),
        "mtq_energy_J_d": round(agg["mtq_energy_J_d"], 4),
        "pei": round(agg["pei"], 4),
        "score_v2": "" if sv2 is None else round(sv2, 4),
        "score_v1": "" if sv1 is None else round(sv1, 4),
        "u575_ms": round(r["u575_ms"], 3),
        "u575_ms_actual": round(r["u575_ms_actual"], 3),
        "in_budget": int(r["in_budget"]),
        "wall_s": r["wall_s"],
        "disqualified": int(sv2 is None),
    }


def _done():
    if not LEDGER.exists():
        return set()
    with open(LEDGER) as f:
        return {r["id"] for r in csv.DictReader(f)}


def _append(row):
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def _jobs(rnd, cfgs):
    RUNS.mkdir(parents=True, exist_ok=True)
    out = []
    seen = set()
    for c in cfgs:
        cid = cfg_id(rnd, c)
        if cid in seen:
            continue
        seen.add(cid)
        out.append({"id": cid, "round": rnd, "cfg": c, "noisy": True,
                    "n_orbits": N_ORBITS, "dest": str(RUNS / f"{cid}.json")})
    return out


def _run(jobs, workers):
    done = _done()
    pending = [j for j in jobs if j["id"] not in done]
    print(f"[v3] {len(jobs)} configs, {len(pending)} to run, workers={workers} "
          f"remain={remaining():.0f}s", flush=True)
    results = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_eval_one, j): j for j in pending}
            n = 0
            for fut in as_completed(futs):
                n += 1
                r = fut.result()
                row = _row(r)
                if r["id"] not in done:
                    _append(row)
                    done.add(r["id"])
                sc = row["score_v2"] if row["score_v2"] != "" else "DQ"
                print(f"[v3] {n}/{len(pending)} {r['id']} score_v2={sc} "
                      f"dec={r['agg']['decay_km_d']:.2f} band={r['agg']['band_pct']:.1f} "
                      f"dl={r['agg']['downlink_min_d']:.1f} "
                      f"ms={r['u575_ms']:.1f} {r['wall_s']}s", flush=True)
                results.append(r)
    for j in jobs:
        p = Path(j["dest"])
        if p.exists() and j["id"] not in {x["id"] for x in results}:
            results.append(json.loads(p.read_text()))
    for r in results:
        r["score_v2"] = score_vs(r["agg"], V2_ANCHOR)
        r["score_v1"] = score_vs(r["agg"], V1_NOISY)
    return results


def _pick(results):
    ok = [r for r in results
          if r.get("score_v2") is not None and r.get("in_budget", True)]
    if not ok:
        ok = [r for r in results if r.get("score_v2") is not None]
    if not ok:
        return None
    return max(ok, key=lambda r: r["score_v2"])


def _print_round(rnd, results, n=12):
    rows = sorted(results, key=lambda r: (-(r.get("score_v2") or -1e9)))
    print(f"\n[v3] round {rnd} top {min(n, len(rows))}/{len(rows)}", flush=True)
    print(f"{'id':64s} {'s_v2':>6s} {'dec':>7s} {'band':>6s} {'dl':>6s} "
          f"{'slew':>7s} {'ms':>5s}", flush=True)
    for r in rows[:n]:
        sc = r.get("score_v2")
        print(f"{r['id']:64s} {('-' if sc is None else f'{sc:.3f}'):>6s} "
              f"{r['agg']['decay_km_d']:7.2f} {r['agg']['band_pct']:6.1f} "
              f"{r['agg']['downlink_min_d']:6.1f} {r['agg']['slews_per_day']:7.1f} "
              f"{r['u575_ms']:5.1f}", flush=True)


# ---------------------------------------------------------------------------
# setup / v2 archive (read-only copy out of v8)
# ---------------------------------------------------------------------------

def stage_setup():
    OUT.mkdir(parents=True, exist_ok=True)
    v2d = OUT / "v2"
    v2d.mkdir(exist_ok=True)
    copies = {
        V2_SRC / "mpc_tune" / "freeze.json": v2d / "freeze.json",
        V2_SRC / "mpc_tune" / "ledger.csv": v2d / "ledger.csv",
        CONFIG_DIR / "mpc_v2.yaml": v2d / "mpc_v2.yaml",
        V2_SRC / "data" / "ref_v2.json": v2d / "ref_v2.json",
        CONFIG_DIR / "mpc_v1.yaml": v2d / "mpc_v1.yaml",
    }
    for src, dst in copies.items():
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    mc_path = V2_SRC / "data" / "mc_decay_v2.json"
    if mc_path.exists() and not (v2d / "mc_summary.json").exists():
        mc = json.loads(mc_path.read_text())
        (v2d / "mc_summary.json").write_text(json.dumps(
            {"summary": mc.get("summary"), "config": mc.get("config"),
             "wall_s": mc.get("wall_s")}, indent=2))
    _write_v2_report()
    print(f"[v3] outputs -> {OUT}", flush=True)


def _write_v2_report():
    freeze = json.loads((OUT / "v2" / "freeze.json").read_text())
    t, p20, mc = freeze["table"], freeze["table"]["MPC_v2_probe20"], freeze["table"]["MPC_v2_mc"]
    v1 = t["MPC_v1"]
    p10 = t["MPC_v2_probe10"]
    text = f"""# MPC_v2 report

Frozen `{freeze['winner']}` on 2026-08-23. Sensors ON. `slew` 0.2 → 0.5 vs v1; all other knobs unchanged.

## How to read the numbers (Deep RL must match the protocol)

| suite | what it is |
|---|---|
| **probe20** | 20 orbits × probes nominal/spike/offnominal × seeds 1001,1002, v8a plant, **sensors on**. This is the v12 comparator row. |
| **probe10** | 10-orbit quiet+storm × 3 seeds — used only for the A–F sweep. Different numbers; do not mix. |
| **MC** | 512 random orbits, 500→295 km, max 20000 steps (~69 d), v10 weather envelope, sensors on. |

## Scoreboard

| metric | MPC_v1 (truth, published) | MPC_v2 probe20 (sensors) | MPC_v2 MC |
|---|---|---|---|
| decay nominal km/d | {v1['decay_nominal']} | {p20['decay_nominal']:.3f} | {mc['decay_500_mean']:.2f} @500 km |
| band % | {v1['band_pct']} | {p20['band_pct']:.2f} | — |
| downlink min/d | {v1['downlink_min_d']} | {p20['downlink_min_d']:.2f} | — |
| brownouts | 0 | {p20['brownouts']:.0f} | — |
| slews / day | — | {p20['slews_per_day']:.1f} | {mc['slews_per_day_mean']:.1f} |
| mtq energy J | {v1['mtq_energy_J']} | {p20['mtq_energy_J']:.2f} | {mc['mtq_energy_J_d_mean']:.2f} J/d |
| PEI | — | {p20['pei']:.4f} | — |
| U575 ms | 8.4 | {p20['u575_ms']:.2f} | — |
| MC lifetime d | 17.9 | — | {mc['days_500_to_300_mean']:.2f} |
| MC decay@300 | 88.2 | — | {mc['decay_300_mean']:.2f} |
| MC reached 300 km | 252/512 | — | {mc['reached_floor']}/512 |

## Sweep lessons (10-orbit, vs noisy v1)

- Worst-metric score of the winner vs noisy v1: **{p10['score']:.4f}** (decay 17.81, band 86.5, dl 18.3, slews 166).
- Horizon 6 beat 3/9/12 on the min-ratio; H=3 had the best decay (14.35) but lost band.
- `n_random` 16/32 bought downlink (27–31 min/d) and destroyed the band (73–75 %). 32 failed the 15 ms gate.
- Raising `gs` / `gs_align` did not help downlink; `soc_k` 6 vs 10 was identical.
- `slew` 0.5 is the only weight that beat v1 on all four 10-orbit metrics at once.
- Probe20 downlink 18.6 vs published 23.2 is **sensor noise**, not the slew tweak.

## Frozen config

`horizon=6 n_random=8 cd=0.5 slew=0.5 gs=2 gs_align=1 power_low_pen=4 soc_k=6 max_cmd=40°`
Sensors ON. Dipole 8-bit: not in the plant.

Deep RL matching target (probe20, sensors on): decay **≤ 9.19 km/d**, band **≥ 84.1 %**, downlink **≥ 18.6 min/d**, brownouts **0**, U575 **≤ 1.5 ms** (10× vs 6.81 ms).
"""
    (OUT / "REPORT_v2.md").write_text(text)


# ---------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------

def round_G(workers):
    """Fine grid around v2: H × cd × slew."""
    bb = v2_cfg()
    cfgs = []
    for H in (4, 5, 6, 7, 8):
        for cd in (0.40, 0.50, 0.60, 0.70):
            for sl in (0.35, 0.50, 0.65, 0.80):
                c = _set(_set(_set(bb, "horizon_steps", H), "cd", cd), "slew", sl)
                cfgs.append(c)
    res = _run(_jobs("G", cfgs), workers)
    _print_round("G", res)
    return res


def round_H(workers):
    """Horizon × n_random at v2 weights (cost vs downlink)."""
    bb = v2_cfg()
    cfgs = []
    for H in (3, 4, 6, 8):
        for R in (6, 8, 10, 12):
            cfgs.append(_set(_set(bb, "horizon_steps", H), "n_random", R))
    res = _run(_jobs("H", cfgs), workers)
    _print_round("H", res)
    return res


def round_I(workers):
    """power_band × gs_pen × gs on the v2 backbone."""
    bb = v2_cfg()
    cfgs = []
    for pb in (0.8, 1.0, 1.3, 1.6):
        for gp in (0.3, 0.5, 0.8):
            for gs in (1.5, 2.0, 2.5):
                c = _set(_set(_set(bb, "power_band", pb), "gs_pen", gp), "gs", gs)
                cfgs.append(c)
    res = _run(_jobs("I", cfgs), workers)
    _print_round("I", res)
    return res


def round_J(workers):
    """max_cmd_deg on the v2 backbone."""
    bb = v2_cfg()
    cfgs = [_set(bb, "max_cmd_deg", d) for d in (20.0, 30.0, 40.0, 50.0, 60.0)]
    res = _run(_jobs("J", cfgs), workers)
    _print_round("J", res)
    return res


def round_K(workers, prior):
    """Pareto hybrids: v2 lessons + G/I/H winners, ±1 step."""
    bb = v2_cfg()
    seeds = [
        _set(_set(_set(bb, "horizon_steps", 3), "slew", 0.65), "power_band", 1.3),
        _set(_set(_set(bb, "horizon_steps", 3), "slew", 0.80), "cd", 0.60),
        _set(_set(_set(bb, "horizon_steps", 4), "n_random", 10), "slew", 0.65),
        _set(_set(_set(bb, "horizon_steps", 5), "cd", 0.60), "slew", 0.65),
        _set(_set(bb, "n_random", 10), "slew", 0.65),
        _set(_set(_set(bb, "cd", 0.60), "slew", 0.80), "power_band", 1.3),
        _set(_set(_set(bb, "gs", 2.5), "gs_pen", 0.3), "slew", 0.65),
        _set(_set(bb, "max_cmd_deg", 30.0), "slew", 0.65),
        _set(_set(_set(bb, "horizon_steps", 8), "slew", 0.65), "power_band", 1.3),
    ]
    winners = []
    for rnd in ("G", "H", "I", "J"):
        pool = [r for r in prior if r.get("round") == rnd]
        w = _pick(pool)
        if w:
            winners.append(w["cfg"])
    combo = v2_cfg()
    for wcfg in winners:
        for k in ("horizon_steps", "n_random", "max_cmd_deg", "soc_k"):
            combo = _set(combo, k, _get(wcfg, k))
        for k in ("cd", "slew", "gs", "gs_align", "power_band", "gs_pen",
                  "power_low_pen"):
            combo = _set(combo, k, _get(wcfg, k))
    seeds.append(combo)
    # ±1 on combo for the knobs we actually grid
    grids = {
        "horizon_steps": [3, 4, 5, 6, 7, 8],
        "n_random": [6, 8, 10, 12],
        "cd": [0.40, 0.50, 0.60, 0.70],
        "slew": [0.35, 0.50, 0.65, 0.80],
        "power_band": [0.8, 1.0, 1.3, 1.6],
        "gs": [1.5, 2.0, 2.5],
        "max_cmd_deg": [20.0, 30.0, 40.0, 50.0],
    }
    for key, grid in grids.items():
        cur = float(_get(combo, key))
        idxs = [i for i, v in enumerate(grid) if abs(float(v) - cur) < 1e-9]
        if not idxs:
            continue
        i = idxs[0]
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                seeds.append(_set(combo, key, grid[j]))
    res = _run(_jobs("K", seeds), workers)
    _print_round("K", res)
    return res


def _all_runs():
    rows = []
    if not RUNS.exists():
        return rows
    for p in RUNS.glob("*.json"):
        r = json.loads(p.read_text())
        r["score_v2"] = score_vs(r["agg"], V2_ANCHOR)
        r["score_v1"] = score_vs(r["agg"], V1_NOISY)
        r.setdefault("in_budget", r.get("u575_ms", 99) <= BUDGET_MS)
        rows.append(r)
    return rows


def stage_confirm(workers):
    """20-orbit 3-probe suite on the top unique configs + v2 baseline."""
    from arlamx_v2.bench_v8 import mpc_fn as _mpc
    pool = _all_runs()
    ranked = sorted((r for r in pool if r.get("score_v2") is not None
                     and r.get("in_budget", True)),
                    key=lambda r: -r["score_v2"])
    picks, seen = [], set()
    def add(r):
        key = json.dumps(r["cfg"], sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        picks.append(r)
    for r in ranked[:8]:
        add(r)
    for r in sorted(pool, key=lambda x: x["agg"]["decay_km_d"])[:3]:
        add(r)
    for r in sorted(pool, key=lambda x: -x["agg"]["downlink_min_d"])[:3]:
        add(r)
    # always include v2
    v2 = {"id": "V2_baseline", "cfg": v2_cfg(), "round": "L"}
    add(v2)
    conf_dir = OUT / "confirm"
    conf_dir.mkdir(exist_ok=True)
    print(f"[v3] 20-orbit confirm on {len(picks)} configs", flush=True)
    out = []
    for r in picks:
        dest = conf_dir / f"{r['id']}.json"
        if dest.exists():
            out.append(json.loads(dest.read_text()))
            continue
        if remaining() < 90:
            print("[v3] time cap: stopping confirm", flush=True)
            break
        print(f"[v3] confirm {r['id']}", flush=True)
        t0 = time.time()
        fn = _mpc(cfg=r["cfg"], noisy=True, seed=0)
        agg, per = probe_suite(fn, VARIANT)
        rec = {"id": r["id"], "cfg": r["cfg"], "agg": agg, "per_probe": per,
               "u575_ms": cost_ms(r["cfg"], "published"),
               "u575_ms_actual": cost_ms(r["cfg"], "actual"),
               "wall_s": round(time.time() - t0, 1)}
        dest.write_text(json.dumps(rec, indent=2, default=str))
        print(f"   decay={agg['decay_nominal']:.2f} band={agg['band_pct']:.1f} "
              f"dl={agg['downlink_min_d']:.1f} brn={agg['brownouts']:.0f} "
              f"{rec['wall_s']}s", flush=True)
        out.append(rec)
    (OUT / "confirm_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def stage_freeze(confirms, workers):
    """Pick the best 20-orbit config (no brownouts), write mpc_v3.yaml, optional MC."""
    ok = [c for c in confirms if float(c["agg"].get("brownouts", 0)) < 0.5]
    if not ok:
        raise SystemExit("no brownout-free confirm config")
    # rank: min-ratio vs v2 probe20, using probe20 metrics
    v2p = json.loads((OUT / "v2" / "freeze.json").read_text())["table"]["MPC_v2_probe20"]
    anchor20 = {
        "decay_km_d": v2p["decay_nominal"],
        "band_pct": v2p["band_pct"],
        "downlink_min_d": v2p["downlink_min_d"],
        "slews_per_day": v2p["slews_per_day"],
        "brownouts": 0.0,
    }
    def s20(c):
        a = c["agg"]
        m = {"decay_km_d": a["decay_nominal"], "band_pct": a["band_pct"],
             "downlink_min_d": a["downlink_min_d"],
             "slews_per_day": a.get("slews_per_day", v2p["slews_per_day"]),
             "brownouts": a["brownouts"]}
        return score_vs(m, anchor20)
    for c in ok:
        c["score20"] = s20(c)
    winner = max(ok, key=lambda c: c["score20"] if c["score20"] is not None else -1e9)
    meta = {
        "id": winner["id"], "score20": winner["score20"],
        "u575_ms": winner["u575_ms"], "u575_ms_actual": winner["u575_ms_actual"],
        "date": _now(), "protocol": "20-orbit × 3 probes × 2 seeds, sensors on, v8a",
        "budget_ms": BUDGET_MS,
    }
    yaml_path = CONFIG_DIR / "mpc_v3.yaml"
    out_yaml = OUT / "mpc_v3.yaml"
    _write_yaml(yaml_path, winner["cfg"], meta)
    shutil.copy2(yaml_path, out_yaml)
    print(f"[v3] froze {winner['id']} score20={winner['score20']:.4f} -> {yaml_path}",
          flush=True)
    (OUT / "ref_v3.json").write_text(json.dumps(
        {"MPC": {"agg": winner["agg"], "per_probe": winner.get("per_probe"),
                 "cfg_id": winner["id"], "sensors": True,
                 "u575_ms": winner["u575_ms"]}}, indent=2, default=str))

    mc_sum = None
    # MC if we have ≥ 25 min left
    if remaining() >= 1500:
        from arlamx_v2.mc_decay import run_study
        pending = OUT / "mpc_v3_pending.json"
        pending.write_text(json.dumps(
            {**winner["cfg"], "sensors": {"enabled": True}}, indent=2))
        n_mc = 512 if remaining() >= 2400 else 256
        print(f"[v3] MC {n_mc}  remain={remaining():.0f}s", flush=True)
        mc = run_study([f"mpcjson:{pending}"], n_mc, variant="v10",
                       workers=workers, max_steps=20000, floor_km=295.0,
                       outfile=str(OUT / "mc_decay_v3.json"))
        mc_sum = mc["summary"].get(f"mpcjson:{pending}", {})
    else:
        print(f"[v3] skip MC (remain {remaining():.0f}s < 25 min)", flush=True)

    table = {
        "MPC_v1": PUBLISHED_V1,
        "MPC_v2_probe20": json.loads((OUT / "v2" / "freeze.json").read_text())
        ["table"]["MPC_v2_probe20"],
        "MPC_v2_mc": json.loads((OUT / "v2" / "freeze.json").read_text())
        ["table"]["MPC_v2_mc"],
        "MPC_v3_probe20": winner["agg"],
        "MPC_v3_mc": mc_sum,
        "winner_id": winner["id"],
        "score20": winner["score20"],
        "u575_ms": winner["u575_ms"],
    }
    (OUT / "freeze.json").write_text(json.dumps(table, indent=2, default=str))
    return table, winner


def _write_yaml(path, cfg, meta):
    w = cfg["weights"]
    doc = {
        "name": "mpc_v3",
        "horizon_steps": int(cfg["horizon_steps"]),
        "n_random": int(cfg["n_random"]),
        "max_cmd_deg": float(cfg.get("max_cmd_deg", 40.0)),
        "advisor_step_s": float(cfg.get("advisor_step_s", 300.0)),
        "use_sun_ephemeris": True,
        "soc_lo": float(cfg.get("soc_lo", 0.4)),
        "soc_hi": float(cfg.get("soc_hi", 0.6)),
        "soc_k": float(cfg.get("soc_k", 6.0)),
        "weights": {k: float(w[k]) for k in SamplingMpcPolicy.DEFAULT_W},
        "sensors": {"enabled": True, "soc_sigma": 0.005},
        "dipole_quant_bits": None,
        "u575_ms": float(meta["u575_ms"]),
        "u575_ms_actual": float(meta["u575_ms_actual"]),
        "tune": meta,
    }
    header = (
        f"# Sampling MPC v3 — FROZEN {meta['date']}.\n"
        f"# Winner of tune_mpc_v3 (id={meta['id']}, score20={meta['score20']}).\n"
        f"# Sensors ON. Outputs live in outputs/mpc/ (v8 tree not written).\n"
        f"# Dipole 8-bit: not in the plant.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False))


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def stage_plots(table=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
    plt.rcParams.update({
        "figure.facecolor": surf, "axes.facecolor": surf, "savefig.facecolor": surf,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": grid, "axes.axisbelow": True,
        "font.size": 12,
    })

    def save(fig, stem):
        for ext in ("png", "pdf"):
            fig.savefig(PLOTS / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[v3] plot {stem}", flush=True)

    rows = []
    if LEDGER.exists():
        with open(LEDGER) as f:
            rows = list(csv.DictReader(f))

    def num(r, k):
        try:
            return float(r[k])
        except (KeyError, ValueError, TypeError):
            return np.nan

    # 1. Pareto decay vs downlink
    if rows:
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        sc = ax.scatter([num(r, "downlink_min_d") for r in rows],
                        [num(r, "decay_km_d") for r in rows],
                        c=[num(r, "band_pct") for r in rows],
                        s=28, cmap="viridis", alpha=0.85, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="band %")
        ax.axvline(V2_ANCHOR["downlink_min_d"], color="#eb6834", ls="--", lw=1,
                   label="v2 10-orbit")
        ax.axhline(V2_ANCHOR["decay_km_d"], color="#eb6834", ls="--", lw=1)
        ax.set_xlabel("downlink min/d  (higher better)")
        ax.set_ylabel("decay km/d  (lower better)")
        ax.set_title("MPC_v3 10-orbit sweep — decay vs downlink")
        ax.legend(frameon=False)
        save(fig, "fig2_pareto_decay_dl")

        # 2. score histogram / leaderboard
        scored = [r for r in rows if r.get("score_v2") not in ("", None)]
        scored.sort(key=lambda r: -num(r, "score_v2"))
        top = scored[:18]
        fig, ax = plt.subplots(figsize=(9, 6))
        ys = np.arange(len(top))
        ax.barh(ys, [num(r, "score_v2") for r in top], color="#2a78d6")
        ax.axvline(1.0, color="#eb6834", ls="--", label="v2")
        ax.set_yticks(ys)
        ax.set_yticklabels([r["id"][:42] for r in top], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("min-ratio vs MPC_v2 (10-orbit, sensors on)")
        ax.set_title("MPC_v3 leaderboard")
        ax.legend(frameon=False)
        save(fig, "fig7_leaderboard")

        # 3. cd × slew heatmap of score (H=6 R=8 subset)
        sub = [r for r in scored
               if int(float(r["horizon_steps"])) == 6 and int(float(r["n_random"])) == 8]
        if sub:
            cds = sorted({num(r, "cd") for r in sub})
            sls = sorted({num(r, "slew") for r in sub})
            M = np.full((len(sls), len(cds)), np.nan)
            for r in sub:
                i, j = sls.index(num(r, "slew")), cds.index(num(r, "cd"))
                v = num(r, "score_v2")
                if np.isnan(M[i, j]) or v > M[i, j]:
                    M[i, j] = v
            fig, ax = plt.subplots(figsize=(6.4, 5.0))
            im = ax.imshow(M, origin="lower", cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(cds))); ax.set_xticklabels([f"{c:g}" for c in cds])
            ax.set_yticks(range(len(sls))); ax.set_yticklabels([f"{s:g}" for s in sls])
            ax.set_xlabel("cd"); ax.set_ylabel("slew")
            ax.set_title("score vs v2  (H=6, R=8)")
            fig.colorbar(im, ax=ax, label="score_v2")
            save(fig, "fig3_cd_slew_heatmap")

        # 4. U575 vs score
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.scatter([num(r, "u575_ms") for r in scored],
                   [num(r, "score_v2") for r in scored],
                   c="#2a78d6", s=22, alpha=0.8)
        ax.axvline(15.0, color="#e34948", ls="--", label="15 ms gate")
        ax.axhline(1.0, color="#eb6834", ls="--", label="v2")
        ax.set_xlabel("U575 ms (published-scale)")
        ax.set_ylabel("score vs v2")
        ax.set_title("Decision cost vs quality")
        ax.legend(frameon=False)
        save(fig, "fig5_u575_vs_score")

        # 5. horizon effect
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for metric, lab, col in (("decay_km_d", "decay km/d", "#e34948"),
                                 ("band_pct", "band %", "#1baf7a"),
                                 ("downlink_min_d", "downlink min/d", "#2a78d6")):
            by = {}
            for r in rows:
                by.setdefault(int(float(r["horizon_steps"])), []).append(num(r, metric))
            xs = sorted(by)
            ax.plot(xs, [np.nanmedian(by[x]) for x in xs], "o-", color=col, label=lab)
        ax.set_xlabel("horizon_steps")
        ax.set_title("Median metric vs horizon (all other knobs mixed)")
        ax.legend(frameon=False)
        save(fig, "fig4_horizon")

    # 6. v1 / v2 / v3 scoreboard (probe20)
    v1 = PUBLISHED_V1
    v2p = json.loads((OUT / "v2" / "freeze.json").read_text())["table"]["MPC_v2_probe20"]
    v3p = None
    if table and table.get("MPC_v3_probe20"):
        v3p = table["MPC_v3_probe20"]
    elif (OUT / "ref_v3.json").exists():
        v3p = json.loads((OUT / "ref_v3.json").read_text())["MPC"]["agg"]
    labels = ["decay km/d", "band %", "downlink min/d", "mtq J", "U575 ms"]
    def vec(d, kind):
        if kind == "v1":
            return [d["decay_nominal"], d["band_pct"], d["downlink_min_d"],
                    d["mtq_energy_J"], d["u575_ms"]]
        return [d.get("decay_nominal", d.get("decay_km_d")),
                d["band_pct"], d["downlink_min_d"],
                d.get("mtq_energy_J", np.nan),
                d.get("u575_ms", np.nan) if "u575_ms" in d else np.nan]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, vec(v1, "v1"), w, label="MPC_v1 published", color="#8a8880")
    ax.bar(x, vec(v2p, "v2"), w, label="MPC_v2 sensors", color="#2a78d6")
    if v3p:
        # attach u575 from freeze if missing
        if "u575_ms" not in v3p and table:
            v3p = dict(v3p)
            v3p["u575_ms"] = table.get("u575_ms")
        ax.bar(x + w, vec(v3p, "v3"), w, label="MPC_v3 sensors", color="#1baf7a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Probe-20 scoreboard (Deep RL must match this protocol)")
    ax.legend(frameon=False)
    save(fig, "fig1_scoreboard")

    # 7. MC
    v2mc = json.loads((OUT / "v2" / "freeze.json").read_text())["table"]["MPC_v2_mc"]
    v3mc = (table or {}).get("MPC_v3_mc") if table else None
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    labs = ["lifetime d", "decay@500", "decay@400", "decay@300", "reached 300 /512"]
    a = [v1["mc_lifetime_d"], 5.40, 14.3, v1["decay_300"], 252]
    b = [v2mc["days_500_to_300_mean"], v2mc["decay_500_mean"], v2mc["decay_400_mean"],
         v2mc["decay_300_mean"], v2mc["reached_floor"]]
    x = np.arange(len(labs)); w = 0.26
    ax.bar(x - w, a, w, label="v1 (28 d cap)", color="#8a8880")
    ax.bar(x, b, w, label="v2 (69 d cap, sensors)", color="#2a78d6")
    if v3mc:
        c = [v3mc.get("days_500_to_300_mean"), v3mc.get("decay_500_mean"),
             v3mc.get("decay_400_mean"), v3mc.get("decay_300_mean"),
             v3mc.get("reached_floor")]
        ax.bar(x + w, c, w, label="v3", color="#1baf7a")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=15, ha="right")
    ax.set_title("Monte Carlo 500→300 km")
    ax.legend(frameon=False)
    save(fig, "fig6_mc")


def _write_final_report(table, winner):
    v1, v2p, v2mc = table["MPC_v1"], table["MPC_v2_probe20"], table["MPC_v2_mc"]
    v3p = table["MPC_v3_probe20"]
    v3mc = table.get("MPC_v3_mc") or {}
    n = sum(1 for _ in open(LEDGER)) - 1 if LEDGER.exists() else 0
    def fmt(x, nd=2):
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}"
    text = f"""# MPC_v2 → MPC_v3 report

For Deep RL matching. **Protocol: 20-orbit × 3 probes × seeds 1001/1002, v8a plant, sensors ON.**
Do not mix with the 10-orbit sweep numbers.

Winner: `{table['winner_id']}`  score20 vs v2 = {fmt(table.get('score20'), 4)}
U575 = {fmt(table.get('u575_ms'))} ms (budget 15). Sweep n = {n} configs. Wall cap 3 h.

## Probe-20 (the numbers to beat)

| metric | MPC_v1 published (truth) | MPC_v2 | MPC_v3 |
|---|---|---|---|
| decay nominal km/d | {v1['decay_nominal']} | {fmt(v2p['decay_nominal'], 3)} | {fmt(v3p.get('decay_nominal'), 3)} |
| decay spike km/d | — | {fmt(v2p.get('decay_spike'), 3)} | {fmt(v3p.get('decay_spike'), 3)} |
| band % | {v1['band_pct']} | {fmt(v2p['band_pct'], 2)} | {fmt(v3p.get('band_pct'), 2)} |
| downlink min/d | {v1['downlink_min_d']} | {fmt(v2p['downlink_min_d'], 2)} | {fmt(v3p.get('downlink_min_d'), 2)} |
| brownouts | 0 | {fmt(v2p['brownouts'], 0)} | {fmt(v3p.get('brownouts'), 0)} |
| slews / day | — | {fmt(v2p.get('slews_per_day'), 1)} | {fmt(v3p.get('slews_per_day'), 1)} |
| mtq energy J | {v1['mtq_energy_J']} | {fmt(v2p.get('mtq_energy_J'), 2)} | {fmt(v3p.get('mtq_energy_J'), 2)} |
| PEI | — | {fmt(v2p.get('pei'), 4)} | {fmt(v3p.get('pei'), 4)} |
| U575 ms | 8.4 | {fmt(v2p.get('u575_ms'))} | {fmt(table.get('u575_ms'))} |

## Monte Carlo 500→295 km (69-day ceiling, sensors on)

| metric | v1 (28 d cap) | v2 | v3 |
|---|---|---|---|
| lifetime d | 17.9 | {fmt(v2mc['days_500_to_300_mean'])} | {fmt(v3mc.get('days_500_to_300_mean'))} |
| decay@500 | 5.40 | {fmt(v2mc['decay_500_mean'])} | {fmt(v3mc.get('decay_500_mean'))} |
| decay@400 | 14.3 | {fmt(v2mc['decay_400_mean'])} | {fmt(v3mc.get('decay_400_mean'))} |
| decay@300 | 88.2 | {fmt(v2mc['decay_300_mean'])} | {fmt(v3mc.get('decay_300_mean'))} |
| reached 300 km | 252/512 | {v2mc['reached_floor']}/512 | {v3mc.get('reached_floor', '—')}/{v3mc.get('n', '—')} |

## Frozen v3 config

See `outputs/mpc/mpc_v3.yaml` and `python/configs/mpc_v3.yaml`.
Horizon {winner['cfg']['horizon_steps']}, n_random {winner['cfg']['n_random']},
weights {winner['cfg']['weights']}, max_cmd {winner['cfg'].get('max_cmd_deg', 40)}.

## Deep RL match bar

Beat or match **MPC_v3 probe20** on decay, band, downlink, brownouts=0, with
decision cost ≤ 1.5 ms on the U575 published-scale model (10× vs this MPC).

Plots: `outputs/mpc/plots/`. Ledger: `outputs/mpc/ledger.csv`.
v2 archive (untouched v8 tree): `outputs/mpc/v2/`.
"""
    (OUT / "REPORT.md").write_text(text)
    print(text, flush=True)


def stage_all(workers):
    global T0
    T0 = time.time()
    stage_setup()
    stage_plots()  # v2-only plots first
    prior = []
    if remaining() > 60:
        prior += round_G(workers)
        stage_plots()
    if remaining() > 60:
        prior += round_H(workers)
    if remaining() > 60:
        prior += round_I(workers)
        stage_plots()
    if remaining() > 60:
        prior += round_J(workers)
    if remaining() > 90:
        prior += round_K(workers, _all_runs())
        stage_plots()
    confirms = stage_confirm(workers)
    table, winner = stage_freeze(confirms, workers)
    stage_plots(table)
    _write_final_report(table, winner)
    print(f"[v3] done in {time.time()-T0:.0f}s  remain_budget={remaining():.0f}s",
          flush=True)


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("setup", "G", "H", "I", "J", "K", "confirm",
                             "freeze", "plots", "all"))
    ap.add_argument("--workers", type=int,
                    default=max(8, (os.cpu_count() or 8) - 4))
    a = ap.parse_args()
    T0 = time.time()
    print(f"[v3] stage={a.stage} workers={a.workers} cpu={os.cpu_count()} {_now()}",
          flush=True)
    if a.stage == "setup":
        stage_setup()
    elif a.stage == "G":
        stage_setup(); round_G(a.workers); stage_plots()
    elif a.stage == "H":
        round_H(a.workers); stage_plots()
    elif a.stage == "I":
        round_I(a.workers); stage_plots()
    elif a.stage == "J":
        round_J(a.workers)
    elif a.stage == "K":
        round_K(a.workers, _all_runs()); stage_plots()
    elif a.stage == "confirm":
        stage_confirm(a.workers)
    elif a.stage == "plots":
        stage_setup(); stage_plots()
    else:
        stage_all(a.workers)


if __name__ == "__main__":
    main()
