"""MPC_v2 3-hour sprint: sensor-noisy sampling MPC, then freeze.

Rounds A–F on a 10-orbit 400 km probe (3 seeds, quiet + storm). Score is the
worst metric ratio vs the v1 config on the SAME protocol (higher is better;
brownout ⇒ disqualified). U575 gate uses the published 24×20 stand-in scaled
to (H, n_random), budget 15 ms.

    PYTHONPATH=python python -m arlamx_v2.tune_mpc --stage all --workers 20

Resumable: each config writes outputs/v8/mpc_tune/runs/<id>.json.
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
from arlamx_v2.config import CONFIG_DIR
from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.inference_budget import mpc_u575_ms
from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

V8 = OUTPUTS / "v8"
OUT = V8 / "mpc_tune"
RUNS = OUT / "runs"
LEDGER = OUT / "ledger.csv"
STATE = OUT / "state.json"
BUDGET_MS = 15.0
N_ORBITS = 10.0
SEEDS = (1001, 1002, 1003)
VARIANT = "v8a"
SLEW_FLOOR = 0.05          # slews/day denominator floor
DECAY_FLOOR = 0.05         # km/d

# Quiet + constant-storm probes (10-orbit protocol). Storm is F10.7 220 / Ap 120
# held for the episode, not a mid-episode jump.
QUIET_OPT = dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001,
                 f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                 nu_deg=0.0, mass_kg=0.625, soc=0.5,
                 omega_dps=[1.0, 1.0, 0.5])
STORM_OPT = dict(QUIET_OPT, f107=220.0, ap=120.0)
TUNE_PROBES = (("quiet", QUIET_OPT, ()), ("storm", STORM_OPT, ()))

PUBLISHED_V1 = {
    "decay_nominal": 9.28, "band_pct": 86.6, "downlink_min_d": 23.2,
    "brownouts": 0.0, "mtq_energy_J": 2.97, "u575_ms": 8.4,
    "mc_lifetime_d": 17.9, "decay_300": 88.2,
}

GRIDS = {
    "horizon_steps": [3, 6, 9, 12],
    "n_random": [8, 16, 32],
    "cd": [0.3, 0.5, 0.8],
    "slew": [0.2, 0.5, 1.0],
    "gs": [2.0, 3.0, 4.0],
    "gs_align": [1.0, 2.0],
    "power_low_pen": [4.0, 8.0],
    "soc_k": [6.0, 10.0],
}

LEDGER_FIELDS = [
    "id", "round", "horizon_steps", "n_random", "cd", "slew", "gs", "gs_align",
    "power_low_pen", "soc_k", "decay_km_d", "band_pct", "downlink_min_d",
    "brownouts", "slews_per_day", "mtq_energy_J_d", "pei", "score",
    "u575_ms", "u575_ms_actual", "in_budget", "wall_s", "disqualified",
]


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _base_cfg():
    return SamplingMpcPolicy().to_config()


def _set_knob(cfg, key, value):
    cfg = deepcopy(cfg)
    if key in ("horizon_steps", "n_random"):
        cfg[key] = int(value)
    elif key == "soc_k":
        cfg[key] = float(value)
    else:
        cfg.setdefault("weights", {})[key] = float(value)
    return cfg


def _get_knob(cfg, key):
    if key in ("horizon_steps", "n_random", "soc_k"):
        return cfg[key]
    return cfg["weights"][key]


def cfg_id(rnd, cfg):
    w = cfg["weights"]
    return (f"{rnd}_H{int(cfg['horizon_steps'])}_R{int(cfg['n_random'])}"
            f"_cd{w['cd']:g}_sl{w['slew']:g}_gs{w['gs']:g}_ga{w['gs_align']:g}"
            f"_pl{w['power_low_pen']:g}_k{cfg['soc_k']:g}")


def cost_ms(cfg, scale="published"):
    return mpc_u575_ms(int(cfg["horizon_steps"]), int(cfg["n_random"]),
                       scale=scale)


def score_vs(metrics, anchor):
    """Worst-metric ratio vs anchor. >1 beats the anchor on every metric.
    None if any brownout."""
    if float(metrics.get("brownouts", 0)) > 0.5:
        return None
    ratios = [
        max(anchor["decay_km_d"], DECAY_FLOOR) / max(metrics["decay_km_d"], DECAY_FLOOR),
        metrics["band_pct"] / max(anchor["band_pct"], 1.0),
        metrics["downlink_min_d"] / max(anchor["downlink_min_d"], 0.1),
        max(anchor["slews_per_day"], SLEW_FLOOR) / max(metrics["slews_per_day"], SLEW_FLOOR),
    ]
    return float(min(ratios))


def _mk_env(opt, n_orbits=N_ORBITS):
    env = ArlamxV2Env(seed=0, variant=VARIANT)
    alt = float(opt["altitude_km"])
    period = 2 * np.pi * np.sqrt((6371e3 + alt * 1e3) ** 3 / 3.986004418e14)
    env._max_steps = int(float(n_orbits) * period / env._advisor_s)
    return env


def _eval_one(job):
    """Worker: one config × 3 seeds × 2 probes. job is a plain dict."""
    import warnings
    warnings.filterwarnings("ignore")
    from arlamx_v2.bench_v8 import _rollout, mpc_fn

    cid, rnd, cfg, noisy = job["id"], job["round"], job["cfg"], job["noisy"]
    dest = RUNS / f"{cid}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    t0 = time.time()
    predict = mpc_fn(cfg=cfg, noisy=noisy, seed=0)
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
    u_pub = cost_ms(cfg, "published")
    u_act = cost_ms(cfg, "actual")
    out = {
        "id": cid, "round": rnd, "cfg": cfg, "noisy": noisy,
        "agg": agg, "per": rows,
        "u575_ms": u_pub, "u575_ms_actual": u_act,
        "in_budget": bool(u_pub <= BUDGET_MS),
        "wall_s": round(time.time() - t0, 1),
        "when": _now(),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    return out


def _ensure_dirs():
    RUNS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def _load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"winners": {}, "anchor_id": None}


def _save_state(st):
    _ensure_dirs()
    STATE.write_text(json.dumps(st, indent=2, default=str))


def _ledger_write(row):
    _ensure_dirs()
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def _row_from(result, score, disqualified):
    cfg, agg = result["cfg"], result["agg"]
    w = cfg["weights"]
    return {
        "id": result["id"], "round": result["round"],
        "horizon_steps": cfg["horizon_steps"], "n_random": cfg["n_random"],
        "cd": w["cd"], "slew": w["slew"], "gs": w["gs"],
        "gs_align": w["gs_align"], "power_low_pen": w["power_low_pen"],
        "soc_k": cfg["soc_k"],
        "decay_km_d": round(agg["decay_km_d"], 4),
        "band_pct": round(agg["band_pct"], 3),
        "downlink_min_d": round(agg["downlink_min_d"], 3),
        "brownouts": int(agg["brownouts"]),
        "slews_per_day": round(agg["slews_per_day"], 3),
        "mtq_energy_J_d": round(agg["mtq_energy_J_d"], 4),
        "pei": round(agg["pei"], 4),
        "score": "" if score is None else round(score, 4),
        "u575_ms": round(result["u575_ms"], 3),
        "u575_ms_actual": round(result["u575_ms_actual"], 3),
        "in_budget": int(result["in_budget"]),
        "wall_s": result["wall_s"],
        "disqualified": int(bool(disqualified)),
    }


def _done_ids():
    if not LEDGER.exists():
        return set()
    with open(LEDGER) as f:
        return {r["id"] for r in csv.DictReader(f)}


def _run_jobs(jobs, workers, anchor=None):
    """Evaluate configs in parallel. `anchor` is an agg dict or None (score later)."""
    _ensure_dirs()
    done = _done_ids()
    pending = [j for j in jobs if j["id"] not in done]
    print(f"[tune] {len(jobs)} configs, {len(pending)} to run, "
          f"{len(jobs) - len(pending)} cached, workers={workers}", flush=True)
    results = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_eval_one, j): j for j in pending}
            n = 0
            for fut in as_completed(futs):
                n += 1
                r = fut.result()
                sc = score_vs(r["agg"], anchor) if anchor is not None else None
                dq = bool(anchor is not None and sc is None)
                if r["id"] not in done:
                    _ledger_write(_row_from(r, sc, dq))
                    done.add(r["id"])
                tag = "DQ" if dq else (f"score={sc:.3f}" if sc is not None else "ok")
                print(f"[tune] {n}/{len(pending)} {r['id']} {tag} "
                      f"dec={r['agg']['decay_km_d']:.2f} band={r['agg']['band_pct']:.1f} "
                      f"dl={r['agg']['downlink_min_d']:.1f} "
                      f"u575={r['u575_ms']:.1f}ms {r['wall_s']}s", flush=True)
                results.append(r)
    # reload cached
    for j in jobs:
        p = RUNS / f"{j['id']}.json"
        if p.exists() and j["id"] not in {r["id"] for r in results}:
            results.append(json.loads(p.read_text()))
    if anchor is not None:
        for r in results:
            r["score"] = score_vs(r["agg"], anchor)
            r["disqualified"] = r["score"] is None
    return results


def _pick(results, budget_only=True):
    ok = [r for r in results
          if r.get("score") is not None
          and (r.get("in_budget", True) if budget_only else True)]
    if not ok and budget_only:
        print("[tune] no in-budget survivor; picking unconstrained", flush=True)
        return _pick(results, budget_only=False)
    if not ok:
        return None
    return max(ok, key=lambda r: r["score"])


def _print_round(rnd, results):
    print(f"\n[tune] round {rnd}  ({len(results)} configs)", flush=True)
    print(f"{'id':52s} {'score':>7s} {'dec':>7s} {'band':>6s} {'dl':>6s} "
          f"{'slew/d':>7s} {'ms':>6s} {'OK':>3s}", flush=True)
    for r in sorted(results, key=lambda x: (-(x.get("score") or -1e9))):
        sc = r.get("score")
        print(f"{r['id']:52s} {('-' if sc is None else f'{sc:.3f}'):>7s} "
              f"{r['agg']['decay_km_d']:7.2f} {r['agg']['band_pct']:6.1f} "
              f"{r['agg']['downlink_min_d']:6.1f} {r['agg']['slews_per_day']:7.2f} "
              f"{r['u575_ms']:6.1f} {'Y' if r['in_budget'] else 'n':>3s}", flush=True)


def _jobs_for(rnd, cfgs, noisy=True):
    return [{"id": cfg_id(rnd, c), "round": rnd, "cfg": c, "noisy": noisy,
             "n_orbits": N_ORBITS} for c in cfgs]


def _preserve_v1_ref():
    src = V8 / "data" / "ref.json"
    dst = V8 / "data" / "ref_v1.json"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"[tune] preserved {src} -> {dst}", flush=True)
    return dst if dst.exists() else src


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------

def stage_anchor(workers):
    """v1 weights on the 10-orbit noisy protocol — the score denominator."""
    _ensure_dirs()
    cfg = _base_cfg()
    jobs = _jobs_for("Z", [cfg])
    jobs[0]["id"] = "Z_v1_anchor"
    results = _run_jobs(jobs, workers, anchor=None)
    r = results[0]
    r["score"] = 1.0
    r["disqualified"] = float(r["agg"]["brownouts"]) > 0.5
    if "Z_v1_anchor" not in _done_ids():
        _ledger_write(_row_from(r, 1.0, r["disqualified"]))
    st = _load_state()
    st["anchor_id"] = r["id"]
    st["anchor_agg"] = r["agg"]
    _save_state(st)
    print(f"[tune] anchor (noisy v1, 10-orbit): decay={r['agg']['decay_km_d']:.2f} "
          f"band={r['agg']['band_pct']:.1f} dl={r['agg']['downlink_min_d']:.1f} "
          f"slew/d={r['agg']['slews_per_day']:.2f} pei={r['agg']['pei']:.3f} "
          f"brn={r['agg']['brownouts']:.0f}", flush=True)
    return r["agg"]


def _anchor():
    st = _load_state()
    if st.get("anchor_agg"):
        return st["anchor_agg"]
    raise SystemExit("no anchor — run --stage anchor first")


def stage_A(workers):
    cfg0 = _base_cfg()
    cfgs = [_set_knob(cfg0, "horizon_steps", h) for h in GRIDS["horizon_steps"]]
    res = _run_jobs(_jobs_for("A", cfgs), workers, _anchor())
    _print_round("A", res)
    w = _pick(res)
    st = _load_state()
    st["winners"]["A"] = w["id"] if w else None
    st["backbone_H"] = int(w["cfg"]["horizon_steps"]) if w else 6
    _save_state(st)
    print(f"[tune] A winner: {w['id'] if w else None}  H={st['backbone_H']}", flush=True)
    return w


def stage_B(workers):
    st = _load_state()
    H = int(st.get("backbone_H", 6))
    cfg0 = _set_knob(_base_cfg(), "horizon_steps", H)
    cfgs = [_set_knob(cfg0, "n_random", n) for n in GRIDS["n_random"]]
    res = _run_jobs(_jobs_for("B", cfgs), workers, _anchor())
    _print_round("B", res)
    w = _pick(res)
    st = _load_state()
    st["winners"]["B"] = w["id"] if w else None
    st["backbone_R"] = int(w["cfg"]["n_random"]) if w else 8
    _save_state(st)
    print(f"[tune] B winner: {w['id'] if w else None}  R={st['backbone_R']}", flush=True)
    return w


def _backbone():
    st = _load_state()
    cfg = _base_cfg()
    cfg = _set_knob(cfg, "horizon_steps", int(st.get("backbone_H", 6)))
    cfg = _set_knob(cfg, "n_random", int(st.get("backbone_R", 8)))
    return cfg


def stage_CDE(workers):
    """C, D, E from the A/B backbone, all in one pool."""
    bb = _backbone()
    jobs = []
    # C: cd × slew
    for cd in GRIDS["cd"]:
        for sl in GRIDS["slew"]:
            c = _set_knob(_set_knob(bb, "cd", cd), "slew", sl)
            jobs.append(_jobs_for("C", [c])[0])
    # D: gs × gs_align
    for gs in GRIDS["gs"]:
        for ga in GRIDS["gs_align"]:
            c = _set_knob(_set_knob(bb, "gs", gs), "gs_align", ga)
            jobs.append(_jobs_for("D", [c])[0])
    # E: power_low_pen × soc_k
    for pen in GRIDS["power_low_pen"]:
        for k in GRIDS["soc_k"]:
            c = _set_knob(_set_knob(bb, "power_low_pen", pen), "soc_k", k)
            jobs.append(_jobs_for("E", [c])[0])
    # de-dup ids
    seen, uniq = set(), []
    for j in jobs:
        if j["id"] not in seen:
            seen.add(j["id"])
            uniq.append(j)
    res = _run_jobs(uniq, workers, _anchor())
    by = {}
    for r in res:
        by.setdefault(r["round"], []).append(r)
    st = _load_state()
    for rnd in ("C", "D", "E"):
        _print_round(rnd, by.get(rnd, []))
        w = _pick(by.get(rnd, []))
        st["winners"][rnd] = w["id"] if w else None
        print(f"[tune] {rnd} winner: {st['winners'][rnd]}", flush=True)
    _save_state(st)
    return res


def _load_run(cid):
    p = RUNS / f"{cid}.json"
    return json.loads(p.read_text()) if cid and p.exists() else None


def stage_F(workers):
    """Combine A–E winners, then ±1 grid step on each knob."""
    st = _load_state()
    combo = _backbone()
    # overlay C/D/E winners' knobs
    for rnd, keys in (("C", ("cd", "slew")),
                      ("D", ("gs", "gs_align")),
                      ("E", ("power_low_pen", "soc_k"))):
        r = _load_run(st["winners"].get(rnd))
        if r is None:
            continue
        for k in keys:
            combo = _set_knob(combo, k, _get_knob(r["cfg"], k))
    cfgs = [combo]
    for key, grid in GRIDS.items():
        cur = _get_knob(combo, key)
        # match as float
        idxs = [i for i, v in enumerate(grid) if abs(float(v) - float(cur)) < 1e-12]
        if not idxs:
            continue
        i = idxs[0]
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                cfgs.append(_set_knob(combo, key, grid[j]))
    # unique
    uniq, seen = [], set()
    jobs = []
    for c in cfgs:
        j = _jobs_for("F", [c])[0]
        if j["id"] not in seen:
            seen.add(j["id"])
            uniq.append(c)
            jobs.append(j)
    res = _run_jobs(jobs, workers, _anchor())
    _print_round("F", res)
    w = _pick(res)
    st = _load_state()
    st["winners"]["F"] = w["id"] if w else None
    _save_state(st)
    print(f"[tune] F winner: {st['winners']['F']}", flush=True)
    return w


def _all_results():
    rows = []
    if not RUNS.exists():
        return rows
    for p in RUNS.glob("*.json"):
        r = json.loads(p.read_text())
        if "agg" in r and r.get("round") != "Z":
            r["score"] = score_vs(r["agg"], _anchor())
            r["disqualified"] = r["score"] is None
            r.setdefault("in_budget", r.get("u575_ms", 99) <= BUDGET_MS)
            rows.append(r)
    return rows


def _write_mpc_yaml(path, cfg, meta):
    sensors = {"enabled": True, "soc_sigma": 0.005}
    doc = {
        "name": "mpc_v2",
        "horizon_steps": int(cfg["horizon_steps"]),
        "n_random": int(cfg["n_random"]),
        "max_cmd_deg": float(cfg.get("max_cmd_deg", 40.0)),
        "advisor_step_s": float(cfg.get("advisor_step_s", 300.0)),
        "use_sun_ephemeris": bool(cfg.get("use_sun_ephemeris", True)),
        "soc_lo": float(cfg.get("soc_lo", 0.4)),
        "soc_hi": float(cfg.get("soc_hi", 0.6)),
        "soc_k": float(cfg.get("soc_k", 6.0)),
        "weights": {k: float(cfg["weights"][k]) for k in SamplingMpcPolicy.DEFAULT_W},
        "sensors": sensors,
        "dipole_quant_bits": None,
        "u575_ms": float(meta["u575_ms"]),
        "u575_ms_actual": float(meta["u575_ms_actual"]),
        "tune": meta,
    }
    header = (
        f"# Sampling MPC v2 — FROZEN {meta.get('date', _now())}.\n"
        f"# Do not edit. Winner of tune_mpc rounds A–F "
        f"(id={meta.get('id')}, score={meta.get('score')}).\n"
        f"# U575 {meta.get('u575_ms'):.2f} ms published-scale "
        f"(budget {BUDGET_MS:.0f} ms); actual-algorithm "
        f"{meta.get('u575_ms_actual'):.2f} ms.\n"
        f"# Sensors ON (GNSS + dual mag + SoC 0.5 % [ASSUME]).\n"
        f"# Dipole 8-bit quantisation: NOT in the C++ plant — skipped "
        f"[NOTE 2026-08-23].\n"
        f"# Tags: [TUNE-A] horizon  [TUNE-B] n_random  [TUNE-C] cd×slew  "
        f"[TUNE-D] gs  [TUNE-E] band  [TUNE-F] combo.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
    return doc


def stage_freeze(workers, skip_mc=False, n_mc=512, max_steps=20000):
    """Write mpc_v2.yaml, full 20-orbit probe suite, optional 512-orbit MC."""
    from arlamx_v2.bench_v8 import mpc_fn, probe_suite

    _preserve_v1_ref()
    pool = _all_results()
    winner = _pick(pool)
    if winner is None:
        raise SystemExit("no surviving config to freeze")
    print(f"[freeze] winner {winner['id']} score={winner['score']:.4f} "
          f"u575={winner['u575_ms']:.2f} ms", flush=True)

    meta = {
        "id": winner["id"],
        "score": winner["score"],
        "u575_ms": winner["u575_ms"],
        "u575_ms_actual": winner["u575_ms_actual"],
        "date": _now(),
        "protocol": "10-orbit × 3 seeds × quiet+storm, sensors on, v8a",
        "budget_ms": BUDGET_MS,
        "winners": _load_state().get("winners", {}),
        "dipole_quant": "skipped — plant has no 8-bit DAC",
    }
    yaml_path = CONFIG_DIR / "mpc_v2.yaml"
    _write_mpc_yaml(yaml_path, winner["cfg"], meta)
    print(f"[freeze] wrote {yaml_path}", flush=True)

    # Full 20-orbit 3-probe suite (same protocol as historical ref.json).
    print("[freeze] 20-orbit probe suite (sensors ON)", flush=True)
    fn = mpc_fn(cfg=winner["cfg"], noisy=True, seed=0)
    agg, per = probe_suite(fn, VARIANT)
    ref_v2 = {
        "MPC": {"agg": agg, "per_probe": per,
                "cfg_id": winner["id"], "sensors": True,
                "u575_ms": winner["u575_ms"]},
        "note": "MPC_v2 freeze; Heuristic is not re-scored here. "
                "See ref_v1.json for the historical pair.",
    }
    (V8 / "data").mkdir(parents=True, exist_ok=True)
    (V8 / "data" / "ref_v2.json").write_text(json.dumps(ref_v2, indent=2))
    # Keep Heuristic from v1, replace MPC.
    v1_path = V8 / "data" / "ref_v1.json"
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())
        merged = dict(v1)
        merged["MPC"] = ref_v2["MPC"]
        merged["MPC_v1"] = v1.get("MPC")
        merged["note"] = "ref.json after MPC_v2 freeze. MPC is v2; MPC_v1 is historical."
        (V8 / "data" / "ref.json").write_text(json.dumps(merged, indent=2))
        print("[freeze] updated ref.json (MPC=v2, MPC_v1 preserved inside)", flush=True)

    table = {
        "MPC_v1": PUBLISHED_V1,
        "MPC_v2_probe10": {**winner["agg"], "score": winner["score"],
                           "u575_ms": winner["u575_ms"],
                           "u575_ms_actual": winner["u575_ms_actual"]},
        "MPC_v2_probe20": {
            "decay_nominal": agg["decay_nominal"],
            "decay_spike": agg["decay_spike"],
            "band_pct": agg["band_pct"],
            "downlink_min_d": agg["downlink_min_d"],
            "brownouts": agg["brownouts"],
            "slews_per_day": agg.get("slews_per_day"),
            "mtq_energy_J": agg.get("mtq_energy_J"),
            "mtq_energy_J_d": agg.get("mtq_energy_J_d"),
            "pei": agg.get("pei"),
            "u575_ms": winner["u575_ms"],
        },
    }

    mc_sum = None
    if not skip_mc:
        from arlamx_v2.mc_decay import run_study
        pending = OUT / "mpc_v2_pending.json"
        pending.write_text(json.dumps(
            {**winner["cfg"], "sensors": {"enabled": True}}, indent=2))
        print(f"[freeze] MC 512 × 500→295 km, max_steps={max_steps}", flush=True)
        mc = run_study([f"mpcjson:{pending}"], n_mc, variant="v10",
                       workers=workers, max_steps=max_steps, floor_km=295.0,
                       outfile=str(V8 / "data" / "mc_decay_v2.json"))
        spec = f"mpcjson:{pending}"
        mc_sum = mc["summary"].get(spec, {})
        table["MPC_v2_mc"] = mc_sum
    (OUT / "freeze.json").write_text(json.dumps(
        {"winner": winner["id"], "table": table, "meta": meta}, indent=2,
        default=str))
    _print_table(table)
    return table


def _print_table(table):
    v1, v2 = table.get("MPC_v1", {}), table.get("MPC_v2_probe20", {})
    print("\n[freeze] deliverable table", flush=True)
    print(f"{'':28s} {'MPC_v1':>12s} {'MPC_v2':>12s}", flush=True)
    rows = [
        ("decay nominal km/d", v1.get("decay_nominal"), v2.get("decay_nominal")),
        ("band %", v1.get("band_pct"), v2.get("band_pct")),
        ("downlink min/d", v1.get("downlink_min_d"), v2.get("downlink_min_d")),
        ("brownouts", v1.get("brownouts"), v2.get("brownouts")),
        ("slews / day", "—", v2.get("slews_per_day")),
        ("mtq energy J", v1.get("mtq_energy_J"), v2.get("mtq_energy_J")),
        ("PEI", "—", v2.get("pei")),
        ("U575 ms", v1.get("u575_ms"), v2.get("u575_ms")),
    ]
    mc = table.get("MPC_v2_mc") or {}
    rows.append(("MC lifetime d", v1.get("mc_lifetime_d"),
                 mc.get("days_500_to_300_mean")))
    rows.append(("MC decay@300", v1.get("decay_300"), mc.get("decay_300_mean")))
    for name, a, b in rows:
        def fmt(x):
            if x is None or x == "—":
                return "—"
            return f"{x:.2f}" if isinstance(x, float) else str(x)
        print(f"{name:28s} {fmt(a):>12s} {fmt(b):>12s}", flush=True)


def stage_smoke(workers):
    """2-orbit × 1 seed × quiet, default weights — proves the pipeline."""
    _ensure_dirs()
    cfg = _base_cfg()
    job = {"id": "SMOKE_v1", "round": "S", "cfg": cfg, "noisy": True,
           "n_orbits": 2.0}
    # temporarily shrink probe list by monkeypatching is too messy; just run
    from arlamx_v2.bench_v8 import _rollout, mpc_fn
    predict = mpc_fn(cfg=cfg, noisy=True, seed=0)
    env = _mk_env(QUIET_OPT, n_orbits=2.0)
    m = _rollout(env, predict, "quiet", 1001, opt=QUIET_OPT, jumps=())
    env.close()
    print(f"[smoke] decay={m['decay_km_d']:.2f} band={m['band_pct']:.1f} "
          f"dl={m['downlink_min_d']:.1f} slew/d={m['slews_per_day']:.2f} "
          f"pei={m['pei']:.3f} brn={m['brownouts']} "
          f"u575={cost_ms(cfg):.2f} ms", flush=True)
    assert np.isfinite(m["decay_km_d"])
    assert m["brownouts"] >= 0
    (OUT / "smoke.json").write_text(json.dumps(m, indent=2, default=str))
    print("[smoke] PASS", flush=True)
    return m


def stage_all(workers, skip_mc=False, n_mc=512):
    _preserve_v1_ref()
    print("[tune] notes: dipole 8-bit quantisation NOT in plant — skipped.",
          flush=True)
    stage_anchor(workers)
    stage_A(workers)
    stage_B(workers)
    stage_CDE(workers)
    stage_F(workers)
    return stage_freeze(workers, skip_mc=skip_mc, n_mc=n_mc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("smoke", "anchor", "A", "B", "CDE", "F",
                             "freeze", "all"))
    ap.add_argument("--workers", type=int,
                    default=max(4, min(20, (os.cpu_count() or 8) - 4)))
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--n-mc", type=int, default=512)
    a = ap.parse_args()
    print(f"[tune] stage={a.stage} workers={a.workers}  {_now()}", flush=True)
    if a.stage == "smoke":
        stage_smoke(a.workers)
    elif a.stage == "anchor":
        stage_anchor(a.workers)
    elif a.stage == "A":
        stage_A(a.workers)
    elif a.stage == "B":
        stage_B(a.workers)
    elif a.stage == "CDE":
        stage_CDE(a.workers)
    elif a.stage == "F":
        stage_F(a.workers)
    elif a.stage == "freeze":
        stage_freeze(a.workers, skip_mc=a.skip_mc, n_mc=a.n_mc)
    else:
        stage_all(a.workers, skip_mc=a.skip_mc, n_mc=a.n_mc)


if __name__ == "__main__":
    main()
