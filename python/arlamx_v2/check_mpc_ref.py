"""Reference-integrity checks for MPC_v1 vs v2. Does not touch outputs/v8.

The v2 freeze mixed three changes at once (slew 0.2→0.5, sensors ON,
MC cap 28 d → 69 d). These checks un-mix them. v12 must keep BOTH v1 and
v2 columns until Check 1 (reproduce published v1) and Check 3 (MC cap)
are in.

    PYTHONPATH=python python -m arlamx_v2.check_mpc_ref --stage probes
    PYTHONPATH=python python -m arlamx_v2.check_mpc_ref --stage mc --workers 2

Writes outputs/mpc/checks/ only.
Level: advanced.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from arlamx_v2.advisors.mpc import SamplingMpcPolicy
from arlamx_v2.bench_v8 import mpc_fn, probe_suite
from arlamx_v2.config import CONFIG_DIR
from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

OUT = OUTPUTS / "mpc" / "checks"
V1_REF = {
    "decay_nominal": 9.28197460693557,
    "decay_spike": 17.59986221502394,
    "band_pct": 86.60605759635257,
    "brownouts": 0.0,
    "downlink_min_d": 23.237560124292344,
    "mtq_energy_J": 2.974241730071979,
}
V2_PROBE20 = {
    "decay_nominal": 9.188965076391653,
    "band_pct": 84.13972010575255,
    "downlink_min_d": 18.64044609032478,
    "mtq_energy_J": 3.7279812989722867,
    "brownouts": 0.0,
}


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _v1_cfg():
    return SamplingMpcPolicy().to_config()  # slew 0.2


def _v2_cfg():
    return SamplingMpcPolicy.from_config(
        yaml.safe_load((CONFIG_DIR / "mpc_v2.yaml").read_text())).to_config()


def _run_probe(label, cfg, noisy):
    print(f"[check] probe20 {label}  noisy={noisy} slew={cfg['weights']['slew']}",
          flush=True)
    fn = mpc_fn(cfg=cfg, noisy=noisy, seed=0)
    agg, per = probe_suite(fn, "v8a")
    rec = {"label": label, "noisy": noisy, "cfg": cfg, "agg": agg, "per_probe": per,
           "when": _now()}
    (OUT / f"{label}.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"   decay={agg['decay_nominal']:.3f} band={agg['band_pct']:.2f} "
          f"dl={agg['downlink_min_d']:.2f} mtqJ={agg['mtq_energy_J']:.2f} "
          f"brn={agg['brownouts']:.0f}", flush=True)
    return rec


def _rel(a, b):
    b = float(b)
    if abs(b) < 1e-12:
        return abs(float(a) - b)
    return abs(float(a) - b) / abs(b)


def check1_pass(agg):
    """Reproduce published v1 on the current plant, truth state, slew 0.2."""
    tests = {
        "decay_nominal": _rel(agg["decay_nominal"], V1_REF["decay_nominal"]) <= 0.03,
        "band_pct": abs(agg["band_pct"] - V1_REF["band_pct"]) <= 1.5,
        "downlink_min_d": _rel(agg["downlink_min_d"], V1_REF["downlink_min_d"]) <= 0.05,
        "brownouts": float(agg["brownouts"]) < 0.5,
        "mtq_energy_J": _rel(agg["mtq_energy_J"], V1_REF["mtq_energy_J"]) <= 0.20,
    }
    return tests, all(tests.values())


def stage_probes():
    OUT.mkdir(parents=True, exist_ok=True)
    v1, v2 = _v1_cfg(), _v2_cfg()
    cells = [
        ("A_slew02_truth", v1, False),   # published v1 protocol — Check 1
        ("B_slew05_truth", v2, False),   # weight only
        ("C_slew02_sensors", v1, True),  # sensors only
        ("D_slew05_sensors", v2, True),  # v2 freeze protocol
    ]
    results = {}
    for label, cfg, noisy in cells:
        dest = OUT / f"{label}.json"
        if dest.exists():
            print(f"[check] cached {label}", flush=True)
            results[label] = json.loads(dest.read_text())
        else:
            results[label] = _run_probe(label, deepcopy(cfg), noisy)

    a = results["A_slew02_truth"]["agg"]
    tests, ok = check1_pass(a)
    gate = {
        "when": _now(),
        "check1_reproduce_v1": {"pass": ok, "tests": tests, "agg": {
            k: a[k] for k in V1_REF}, "published": V1_REF},
        "check2_factorial": {k: results[k]["agg"] for k in results},
        "deltas": {
            "weight_only_vs_v1": _delta(results["B_slew05_truth"]["agg"], a),
            "sensors_only_vs_v1": _delta(results["C_slew02_sensors"]["agg"], a),
            "v2_protocol_vs_v1": _delta(results["D_slew05_sensors"]["agg"], a),
            "v2_vs_published": _delta(V2_PROBE20, V1_REF),
        },
        "verdict": (
            "PASS — plant matches published v1; v2 probe20 deficit is isolated below."
            if ok else
            "FAIL — current plant does not reproduce published v1. "
            "Do not use v2 as the sole RL reference."
        ),
    }
    (OUT / "gate.json").write_text(json.dumps(gate, indent=2, default=str))
    _write_gate_md(gate)
    _plot_factorial(results)
    print(f"[check] Check 1: {'PASS' if ok else 'FAIL'}", flush=True)
    return gate


def _delta(new, old):
    out = {}
    for k in ("decay_nominal", "band_pct", "downlink_min_d", "mtq_energy_J"):
        if k not in new or k not in old:
            continue
        n, o = float(new[k]), float(old[k])
        out[k] = {"new": n, "old": o, "abs": n - o,
                  "pct": (100.0 * (n - o) / o) if abs(o) > 1e-12 else None}
    return out


def _write_gate_md(gate):
    d = gate["deltas"]
    c1 = gate["check1_reproduce_v1"]
    fac = gate["check2_factorial"]

    def row(agg):
        return (f"{agg['decay_nominal']:.3f} | {agg['band_pct']:.2f} | "
                f"{agg['downlink_min_d']:.2f} | {agg['mtq_energy_J']:.2f} | "
                f"{agg['brownouts']:.0f}")

    def dpct(block, k):
        p = block[k]["pct"]
        return "—" if p is None else f"{p:+.1f}%"

    md = f"""# MPC reference gate (Check 1–2)

{gate['verdict']}

v12 Deep RL must be scored against **both** the published v1 column and the
sensor-on column until Check 1 is PASS **and** Check 3 (MC cap, below) shows
the lifetime jump is or is not a terminator artifact. Do not treat `mpc_v2.yaml`
as the sole frozen comparator.

## Check 1 — reproduce published v1 (truth, slew 0.2, probe20)

| | published v1 | this plant, same protocol | pass? |
|---|---|---|---|
| decay km/d | {V1_REF['decay_nominal']:.3f} | {c1['agg']['decay_nominal']:.3f} | {c1['tests']['decay_nominal']} |
| band % | {V1_REF['band_pct']:.2f} | {c1['agg']['band_pct']:.2f} | {c1['tests']['band_pct']} |
| downlink min/d | {V1_REF['downlink_min_d']:.2f} | {c1['agg']['downlink_min_d']:.2f} | {c1['tests']['downlink_min_d']} |
| mtq J | {V1_REF['mtq_energy_J']:.2f} | {c1['agg']['mtq_energy_J']:.2f} | {c1['tests']['mtq_energy_J']} |
| brownouts | 0 | {c1['agg']['brownouts']:.0f} | {c1['tests']['brownouts']} |

**Check 1: {'PASS' if c1['pass'] else 'FAIL'}**

## Check 2 — 2×2 factorial on the SAME probe20 suite

| cell | sensors | slew | decay | band % | downlink | mtq J | brn |
|---|---|---|---|---|---|---|---|
| A published protocol | off | 0.2 | {row(fac['A_slew02_truth'])} |
| B weight only | off | 0.5 | {row(fac['B_slew05_truth'])} |
| C sensors only | on | 0.2 | {row(fac['C_slew02_sensors'])} |
| D v2 freeze protocol | on | 0.5 | {row(fac['D_slew05_sensors'])} |

Deltas vs cell A (this-plant v1):

| change | decay | band | downlink | mtq J |
|---|---|---|---|---|
| slew 0.2→0.5, truth | {dpct(d['weight_only_vs_v1'], 'decay_nominal')} | {dpct(d['weight_only_vs_v1'], 'band_pct')} | {dpct(d['weight_only_vs_v1'], 'downlink_min_d')} | {dpct(d['weight_only_vs_v1'], 'mtq_energy_J')} |
| sensors on, slew 0.2 | {dpct(d['sensors_only_vs_v1'], 'decay_nominal')} | {dpct(d['sensors_only_vs_v1'], 'band_pct')} | {dpct(d['sensors_only_vs_v1'], 'downlink_min_d')} | {dpct(d['sensors_only_vs_v1'], 'mtq_energy_J')} |
| both (v2 protocol) | {dpct(d['v2_protocol_vs_v1'], 'decay_nominal')} | {dpct(d['v2_protocol_vs_v1'], 'band_pct')} | {dpct(d['v2_protocol_vs_v1'], 'downlink_min_d')} | {dpct(d['v2_protocol_vs_v1'], 'mtq_energy_J')} |

Reading: if the downlink/band/mtq hit lives in **C** not **B**, v2 is a noisier
comparator, not a stronger controller. The 0.2→0.5 slew tweak is then a small
weight move on top of a protocol change.

## Check 3 — MC lifetime (separate `--stage mc`)

v1 MC: 8000 steps (~28 d), truth, slew 0.2 → 252/512 reached, 17.9 d mean.
v2 MC: 20000 steps (~69 d), sensors, slew 0.5 → 402/512 reached, 30.0 d mean.

A 12-day lifetime jump from a 0.3 slew-weight bump is not credible. Check 3
re-runs v1-truth at both caps. If 8000→20000 on the SAME policy reproduces
~18 d → ~30 d, the "win" is the terminator.

## v3 note

`mpc_v3.yaml` froze the **same knobs as v2** (score20 = 1.000 on probe20).
The 10-orbit "better decay" rows did not survive the 20-orbit confirm.
v3 is not a new controller.

## What v12 should do

Score every RL run against **two** columns: published v1 (truth, probe20) and
sensor-on v1/v2 (cell C or D, once Check 1 passes). Do not hide a downlink
loss behind a 69-day MC cap.
"""
    (OUT / "GATE.md").write_text(md)


def _plot_factorial(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["A_slew02_truth", "B_slew05_truth", "C_slew02_sensors", "D_slew05_sensors"]
    labs = ["A truth\nslew 0.2", "B truth\nslew 0.5", "C sensors\nslew 0.2",
            "D sensors\nslew 0.5"]
    metrics = [("decay_nominal", "decay km/d", True),
               ("band_pct", "band %", False),
               ("downlink_min_d", "downlink min/d", False),
               ("mtq_energy_J", "mtq J", True)]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6))
    cols = ["#8a8880", "#2a78d6", "#eb6834", "#1baf7a"]
    for ax, (k, title, low) in zip(axes, metrics):
        ys = [results[c]["agg"][k] for c in keys]
        ax.bar(range(4), ys, color=cols)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(title, fontsize=11)
        if k in V1_REF:
            ax.axhline(V1_REF[k], color="#e34948", ls="--", lw=0.8)
    fig.suptitle("Check 2: sensors × slew on probe20 (red dashed = published v1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_check2_factorial.png", dpi=150)
    fig.savefig(OUT / "fig_check2_factorial.pdf")
    plt.close(fig)
    print(f"[check] wrote {OUT / 'fig_check2_factorial.png'}", flush=True)


def stage_mc(workers, n_runs=64):
    """Same v1-truth policy, two caps. Isolates the terminator."""
    from arlamx_v2.mc_decay import run_study
    OUT.mkdir(parents=True, exist_ok=True)
    pending = OUT / "v1_truth_pending.json"
    cfg = _v1_cfg()
    pending.write_text(json.dumps({**cfg, "sensors": {"enabled": False}}, indent=2))
    # mpcjson path uses noisy from yaml sensors.enabled — False here.
    spec = f"mpcjson:{pending}"
    out = {}
    for cap, tag in ((8000, "cap28d"), (20000, "cap69d")):
        dest = OUT / f"mc_v1_{tag}.json"
        print(f"[check] MC v1-truth n={n_runs} max_steps={cap}", flush=True)
        rec = run_study([spec], n_runs, variant="v10", workers=workers,
                        max_steps=cap, floor_km=295.0, outfile=str(dest))
        out[tag] = rec["summary"][spec]
        print(f"   reached={out[tag]['reached_floor']}/{n_runs}  "
              f"life={out[tag]['days_500_to_300_mean']:.2f}d  "
              f"dec300={out[tag]['decay_300_mean']}", flush=True)
    (OUT / "check3_mc.json").write_text(json.dumps(out, indent=2))
    a, b = out["cap28d"], out["cap69d"]
    # If lifetime jumps > 40% on the SAME policy, the v2 "win" is the cap.
    life_a = a.get("days_500_to_300_mean") or 0.0
    life_b = b.get("days_500_to_300_mean") or 0.0
    jump = (life_b / life_a - 1.0) if life_a > 0 else None
    cap_explains = bool(jump is not None and jump >= 0.25)
    verdict = {
        "cap28d": a, "cap69d": b,
        "lifetime_jump_frac": jump,
        "cap_explains_v2_lifetime": cap_explains,
        "n_runs": n_runs,
        "note": ("Same v1-truth policy at two caps. A large jump means the "
                 "v2 17.9→30 d number is the 69-day terminator, not slew 0.5."),
    }
    (OUT / "check3_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"[check] Check 3 lifetime jump={jump}  "
          f"cap_explains={cap_explains}", flush=True)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("probes", "mc", "all"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-mc", type=int, default=64)
    a = ap.parse_args()
    print(f"[check] stage={a.stage} {_now()}", flush=True)
    if a.stage in ("probes", "all"):
        stage_probes()
    if a.stage in ("mc", "all"):
        stage_mc(a.workers, a.n_mc)


if __name__ == "__main__":
    main()
