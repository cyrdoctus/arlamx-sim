"""Two presentation animations: the sampling MPC and the trained policy, each
through the storm segment where the two behave most differently.

Records both policies on the 1-day storm scenario, then renders the storm-onset
segment with the existing 3D IPC renderer (real simplified SolarCat CAD, live
disturbance/gate/power panels).

    PYTHONPATH=python python -m arlamx_v2.anim_production [--segment storm_onset]

Outputs -> outputs/presentation/animations/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from arlamx_v2 import ipc_animation as IPC
from arlamx_v2 import showcase_runs as SR
from arlamx_v2.bench_production import (ALGOS, BUDGETS, DATA, RUNS, V7_ENV_KW,
                                        cell_label, repr_seed, run_name)
from arlamx_v2.paths import OUTPUTS

ANIM = OUTPUTS / "presentation" / "animations"
SCENARIO = "storm1d"


def _best_rl():
    """(algo, budget, seed) of the best-scoring RL cell in the bake-off."""
    bake = json.loads((DATA / "bakeoff.json").read_text())
    best, pick = None, None
    for algo in ALGOS:
        for bud in BUDGETS:
            gaps = [v["gap_index"] for k, v in bake.items()
                    if k.startswith(f"{cell_label(algo, bud)} s")]
            if not gaps:
                continue
            m = float(np.median(gaps))
            if best is None or m < best:
                best, pick = m, (algo, bud)
    algo, bud = pick
    return algo, bud, repr_seed(bake, algo, bud)


def record_pair():
    """Record the MPC and the best RL policy on the same storm scenario."""
    algo, bud, seed = _best_rl()
    name = run_name(algo, bud, seed)
    print(f"[anim] RL policy: {algo.upper()} {bud}k seed {seed}", flush=True)

    jobs = [
        ("mpc", "advisor", dict(which="mpc")),
        (f"rl_{algo}", "model", dict(algo=algo, run=name, root=str(RUNS),
                                     env_kw=dict(V7_ENV_KW))),
    ]
    for slug, kind, spec in jobs:
        print(f"[anim] recording {slug} / {SCENARIO}", flush=True)
        SR.record(slug, kind, spec, SCENARIO)
    return f"rl_{algo}", algo


def render(slug, segment):
    d = np.load(SR.OUT / f"{slug}__{SCENARIO}.npz")
    tris, tri_n = IPC.simplified_model()
    segs = IPC.find_segments(d)
    if segment not in segs:
        segment = "full" if "full" in segs else list(segs)[0]
    i0, i1 = segs[segment]
    print(f"[anim] rendering {slug} / {segment} ({i1 - i0} frames)", flush=True)
    IPC.render(d, tris, tri_n, slug, SCENARIO, segment, i0, i1)
    ANIM.mkdir(parents=True, exist_ok=True)
    stem = f"ipc_{slug}_{SCENARIO}_{segment}"
    for ext in ("mp4", "gif"):
        src = IPC.SHOW / f"{stem}.{ext}"
        if src.exists():
            shutil.copy2(src, ANIM / f"{slug}_{segment}.{ext}")
    return segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="storm_onset")
    ap.add_argument("--skip-record", action="store_true")
    a = ap.parse_args()

    if a.skip_record:
        bake_algo, _ = _best_rl()[0], None
        rl_slug = f"rl_{bake_algo}"
    else:
        rl_slug, _algo = record_pair()

    for slug in ("mpc", rl_slug):
        render(slug, a.segment)
    print(f"[anim] wrote {ANIM}", flush=True)


if __name__ == "__main__":
    main()
