"""Membrane damage (hole, tear, multi-hole): plan and apply to the panel table."""
from __future__ import annotations

import numpy as np

MEMBRANE_NZ = 0.9
R_MEMBRANE = 0.45


def plan_damage(rng, kind=None):
    kind = kind or rng.choice(["hole", "tear", "multi"])
    def pt():
        r = float(rng.uniform(0.10, R_MEMBRANE * 0.9))
        th = float(rng.uniform(0.0, 2.0 * np.pi))
        return (r * np.cos(th), r * np.sin(th))
    if kind == "hole":
        holes = [dict(c=pt(), a=float(rng.uniform(0.05, 0.08)), b=None, th=0.0)]
    elif kind == "tear":
        holes = [dict(c=pt(), a=float(rng.uniform(0.18, 0.25)),
                      b=float(rng.uniform(0.06, 0.08)),
                      th=float(rng.uniform(0.0, np.pi)))]
    else:
        holes = [dict(c=pt(), a=float(rng.uniform(0.03, 0.06)), b=None, th=0.0)
                 for _ in range(int(rng.integers(3, 6)))]
    return {"kind": kind, "holes": holes}


def _removal_fraction(cx, cy, hole):
    dx, dy = cx - hole["c"][0], cy - hole["c"][1]
    if hole["b"] is None:
        d = np.hypot(dx, dy) / hole["a"]
    else:
        ct, st = np.cos(hole["th"]), np.sin(hole["th"])
        u = (dx * ct + dy * st) / hole["a"]
        v = (-dx * st + dy * ct) / hole["b"]
        d = np.hypot(u, v)
    return float(np.clip(1.0 - d, 0.0, 1.0))


def apply_damage(normals, areas, centroids, plan):
    n = np.asarray(normals, float)
    A = np.asarray(areas, float).copy()
    c = np.asarray(centroids, float)
    kept = np.ones(len(A))
    membrane = np.abs(n[:, 2]) > MEMBRANE_NZ
    for i in np.where(membrane)[0]:
        rem = 0.0
        for hole in plan["holes"]:
            rem = max(rem, _removal_fraction(c[i, 0], c[i, 1], hole))
        kept[i] = 1.0 - rem
    A_dam = A * kept
    lost = float((A - A_dam)[membrane].sum())
    total_mem = float(A[membrane].sum())
    summary = {
        "kind": plan["kind"],
        "n_holes": len(plan["holes"]),
        "area_lost_m2": lost,
        "area_lost_frac": lost / max(total_mem, 1e-12),
        "panels_hit": int(np.sum(kept < 0.999)),
        "cp_shift_m": _cp_shift(A, A_dam, c, membrane),
    }
    return A_dam, kept, summary


def _cp_shift(A0, A1, c, membrane):
    def cp(A):
        w = A[membrane]
        if w.sum() < 1e-12:
            return np.zeros(2)
        return (c[membrane, :2] * w[:, None]).sum(0) / w.sum()
    return float(np.linalg.norm(cp(A1) - cp(A0)))
