"""Six-attitude + 1° AoA projected-area check of a simplified plate set vs the STL.

Attitudes (flow direction d in the body frame; ram area = Σ A max(0, n·d)):

1. side_flat   — in-plane, largest side silhouette
                 (one tube facing the flow, two hex vertices, two angled sides)
2. side_vertex — in-plane, smallest side silhouette
                 (a hex vertex on the centreline, two angled tubes)
3. max_drag    — +Z and −Z face-on (sail normal); counted as one check
4. tilt_x_45   — 45° rotation about body X (Z = sail normal)
5. tilt_y_45   — 45° rotation about body Y
6. aoa_1deg    — from the min-drag in-plane direction toward face-on by 1°

Quality toggle: 1pct (preferred) / 5pct / 10pct (crude).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from arlamx_v2.geometry import (
    QUALITY,
    load_mesh,
    projected_area,
    simplify_sides,
)
from arlamx_v2.paths import SOLARCAT_STL

# Default is paths.SOLARCAT_STL (repo data/ first). A missing file fails
# in run_solarcat; do not silently load a V1.7 ghost.
SOLAR_STL = SOLARCAT_STL


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def sail_normal(n, A):
    """Dominant face normal (area-weighted, sign chosen so n_z ≥ 0 after a flip)."""
    n = np.asarray(n, float)
    A = np.asarray(A, float)
    # Bin to 2 decimals so +Z membrane votes together.
    keys = np.round(n, 2)
    acc = {}
    for i, k in enumerate(map(tuple, keys)):
        acc[k] = acc.get(k, 0.0) + float(A[i])
    best = max(acc, key=acc.get)
    n0 = np.array(best, float)
    n0 = n0 / max(np.linalg.norm(n0), 1e-15)
    if n0[2] < 0:
        n0 = -n0
    return n0


def inplane_extrema(n, A, n_sail, steps=360):
    """Largest and smallest ram area among in-plane flow directions."""
    # Build an in-plane basis.
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_sail[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_sail, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(n_sail, u)
    d_max, a_max = u.copy(), -1.0
    d_min, a_min = u.copy(), 1e300
    for k in range(steps):
        th = 2.0 * np.pi * k / steps
        d = np.cos(th) * u + np.sin(th) * v
        a = projected_area(n, A, d)
        if a > a_max:
            a_max, d_max = a, d.copy()
        if a < a_min:
            a_min, d_min = a, d.copy()
    return d_max, d_min


def attitude_set(n, A):
    """Named flow directions for the six checks."""
    nz = sail_normal(n, A)
    d_flat, d_vtx = inplane_extrema(n, A, nz)
    return {
        "side_flat": d_flat,
        "side_vertex": d_vtx,
        "max_drag_plus_z": nz,
        "max_drag_minus_z": -nz,
        "tilt_x_45": _rot_x(45.0).T @ nz,
        "tilt_y_45": _rot_y(45.0).T @ nz,
        "min_drag": d_vtx,
    }


def aoa_dir(d_min, n_sail, deg):
    a = np.radians(deg)
    d = np.cos(a) * d_min + np.sin(a) * n_sail
    return d / np.linalg.norm(d)


def compare_table(n0, A0, n1, A1, dirs):
    rows = []
    for name, d in dirs.items():
        a0 = projected_area(n0, A0, d)
        a1 = projected_area(n1, A1, d)
        rel = abs(a1 - a0) / max(a0, 1e-12)
        rows.append({"name": name, "A_stl": a0, "A_simp": a1, "rel_err": rel})
    return rows


def max_drag_pair_err(n0, A0, n1, A1, dirs):
    """+Z/−Z counted as one check: worse of the two relative errors."""
    e_p = abs(
        projected_area(n1, A1, dirs["max_drag_plus_z"])
        - projected_area(n0, A0, dirs["max_drag_plus_z"])
    ) / max(projected_area(n0, A0, dirs["max_drag_plus_z"]), 1e-12)
    e_m = abs(
        projected_area(n1, A1, dirs["max_drag_minus_z"])
        - projected_area(n0, A0, dirs["max_drag_minus_z"])
    ) / max(projected_area(n0, A0, dirs["max_drag_minus_z"]), 1e-12)
    return max(e_p, e_m)


def run_solarcat(quality="1pct", stl=SOLAR_STL, units="mm"):
    stl_path = Path(stl)
    if not stl_path.is_file():
        raise FileNotFoundError(
            f"SolarCat STL not found: {stl_path}. "
            "Expected paths.SOLARCAT_STL (repo data/SolarCat_Assembly.STL)."
        )
    V, F = load_mesh(stl_path, units=units)
    n0, A0 = _raw_panels(V, F)
    n1, A1, _c, rep = simplify_sides(V, F, quality=quality)
    dirs = attitude_set(n0, A0)
    named = {
        "1_side_flat": dirs["side_flat"],
        "2_side_vertex": dirs["side_vertex"],
        "4_tilt_x_45": dirs["tilt_x_45"],
        "5_tilt_y_45": dirs["tilt_y_45"],
    }
    rows = compare_table(n0, A0, n1, A1, named)
    rows.append(
        {
            "name": "3_max_drag_pm_z",
            "A_stl": 0.5
            * (
                projected_area(n0, A0, dirs["max_drag_plus_z"])
                + projected_area(n0, A0, dirs["max_drag_minus_z"])
            ),
            "A_simp": 0.5
            * (
                projected_area(n1, A1, dirs["max_drag_plus_z"])
                + projected_area(n1, A1, dirs["max_drag_minus_z"])
            ),
            "rel_err": max_drag_pair_err(n0, A0, n1, A1, dirs),
        }
    )
    # 6: 0° min-drag → 1° AoA growth
    d0 = aoa_dir(dirs["min_drag"], dirs["max_drag_plus_z"], 0.0)
    d1 = aoa_dir(dirs["min_drag"], dirs["max_drag_plus_z"], 1.0)
    g_stl = projected_area(n0, A0, d1) - projected_area(n0, A0, d0)
    g_sim = projected_area(n1, A1, d1) - projected_area(n1, A1, d0)
    grow_rel = abs(g_sim - g_stl) / max(abs(g_stl), 1e-12)
    rows.append(
        {
            "name": "6_aoa_0_to_1deg_growth",
            "A_stl": g_stl,
            "A_simp": g_sim,
            "rel_err": grow_rel,
        }
    )
    return {
        "quality": quality,
        "tol": QUALITY[quality]["tol"],
        "report": rep,
        "rows": rows,
        "wetted_rel": rep.area_rel_err,
        "dirs": dirs,
        "n0": n0,
        "A0": A0,
        "n1": n1,
        "A1": A1,
    }


def _raw_panels(V, F):
    from arlamx_v2.geometry import _face_normals_areas

    n, A = _face_normals_areas(V, F)
    ok = A > 1e-15
    return n[ok], A[ok]


def write_outputs(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = result["quality"]
    csv_path = out_dir / f"attitude_check_{q}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "A_stl", "A_simp", "rel_err"])
        w.writeheader()
        for row in result["rows"]:
            w.writerow(
                {
                    "name": row["name"],
                    "A_stl": f"{row['A_stl']:.6e}",
                    "A_simp": f"{row['A_simp']:.6e}",
                    "rel_err": f"{row['rel_err']:.6e}",
                }
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["name"] for r in result["rows"]]
        stl = [r["A_stl"] for r in result["rows"]]
        smp = [r["A_simp"] for r in result["rows"]]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(10.5, 4.2))
        ax.bar(x - 0.18, stl, 0.36, label="STL", color="#8AA4B5")
        ax.bar(x + 0.18, smp, 0.36, label=f"simplified ({q})", color="#C46B2C")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Ram area [m²]  (growth for check 6)")
        ax.set_title(
            f"SolarCat projected area  quality={q}  "
            f"wetted rel={result['wetted_rel']:.3%}  "
            f"budget={result['tol']:.0%}"
        )
        ax.legend()
        fig.tight_layout()
        png = out_dir / f"attitude_check_{q}.png"
        fig.savefig(png, dpi=150, facecolor="white")
        plt.close(fig)
    except Exception:
        png = None
    return csv_path, png


def all_within_budget(result):
    tol = result["tol"]
    if result["wetted_rel"] > tol:
        return False
    return all(r["rel_err"] <= tol for r in result["rows"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=list(QUALITY), default="1pct")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "results" / "geometry",
    )
    args = ap.parse_args()
    res = run_solarcat(args.quality)
    csv_p, png = write_outputs(res, args.out)
    print(f"quality={args.quality}  wetted_rel={res['wetted_rel']:.4%}")
    for r in res["rows"]:
        flag = "OK" if r["rel_err"] <= res["tol"] else "FAIL"
        print(f"  {r['name']:28s}  STL={r['A_stl']:.5f}  simp={r['A_simp']:.5f}  "
              f"err={r['rel_err']:.3%}  {flag}")
    print("wrote", csv_p, png)
    if not all_within_budget(res):
        raise SystemExit(1)
