"""3D comparison: input mesh vs V2.1 sealed-side plates.

Writes PNGs under outputs/results/geometry/. Does not touch V1.7.
V1.7 drew .geom facets as squares on centroids (no vertices). Here the
right-hand panel is the actual simplified triangles the plant would use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from arlamx_v2.geometry import cube_mesh, load_mesh, simplify_sides

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "results" / "geometry"

# Distinct side colours (cycled).
PALETTE = np.array(
    [
        [0.15, 0.40, 0.55],
        [0.80, 0.45, 0.20],
        [0.25, 0.55, 0.35],
        [0.65, 0.25, 0.35],
        [0.45, 0.40, 0.70],
        [0.75, 0.65, 0.20],
        [0.20, 0.55, 0.60],
        [0.55, 0.35, 0.20],
    ]
)


def hex_prism_mesh(r=0.5, h=0.2):
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    top = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(6, h / 2)], 1)
    bot = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(6, -h / 2)], 1)
    V = np.vstack([top, bot])
    F = []
    for i in range(1, 5):
        F.append([0, i, i + 1])
        F.append([6, 6 + i + 1, 6 + i])
    for i in range(6):
        j = (i + 1) % 6
        F.append([i, j, 6 + j])
        F.append([i, 6 + j, 6 + i])
    return V, np.asarray(F, int)


def _side_colors(tris):
    """Colour by rounded normal so one side = one colour."""
    cols = []
    key_to_i = {}
    for vs in tris:
        n = np.cross(vs[1] - vs[0], vs[2] - vs[0])
        n = n / max(np.linalg.norm(n), 1e-15)
        key = tuple(np.round(n, 3))
        if key not in key_to_i:
            key_to_i[key] = len(key_to_i)
        cols.append(PALETTE[key_to_i[key] % len(PALETTE)])
    return cols


def _set_equal(ax, pts, pad=0.08):
    pts = np.asarray(pts)
    c = pts.mean(axis=0)
    m = (pts.max(axis=0) - pts.min(axis=0)).max() * (0.5 + pad)
    if m < 1e-9:
        m = 1.0
    ax.set_xlim(c[0] - m, c[0] + m)
    ax.set_ylim(c[1] - m, c[1] + m)
    ax.set_zlim(c[2] - m, c[2] + m)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


def _draw_input(ax, V, F, elev, azim):
    tris = V[F]
    coll = Poly3DCollection(tris, alpha=0.35, facecolor="#8AA4B5", edgecolor="#2A3540", linewidth=0.15)
    ax.add_collection3d(coll)
    _set_equal(ax, V)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Input mesh  ({len(F)} triangles)", fontsize=10)


def _draw_simple(ax, tris, elev, azim):
    cols = _side_colors(tris)
    coll = Poly3DCollection(tris, alpha=0.85, facecolors=cols, edgecolor="#1A1F25", linewidth=0.45)
    ax.add_collection3d(coll)
    pts = np.vstack(tris)
    _set_equal(ax, pts)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Sealed sides  ({len(tris)} plates)", fontsize=10)


def plot_pair(V, F, title, out_path, views=((22, -60), (18, 25)), theta_deg=None, quality="1pct"):
    n, A, c, rep, tris = simplify_sides(
        V, F, theta_deg=theta_deg, return_tris=True, quality=quality
    )
    fig = plt.figure(figsize=(11.2, 5.0 * len(views)))
    for i, (el, az) in enumerate(views):
        ax1 = fig.add_subplot(len(views), 2, 2 * i + 1, projection="3d")
        _draw_input(ax1, V, F, el, az)
        ax2 = fig.add_subplot(len(views), 2, 2 * i + 2, projection="3d")
        _draw_simple(ax2, tris, el, az)
    fig.suptitle(
        f"{title}\n"
        f"sides={rep.n_sides}  panels={rep.n_panels}  "
        f"A_in={rep.area_in:.4f} m²  A_out={rep.area_out:.4f} m²  "
        f"unmatched={rep.unmatched_m:.2e} m",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rep, out_path


def maybe_decimate(V, F, target=12000):
    if len(F) <= target:
        return V, F, False
    try:
        import trimesh

        m = trimesh.Trimesh(vertices=V, faces=F, process=False)
        m = m.simplify_quadric_decimation(target)
        return np.asarray(m.vertices), np.asarray(m.faces, int), True
    except Exception:
        return V, F, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--catsat",
        action="store_true",
        help="also try the 2.5M-triangle CatSat STL (slow; decimates first)",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    written = []

    V, F = cube_mesh(1.0)
    rep, p = plot_pair(V, F, "Classroom cube (12 triangles → 6 sides)", args.out / "compare_cube.png")
    written.append((p, rep))

    V, F = hex_prism_mesh()
    rep, p = plot_pair(
        V, F, "Hex prism (sail-like closed shell)", args.out / "compare_hex_prism.png", theta_deg=15.0
    )
    written.append((p, rep))

    solar = Path("/mnt/storage/ARLAMX_Dev/Thesis/ARLAMX-V1.7/geometry/models/SolarCat_Assembly.STL")
    if solar.exists():
        V, F = load_mesh(solar, units="mm")
        # Centre for a readable frame (does not change areas).
        V = V - V.mean(axis=0)
        rep, p = plot_pair(
            V,
            F,
            "SolarCat_Assembly.STL  (read as data; not copied) → sealed sides",
            args.out / "compare_solarcat_stl.png",
            views=((22, -55), (70, -40)),
            theta_deg=12.0,
        )
        written.append((p, rep))

    if args.catsat:
        catsat = Path(
            "/mnt/storage/ARLAMX_Dev/Thesis/ARLAMX-V1.7/geometry/models/CATSAT_V7_Deployed_External.STL"
        )
        if catsat.exists():
            V, F = load_mesh(catsat, units="mm")
            V = V - V.mean(axis=0)
            V, F, dec = maybe_decimate(V, F, target=8000)
            tag = " (CAD decimated to ~8k for drawing only)" if dec else ""
            rep, p = plot_pair(
                V,
                F,
                f"CatSat V7 deployed STL{tag} → sealed sides",
                args.out / "compare_catsat_stl.png",
                views=((18, -60), (16, 25)),
                theta_deg=18.0,
            )
            written.append((p, rep))
    else:
        (args.out / "compare_catsat_stl.SKIP.txt").write_text(
            "CatSat deployed CAD is ~2.5M triangles. Default figures skip it.\n"
            "Re-run with: PYTHONPATH=python python -m arlamx_v2.plot_simplify --catsat\n"
        )

    print("Wrote:")
    for p, r in written:
        print(f"  {p}  sides={r.n_sides} panels={r.n_panels} unmatched={r.unmatched_m:.3e}")


if __name__ == "__main__":
    main()
