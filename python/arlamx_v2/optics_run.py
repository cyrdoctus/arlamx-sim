"""Sweep sail optical presets: face-on / 45 deg force and a 1-orbit energy.

Sources: Montenbruck & Gill 2000 §3.4; Vallado 2013 §8.6.4;
McInnes 1999 (sail coatings). Level: advanced.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.decay_run import coe_to_rv, load_panels
from arlamx_v2.paths import resolve_ggm
from arlamx_v2.sail_optics import PRESETS, apply_optics


def plate_force(name, n, A, c, sun):
    spec = PRESETS[name]
    if spec["mode"] == "cannonball":
        return np.asarray(cpp.panel_srp_force(n, A, c, sun, 1.0, cpp.P_SRP_1AU, spec["Cr"]))
    return np.asarray(
        cpp.panel_srp_optical(n, A, c, sun, spec["ca"], spec["cs"], spec["cd"], 1.0, cpp.P_SRP_1AU)[0]
    )


def sweep_attitudes(geom="hex"):
    n, A, c = load_panels(geom)
    sun = np.array([0.0, 0.0, 1.0])
    rows = []
    for name in PRESETS:
        for label, nrm in (
            ("face", np.array([[0.0, 0.0, 1.0]])),
            ("tilt45", np.array([[0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])),
        ):
            nn = nrm if name != "use_full" else n
            aa = np.array([1.0]) if nn.shape[0] == 1 else A
            cc = np.zeros((nn.shape[0], 3))
            if nn.shape[0] == 1:
                F = plate_force(name, nn, aa, cc, sun)
            else:
                F = plate_force(name, n, A, c, sun)
            mag = float(np.linalg.norm(F))
            along = float(-np.dot(F, sun) / max(mag, 1e-30))
            rows.append(
                {
                    "preset": name,
                    "attitude": label,
                    "Fx": F[0],
                    "Fy": F[1],
                    "Fz": F[2],
                    "F_N": mag,
                    "along_minus_sun": along,
                    "ca": PRESETS[name]["ca"],
                    "cs": PRESETS[name]["cs"],
                    "cd": PRESETS[name]["cd"],
                }
            )
    return rows


def one_orbit_energy(name, geom="hex"):
    n, A, c = load_panels(geom)
    p = cpp.SimParams()
    p.mass = 0.625
    p.dt_s = 5.0
    p.advisor_step_s = 5.0
    p.rk4_step_s = 5.0
    p.sh_degree = 4
    p.use_panel_srp = True
    p.corotating = True
    p.ggm_path = resolve_ggm()
    p.mode = "prescribed"
    p.max_slew_rad = np.pi
    apply_optics(p, name)
    sim = cpp.Simulator(p)
    sim.set_mode("prescribed")
    sim.set_panels(n, A, c)
    sim.set_atmosphere(1.3e-12, 1200.0, 2.656e-26)
    r0, v0 = coe_to_rv(cpp.RE_WGS + 500e3, 0.001, np.radians(23.0), 0.0, 0.0, 0.0)
    sim.reset(r0, v0, np.zeros(3), np.zeros(3))
    sma0 = None
    de_a = de_l = 0.0
    q = np.array([1.0, 0.0, 0.0, 0.0])
    nstep = int(5600.0 / 5.0)
    for _ in range(nstep):
        out = sim.step(q)
        if sma0 is None:
            sma0 = float(out["sma_m"])
        de_a += float(out["dE_actual"])
        de_l += float(out.get("dE_lift", 0.0))
    return {
        "preset": name,
        "dsma_m": float(out["sma_m"]) - sma0,
        "dE": de_a,
        "dE_lift": de_l,
    }


def plot_forces(rows, png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["preset"] for r in rows if r["attitude"] == "face"]
    face = [r["F_N"] for r in rows if r["attitude"] == "face"]
    tilt = [r["F_N"] for r in rows if r["attitude"] == "tilt45"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(x - 0.18, face, 0.36, label="face-on 1 m2", color="#3D5A80")
    ax.bar(x + 0.18, tilt, 0.36, label="45 deg 1 m2", color="#A33B2B")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("SRP |F| [N]")
    ax.set_title("Sail optical presets  P = 4.56e-6 Pa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom", default="hex")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "results" / "optics",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = sweep_attitudes(args.geom)
    with open(args.out / "optics_forces.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_forces(rows, args.out / "optics_forces.png")
    e_rows = []
    for name in PRESETS:
        print("orbit", name, flush=True)
        e_rows.append(one_orbit_energy(name, args.geom))
    with open(args.out / "optics_orbit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(e_rows[0].keys()))
        w.writeheader()
        w.writerows(e_rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
