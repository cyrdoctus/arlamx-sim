"""Can solar radiation pressure raise (or hold) the orbit of the sail against
drag if the attitude is chosen optimally at every point of the orbit?

For each altitude a circular orbit (i, RAAN, epoch as in the decay studies)
is sampled at ``n_pts`` argument-of-latitude points with the local MSIS 2.1
density, the co-rotating relative wind, the analytic Sun direction and the
cylindrical eclipse. At every point the sail normal is swept over a
Fibonacci grid on the sphere; for each orientation the plant's own kernels
give the aero force (Sentman) and the SRP force, and the along-track power
(F_aero + F_srp) . v / m is evaluated. The best orientation per point is the
one an ideal attitude controller would fly; integrating that maximum around
the orbit gives the best achievable energy change per orbit. If it is
positive the orbit can be raised; if it is negative but smaller in magnitude
than the edge-on loss, SRP can only slow the decay.

Two SRP models: the one the decay runs used (``panel_srp_force``: force along
the anti-Sun direction, Cr = 1.8, per lit panel) and an ideal specular sail
(``panel_srp_optical`` with cs = 1: force 2 P A cos^2 along minus the normal,
the upper bound for a flat reflective membrane).

Sources: McInnes 1999 (solar sailing), Vallado 2013 sec. 8.6 / 9.7,
Sentman 1961. Level: advanced.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import OMEGA_EARTH, dcm_bn_axis_along, drag_extreme_axes
from arlamx_v2.aoa_decay import MASS
from arlamx_v2.atmosphere import jd_to_datetime, query_msis
from arlamx_v2.decay_run import coe_to_rv, geodetic_from_eci, load_panels

MU = cpp.MU_WGS
RE = cpp.RE_WGS
P0 = cpp.P_SRP_1AU
CR = 1.8
COLOR = {"hex": "#7f8c8d", "stl1pct": "#c9a227"}
NAME = {"hex": "original hexagon (72-plate .geom) 0.72 kg", "stl1pct": "six-petal STL (1 % simplified) 1.75 kg"}


def fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5**0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def _local_normals(n0, half_deg, n):
    """n x n directions within +-half_deg of n0 (two tilt angles)."""
    n0 = n0 / np.linalg.norm(n0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n0, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)
    out = []
    for t1 in np.radians(np.linspace(-half_deg, half_deg, n)):
        for t2 in np.radians(np.linspace(-half_deg, half_deg, n)):
            v = n0 + np.tan(t1) * e1 + np.tan(t2) * e2
            out.append(v / np.linalg.norm(v))
    return np.array(out)


def orbit_points(alt_km, inc_deg, jd, f107, ap, n_pts):
    a = RE + alt_km * 1e3
    sun = np.asarray(cpp.sun_unit_analytic(jd), float)
    when = jd_to_datetime(jd)
    pts = []
    for u in np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False):
        r, v = coe_to_rv(a, 0.0, np.radians(inc_deg), 0.0, 0.0, u)
        lat, lon_e, alt_m, _g = geodetic_from_eci(r, jd)
        rho, T, mb = query_msis(alt_m / 1e3, np.degrees(lat), np.degrees(lon_e), when, f107, ap)
        v_rel = v - np.cross([0.0, 0.0, OMEGA_EARTH], r)
        ecl = float(cpp.eclipse_cylindrical(r, sun, RE))
        pts.append(dict(u=u, r=r, v=v, v_rel=v_rel, rho=float(rho), T=float(T), mb=float(mb), ecl=ecl, sun=sun))
    return a, pts


def sweep_point(nrm, area, cen, max_ax, pt, normals, mass, cs_ideal=1.0):
    """Power [W/kg] for every candidate sail normal: aero, SRP (as flown),
    SRP (ideal specular). Returns arrays (K,) of aero power, srp_cb power,
    srp_sp power and the drag / srp force magnitudes."""
    v = pt["v"]
    vn = float(np.linalg.norm(v))
    vhat = v / vn
    K = len(normals)
    p_aero = np.empty(K)
    p_cb = np.empty(K)
    p_sp = np.empty(K)
    f_aero = np.empty(K)
    f_cb = np.empty(K)
    f_sp = np.empty(K)
    for k, n_hat in enumerate(normals):
        c_bn = dcm_bn_axis_along(n_hat, max_ax, helper_n=vhat)
        v_b_gas = -(c_bn @ pt["v_rel"])
        a = cpp.spacecraft_aero(nrm, area, cen, v_b_gas, pt["rho"], pt["T"], pt["mb"], 300.0, 0.93, True)
        fa = c_bn.T @ np.asarray(a["force"], float)
        sun_b = c_bn @ pt["sun"]
        fcb = c_bn.T @ np.asarray(cpp.panel_srp_force(nrm, area, cen, sun_b, pt["ecl"], P0, CR), float)
        fsp = c_bn.T @ np.asarray(cpp.panel_srp_optical(nrm, area, cen, sun_b, 0.0, cs_ideal, 0.0, pt["ecl"], P0)[0], float)
        p_aero[k] = float(fa @ v) / mass
        p_cb[k] = float(fcb @ v) / mass
        p_sp[k] = float(fsp @ v) / mass
        f_aero[k] = float(np.linalg.norm(fa))
        f_cb[k] = float(np.linalg.norm(fcb))
        f_sp[k] = float(np.linalg.norm(fsp))
    return p_aero, p_cb, p_sp, f_aero, f_cb, f_sp


def assess(geom, alts, inc_deg, jd, f107, ap, n_pts, n_dirs, log=print):
    nrm, area, cen = load_panels(geom)
    min_ax, max_ax, _proj = drag_extreme_axes(nrm, area)
    mass = MASS[geom]
    normals = fibonacci_sphere(n_dirs)
    rows = []
    for alt in alts:
        a, pts = orbit_points(alt, inc_deg, jd, f107, ap, n_pts)
        T_orb = 2.0 * np.pi * np.sqrt(a**3 / MU)
        dt = T_orb / n_pts
        acc = dict(E_edge=0.0, E_face=0.0, E_edge_noSRP=0.0, E_opt_cb=0.0, E_opt_sp=0.0, E_opt_aero_only=0.0,
                   E_srp_only_cb_best=0.0, E_srp_only_sp_best=0.0, sunlit=0.0,
                   D_edge=0.0, D_face=0.0, S_cb_max=0.0, S_sp_max=0.0, rho=0.0)
        for pt in pts:
            res = sweep_point(nrm, area, cen, max_ax, pt, normals, mass)
            # refine each objective locally around the best grid direction
            # (the grid is ~10 deg coarse), and evaluate the flown references
            # (edge-on: normal along -h as in the decay runs; face-on: along
            # the relative wind) at their exact directions
            best = {}
            for name, obj in (("cb", res[0] + res[1]), ("sp", res[0] + res[2]), ("aero", res[0]),
                              ("cb_only", res[1]), ("sp_only", res[2])):
                k = int(np.argmax(obj))
                loc = _local_normals(normals[k], 12.0, 5)
                r2 = sweep_point(nrm, area, cen, max_ax, pt, loc, mass)
                obj2 = {"cb": r2[0] + r2[1], "sp": r2[0] + r2[2], "aero": r2[0], "cb_only": r2[1], "sp_only": r2[2]}[name]
                best[name] = max(float(obj[k]), float(np.max(obj2)))
            h_hat = np.cross(pt["r"], pt["v"])
            h_hat /= np.linalg.norm(h_hat)
            vr = pt["v_rel"] / np.linalg.norm(pt["v_rel"])
            ref = sweep_point(nrm, area, cen, max_ax, pt, np.vstack([-h_hat, vr]), mass)
            acc["E_edge"] += (ref[0][0] + ref[1][0]) * dt
            acc["E_edge_noSRP"] += ref[0][0] * dt
            acc["E_face"] += (ref[0][1] + ref[1][1]) * dt
            acc["E_opt_cb"] += best["cb"] * dt
            acc["E_opt_sp"] += best["sp"] * dt
            acc["E_opt_aero_only"] += best["aero"] * dt
            acc["E_srp_only_cb_best"] += best["cb_only"] * dt  # SRP push with drag ignored
            acc["E_srp_only_sp_best"] += best["sp_only"] * dt
            acc["sunlit"] += pt["ecl"] * dt
            acc["D_edge"] += ref[3][0] * dt
            acc["D_face"] += ref[3][1] * dt
            acc["S_cb_max"] += float(np.max(res[4])) * dt
            acc["S_sp_max"] += float(np.max(res[5])) * dt
            acc["rho"] += pt["rho"] * dt
        row = {"geom": geom, "mass_kg": mass, "f107": f107, "alt_km": alt, "T_orbit_s": T_orb,
               "rho_mean": acc["rho"] / T_orb, "sunlit_frac": acc["sunlit"] / T_orb,
               "drag_edge_on_uN": acc["D_edge"] / T_orb * 1e6, "drag_face_on_uN": acc["D_face"] / T_orb * 1e6,
               "srp_asflown_max_uN": acc["S_cb_max"] / T_orb * 1e6, "srp_ideal_max_uN": acc["S_sp_max"] / T_orb * 1e6}
        conv = 2.0 * a * a / MU / 1e3 * (86400.0 / T_orb)  # J/kg per orbit -> km/day
        for key in ("E_edge", "E_edge_noSRP", "E_face", "E_opt_cb", "E_opt_sp", "E_opt_aero_only",
                    "E_srp_only_cb_best", "E_srp_only_sp_best"):
            row[key + "_J_kg_orbit"] = acc[key]
            row[key + "_km_day"] = acc[key] * conv
        rows.append(row)
        log(
            f"{geom:8s} F10.7={f107:4.0f} {alt:5.0f} km  rho {row['rho_mean']:.2e}  "
            f"drag edge {row['drag_edge_on_uN']:6.2f} face {row['drag_face_on_uN']:7.2f} uN | SRP max as-flown {row['srp_asflown_max_uN']:5.2f} ideal {row['srp_ideal_max_uN']:5.2f} uN | "
            f"da/dt km/day: edge-on {row['E_edge_km_day']:+8.3f}  best(as-flown SRP) {row['E_opt_cb_km_day']:+8.3f}  "
            f"best(ideal sail) {row['E_opt_sp_km_day']:+8.3f}  SRP-only ceiling {row['E_srp_only_sp_best_km_day']:+7.3f}"
        )
    return rows


def plot(rows, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geoms = sorted({r["geom"] for r in rows}, key=lambda g: 0 if g == "hex" else 1)
    f107s = sorted({r["f107"] for r in rows}, reverse=True)
    ls = {f107s[0]: "-"}
    if len(f107s) > 1:
        ls[f107s[1]] = "--"

    # figure 1: forces vs altitude
    fig, ax = plt.subplots(1, len(geoms), figsize=(6.4 * len(geoms), 4.8), squeeze=False)
    for j, g in enumerate(geoms):
        a_ = ax[0, j]
        for f in f107s:
            rr = sorted([r for r in rows if r["geom"] == g and r["f107"] == f], key=lambda r: r["alt_km"])
            alt = [r["alt_km"] for r in rr]
            a_.plot(alt, [r["drag_face_on_uN"] for r in rr], color="#A33B2B", ls=ls[f], label=f"drag face-on, F10.7={f:g}")
            a_.plot(alt, [r["drag_edge_on_uN"] for r in rr], color="#2E6B4E", ls=ls[f], label=f"drag edge-on, F10.7={f:g}")
            if f == f107s[0]:
                a_.plot(alt, [r["srp_asflown_max_uN"] for r in rr], color="#2a78d6", ls="-", lw=2, label="SRP max, model as flown (Cr 1.8)")
                a_.plot(alt, [r["srp_ideal_max_uN"] for r in rr], color="#2a78d6", ls=":", lw=2, label="SRP max, ideal specular sail")
        a_.set_yscale("log")
        a_.set_xlabel("altitude [km]")
        a_.set_ylabel("orbit-mean force [µN]")
        a_.set_title(NAME[g], fontsize=9)
        a_.grid(alpha=0.3, which="both")
        a_.legend(frameon=False, fontsize=7)
    fig.suptitle("SRP against drag: force magnitudes on a circular orbit, i=23°, epoch of the decay study", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "srp_vs_drag_force.png", dpi=150, facecolor="white")
    plt.close(fig)

    # figure 2: best achievable altitude rate
    fig, ax = plt.subplots(1, len(geoms), figsize=(6.4 * len(geoms), 4.8), squeeze=False)
    for j, g in enumerate(geoms):
        a_ = ax[0, j]
        for f in f107s:
            rr = sorted([r for r in rows if r["geom"] == g and r["f107"] == f], key=lambda r: r["alt_km"])
            alt = [r["alt_km"] for r in rr]
            a_.plot(alt, [r["E_edge_km_day"] for r in rr], color="#2E6B4E", ls=ls[f], label=f"edge-on as flown (SRP on), F10.7={f:g}")
            a_.plot(alt, [r["E_opt_cb_km_day"] for r in rr], color="#2a78d6", ls=ls[f], lw=2, label=f"best attitude, SRP model as flown, F10.7={f:g}")
            a_.plot(alt, [r["E_opt_sp_km_day"] for r in rr], color="#7b3fb3", ls=ls[f], lw=2, label=f"best attitude, ideal specular sail, F10.7={f:g}")
            if f == f107s[0]:
                a_.plot(alt, [r["E_srp_only_sp_best_km_day"] for r in rr], color="#7b3fb3", ls=":", lw=1, label="ideal sail, drag ignored (ceiling)")
        a_.axhline(0.0, color="#333", lw=0.8)
        a_.set_yscale("symlog", linthresh=0.05)
        a_.set_xlabel("altitude [km]")
        a_.set_ylabel("semi-major-axis rate [km/day]  (symlog)")
        a_.set_title(NAME[g], fontsize=9)
        a_.grid(alpha=0.3, which="both")
        a_.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Best achievable orbit-energy rate with the attitude re-optimised at every point of the orbit\n"
        "above the zero line the orbit can be raised; between the green and blue/purple lines SRP only slows the decay",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "srp_boost_feasibility.png", dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--geoms", nargs="+", default=["hex", "stl1pct"], choices=("hex", "stl1pct"))
    ap_.add_argument("--alts", type=float, nargs="+", default=[300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 700.0, 800.0])
    ap_.add_argument("--f107", type=float, nargs="+", default=[150.0, 70.0])
    ap_.add_argument("--ap", type=float, default=4.0)
    ap_.add_argument("--inc", type=float, default=23.0)
    ap_.add_argument("--epoch-jd", type=float, default=2461060.5)
    ap_.add_argument("--n-pts", type=int, default=48)
    ap_.add_argument("--n-dirs", type=int, default=300)
    ap_.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "results" / "decay" / "aoa45_500to300" / "srp_assessment")
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "run.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    rows = []
    for geom in args.geoms:
        for f in args.f107:
            rows += assess(geom, args.alts, args.inc, args.epoch_jd, f, args.ap, args.n_pts, args.n_dirs, log=log)
    with open(args.out / "srp_assessment.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot(rows, args.out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
