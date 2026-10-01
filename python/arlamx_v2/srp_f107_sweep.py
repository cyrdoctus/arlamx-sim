"""F10.7 sweep of an SRP-aware attitude strategy in the 500-450 km band.

Strategies evaluated at every point of a circular orbit (same sampling as
``srp_assess``: local MSIS density, co-rotating wind, analytic Sun,
cylindrical eclipse):

* ``edge``     edge-on all the time (min drag, sail normal along -h, as flown
               in the decay runs); SRP acts on it whatever it does.
* ``maxsun``   the user's two-mode rule: in eclipse edge-on; in sunlight the
               orientation that maximises the SRP push along the velocity,
               F_srp . v, ignoring drag when choosing. When no orientation
               gives a positive push (moving toward the Sun) fall back to
               edge-on. Drag is of course charged in the result.
* ``best``     in sunlight the orientation that maximises the NET power
               (F_srp + F_aero) . v; in eclipse edge-on. The ceiling for any
               attitude controller.

Each with the two SRP models of ``srp_assess`` (as flown: force along the
Sun line, Cr = 1.8; ideal specular: 2 P A cos^2 along the normal). Rates are
converted to days to descend from the top to the bottom of the band by
integrating 1/(da/dt) over altitude; a non-negative rate anywhere in the
band means the orbit is held.

Level: advanced.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import drag_extreme_axes
from arlamx_v2.aoa_decay import MASS
from arlamx_v2.decay_run import load_panels
from arlamx_v2.srp_assess import NAME, _local_normals, fibonacci_sphere, orbit_points, sweep_point

MU = cpp.MU_WGS
RE = cpp.RE_WGS
STRATS = ("edge", "maxsun_cb", "maxsun_sp", "best_cb", "best_sp")
LABEL = {
    "edge": "edge-on always (SRP as flown acting on it)",
    "maxsun_cb": "eclipse edge-on, sunlit max Sun push — SRP model as flown",
    "maxsun_sp": "eclipse edge-on, sunlit max Sun push — ideal specular sail",
    "best_cb": "eclipse edge-on, sunlit best net (SRP − drag) — as flown",
    "best_sp": "eclipse edge-on, sunlit best net (SRP − drag) — ideal specular",
}
STYLE = {
    "edge": dict(color="#2E6B4E", ls="-", lw=1.6),
    "maxsun_cb": dict(color="#d97706", ls="-", lw=2.0),
    "maxsun_sp": dict(color="#d97706", ls="--", lw=2.0),
    "best_cb": dict(color="#2a78d6", ls="-", lw=1.4),
    "best_sp": dict(color="#7b3fb3", ls="--", lw=1.4),
}


def _refined_max(nrm, area, cen, max_ax, pt, normals, mass, res, pick):
    """Max over the grid + a local refinement of the objective ``pick(res)``.
    Returns (value of the objective, aero power, srp_cb power, srp_sp power)
    at the winning orientation."""
    obj = pick(res)
    k = int(np.argmax(obj))
    loc = _local_normals(normals[k], 12.0, 5)
    r2 = sweep_point(nrm, area, cen, max_ax, pt, loc, mass)
    obj2 = pick(r2)
    k2 = int(np.argmax(obj2))
    if obj2[k2] > obj[k]:
        return float(obj2[k2]), float(r2[0][k2]), float(r2[1][k2]), float(r2[2][k2])
    return float(obj[k]), float(res[0][k]), float(res[1][k]), float(res[2][k])


def rates_at(geom, nrm, area, cen, max_ax, mass, normals, alt_km, inc_deg, jd, f107, ap, n_pts):
    a, pts = orbit_points(alt_km, inc_deg, jd, f107, ap, n_pts)
    T_orb = 2.0 * np.pi * np.sqrt(a**3 / MU)
    dt = T_orb / n_pts
    E = {s: 0.0 for s in STRATS}
    sunlit = 0.0
    for pt in pts:
        h_hat = np.cross(pt["r"], pt["v"])
        h_hat /= np.linalg.norm(h_hat)
        ref = sweep_point(nrm, area, cen, max_ax, pt, np.vstack([-h_hat]), mass)
        p_edge_aero, p_edge_cb, p_edge_sp = float(ref[0][0]), float(ref[1][0]), float(ref[2][0])
        E["edge"] += (p_edge_aero + p_edge_cb) * dt
        if pt["ecl"] <= 0.0:
            for s in STRATS[1:]:
                E[s] += p_edge_aero * dt  # eclipse: edge-on, no SRP
            continue
        sunlit += dt
        res = sweep_point(nrm, area, cen, max_ax, pt, normals, mass)
        # max Sun push, drag ignored in the choice
        v_cb, pa, pcb, _ = _refined_max(nrm, area, cen, max_ax, pt, normals, mass, res, lambda r: r[1])
        E["maxsun_cb"] += ((pa + pcb) if v_cb > 0.0 else (p_edge_aero + p_edge_cb)) * dt
        v_sp, pa, _, psp = _refined_max(nrm, area, cen, max_ax, pt, normals, mass, res, lambda r: r[2])
        E["maxsun_sp"] += ((pa + psp) if v_sp > 0.0 else (p_edge_aero + p_edge_sp)) * dt
        # best net
        v, _, _, _ = _refined_max(nrm, area, cen, max_ax, pt, normals, mass, res, lambda r: r[0] + r[1])
        E["best_cb"] += max(v, p_edge_aero + p_edge_cb) * dt
        v, _, _, _ = _refined_max(nrm, area, cen, max_ax, pt, normals, mass, res, lambda r: r[0] + r[2])
        E["best_sp"] += max(v, p_edge_aero + p_edge_sp) * dt
    conv = 2.0 * a * a / MU / 1e3 * (86400.0 / T_orb)  # J/kg per orbit -> km/day
    row = {"geom": geom, "mass_kg": mass, "f107": f107, "alt_km": alt_km, "sunlit_frac": sunlit / T_orb,
           "rho_mean": float(np.mean([p["rho"] for p in pts]))}
    for s in STRATS:
        row[f"{s}_km_day"] = E[s] * conv
    return row


def days_through_band(alts, rates):
    """Days to go from max(alts) down to min(alts) with da/dt = rates(alt)
    [km/day], piecewise linear in altitude. inf if the rate is >= 0 anywhere."""
    order = np.argsort(alts)[::-1]
    h = np.asarray(alts, float)[order]
    r = np.asarray(rates, float)[order]
    if np.any(r >= 0.0):
        return float("inf")
    days = 0.0
    for k in range(len(h) - 1):
        dh = h[k] - h[k + 1]
        r0, r1 = -r[k], -r[k + 1]  # positive descent rates
        # exact integral of dh / r(h) for linear r(h)
        days += dh / (r1 - r0) * np.log(r1 / r0) if abs(r1 - r0) > 1e-12 else dh / r0
    return float(days)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--geoms", nargs="+", default=["hex", "stl1pct"], choices=("hex", "stl1pct"))
    ap_.add_argument("--f107", type=float, nargs="+", default=list(np.arange(20.0, 201.0, 10.0)))
    ap_.add_argument("--alts", type=float, nargs="+", default=[500.0, 475.0, 450.0])
    ap_.add_argument("--ap", type=float, default=4.0)
    ap_.add_argument("--inc", type=float, default=23.0)
    ap_.add_argument("--epoch-jd", type=float, default=2461060.5)
    ap_.add_argument("--n-pts", type=int, default=36)
    ap_.add_argument("--n-dirs", type=int, default=200)
    ap_.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "results" / "decay" / "aoa45_500to300" / "srp_f107_sweep")
    ap_.add_argument("--plot-only", action="store_true")
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "run.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    csv_path = args.out / "rates.csv"
    if args.plot_only and csv_path.exists():
        with open(csv_path) as fh:
            rows = [{k: (v if k == "geom" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
    else:
        rows = []
        normals = fibonacci_sphere(args.n_dirs)
        for geom in args.geoms:
            nrm, area, cen = load_panels(geom)
            _mn, max_ax, _p = drag_extreme_axes(nrm, area)
            for f in args.f107:
                for alt in args.alts:
                    row = rates_at(geom, nrm, area, cen, max_ax, MASS[geom], normals, alt, args.inc, args.epoch_jd, f, args.ap, args.n_pts)
                    rows.append(row)
                    log(
                        f"{geom:8s} F10.7={f:5.0f} {alt:5.0f} km rho {row['rho_mean']:.2e} | km/day: "
                        + "  ".join(f"{s} {row[f'{s}_km_day']:+8.4f}" for s in STRATS)
                    )
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # days through the band per (geom, f107, strategy)
    alts = sorted({r["alt_km"] for r in rows})
    band = []
    for geom in sorted({r["geom"] for r in rows}, key=lambda g: 0 if g == "hex" else 1):
        for f in sorted({r["f107"] for r in rows}):
            rr = [r for r in rows if r["geom"] == geom and r["f107"] == f]
            rec = {"geom": geom, "f107": f}
            for s in STRATS:
                rec[f"{s}_days_{max(alts):.0f}_to_{min(alts):.0f}"] = days_through_band(
                    [r["alt_km"] for r in rr], [r[f"{s}_km_day"] for r in rr]
                )
            band.append(rec)
    with open(args.out / "days_through_band.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(band[0].keys()))
        w.writeheader()
        w.writerows(band)
    for rec in band:
        log(
            f"{rec['geom']:8s} F10.7={rec['f107']:5.0f}  days {max(alts):.0f}->{min(alts):.0f} km: "
            + "  ".join(f"{s} {rec[f'{s}_days_{max(alts):.0f}_to_{min(alts):.0f}']:8.1f}" for s in STRATS)
        )
    plot(rows, band, alts, args.out, args.ap)
    log(f"wrote {args.out}")


def plot(rows, band, alts, out_dir, ap):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geoms = sorted({r["geom"] for r in rows}, key=lambda g: 0 if g == "hex" else 1)
    top, bot = max(alts), min(alts)

    # 1. rate vs F10.7 at the top and bottom of the band
    fig, ax = plt.subplots(2, len(geoms), figsize=(6.6 * len(geoms), 8.6), squeeze=False, sharex=True)
    for j, g in enumerate(geoms):
        for i, alt in enumerate((top, bot)):
            a_ = ax[i, j]
            rr = sorted([r for r in rows if r["geom"] == g and r["alt_km"] == alt], key=lambda r: r["f107"])
            f = [r["f107"] for r in rr]
            for s in STRATS:
                a_.plot(f, [r[f"{s}_km_day"] for r in rr], label=LABEL[s], **STYLE[s])
            a_.axhline(0.0, color="#333", lw=0.8)
            a_.axvspan(0, 65, color="#ccc", alpha=0.35, lw=0)
            a_.set_yscale("symlog", linthresh=0.02)
            a_.set_ylabel(f"da/dt at {alt:.0f} km [km/day]  (symlog)")
            a_.grid(alpha=0.3, which="both")
            a_.set_title(NAME[g] if i == 0 else "", fontsize=9)
            if i == 1:
                a_.set_xlabel("F10.7 [sfu]   (grey: below the observed solar-minimum floor ~65)")
        ax[0, j].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(
        f"Semi-major-axis rate vs F10.7 (Ap = {ap:g}), circular orbit i=23°, epoch of the decay study\n"
        "positive = orbit raised; strategies: edge-on in eclipse, then either max Sun push or best net in sunlight",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "rate_vs_f107.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 2. days through the band vs F10.7
    fig, ax = plt.subplots(1, len(geoms), figsize=(6.6 * len(geoms), 4.8), squeeze=False)
    for j, g in enumerate(geoms):
        a_ = ax[0, j]
        bb = sorted([b for b in band if b["geom"] == g], key=lambda b: b["f107"])
        f = np.array([b["f107"] for b in bb])
        ymax = 0.0
        for s in STRATS:
            d = np.array([b[f"{s}_days_{top:.0f}_to_{bot:.0f}"] for b in bb])
            fin = np.isfinite(d)
            if fin.any():
                ymax = max(ymax, float(np.max(d[fin])))
        cap = ymax * 3.0 if ymax > 0 else 1e4
        for s in STRATS:
            d = np.array([b[f"{s}_days_{top:.0f}_to_{bot:.0f}"] for b in bb])
            held = ~np.isfinite(d)
            dplot = np.where(held, cap, d)
            a_.plot(f, dplot, label=LABEL[s], **STYLE[s])
            if held.any():
                a_.plot(f[held], dplot[held], "^", color=STYLE[s]["color"], ms=7, mfc="none")
        a_.axhline(cap, color="#333", lw=0.6, ls=":")
        a_.text(f[-1], cap, " held (rate ≥ 0)", va="bottom", ha="right", fontsize=8)
        a_.axvspan(0, 65, color="#ccc", alpha=0.35, lw=0)
        a_.set_yscale("log")
        a_.set_xlabel("F10.7 [sfu]")
        a_.set_ylabel(f"days from {top:.0f} km to {bot:.0f} km")
        a_.grid(alpha=0.3, which="both")
        a_.set_title(NAME[g], fontsize=9)
        a_.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Time to descend through the {top:.0f}-{bot:.0f} km band vs F10.7 (Ap = {ap:g}); triangles = orbit held", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "days_through_band_vs_f107.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
