"""Free-molecular lift and drag of a panel model versus angle of attack and
altitude, and what the best lift could buy.

For each altitude the atmosphere (MSIS 2.1, fixed F10.7 / Ap) and the
relative wind |v - omega_E x r| are averaged around one circular orbit at the
requested inclination. The sail is then swept from edge-on (alpha = 0, flow
along the min-drag body axis) to face-on (alpha = 90 deg, flow along the
max-drag axis) through the plant's own Sentman kernel (``cpp.spacecraft_aero``),
and the body-frame force is split into the along-flow part (drag) and the
perpendicular part (lift).

Usefulness of lift is scored with the Gauss variational equations for a
circular orbit (Vallado 2013 sec. 9.7): a normal acceleration a_N flipped in
sign every half orbit gives <di/dt> = (2/pi) a_N / v, while a tangential
deceleration a_T gives da/dt = 2 a_T / n. Their ratio is
di/da = (L/D) / (pi a), so the inclination bought per kilometre of altitude
spent depends only on L/D.

Sources: Sentman 1961 (GSI); Doornbos 2012 (accommodation); Vallado 2013.
Level: advanced.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import drag_extreme_axes
from arlamx_v2.atmosphere import jd_to_datetime, query_msis
from arlamx_v2.decay_run import coe_to_rv, load_panels

OMEGA_E = 7.2921150e-5  # rad/s, Earth rotation
MU = cpp.MU_WGS
RE = cpp.RE_WGS
AXIS = {"x": 0, "y": 1, "z": 2}


def orbit_mean_atmosphere(alt_km, inc_deg, jd, f107, ap, n_pts=36):
    """Arithmetic-mean rho, rho-weighted T, mean m_bar and mean |v_rel| around
    one circular orbit of the given inclination."""
    a = RE + alt_km * 1e3
    gmst = cpp.gmst_rad(jd)
    when = jd_to_datetime(jd)
    rho, T, mb, vrel = [], [], [], []
    for nu in np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False):
        r, v = coe_to_rv(a, 0.0, np.radians(inc_deg), 0.0, 0.0, nu)
        lat, lon_eci, alt_m = cpp.ecef_to_geodetic(r)
        lon_e = (lon_eci - gmst + np.pi) % (2.0 * np.pi) - np.pi
        rh, tt, mm = query_msis(alt_m / 1e3, np.degrees(lat), np.degrees(lon_e), when, f107, ap)
        rho.append(rh)
        T.append(tt)
        mb.append(mm)
        vrel.append(np.linalg.norm(v - np.cross([0.0, 0.0, OMEGA_E], r)))
    rho = np.asarray(rho)
    return (
        float(rho.mean()),
        float(np.sum(rho * np.asarray(T)) / rho.sum()),
        float(np.mean(mb)),
        float(np.mean(vrel)),
    )


def sweep(nrm, area, cen, min_ax, max_ax, vrel, rho, T, mb, alphas_deg, T_w=300.0, alpha_E=0.93):
    """Drag (along flow), lift (perpendicular) [N] and Cd, Cl for each alpha."""
    out = []
    for al in np.radians(alphas_deg):
        d = np.zeros(3)
        d[AXIS[min_ax]] = np.cos(al)
        d[AXIS[max_ax]] = np.sin(al)
        v_b = vrel * d
        a = cpp.spacecraft_aero(nrm, area, cen, v_b, rho, T, mb, T_w, alpha_E, True)
        F = np.asarray(a["force"], float)
        f_along = float(F @ d)
        F_perp = F - f_along * d
        out.append((abs(f_along), float(np.linalg.norm(F_perp)), float(a["Cd"]), float(a["Cl"]), float(a["A_ref"])))
    return np.asarray(out)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--geom", choices=("hex", "stl1pct"), default="stl1pct")
    ap_.add_argument("--mass", type=float, default=1.75)
    ap_.add_argument("--inc", type=float, default=45.0)
    ap_.add_argument("--alts", type=float, nargs="+", default=[500.0, 450.0, 400.0, 350.0, 300.0])
    ap_.add_argument("--f107", type=float, default=150.0)
    ap_.add_argument("--ap", type=float, default=4.0)
    ap_.add_argument("--epoch-jd", type=float, default=2461060.5)
    ap_.add_argument("--out", type=Path, required=True)
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nrm, area, cen = load_panels(args.geom)
    min_ax, max_ax, proj = drag_extreme_axes(nrm, area)
    print("geom", args.geom, "min_axis", min_ax, "max_axis", max_ax, "proj_m2", proj)
    alphas = np.arange(0.0, 90.5, 1.0)

    rows = []
    summary = []
    for alt in args.alts:
        rho, T, mb, vrel = orbit_mean_atmosphere(alt, args.inc, args.epoch_jd, args.f107, args.ap)
        s = sweep(nrm, area, cen, min_ax, max_ax, vrel, rho, T, mb, alphas)
        a_sma = RE + alt * 1e3
        n_rate = np.sqrt(MU / a_sma**3)
        v_circ = np.sqrt(MU / a_sma)
        drag, lift = s[:, 0], s[:, 1]
        ld = np.divide(lift, drag, out=np.zeros_like(lift), where=drag > 0)
        # Gauss (circular): <di/dt> = (2/pi) a_N / v ; da/dt = -2 a_T / n
        di_deg_day = np.degrees((2.0 / np.pi) * (lift / args.mass) / v_circ) * 86400.0
        da_km_day = -2.0 * (drag / args.mass) / n_rate / 1e3 * 86400.0
        for al, dr, li, cd, cl, aref, l_d, di, da in zip(alphas, drag, lift, s[:, 2], s[:, 3], s[:, 4], ld, di_deg_day, da_km_day):
            rows.append(
                {
                    "alt_km": alt, "alpha_deg": al, "rho": rho, "T": T, "m_bar": mb, "v_rel": vrel,
                    "drag_N": dr, "lift_N": li, "Cd": cd, "Cl": cl, "A_ref_m2": aref, "L_over_D": l_d,
                    "di_deg_per_day": di, "da_km_per_day": da,
                }
            )
        k_lmax = int(np.argmax(lift))
        k_ldmax = int(np.argmax(ld))
        summary.append(
            {
                "alt_km": alt, "rho": rho, "T": T, "v_rel": vrel,
                "drag_edge_on_N": drag[0], "drag_face_on_N": drag[-1],
                "alpha_max_lift_deg": alphas[k_lmax], "lift_max_N": lift[k_lmax], "drag_at_max_lift_N": drag[k_lmax],
                "alpha_max_LD_deg": alphas[k_ldmax], "LD_max": ld[k_ldmax], "lift_at_max_LD_N": lift[k_ldmax], "drag_at_max_LD_N": drag[k_ldmax],
                "di_deg_day_at_max_lift": di_deg_day[k_lmax], "da_km_day_at_max_lift": da_km_day[k_lmax],
                "da_km_day_edge_on": da_km_day[0], "da_km_day_face_on": da_km_day[-1],
                "deg_incl_per_km_lost_at_max_LD": np.degrees(ld[k_ldmax] / (np.pi * a_sma / 1e3)),
            }
        )
        print(
            f"alt {alt:5.0f} km  rho {rho:.3e}  T {T:5.0f}  vrel {vrel:6.0f} | "
            f"D edge {drag[0]:.3e} N  D face {drag[-1]:.3e} N | "
            f"L max {lift[k_lmax]:.3e} N @ {alphas[k_lmax]:.0f} deg (D there {drag[k_lmax]:.3e}) | "
            f"L/D max {ld[k_ldmax]:.3f} @ {alphas[k_ldmax]:.0f} deg | "
            f"di {di_deg_day[k_lmax]:.4f} deg/day  da {da_km_day[k_lmax]:.2f} km/day at max lift"
        )

    with open(args.out / "lift_drag_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "lift_drag_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    plot(args, alphas, rows, summary, min_ax, max_ax)


def plot(args, alphas, rows, summary, min_ax, max_ax):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alts = [s["alt_km"] for s in summary]
    cmap = plt.get_cmap("viridis")
    cols = {a: cmap(i / max(1, len(alts) - 1)) for i, a in enumerate(alts)}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for alt in alts:
        rr = [r for r in rows if r["alt_km"] == alt]
        al = np.array([r["alpha_deg"] for r in rr])
        dr = np.array([r["drag_N"] for r in rr])
        li = np.array([r["lift_N"] for r in rr])
        ld = np.array([r["L_over_D"] for r in rr])
        ax[0, 0].plot(al, dr * 1e3, color=cols[alt], label=f"{alt:.0f} km")
        ax[0, 1].plot(al, li * 1e3, color=cols[alt], label=f"{alt:.0f} km")
        ax[1, 0].plot(al, ld, color=cols[alt], label=f"{alt:.0f} km")
    ax[0, 0].set_ylabel("drag (along flow) [mN]")
    ax[0, 1].set_ylabel("lift (perpendicular to flow) [mN]")
    ax[1, 0].set_ylabel("L / D")
    for a_ in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a_.set_xlabel(f"angle of attack alpha [deg]  (0 = edge-on, body {min_ax};  90 = face-on, body {max_ax})")
        a_.set_xlim(0, 90)
        a_.grid(alpha=0.3)
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_yscale("log")
    # lift is exactly zero at 0 and 90 deg by symmetry; clip the log axis so
    # those two points do not swallow the plot
    lift_top = max(r["lift_N"] for r in rows) * 1e3
    ax[0, 1].set_ylim(lift_top * 1e-3, lift_top * 2.0)
    ax[0, 0].legend(frameon=False, title="altitude")
    # panel 4: vs altitude
    a4 = ax[1, 1]
    ah = np.array(alts)
    a4.plot(ah, [s["drag_edge_on_N"] * 1e3 for s in summary], "o-", color="#2E6B4E", label="drag, edge-on (min)")
    a4.plot(ah, [s["drag_face_on_N"] * 1e3 for s in summary], "s-", color="#A33B2B", label="drag, face-on (max)")
    a4.plot(ah, [s["lift_max_N"] * 1e3 for s in summary], "^-", color="#2a78d6", label="max lift (best alpha)")
    a4.plot(ah, [s["drag_at_max_lift_N"] * 1e3 for s in summary], "^--", color="#2a78d6", alpha=0.6, label="drag at that alpha")
    a4.set_yscale("log")
    a4.set_xlabel("altitude [km]")
    a4.set_ylabel("force [mN]")
    a4.grid(alpha=0.3)
    a4.legend(frameon=False, fontsize=8)
    a4.invert_xaxis()
    fig.suptitle(
        f"{args.geom} panel model, {args.mass:g} kg, i={args.inc:g} deg, F10.7={args.f107:g}, Ap={args.ap:g}: "
        "free-molecular lift and drag (ARLAMX v2.1 Sentman)"
    )
    fig.tight_layout()
    fig.savefig(args.out / "lift_drag_vs_alpha_alt.png", dpi=150, facecolor="white")
    plt.close(fig)

    # What lift buys: inclination per day at max lift vs altitude lost per day
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ah, [s["di_deg_day_at_max_lift"] for s in summary], "^-", color="#2a78d6")
    ax[0].set_ylabel("inclination change [deg/day] at max-lift alpha")
    ax[0].set_xlabel("altitude [km]")
    ax[0].set_yscale("log")
    ax[1].plot(ah, [-s["da_km_day_edge_on"] for s in summary], "o-", color="#2E6B4E", label="edge-on (min drag)")
    ax[1].plot(ah, [-s["da_km_day_at_max_lift"] for s in summary], "^-", color="#2a78d6", label="at max-lift alpha")
    ax[1].plot(ah, [-s["da_km_day_face_on"] for s in summary], "s-", color="#A33B2B", label="face-on (max drag)")
    ax[1].set_ylabel("altitude loss [km/day]")
    ax[1].set_xlabel("altitude [km]")
    ax[1].set_yscale("log")
    ax[1].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.invert_xaxis()
    ld = summary[0]["LD_max"]
    fig.suptitle(
        f"What lift buys ({args.geom}, {args.mass:g} kg): L/D max = {ld:.2f}  ->  "
        f"{summary[0]['deg_incl_per_km_lost_at_max_LD']:.4f} deg of inclination per km of altitude spent"
    )
    fig.tight_layout()
    fig.savefig(args.out / "lift_utility_vs_alt.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", args.out / "lift_drag_vs_alpha_alt.png", args.out / "lift_utility_vs_alt.png")


if __name__ == "__main__":
    main()
