"""Prescribed min/max-drag orbital decay runs and plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import (
    MaxDragPolicy,
    MinDragPolicy,
    drag_extreme_axes,
    ram_area,
)
from arlamx_v2.atmosphere import jd_to_datetime, query_msis
from arlamx_v2.geometry import load_geom, load_mesh, simplify_sides
from arlamx_v2.paths import HEX_GEOM, SOLARCAT_STL, resolve_ggm
from arlamx_v2 import physics as physics_mod

MU = cpp.MU_WGS
RE = cpp.RE_WGS


# Classical elements -> inertial r, v (Vallado 2013, Alg. 10 COE2RV).
def coe_to_rv(a, e, inc, raan, argp, nu):
    p = a * (1.0 - e * e)
    cnu, snu = np.cos(nu), np.sin(nu)
    r_w = (p / (1.0 + e * cnu)) * np.array([cnu, snu, 0.0])
    v_w = np.sqrt(MU / p) * np.array([-snu, e + cnu, 0.0])
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    rot = np.array(
        [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ]
    )
    return rot @ r_w, rot @ v_w


def load_panels(kind):
    if kind == "hex":
        return load_geom(HEX_GEOM)
    verts, faces = load_mesh(SOLARCAT_STL, units="mm")
    nrm, area, cen, _rep = simplify_sides(verts, faces, quality="1pct")
    return nrm, area, cen


def geodetic_from_eci(r_n, jd):
    gmst = cpp.gmst_rad(jd)
    lat, lon_eci, alt_m = cpp.ecef_to_geodetic(r_n)
    lon_e = (lon_eci - gmst + np.pi) % (2.0 * np.pi) - np.pi
    return lat, lon_e, alt_m, gmst


def run_one(
    kind,
    geom,
    out_csv,
    dt_s=5.0,
    alt_stop=250.0,
    max_days=800.0,
    f107=150.0,
    ap=4.0,
    msis_every_s=600.0,
    log_every_s=1800.0,
    alt0_km=500.0,
    inc_deg=23.0,
    ecc=0.001,
    mass_kg=0.625,
    physics="standard",
    gsi=None,
):
    nrm, area, cen = load_panels(geom)
    min_ax, max_ax, proj = drag_extreme_axes(nrm, area)
    print(kind, geom, "min_axis", min_ax, "max_axis", max_ax, "proj_m2", proj)
    phys = physics_mod.load(physics)
    if gsi is not None:
        from arlamx_v2.config import merge as _merge
        phys = physics_mod.validate(_merge(phys, {"aero": {"gsi": str(gsi).lower()}}))
    atmo_cfg = phys.get("atmosphere") or {}
    if f107 is None:
        f107 = float(atmo_cfg.get("f107", 150.0))
    if ap is None:
        ap = float(atmo_cfg.get("ap", 4.0))
    msis_ver = physics_mod.msis_version(phys)
    want_chi = physics_mod.gsi_name(phys) == "cll"
    atmo_model = str(atmo_cfg.get("model", "msis21")).lower()
    params = cpp.SimParams()
    params.mass = mass_kg
    params.dt_s = dt_s
    params.advisor_step_s = dt_s
    params.epoch_jd = 2461060.5
    params.max_slew_rad = np.pi
    params.mode = "prescribed"
    physics_mod.apply_to_params(params, phys, ggm_path=resolve_ggm())
    params.rk4_step_s = min(float(params.rk4_step_s), float(dt_s))
    sim = cpp.Simulator(params)
    sim.set_mode("prescribed")
    sim.set_panels(nrm, area, cen)
    sim.set_inertia_diag(np.array([0.0125, 0.0125, 0.025]))
    r0, v0 = coe_to_rv(RE + alt0_km * 1e3, ecc, np.radians(inc_deg), 0.0, 0.0, 0.0)
    pol = (
        MinDragPolicy(body_axis=min_ax) if kind == "min" else MaxDragPolicy(body_axis=max_ax)
    )
    extra = {"min_drag_axis": min_ax, "max_drag_axis": max_ax}
    sim.reset(r0, v0, np.zeros(3), np.zeros(3))
    rows = []
    nmax = int(max_days * 86400.0 / dt_s) + 2
    log_every = max(1, int(round(log_every_s / dt_s)))
    msis_every = max(1, int(round(msis_every_s / dt_s)))
    if atmo_model == "constant":
        rho, temp, mbar = physics_mod.constant_atmo(phys)
    else:
        rho, temp, mbar = 1e-12, 900.0, 2.656e-26
    sim.set_atmosphere(rho, temp, mbar)
    last_ok = None
    for k in range(nmax):
        st = sim.get_state()
        r = np.asarray(st["r"], float)
        v = np.asarray(st["v"], float)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            break
        if atmo_model != "constant" and k % msis_every == 0:
            jd = params.epoch_jd + float(st["t"]) / 86400.0
            lat, lon_e, alt_m, _gmst = geodetic_from_eci(r, jd)
            try:
                out_msis = query_msis(
                    alt_m / 1e3,
                    np.degrees(lat),
                    np.degrees(lon_e),
                    jd_to_datetime(jd),
                    f107,
                    ap,
                    version=msis_ver,
                    return_species=want_chi,
                )
                if want_chi:
                    rho, temp, mbar, chi = out_msis
                else:
                    rho, temp, mbar = out_msis
                    chi = None
                if np.isfinite(rho) and np.isfinite(temp) and rho > 0.0 and temp > 0.0:
                    if chi is not None:
                        sim.set_atmosphere(float(rho), float(temp), float(mbar), chi)
                    else:
                        sim.set_atmosphere(float(rho), float(temp), float(mbar))
            except Exception:
                pass
        st_pol = {"r": r, "v": v, **extra}
        q, _ = pol.predict(None, st_pol)
        out = sim.step(q)
        t = float(out["t"])
        alt = float(out["altitude_km"])
        sma = float(out["sma_m"])
        if not np.isfinite(alt) or not np.isfinite(sma):
            break
        last_ok = out
        if k % log_every == 0 or alt <= alt_stop:
            c_bn = cpp.mrp_to_dcm(np.asarray(out["sigma"], float))
            vhat_b = c_bn @ (v / max(float(np.linalg.norm(v)), 1.0))
            cd_, cl_ = float(out["Cd"]), float(out["Cl"])
            f_tot = float(out["drag_N"])
            cmag = max(float(np.hypot(cd_, cl_)), 1e-30)
            rows.append(
                {
                    "t_s": t,
                    "t_day": t / 86400.0,
                    "alt_km": alt,
                    "sma_km": sma / 1e3,
                    "Cd": cd_,
                    "Cl": cl_,
                    "ram_m2": ram_area(nrm, area, vhat_b),
                    "rho": float(rho),
                    "T": float(temp),
                    "eclipse": float(out["eclipse"]),
                    "drag_N": f_tot,
                    "drag_along_N": f_tot * cd_ / cmag,
                    "lift_N": f_tot * cl_ / cmag,
                    "dE_actual": float(out["dE_actual"]),
                }
            )
        if alt <= alt_stop:
            break
    if not rows and last_ok is not None:
        rows.append(
            {
                "t_s": float(last_ok["t"]),
                "t_day": float(last_ok["t"]) / 86400.0,
                "alt_km": float(last_ok["altitude_km"]),
                "sma_km": float(last_ok["sma_m"]) / 1e3,
                "Cd": float(last_ok["Cd"]),
                "rho": float(rho),
                "T": float(temp),
                "eclipse": float(last_ok["eclipse"]),
                "drag_N": float(last_ok["drag_N"]),
                "dE_actual": float(last_ok["dE_actual"]),
            }
        )
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"{kind}-drag produced no finite samples")
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows, out_csv


def plot_decay(min_csv, max_csv, png, alt_stop=250.0, alt0_km=500.0, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(path):
        t_day, alt = [], []
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for rec in reader:
                t_day.append(float(rec["t_day"]))
                alt.append(float(rec["alt_km"]))
        return np.array(t_day), np.array(alt)

    tmin, amin = load(min_csv)
    tmax, amax = load(max_csv)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(tmin, amin, label="min-drag (edge-on)", color="#2E6B4E")
    ax.plot(tmax, amax, label="max-drag (face-on)", color="#A33B2B")
    ax.axhline(alt_stop, color="#888", ls="--", lw=0.8)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Spherical altitude |r|-Re [km]")
    ax.set_title(title or "SolarCat decay  i=23°  e=0.001  F10.7=150  Ap=4  corotating+SRP")
    ax.legend()
    lo = min(alt_stop - 10.0, float(np.nanmin(amin)), float(np.nanmin(amax)))
    ax.set_ylim(lo - 10.0, alt0_km + 10.0)
    fig.tight_layout()
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)


def plot_forces(min_csv, max_csv, png, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(path):
        alt, d, l = [], [], []
        with open(path) as fh:
            for rec in csv.DictReader(fh):
                if "lift_N" not in rec:
                    return None
                alt.append(float(rec["alt_km"]))
                d.append(float(rec["drag_along_N"]))
                l.append(float(rec["lift_N"]))
        return np.array(alt), np.array(d), np.array(l)

    dmin, dmax = load(min_csv), load(max_csv)
    if dmin is None or dmax is None:
        print("plot_forces: CSVs lack lift columns, skipped")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for (alt, d, l), lab, col in ((dmin, "min-drag (edge-on)", "#2E6B4E"), (dmax, "max-drag (face-on)", "#A33B2B")):
        ax[0].plot(alt, d * 1e3, ".", ms=2.5, color=col, label=lab)
        ax[1].plot(alt, l * 1e3, ".", ms=2.5, color=col, label=lab)
    ax[0].set_ylabel("drag along the relative wind [mN]")
    ax[1].set_ylabel("lift, perpendicular to the wind [mN]")
    for a_ in ax:
        a_.set_xlabel("spherical altitude [km]")
        a_.set_yscale("log")
        a_.grid(alpha=0.3)
        a_.invert_xaxis()
        a_.legend(frameon=False, fontsize=8)
    fig.suptitle(title or "Aero forces along the decay (samples every 30 min)")
    fig.tight_layout()
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom", choices=("hex", "stl1pct"), default="hex")
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--max-days", type=float, default=800.0)
    parser.add_argument("--kind", choices=("min", "max", "both"), default="both")
    parser.add_argument("--alt0", type=float, default=500.0, help="initial altitude [km]")
    parser.add_argument("--alt-stop", type=float, default=250.0, help="stop altitude [km]")
    parser.add_argument("--inc", type=float, default=23.0, help="inclination [deg]")
    parser.add_argument("--ecc", type=float, default=0.001, help="eccentricity")
    parser.add_argument("--f107", type=float, default=None,
                        help="override atmosphere.f107 from the physics file")
    parser.add_argument("--ap", type=float, default=None,
                        help="override atmosphere.ap from the physics file")
    parser.add_argument("--mass", type=float, default=0.625, help="spacecraft mass [kg]")
    parser.add_argument("--physics", default="standard",
                        help="physics preset (fast|standard|high) or YAML path")
    parser.add_argument("--gsi", default=None, choices=("sentman", "cll"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "results" / "decay",
    )
    args = parser.parse_args()
    if args.f107 is None or args.ap is None:
        atmo = (physics_mod.load(args.physics).get("atmosphere") or {})
        if args.f107 is None:
            args.f107 = float(atmo.get("f107", 150.0))
        if args.ap is None:
            args.ap = float(atmo.get("ap", 4.0))
    args.out.mkdir(parents=True, exist_ok=True)
    pmin = args.out / "min_drag_decay.csv"
    pmax = args.out / "max_drag_decay.csv"
    rmin = rmax = None
    common = dict(
        dt_s=args.dt,
        max_days=args.max_days,
        alt_stop=args.alt_stop,
        f107=args.f107,
        ap=args.ap,
        alt0_km=args.alt0,
        inc_deg=args.inc,
        ecc=args.ecc,
        mass_kg=args.mass,
        physics=args.physics,
        gsi=args.gsi,
    )
    if args.kind in ("min", "both"):
        rmin, pmin = run_one("min", args.geom, pmin, **common)
        print("min last", rmin[-1], "n", len(rmin))
        print(f"min-drag: {args.alt0:g} -> {rmin[-1]['alt_km']:.1f} km in {rmin[-1]['t_day']:.2f} days")
    if args.kind in ("max", "both"):
        rmax, pmax = run_one("max", args.geom, pmax, **common)
        print("max last", rmax[-1], "n", len(rmax))
        print(f"max-drag: {args.alt0:g} -> {rmax[-1]['alt_km']:.1f} km in {rmax[-1]['t_day']:.2f} days")
    if rmin is not None and rmax is not None:
        png = args.out / "min_max_drag_decay.png"
        title = (
            f"SolarCat ({args.geom}, {args.mass:g} kg) decay  i={args.inc:g}°  e={args.ecc:g}  "
            f"F10.7={args.f107:g}  Ap={args.ap:g}  corotating+SRP"
        )
        plot_decay(pmin, pmax, png, alt_stop=args.alt_stop, alt0_km=args.alt0, title=title)
        png_f = args.out / "lift_drag_vs_alt.png"
        plot_forces(pmin, pmax, png_f, title=title.replace("decay", "aero forces along the decay"))
        print("wrote", pmin, pmax, png, png_f)


if __name__ == "__main__":
    main()
