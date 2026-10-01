"""Closed-loop heuristic / MPC campaign on the 1% SolarCat plant.

The policy never reads a sun-sensor vector. Power search uses only
power_gen_norm. MPC may use the analytic sun ephemeris (onboard clock +
orbit), which is not a sun sensor.

Sources: Wertz 1978 (mode switching); Rawlings et al. 2017 (MPC);
Vallado 2013 (two-body / eclipse). Level: advanced.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors import (
    HEURISTIC_BANK,
    GroundStationPolicy,
    MaxDragPolicy,
    MinDragPolicy,
    NadirPolicy,
    OffNadirPolicy,
    SamplingMpcPolicy,
)
from arlamx_v2.advisors.heuristic import HeuristicPowerPolicy
from arlamx_v2.advisors.stations import nearest_visible
from arlamx_v2.atmosphere import jd_to_datetime, query_msis
from arlamx_v2.decay_run import coe_to_rv, geodetic_from_eci, load_panels
from arlamx_v2.paths import resolve_ggm

BATT_CAP_J = 0.53 * 3600.0
PANEL_PEAK = 0.78 * 0.85
LOAD_BASE = 0.007
LOAD_GPS = 0.205
LOAD_TX = 0.400
TX_ALIGN = 0.7


def build_policies(panels):
    bank = {}
    for name, pol in HEURISTIC_BANK.items():
        p = deepcopy(pol)
        p.reset()
        bank[name] = p
    bank["min_drag"] = MinDragPolicy()
    bank["max_drag"] = MaxDragPolicy()
    bank["nadir"] = NadirPolicy()
    bank["ground_station"] = GroundStationPolicy()
    for ang in (0.0, 15.0, 30.0, 45.0, 60.0):
        bank[f"off_nadir_{ang:.0f}"] = OffNadirPolicy(ang, toward="velocity")
    for floor, tgt in ((0.02, 0.30), (0.05, 0.50), (0.10, 0.70), (0.20, 0.85)):
        key = f"search_f{int(floor*100):02d}_t{int(tgt*100):02d}"
        bank[key] = HeuristicPowerPolicy(
            soc_thresholds=(0.99, 0.98, 0.50, 0.30, 0.15),
            sweep_step_deg=15.0,
            climb_step_deg=6.0,
            sense_floor=floor,
            gen_target=tgt,
            eclipse_hold_steps=8,
            label=key,
            allow_gs=False,
        )
    mpc = SamplingMpcPolicy(horizon_steps=6, n_random=8, seed=0, panels=panels)
    mpc.reset()
    bank["mpc"] = mpc
    mpc_nosun = SamplingMpcPolicy(
        horizon_steps=6, n_random=8, seed=1, panels=panels, use_sun_ephemeris=False
    )
    mpc_nosun.reset()
    bank["mpc_no_sun_eph"] = mpc_nosun
    return bank


def update_power(sun_b, eclipse, dt_s, gs_cos, gs_vis):
    illum = abs(float(sun_b[2])) if eclipse > 0.5 else 0.0
    p_gen = PANEL_PEAK * illum
    p_load = LOAD_BASE
    if eclipse > 0.5:
        p_load += LOAD_GPS
    if gs_vis > 0.5 and gs_cos > TX_ALIGN and eclipse > 0.5:
        p_load += LOAD_TX
    return p_gen, p_load, illum


def simulate_policy(name, policy, panels, days=2.0, advisor_s=300.0, dt_s=2.0):
    params = cpp.SimParams()
    params.mass = 0.625
    params.dt_s = dt_s
    params.advisor_step_s = advisor_s
    params.rk4_step_s = dt_s
    params.sh_degree = 4
    params.lunisolar = False
    params.use_panel_srp = True
    params.corotating = True
    params.alpha_E = 0.93
    params.T_w = 300.0
    params.epoch_jd = 2461060.5
    params.ggm_path = resolve_ggm()
    params.max_slew_rad = np.radians(40.0)
    params.mode = "point"
    sim = cpp.Simulator(params)
    sim.set_panels(panels[0], panels[1], panels[2])
    sim.set_inertia_diag(np.array([0.0125, 0.0125, 0.025]))
    r0, v0 = coe_to_rv(cpp.RE_WGS + 500e3, 0.001, np.radians(23.0), 0.0, 0.0, 0.0)
    sim.reset(r0, v0, np.zeros(3), np.zeros(3))
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "panels"):
        policy.panels = panels
    if hasattr(policy, "advisor_step_s"):
        policy.advisor_step_s = advisor_s
    nstep = int(days * 86400.0 / advisor_s)
    batt_e = 0.5 * BATT_CAP_J
    rho, temp, mbar = 1e-12, 900.0, 2.656e-26
    sim.set_atmosphere(rho, temp, mbar)
    hist = []
    brownouts = 0
    ecl_entry_soc = []
    ecl_exit_soc = []
    prev_ecl = 1.0
    for k in range(nstep):
        st = sim.get_state()
        r = np.asarray(st["r"], float)
        v = np.asarray(st["v"], float)
        if not np.all(np.isfinite(r)):
            break
        jd = params.epoch_jd + float(st["t"]) / 86400.0
        lat, lon_e, alt_m, gmst = geodetic_from_eci(r, jd)
        if k % 2 == 0:
            try:
                rho, temp, mbar = query_msis(
                    alt_m / 1e3,
                    np.degrees(lat),
                    np.degrees(lon_e),
                    jd_to_datetime(jd),
                    150.0,
                    4.0,
                )
                sim.set_atmosphere(float(rho), float(temp), float(mbar))
            except Exception:
                pass
        gs_dir, gs_vis, gs_el = nearest_visible(r, gmst)
        soc = batt_e / BATT_CAP_J
        sun_n = np.asarray(st["sun_N"], float)
        c_bn = np.asarray(st["C_BN"], float)
        z_n = c_bn.T @ np.array([0.0, 0.0, 1.0])
        nadir = -r / max(float(np.linalg.norm(r)), 1.0)
        gen_norm = 0.0 if not hist else hist[-1]["gen_norm"]
        state = {
            "r": r,
            "v": v,
            "C_BN": c_bn,
            "sun_N": sun_n,
            "battery_soc": soc,
            "power_gen_norm": gen_norm,
            "gs_dir_N": gs_dir,
            "gs_visible": gs_vis,
            "rho": rho,
            "T": temp,
            "m_bar": mbar,
            "eclipse": float(st.get("eclipse", 1.0)),
        }
        q, info = policy.predict(None, state)
        out = sim.step(q)
        ecl = float(out["eclipse"])
        sun_b = np.asarray(out["sun_B"], float)
        c2 = cpp.mrp_to_dcm(np.asarray(out["sigma"], float))
        z_n = c2.T @ np.array([0.0, 0.0, 1.0])
        gs_cos = float(np.dot(z_n, gs_dir)) if gs_vis > 0.5 else 0.0
        nadir_cos = float(np.dot(z_n, nadir))
        p_gen, p_load, illum = update_power(sun_b, ecl, advisor_s, gs_cos, gs_vis)
        batt_e = min(BATT_CAP_J, max(0.0, batt_e + (p_gen - p_load) * advisor_s))
        soc = batt_e / BATT_CAP_J
        if soc <= 1e-6:
            brownouts += 1
        if prev_ecl > 0.5 and ecl < 0.5:
            ecl_entry_soc.append(soc)
        if prev_ecl < 0.5 and ecl > 0.5:
            ecl_exit_soc.append(soc)
        prev_ecl = ecl
        pick = ""
        if isinstance(info, dict):
            pick = str(info.get("pick", info.get("phase", info.get("mode", ""))))
        rec = {
            "t_s": float(out["t"]),
            "alt_km": float(out["altitude_km"]),
            "sma_m": float(out["sma_m"]),
            "Cd": float(out["Cd"]),
            "soc": soc,
            "gen_norm": p_gen / max(PANEL_PEAK, 1e-9),
            "p_gen": p_gen,
            "p_load": p_load,
            "eclipse": ecl,
            "gs_vis": gs_vis,
            "gs_cos": gs_cos,
            "nadir_cos": nadir_cos,
            "illum": illum,
            "rho": float(rho),
            "pick": pick,
        }
        hist.append(rec)
    return {
        "name": name,
        "hist": hist,
        "brownouts": brownouts,
        "ecl_entry_soc": ecl_entry_soc,
        "ecl_exit_soc": ecl_exit_soc,
    }


def summarise(run):
    h = run["hist"]
    if not h:
        return {"name": run["name"], "steps": 0}
    soc = np.array([r["soc"] for r in h])
    gen = np.array([r["gen_norm"] for r in h])
    cd = np.array([r["Cd"] for r in h])
    alt = np.array([r["alt_km"] for r in h])
    ecl = np.array([r["eclipse"] for r in h])
    gs_vis = np.array([r["gs_vis"] for r in h])
    gs_cos = np.array([r["gs_cos"] for r in h])
    nadir_cos = np.array([r["nadir_cos"] for r in h])
    lit = ecl > 0.5
    vis = gs_vis > 0.5
    entry = np.array(run["ecl_entry_soc"]) if run["ecl_entry_soc"] else np.array([np.nan])
    exit_ = np.array(run["ecl_exit_soc"]) if run["ecl_exit_soc"] else np.array([np.nan])
    return {
        "name": run["name"],
        "steps": len(h),
        "days": h[-1]["t_s"] / 86400.0,
        "soc_min": float(np.min(soc)),
        "soc_mean": float(np.mean(soc)),
        "soc_max": float(np.max(soc)),
        "brownouts": int(run["brownouts"]),
        "gen_mean": float(np.mean(gen)),
        "gen_mean_lit": float(np.mean(gen[lit])) if np.any(lit) else 0.0,
        "cd_mean": float(np.mean(cd)),
        "dalt_km": float(alt[-1] - alt[0]),
        "eclipse_frac": float(np.mean(1.0 - ecl)),
        "gs_vis_frac": float(np.mean(gs_vis)),
        "gs_lock_frac": float(np.mean((gs_cos > TX_ALIGN) & vis)) if np.any(vis) else 0.0,
        "nadir_30_frac": float(np.mean(nadir_cos > np.cos(np.radians(30.0)))),
        "nadir_mean_cos": float(np.mean(nadir_cos)),
        "ecl_entry_soc_mean": float(np.nanmean(entry)),
        "ecl_exit_soc_mean": float(np.nanmean(exit_)),
        "survived": bool(np.min(soc) > 1e-6 and np.all(np.isfinite(alt))),
    }


def write_hist(path, hist):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hist:
        return
    keys = [k for k in hist[0].keys() if k != "pick"] + ["pick"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(hist)


def plot_campaign(summaries, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s["name"] for s in summaries if s.get("steps", 0) > 0]
    soc = [next(x["soc_min"] for x in summaries if x["name"] == n) for n in names]
    gen = [next(x["gen_mean_lit"] for x in summaries if x["name"] == n) for n in names]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(names))
    ax.bar(x - 0.18, soc, 0.36, label="min SoC", color="#2E6B4E")
    ax.bar(x + 0.18, gen, 0.36, label="mean gen (lit)", color="#3D5A80")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Advisor campaign — 1% SolarCat, i=23°, 500 km, F10.7=150")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=2.0)
    parser.add_argument("--geom", choices=("hex", "stl1pct"), default="stl1pct")
    parser.add_argument("--only", type=str, default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "results" / "advisors",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    panels = load_panels(args.geom)
    bank = build_policies(panels)
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        bank = {k: v for k, v in bank.items() if k in want}
    summaries = []
    for name, pol in bank.items():
        print("running", name, flush=True)
        run = simulate_policy(name, pol, panels, days=args.days)
        summ = summarise(run)
        summaries.append(summ)
        write_hist(args.out / f"{name}.csv", run["hist"])
        print(" ", {k: summ[k] for k in ("soc_min", "brownouts", "gen_mean_lit", "gs_lock_frac", "survived")})
    with open(args.out / "summary.json", "w") as fh:
        json.dump(summaries, fh, indent=2)
    if summaries:
        keys = [k for k in summaries[0].keys() if k != "name"]
        with open(args.out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name"] + keys)
            w.writeheader()
            w.writerows(summaries)
        plot_campaign(summaries, args.out / "campaign_soc_gen.png")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
