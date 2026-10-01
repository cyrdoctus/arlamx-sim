"""Fixed angle-of-attack orbital decay with lift / drag / orbit-element logging.

Same plant setup as ``decay_run`` (MSIS 2.1 F10.7 / Ap, co-rotating wind,
GGM03S degree 4, panel SRP, prescribed attitude, dt = 5 s) but the attitude
is ``FixedAoAPolicy``: the sail is held at a fixed angle of attack against
the relative wind, with the lift aimed out-of-plane (+h) or radially (+r).
alpha = 0 / 90 reproduce the edge-on / face-on runs of ``decay_run``.

Every ``log_every_s`` the run records, besides altitude and sma, the
osculating elements from (r, v), the aero force re-evaluated with the
plant's own Sentman kernel at the logged state and split into the along-wind
(drag) and perpendicular (lift) parts in the radial / transverse / normal
frame, the measured angle of attack, and the plant's cumulative work
bookkeeping (dE_drag, dE_lift, dE_actual).

Post-processing integrates the Gauss variational equations (Vallado 2013
sec. 9.7) with the logged lift-only and drag-only accelerations so the change
of a, e, i, RAAN attributable to lift alone can be read off; the eccentricity
is followed as the vector (v x h)/mu - r_hat so e ~ 0.001 is not singular.

Sources: Sentman 1961; Vallado 2013 secs. 8.6, 9.7; Picone 2002 / Emmert 2021.
Level: advanced.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import OMEGA_EARTH, FixedAoAPolicy, drag_extreme_axes, ram_area
from arlamx_v2.atmosphere import jd_to_datetime, query_msis
from arlamx_v2.decay_run import coe_to_rv, geodetic_from_eci, load_panels
from arlamx_v2.paths import resolve_ggm

MU = cpp.MU_WGS
RE = cpp.RE_WGS
AXIS = {"x": 0, "y": 1, "z": 2}

# study-wide defaults (the 2026-09-05 min/max-drag campaign)
MASS = {"hex": 0.72, "stl1pct": 1.75}
LABEL = {"hex": "prior hex plates (.geom), 0.72 kg", "stl1pct": "six-petal STL (1pct), 1.75 kg"}
COLOR = {"hex": "#7f8c8d", "stl1pct": "#c9a227"}


def rv_to_coe(r, v, mu=MU):
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = float(np.linalg.norm(r))
    vn = float(np.linalg.norm(v))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    hhat = h / hn
    nvec = np.array([-h[1], h[0], 0.0])
    nn = float(np.linalg.norm(nvec))
    nhat = nvec / nn if nn > 0 else np.array([1.0, 0.0, 0.0])
    evec = ((vn * vn - mu / rn) * r - float(r @ v) * v) / mu
    e = float(np.linalg.norm(evec))
    energy = 0.5 * vn * vn - mu / rn
    a = -mu / (2.0 * energy)
    inc = float(np.arccos(np.clip(h[2] / hn, -1.0, 1.0)))
    raan = float(np.arctan2(nvec[1], nvec[0]))
    rhat = r / rn
    ehat = evec / e if e > 1e-15 else nhat
    argp = float(np.arctan2(float(np.cross(nhat, ehat) @ hhat), float(nhat @ ehat)))
    nu = float(np.arctan2(float(np.cross(ehat, rhat) @ hhat), float(ehat @ rhat)))
    u = float(np.arctan2(float(np.cross(nhat, rhat) @ hhat), float(nhat @ rhat)))
    return dict(a=a, e=e, inc=inc, raan=raan, argp=argp, nu=nu, u=u, energy=energy, h=h, evec=evec, p=hn * hn / mu)


def _midstep(r, v, tau):
    """Two-body Taylor step of the state by tau seconds (|tau| <= a few s)."""
    a = -MU * r / float(np.linalg.norm(r)) ** 3
    return r + v * tau + 0.5 * a * tau * tau, v + a * tau


def rtn_frame(r, v):
    rhat = r / np.linalg.norm(r)
    n = np.cross(r, v)
    nhat = n / np.linalg.norm(n)
    that = np.cross(nhat, rhat)
    return rhat, that, nhat


def make_sim(nrm, area, cen, mass_kg, dt_s, f107_unused=None):
    params = cpp.SimParams()
    params.mass = mass_kg
    params.dt_s = dt_s
    params.advisor_step_s = dt_s
    params.rk4_step_s = min(5.0, dt_s)
    params.sh_degree = 4
    params.lunisolar = False
    params.use_panel_srp = True
    params.corotating = True
    params.alpha_E = 0.93
    params.T_w = 300.0
    params.epoch_jd = 2461060.5
    params.ggm_path = resolve_ggm()
    params.max_slew_rad = np.pi
    params.mode = "prescribed"
    sim = cpp.Simulator(params)
    sim.set_mode("prescribed")
    sim.set_panels(nrm, area, cen)
    sim.set_inertia_diag(np.array([0.0125, 0.0125, 0.025]))
    return sim, params


FIELDS = [
    "t_s", "t_day", "alt_km", "sma_km", "ecc", "inc_deg", "raan_deg", "argp_deg", "nu_deg", "u_deg",
    "energy_J_kg", "rho", "T", "m_bar", "v_rel", "eclipse", "aoa_deg", "ram_m2", "Cd", "Cl", "A_ref_m2",
    "F_aero_N", "drag_N", "lift_N", "L_over_D",
    "F_R_N", "F_T_N", "F_N_N", "drag_R_N", "drag_T_N", "drag_N_N", "lift_R_N", "lift_T_N", "lift_N_N",
    "cum_dE_actual", "cum_dE_drag", "cum_dE_lift", "cum_dE_baseline",
]


def run_case(
    geom,
    alpha_deg,
    lift_toward,
    out_csv,
    mass_kg=None,
    dt_s=5.0,
    alt0_km=500.0,
    alt_stop_km=300.0,
    inc_deg=23.0,
    ecc=0.001,
    f107=150.0,
    ap=4.0,
    max_days=800.0,
    msis_every_s=600.0,
    log_every_s=120.0,
    log=print,
):
    mass_kg = MASS[geom] if mass_kg is None else mass_kg
    nrm, area, cen = load_panels(geom)
    min_ax, max_ax, proj = drag_extreme_axes(nrm, area)
    e_min = np.eye(3)[AXIS[min_ax]]
    e_max = np.eye(3)[AXIS[max_ax]]
    pol = FixedAoAPolicy(alpha_deg, min_ax, max_ax, lift_toward=lift_toward, corotating=True)
    sim, params = make_sim(nrm, area, cen, mass_kg, dt_s)
    log(
        f"[{geom} a={alpha_deg:g} {lift_toward}] mass {mass_kg} kg  panels {len(area)}  "
        f"min_axis {min_ax} max_axis {max_ax} proj_m2 {proj}"
    )
    r0, v0 = coe_to_rv(RE + alt0_km * 1e3, ecc, np.radians(inc_deg), 0.0, 0.0, 0.0)
    sim.reset(r0, v0, np.zeros(3), np.zeros(3))
    rho, temp, mbar = 1e-12, 900.0, 2.656e-26
    sim.set_atmosphere(rho, temp, mbar)
    nmax = int(max_days * 86400.0 / dt_s) + 2
    log_every = max(1, int(round(log_every_s / dt_s)))
    msis_every = max(1, int(round(msis_every_s / dt_s)))
    cum = dict(dE_actual=0.0, dE_drag=0.0, dE_lift=0.0, dE_baseline=0.0)
    rows = []
    t_wall = time.time()
    for k in range(nmax):
        st = sim.get_state()
        r = np.asarray(st["r"], float)
        v = np.asarray(st["v"], float)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            break
        if k % msis_every == 0:
            jd = params.epoch_jd + float(st["t"]) / 86400.0
            lat, lon_e, alt_m, _g = geodetic_from_eci(r, jd)
            try:
                rho_, temp_, mbar_ = query_msis(
                    alt_m / 1e3, np.degrees(lat), np.degrees(lon_e), jd_to_datetime(jd), f107, ap
                )
                if np.isfinite(rho_) and np.isfinite(temp_) and rho_ > 0.0 and temp_ > 0.0:
                    rho, temp, mbar = float(rho_), float(temp_), float(mbar_)
                    sim.set_atmosphere(rho, temp, mbar)
            except Exception:
                pass
        # the plant holds the commanded attitude for the whole dt step, so
        # command it for the mid-step state (two-body prediction) rather than
        # the start; otherwise the hold lags by n*dt/2 ~ 0.16 deg at dt = 5 s
        r_mid, v_mid = _midstep(r, v, +0.5 * dt_s)
        q, _ = pol.predict(None, {"r": r_mid, "v": v_mid})
        out = sim.step(q)
        for key in cum:
            cum[key] += float(out[key])
        t = float(out["t"])
        alt = float(out["altitude_km"])
        if not np.isfinite(alt):
            break
        if k % log_every == 0 or alt <= alt_stop_km:
            # log at the mid-step epoch, where the held attitude is exact
            r, v = _midstep(np.asarray(out["r"], float), np.asarray(out["v"], float), -0.5 * dt_s)
            t = t - 0.5 * dt_s
            alt = (float(np.linalg.norm(r)) - RE) / 1e3
            c_bn = cpp.mrp_to_dcm(np.asarray(out["sigma"], float))
            v_rel = v - np.cross([0.0, 0.0, OMEGA_EARTH], r)
            vrel = float(np.linalg.norm(v_rel))
            w = v_rel / vrel
            v_b_gas = -(c_bn @ v_rel)
            aero = cpp.spacecraft_aero(nrm, area, cen, v_b_gas, rho, temp, mbar, params.T_w, params.alpha_E, True)
            f_b = np.asarray(aero["force"], float)
            f_n = c_bn.T @ f_b
            f_along = float(f_n @ w)
            drag_vec = f_along * w
            lift_vec = f_n - drag_vec
            rhat, that, nhat = rtn_frame(r, v)
            w_b = c_bn @ w
            aoa = float(np.degrees(np.arctan2(float(w_b @ e_max), float(w_b @ e_min))))
            coe = rv_to_coe(r, v)
            drag = abs(f_along)
            lift = float(np.linalg.norm(lift_vec))
            rows.append(
                {
                    "t_s": t, "t_day": t / 86400.0, "alt_km": alt, "sma_km": coe["a"] / 1e3,
                    "ecc": coe["e"], "inc_deg": np.degrees(coe["inc"]), "raan_deg": np.degrees(coe["raan"]),
                    "argp_deg": np.degrees(coe["argp"]), "nu_deg": np.degrees(coe["nu"]), "u_deg": np.degrees(coe["u"]),
                    "energy_J_kg": coe["energy"], "rho": rho, "T": temp, "m_bar": mbar, "v_rel": vrel,
                    "eclipse": float(out["eclipse"]), "aoa_deg": aoa, "ram_m2": ram_area(nrm, area, w_b),
                    "Cd": float(aero["Cd"]), "Cl": float(aero["Cl"]), "A_ref_m2": float(aero["A_ref"]),
                    "F_aero_N": float(np.linalg.norm(f_n)), "drag_N": drag, "lift_N": lift,
                    "L_over_D": lift / drag if drag > 0 else 0.0,
                    "F_R_N": float(f_n @ rhat), "F_T_N": float(f_n @ that), "F_N_N": float(f_n @ nhat),
                    "drag_R_N": float(drag_vec @ rhat), "drag_T_N": float(drag_vec @ that), "drag_N_N": float(drag_vec @ nhat),
                    "lift_R_N": float(lift_vec @ rhat), "lift_T_N": float(lift_vec @ that), "lift_N_N": float(lift_vec @ nhat),
                    "cum_dE_actual": cum["dE_actual"], "cum_dE_drag": cum["dE_drag"], "cum_dE_lift": cum["dE_lift"],
                    "cum_dE_baseline": cum["dE_baseline"],
                }
            )
        if alt <= alt_stop_km:
            break
    if not rows:
        raise RuntimeError(f"{geom} alpha={alpha_deg} produced no samples")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: f"{v:.10g}" for k, v in row.items()})
    last = rows[-1]
    log(
        f"[{geom} a={alpha_deg:g} {lift_toward}] {alt0_km:g} -> {last['alt_km']:.1f} km in {last['t_day']:.2f} days "
        f"({len(rows)} samples, {time.time() - t_wall:.0f} s wall); aoa held {min(x['aoa_deg'] for x in rows):.3f}.."
        f"{max(x['aoa_deg'] for x in rows):.3f} deg; mean L/D {np.mean([x['L_over_D'] for x in rows]):.3f}; "
        f"work drag {last['cum_dE_drag']:.1f} lift {last['cum_dE_lift']:.3f} J/kg"
    )
    return rows, out_csv


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------


def load_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def orbit_mean(t, y, a_m):
    """Centered running mean over one local orbital period (variable window)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return y.copy()
    dt = float(np.median(np.diff(t)))
    per = 2.0 * np.pi * np.sqrt(np.asarray(a_m, float) ** 3 / MU)
    half = np.maximum(1, np.round(0.5 * per / dt).astype(int))
    cs = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty(n)
    for k in range(n):
        kk = min(max(k, half[k]), n - 1 - half[k])  # clamp so the window always fits
        if kk < 0 or kk >= n:
            kk = k
        lo = max(0, kk - half[k])
        hi = min(n, kk + half[k] + 1)
        out[k] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


def gauss_integrate(d, mass_kg, which="lift"):
    """Integrate the Gauss variational equations along the logged trajectory
    with the logged lift-only (or drag-only) acceleration. Returns cumulative
    da [km], de (along e_hat) and |de_vec|, di [deg], dRAAN [deg], dE [J/kg]."""
    t = d["t_s"]
    a = d["sma_km"] * 1e3
    e = d["ecc"]
    inc = np.radians(d["inc_deg"])
    nu = np.radians(d["nu_deg"])
    u = np.radians(d["u_deg"])
    p = a * (1.0 - e * e)
    h = np.sqrt(MU * p)
    r = p / (1.0 + e * np.cos(nu))
    aR = d[f"{which}_R_N"] / mass_kg
    aT = d[f"{which}_T_N"] / mass_kg
    aN = d[f"{which}_N_N"] / mass_kg
    da = 2.0 * a * a / h * (e * np.sin(nu) * aR + p / r * aT)
    de = (p * np.sin(nu) * aR + ((p + r) * np.cos(nu) + r * e) * aT) / h
    di = r * np.cos(u) * aN / h
    dO = r * np.sin(u) * aN / (h * np.sin(inc))
    # eccentricity vector: d e_vec/dt = (a x h + v x (r x a)) / mu, in RTN
    # with h = h N, r = r R, v = vR R + vT T
    vR = MU / h * e * np.sin(nu)
    vT = h / r
    aXh = np.stack([aT * h, -aR * h, np.zeros_like(h)], 1)  # (aR,aT,aN) x (0,0,h)
    rXa = np.stack([np.zeros_like(h), -r * aN, r * aT], 1)  # (r,0,0) x (aR,aT,aN)
    vX = np.stack(
        [vT * rXa[:, 2], -vR * rXa[:, 2], vR * rXa[:, 1] - vT * rXa[:, 0]], 1
    )  # (vR,vT,0) x rXa
    devec_rtn = (aXh + vX) / MU
    # rotate RTN -> inertial before integrating (RTN turns once per orbit)
    raan = np.radians(d["raan_deg"])
    cO, sO, ci, si, cu, su = np.cos(raan), np.sin(raan), np.cos(inc), np.sin(inc), np.cos(u), np.sin(u)
    R_hat = np.stack([cO * cu - sO * su * ci, sO * cu + cO * su * ci, su * si], 1)
    N_hat = np.stack([sO * si, -cO * si, ci], 1)
    T_hat = np.cross(N_hat, R_hat)
    devec = devec_rtn[:, :1] * R_hat + devec_rtn[:, 1:2] * T_hat + devec_rtn[:, 2:3] * N_hat
    dE = aR * vR + aT * vT

    def cum(y):
        out = np.zeros_like(t)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
        return out

    dev = np.stack([cum(devec[:, 0]), cum(devec[:, 1]), cum(devec[:, 2])], 1)
    # the scalar Gauss de/dt is kept for reference only: at e ~ 1e-3 the
    # osculating perigee direction is dominated by the J2 short-period term
    # and tracks the satellite, so cos(nu) does not average out and the
    # scalar integral is not a clean 'change of e'. The inertial e-vector
    # displacement |dev| is the singularity-free measure of a shape change.
    return {
        "t_day": d["t_day"],
        "da_km": cum(da) / 1e3,
        "de_scalar": cum(de),
        "de_vec": dev,
        "de_vec_mag": np.linalg.norm(dev, axis=1),
        "di_deg": np.degrees(cum(di)),
        "draan_deg": np.degrees(cum(dO)),
        "dE_J_kg": cum(dE),
    }


def _unwrap_deg(x):
    return np.degrees(np.unwrap(np.radians(x)))


def _style(alpha, lift_toward):
    if alpha == 45.0:
        return dict(ls="-" if lift_toward == "normal" else ":", lw=1.8 if lift_toward == "normal" else 1.6)
    if alpha == 90.0:
        return dict(ls="--", lw=1.0, alpha=0.7)
    return dict(ls="-.", lw=1.0, alpha=0.7)


def _label(geom, alpha, lift_toward):
    g = "original hexagon (72-plate .geom) 0.72 kg" if geom == "hex" else "six-petal STL (1 % simplified) 1.75 kg"
    if alpha == 45.0:
        return f"{g}, AoA 45°, lift {'out-of-plane (+h)' if lift_toward == 'normal' else 'radial (+r)'}"
    if alpha == 90.0:
        return f"{g}, face-on (max drag)"
    if alpha == 0.0:
        return f"{g}, edge-on (min drag)"
    return f"{g}, AoA {alpha:g}°"


def make_plots(runs, out_dir, alt0, alt_stop, inc, ecc, f107, ap):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = f"i={inc:g}°  e={ecc:g}  F10.7={f107:g}  Ap={ap:g}  MSIS2.1 co-rotating + SRP + GGM03S(4)  ARLAMX v2.1"
    data = {key: load_csv(path) for key, path in runs.items()}
    for key in data:
        d = data[key]
        d["sma_m"] = d["sma_km"] * 1e3
        # orbit-mean altitude for the x-axis of the vs-altitude plots: the
        # instantaneous altitude swings +-a*e ~ 10 km inside every orbit and
        # would draw orbit-averaged quantities as loops
        d["alt_mean_km"] = orbit_mean(d["t_s"], d["alt_km"], d["sma_m"])

    def _days_to_stop(d):
        hit = np.where(d["alt_km"] <= alt_stop)[0]
        return float(d["t_day"][hit[0]]) if len(hit) else float(d["t_day"][-1])

    def _decay_axes(ax, only45):
        for (geom, alpha, tilt), d in data.items():
            if only45 and alpha != 45.0:
                continue
            days = _days_to_stop(d)
            ax.plot(d["t_day"], d["alt_mean_km"], color=COLOR[geom],
                    label=f"{_label(geom, alpha, tilt)}:  {days:.1f} d", **_style(alpha, tilt))
            if alpha == 45.0 and tilt == "normal":
                ax.fill_between(d["t_day"], orbit_mean(d["t_s"], d["alt_km"] - 0.0, d["sma_m"]) - d["sma_km"] * d["ecc"],
                                d["alt_mean_km"] + d["sma_km"] * d["ecc"], color=COLOR[geom], alpha=0.12, lw=0)
            if alpha == 45.0:
                ax.annotate(f"{days:.1f} d", (days, alt_stop), xytext=(4, 6 if tilt == "normal" else -12),
                            textcoords="offset points", fontsize=8, color=COLOR[geom])
        ax.axhline(alt_stop, color="#888", ls="--", lw=0.8)
        ax.set_xlabel("time [days]")
        ax.set_ylabel("orbit-mean spherical altitude [km]\n(band: perigee..apogee of the 45° out-of-plane run)")
        ax.set_ylim(alt_stop - 15, alt0 + 10)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    # 1. decay: altitude vs time, everything ------------------------------------------
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    _decay_axes(ax, only45=False)
    ax.set_title(f"Decay at fixed 45° angle of attack vs edge-on / face-on, {alt0:g} -> {alt_stop:g} km\n{env}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "decay_aoa45_hex_vs_stl.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 1b. just the 45 deg curves
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    _decay_axes(ax, only45=True)
    ax.set_title(f"Decay at fixed 45° angle of attack, original hexagon vs six-petal, {alt0:g} -> {alt_stop:g} km\n{env}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "decay_aoa45_only.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 2. forces vs altitude ------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
    for (geom, alpha, tilt), d in data.items():
        if tilt != "normal" and alpha != 45.0:
            continue
        if alpha == 45.0 and tilt != "normal":
            continue
        st = _style(alpha, tilt)
        lab = _label(geom, alpha, tilt)
        am = d["sma_m"]
        x = d["alt_mean_km"]
        ax[0, 0].plot(x, orbit_mean(d["t_s"], d["drag_N"], am) * 1e3, color=COLOR[geom], label=lab, **st)
        if alpha == 45.0:
            ax[0, 1].plot(x, orbit_mean(d["t_s"], d["lift_N"], am) * 1e3, color=COLOR[geom], label=lab, **st)
            ax[1, 0].plot(x, orbit_mean(d["t_s"], d["L_over_D"], am), color=COLOR[geom], label=lab, **st)
            ax[1, 1].plot(x, orbit_mean(d["t_s"], d["Cd"], am), color=COLOR[geom], label=f"Cd, {lab}", **st)
            ax[1, 1].plot(x, orbit_mean(d["t_s"], d["Cl"], am), color=COLOR[geom], ls="--", lw=1.2, label=f"Cl, {lab}")
    ax[0, 0].set_ylabel("drag along the wind [mN]  (orbit mean)")
    ax[0, 0].set_yscale("log")
    ax[0, 1].set_ylabel("lift perpendicular to the wind [mN]  (orbit mean)")
    ax[0, 1].set_yscale("log")
    ax[1, 0].set_ylabel("L / D  (orbit mean)")
    ax[1, 1].set_ylabel("Cd (solid), Cl (dashed), A_ref = wetted projection")
    for a_ in ax.ravel():
        a_.set_xlabel("orbit-mean spherical altitude [km]")
        a_.invert_xaxis()
        a_.grid(alpha=0.3, which="both")
        a_.legend(frameon=False, fontsize=7, loc="lower right" if a_ is ax[0, 0] else "best")
    fig.suptitle(f"Aero forces along the decay at 45° angle of attack\n{env}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "forces_vs_alt.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 3. RTN components of drag and lift vs altitude (45 deg runs) -----------------
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)
    comps = (("R", "radial (+ = up)"), ("T", "transverse (+ = along track)"), ("N", "normal (+ = orbit normal h)"))
    for (geom, alpha, tilt), d in data.items():
        if alpha != 45.0:
            continue
        st = _style(alpha, tilt)
        lab = _label(geom, alpha, tilt)
        am = d["sma_m"]
        for j, (c, _cl) in enumerate(comps):
            ax[0, j].plot(d["alt_mean_km"], orbit_mean(d["t_s"], d[f"drag_{c}_N"], am) * 1e6, color=COLOR[geom], label=lab, **st)
            ax[1, j].plot(d["alt_mean_km"], orbit_mean(d["t_s"], d[f"lift_{c}_N"], am) * 1e6, color=COLOR[geom], label=lab, **st)
    for j, (c, cl) in enumerate(comps):
        ax[0, j].set_title(f"drag part, {cl}", fontsize=9)
        ax[1, j].set_title(f"lift part, {cl}", fontsize=9)
        ax[1, j].set_xlabel("orbit-mean spherical altitude [km]")
    for a_ in ax.ravel():
        a_.set_ylabel("force [µN]  (orbit mean)")
        a_.grid(alpha=0.3)
        a_.invert_xaxis()
        a_.axhline(0.0, color="#999", lw=0.6)
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Aero force split into the along-wind (drag) and perpendicular (lift) parts, in the radial / transverse / normal frame\n"
        + env,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "forces_rtn_vs_alt.png", dpi=150, facecolor="white")
    plt.close(fig)

    # mean eccentricity vector in the node frame (e cos w, e sin w averaged
    # over one orbit): at e ~ 1e-3 the osculating perigee is dominated by the
    # J2 short-period term and tracks the satellite, so the osculating w (and
    # a plain running mean of e) is not a shape measure; the averaged vector is
    for key, d in data.items():
        w = np.radians(d["argp_deg"])
        ex = orbit_mean(d["t_s"], d["ecc"] * np.cos(w), d["sma_m"])
        ey = orbit_mean(d["t_s"], d["ecc"] * np.sin(w), d["sma_m"])
        d["e_mean"] = np.hypot(ex, ey)
        d["argp_mean_deg"] = np.degrees(np.unwrap(np.arctan2(ey, ex)))

    # 4. simulated orbit elements vs time (orbit-mean) ----------------------------
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.2))
    panels = (
        ("sma_km", "semi-major axis [km]  (orbit mean)", 1.0),
        ("energy_J_kg", "specific orbital energy [MJ/kg]  (orbit mean)", 1e-6),
        ("e_mean", "mean eccentricity |<e cos w, e sin w>|", 1.0),
        ("inc_deg", "inclination [deg]  (orbit mean)", 1.0),
        ("raan_deg", "RAAN [deg]  (unwrapped, orbit mean; J2 regression)", 1.0),
        ("argp_mean_deg", "mean argument of perigee [deg]  (unwrapped; J2 advance)", 1.0),
    )
    for (geom, alpha, tilt), d in data.items():
        st = _style(alpha, tilt)
        lab = _label(geom, alpha, tilt)
        am = d["sma_m"]
        for a_, (key, _yl, sc) in zip(ax.ravel(), panels):
            y = d[key]
            if key == "raan_deg":
                y = orbit_mean(d["t_s"], _unwrap_deg(y), am)
            elif key in ("sma_km", "energy_J_kg", "inc_deg"):
                y = orbit_mean(d["t_s"], y, am)
            a_.plot(d["t_day"], y * sc, color=COLOR[geom], label=lab, **st)
    for a_, (_k, yl, _s) in zip(ax.ravel(), panels):
        a_.set_ylabel(yl)
        a_.set_xlabel("time [days]")
        a_.grid(alpha=0.3)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"Simulated orbit elements, running mean over one orbit\n{env}", fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out_dir / "orbit_elements_vs_time.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 4b. same elements against altitude, so runs with different lifetimes overlap
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.4))
    for (geom, alpha, tilt), d in data.items():
        st = _style(alpha, tilt)
        lab = _label(geom, alpha, tilt)
        am = d["sma_m"]
        for a_, key in zip(ax, ("e_mean", "inc_deg", "energy_J_kg")):
            sc = 1e-6 if key == "energy_J_kg" else 1.0
            y = d[key] if key == "e_mean" else orbit_mean(d["t_s"], d[key], am)
            a_.plot(d["alt_mean_km"], y * sc, color=COLOR[geom], label=lab, **st)
    for a_, yl in zip(ax, ("mean eccentricity |<e cos w, e sin w>|", "inclination [deg]  (orbit mean)", "specific orbital energy [MJ/kg]")):
        a_.set_ylabel(yl)
        a_.set_xlabel("orbit-mean spherical altitude [km]")
        a_.invert_xaxis()
        a_.grid(alpha=0.3)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"Orbit shape and energy against altitude, all attitudes\n{env}", fontsize=10)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(out_dir / "orbit_elements_vs_alt.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 5. what lift alone did (Gauss integration), against what drag did -----------
    gauss = {}
    for (geom, alpha, tilt), d in data.items():
        gauss[(geom, alpha, tilt)] = (
            gauss_integrate(d, MASS[geom], "lift"),
            gauss_integrate(d, MASS[geom], "drag"),
        )
    fig, ax = plt.subplots(2, 4, figsize=(15, 8.0))
    q = (("da_km", "Δa [km]"), ("de_vec_mag", "|Δe-vector|  (e0 = 0.001)"), ("di_deg", "Δi [deg]"), ("draan_deg", "ΔRAAN [deg]"))
    for (geom, alpha, tilt), (gl, gd) in gauss.items():
        if alpha not in (45.0, 90.0):
            continue
        st = _style(alpha, tilt)
        lab = _label(geom, alpha, tilt)
        for j, (key, _yl) in enumerate(q):
            ax[0, j].plot(gl["t_day"], gl[key], color=COLOR[geom], label=lab + (" (control)" if alpha == 90.0 else ""), **st)
            if tilt == "normal":
                ax[1, j].plot(gd["t_day"], gd[key], color=COLOR[geom], label=lab.replace(", lift out-of-plane (+h)", ""), **st)
    for j, (_k, yl) in enumerate(q):
        ax[0, j].set_title(f"lift alone: {yl}", fontsize=9)
        ax[1, j].set_title(f"drag alone: {yl}", fontsize=9)
        ax[1, j].set_xlabel("time [days]")
        for i_ in (0, 1):
            ax[i_, j].grid(alpha=0.3)
            ax[i_, j].axhline(0.0, color="#999", lw=0.6)
            ax[i_, j].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    h0, l0 = ax[0, 0].get_legend_handles_labels()
    h1, l1 = ax[1, 0].get_legend_handles_labels()
    fig.legend(h0, ["top: " + x for x in l0], loc="lower left", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.01, 0.0))
    fig.legend(h1, ["bottom: " + x for x in l1], loc="lower right", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.99, 0.0))
    fig.suptitle(
        "Gauss variational equations integrated with the logged lift-only (top) and drag-only (bottom) accelerations\n"
        "top = what the fixed-attitude lift did to the orbit over the whole decay;  " + env,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_dir / "lift_effect_gauss.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 6. energy budget --------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for (geom, alpha, tilt), d in data.items():
        if alpha != 45.0:
            continue
        st = _style(alpha, tilt)
        lab = _label(geom, alpha, tilt)
        ax[0].plot(d["t_day"], d["cum_dE_drag"] / 1e3, color=COLOR[geom], label=lab, **st)
        ax[1].plot(d["t_day"], d["cum_dE_lift"], color=COLOR[geom], label=lab, **st)
        srp = d["cum_dE_actual"] - d["cum_dE_drag"] - d["cum_dE_lift"]
        ax[2].plot(d["t_day"], srp, color=COLOR[geom], label=lab, **st)
    ax[0].set_ylabel("cumulative work by drag on the orbit [kJ/kg]")
    ax[1].set_ylabel("cumulative work by lift on the orbit [J/kg]")
    ax[2].set_ylabel("cumulative work by SRP on the orbit [J/kg]")
    for a_ in ax:
        a_.set_xlabel("time [days]")
        a_.grid(alpha=0.3)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(
        "Specific-energy bookkeeping from the plant (dE_drag, dE_lift, dE_actual - dE_drag - dE_lift)\n"
        "lift is perpendicular to the relative wind, so its work comes only through the co-rotation wind angle;  " + env,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_dir / "energy_budget.png", dpi=150, facecolor="white")
    plt.close(fig)

    # 7. angle-of-attack hold check + eclipse -----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for (geom, alpha, tilt), d in data.items():
        if alpha != 45.0:
            continue
        ax.plot(d["t_day"], d["aoa_deg"], color=COLOR[geom], label=_label(geom, alpha, tilt), **_style(alpha, tilt))
    ax.set_xlabel("time [days]")
    ax.set_ylabel("measured angle of attack [deg]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "aoa_hold_check.png", dpi=150, facecolor="white")
    plt.close(fig)

    return data, gauss


def summarize(data, gauss, out_dir, alt_stop):
    rows = []
    for (geom, alpha, tilt), d in data.items():
        k_end = int(np.argmax(d["alt_km"] <= alt_stop)) if np.any(d["alt_km"] <= alt_stop) else len(d["alt_km"]) - 1
        am = d["sma_m"]
        inc_m = orbit_mean(d["t_s"], d["inc_deg"], am)
        ecc_m = orbit_mean(d["t_s"], d["ecc"], am)
        row = {
            "geom": geom,
            "mass_kg": MASS[geom],
            "alpha_deg": alpha,
            "lift_toward": tilt,
            "days_to_stop": d["t_day"][k_end],
            "alt_end_km": d["alt_km"][k_end],
            "aoa_min_deg": d["aoa_deg"].min(),
            "aoa_max_deg": d["aoa_deg"].max(),
            "mean_L_over_D": float(np.mean(d["L_over_D"])),
            "mean_Cd": float(np.mean(d["Cd"])),
            "mean_Cl": float(np.mean(d["Cl"])),
        }
        drag_m = orbit_mean(d["t_s"], d["drag_N"], am)
        lift_m = orbit_mean(d["t_s"], d["lift_N"], am)
        alt_m = orbit_mean(d["t_s"], d["alt_km"], am)
        row["alt_mean_start_km"] = float(alt_m[0])
        row["drag_uN_start"] = float(drag_m[0]) * 1e6
        row["lift_uN_start"] = float(lift_m[0]) * 1e6
        for alt in (400.0, 300.0):
            k = int(np.argmin(np.abs(alt_m - alt)))
            row[f"drag_uN_{alt:.0f}km"] = float(drag_m[k]) * 1e6
            row[f"lift_uN_{alt:.0f}km"] = float(lift_m[k]) * 1e6
        row.update(
            {
                "d_inc_mean_deg_sim": float(inc_m[k_end] - inc_m[0]),
                "d_ecc_mean_sim": float(ecc_m[k_end] - ecc_m[0]),
                "d_energy_J_kg_sim": float(d["energy_J_kg"][k_end] - d["energy_J_kg"][0]),
                "work_drag_J_kg": float(d["cum_dE_drag"][k_end]),
                "work_lift_J_kg": float(d["cum_dE_lift"][k_end]),
                "work_srp_J_kg": float(d["cum_dE_actual"][k_end] - d["cum_dE_drag"][k_end] - d["cum_dE_lift"][k_end]),
            }
        )
        if (geom, alpha, tilt) in gauss:
            gl, gd = gauss[(geom, alpha, tilt)]
            row.update(
                {
                    "lift_only_da_km_end": float(gl["da_km"][k_end]),
                    "lift_only_da_km_peak": float(np.max(np.abs(gl["da_km"][: k_end + 1]))),
                    "lift_only_de_vec_end": float(gl["de_vec_mag"][k_end]),
                    "lift_only_de_vec_peak": float(np.max(gl["de_vec_mag"][: k_end + 1])),
                    "lift_only_di_deg_end": float(gl["di_deg"][k_end]),
                    "lift_only_di_deg_peak": float(np.max(np.abs(gl["di_deg"][: k_end + 1]))),
                    "lift_only_draan_deg_end": float(gl["draan_deg"][k_end]),
                    "lift_only_draan_deg_peak": float(np.max(np.abs(gl["draan_deg"][: k_end + 1]))),
                    "drag_only_da_km_end": float(gd["da_km"][k_end]),
                    "drag_only_di_deg_end": float(gd["di_deg"][k_end]),
                    "drag_only_draan_deg_end": float(gd["draan_deg"][k_end]),
                    "drag_only_de_vec_end": float(gd["de_vec_mag"][k_end]),
                    "drag_only_de_vec_peak": float(np.max(gd["de_vec_mag"][: k_end + 1])),
                    "lift_only_dE_J_kg_end": float(gl["dE_J_kg"][k_end]),
                }
            )
        rows.append(row)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    return rows


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--geoms", nargs="+", default=["hex", "stl1pct"], choices=("hex", "stl1pct"))
    ap_.add_argument("--alpha", type=float, default=45.0, help="angle of attack for the study runs [deg]")
    ap_.add_argument("--tilts", nargs="+", default=["normal", "zenith"], choices=("normal", "anti_normal", "zenith", "nadir"))
    ap_.add_argument("--refs", nargs="*", type=float, default=[0.0, 90.0], help="reference AoAs (edge-on / face-on) to rerun with the same logging")
    ap_.add_argument("--mass-hex", type=float, default=MASS["hex"])
    ap_.add_argument("--mass-stl", type=float, default=MASS["stl1pct"])
    ap_.add_argument("--dt", type=float, default=5.0)
    ap_.add_argument("--alt0", type=float, default=500.0)
    ap_.add_argument("--alt-stop", type=float, default=300.0)
    ap_.add_argument("--inc", type=float, default=23.0)
    ap_.add_argument("--ecc", type=float, default=0.001)
    ap_.add_argument("--f107", type=float, default=150.0)
    ap_.add_argument("--ap", type=float, default=4.0)
    ap_.add_argument("--log-every", type=float, default=120.0, help="log cadence [s] (~46 samples per orbit)")
    ap_.add_argument("--max-days", type=float, default=800.0)
    ap_.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "results" / "decay" / "aoa45_500to300")
    ap_.add_argument("--plot-only", action="store_true", help="skip the runs, regenerate plots from the CSVs")
    args = ap_.parse_args()
    MASS["hex"] = args.mass_hex
    MASS["stl1pct"] = args.mass_stl
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "run.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    cases = []
    for geom in args.geoms:
        for tilt in args.tilts:
            cases.append((geom, args.alpha, tilt))
        for a in args.refs:
            cases.append((geom, float(a), "normal"))
    runs = {}
    for geom, alpha, tilt in cases:
        path = args.out / f"{geom}_aoa{alpha:g}_{tilt}.csv"
        runs[(geom, alpha, tilt)] = path
        if args.plot_only and path.exists():
            continue
        run_case(
            geom, alpha, tilt, path, mass_kg=MASS[geom], dt_s=args.dt, alt0_km=args.alt0, alt_stop_km=args.alt_stop,
            inc_deg=args.inc, ecc=args.ecc, f107=args.f107, ap=args.ap, max_days=args.max_days,
            log_every_s=args.log_every, log=log,
        )
    data, gauss = make_plots(runs, args.out, args.alt0, args.alt_stop, args.inc, args.ecc, args.f107, args.ap)
    rows = summarize(data, gauss, args.out, args.alt_stop)
    for r in rows:
        log(
            f"{r['geom']:8s} a={r['alpha_deg']:4.0f} {r['lift_toward']:7s} {r['days_to_stop']:8.2f} d  "
            f"L/D {r['mean_L_over_D']:.3f}  lift@400km {r['lift_uN_400km']:8.2f} uN  drag@400km {r['drag_uN_400km']:8.2f} uN  "
            f"lift-only Δi peak {r.get('lift_only_di_deg_peak', float('nan')):.2e} deg  end {r.get('lift_only_di_deg_end', float('nan')):.2e}  "
            f"|Δe-vec| lift {r.get('lift_only_de_vec_end', float('nan')):.2e} drag {r.get('drag_only_de_vec_end', float('nan')):.2e}  work lift {r['work_lift_J_kg']:.3f} / drag {r['work_drag_J_kg']:.0f} J/kg"
        )
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
