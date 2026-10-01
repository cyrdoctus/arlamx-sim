"""70x70 gradient sweeps on the V2 plant: space weather + orbits beyond training.

Panels (each a full 70x70 grid, one 2-orbit deterministic episode per cell):
  sw    F10.7 60-320  x  Ap 0-180          (trained: 65-250 x 2-40), 400 km
  alt   altitude 280-700 km  x  incl 0-98  (trained: 300-500 x 20-30)
  ecc   eccentricity 0-0.02  x  incl 0-98  (trained: 0.001-0.01 x 20-30), 500 km

Policies (fixed identity order, matching the figure palette): MPC, heuristic,
min-drag, and the retrained 50k deep-RL trio (PPO / SAC / TD3) — each RL model
evaluated with exactly the env options it was trained with.

Usage:  PYTHONPATH=python python -m arlamx_v2.gradient70 [--test] [--panels sw,alt,ecc]
Outputs -> outputs/analysis/gradient70/grid70_{panel}.npz
Level: advanced.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

TEST = "--test" in sys.argv
GRID_N = 8 if TEST else 70
CHUNKS = 2 if TEST else 14
MAX_WORKERS = 8 if TEST else 44
EP_ORBITS = 3.0
SEED = 777

_arg = [a for a in sys.argv if a.startswith("--panels")]
PANELS = (_arg[0].split("=", 1)[1].split(",") if _arg and "=" in _arg[0]
          else ["sw", "alt", "ecc"])

# x axis, y axis per panel (x = columns, y = rows of the stored 2D arrays)
AXES = {
    "sw":  dict(x=("f107", np.linspace(60.0, 320.0, GRID_N)),
                y=("ap", np.linspace(0.0, 180.0, GRID_N)),
                base=dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001)),
    "alt": dict(x=("inc_deg", np.linspace(0.0, 98.0, GRID_N)),
                y=("altitude_km", np.linspace(280.0, 700.0, GRID_N)),
                base=dict(f107=150.0, ap=4.0, ecc=0.001)),
    "ecc": dict(x=("inc_deg", np.linspace(0.0, 98.0, GRID_N)),
                y=("ecc", np.linspace(0.0, 0.02, GRID_N)),
                base=dict(altitude_km=500.0, f107=150.0, ap=4.0)),
}
BASE_OPT = dict(raan_deg=0.0, argp_deg=0.0, nu_deg=0.0, mass_kg=0.625, soc=0.5,
                omega_dps=[1.0, 1.0, 0.5])

# (slug, kind, spec). RL entries carry the env_kw they were trained with and
# their campaign run name; advisor entries carry the advisor id.
WINNER_W = {"band_above_mode": "exp", "dE_weight": 3.5, "thrift_weight": 0.9}
PLAIN_W = {"band_above_mode": "exp", "dE_weight": 3.5}
GAINS = {"kp": 1e-4, "kd": 2e-3}
POLICIES = [
    ("mpc",       "advisor", dict(which="mpc")),
    ("heuristic", "advisor", dict(which="heuristic")),
    ("min_drag",  "advisor", dict(which="min_drag")),
    ("rl_ppo",    "model", dict(algo="ppo", run="s4_f0p3_th0p9_obs",
                                env_kw={**GAINS, "obs_extra": "observer",
                                        "gate_floor": 0.3, "reward_w": WINNER_W})),
    ("rl_sac",    "model", dict(algo="sac", run="s4_f0p3_th0p9_obs_sac",
                                env_kw={**GAINS, "obs_extra": "observer",
                                        "gate_floor": 0.3, "reward_w": WINNER_W})),
    ("rl_td3",    "model", dict(algo="td3", run="s4_f0p3_th0p9_obs_td3",
                                env_kw={**GAINS, "obs_extra": "observer",
                                        "gate_floor": 0.3, "reward_w": WINNER_W})),
]
if TEST:
    POLICIES = [POLICIES[0], POLICIES[3]]

QUANTS = ("decay", "soc_mean", "soc_min", "brownouts", "downlink", "cd",
          "effort", "lifetime_d", "nsteps")


def _advisor_state(env, cpp, last):
    """Flight-computer state dict for the advisor policies (same inputs the
    campaign's MPC reference uses)."""
    from arlamx_v2.advisors.stations import nearest_visible
    from arlamx_v2.atmosphere import jd_to_datetime, query_msis
    from arlamx_v2.env import EPOCH_JD

    st = env.sim.get_state()
    r = np.asarray(st["r"], float)
    jd = EPOCH_JD + float(st["t"]) / 86400.0
    gmst = cpp.gmst_rad(jd)
    lat, lon, alt_m = cpp.ecef_to_geodetic(r)
    lon_e = (lon - gmst + np.pi) % (2 * np.pi) - np.pi
    try:
        rho, T, mb = query_msis(alt_m / 1e3, np.degrees(lat), np.degrees(lon_e),
                                jd_to_datetime(jd), env._f107, env._ap)
    except Exception:
        rho, T, mb = None, 900.0, 2.656e-26
    gs_n, gs_vis, _el = nearest_visible(r, gmst)
    c_bn = cpp.mrp_to_dcm(np.asarray(st["sigma"], float))
    return {
        "r": r, "v": np.asarray(st["v"], float),
        "sun_N": np.asarray(st.get("sun_N", np.zeros(3)), float),
        "C_BN": c_bn,
        "battery_soc": env._batt_E / env._batt_cap_J,
        "power_gen_norm": float(last.get("power_gen_norm", 0.0)),
        "rho": rho, "T": T, "m_bar": mb,
        "gs_dir_N": gs_n if gs_vis > 0.5 else None,
        "gs_dir_B": (c_bn @ gs_n) if gs_vis > 0.5 else None,
        "gs_visible": gs_vis,
    }


def _build_policy(kind, spec):
    from arlamx_v2 import cpp

    if kind == "model":
        from stable_baselines3 import PPO, SAC, TD3
        cls = {"sac": SAC, "td3": TD3}.get(spec["algo"], PPO)
        # spec["root"] lets a caller point at a run tree other than the tuning
        # campaign's (e.g. the production bake-off runs). Default unchanged.
        root = Path(spec.get("root") or (ROOT / "outputs" / ".old" / "campaign" / "runs"))
        path = root / spec["run"] / "models" / f"{spec['algo']}_{spec['run']}.zip"
        model = cls.load(str(path))

        def f(obs, env, last):
            a, _ = model.predict(obs, deterministic=True)
            return a
        return f, spec["env_kw"]

    which = spec["which"]
    if which == "mpc":
        from arlamx_v2.advisors.mpc import SamplingMpcPolicy
        pol = SamplingMpcPolicy(seed=0)
        needs_panels = True
    elif which == "heuristic":
        from arlamx_v2.advisors.heuristic import HeuristicPowerPolicy
        pol = HeuristicPowerPolicy()
        needs_panels = False
    else:
        from arlamx_v2.advisors.mpc import MinDragPolicy
        pol = MinDragPolicy()
        needs_panels = False

    holder = {"env": None}

    def f(obs, env, last):
        if holder["env"] is not env or env._step == 0:
            if hasattr(pol, "reset"):
                pol.reset()
            if needs_panels:
                pol.panels = (env._n, env._A, env._c)
                pol.advisor_step_s = env._advisor_s
            holder["env"] = env
        state = _advisor_state(env, cpp, last)
        q, _meta = pol.predict(obs, state)
        return np.concatenate([np.asarray(q, float), [1.0, 1.0, 1.0]])
    # Advisors run in the v7 env with gates held open and no extra obs.
    return f, {"obs_extra": "none", "gate_floor": 1.0, **GAINS}


def _cell_metrics(env, predict_fn, opt):
    obs, _ = env.reset(seed=SEED, options=opt)
    env._weather_jumps = []
    env._srp_flashes = []
    soc_l, dep_l, sma_l, dl_l, cd_l, ef_l = [], [], [], [], [], []
    last = {"power_gen_norm": 0.0}
    done, n = False, 0
    while not done:
        a = predict_fn(obs, env, last)
        obs, _r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        done = term or trunc
        n += 1
        last["power_gen_norm"] = float(info.get("power_gen_norm", 0.0))
        soc_l.append(float(info["battery_soc"]))
        dep_l.append(1.0 if info.get("battery_depleted") else 0.0)
        sma_l.append(float(info["sma_m"]))
        vis = float(info.get("gs_visible", 0.0)) > 0.5
        lit = float(info.get("eclipse", 1.0)) > 0.5
        dl_l.append(1.0 if (vis and lit and float(info.get("gs_point_cos", 0)) > 0.7)
                    else 0.0)
        cd_l.append(float(info.get("Cd", np.nan)))
        ef_l.append(float(info.get("torque_effort", 0.0)))
    soc = np.asarray(soc_l)
    dep = np.asarray(dep_l) > 0.5
    sma = np.asarray(sma_l)
    days = max(n * env._advisor_s / 86400.0, 1e-9)
    # J2-robust decay: mean osculating SMA over the FIRST FULL ORBIT vs the
    # LAST FULL ORBIT (the episode is an integer number of orbits, so the
    # several-km J2 short-period oscillation averages out of each window),
    # divided by the time between window centres. An endpoint difference is
    # dominated by that oscillation wherever the true decay is small.
    w = max(1, n // int(EP_ORBITS))
    dt_days = max((n - w) * env._advisor_s / 86400.0, 1e-9)
    decay = float((np.mean(sma[:w]) - np.mean(sma[-w:])) / 1e3 / dt_days)
    alt0 = float(opt.get("altitude_km", 400.0))
    lifetime = float(np.clip((alt0 - 250.0) / max(decay, 1e-3), 0.0, 3650.0))
    return (decay, float(np.mean(soc)), float(np.min(soc)),
            float(np.sum(np.diff(dep.astype(int)) > 0) + (1 if dep[:1].any() else 0)),
            float(np.mean(dl_l)), float(np.nanmean(cd_l)), float(np.mean(ef_l)),
            lifetime, float(n))


def worker(task):
    slug, kind, spec, panel, j0, j1 = task
    sys.path.insert(0, str(ROOT / "python"))
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from arlamx_v2.env import ArlamxV2Env

    predict_fn, env_kw = _build_policy(kind, spec)
    env = ArlamxV2Env(seed=0, variant="v7", **env_kw)
    ax = AXES[panel]
    xkey, xv = ax["x"]
    ykey, yv = ax["y"]
    out = {q: np.full((j1 - j0, GRID_N), np.nan) for q in QUANTS}
    for jj, j in enumerate(range(j0, j1)):        # rows (y)
        for i in range(GRID_N):                   # cols (x)
            opt = dict(BASE_OPT)
            opt.update(ax["base"])
            opt[xkey] = float(xv[i])
            opt[ykey] = float(yv[j])
            alt = float(opt.get("altitude_km", 400.0))
            period = 2 * np.pi * np.sqrt((6371e3 + alt * 1e3) ** 3 / 3.986004418e14)
            env._max_steps = int(EP_ORBITS * period / env._advisor_s)
            vals = _cell_metrics(env, predict_fn, opt)
            for k, q in enumerate(QUANTS):
                out[q][jj, i] = vals[k]
    env.close()
    return slug, panel, j0, j1, out


def main():
    tasks = []
    for panel in PANELS:
        for slug, kind, spec in POLICIES:
            edges = np.linspace(0, GRID_N, CHUNKS + 1).astype(int)
            for k in range(CHUNKS):
                if edges[k + 1] > edges[k]:
                    tasks.append((slug, kind, spec, panel,
                                  int(edges[k]), int(edges[k + 1])))
    pts = len(POLICIES) * len(PANELS) * GRID_N * GRID_N
    print(f"[gradient70] {len(POLICIES)} policies x {len(PANELS)} panels x "
          f"{GRID_N}x{GRID_N} = {pts} cells in {len(tasks)} tasks on "
          f"{MAX_WORKERS} workers", flush=True)

    results = {}
    done_n = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for slug, panel, j0, j1, out in ex.map(worker, tasks):
            key = f"{panel}__{slug}"
            r = results.setdefault(key, {q: np.full((GRID_N, GRID_N), np.nan)
                                         for q in QUANTS})
            for q in QUANTS:
                r[q][j0:j1, :] = out[q]
            done_n += 1
            print(f"[chunk {done_n}/{len(tasks)}] {key} rows[{j0}:{j1}] "
                  f"decay~{np.nanmean(out['decay']):.2f}", flush=True)

    outdir = ROOT / "outputs" / ".old" / "analysis" / "gradient70"
    outdir.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        ax = AXES[panel]
        np.savez_compressed(
            outdir / (f"grid70_{panel}{'_test' if TEST else ''}.npz"),
            X=ax["x"][1], Y=ax["y"][1],
            XKEY=np.array(ax["x"][0]), YKEY=np.array(ax["y"][0]),
            SLUGS=np.array([p[0] for p in POLICIES]),
            **{f"{slug}__{q}": results[f"{panel}__{slug}"][q]
               for slug, _k, _s in POLICIES for q in QUANTS
               if f"{panel}__{slug}" in results})
    print(f"[all done] -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
