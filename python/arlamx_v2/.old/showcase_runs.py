"""Showcase episode recorder: full time-series for the conference figure set.

Runs each policy (MPC, heuristic, min-drag, RL PPO/SAC/TD3) through two
deterministic scenarios — a 2-day nominal episode and a 1-day episode with a
mid-episode storm spike — recording everything the figures and the 3D
animation need per advisor step (attitude, torques, gates, observer, power,
pointing, downlink). Writes one npz per (policy, scenario).

Usage:  PYTHONPATH=python python -m arlamx_v2.showcase_runs
Outputs -> outputs/analysis/showcase/{policy}__{scenario}.npz
Level: advanced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from arlamx_v2 import cpp                                  # noqa: E402
from arlamx_v2.env import ArlamxV2Env                      # noqa: E402
from arlamx_v2.gradient70 import POLICIES, _build_policy   # noqa: E402

OUT = ROOT / "outputs" / "analysis" / "showcase"
SEED = 314

SCENARIOS = {
    # 2-day nominal mission at the 400 km reference orbit
    "nominal2d": dict(opt=dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001,
                               f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                               nu_deg=0.0, mass_kg=0.625, soc=0.5,
                               omega_dps=[1.0, 1.0, 0.5]),
                      days=2.0, jumps=[]),
    # 1-day episode, severe storm arriving at 30% and clearing at 70%
    "storm1d": dict(opt=dict(altitude_km=350.0, inc_deg=23.0, ecc=0.001,
                             f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                             nu_deg=0.0, mass_kg=0.625, soc=0.5,
                             omega_dps=[1.0, 1.0, 0.5]),
                    days=1.0, jumps=[(0.30, 130.0, 76.0), (0.70, -130.0, -76.0)]),
    # 1-day post-deployment recovery: supercap nearly empty, hard tumble,
    # brownout mode engaged from the start -> B-dot detumble -> recovery ->
    # the advisor takes over.
    "recovery1d": dict(opt=dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001,
                                f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                                nu_deg=0.0, mass_kg=0.625, soc=0.02,
                                omega_dps=[4.0, 4.0, 2.0], force_brownout=True),
                       days=1.0, jumps=[]),
}

KEYS = ["t_h", "sigma", "r", "v", "sun_N", "eclipse", "gates", "tau_ctrl",
        "tau_dist", "tau_aero", "effort", "soc", "gen", "gs_vis", "gs_cos",
        "gs_err_axes", "downlink", "cmd_track_deg", "q_cmd", "cd", "sma_km",
        "alt_km", "f107", "ap", "omega_dps", "brownout",
        "dE_drag", "dE_lift", "dE_srp"]


def record(slug, kind, spec, scen_name):
    scen = SCENARIOS[scen_name]
    predict_fn, env_kw = _build_policy(kind, spec)
    env = ArlamxV2Env(seed=0, variant="v7", **env_kw)
    env._max_steps = int(scen["days"] * 86400.0 / env._advisor_s)
    obs, _ = env.reset(seed=SEED, options=dict(scen["opt"]))
    env._weather_jumps = [(f * env._max_steps, df, da) for f, df, da in scen["jumps"]]
    env._srp_flashes = []
    H = {k: [] for k in KEYS}
    last = {"power_gen_norm": 0.0}
    done, n = False, 0
    while not done:
        a = predict_fn(obs, env, last)
        a = np.asarray(a, dtype=np.float32)
        obs, _r, term, trunc, info = env.step(a)
        done = term or trunc
        n += 1
        last["power_gen_norm"] = float(info.get("power_gen_norm", 0.0))
        st = env.sim.get_state()
        sigma = np.asarray(st["sigma"], float)
        c_bn = cpp.mrp_to_dcm(sigma)
        # command-vs-achieved: angle between the commanded attitude (unit
        # quaternion from the action) and the achieved attitude.
        q_cmd = a[:4] / max(np.linalg.norm(a[:4]), 1e-9)
        c_cmd = cpp.mrp_to_dcm(np.asarray(
            [q_cmd[1], q_cmd[2], q_cmd[3]], float) / max(1.0 + q_cmd[0], 1e-9))
        r_err = c_cmd @ c_bn.T
        ang = float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(r_err) - 1.0),
                                                 -1.0, 1.0))))
        vis = float(info.get("gs_visible", 0.0))
        lit = float(info.get("eclipse", 1.0))
        cosg = float(info.get("gs_point_cos", 0.0))
        H["t_h"].append(n * env._advisor_s / 3600.0)
        H["sigma"].append(sigma)
        H["r"].append(np.asarray(st["r"], float))
        H["v"].append(np.asarray(st["v"], float))
        H["sun_N"].append(np.asarray(st.get("sun_N", np.zeros(3)), float))
        H["eclipse"].append(lit)
        H["gates"].append(np.asarray(info.get("gates", np.ones(3)), float))
        H["tau_ctrl"].append(np.asarray(info.get("tau_ctrl_mean", np.zeros(3)), float))
        H["tau_dist"].append(np.asarray(info.get("tau_dist_est", np.zeros(3)), float))
        H["tau_aero"].append(np.asarray(info.get("tau_aero_mean", np.zeros(3)), float))
        H["effort"].append(float(info.get("torque_effort", 0.0)))
        H["soc"].append(float(info["battery_soc"]))
        H["gen"].append(float(info.get("power_gen_norm", 0.0)))
        H["gs_vis"].append(vis)
        H["gs_cos"].append(cosg)
        H["gs_err_axes"].append(np.asarray(info.get("gs_err_axes_deg", np.zeros(3)),
                                           float))
        H["downlink"].append(1.0 if (vis > 0.5 and lit > 0.5 and cosg > 0.7) else 0.0)
        H["cmd_track_deg"].append(ang)
        H["q_cmd"].append(q_cmd)
        H["cd"].append(float(info.get("Cd", np.nan)))
        H["sma_km"].append(float(info["sma_m"]) / 1e3)
        H["alt_km"].append(float(info["altitude_km"]))
        H["f107"].append(float(env._f107))
        H["ap"].append(float(env._ap))
        H["omega_dps"].append(float(info.get("omega_dps", 0.0)))
        H["brownout"].append(1.0 if info.get("brownout_active") else 0.0)
        H["dE_drag"].append(float(info.get("dE_drag", 0.0)))
        H["dE_lift"].append(float(info.get("dE_lift", 0.0)))
        H["dE_srp"].append(float(info.get("dE_srp", 0.0)))
    env.close()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{slug}__{scen_name}.npz",
                        **{k: np.asarray(v) for k, v in H.items()})
    soc = np.asarray(H["soc"])
    print(f"[showcase] {slug}/{scen_name}: {n} steps, "
          f"soc {soc.min():.2f}-{soc.max():.2f}, "
          f"downlink {int(np.sum(H['downlink']))} steps, "
          f"track_err median {np.median(H['cmd_track_deg']):.1f} deg", flush=True)


def main():
    # geometry (for the 3D animation) saved once
    from arlamx_v2.geometry import load_geom
    from arlamx_v2.paths import HEX_GEOM
    n, a, c = load_geom(str(HEX_GEOM))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "geometry.npz", normals=n, areas=a, centroids=c)

    for slug, kind, spec in POLICIES:
        for scen in SCENARIOS:
            record(slug, kind, spec, scen)
    print("[showcase] DONE", flush=True)


if __name__ == "__main__":
    main()
