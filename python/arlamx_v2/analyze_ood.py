"""OOD / gradient / weather-spike / brownout-recovery analysis.

Train box: alt 300-500 km, i 20-30 deg, F10.7 65-250.
Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from arlamx_v2.advisors import HEURISTIC_BANK, MinDragPolicy, SamplingMpcPolicy
from arlamx_v2.env import ArlamxV2Env

ROOT = Path(__file__).resolve().parents[2]

RESET_BASE = {"ecc": 0.001, "nu_deg": 90.0, "mass_kg": 0.625}


def _load(name):
    from stable_baselines3 import PPO, SAC

    from arlamx_v2.paths import run_dir as _run_dir
    folder = _run_dir(name) / "models"
    algo = "sac" if "SAC" in name else "ppo"
    z = folder / f"{algo}_{name}.zip"
    if not z.exists():
        return None
    cls = SAC if algo == "sac" else PPO
    return cls.load(str(z))


def _policies():
    out = {}
    md = MinDragPolicy()
    out["min_drag"] = ("v4a", lambda obs, env: md.predict(obs, env.sim.get_state())[0])
    h = HEURISTIC_BANK["mission_v17"]
    h.reset()

    def hf(obs, env, pol=h):
        st = env.sim.get_state()
        st["battery_soc"] = env._batt_E / env._batt_cap_J
        return pol.predict(obs, st)[0]

    out["heuristic"] = ("v4a", hf)
    mpc = SamplingMpcPolicy(horizon_steps=4, n_random=4, seed=0)
    mpc.reset()

    def mf(obs, env, pol=mpc):
        st = env.sim.get_state()
        st["battery_soc"] = env._batt_E / env._batt_cap_J
        return pol.predict(obs, st)[0]

    out["mpc"] = ("v4a", mf)
    for name in (
        "SC_v4a_4x16_PPO_300k",
        "SC_v4b_4x16_PPO_300k",
        "SC_v4a_4x16_SAC_300k",
        "SC_v4b_4x16_SAC_300k",
        "SC_v5a_4x16_PPO_tuned",
        "SC_v5b_4x16_PPO_tuned",
        "SC_v6_4x16_PPO_100k",
        "SC_v6_4x16_PPO_250k",
        "SC_v6_4x16_PPO_500k",
        "SC_v6_4x16_PPO_4000k",
    ):
        m = _load(name)
        if m is None:
            continue
        if "v6" in name:
            var = "v6"
        elif "v5b" in name:
            var = "v5b"
        elif "v5a" in name:
            var = "v5a"
        elif "v4b" in name:
            var = "v4b"
        else:
            var = "v4a"

        def fn(obs, env, model=m):
            a, _ = model.predict(obs, deterministic=True)
            return a

        out[name] = (var, fn)
    return out


def _pad_action(act, env):
    a = np.asarray(act, float).reshape(-1)
    n = int(env.action_space.shape[0])
    if a.size == n:
        return a.astype(np.float32)
    if a.size > n:
        return a[:n].astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: a.size] = a
    if n == 5:
        out[4] = 1.0
    return out


def plot_lines(rows, xkey, ykey, hue, png, title, xlabel, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    names = sorted({r[hue] for r in rows})
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for name in names:
        xs = [r[xkey] for r in rows if r[hue] == name]
        ys = [r[ykey] for r in rows if r[hue] == name]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", label=name, lw=1.2, ms=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=130, facecolor="white")
    plt.close(fig)


def _run_episode(env, fn, n_steps, reset_opt, weather=None, srp_flash=None):
    obs, _ = env.reset(seed=int(reset_opt.get("seed", 0)), options=reset_opt)
    ret = 0.0
    socs, alts, gens, smas = [], [], [], []
    rec_at = -1
    rec_stable = -1
    for k in range(n_steps):
        if weather is not None and k == weather["at"]:
            env._f107 = float(weather["f107"])
            env._ap = float(weather["ap"])
        if srp_flash is not None and srp_flash["at"] <= k < srp_flash["at"] + srp_flash["dur"]:
            env._srp_now = float(srp_flash["scale"])
            env.sim.set_srp_scale(float(srp_flash["scale"]))
        act = _pad_action(fn(obs, env), env)
        obs, rew, term, trunc, info = env.step(act)
        ret += float(rew)
        socs.append(float(info.get("battery_soc", 0.0)))
        alts.append(float(info.get("altitude_km", 0.0)))
        gens.append(float(info.get("power_gen_norm", 0.0)))
        smas.append(float(info.get("sma_m", 0.0)))
        if rec_at < 0 and info.get("brownout_recovered", False):
            rec_at = k + 1
        if rec_stable < 0 and rec_at > 0 and (not info.get("brownout_active", False)):
            omg = float(info.get("omega_dps", 99.0))
            if omg <= 0.5 and socs[-1] >= 0.15:
                rec_stable = k + 1
        if term or trunc:
            break
    return {
        "return": ret,
        "soc_min": float(np.min(socs)) if socs else 0.0,
        "soc_end": socs[-1] if socs else 0.0,
        "dalt": (alts[-1] - alts[0]) if len(alts) > 1 else 0.0,
        "steps": len(socs),
        "recover_steps": rec_at,
        "stabilize_steps": rec_stable,
        "gen_mean": float(np.mean(gens)) if gens else 0.0,
        "alt0": alts[0] if alts else 0.0,
        "alt1": alts[-1] if alts else 0.0,
        "dsma_km": ((smas[-1] - smas[0]) / 1e3) if len(smas) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--recover-steps", type=int, default=80)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / ".old" / "analysis" / "ood")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pols = _policies()
    alt_rows, wx_rows, inc_rows, rec_rows, storm_rows, flash_rows = [], [], [], [], [], []

    alts = [260, 300, 350, 400, 500, 550, 650]
    f107s = [65, 100, 150, 200, 250, 300]
    incs = [0, 23, 35, 51.6, 80, 97.4]
    nus = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    storms = [
        {"name": "quiet", "f107": 70, "ap": 4, "at": 8},
        {"name": "moderate", "f107": 150, "ap": 15, "at": 8},
        {"name": "storm", "f107": 220, "ap": 40, "at": 8},
        {"name": "superstorm", "f107": 300, "ap": 80, "at": 8},
    ]

    for pname, (var, fn) in pols.items():
        print("ood", pname, flush=True)
        for alt in alts:
            env_a = ArlamxV2Env(seed=2, variant=var)
            rec = _run_episode(
                env_a,
                fn,
                args.steps,
                {**RESET_BASE, "altitude_km": alt, "inc_deg": 23.0, "f107": 150.0, "ap": 4.0, "soc": 0.5, "seed": 2},
            )
            alt_rows.append(
                {
                    "policy": pname,
                    "alt_km": alt,
                    "ood": alt < 300 or alt > 500,
                    "return": rec["return"],
                    "soc_min": rec["soc_min"],
                    "dalt": rec["dalt"],
                    "dsma_km": rec["dsma_km"],
                    "steps": rec["steps"],
                }
            )
            env_a.close()
        for f in f107s:
            env_w = ArlamxV2Env(seed=3, variant=var)
            rec = _run_episode(
                env_w,
                fn,
                args.steps,
                {**RESET_BASE, "altitude_km": 400.0, "inc_deg": 23.0, "f107": 150.0, "ap": 4.0, "soc": 0.5, "seed": 3},
                weather={"at": 6, "f107": f, "ap": 40.0 if f >= 200 else 4.0},
            )
            wx_rows.append(
                {
                    "policy": pname,
                    "f107_spike": f,
                    "return": rec["return"],
                    "soc_min": rec["soc_min"],
                    "soc_end": rec["soc_end"],
                    "dalt": rec["dalt"],
                }
            )
            env_w.close()
        for storm in storms:
            env_s = ArlamxV2Env(seed=7, variant=var)
            rec = _run_episode(
                env_s,
                fn,
                args.steps,
                {**RESET_BASE, "altitude_km": 400.0, "inc_deg": 23.0, "f107": 150.0, "ap": 4.0, "soc": 0.5, "seed": 7},
                weather={"at": storm["at"], "f107": storm["f107"], "ap": storm["ap"]},
            )
            storm_rows.append(
                {
                    "policy": pname,
                    "storm": storm["name"],
                    "f107": storm["f107"],
                    "ap": storm["ap"],
                    "return": rec["return"],
                    "soc_min": rec["soc_min"],
                    "dalt": rec["dalt"],
                }
            )
            env_s.close()
        env_f = ArlamxV2Env(seed=8, variant=var)
        rec = _run_episode(
            env_f,
            fn,
            args.steps,
            {**RESET_BASE, "altitude_km": 500.0, "inc_deg": 23.0, "f107": 150.0, "ap": 4.0, "soc": 0.5, "seed": 8},
            srp_flash={"at": 4, "dur": 6, "scale": 5.0},
        )
        flash_rows.append(
            {
                "policy": pname,
                "return": rec["return"],
                "soc_min": rec["soc_min"],
                "dalt": rec["dalt"],
            }
        )
        env_f.close()
        for inc in incs:
            env_i = ArlamxV2Env(seed=4, variant=var)
            rec = _run_episode(
                env_i,
                fn,
                args.steps,
                {**RESET_BASE, "altitude_km": 400.0, "inc_deg": inc, "f107": 150.0, "ap": 4.0, "soc": 0.5, "seed": 4},
            )
            inc_rows.append(
                {
                    "policy": pname,
                    "inc_deg": inc,
                    "ood": inc < 20 or inc > 30,
                    "return": rec["return"],
                    "soc_min": rec["soc_min"],
                }
            )
            env_i.close()
        if var in ("v4b", "v5b", "v6"):
            for nu in nus:
                env_b = ArlamxV2Env(seed=5, variant=var)
                rec = _run_episode(
                    env_b,
                    fn,
                    args.recover_steps,
                    {
                        **RESET_BASE,
                        "altitude_km": 400.0,
                        "inc_deg": 23.0,
                        "nu_deg": nu,
                        "soc": 0.0,
                        "force_brownout": True,
                        "f107": 150.0,
                        "ap": 4.0,
                        "omega_dps": [4.0, -3.5, 3.0],
                        "seed": 5,
                    },
                )
                rec_rows.append(
                    {
                        "policy": pname,
                        "nu_deg": nu,
                        "recover_steps": rec["recover_steps"],
                        "stabilize_steps": rec["stabilize_steps"],
                        "recover_min": rec["recover_steps"] * env_b._advisor_s / 60.0 if rec["recover_steps"] > 0 else -1,
                        "stabilize_min": rec["stabilize_steps"] * env_b._advisor_s / 60.0 if rec["stabilize_steps"] > 0 else -1,
                        "soc_end": rec["soc_end"],
                    }
                )
                env_b.close()

    def dump(name, rows):
        if not rows:
            return
        with open(args.out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("alt_sweep", alt_rows)
    dump("weather_spike", wx_rows)
    dump("inc_sweep", inc_rows)
    dump("brownout_phase", rec_rows)
    dump("storm_sweep", storm_rows)
    dump("srp_flash_500km", flash_rows)
    plot_lines(alt_rows, "alt_km", "return", "policy", args.out / "grad_return_vs_alt.png",
               "Return vs altitude (OOD <300 and >500 km)", "Start altitude [km]", "episode return")
    plot_lines(alt_rows, "alt_km", "soc_min", "policy", args.out / "grad_soc_vs_alt.png",
               "Min SoC vs altitude", "Start altitude [km]", "min SoC")
    plot_lines(alt_rows, "alt_km", "dalt", "policy", args.out / "grad_dalt_vs_alt.png",
               "Geodetic altitude change vs start altitude", "Start altitude [km]", "delta altitude [km]")
    plot_lines(alt_rows, "alt_km", "dsma_km", "policy", args.out / "grad_dsma_vs_alt.png",
               "SMA change vs start altitude (decay proxy)", "Start altitude [km]", "delta SMA [km]")
    plot_lines(wx_rows, "f107_spike", "soc_min", "policy", args.out / "grad_soc_vs_f107spike.png",
               "Min SoC after F10.7 spike at step 6", "F10.7 after spike", "min SoC")
    plot_lines(wx_rows, "f107_spike", "return", "policy", args.out / "grad_return_vs_f107spike.png",
               "Return after F10.7 spike at step 6", "F10.7 after spike", "episode return")
    plot_lines(inc_rows, "inc_deg", "return", "policy", args.out / "grad_return_vs_inc.png",
               "Return vs inclination (train band 20-30 deg)", "inclination [deg]", "episode return")
    plot_lines(storm_rows, "ap", "soc_min", "policy", args.out / "grad_soc_vs_apstorm.png",
               "Min SoC after geomagnetic storm", "Ap after spike", "min SoC")
    if rec_rows:
        plot_lines(rec_rows, "nu_deg", "recover_steps", "policy", args.out / "grad_recover_vs_nu.png",
                   "Advisor steps to hand back after forced brownout", "true anomaly [deg]", "advisor steps")
        plot_lines(rec_rows, "nu_deg", "stabilize_steps", "policy", args.out / "grad_stabilize_vs_nu.png",
                   "Steps to stabilize (SoC>=0.15, |w|<=0.5 deg/s)", "true anomaly [deg]", "advisor steps")
        plot_lines(rec_rows, "nu_deg", "recover_min", "policy", args.out / "grad_recover_min_vs_nu.png",
                   "Minutes to recover after forced brownout", "true anomaly [deg]", "minutes")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "n_policies": len(pols),
                "policies": list(pols),
                "alts": alts,
                "steps": args.steps,
                "recover_steps": args.recover_steps,
            },
            indent=2,
        )
    )
    print("wrote", args.out, "policies", list(pols))


if __name__ == "__main__":
    main()
