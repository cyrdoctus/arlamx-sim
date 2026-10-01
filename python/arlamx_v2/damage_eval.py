"""SC_v11 damage verification: hole / tear / multi, measured and animated.

For each damage kind and each policy, forced-strike episodes measure what the
strike costs and how fast the policy re-trims:

    pre/post decay rate         does the wounded craft still hold its orbit
    pre/post tracking error     does the controller re-trim
    recovery time               steps until tracking error returns under
                                2x the pre-strike median
    post-strike delegation B    does the policy still use the (changed)
                                environmental torque correctly

Policies: the best v11 (trained on strikes), the best v10 (identical brief,
never saw a strike — the ablation), and the MPC (whose internal Sentman model
silently keeps the PRE-strike panel table: exactly what an onboard model-based
controller would suffer).

    PYTHONPATH=python python -m arlamx_v2.damage_eval --stage eval
    PYTHONPATH=python python -m arlamx_v2.damage_eval --stage anim
Outputs: outputs/v8/data/damage_eval.json, damage traces npz,
         outputs/presentation/animations_v11/*.mp4/.gif
Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from arlamx_v2 import bench_v8 as B
from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

V8 = OUTPUTS / "v8"
ANIM = OUTPUTS / "presentation" / "animations_v11"
KINDS = ("hole", "tear", "multi")
SEEDS = tuple(range(3001, 3009))
STRIKE_FRAC = 0.35
EP_ORBITS = 30.0


def _best(variant):
    rows = sorted((r for r in csv.DictReader(open(V8 / "ledger.csv"))
                   if r["variant"] == variant),
                  key=lambda r: float(r["gap_index"]))
    return rows[0] if rows else None


def _policies():
    out = [("MPC", B.mpc_fn())]
    for var in ("v10", "v11"):
        r = _best(var)
        if r:
            out.append((f"{var} {r['algo']} {r['arch']}",
                        B.sb3_fn(B.load_model(r["algo"], r["name"]))))
    return out


def _episode(predict, kind, seed, variant="v11", record=False):
    env = ArlamxV2Env(seed=seed, variant=variant)
    period = 2 * np.pi * np.sqrt((6371e3 + 400e3) ** 3 / 3.986004418e14)
    env._max_steps = int(EP_ORBITS * period / env._advisor_s)
    obs, _ = env.reset(seed=seed, options=dict(
        altitude_km=400.0, inc_deg=25.0, ecc=0.001, f107=150.0, ap=8.0,
        soc=0.5, omega_dps=[0.5, 0.5, 0.2],
        damage_kind=kind, damage_step_frac=STRIKE_FRAC))
    H = {k: [] for k in ("t_h", "alt", "sma", "terr", "B", "split", "Cd",
                         "sigma", "r", "sun_N", "tau_env", "tau_aero",
                         "soc", "gates", "damage")}
    done, k = False, 0
    while not done:
        a = predict(obs, env)
        obs, _r, te, tr, info = env.step(np.asarray(a, np.float32))
        done = te or tr
        k += 1
        H["t_h"].append(k * env._advisor_s / 3600.0)
        H["alt"].append(float(info["altitude_km"]))
        H["sma"].append(float(info["sma_m"]))
        H["terr"].append(float(info.get("tracking_err_rad", 0.0)))
        H["B"].append(float(info.get("deleg_B", 0.0)))
        H["split"].append(np.asarray(info.get("deleg_split", np.zeros(3))).tolist())
        H["Cd"].append(float(info.get("Cd", np.nan)))
        H["soc"].append(float(info["battery_soc"]))
        H["gates"].append(np.asarray(info.get("gates", np.ones(3))).tolist())
        H["damage"].append(1.0 if info.get("damage_active") else 0.0)
        if record:
            st = env.sim.get_state()
            H["sigma"].append(np.asarray(st["sigma"], float).tolist())
            H["r"].append(np.asarray(st["r"], float).tolist())
            H["sun_N"].append(np.asarray(st.get("sun_N", np.zeros(3)), float).tolist())
            H["tau_env"].append(float(np.linalg.norm(info.get("tau_env_filt", np.zeros(3)))))
            H["tau_aero"].append(float(np.linalg.norm(info.get("tau_aero_mean", np.zeros(3)))))
    summ = dict(info.get("damage_summary") or {})
    plan = getattr(env, "_damage_plan", None)
    kept = getattr(env, "_damage_kept", None)
    env.close()
    return H, summ, plan, (kept.tolist() if kept is not None else None)


def _metrics(H):
    dmg = np.asarray(H["damage"]) > 0.5
    if not dmg.any():
        return None
    i0 = int(np.argmax(dmg))
    sma = np.asarray(H["sma"]) / 1e3
    t_d = np.asarray(H["t_h"]) / 24.0
    terr = np.asarray(H["terr"])
    Bv = np.asarray(H["B"])

    def rate(a, b):
        if b - a < 3:
            return None
        return float((sma[a] - sma[b - 1]) / max(t_d[b - 1] - t_d[a], 1e-9))
    pre_ref = float(np.median(terr[max(0, i0 - 40):i0])) if i0 > 3 else 0.0
    rec = None
    for j in range(i0 + 1, len(terr)):
        if terr[j] <= max(2.0 * pre_ref, 1e-4):
            rec = j - i0
            break
    return {
        "decay_pre": rate(max(0, i0 - 60), i0),
        "decay_post": rate(i0, len(sma)),
        "terr_pre_deg": float(np.degrees(pre_ref)),
        "terr_post_deg": float(np.degrees(np.median(terr[i0:]))),
        "recovery_steps": rec,
        "B_pre": float(np.mean(Bv[max(0, i0 - 40):i0])),
        "B_post": float(np.mean(Bv[i0:])),
    }


def stage_eval():
    pols = _policies()
    out = {}
    for label, fn in pols:
        out[label] = {}
        for kind in KINDS:
            rows = []
            for sd in SEEDS:
                H, summ, _plan, _kept = _episode(fn, kind, sd)
                m = _metrics(H)
                if m:
                    m["area_lost_frac"] = summ.get("area_lost_frac")
                    m["cp_shift_m"] = summ.get("cp_shift_m")
                    rows.append(m)
            agg = {}
            for k in rows[0]:
                vals = [r[k] for r in rows if r[k] is not None]
                agg[k] = float(np.median(vals)) if vals else None
            out[label][kind] = {"n": len(rows), "median": agg}
            print(f"[dmg] {label:18s} {kind:5s}: decay {agg['decay_pre'] and round(agg['decay_pre'],2)}->"
                  f"{agg['decay_post'] and round(agg['decay_post'],2)} km/d  "
                  f"terr {agg['terr_pre_deg']:.2f}->{agg['terr_post_deg']:.2f} deg  "
                  f"rec {agg['recovery_steps']} steps  B {agg['B_pre']:.2f}->{agg['B_post']:.2f}",
                  flush=True)
    (V8 / "data" / "damage_eval.json").write_text(json.dumps(out, indent=2))
    print("[dmg] wrote damage_eval.json")


# ---------------------------------------------------------------------------
# Animation: the strike happens ON CAMERA
# ---------------------------------------------------------------------------

def stage_anim():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation as manim
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from arlamx_v2 import ipc_animation as IPC
    from arlamx_v2.damage import _removal_fraction

    ANIM.mkdir(parents=True, exist_ok=True)
    tris, _tn = IPC.simplified_model()
    tri_cent = np.asarray(tris).mean(axis=1)

    for label, fn in _policies():
        if label == "MPC" and _best("v11") is None:
            continue
        slug = label.split()[0].lower()
        H, summ, plan, _kept = _episode(fn, "tear", 3001, record=True)
        if plan is None:
            continue
        dmg = np.asarray(H["damage"]) > 0.5
        i_strike = int(np.argmax(dmg)) if dmg.any() else -1
        # CAD triangles inside the tear disappear at the strike frame.
        gone = np.array([max((_removal_fraction(c[0], c[1], h)
                              for h in plan["holes"]), default=0.0) > 0.5
                         for c in tri_cent])
        n = len(H["t_h"])
        lo, hi = max(0, i_strike - 20), min(n, i_strike + 60)

        fig = plt.figure(figsize=(12.6, 6.4))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], hspace=0.45,
                              left=0.02, right=0.96, top=0.88, bottom=0.10)
        ax3 = fig.add_subplot(gs[:, 0], projection="3d")
        axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
        title = fig.suptitle("", fontsize=12)
        t = np.asarray(H["t_h"][lo:hi])
        for ax, series, lab in ((axes[0], np.degrees(H["terr"][lo:hi]), "tracking error (deg)"),
                                (axes[1], H["Cd"][lo:hi], "realized Cd"),
                                (axes[2], H["B"][lo:hi], "delegation benefit B")):
            ax.plot(t, series, lw=1.2, color="#4a3aa7")
            ax.axvline(H["t_h"][i_strike], color="#e34948", lw=1.4, ls="--")
            ax.set_ylabel(lab, fontsize=8)
            ax.grid(True, color="0.9", lw=0.5)
        axes[2].set_xlabel("mission time (h)", fontsize=8)
        cursors = [ax.axvline(t[0], color="0.1", lw=0.9) for ax in axes]
        coll = Poly3DCollection([], linewidths=0.15, edgecolors=(0, 0, 0, 0.3))
        ax3.add_collection3d(coll)
        ext = np.abs(tris).max() * 1.25

        def draw(f):
            k = lo + f
            struck = k >= i_strike
            sig = np.asarray(H["sigma"][k])
            c_nb = np.asarray(IPC._quat_dcm_bn(IPC._mrp_to_quat(sig))).T
            keep = ~gone if struck else np.ones(len(tris), bool)
            tt = np.einsum("ij,tkj->tki", c_nb, np.asarray(tris)[keep])
            coll.set_verts(tt)
            coll.set_facecolor("#8a2b2b" if struck else "#5b6470")
            ax3.set_xlim(-ext, ext); ax3.set_ylim(-ext, ext); ax3.set_zlim(-ext, ext)
            ax3.set_axis_off()
            ax3.view_init(elev=18, azim=35)
            area = 1.0 - (summ.get("area_lost_frac", 0.0) if struck else 0.0)
            title.set_text(
                f"{label} — membrane tear at t = {H['t_h'][i_strike]:.1f} h   |   "
                f"{'AFTER STRIKE' if struck else 'nominal'}   "
                f"membrane {area*100:.1f} %   cp shift "
                f"{(summ.get('cp_shift_m', 0.0)*100 if struck else 0):.1f} cm   "
                f"Cd {H['Cd'][k]:.2f}")
            for cur in cursors:
                cur.set_xdata([H["t_h"][k]])
            return [coll]

        anim = manim.FuncAnimation(fig, draw, frames=hi - lo, interval=80, blit=False)
        stem = ANIM / f"damage_{slug}_tear"
        anim.save(str(stem) + ".mp4", writer=manim.FFMpegWriter(fps=12, bitrate=2200))
        anim.save(str(stem) + ".gif", writer=manim.PillowWriter(fps=6))
        plt.close(fig)
        print(f"[dmg-anim] wrote {stem}.mp4/.gif ({hi-lo} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("eval", "anim"))
    a = ap.parse_args()
    if a.stage == "eval":
        stage_eval()
    else:
        stage_anim()


if __name__ == "__main__":
    main()
