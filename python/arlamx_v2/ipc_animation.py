"""3D animation: the V2-simplified spacecraft responding to disturbances (IPC).

The spacecraft model is the REAL SolarCat CAD run through the V2.1 sealed-side
simplifier (`geometry.simplify_sides`, quality 1pct) — the actual simplified
triangles the algorithm emits, sun-shaded per plate, dimmed in eclipse. NOT the
V1.7 hand-made .geom.

Left: the simplified model in inertial axes — attitude from the recorded
episode — with Sun / velocity / antenna (+Z) / disturbance-estimate arrows.
Right: live side panels sweeping a time cursor — disturbance detection
(observer vs plant truth), per-axis authority gates, magnetorquer power, SoC.

Segments — one animation per mission segment, auto-located in the recording:
  full           the whole episode
  quiet_cruise   the first quiet hours (before any storm)
  storm_onset    1 h before the storm arrives -> 3 h after
  eclipse_pass   one full eclipse transit (with sunlit margins)
  downlink_pass  a window around an actual downlink event

Usage:
  PYTHONPATH=python python -m arlamx_v2.ipc_animation \
      [--policy rl_ppo] [--scenario storm1d] [--segments all|name,name,...]
Outputs -> outputs/analysis/showcase/ipc_{policy}_{scenario}_{segment}.{mp4,gif}
Level: advanced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
SHOW = ROOT / "outputs" / ".old" / "analysis" / "showcase"

from arlamx_v2 import cpp                                   # noqa: E402
from arlamx_v2.geometry import load_mesh, simplify_sides    # noqa: E402
from arlamx_v2.paths import SOLARCAT_STL                    # noqa: E402

INK = "#33322e"
plt.rcParams.update({
    "font.size": 8.5, "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def simplified_model():
    """SolarCat CAD -> V2 sealed-side plates (cached). Returns (tris, normals):
    tris (T,3,3) body-frame triangle vertices centred on the mesh centroid,
    normals (T,3) unit."""
    cache = SHOW / "geometry_simplified.npz"
    if cache.exists():
        d = np.load(cache)
        return d["tris"], d["tri_normals"]
    V, F = load_mesh(str(SOLARCAT_STL), units="mm")
    _n, _A, _c, report, tris = simplify_sides(V, F, return_tris=True,
                                              quality="1pct")
    T = np.stack([np.asarray(t, float) for t in tris])
    T -= T.reshape(-1, 3).mean(axis=0)
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    n = np.cross(e1, e2)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    SHOW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tris=T, tri_normals=n)
    print(f"  simplified model: {len(T)} plates "
          f"(sides {report.get('n_sides', '?')}, quality 1pct)")
    return T, n


def find_segments(d):
    """Locate the mission segments in a recording. Returns {name: (i0, i1)}."""
    t = d["t_h"]
    n = len(t)
    segs = {"full": (0, n)}
    storm = d["ap"] > 40.0
    if np.any(storm):
        k = int(np.argmax(storm))
        segs["quiet_cruise"] = (0, max(2, k - 1))
        i0 = max(0, k - int(1.0 / (t[1] - t[0])))
        i1 = min(n, k + int(3.0 / (t[1] - t[0])))
        segs["storm_onset"] = (i0, i1)
    else:
        segs["quiet_cruise"] = (0, min(n, int(3.0 / (t[1] - t[0]))))
    ecl = d["eclipse"] <= 0.5          # in shadow
    if np.any(ecl):
        k = int(np.argmax(ecl))        # first eclipse entry
        k1 = k
        while k1 < n and ecl[k1]:
            k1 += 1
        pad = max(2, (k1 - k) // 2)
        segs["eclipse_pass"] = (max(0, k - pad), min(n, k1 + pad))
    bo = d["brownout"] > 0.5
    if np.any(bo):
        # from the browned-out start through recovery plus two hours of the
        # advisor back in control
        k_rec = int(np.argmax(~bo)) if np.any(~bo) else n
        segs["brownout_recovery"] = (0, min(n, k_rec + int(2.0 / (t[1] - t[0]))))
    dl = d["downlink"] > 0.5
    if np.any(dl):
        k = int(np.argmax(dl))
        pad = int(0.75 / (t[1] - t[0]))
        segs["downlink_pass"] = (max(0, k - pad), min(n, k + pad + 1))
    return segs


def _mrp_to_quat(sig):
    s2 = float(np.dot(sig, sig))
    q = np.array([(1.0 - s2), 2.0 * sig[0], 2.0 * sig[1], 2.0 * sig[2]]) / (1.0 + s2)
    n = np.linalg.norm(q)
    q = q / max(n, 1e-12)
    return -q if q[0] < 0 else q


def _slerp(qa, qb, u):
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    dot = min(1.0, dot)
    ang = np.arccos(dot)
    if ang < 1e-6:
        out = (1 - u) * qa + u * qb
    else:
        out = (np.sin((1 - u) * ang) * qa + np.sin(u * ang) * qb) / np.sin(ang)
    return out / max(np.linalg.norm(out), 1e-12)


def _quat_dcm_bn(q):
    q0, q1, q2, q3 = q
    return np.array([
        [q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
        [2*(q1*q2-q0*q3), q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3+q0*q1)],
        [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0*q0-q1*q1-q2*q2+q3*q3]])


def render(d, tris, tri_n, slug, scen, seg_name, i0, i1, interp=None):
    idx = np.arange(i0, i1)
    # Attitude interpolation: the recording is one sample per 300 s, so raw
    # playback jumps 5 minutes of slew per frame. SLERP `interp` sub-frames
    # between advisor steps for smooth, stable motion; the camera is FIXED.
    if interp is None:
        interp = 1 if len(idx) > 220 else max(1, int(np.ceil(360 / max(len(idx), 1))))
    quats = [_mrp_to_quat(np.asarray(d["sigma"][k], float)) for k in idx]
    frames = []            # (step_index_within_idx, dcm_bn, t_lerped)
    for j in range(len(idx)):
        if j == len(idx) - 1 or interp == 1:
            frames.append((j, _quat_dcm_bn(quats[j]), float(d["t_h"][idx[j]])))
            continue
        t0, t1 = float(d["t_h"][idx[j]]), float(d["t_h"][idx[j + 1]])
        for u in np.linspace(0.0, 1.0, interp, endpoint=False):
            frames.append((j, _quat_dcm_bn(_slerp(quats[j], quats[j + 1], u)),
                           t0 + u * (t1 - t0)))
    t = d["t_h"][idx]
    cap = np.array([1.8e-5, 1.8e-5, 5.7e-6])
    tau_d = np.linalg.norm(d["tau_dist"][idx], axis=1) * 1e6
    tau_a = np.linalg.norm(d["tau_aero"][idx], axis=1) * 1e6
    storm = d["ap"][idx] > 40.0
    ext = np.abs(tris).max() * 1.30

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1.0], hspace=0.42,
                          wspace=0.16, left=0.02, right=0.965, top=0.90,
                          bottom=0.09)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    title = fig.suptitle("", fontsize=12, y=0.965)

    specs = [("torque (µN·m)", None), ("axis authority gate", (0, 1.06)),
             ("MTQ power (mW)", None), ("supercap SoC (%)", (0, 105))]
    bo = d["brownout"][idx] > 0.5
    for ax, (lab, ylim) in zip(axes, specs):
        ax.fill_between(t, 0, 1, where=storm, color="#e34948", alpha=0.07,
                        transform=ax.get_xaxis_transform())
        ax.fill_between(t, 0, 1, where=bo, color="0.35", alpha=0.10,
                        transform=ax.get_xaxis_transform())
        ax.set_ylabel(lab, fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(t[0], t[-1])
        ax.grid(True, color="0.88", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    # Red matches the 3D disturbance arrow — the legend claimed that pairing and
    # the panel did not honour it.
    axes[0].plot(t, tau_a, color="0.55", lw=0.9, label="true aero torque")
    axes[0].plot(t, tau_d, color="#e34948", lw=1.3, ls=(0, (3, 2)),
                 label="disturbance est. (onboard)")
    axes[0].axhline(np.linalg.norm(cap) * 1e6, color="0.6", lw=0.7, ls=":")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    for i, (al, c) in enumerate(zip("XYZ", ("#2a78d6", "#eb6834", "#1baf7a"))):
        axes[1].step(t, d["gates"][idx, i], where="mid", color=c, lw=1.1, label=al)
    axes[1].legend(fontsize=6.5, frameon=False, ncol=3, loc="lower right")
    # Physical coil power, not the legacy linear proxy: a magnetorquer dissipates
    # I^2 R and its dipole is proportional to I, so P scales as torque squared.
    # See python/arlamx_v2/power.py. Falls back to the recorded effort trace if a
    # recording predates the per-axis torque log.
    if "tau_ctrl" in d and "mtq_power_mW" not in d:
        from arlamx_v2.power import mtq_power_w
        from arlamx_v2.config import load as _load
        _pc = _load("power_mtq")
        _bref = float(_pc["magnetorquers"]["b_ref_T"])
        pw = np.array([mtq_power_w(tc, np.array([0.0, 0.0, _bref]), _pc)[0]
                       for tc in d["tau_ctrl"][idx]]) * 1e3
    else:
        pw = d["effort"][idx] * 270.0
    axes[2].plot(t, pw, color="#4a3aa7", lw=1.1)
    axes[3].axhspan(40, 60, color="#1baf7a", alpha=0.10)
    axes[3].plot(t, d["soc"][idx] * 100, color="#eda100", lw=1.3)
    axes[3].set_xlabel("mission time (h)", fontsize=8)
    cursors = [ax.axvline(t[0], color=INK, lw=0.9) for ax in axes]

    coll = Poly3DCollection([], linewidths=0.15, edgecolors=(0, 0, 0, 0.25))
    ax3.add_collection3d(coll)
    quivers = {}
    mode_txt = ax3.text2D(0.02, 0.96, "", transform=ax3.transAxes, fontsize=11,
                          fontweight="bold", va="top")
    base_rgb = np.array([0.95, 0.76, 0.31])       # sunlit membrane gold
    dark_rgb = np.array([0.36, 0.39, 0.44])       # eclipse slate

    def draw(kk):
        j, c_bn, t_cur = frames[kk]
        k = idx[j]
        c_nb = c_bn.T
        world = tris @ c_nb.T                     # (T,3,3) body -> inertial
        lit = d["eclipse"][k] > 0.5
        sun_b = c_bn @ d["sun_N"][k]
        if lit:
            shade = 0.35 + 0.65 * np.abs(tri_n @ sun_b)   # two-sided membrane
            colors = np.clip(shade[:, None] * base_rgb[None, :], 0, 1)
        else:
            colors = np.tile(dark_rgb, (len(tris), 1))
        coll.set_verts(list(world))
        coll.set_facecolor(np.c_[colors, np.full(len(colors), 0.95)])

        for q in quivers.values():
            q.remove()
        quivers.clear()
        sun = d["sun_N"][k]
        v = d["v"][k] / max(np.linalg.norm(d["v"][k]), 1e-9)
        z_b = c_nb @ np.array([0.0, 0.0, 1.0])
        tau = d["tau_dist"][k]
        tau_nrm = c_nb @ (tau / max(np.linalg.norm(tau), 1e-12))
        # Arrows must be LONGER than the body's in-plane half-extent (ext/1.30)
        # or matplotlib's depth sort buries them inside the sail. The +Z antenna
        # arrow was visible only because the plate is thin in Z; the disturbance
        # arrow lies in the plate plane and was hidden every frame.
        s = ext * 1.15
        quivers["sun"] = ax3.quiver(0, 0, 0, *(sun * s), color="#eda100",
                                    lw=2.0, arrow_length_ratio=0.12, zorder=6)
        quivers["v"] = ax3.quiver(0, 0, 0, *(v * s * 0.92), color="0.45", lw=1.6,
                                  arrow_length_ratio=0.12, zorder=6)
        quivers["z"] = ax3.quiver(0, 0, 0, *(z_b * s * 0.98), color="#2a78d6",
                                  lw=2.0, arrow_length_ratio=0.14, zorder=7)
        quivers["tau"] = ax3.quiver(0, 0, 0, *(tau_nrm * s * 1.06),
                                    color="#e34948", lw=2.6,
                                    arrow_length_ratio=0.16, zorder=8)
        ax3.set_xlim(-ext, ext); ax3.set_ylim(-ext, ext); ax3.set_zlim(-ext, ext)
        ax3.set_axis_off()
        ax3.view_init(elev=18, azim=35)          # fixed camera — no orbiting

        # Flight mode: brownout + fast tumble -> B-dot detumble; brownout at
        # low rates -> sun-point recovery; otherwise the advisor commands.
        if d["brownout"][k] > 0.5:
            if d["omega_dps"][k] > 2.0:
                mode_txt.set_text("MODE: B-DOT DETUMBLE")
                mode_txt.set_color("#e34948")
            else:
                mode_txt.set_text("MODE: SUN-POINT RECOVERY")
                mode_txt.set_color("#c98500")
        else:
            mode_txt.set_text("MODE: ADVISOR IN CONTROL")
            mode_txt.set_color("#2a78d6")
        state = "SEVERE STORM" if storm[j] else "quiet"
        title.set_text(
            f"IPC — {slug.replace('_', ' ').upper()} · {seg_name.replace('_', ' ')}"
            f" · t = {d['t_h'][k]:5.1f} h · alt {d['alt_km'][k]:5.1f} km · "
            f"F10.7 {d['f107'][k]:.0f} / Ap {d['ap'][k]:.0f} ({state}) · "
            f"{'sunlit' if lit else 'eclipse'} · |ω| {d['omega_dps'][k]:.2f}°/s")
        for cur in cursors:
            cur.set_xdata([t_cur] * 2)
        return [coll, title, mode_txt, *cursors]

    for lbl, c in (("Sun", "#eda100"), ("velocity", "0.45"),
                   ("antenna +Z", "#2a78d6"), ("disturbance est.", "#e34948")):
        ax3.plot([], [], color=c, lw=2.0, label=lbl)
    ax3.legend(loc="lower left", fontsize=7.5, frameon=False)

    nfr = len(frames)
    stem = f"ipc_{slug}_{scen}_{seg_name}"
    anim = animation.FuncAnimation(fig, draw, frames=nfr, interval=50,
                                   blit=False)
    anim.save(str(SHOW / f"{stem}.mp4"),
              writer=animation.FFMpegWriter(fps=20, bitrate=2400))
    step = max(1, nfr // 90)
    anim2 = animation.FuncAnimation(fig, draw, frames=range(0, nfr, step),
                                    interval=125, blit=False)
    anim2.save(str(SHOW / f"{stem}.gif"), writer=animation.PillowWriter(fps=8))
    plt.close(fig)
    print(f"  wrote {stem}.mp4/.gif ({nfr} frames)")


def main():
    slug = "rl_ppo"
    scen = "storm1d"
    want = "all"
    if "--policy" in sys.argv:
        slug = sys.argv[sys.argv.index("--policy") + 1]
    if "--scenario" in sys.argv:
        scen = sys.argv[sys.argv.index("--scenario") + 1]
    if "--segments" in sys.argv:
        want = sys.argv[sys.argv.index("--segments") + 1]
    d = np.load(SHOW / f"{slug}__{scen}.npz")
    tris, tri_n = simplified_model()
    segs = find_segments(d)
    names = list(segs) if want == "all" else [w.strip() for w in want.split(",")]
    for name in names:
        if name not in segs:
            print(f"  [skip] segment '{name}' not present in {slug}/{scen}")
            continue
        i0, i1 = segs[name]
        render(d, tris, tri_n, slug, scen, name, i0, i1)


if __name__ == "__main__":
    main()
