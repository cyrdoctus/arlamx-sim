"""The two conference-slide figures (8-minute talk, technical audience).

  slide1_perturbation_exploitation  How the advisors exploit drag & SRP as
      actuation: (a) integrated non-gravitational work decomposition per
      policy; (b) the IPC mechanism during the storm (observer -> gates ->
      magnetorquer power); (c) who-wins-where map across space weather.
  slide2_adcs_architecture          Onboard ADCS flow in IPC mode, with the
      actual equations: sensors -> observer -> RL advisor (quaternion +
      per-axis authority gates) -> feasibility clip -> MRP feedback with gated
      per-axis saturation -> magnetorquers; environment torques close the
      exploitation loop; mode supervisor handles brownout -> B-dot detumble ->
      recovery.

Also writes the standalone winner-map figure + wins summary
(fig_winner_maps.{pdf,png}, winner_summary.json).

Usage:  PYTHONPATH=python python -m arlamx_v2.slides
Outputs -> outputs/analysis/slides/
Level: advanced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SHOW = ROOT / "outputs" / ".old" / "analysis" / "showcase"
G70 = ROOT / "outputs" / ".old" / "analysis" / "gradient70"
OUT = ROOT / "outputs" / ".old" / "analysis" / "slides"

FOCUS = [("heuristic", "Heuristic", "#eb6834"),
         ("mpc", "MPC", "#2a78d6"),
         ("rl_ppo", "RL + IPC", "#1baf7a")]
INK = "#33322e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "0.55",
    "axes.linewidth": 0.7, "savefig.dpi": 300,
})


def _load(slug, scen):
    f = SHOW / f"{slug}__{scen}.npz"
    return np.load(f) if f.exists() else None


# ===========================================================================
# Winner analysis: which policy wins where (lowest decay, brownout-free
# cells only; a policy that browns out in a cell is disqualified there).
# ===========================================================================

def winner_grid(panel, criterion="longevity"):
    """Winner per cell; a policy that browns out in a cell is disqualified.
    criterion: "longevity" (lowest decay) or "downlink" (highest duty)."""
    d = np.load(G70 / f"grid70_{panel}.npz")
    slugs = [s for s, _l, _c in FOCUS]
    socmin = np.stack([d[f"{s}__soc_min"] for s in slugs])
    if criterion == "downlink":
        val = np.stack([d[f"{s}__downlink"] for s in slugs])
        vq = np.where(socmin < 0.10, -np.inf, val)
        win = np.argmax(vq, axis=0).astype(float)
        win[np.all(~np.isfinite(vq), axis=0) | (np.nanmax(vq, axis=0) <= 0)] = np.nan
    else:
        val = np.stack([d[f"{s}__decay"] for s in slugs])
        vq = np.where(socmin < 0.10, np.inf, val)
        win = np.argmin(vq, axis=0).astype(float)
        win[np.all(~np.isfinite(vq), axis=0)] = np.nan
    return d, win


def fig_winner_maps():
    panels = [("sw", "F10.7 (sfu)", "Ap index", (65, 250, 2, 40)),
              ("alt", "inclination (deg)", "altitude (km)", (20, 30, 300, 500))]
    crits = [("longevity", "longevity winner (lowest decay)"),
             ("downlink", "downlink winner (highest duty)")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    cmap = ListedColormap([c for _s, _l, c in FOCUS])
    summary = {}
    for row, (crit, ctitle) in enumerate(crits):
        for col, (panel, xl, yl, box) in enumerate(panels):
            ax = axes[row][col]
            d, win = winner_grid(panel, crit)
            X, Y = d["X"], d["Y"]
            ax.imshow(win, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5,
                      vmax=len(FOCUS) - 0.5, extent=[X[0], X[-1], Y[0], Y[-1]],
                      interpolation="nearest")
            x0, x1, y0, y1 = box
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec="white", lw=1.1, ls="--"))
            ax.set_xlabel(xl, fontsize=10)
            ax.set_ylabel(yl, fontsize=10)
            counts = {l: float(np.nanmean(win == i) * 100)
                      for i, (_s, l, _c) in enumerate(FOCUS)}
            summary[f"{panel}_{crit}"] = counts
            ax.set_title(f"{ctitle} — {panel}\n" + "  ".join(
                f"{l} {v:.0f}%" for l, v in counts.items()), fontsize=9)
    handles = [Line2D([0], [0], marker="s", ls="", mfc=c, mec="none", ms=10,
                      label=l) for _s, l, c in FOCUS]
    handles.append(Line2D([0], [0], ls="--", color="0.4",
                          label="deep-RL training envelope"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle("Lowest orbital decay with no brown-out — who wins where",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"fig_winner_maps.{e}", bbox_inches="tight")
    plt.close(fig)
    (OUT / "winner_summary.json").write_text(json.dumps(summary, indent=2))
    print("  wrote fig_winner_maps + winner_summary.json")
    return summary


# ===========================================================================
# Slide 1 — exploiting drag & SRP as actuation
# ===========================================================================

def slide1():
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], hspace=0.55,
                          wspace=0.22, left=0.07, right=0.985, top=0.86,
                          bottom=0.09)
    fig.suptitle("Exploiting aerodynamic and radiation-pressure perturbations "
                 "for sail control", fontsize=16, y=0.965)
    fig.text(0.07, 0.905, "0.625 kg hex sail · magnetorquer-only ADCS "
             "(τ_max X/Y 18 µN·m, Z 5.7 µN·m) — environmental torques exceed "
             "actuator authority by up to 100× below 400 km",
             fontsize=10, color="0.35")

    # (a) integrated non-gravitational specific work, 2-day nominal mission.
    # Drag loss and harvested lift/SRP are orders of magnitude apart, so they
    # get two separate sub-axes (never a dual-axis chart).
    ga = gs[0, 0].subgridspec(1, 2, wspace=0.08, width_ratios=[1.0, 1.0])
    axl = fig.add_subplot(ga[0, 0])
    axr = fig.add_subplot(ga[0, 1])
    labels, drag_v, lift_v, srp_v = [], [], [], []
    for slug, label, color in FOCUS:
        d = _load(slug, "nominal2d")
        labels.append(label)
        drag_v.append(np.sum(d["dE_drag"]))
        lift_v.append(np.sum(d["dE_lift"]))
        srp_v.append(np.sum(d["dE_srp"]))
    y = np.arange(len(labels))
    axl.barh(y, np.abs(np.asarray(drag_v)) / 1e3, height=0.5, color="#e34948")
    axl.set_yticks(y, labels, fontsize=10)
    axl.invert_xaxis()
    axl.set_xlabel("drag energy lost (kJ/kg)", fontsize=9)
    axr.barh(y + 0.22, lift_v, height=0.36, color="#1baf7a", label="lift")
    axr.barh(y - 0.22, srp_v, height=0.36, color="#eda100", label="SRP")
    axr.axvline(0, color="0.4", lw=0.6)
    axr.set_yticks([])
    axr.set_xlabel("harvested lift / SRP work (J/kg)", fontsize=9)
    axr.legend(fontsize=8, frameon=False, loc="lower right")
    for a_ in (axl, axr):
        a_.grid(True, color="0.88", lw=0.5, axis="x")
        a_.spines[["top", "right", "left"]].set_visible(False)
    axl.set_title("(a) Environmental work over 2 days: cost vs harvest",
                  fontsize=11, loc="left", x=0.05)

    # (b) who wins where, by criterion (space weather, 4,900 episodes each)
    gb = gs[1:, 0].subgridspec(2, 1, hspace=0.45)
    cmap = ListedColormap([c for _s, _l, c in FOCUS])
    for row, (crit, ctitle) in enumerate(
            (("longevity", "longevity (lowest decay, no brown-out)"),
             ("downlink", "downlink (highest duty, no brown-out)"))):
        ax = fig.add_subplot(gb[row, 0])
        d, win = winner_grid("sw", crit)
        X, Y = d["X"], d["Y"]
        ax.imshow(win, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5,
                  vmax=len(FOCUS) - 0.5, extent=[X[0], X[-1], Y[0], Y[-1]],
                  interpolation="nearest")
        ax.add_patch(plt.Rectangle((65, 2), 185, 38, fill=False, ec="white",
                                   lw=1.1, ls="--"))
        counts = "   ".join(
            f"{l} {np.nanmean(win == i) * 100:.0f}%"
            for i, (_s, l, _c) in enumerate(FOCUS))
        ax.set_title(f"(b{row + 1}) winner by {ctitle}:  {counts}", fontsize=10)
        ax.set_ylabel("Ap index", fontsize=9)
        if row == 1:
            ax.set_xlabel("F10.7 (sfu)", fontsize=10)
        ax.tick_params(labelsize=8)
    handles = [Line2D([0], [0], marker="s", ls="", mfc=c, mec="none", ms=9,
                      label=l) for _s, l, c in FOCUS]
    handles.append(Line2D([0], [0], marker="s", ls="", mfc="white", mec="0.6",
                          ms=9, label="white: no policy downlinks"))
    fig.legend(handles=handles, loc="lower left", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.055, -0.005))

    # (c) the IPC mechanism through a severe storm (RL + IPC, 350 km)
    d = _load("rl_ppo", "storm1d")
    t = d["t_h"]
    storm = d["ap"] > 40.0
    sub = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for ax in sub:
        ax.fill_between(t, 0, 1, where=storm, color="#e34948", alpha=0.08,
                        transform=ax.get_xaxis_transform())
        ax.grid(True, color="0.9", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    sub[0].plot(t, np.linalg.norm(d["tau_aero"], axis=1) * 1e6, color="0.5",
                lw=1.0, label="true aero torque")
    sub[0].plot(t, np.linalg.norm(d["tau_dist"], axis=1) * 1e6,
                color="#2a78d6", lw=1.5, ls=(0, (3, 2)),
                label="onboard estimate  τ̂ = IΔω/Δt + ω×Iω − τ_ctrl")
    sub[0].set_yscale("log")
    sub[0].set_ylabel("torque (µN·m)", fontsize=9)
    sub[0].legend(fontsize=8, frameon=False, loc="lower right")
    sub[0].set_title("(c) IPC through a severe storm — RL + IPC at 350 km "
                     "(storm shaded)", fontsize=11)
    for i, (al, c) in enumerate(zip("XYZ", ("#2a78d6", "#eb6834", "#1baf7a"))):
        sub[1].step(t, d["gates"][:, i], where="mid", color=c, lw=1.2,
                    label=f"gate {al}")
    sub[1].set_ylim(0, 1.05)
    sub[1].set_ylabel("axis authority", fontsize=9)
    sub[1].legend(fontsize=8, frameon=False, ncol=3, loc="lower right")
    sub[2].plot(t, d["effort"] * 270.0, color="#4a3aa7", lw=1.2)
    sub[2].set_ylabel("MTQ power (mW)", fontsize=9)
    sub[2].set_xlabel("mission time (h)", fontsize=9)

    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"slide1_perturbation_exploitation.{e}",
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote slide1_perturbation_exploitation")


# ===========================================================================
# Slide 2 — onboard ADCS architecture in IPC mode
# ===========================================================================

def _box(ax, x, y, w, h, title, lines, fc, title_c=INK, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="0.45", lw=1.0))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top",
            fontsize=fs + 1, fontweight="bold", color=title_c)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.078 - 0.036 * i, ln, ha="center",
                va="top", fontsize=fs - 1.2, color=INK)


def _arrow(ax, p0, p1, color="0.35", lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color,
                                 lw=lw, ls=ls, mutation_scale=16,
                                 shrinkA=2, shrinkB=2))


def slide2():
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("Onboard ADCS in intelligent-perturbation-control mode",
                 fontsize=16, y=0.97)

    # row 1: sensing -> estimation -> advisor
    _box(ax, 0.015, 0.62, 0.185, 0.24, "SENSORS",
         ["GPS  r, v", "gyro  ω", "magnetometer  B_B", "EPS  SoC, P_gen"],
         "#eef3fa")
    _box(ax, 0.245, 0.62, 0.24, 0.24, "ONBOARD ESTIMATION",
         ["observation vector (26)", "disturbance observer:",
          r"$\hat{\tau}_d = I\,\Delta\omega/\Delta t + \omega\times I\omega - \tau_{ctrl}$",
          "gate state, sun phase"], "#eaf6f0")
    _box(ax, 0.53, 0.62, 0.215, 0.24, "RL ADVISOR (IPC)",
         ["MLP 4×16 · every 300 s", "outputs:  q_cmd (4)",
          "per-axis gates  g ∈ [0.3, 1]³", "trained: PPO, 50k steps"],
         "#e8f0e2")
    _box(ax, 0.79, 0.62, 0.195, 0.24, "FEASIBILITY",
         ["SLERP clip ≤ 40°", "per inference", "(matches training", "reward band)"],
         "#f7f0e4")

    # row 2: controller -> actuators -> plant
    _box(ax, 0.53, 0.24, 0.215, 0.28, "MRP FEEDBACK",
         [r"$\tau=-k_p\sigma_e-k_d\omega_e+\omega\times I\omega$",
          r"per-axis saturation:", r"$|\tau_i|\leq g_i\,\tau_{max,i}$",
          "k_p 1e-4 · k_d 2e-3", "gates scale authority"], "#eef3fa")
    _box(ax, 0.79, 0.24, 0.195, 0.28, "MAGNETORQUERS",
         ["τ_max X/Y 18 µN·m", "τ_max Z 5.7 µN·m",
          "released axis ⇒", "≈ zero power draw"], "#f3e9f2")
    _box(ax, 0.245, 0.06, 0.24, 0.16, "SPACECRAFT DYNAMICS",
         ["rigid body + orbit (RK4, 2 s)", "GGM03S 4×4 · NRLMSISE-00"],
         "#f2f2ee")
    _box(ax, 0.015, 0.06, 0.185, 0.30, "ENVIRONMENT",
         ["aero torque τ_a (Sentman)", "SRP force/torque",
          "· exceeds actuators", "  up to 100× @ ≤400 km",
          "· sensed via observer", "· exploited via gates"], "#fdeeee",
         title_c="#b33")

    # mode supervisor
    _box(ax, 0.245, 0.30, 0.24, 0.26, "MODE SUPERVISOR",
         ["SoC = 0 → BROWNOUT:", "  B-dot detumble  m = −k·dB/dt",
          "  loads shed (GPS, TX off)", "recover: SoC ≥ 15% ∧ ω ≤ 0.5°/s",
          "→ advisor resumes control"], "#fbf3df", title_c="#8a6d00")

    # arrows: main loop
    _arrow(ax, (0.20, 0.74), (0.245, 0.74))
    _arrow(ax, (0.485, 0.74), (0.53, 0.74))
    _arrow(ax, (0.745, 0.74), (0.79, 0.74))
    _arrow(ax, (0.887, 0.62), (0.887, 0.55))          # feasibility -> ctrl col
    _arrow(ax, (0.887, 0.55), (0.746, 0.44))
    _arrow(ax, (0.745, 0.38), (0.79, 0.38))           # ctrl -> mtq
    _arrow(ax, (0.887, 0.24), (0.44, 0.14))           # mtq -> plant
    _arrow(ax, (0.245, 0.14), (0.20, 0.14))           # env -> plant (reversed)
    _arrow(ax, (0.20, 0.18), (0.245, 0.18))
    _arrow(ax, (0.30, 0.22), (0.30, 0.30))            # plant -> supervisor
    _arrow(ax, (0.365, 0.56), (0.365, 0.62))          # supervisor -> estimation
    # sensing feedback from plant
    _arrow(ax, (0.245, 0.10), (0.10, 0.10), color="0.6")
    _arrow(ax, (0.10, 0.10), (0.10, 0.62), color="0.6")
    # gates highlight: advisor -> controller saturation
    _arrow(ax, (0.638, 0.62), (0.638, 0.52), color="#1baf7a", lw=2.4)
    ax.text(0.648, 0.575, "authority gates g", fontsize=9.5, color="#1baf7a",
            fontweight="bold")
    # exploitation loop annotation
    _arrow(ax, (0.108, 0.38), (0.245, 0.50), color="#e34948", lw=2.2,
           ls=(0, (5, 3)))
    ax.text(0.015, 0.475, "EXPLOITATION LOOP\nreleased axis lets τ_a / SRP\n"
            "rotate the craft for free;\nfloor g ≥ 0.3 keeps damping",
            fontsize=9.5, color="#e34948", va="top",
            bbox=dict(fc="white", ec="#e34948", lw=0.8, alpha=0.9,
                      boxstyle="round,pad=0.3"))

    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"slide2_adcs_architecture.{e}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote slide2_adcs_architecture")


if __name__ == "__main__":
    if (G70 / "grid70_sw.npz").exists():
        fig_winner_maps()
        slide1()
    else:
        print("  [wait] grid70_sw.npz missing — run after the sweep finishes")
    slide2()
    print("[done]")
