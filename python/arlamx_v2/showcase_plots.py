"""Conference figure set from the showcase recordings.

  fig_pointing_accuracy   CDFs: command-vs-achieved attitude error, and antenna
                          boresight error during daylight passes (downlink view)
  fig_advisor_dashboard   Heuristic vs MPC, 2-day nominal mission: SoC + band,
                          downlink windows, drag coefficient, altitude retained
  fig_ipc_timeline        The IPC mechanism during a severe storm: disturbance
                          detection (observer vs truth), per-axis authority
                          gates, magnetorquer power, SoC — annotated onset/clear

All PDF (vector) + 300 dpi PNG, fixed validated policy palette.

Usage:  PYTHONPATH=python python -m arlamx_v2.showcase_plots
Outputs -> outputs/analysis/showcase/fig_*.{pdf,png}
Level: advanced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SHOW = ROOT / "outputs" / ".old" / "analysis" / "showcase"

POLICY_META = [
    ("mpc", "MPC", "#2a78d6"),
    ("heuristic", "Heuristic", "#eb6834"),
    ("min_drag", "Min-drag", "#008300"),
    ("rl_ppo", "RL PPO (IPC)", "#1baf7a"),
    ("rl_sac", "RL SAC (IPC)", "#eda100"),
    ("rl_td3", "RL TD3 (IPC)", "#e87b a4".replace(" ", "")),
]
INK = "#33322e"
GRID = dict(color="0.85", lw=0.5)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 8.5,
    "font.family": "DejaVu Sans", "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": "0.55", "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK,
})


def _load(slug, scen):
    f = SHOW / f"{slug}__{scen}.npz"
    return np.load(f) if f.exists() else None


def _cdf(ax, vals, color, label):
    v = np.sort(np.asarray(vals, float))
    v = v[np.isfinite(v)]
    if not len(v):
        return
    y = np.arange(1, len(v) + 1) / len(v) * 100.0
    ax.plot(v, y, color=color, lw=1.8, label=label)


def fig_pointing_accuracy():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for slug, label, color in POLICY_META:
        d = _load(slug, "nominal2d")
        if d is None:
            continue
        _cdf(ax1, d["cmd_track_deg"], color, label)
        mask = (d["gs_vis"] > 0.5) & (d["eclipse"] > 0.5)
        if np.any(mask):
            err = np.degrees(np.arccos(np.clip(d["gs_cos"][mask], -1.0, 1.0)))
            _cdf(ax2, err, color, label)
    ax1.set_xscale("log")
    ax1.set_xlim(0.02, 200)
    ax1.set_xlabel("command-vs-achieved attitude error (deg)")
    ax1.set_ylabel("steps at or below (%)")
    ax1.set_title("(a) Commanded attitude is reached — tracking fidelity")
    ax2.set_xlim(0, 180)
    ax2.axvspan(0, np.degrees(np.arccos(0.7)), color="#1baf7a", alpha=0.08)
    ax2.text(np.degrees(np.arccos(0.7)) + 2, 6, "LoRa downlink cone (45.6°)",
             fontsize=7.5, color="0.35")
    ax2.set_xlabel("antenna boresight-to-station error during daylight passes (deg)")
    ax2.set_ylabel("pass steps at or below (%)")
    ax2.set_title("(b) Downlink pointing during passes")
    for ax in (ax1, ax2):
        ax.set_ylim(0, 100)
        ax.grid(True, **GRID)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Pointing accuracy — 2-day nominal mission (400 km)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(SHOW / f"fig_pointing_accuracy.{e}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_pointing_accuracy")


def fig_advisor_dashboard():
    pols = [("heuristic", "Heuristic", "#eb6834"), ("mpc", "MPC", "#2a78d6")]
    data = {s: _load(s, "nominal2d") for s, _l, _c in pols}
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 8.6), sharex=True)

    ax = axes[0]
    ax.axhspan(40, 60, color="#1baf7a", alpha=0.10)
    ax.text(0.4, 61.5, "target band 40–60%", fontsize=7.5, color="0.35")
    for s, l, c in pols:
        d = data[s]
        ax.plot(d["t_h"], d["soc"] * 100, color=c, lw=1.4, label=l)
    ax.set_ylabel("supercap SoC (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Heuristic vs MPC — 2-day nominal mission at 400 km", fontsize=11)

    ax = axes[1]
    for k, (s, l, c) in enumerate(pols):
        d = data[s]
        dl = d["downlink"] > 0.5
        vis = (d["gs_vis"] > 0.5) & (d["eclipse"] > 0.5)
        y = 1.0 - 0.45 * k
        ax.fill_between(d["t_h"], y - 0.16, y + 0.16, where=vis,
                        color=c, alpha=0.18, step="mid")
        ax.fill_between(d["t_h"], y - 0.16, y + 0.16, where=dl,
                        color=c, alpha=0.95, step="mid")
        ax.text(d["t_h"][2], y + 0.20, l, fontsize=7.5, color=c)
    ax.set_ylim(0.2, 1.45)
    ax.set_yticks([])
    ax.set_ylabel("downlink\nwindows")
    ax.text(0.995, 0.04, "light = station visible in daylight · solid = downlinking",
            fontsize=7, color="0.4", ha="right", transform=ax.transAxes)

    ax = axes[2]
    for s, l, c in pols:
        d = data[s]
        ax.plot(d["t_h"], d["cd"], color=c, lw=1.0, alpha=0.9)
    ax.set_ylabel("effective $C_d$")

    ax = axes[3]
    for s, l, c in pols:
        d = data[s]
        ax.plot(d["t_h"], d["sma_km"] - d["sma_km"][0], color=c, lw=1.4)
    ax.set_ylabel("SMA change (km)")
    ax.set_xlabel("mission time (h)")

    for ax in axes:
        ax.grid(True, **GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(SHOW / f"fig_advisor_dashboard.{e}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_advisor_dashboard")


def fig_ipc_timeline(slug="rl_ppo"):
    d = _load(slug, "storm1d")
    label = dict((s, l) for s, l, _c in POLICY_META)[slug]
    t = d["t_h"]
    cap = np.array([1.8e-5, 1.8e-5, 5.7e-6])
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 8.8), sharex=True)

    # storm shading from the recorded ap trace
    storm = d["ap"] > 40.0
    for ax in axes:
        ax.fill_between(t, 0, 1, where=storm, color="#e34948", alpha=0.06,
                        transform=ax.get_xaxis_transform())
    t_on = t[np.argmax(storm)] if np.any(storm) else None

    ax = axes[0]
    ax.plot(t, np.linalg.norm(d["tau_aero"], axis=1) * 1e6, color="0.45", lw=1.0,
            label="true aero torque (plant)")
    ax.plot(t, np.linalg.norm(d["tau_dist"], axis=1) * 1e6, color="#2a78d6",
            lw=1.6, ls=(0, (4, 2)), label="onboard disturbance estimate")
    ax.axhline(np.linalg.norm(cap) * 1e6, color="0.6", lw=0.8, ls=":")
    ax.text(t[-1], np.linalg.norm(cap) * 1e6 * 1.1, "magnetorquer authority",
            fontsize=7, color="0.45", ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("torque (µN·m)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(f"Intelligent perturbation control during a severe storm — {label}, 350 km",
                 fontsize=11)

    ax = axes[1]
    for i, (axis_l, c) in enumerate(zip("XYZ", ("#2a78d6", "#eb6834", "#1baf7a"))):
        ax.step(t, d["gates"][:, i], where="mid", color=c, lw=1.3,
                label=f"gate {axis_l}")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("axis authority gate")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right")

    ax = axes[2]
    ax.plot(t, d["effort"] * 270.0, color="#4a3aa7", lw=1.2)
    ax.set_ylabel("magnetorquer power (mW)")

    ax = axes[3]
    ax.axhspan(40, 60, color="#1baf7a", alpha=0.10)
    ax.plot(t, d["soc"] * 100, color="#eda100", lw=1.5)
    ax.set_ylabel("supercap SoC (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("mission time (h)")

    if len(t) and t[-1] < 22.0:
        axes[3].annotate("decayed below the 250 km floor",
                         xy=(t[-1], 20), xytext=(t[-1] - 5.5, 8),
                         fontsize=8, color="#e34948",
                         arrowprops=dict(arrowstyle="->", color="#e34948", lw=0.9))
    if t_on is not None:
        axes[0].annotate("storm onset\n(F10.7 +130, Ap +76)",
                         xy=(t_on, np.linalg.norm(d["tau_dist"][np.argmax(storm)])
                             * 1e6),
                         xytext=(t_on + 1.2, 8e-1), fontsize=8, color="#e34948",
                         arrowprops=dict(arrowstyle="->", color="#e34948", lw=0.9))
    for ax in axes:
        ax.grid(True, **GRID)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(SHOW / f"fig_ipc_timeline.{e}", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_ipc_timeline")


def summary_json():
    out = {}
    for slug, label, _c in POLICY_META:
        row = {}
        for scen in ("nominal2d", "storm1d"):
            d = _load(slug, scen)
            if d is None:
                continue
            n = len(d["t_h"])
            days = float(d["t_h"][-1]) / 24.0
            mask = (d["gs_vis"] > 0.5) & (d["eclipse"] > 0.5)
            err = (np.degrees(np.arccos(np.clip(d["gs_cos"][mask], -1, 1)))
                   if np.any(mask) else np.array([np.nan]))
            row[scen] = {
                "survived_steps": int(n),
                "decay_km_d": float((d["sma_km"][0] - d["sma_km"][-1]) / max(days, 1e-9)),
                "band_pct": float(np.mean((d["soc"] >= 0.4) & (d["soc"] <= 0.6)) * 100),
                "downlink_min_d": float(np.sum(d["downlink"]) * 5.0 / max(days, 1e-9)),
                "track_err_median_deg": float(np.median(d["cmd_track_deg"])),
                "gs_err_median_deg": float(np.nanmedian(err)),
                "mtq_energy_J": float(np.sum(d["effort"]) * 0.270 * 300.0),
            }
        out[slug] = row
    (SHOW / "showcase_summary.json").write_text(json.dumps(out, indent=2))
    print("  wrote showcase_summary.json")


if __name__ == "__main__":
    fig_pointing_accuracy()
    fig_advisor_dashboard()
    fig_ipc_timeline()
    summary_json()
    print("[done]")
