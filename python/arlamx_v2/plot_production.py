"""Presentation figures for the SC_v7 bake-off (RL vs MPC vs heuristic).

Six slide-ready figures written to outputs/presentation/plots/ as PNG + PDF.
Reads only what bench_production wrote; runs no simulation of its own.

    PYTHONPATH=python python -m arlamx_v2.plot_production
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from arlamx_v2.env import MTQ_TAU_NM
from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

OUT = OUTPUTS / "presentation"
DATA = OUT / "data"
PLOTS = OUT / "plots"

# Validated categorical palette (light surface). Assigned by entity, in fixed
# order — never by rank, never cycled. Verified with the dataviz validator:
# worst adjacent CVD dE 9.2, normal-vision dE 27.6. Aqua sits below 3:1 on the
# light surface, so every series carrying it is also directly labelled.
C = {
    "MPC": "#2a78d6",        # slot 1 blue
    "Heuristic": "#eb6834",  # slot 2 orange
    "PPO": "#1baf7a",        # slot 3 aqua
    "SAC": "#4a3aa7",        # slot 7 violet
    "TD3": "#e34948",        # slot 8 red
}
MARK = {"MPC": "o", "Heuristic": "s", "PPO": "^", "SAC": "D", "TD3": "v"}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e4e3de"
SURF = "#fcfcfb"

ALGOS = ("PPO", "SAC", "TD3")
BUDGETS = (50, 250)
SETTLE_H = 1.0   # drop the post-deployment detumble transient from observer stats


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "figure.dpi": 110,
    })


def _header(fig, title, subtitle, top=0.84):
    """Title block with guaranteed clearance above the axes."""
    fig.subplots_adjust(top=top)
    fig.text(0.006, 0.985, title, fontsize=21, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.006, 0.925, subtitle, fontsize=12.5, color=INK2, ha="left", va="top")


def _save(fig, stem, note=None):
    if note:
        fig.text(0.006, 0.004, note, fontsize=9, color=MUTED, ha="left", va="bottom")
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def load():
    return (json.loads((DATA / "bakeoff.json").read_text()),
            json.loads((DATA / "traces.json").read_text()),
            json.loads((DATA / "inference_cost.json").read_text()),
            json.loads((DATA / "train_metrics.json").read_text()))


def cells(bake):
    out = {}
    for k, v in bake.items():
        if k in ("MPC", "Heuristic"):
            out[(k, None)] = [v]
            continue
        algo, bud, _seed = k.split()
        out.setdefault((algo, int(bud[:-1])), []).append(v)
    return out


def med(rows, key):
    return float(np.median([r["agg"][key] for r in rows]))


def med_gap(rows):
    return float(np.median([r["gap_index"] for r in rows]))


def best_cell(cl, algo):
    opts = [(med_gap(cl[(algo, b)]), b) for b in BUDGETS if (algo, b) in cl]
    return min(opts)[1] if opts else BUDGETS[0]


def _rl_label(traces, bake):
    cl = cells(bake)
    order = sorted(ALGOS, key=lambda a: med_gap(cl[(a, best_cell(cl, a))]))
    for a in order:
        lab = f"{a} {best_cell(cl, a)}k"
        if lab in traces:
            return lab
    return [k for k in traces if k not in ("MPC", "Heuristic")][0]


def _torque_nNm(tr, key):
    """Per-axis normalised torque -> magnitude in nN.m."""
    return np.linalg.norm(np.asarray(tr[key], float) * np.asarray(MTQ_TAU_NM) * 1e9,
                          axis=1)


# ---------------------------------------------------------------------------
# 1 — mission scorecard
# ---------------------------------------------------------------------------

def fig1_scorecard(bake):
    cl = cells(bake)
    names = ["MPC", "Heuristic"]
    cols = [C["MPC"], C["Heuristic"]]
    rows = [cl[("MPC", None)], cl[("Heuristic", None)]]
    for a in ALGOS:
        b = best_cell(cl, a)
        names.append(f"{a}\n{b}k")
        cols.append(C[a])
        rows.append(cl[(a, b)])

    panels = [
        ("decay_nominal", "Orbit decay, nominal\n(km/day — lower is better)", "%.1f", True),
        ("band_pct", "Battery in the 40–60 % band\n(% of time — higher is better)", "%.0f", False),
        ("downlink_min_d", "Downlink contact\n(min/day — higher is better)", "%.1f", False),
        ("act_energy_J", "Magnetorquer energy\n(J per probe set — lower is better)", "%.0f", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.8))
    x = np.arange(len(names))
    for ax, (key, title, fmt, lower_better) in zip(axes, panels):
        vals = [med(r, key) for r in rows]
        bars = ax.bar(x, vals, color=cols, width=0.66, linewidth=2,
                      edgecolor=SURF, zorder=3)
        best = (min if lower_better else max)(vals)
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(vals) * 0.035, fmt % v, ha="center", va="bottom",
                    fontsize=12.5, color=INK,
                    fontweight="bold" if v == best else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11.5, color=INK2)
        ax.set_title(title, color=INK, pad=12, fontsize=13.5)
        ax.set_ylim(0, max(vals) * 1.24)
        ax.grid(axis="x", visible=False)
        _ = bars
    _header(fig, "Mission outcome — the sampling MPC still sets the bar",
            "Median of 3 seeds per learner; each bar is 3 probe scenarios × 2 seeds × 20 orbits. "
            "Each algorithm is shown at its better training budget.", top=0.80)
    _save(fig, "fig1_mission_scorecard",
          "Source: outputs/presentation/data/bakeoff.json — the probe suite is identical for every policy.")


# ---------------------------------------------------------------------------
# 2 — decision cost vs quality
# ---------------------------------------------------------------------------

def fig2_cost_vs_quality(bake, infer):
    cl = cells(bake)
    pts = [(n, n, infer[n]["ms_per_decision"], med_gap(cl[(n, None)]))
           for n in ("MPC", "Heuristic")]
    for a in ALGOS:
        b = best_cell(cl, a)
        lab = f"{a} {b}k"
        pts.append((a, lab, infer[lab]["ms_per_decision"], med_gap(cl[(a, b)])))

    # TD3 and SAC sit close enough that both labels cannot go above their
    # markers; SAC is pushed to the right instead.
    OFFSET = {"MPC": (0, 24, "center"), "Heuristic": (0, 24, "center"),
              "PPO": (0, 24, "center"), "SAC": (26, -6, "left"),
              "TD3": (0, 24, "center")}
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    for key, lab, ms, gap in pts:
        ax.scatter(ms, gap, s=340, color=C[key], marker=MARK[key],
                   edgecolor=SURF, linewidth=2, zorder=4)
        dx, dy, ha = OFFSET[key]
        ax.annotate(f"{lab} · {ms:.2f} ms", (ms, gap), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=12.5, color=INK,
                    fontweight="bold")
    mpc_ms = infer["MPC"]["ms_per_decision"]
    rl_ms = float(np.median([infer[f"{a} {best_cell(cl,a)}k"]["ms_per_decision"]
                             for a in ALGOS]))
    ax.set_xscale("log")
    ax.set_xlabel("Compute per control decision  (ms, log scale)")
    ax.set_ylabel("gap index vs MPC   (0 = MPC, lower is better)")
    ax.axhline(0, color=C["MPC"], linewidth=2, linestyle="--", zorder=2)
    ymax = max(g for *_, g in pts) * 1.28
    ax.set_ylim(-0.6, ymax)
    ax.text(ax.get_xlim()[0] * 1.06, 0.05, "MPC reference", fontsize=12,
            color=C["MPC"], va="bottom", fontweight="bold")
    ax.annotate("", xy=(rl_ms, -0.36), xytext=(mpc_ms, -0.36),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.8))
    ax.text(float(np.sqrt(rl_ms * mpc_ms)), -0.31,
            f"{mpc_ms / rl_ms:,.0f}× cheaper per decision", ha="center",
            fontsize=13.5, color=INK, fontweight="bold")
    _header(fig, "What a decision costs, and what it buys",
            "The MPC buys its lead by rolling the dynamics forward many times per decision. The network reaches its answer at fixed\n"
            "cost and needs no onboard propagator — the difference between a controller that fits a CubeSat duty cycle and one that does not.",
            top=0.82)
    _save(fig, "fig2_cost_vs_quality",
          "Wall-clock for one decision on this workstation. For MPC and the heuristic this includes the onboard state assembly they need "
          "(density lookup ~0.02 ms, station lookup); for the network it is the forward pass.")


# ---------------------------------------------------------------------------
# 3 — learner x budget, with seed spread
# ---------------------------------------------------------------------------

def fig3_learner_budget(bake):
    cl = cells(bake)
    fig, ax = plt.subplots(figsize=(13.5, 7.2))

    allg = [g for a in ALGOS for b in BUDGETS if (a, b) in cl
            for g in [r["gap_index"] for r in cl[(a, b)]]]
    # One SAC seed diverges to ~10; clipping keeps the other 17 runs readable and
    # the outlier is called out explicitly rather than dropped.
    cap = float(np.percentile(allg, 90)) * 1.35
    labels, xs, pos = [], [], 0
    for a in ALGOS:
        for b in BUDGETS:
            rows = cl.get((a, b))
            if not rows:
                continue
            gaps = np.array([r["gap_index"] for r in rows])
            inside, outside = gaps[gaps <= cap], gaps[gaps > cap]
            ax.scatter([pos] * len(inside), inside, s=155, color=C[a], marker=MARK[a],
                       edgecolor=SURF, linewidth=1.8, zorder=4)
            for g in outside:
                ax.scatter([pos], [cap], s=155, color=C[a], marker=MARK[a],
                           edgecolor=SURF, linewidth=1.8, zorder=4, alpha=0.55)
                ax.annotate(f"{g:.1f} ↑", (pos, cap), textcoords="offset points",
                            xytext=(12, 2), fontsize=11, color=C[a],
                            fontweight="bold")
            m = float(np.median(gaps))
            ax.plot([pos - 0.27, pos + 0.27], [m, m], color=C[a], linewidth=3.6,
                    zorder=5, solid_capstyle="round")
            ax.annotate(f"{m:.2f}", (pos, m), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=12, color=INK,
                        fontweight="bold", zorder=6)
            labels.append(f"{a}\n{b}k")
            xs.append(pos)
            pos += 1
        pos += 0.45

    hg = med_gap(cl[("Heuristic", None)])
    ax.axhline(0, color=C["MPC"], linewidth=2.4, linestyle="--", zorder=2)
    ax.axhline(hg, color=C["Heuristic"], linewidth=2.4, linestyle=":", zorder=2)
    ax.set_xlim(-0.7, pos - 0.2)
    ax.text(pos - 0.3, 0.03, "MPC", color=C["MPC"], fontsize=12.5,
            fontweight="bold", ha="right", va="bottom")
    ax.text(pos - 0.3, hg + 0.03, "Heuristic advisor", color=C["Heuristic"],
            fontsize=12.5, fontweight="bold", ha="right", va="bottom")
    ax.set_ylim(-0.25, cap * 1.10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12, color=INK2)
    ax.set_ylabel("gap index vs MPC   (lower is better)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=C[a], marker=MARK[a], linestyle="none",
                      markersize=11, markeredgecolor=SURF, label=a) for a in ALGOS]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=3)
    _header(fig, "More training steps did not buy a better policy",
            "Each marker is one seed (42 / 43 / 44); the bar is the median. Going from 50k to 250k steps made every learner worse on this\n"
            "reward — the extra budget is spent driving drag harder than the power system can support.",
            top=0.82)
    _save(fig, "fig3_learner_and_budget",
          "3 seeds per cell. Markers above the axis cap are drawn at the cap with their true value labelled. "
          "The published campaign leaderboard reported a best-of-53 trial and is not comparable with a single seed.")


# ---------------------------------------------------------------------------
# 4 — disturbance observer
# ---------------------------------------------------------------------------

def fig4_observer(traces, bake):
    rl = _rl_label(traces, bake)
    tr = traces[rl]["nominal"]
    t = np.asarray(tr["t_h"], float)
    keep = t >= SETTLE_H
    true_m, est_m = _torque_nNm(tr, "tau_aero")[keep], _torque_nNm(tr, "tau_dist")[keep]
    tt = t[keep]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.6),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.plot(tt, true_m, color=INK2, linewidth=2.6, label="True aerodynamic torque (plant)", zorder=3)
    ax.plot(tt, est_m, color=C[rl.split()[0]], linewidth=2.0, linestyle="--",
            label="Onboard estimate (gyro + own commands)", zorder=4)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Disturbance torque magnitude  (nN·m)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"{rl} — the estimate follows the real torque", color=INK,
                 fontsize=15, pad=10, loc="left")

    ax2 = axes[1]
    ax2.scatter(true_m, est_m, s=32, color=C[rl.split()[0]], alpha=0.5,
                edgecolor="none", zorder=3)
    hi = float(max(true_m.max(), est_m.max())) * 1.05
    ax2.plot([0, hi], [0, hi], color=INK2, linewidth=1.8, linestyle=":", zorder=4)
    ax2.set_xlim(0, hi)
    ax2.set_ylim(0, hi)
    ax2.set_xlabel("True torque (nN·m)")
    ax2.set_ylabel("Estimated torque (nN·m)")
    ax2.set_title("Estimate vs truth", color=INK, fontsize=15, pad=10, loc="left")

    # Accuracy depends on how hard the vehicle is slewing — report all three.
    lines = []
    for lab in ("Heuristic", "MPC", rl):
        s = traces[lab]["nominal"]
        k = np.asarray(s["t_h"], float) >= SETTLE_H
        a, e = _torque_nNm(s, "tau_aero")[k], _torque_nNm(s, "tau_dist")[k]
        lines.append((lab, float(np.corrcoef(a, e)[0, 1])))
    txt = "\n".join(f"{lab:>12s}   r = {r:.2f}" for lab, r in lines)
    ax2.text(0.045, 0.97, txt, transform=ax2.transAxes, fontsize=12.5, va="top",
             family="monospace", color=INK,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                       edgecolor=GRID, linewidth=1.2))
    ax2.text(0.97, 0.05, "the quieter the vehicle, the better the estimate —\n"
                         "hard slewing corrupts the finite-difference term",
             transform=ax2.transAxes, fontsize=11.5, color=MUTED, va="bottom",
             ha="right")

    _header(fig, "Sensing the environment with no extra hardware",
            "τ_dist = I·ω̇ + ω×Iω − τ_cmd — the rate gyro and the torque the controller already commanded. No accelerometer, no extra sensor.\n"
            f"The first {SETTLE_H:.0f} h of post-deployment detumble is excluded: the finite-difference term is meaningless while the craft is still spinning down.",
            top=0.80)
    _save(fig, "fig4_disturbance_observer",
          f"Policy: {rl}, nominal probe, 20 orbits. The true torque is the plant's aerodynamic torque and is never visible to the controller.")


# ---------------------------------------------------------------------------
# 5 — authority gating and actuator economy
# ---------------------------------------------------------------------------

def fig5_gates_and_economy(bake, traces):
    cl = cells(bake)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    order = [("MPC", "MPC"), ("Heuristic", "Heuristic")] + \
            [(f"{a} {best_cell(cl, a)}k", a) for a in ALGOS]
    gate_note = []
    for lab, ckey in order:
        if lab not in traces:
            continue
        tr = traces[lab]["nominal"]
        t = np.asarray(tr["t_h"], float)
        keep = t >= SETTLE_H          # drop the post-deployment detumble spike
        t, tau = t[keep], _torque_nNm(tr, "tau_ctrl")[keep]
        # ~1-orbit boxcar: the per-step signal is spiky and the comparison is
        # about sustained duty, not individual slews.
        w = max(1, int(round(1.55 / max(t[1] - t[0], 1e-9))))
        sm = np.convolve(tau, np.ones(w) / w, mode="valid")
        ts = t[w - 1:]
        ax.plot(ts, sm, color=C[ckey], linewidth=2.4, zorder=4)
        ax.text(ts[-1], sm[-1], f"  {ckey}", color=C[ckey], fontsize=12.5,
                fontweight="bold", va="center", ha="left")
        if ckey in ALGOS:
            gate_note.append(f"{ckey} ceiling {np.mean(tr['gate']):.2f}")
    ax.set_ylim(bottom=0)
    xr = ax.get_xlim()
    ax.set_xlim(0, xr[1] * 1.12)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Commanded magnetorquer torque  (nN·m, 1-orbit mean)")
    ax.set_title("What each policy actually asks its magnetorquers to do",
                 color=INK, fontsize=15, pad=10, loc="left")
    ax.text(0.985, 0.965,
            "authority ceilings set by the learners:\n" + "\n".join(gate_note),
            transform=ax.transAxes, fontsize=11.5, color=MUTED, va="top",
            ha="right", linespacing=1.45)

    ax2 = axes[1]
    names = ["MPC", "Heuristic"] + [f"{a} {best_cell(cl,a)}k" for a in ALGOS]
    keys = ["MPC", "Heuristic"] + list(ALGOS)
    rows = [cl[("MPC", None)], cl[("Heuristic", None)]] + \
           [cl[(a, best_cell(cl, a))] for a in ALGOS]
    vals = [med(r, "act_energy_J") for r in rows]
    y = np.arange(len(names))[::-1]
    ax2.barh(y, vals, color=[C[k] for k in keys], height=0.62, zorder=3,
             edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, vals):
        ax2.text(v + max(vals) * 0.02, yi, f"{v:.0f} J", va="center", ha="left",
                 fontsize=12.5, color=INK, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=12, color=INK2)
    ax2.set_xlim(0, max(vals) * 1.20)
    ax2.set_xlabel("Magnetorquer energy over the probe set  (J)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("What that costs in actuator energy", color=INK, fontsize=15,
                  pad=10, loc="left")

    _header(fig, "Six times the actuator duty, for no better mission result",
            "Peak magnetorquer torque is 18 nN·m on X/Y — the same order as the aerodynamic torque already acting — so how hard the policy pushes is a real choice.\n"
            "TD3 holds attitude on a third of the MPC's torque and a sixth of PPO's, and pays for it in orbit-energy performance (fig 1). The learners' own\n"
            "authority ceilings barely differ, so it is the commanded torque, not the gate action, that separates them.",
            top=0.78)
    _save(fig, "fig5_environment_and_economy",
          "Left: nominal probe, one representative seed per learner, smoothed over one orbit. Right: median of 3 seeds. "
          "Advisors run with the gates open, so their ceiling is 1.0 by construction and not a learned quantity.")


# ---------------------------------------------------------------------------
# 6 — storm response
# ---------------------------------------------------------------------------

def fig6_storm(traces, bake):
    rl = _rl_label(traces, bake)
    show = [("MPC", "MPC"), ("Heuristic", "Heuristic"), (rl, rl.split()[0])]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4))

    for ax, key, ylab, title in (
        (axes[0], "alt_km", "Altitude (km)", "Altitude through the storm"),
        (axes[1], "soc", "Battery state of charge", "Battery through the storm"),
    ):
        for lbl, ckey in show:
            tr = traces[lbl]["spike"]
            t = np.asarray(tr["t_h"], float)
            v = np.asarray(tr[key], float)
            ax.plot(t, v, color=C[ckey], linewidth=2.5, zorder=4, label=lbl)
            ax.text(t[-1], v[-1], f"  {lbl}", color=C[ckey], fontsize=12,
                    fontweight="bold", va="center", ha="left")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(ylab)
        ax.set_title(title, color=INK, fontsize=15, pad=10, loc="left")
        ax.set_xlim(0, float(np.asarray(traces["MPC"]["spike"]["t_h"], float)[-1]) * 1.18)

    axes[1].axhspan(0.4, 0.6, color=C["MPC"], alpha=0.08, zorder=1)
    axes[1].set_ylim(0, 1.18)
    axes[1].text(0.985, 0.50, "target band", transform=axes[1].get_yaxis_transform(),
                 fontsize=11.5, color=MUTED, va="center", ha="right")

    tmax = float(np.asarray(traces["MPC"]["spike"]["t_h"], float)[-1])
    for ax in axes:
        for frac, note in ((0.25, "storm onset"), (0.60, "storm clears")):
            xv = tmax * frac
            ax.axvline(xv, color=MUTED, linewidth=1.5, linestyle=":", zorder=2)
            ax.annotate(note, (xv, 1.0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(4, -12), fontsize=11,
                        color=MUTED, ha="left", va="top")

    _header(fig, "Behaviour when the atmosphere changes underneath the policy",
            "Storm probe: F10.7 stepped +100 and Ap +36 a quarter of the way in, released at 60 %. Identical disturbance for all three.\n"
            "The MPC gives up the least altitude while holding the battery in band; the heuristic protects the battery and pays in altitude.",
            top=0.80)
    _save(fig, "fig6_storm_response",
          "One deterministic episode per policy (seed 1001); the scorecard averages the full probe set.")


def main():
    _style()
    bake, traces, infer, _train = load()
    print("[plots] writing to", PLOTS)
    fig1_scorecard(bake)
    fig2_cost_vs_quality(bake, infer)
    fig3_learner_budget(bake)
    fig4_observer(traces, bake)
    fig5_gates_and_economy(bake, traces)
    fig6_storm(traces, bake)
    print("[plots] done")


if __name__ == "__main__":
    main()
