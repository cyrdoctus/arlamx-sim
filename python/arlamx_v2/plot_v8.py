"""Conference v2 figures for SC_v8 / v8Duo / v9.

Three sets, matching the presentation plan:
    --set 1   v8a vs v8b (+ MPC, heuristic)          figures f1-f6
    --set 2   set 1 + v8Duo                          + f7
    --set 3   set 2 + v9a/v9b                        + f8, f9

Reads outputs/v8/{ledger.csv, data/*}; writes outputs/presentation/plots_v8/
as PNG + PDF. Figures whose inputs are missing are skipped with a notice, so
the sets can be rendered incrementally as the campaign progresses.

    PYTHONPATH=python python -m arlamx_v2.plot_v8 --set 3
Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

V8 = OUTPUTS / "v8"
DATA = V8 / "data"
PLOTS = OUTPUTS / "presentation" / "plots_v8"

# Validated categorical palette (dataviz validator: worst adjacent CVD dE 9.2).
# Colour follows the ENTITY, fixed order, never cycled.
C = {"MPC": "#2a78d6", "Heuristic": "#eb6834", "v8a": "#1baf7a",
     "v8b": "#4a3aa7", "Duo": "#e34948", "v9a": "#eda100", "v9b": "#e87ba4",
     "v9c": "#008300", "v9d": "#7c3aed", "v9Duo": "#b45309", "v10": "#0f766e",
     "v11": "#9d174d"}
MARK = {"MPC": "o", "Heuristic": "s", "v8a": "^", "v8b": "D", "Duo": "v",
        "v9a": "P", "v9b": "X", "v9c": "<", "v9d": ">", "v9Duo": "*",
        "v10": "h", "v11": "8"}
ALGO_LS = {"ppo": "-", "sac": "--", "td3": ":"}

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de", "#fcfcfb"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 13, "axes.titlesize": 15,
        "legend.fontsize": 12, "figure.dpi": 110,
    })


def _header(fig, title, subtitle, top=0.82):
    fig.subplots_adjust(top=top)
    fig.text(0.006, 0.985, title, fontsize=20, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.006, 0.928, subtitle, fontsize=12.5, color=INK2, ha="left", va="top")


def _save(fig, stem, note=None):
    if note:
        fig.text(0.006, 0.004, note, fontsize=9, color=MUTED, ha="left", va="bottom")
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def _need(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"  [skip] missing: {missing}")
        return False
    return True


def ledger():
    with open(V8 / "ledger.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("decay_nominal", "decay_spike", "band_pct", "brownouts",
                  "downlink_min_d", "act_energy_J", "mtq_energy_J",
                  "deleg_benefit", "deleg_P", "deleg_S", "kf_r", "raw_r",
                  "gap_index", "timesteps"):
            r[k] = float(r[k])
    return rows


def best_row(rows, variant, budget=None):
    pool = [r for r in rows if r["variant"] == variant
            and (budget is None or r["timesteps"] == budget)]
    return min(pool, key=lambda r: r["gap_index"]) if pool else None


def ref():
    return json.loads((DATA / "ref.json").read_text())


def trace_key(traces, variant):
    for k in traces:
        if k.startswith(variant + " "):
            return k
    return None


# ---------------------------------------------------------------------------
# f1 — mission scorecard
# ---------------------------------------------------------------------------

def f1_scorecard(rows):
    if not _need(DATA / "ref.json"):
        return
    R = ref()
    entries = [("MPC", R["MPC"]["agg"]), ("Heuristic", R["Heuristic"]["agg"])]
    for var in ("v8a", "v8b"):
        b = best_row(rows, var)
        if b:
            entries.append((var, b))
    panels = [
        ("decay_nominal", "Orbit decay, nominal\n(km/day — lower better)", "%.1f", True),
        ("band_pct", "Battery in 40-60 % band\n(% of time — higher better)", "%.0f", False),
        ("downlink_min_d", "Downlink contact\n(min/day — higher better)", "%.1f", False),
        ("mtq_energy_J", "Coil energy, physical I²R\n(J per probe set — lower better)", "%.2f", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.6))
    x = np.arange(len(entries))
    for ax, (key, title, fmt, lower) in zip(axes, panels):
        vals = [float(e[1][key]) for e in entries]
        ax.bar(x, vals, color=[C[e[0]] for e in entries], width=0.66,
               edgecolor=SURF, linewidth=2, zorder=3)
        best = (min if lower else max)(vals)
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(vals) * 0.035, fmt % v, ha="center", va="bottom",
                    fontsize=12.5, color=INK,
                    fontweight="bold" if v == best else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in entries], fontsize=11.5)
        ax.set_title(title, pad=12, fontsize=13.0, color=INK)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="x", visible=False)
    _header(fig, "SC_v8 on the 300 km-capable plant — one arm delegates, one cannot",
            "All policies on the identical plant: 1 cm cp-cm offset, custom 38 µN·m coils, quadratic coil power, sensor-fed torque filter.\n"
            "v8a pins full magnetorquer authority (control arm); v8b may delegate part of each turn to the environmental torque.",
            top=0.80)
    _save(fig, "v8_f1_scorecard",
          "Best run per arm over the size sweep. Source: outputs/v8/ledger.csv, data/ref.json — probe suite identical for every policy.")


# ---------------------------------------------------------------------------
# f2 — the size sweep
# ---------------------------------------------------------------------------

def f2_size_sweep(rows):
    widths = {"4x14": 14, "4x16": 16, "4x18": 18, "4x20": 20, "4x24": 24, "4x32": 32}
    budgets = sorted({int(r["timesteps"]) for r in rows if r["variant"] in ("v8a", "v8b")})
    if not budgets:
        print("  [skip] f2: no sweep rows yet")
        return
    CAP = 3.0     # divergent runs are drawn AT the cap with their true value —
                  # the story lives below 1.5 and the spikes would squash it.
    fig, axes = plt.subplots(1, len(budgets), figsize=(9.0 * len(budgets), 6.2),
                             sharey=True, squeeze=False)
    for ax, bud in zip(axes[0], budgets):
        for var in ("v8a", "v8b"):
            for algo in ("ppo", "sac", "td3"):
                pts = sorted(((widths[r["arch"]], r["gap_index"]) for r in rows
                              if r["variant"] == var and r["algo"] == algo
                              and int(r["timesteps"]) == bud and r["arch"] in widths),
                             key=lambda t: t[0])
                if not pts:
                    continue
                w, g = zip(*pts)
                gc = np.minimum(g, CAP)
                ax.plot(w, gc, color=C[var], linestyle=ALGO_LS[algo], linewidth=2.2,
                        marker=MARK[var], markersize=7, markeredgecolor=SURF, zorder=3)
                for wi, gi in zip(w, g):
                    if gi > CAP:
                        ax.annotate(f"{gi:.1f}↑", (wi, CAP),
                                    textcoords="offset points", xytext=(0, 6),
                                    ha="center", fontsize=10, color=C[var])
        ax.set_ylim(0, CAP * 1.12)
        ax.set_title(f"{bud // 1000}k steps", color=INK, fontsize=15, loc="left")
        ax.set_xlabel("network width (4 layers)")
        ax.set_xticks(list(widths.values()))
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("gap index vs MPC (lower is better)")
    handles = ([Line2D([], [], color=C[v], marker=MARK[v], linestyle="none",
                       markersize=9, markeredgecolor=SURF, label=v) for v in ("v8a", "v8b")]
               + [Line2D([], [], color=INK2, linestyle=ALGO_LS[a], label=a.upper())
                  for a in ("ppo", "sac", "td3")])
    axes[0][0].legend(handles=handles, frameon=False, ncol=5, loc="upper left")
    _header(fig, "Does capacity buy control? The 4-layer width sweep",
            "Six widths x three learners x both arms at each budget, one seed each. Colour = arm, line style = learner.",
            top=0.80)
    _save(fig, "v8_f2_size_sweep", "Source: outputs/v8/ledger.csv. Diverged runs are drawn at the axis cap with their true gap labelled.")


# ---------------------------------------------------------------------------
# f3 — delegation mechanics
# ---------------------------------------------------------------------------

def f3_delegation(rows):
    if not _need(DATA / "traces.json"):
        return
    traces = json.loads((DATA / "traces.json").read_text())
    key = trace_key(traces, "v8b")
    if not key:
        print("  [skip] f3: no v8b trace")
        return
    tr = traces[key]["nominal"]
    t = np.asarray(tr["t_h"], float)
    split = np.asarray(tr["split"], float)
    B = np.asarray(tr["deleg_B"], float)
    P = np.asarray(tr["deleg_P"], float)

    fig, axes = plt.subplots(3, 1, figsize=(15, 9.4), sharex=True)
    ax = axes[0]
    for i, (lab, col) in enumerate((("X", "#2a78d6"), ("Y", "#eb6834"), ("Z", "#1baf7a"))):
        ax.step(t, split[:, i], where="mid", color=col, linewidth=1.8, label=f"axis {lab}")
    ax.set_ylabel("delegation split")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title(f"{key} — how much of each axis the policy hands to the environment",
                 loc="left", fontsize=14, color=INK)

    ax = axes[1]
    ax.axhline(0, color=MUTED, linewidth=1.2)
    ax.fill_between(t, 0, B, where=B >= 0, color=C["v8b"], alpha=0.55,
                    step="mid", label="B > 0 — environment helping")
    ax.fill_between(t, 0, B, where=B < 0, color="#e34948", alpha=0.55,
                    step="mid", label="B < 0 — environment fighting")
    ax.set_ylabel("delegation benefit B")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    ax = axes[2]
    ax.step(t, P, where="mid", color="#b45309", linewidth=1.8)
    ax.set_ylabel("actuation saved P")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time (hours)")
    ax.set_title("P > 0 only where the authority gate genuinely withheld torque "
                 "(saturating demand)", loc="left", fontsize=12.5, color=INK2)
    _header(fig, "The delegation mechanism, step by step",
            "Split -> gate = 1 - split·0.7. B is signed: delegating into a torque that fights the controller costs reward. "
            "P is the counterfactual\nactuation saving, non-zero only when the gated controller was demand-limited — closing a gate that changed nothing pays nothing.",
            top=0.86)
    _save(fig, "v8_f3_delegation", "Nominal probe, seed 1001. Source: outputs/v8/data/traces.json.")


# ---------------------------------------------------------------------------
# f4 — sensing and predicting the environmental torque
# ---------------------------------------------------------------------------

def f4_torque_prediction(rows):
    if not _need(DATA / "traces.json"):
        return
    traces = json.loads((DATA / "traces.json").read_text())
    key = trace_key(traces, "v8b")
    if not key:
        print("  [skip] f4")
        return
    tr = traces[key]["nominal"]
    t = np.asarray(tr["t_h"], float)
    keep = t >= 1.0
    aero = np.asarray(tr["tau_aero"], float)[keep] * 1e6
    filt = np.asarray(tr["tau_env_filt"], float)[keep] * 1e6
    raw = np.asarray(tr["tau_env_raw"], float)[keep] * 1e6
    short = np.asarray(tr["shortfall"], float)[keep]
    tt = t[keep]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8),
                             gridspec_kw={"width_ratios": [1.8, 1.0, 1.0]})
    ax = axes[0]
    ax.plot(tt, aero, color=INK2, linewidth=2.4, label="true aero torque (plant)")
    ax.plot(tt, raw, color=MUTED, linewidth=1.0, alpha=0.8,
            label="raw 300 s observer")
    ax.plot(tt, filt, color=C["v8b"], linewidth=1.8, linestyle="--",
            label="Kalman-filtered estimate")
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("|τ| (µN·m)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Onboard estimate vs truth", loc="left", fontsize=14, color=INK)

    ax = axes[1]
    ax.scatter(aero, filt, s=26, color=C["v8b"], alpha=0.55, edgecolor="none")
    hi = max(aero.max(), filt.max()) * 1.05
    ax.plot([0, hi], [0, hi], color=INK2, linewidth=1.6, linestyle=":")
    r = float(np.corrcoef(aero, filt)[0, 1])
    ax.text(0.06, 0.94, f"r = {r:.3f}", transform=ax.transAxes, fontsize=15,
            fontweight="bold", color=INK, va="top")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("true (µN·m)")
    ax.set_ylabel("estimated (µN·m)")
    ax.set_title("Estimate vs truth", loc="left", fontsize=14, color=INK)

    ax = axes[2]
    ax.plot(tt, short * 100, color="#e34948", linewidth=1.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("unproducible torque (%)")
    ax.set_title("Field-geometry shortfall:\nτ ∥ B cannot be made at any power",
                 loc="left", fontsize=12.5, color=INK)
    _header(fig, "The spacecraft knows what the environment is doing to it",
            "τ_env = KF( I·ω̇ + ω×Iω − τ_cmd ) from the measured gyro and the controller's own commands — no extra sensor. The right panel is why\n"
            "delegation exists at all: whenever the needed torque lies near B, the magnetorquers are powerless while the aerodynamic torque is not.",
            top=0.82)
    _save(fig, "v8_f4_torque_prediction",
          f"Policy: {key}, nominal probe; first hour (detumble) excluded. Sources: traces.json.")


# ---------------------------------------------------------------------------
# f5 — storm response
# ---------------------------------------------------------------------------

def f5_storm(rows):
    if not _need(DATA / "traces.json"):
        return
    traces = json.loads((DATA / "traces.json").read_text())
    show = [("MPC", "MPC")]
    for var in ("v8a", "v8b"):
        k = trace_key(traces, var)
        if k:
            show.append((k, var))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.0))
    for ax, key, ylab, title in ((axes[0], "alt_km", "altitude (km)", "Altitude through the storm"),
                                 (axes[1], "soc", "battery state of charge", "Battery through the storm")):
        for lab, ckey in show:
            tr = traces[lab]["spike"]
            t = np.asarray(tr["t_h"], float)
            y = np.asarray(tr[key], float)
            ax.plot(t, y, color=C[ckey], linewidth=2.4, zorder=4)
            ax.text(t[-1], y[-1], f"  {ckey}", color=C[ckey], fontsize=12.5,
                    fontweight="bold", va="center")
        ax.set_xlabel("time (hours)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=14, color=INK)
        xr = ax.get_xlim()
        ax.set_xlim(0, xr[1] * 1.14)
    axes[1].axhspan(0.4, 0.6, color=C["MPC"], alpha=0.08)
    axes[1].set_ylim(0, 1.1)
    tmax = float(np.asarray(traces["MPC"]["spike"]["t_h"], float)[-1])
    for ax in axes:
        for frac, note in ((0.25, "storm onset"), (0.60, "storm clears")):
            ax.axvline(tmax * frac, color=MUTED, linewidth=1.4, linestyle=":")
            ax.annotate(note, (tmax * frac, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points",
                        fontsize=11, color=MUTED)
    _header(fig, "Storm response with the stability terms live",
            "F10.7 +100 / Ap +36 at a quarter of the run. Both v8 arms carry the altitude-margin barrier and the one-sided decay-stability term;\n"
            "only v8b may additionally hand axes to the (now much stronger) storm torque.",
            top=0.82)
    _save(fig, "v8_f5_storm", "Spike probe, seed 1001, one deterministic episode per policy.")


# ---------------------------------------------------------------------------
# f6 — where each controller shines
# ---------------------------------------------------------------------------

def f6_division(rows):
    if not _need(DATA / "ref.json"):
        return
    R = ref()
    b8 = best_row(rows, "v8b")
    if b8 is None:
        print("  [skip] f6: no v8b row")
        return
    ent = {"MPC": R["MPC"]["agg"], "Heuristic": R["Heuristic"]["agg"], "v8b": b8}
    # Dimension: (label, key, lower_is_better, unit). Compute per-dimension
    # normalized score in [0,1] where 1 = best of the three.
    dims = [("orbit longevity", "decay_nominal", True),
            ("storm longevity", "decay_spike", True),
            ("battery discipline", "band_pct", False),
            ("downlink volume", "downlink_min_d", False),
            ("actuation economy", "mtq_energy_J", True)]
    names = list(ent)
    scores = np.zeros((len(names), len(dims) + 1))
    for j, (_lab, key, lower) in enumerate(dims):
        vals = np.array([float(ent[n][key]) for n in names])
        lo, hi = vals.min(), vals.max()
        u = np.zeros_like(vals) + 0.5 if hi - lo < 1e-12 else (vals - lo) / (hi - lo)
        scores[:, j] = 1.0 - u if lower else u
    # compute-per-decision: MPC ~6 ms, heuristic ~0.05, network ~0.1 (measured
    # in the v7 study; unchanged mechanisms). Lower better.
    ms = np.array([6.16, 0.05, 0.10])
    u = (ms - ms.min()) / (ms.max() - ms.min())
    scores[:, len(dims)] = 1.0 - u
    dim_labels = [d[0] for d in dims] + ["onboard compute"]

    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    n_d = len(dim_labels)
    x = np.arange(n_d)
    w = 0.26
    for i, name in enumerate(names):
        ax.bar(x + (i - 1) * w, scores[i], width=w * 0.92, color=C[name],
               edgecolor=SURF, linewidth=1.6, zorder=3, label=name)
        for xi, v in zip(x + (i - 1) * w, scores[i]):
            if v >= 0.999:
                ax.text(xi, v + 0.03, "◆", ha="center", fontsize=11, color=C[name])
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=12)
    ax.set_ylabel("normalised score (1 = best of the three)")
    ax.set_ylim(0, 1.18)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _header(fig, "Division of strengths — who wins which axis of the mission",
            "Each dimension normalised across the three controllers on the identical plant; ◆ marks the winner. Compute-per-decision from the\n"
            "measured v7 timings (the decision mechanisms are unchanged): MPC 6.2 ms, heuristic 0.05 ms, network 0.10 ms.",
            top=0.82)
    _save(fig, "v8_f6_division",
          "Sources: outputs/v8/ledger.csv (best v8b), data/ref.json. Normalisation is within this trio; it shows WHERE each wins, not by how much in absolute units.")


# ---------------------------------------------------------------------------
# f7 — Duo (set 2)
# ---------------------------------------------------------------------------

def f7_duo(rows):
    if not _need(DATA / "duo_eval.json", DATA / "ref.json"):
        return
    duo = json.loads((DATA / "duo_eval.json").read_text())
    R = ref()
    b8 = best_row(rows, "v8b")
    entries = [("MPC", R["MPC"]["agg"]), ("v8b", b8), ("Duo", duo["agg"])]
    panels = [("decay_nominal", "decay nominal (km/d)", True),
              ("decay_spike", "decay in storm (km/d)", True),
              ("band_pct", "battery band (%)", False),
              ("downlink_min_d", "downlink (min/d)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    x = np.arange(len(entries))
    for ax, (key, title, lower) in zip(axes, panels):
        vals = [float(e[1][key]) for e in entries]
        ax.bar(x, vals, color=[C[e[0]] for e in entries], width=0.62,
               edgecolor=SURF, linewidth=2, zorder=3)
        best = (min if lower else max)(vals)
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(vals) * 0.03, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=12.5, fontweight="bold" if v == best else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in entries])
        ax.set_title(title, loc="left", fontsize=13, color=INK)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="x", visible=False)
    _header(fig, "SC_v8Duo — does a 6-hour conscience help the 300-second reflex?",
            f"Main advisor: {duo.get('main', '?')}. The horizon advisor sees only the FP32 propagation of the pending command; a deterministic\n"
            "arbiter (feasibility -> battery/decay/downlink overrides -> fixed weights) picks between them. No learned mixing.",
            top=0.80)
    _save(fig, "v8_f7_duo", "Sources: data/duo_eval.json, ledger best v8b, ref.json.")


# ---------------------------------------------------------------------------
# f8 / f9 — v9 (set 3)
# ---------------------------------------------------------------------------

def f8_v9(rows):
    have = [v for v in ("v8b", "v9a", "v9b", "v9c", "v9d", "v10", "v11")
            if best_row(rows, v)]
    if len(have) < 2:
        print("  [skip] f8: v9 rows not in ledger yet")
        return
    entries = [(v, best_row(rows, v)) for v in have]
    panels = [("decay_nominal", "decay nominal (km/d)", True),
              ("decay_spike", "decay in storm (km/d)", True),
              ("band_pct", "battery band (%)", False),
              ("brownouts", "brownouts", True)]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
    x = np.arange(len(entries))
    for ax, (key, title, lower) in zip(axes, panels):
        vals = [float(e[1][key]) for e in entries]
        ax.bar(x, vals, color=[C[e[0]] for e in entries], width=0.6,
               edgecolor=SURF, linewidth=2, zorder=3)
        best = (min if lower else max)(vals)
        for xi, v in enumerate(vals):
            ax.text(xi, v + max(max(vals), 1) * 0.03, f"{v:.1f}", ha="center",
                    fontsize=12.5, fontweight="bold" if v == best else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([e[0] for e in entries])
        ax.set_title(title, loc="left", fontsize=13, color=INK)
        ax.set_ylim(0, max(max(vals), 1) * 1.25)
        ax.grid(axis="x", visible=False)
    _header(fig, "The temporal-context ladder",
            "v9a: +3 future propagations · v9b: +3 stored past · v9c: 6 past (150-900 s) · v9d: 6 past AND 6 future · v10: the winning\n"
            "layout plus the widened-weather, tripled-horizon training envelope. Reward identical within each generation.",
            top=0.80)
    _save(fig, "v8_f8_v9", "Best run per variant. Source: outputs/v8/ledger.csv.")


def f9_leaderboard(rows):
    if not rows:
        return
    fams = ("v8a", "v8b", "v9a", "v9b", "v9c", "v9d", "v10", "v11")
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    pos, ticks, labels = 0, [], []
    for fam in fams:
        pool = [r for r in rows if r["variant"] == fam]
        if not pool:
            continue
        g = [r["gap_index"] for r in pool]
        ax.scatter([pos] * len(g), g, s=90, color=C[fam], marker=MARK[fam],
                   edgecolor=SURF, linewidth=1.4, zorder=4, alpha=0.85)
        m = float(np.median(g))
        ax.plot([pos - 0.24, pos + 0.24], [m, m], color=C[fam], linewidth=3.2,
                zorder=5, solid_capstyle="round")
        b = min(g)
        ax.annotate(f"best {b:.2f}", (pos, b), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=11, color=INK)
        ticks.append(pos)
        labels.append(f"{fam}\n(n={len(g)})")
        pos += 1
    for tag, key in (("duo", "Duo"), ("v9duo", "v9Duo")):
        try:
            duo = json.loads((DATA / f"{tag}_eval.json").read_text())
            ax.scatter([pos], [duo["gap_index"]], s=200, color=C[key],
                       marker=MARK[key], edgecolor=SURF, linewidth=2, zorder=5)
            ticks.append(pos)
            labels.append(f"{key}\n(assembled)")
            pos += 1
        except FileNotFoundError:
            pass
    ax.axhline(0, color=C["MPC"], linewidth=2.2, linestyle="--")
    ax.text(pos - 0.4, 0.03, "MPC", color=C["MPC"], fontsize=12.5, fontweight="bold",
            ha="right", va="bottom")
    hg = None
    try:
        hg_ref = ref()
        from arlamx_v2.bench_v8 import gap_index as _gi
        hg = _gi(hg_ref["Heuristic"]["agg"], hg_ref["MPC"]["agg"])
        ax.axhline(hg, color=C["Heuristic"], linewidth=2.2, linestyle=":")
        ax.text(pos - 0.4, hg + 0.03, "Heuristic", color=C["Heuristic"], fontsize=12.5,
                fontweight="bold", ha="right", va="bottom")
    except Exception:
        pass
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("gap index vs MPC (lower is better)")
    ax.grid(axis="x", visible=False)
    _header(fig, "Every model of the campaign, one axis",
            "Each marker is one trained run (all sizes and budgets); the bar is the family median. The Duo point is the assembled\n"
            "two-advisor system. Both scripted baselines were re-scored on this same v8 plant.",
            top=0.82)
    _save(fig, "v8_f9_leaderboard", "Source: outputs/v8/ledger.csv, data/duo_eval.json, data/ref.json.")


def f10_budget_curve(rows):
    pool = [r for r in rows if r["variant"] == "v10"]
    if not pool:
        print("  [skip] f10: no v10 rows")
        return
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    CAP = 3.0
    combos = sorted({(r["algo"], r["arch"]) for r in pool})
    arch_m = {"4x14": "o", "4x16": "s", "4x18": "^", "4x20": "D", "4x24": "v",
              "4x32": "P", "4x40": "*"}
    algo_c = {"ppo": "#1baf7a", "sac": "#4a3aa7", "td3": "#e34948"}
    for algo, arch in combos:
        pts = sorted(((int(r["timesteps"]), r["gap_index"]) for r in pool
                      if r["algo"] == algo and r["arch"] == arch
                      and int(r["seed"] if "seed" in r else 42) == 42),
                     key=lambda t: t[0])
        if not pts:
            continue
        x, g = zip(*pts)
        ax.plot(np.asarray(x) / 1e3, np.minimum(g, CAP), color=algo_c[algo],
                marker=arch_m.get(arch, "o"), markersize=7, linewidth=1.8,
                markeredgecolor=SURF, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 300, 500, 1000])
    ax.set_xticklabels(["50k", "100k", "200k", "300k", "500k", "1M"])
    ax.axhline(0, color=C["MPC"], linewidth=2.2, linestyle="--")
    ax.set_ylim(0, CAP * 1.05)
    ax.set_xlabel("training steps")
    ax.set_ylabel("gap index vs MPC (lower is better)")
    handles = ([Line2D([], [], color=algo_c[a], label=a.upper()) for a in algo_c]
               + [Line2D([], [], color=INK2, marker=m, linestyle="none",
                         markersize=8, label=k) for k, m in arch_m.items()
                  if any(r["arch"] == k for r in pool)])
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper right")
    _header(fig, "v10 — does more training buy MPC-level control?",
            "Every v10 cell across six budgets (seed 42; values above 3 clipped). Colour = learner, marker = size.",
            top=0.84)
    _save(fig, "v8_f10_budget_curve", "Source: outputs/v8/ledger.csv (v10 rows).")


def f11_v10_target(rows):
    if not _need(DATA / "ref.json"):
        return
    b = best_row(rows, "v10")
    if b is None:
        print("  [skip] f11: no v10 rows")
        return
    R = ref()
    mpc = R["MPC"]["agg"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6),
                             gridspec_kw={"width_ratios": [1.1, 1.0, 1.4]})
    # panel 1: decay ratio vs the 10 % target
    ax = axes[0]
    ratio = b["decay_nominal"] / mpc["decay_nominal"]
    ax.barh([1, 0], [1.10, ratio], color=[GRID, C["v10"]], height=0.5,
            edgecolor=SURF, linewidth=2)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["target\n(1.10 x MPC)", "best v10"])
    ax.axvline(1.0, color=C["MPC"], linewidth=2, linestyle="--")
    ax.text(1.0, 1.55, "MPC", color=C["MPC"], fontsize=12, fontweight="bold",
            ha="center")
    ax.text(ratio + 0.02, 0, f"{ratio:.2f}x", va="center", fontsize=14,
            fontweight="bold", color=INK)
    ax.set_xlim(0, max(2.0, ratio * 1.2))
    ax.set_xlabel("nominal decay / MPC decay")
    ax.grid(axis="y", visible=False)
    ax.set_title("The 10 % longevity target", loc="left", fontsize=14, color=INK)
    # panel 2: what it holds elsewhere
    ax = axes[1]
    keys = [("band_pct", "battery band %"), ("downlink_min_d", "downlink min/d"),
            ("brownouts", "brownouts")]
    x = np.arange(len(keys))
    ax.bar(x - 0.18, [mpc[k] for k, _ in keys], width=0.34, color=C["MPC"],
           edgecolor=SURF, linewidth=2, label="MPC")
    ax.bar(x + 0.18, [float(b[k]) for k, _ in keys], width=0.34, color=C["v10"],
           edgecolor=SURF, linewidth=2, label="best v10")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], fontsize=11.5)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    ax.set_title("Without giving these up", loc="left", fontsize=14, color=INK)
    # panel 3: U575 inference budget
    ax = axes[2]
    try:
        inf = json.loads((V8 / "data" / "inference_u575.json").read_text())
        names = [r["name"] for r in inf["rows"]]
        ms = [r["u575_ms"] for r in inf["rows"]]
        y = np.arange(len(names))[::-1]
        cols = [C["MPC"] if "MPC" in n else (C["v9Duo"] if "Duo" in n else C["v10"])
                for n in names]
        ax.barh(y, ms, color=cols, height=0.62, edgecolor=SURF, linewidth=1.6)
        for yi, v in zip(y, ms):
            ax.text(v * 1.05, yi, f"{v:.2f}", va="center", fontsize=10.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9.5)
        ax.set_xscale("log")
        ax.set_xlabel("ms per decision on the STM32U575 (analytic, conservative)")
        ax.grid(axis="y", visible=False)
        ax.set_title("Flight-computer cost", loc="left", fontsize=14, color=INK)
    except FileNotFoundError:
        ax.axis("off")
    _header(fig, "v10 against its own brief",
            f"Best v10: {b['name']}. Targets from the brief: within 10 % of the MPC's decay, battery discipline held, best coverage,\n"
            "and everything still trivially cheap next to the 300 s decision period on the flight computer.",
            top=0.80)
    _save(fig, "v8_f11_v10_target",
          "Sources: ledger (best v10), ref.json, inference_u575.json (0.35 FLOP/cycle sustained on the M33 FPU — bench on hardware).")


def f12_monte_carlo(rows):
    mc_path = DATA / "mc_decay.json"
    if not mc_path.exists():
        print("  [skip] f12: no mc_decay.json")
        return
    mc = json.loads(mc_path.read_text())
    runs = mc["runs"]
    labels, colors = [], []
    per_cp = {500: [], 400: [], 300: []}
    days = []
    for spec, rws in runs.items():
        if spec == "mpc":
            lab, col = "MPC", C["MPC"]
        elif spec == "heuristic":
            lab, col = "Heuristic", C["Heuristic"]
        else:
            nm = spec[4:]
            lab = " ".join(nm.split("_")[:2]).replace("v10 ", "v10 ") +                   f" {nm.split('_')[3]}"
            col = C["v10"]
        labels.append(lab)
        colors.append(col)
        for cp in (500, 400, 300):
            per_cp[cp].append([r[f"decay_{cp}"] for r in rws
                               if r[f"decay_{cp}"] is not None])
        days.append([r["days_500_to_300"] for r in rws if r["days_500_to_300"]])

    fig, axes = plt.subplots(1, 4, figsize=(19, 6.0))
    for ax, cp in zip(axes[:3], (500, 400, 300)):
        data = per_cp[cp]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=2))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
            patch.set_edgecolor(col)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5)
        ax.set_title(f"decay crossing {cp} km (km/day)", loc="left",
                     fontsize=13.5, color=INK)
        ax.grid(axis="x", visible=False)
    ax = axes[3]
    med = [float(np.median(d)) for d in days]
    ax.bar(np.arange(len(labels)), med, color=colors, edgecolor=SURF,
           linewidth=2, zorder=3)
    for xi, v in enumerate(med):
        ax.text(xi, v + max(med) * 0.02, f"{v:.1f}", ha="center", fontsize=11.5,
                fontweight="bold", color=INK)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5)
    ax.set_title("median days, 500 → 300 km", loc="left", fontsize=13.5, color=INK)
    ax.grid(axis="x", visible=False)
    n = mc["config"]["n_runs"]
    _header(fig, "Monte Carlo: the same policies across the whole altitude band",
            f"{n} randomized orbits per policy (inclination, node, phase, epoch weather, in-episode storms), each flown 500 → 295 km.\n"
            "Decay is sampled as the vehicle CROSSES each checkpoint, so policies are compared at the same physics. Boxes: quartiles, no fliers.",
            top=0.82)
    _save(fig, "v8_f12_monte_carlo",
          "Source: outputs/v8/data/mc_decay.json. The best v10 model's advantage concentrates at the 300 km deep end — the regime its coils and delegation were sized for.")


def f13_damage(rows):
    dpath = DATA / "damage_eval.json"
    if not dpath.exists():
        print("  [skip] f13: no damage_eval.json")
        return
    dmg = json.loads(dpath.read_text())
    kinds = ("hole", "tear", "multi")
    metrics = [("terr_post_deg", "post-strike tracking error (deg)", True),
               ("recovery_steps", "recovery time (steps)", True),
               ("decay_post", "post-strike decay (km/day)", True),
               ("B_post", "post-strike delegation B", False)]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.8))
    pols = list(dmg)
    width = 0.8 / len(pols)
    for ax, (key, title, lower) in zip(axes, metrics):
        for i, pol in enumerate(pols):
            vals = [dmg[pol][k]["median"].get(key) for k in kinds]
            vals = [v if v is not None else np.nan for v in vals]
            ckey = pol.split()[0] if pol.split()[0] in C else "MPC"
            ax.bar(np.arange(len(kinds)) + (i - (len(pols) - 1) / 2) * width,
                   vals, width * 0.9, color=C.get(ckey, INK2), edgecolor=SURF,
                   linewidth=1.6, label=pol if ax is axes[0] else None, zorder=3)
        ax.set_xticks(np.arange(len(kinds)))
        ax.set_xticklabels(kinds)
        ax.set_title(title, loc="left", fontsize=13, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=10.5)
    _header(fig, "When the sail tears: who keeps flying",
            "Forced strikes (8 seeds per kind, medians). v11 trained on strikes; v10 is the identical brief that never saw one;\n"
            "the MPC's internal model silently keeps the pre-strike panel table — exactly what an onboard model-based controller would suffer.",
            top=0.80)
    _save(fig, "v8_f13_damage", "Source: outputs/v8/data/damage_eval.json.")


SETS = {1: (f1_scorecard, f2_size_sweep, f3_delegation, f4_torque_prediction,
            f5_storm, f6_division),
        2: (f7_duo,),
        3: (f8_v9, f9_leaderboard, f10_budget_curve, f11_v10_target,
            f12_monte_carlo, f13_damage)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=3, choices=(1, 2, 3))
    a = ap.parse_args()
    _style()
    rows = ledger() if (V8 / "ledger.csv").exists() else []
    print(f"[plots_v8] ledger rows: {len(rows)} -> {PLOTS}")
    for s in range(1, a.set + 1):
        for fn in SETS[s]:
            fn(rows)
    print("[plots_v8] done")


if __name__ == "__main__":
    main()
