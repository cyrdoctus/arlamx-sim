"""Talk-set figures that do not mix quiet total-J with MC J/day.

Writes ONLY to outputs/v14fix/. Reads existing JSON. Does not train or MC.

    PYTHONPATH=python python -m arlamx_v2.plot_v14fix
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

OUT = OUTPUTS / "v14fix"
V13 = OUTPUTS / "v13"
MC_JSON = V13 / "traces" / "mc_four.json"
EVAL_JSON = V13 / "traces" / "eval10.json"
TALK_JSON = V13 / "traces" / "talk_eval.json"
ARCH_SRC = OUTPUTS / "v14" / "plots"

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de", "#fcfcfb"
COLS = {
    "MPC": "#2a78d6",
    "SC_v13": "#b45309",
    "SC_v14d": "#155e75",
    "SC_v14b": "#6d28d9",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 12, "axes.titlesize": 13.5,
        "legend.fontsize": 11, "figure.dpi": 120,
    })


def _save(fig, stem, note=None):
    if note:
        fig.text(0.006, 0.004, note, fontsize=8.5, color=MUTED,
                 ha="left", va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / stem}.png/.pdf", flush=True)


def _require(path: Path):
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def load_mc():
    payload = _require(MC_JSON)
    s = payload.get("summary") or {}
    u575 = payload.get("u575_ms") or {}
    need = ("MPC", "SC_v13", "SC_v14d", "SC_v14b")
    missing = [n for n in need if n not in s]
    if missing:
        raise SystemExit(f"mc_four.json summary missing {missing}")
    for n in need:
        sm = s[n]
        for k in ("days_500_to_300", "band_pct", "mtq_J_per_day",
                  "downlink_min_d", "deleg_B"):
            if not sm.get(k) or sm[k].get("mean") is None:
                raise SystemExit(f"mc_four.json {n}.{k}.mean missing")
        if sm.get("brownouts_any") is None:
            raise SystemExit(f"mc_four.json {n}.brownouts_any missing")
        if "n" not in sm or "reached_floor" not in sm:
            raise SystemExit(f"mc_four.json {n} n/reached_floor missing")
    if "MPC" not in u575 or "SC_v13" not in u575 or "SC_v14d" not in u575:
        raise SystemExit("mc_four.json u575_ms missing MPC/SC_v13/SC_v14d")
    return payload


def load_quiet():
    ev = _require(EVAL_JSON)
    talk = _require(TALK_JSON)
    if "MPC" not in ev:
        raise SystemExit("eval10.json missing MPC")
    if "SC_v13" not in ev and "SC_v13" not in talk:
        raise SystemExit("no SC_v13 in eval10/talk_eval")
    if "SC_v14d" not in talk:
        raise SystemExit("talk_eval.json missing SC_v14d")
    for src, name in ((ev, "MPC"), (ev, "SC_v13"), (talk, "SC_v14d")):
        row = src[name]
        for k in ("decay_km_d", "band_pct", "downlink_min_d", "mtq_energy_J"):
            if k not in row or row[k] is None:
                raise SystemExit(f"{name}.{k} missing")
    return ev, talk


def _mc_row(sm):
    return {
        "days": float(sm["days_500_to_300"]["mean"]),
        "still": int(sm["n"] - sm["reached_floor"]),
        "n": int(sm["n"]),
        "band": float(sm["band_pct"]["mean"]),
        "jday": float(sm["mtq_J_per_day"]["mean"]),
        "dl": float(sm["downlink_min_d"]["mean"]),
        "B": float(sm["deleg_B"]["mean"]),
        "brown": int(sm["brownouts_any"]),
    }


def fig_mc_scorecard(payload):
    """02: same 24-draw MC on every panel. No freeze 3.73 J."""
    _style()
    s = payload["summary"]
    u575 = payload["u575_ms"]
    order = ("MPC", "SC_v13", "SC_v14d", "SC_v14b")
    rows = {n: _mc_row(s[n]) for n in order}
    for n in order:
        rows[n]["u575"] = float(u575[n])

    print("=== 02_mc_scorecard numbers from mc_four.json ===", flush=True)
    for n in order:
        r = rows[n]
        print(f"  {n:8s}  days={r['days']:.6f}  still={r['still']}/{r['n']}  "
              f"band={r['band']:.6f}  J/day={r['jday']:.6f}  "
              f"dl={r['dl']:.6f}  brown={r['brown']}  "
              f"U575={r['u575']:.3f} ms", flush=True)

    panels = [
        ("days", "Days 500→300 km\nhigher better  (still >300 / 24)", "{:.1f}", False, True),
        ("band", "SoC in 40–60 % band\n% of time, higher better", "{:.1f}", False, False),
        ("dl", "Downlink contact\nmin/day, higher better", "{:.1f}", False, False),
        ("jday", "Coil energy\nJ/day, lower better", "{:.2f}", True, False),
        ("brown", "Brownout runs\nout of 24, lower better", "{:.0f}", True, False),
        ("u575", "STM32U575 decision\nms, lower better", "{:.2f}", True, False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.6))
    fig.subplots_adjust(top=0.78, hspace=0.55, wspace=0.28,
                        left=0.06, right=0.98, bottom=0.12)
    x = np.arange(len(order))
    for ax, (key, title, fmt, lower, still) in zip(axes.ravel(), panels):
        vals = [float(rows[n][key]) for n in order]
        cols = [COLS[n] for n in order]
        ax.bar(x, vals, color=cols, width=0.62, edgecolor=SURF, linewidth=2, zorder=3)
        best = (min if lower else max)(vals)
        ymax = max(vals) if max(vals) > 0 else 1.0
        for xi, n, v in zip(x, order, vals):
            win = abs(v - best) < 1e-12
            ax.text(xi, v + ymax * 0.045, fmt.format(v), ha="center", va="bottom",
                    fontsize=12, color=INK, fontweight="bold" if win else "normal")
        ax.set_xticks(x)
        if still:
            ax.set_xticklabels(
                [f"{n}\n{rows[n]['still']}/{rows[n]['n']}" for n in order],
                fontsize=10)
        else:
            ax.set_xticklabels(order, fontsize=11)
        ax.set_title(title, pad=8, fontsize=12.0, color=INK)
        ax.set_ylim(0, ymax * 1.32)
        ax.grid(axis="x", visible=False)
    fig.text(0.006, 0.985, "24-draw Monte Carlo — same weather, same metric on every bar",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.006, 0.928,
             "Ship bar. Not the 10-orbit quiet probe. Coil is J/day, not total J. "
             "SC_v14d is the dual-gated v14 zip; v14e is not shipped.",
             fontsize=11.5, color=INK2, ha="left", va="top")
    _save(fig, "02_mc_scorecard",
          "MPC still wins days, band, coil, brownouts. SC wins downlink slightly "
          "and compute (6.8×). Quiet 0.56 J is a different test.")


def fig_quiet_vs_mc(payload, ev, talk):
    """03 spare: quiet TOTAL J vs MC J/day on one sheet, live MPC not freeze."""
    _style()
    s = payload["summary"]
    q = {
        "MPC": ev["MPC"],
        "SC_v13": ev["SC_v13"],
        "SC_v14d": talk["SC_v14d"],
    }
    m = {n: _mc_row(s[n]) for n in ("MPC", "SC_v13", "SC_v14d")}
    order = ("MPC", "SC_v13", "SC_v14d")
    print("=== 03_quiet_vs_mc quiet (eval10 / talk_eval) ===", flush=True)
    for n in order:
        a = q[n]
        print(f"  {n:8s}  dec={a['decay_km_d']:.6f}  band={a['band_pct']:.6f}  "
              f"dl={a['downlink_min_d']:.6f}  J={a['mtq_energy_J']:.6f}", flush=True)
    print("=== 03_quiet_vs_mc MC (mc_four.json) ===", flush=True)
    for n in order:
        r = m[n]
        print(f"  {n:8s}  days={r['days']:.6f}  band={r['band']:.6f}  "
              f"dl={r['dl']:.6f}  J/day={r['jday']:.6f}", flush=True)

    left = [
        ("decay_km_d", "Decay (km/day)\nlower better", "{:.2f}", True),
        ("band_pct", "Band 40–60 %\nhigher better", "{:.1f}", False),
        ("downlink_min_d", "Downlink (min/day)\nhigher better", "{:.1f}", False),
        ("mtq_energy_J", "Coil TOTAL J\n~0.64 d, lower better", "{:.2f}", True),
    ]
    right = [
        ("days", "Days 500→300\nhigher better", "{:.1f}", False),
        ("band", "Band 40–60 %\nhigher better", "{:.1f}", False),
        ("dl", "Downlink (min/day)\nhigher better", "{:.1f}", False),
        ("jday", "Coil J/day\nlower better", "{:.2f}", True),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(12.8, 11.2))
    fig.subplots_adjust(top=0.84, hspace=0.55, wspace=0.32,
                        left=0.08, right=0.98, bottom=0.08)
    x = np.arange(len(order))
    for i, ((lk, lt, lf, llower), (rk, rt, rf, rlower)) in enumerate(zip(left, right)):
        for col, key, title, fmt, lower, src in (
            (0, lk, lt, lf, llower, "q"),
            (1, rk, rt, rf, rlower, "m"),
        ):
            ax = axes[i, col]
            if src == "q":
                vals = [float(q[n][key]) for n in order]
            else:
                vals = [float(m[n][key]) for n in order]
            ax.bar(x, vals, color=[COLS[n] for n in order], width=0.62,
                   edgecolor=SURF, lw=2, zorder=3)
            best = (min if lower else max)(vals)
            ymax = max(vals) if max(vals) > 0 else 1.0
            for xi, v in zip(x, vals):
                win = abs(v - best) < 1e-12
                ax.text(xi, v + ymax * 0.05, fmt.format(v), ha="center",
                        va="bottom", fontsize=11, color=INK,
                        fontweight="bold" if win else "normal")
            ax.set_xticks(x)
            ax.set_xticklabels(order, fontsize=10)
            ax.set_title(title, pad=6, fontsize=11.5, color=INK)
            ax.set_ylim(0, ymax * 1.35)
            ax.grid(axis="x", visible=False)
    axes[0, 0].text(0.5, 1.38, "Quiet 10-orbit  (~15 h, TOTAL J)",
                    transform=axes[0, 0].transAxes, ha="center", va="bottom",
                    fontsize=13.5, fontweight="bold", color=INK)
    axes[0, 1].text(0.5, 1.38, "Monte Carlo 24 draws  (J/day, 28 d cap)",
                    transform=axes[0, 1].transAxes, ha="center", va="bottom",
                    fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.006, 0.985, "Same four metrics, two protocols — do not mix the coil units",
             fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.006, 0.932,
             "Left uses live 10-orbit MPC from eval10.json (coil 0.16 J), not freeze 3.73 J. "
             "Right uses mc_four.json. Live quiet downlink 7.8 min/d undersamples ground stations.",
             fontsize=11, color=INK2, ha="left", va="top")
    _save(fig, "03_quiet_vs_mc",
          "Left coil is total J over 0.64 d. Right coil is J/day. "
          "0.56 J on the left is not better than 1.83 J/day on the right.")


RADAR_RMAX = 2.0


def fig_mc_simple(payload):
    """04: the MC numbers as a plain table. Same JSON as 02."""
    _style()
    s = payload["summary"]
    u575 = payload["u575_ms"]
    order = ("MPC", "SC_v13", "SC_v14d", "SC_v14b")
    rows = {n: _mc_row(s[n]) for n in order}
    for n in order:
        rows[n]["u575"] = float(u575[n])
    metrics = [
        ("Days 500→300 (mean)", lambda r: f"{r['days']:.1f}"),
        ("Still above 300 km / 24", lambda r: f"{r['still']} / {r['n']}"),
        ("Band %  40–60 (mean)", lambda r: f"{r['band']:.1f}"),
        ("Coil J/day (mean)", lambda r: f"{r['jday']:.2f}"),
        ("Downlink min/day (mean)", lambda r: f"{r['dl']:.1f}"),
        ("B (mean)", lambda r: f"{r['B']:+.2f}"),
        ("Brownout runs / 24", lambda r: f"{r['brown']}"),
        ("U575 ms", lambda r: f"{r['u575']:.2f}"),
    ]
    header = ["Task"] + list(order)
    body = [header]
    for lab, fn in metrics:
        body.append([lab] + [fn(rows[n]) for n in order])
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.axis("off")
    fig.patch.set_facecolor(SURF)
    fig.text(0.02, 0.96, "24-draw Monte Carlo — the numbers on one sheet",
             fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.89,
             "Same 24 weather draws. Coil is J/day, not quiet total J, not freeze 3.73 J. "
             "SC_v14d is the dual-gated v14 zip. v14e is not here.",
             fontsize=11, color=INK2, va="top")
    col_w = [0.34] + [0.16] * len(order)
    tab = ax.table(cellText=body, loc="upper left", cellLoc="center",
                   colWidths=col_w, bbox=[0.02, 0.10, 0.96, 0.72])
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1.0, 1.55)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(weight="bold", color=INK)
            if c > 0:
                cell.set_text_props(weight="bold", color=COLS[order[c - 1]])
        else:
            cell.set_facecolor(SURF if r % 2 else "#F4F4F4")
            cell.set_text_props(color=INK)
            if c == 0:
                cell.set_text_props(ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "04_mc_simple",
          "MPC wins days, band, coil J/day, brownouts. SC wins downlink slightly and compute.")


def fig_mc_radar(payload):
    """05: one radar, 24-draw MC vs MPC MC mean. Not column D, not quiet J."""
    _style()
    s = payload["summary"]
    u575 = payload["u575_ms"]
    mpc = _mc_row(s["MPC"])
    mpc["u575"] = float(u575["MPC"])
    names = ("SC_v13", "SC_v14d", "SC_v14b")
    labels = ["days", "band", "downlink", "B\nalign", "coil J/day\n(lower)",
              "no brownout", "compute\n(U575)"]

    def orient(name):
        r = _mc_row(s[name])
        r["u575"] = float(u575[name])
        days = r["days"] / max(mpc["days"], 0.05)
        band = r["band"] / max(mpc["band"], 1.0)
        dl = r["dl"] / max(mpc["dl"], 0.1)
        b = (r["B"] + 0.5) / 0.5
        j = mpc["jday"] / max(r["jday"], 0.01)
        nb = 1.0 - r["brown"] / max(r["n"], 1)
        cmpu = mpc["u575"] / max(r["u575"], 1e-6)
        vals = [days, band, dl, b, j, nb, min(cmpu / 3.4, RADAR_RMAX)]
        return [float(np.clip(v, 0.0, RADAR_RMAX)) for v in vals]

    print("=== 05_mc_radar ratios vs MPC MC mean (1.0) ===", flush=True)
    for n in names:
        print(f"  {n:8s}  {orient(n)}", flush=True)

    fig, ax = plt.subplots(figsize=(9.6, 8.6), subplot_kw=dict(polar=True))
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.08, right=0.78)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    ax.plot(ang, [1.0] * (len(labels) + 1), color=COLS["MPC"], lw=2.0, ls="--",
            label="MPC MC mean (1.0)", clip_on=True, zorder=2)
    ax.fill(ang, [1.0] * (len(labels) + 1), color=COLS["MPC"], alpha=0.05,
            clip_on=True)
    for name in names:
        vals = orient(name) + orient(name)[:1]
        ax.plot(ang, vals, color=COLS[name], lw=2.3, label=name, clip_on=True)
        ax.fill(ang, vals, color=COLS[name], alpha=0.10, clip_on=True)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, RADAR_RMAX)
    try:
        ax.set_rmax(RADAR_RMAX)
    except Exception:
        pass
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5", "1.0 match", "1.5", "2.0"], color=INK2, fontsize=9)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.12, 0.55),
              fontsize=11)
    fig.suptitle("24-draw Monte Carlo vs sampling MPC (same weather)",
                 fontsize=16, fontweight="bold", y=0.97, x=0.42)
    fig.text(0.5, 0.035,
             "Dashed ring = MPC mean on these 24 draws (16.4 d, 76.9 % band, 1.83 J/day, "
             "0 brownouts, 6.81 ms) — not the quiet freeze 3.73 J. "
             "Outside the ring = better than MPC on that spoke. Coil spoke is "
             "(1.83 J/day) / (policy J/day). Compute 6.8× is clipped at 2.",
             ha="center", fontsize=9, color=INK2)
    _save(fig, "05_mc_radar",
          "Dynamic weather, 24 draws. Not the 10-orbit quiet radar. Do not mix with key_radar.")


def fig_flight_strip():
    """06: ~4 h around storm onset. Tall/narrow so it sits beside 05_mc_radar."""
    path = V13 / "traces" / "sc_v14d__storm1d.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    d = np.load(path)
    t = np.asarray(d["t_h"], float)
    ap = np.asarray(d["ap"], float)
    storm = ap > 40.0
    if not np.any(storm):
        raise SystemExit("sc_v14d storm1d has no Ap>40 window")
    k_on = int(np.argmax(storm))
    t_on = float(t[k_on])
    t0, t1 = t_on - 1.5, t_on + 2.5
    m = (t >= t0) & (t <= t1)
    if int(np.sum(m)) < 6:
        raise SystemExit("storm window too short")
    tw = t[m]
    B = np.asarray(d["deleg_B"], float)[m]
    soc = np.asarray(d["soc"], float)[m] * 100.0
    dl = np.asarray(d["downlink"], float)[m]
    p_mw = np.asarray(d["mtq_quad_W"], float)[m] * 1e3
    sma = np.asarray(d["sma_km"], float)[m]
    dsma = sma - sma[0]
    split = np.asarray(d["split"], float)[m]
    smean = np.mean(split, axis=1)
    storm_w = ap[m] > 40.0

    print(f"=== 06_flight_strip SC_v14d  t={t0:.2f}–{t1:.2f} h  "
          f"onset={t_on:.2f} h  n={int(np.sum(m))} ===", flush=True)

    _style()
    # Match 05_mc_radar height (~8.6 in) and stay narrow for a PDF side-by-side.
    fig, axes = plt.subplots(4, 1, figsize=(5.4, 8.6), sharex=True)
    fig.subplots_adjust(top=0.88, hspace=0.18, left=0.18, right=0.96, bottom=0.08)
    for ax in axes:
        if np.any(storm_w):
            ax.fill_between(tw, 0, 1, where=storm_w, color="#e34948", alpha=0.10,
                            transform=ax.get_xaxis_transform(), zorder=0)
        ax.axvline(t_on, color="#ea580c", lw=1.2, ls="--", zorder=2)
        ax.set_xlim(t0, t1)
        ax.grid(axis="x", visible=False)

    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].plot(tw, B, color=COLS["SC_v14d"], lw=1.8, zorder=3)
    axes[0].plot(tw, smean, color=MUTED, lw=1.0, ls=":", zorder=3)
    axes[0].set_ylabel("B  /  s̄")
    axes[0].set_ylim(-1.05, 1.15)
    axes[0].text(0.02, 0.90, "B  env as actuator", transform=axes[0].transAxes,
                 fontsize=9, color=COLS["SC_v14d"], va="top")
    axes[0].text(0.98, 0.90, "s̄  mean split", transform=axes[0].transAxes,
                 fontsize=9, color=MUTED, va="top", ha="right")

    axes[1].plot(tw, np.maximum(p_mw, 0.0), color="#4a3aa7", lw=1.6)
    axes[1].set_ylabel("coil mW")
    ymax_p = max(float(np.max(p_mw)), 0.05)
    axes[1].set_ylim(0, ymax_p * 1.25)

    axes[2].axhspan(40, 60, color="#1baf7a", alpha=0.12)
    axes[2].plot(tw, soc, color="#eda100", lw=1.7)
    axes[2].fill_between(tw, 0, 8, where=dl > 0.5, color="#2a78d6", step="mid",
                         alpha=0.85, zorder=3)
    axes[2].set_ylabel("SoC %")
    axes[2].set_ylim(0, 105)
    axes[2].text(0.02, 0.90, "SoC", transform=axes[2].transAxes,
                 fontsize=9, color="#c98500", va="top")
    axes[2].text(0.98, 0.90, "dl", transform=axes[2].transAxes,
                 fontsize=9, color="#2a78d6", va="top", ha="right")

    axes[3].plot(tw, dsma, color=INK, lw=1.7)
    axes[3].set_ylabel("ΔSMA km")
    axes[3].set_xlabel("mission time (h)")
    lo, hi = float(np.min(dsma)), float(np.max(dsma))
    pad = max(0.05, 0.2 * (hi - lo if hi > lo else 1.0))
    axes[3].set_ylim(lo - pad, hi + pad)

    fig.text(0.02, 0.985, "SC_v14d  ·  storm onset  ·  4 h",
             fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.948,
             "Dashed = Ap jump. Shade = Ap > 40. Not the 24-draw MC.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    _save(fig, "06_flight_strip",
          "Companion to 05_mc_radar. Coil is mW this hour, not J/day. Dual-gated v14d zip.")


def _mrp_quat(sig):
    s2 = float(np.dot(sig, sig))
    q = np.array([(1.0 - s2), 2.0 * sig[0], 2.0 * sig[1], 2.0 * sig[2]]) / (1.0 + s2)
    n = np.linalg.norm(q)
    q = q / max(n, 1e-12)
    return -q if q[0] < 0 else q


def _dcm_bn(q):
    q0, q1, q2, q3 = q
    return np.array([
        [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
        [2 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 + q0 * q1)],
        [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])


def _ang_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _fmt_force(n):
    u = abs(float(n)) * 1e6
    if u >= 1000.0:
        return f"{u / 1000.0:.2f} mN"
    return f"{u:.0f} µN"


def _load_simplified():
    cache = OUTPUTS / "analysis" / "showcase" / "geometry_simplified.npz"
    if cache.exists():
        g = np.load(cache)
        return g["tris"], g["tri_normals"]
    from arlamx_v2 import ipc_animation as IA
    return IA.simplified_model()


def _pick_attitude_frames(d):
    """Quiet: sunlit, max |body +X · v| (ram along the 1 cm cp axis).
    Storm: sunlit Ap>40, max |F_est| (aero on the sail)."""
    n = len(d["t_h"])
    best_q, best_s = (-1.0, 0), (-1.0, max(0, n - 1))
    for k in range(n):
        if float(d["eclipse"][k]) <= 0.5:
            continue
        c_bn = _dcm_bn(_mrp_quat(np.asarray(d["sigma"][k], float)))
        vn = d["v"][k] / max(np.linalg.norm(d["v"][k]), 1e-12)
        xv = abs(float((c_bn @ vn)[0]))
        if float(d["ap"][k]) <= 40.0:
            if xv > best_q[0]:
                best_q = (xv, k)
        else:
            f = float(np.linalg.norm(d["F_est"][k]))
            if f > best_s[0]:
                best_s = (f, k)
    return int(best_q[1]), int(best_s[1])


def _label3d(ax, vec, s, lab, col, mesh_r, nudge=0.10):
    p = np.asarray(vec, float) * s * 1.06
    n = np.cross(vec, np.array([0.12, 0.35, 0.92]))
    nn = n / max(float(np.linalg.norm(n)), 1e-9)
    p = p + nn * mesh_r * nudge
    ax.text(p[0], p[1], p[2], lab, color=col, fontsize=13,
            fontweight="bold", ha="center", va="center", zorder=12)


def _draw_attitude_still(ax, d, k, tris, tri_n, f_ref):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sig = np.asarray(d["sigma"][k], float)
    c_bn = _dcm_bn(_mrp_quat(sig))
    c_nb = c_bn.T
    world = tris @ c_nb.T
    lit = float(d["eclipse"][k]) > 0.5
    sun_n = np.asarray(d["sun_N"][k], float)
    sn = sun_n / max(np.linalg.norm(sun_n), 1e-12)
    sun_b = c_bn @ sn
    base_rgb = np.array([0.95, 0.76, 0.31])
    dark_rgb = np.array([0.36, 0.39, 0.44])
    if lit:
        shade = 0.32 + 0.68 * np.abs(tri_n @ sun_b)
        colors = np.clip(shade[:, None] * base_rgb[None, :], 0, 1)
    else:
        colors = np.tile(dark_rgb, (len(tris), 1))
    coll = Poly3DCollection(list(world), linewidths=0.22,
                            edgecolors=(0.12, 0.11, 0.10, 0.32))
    coll.set_facecolor(np.c_[colors, np.full(len(colors), 0.97)])
    ax.add_collection3d(coll)

    mesh_r = float(np.abs(tris).max())
    L = mesh_r * 1.22
    s_body = mesh_r * 0.98
    s_env = mesh_r * 1.16
    vhat = d["v"][k] / max(np.linalg.norm(d["v"][k]), 1e-12)
    x_b = c_nb @ np.array([1.0, 0.0, 0.0])
    z_b = c_nb @ np.array([0.0, 0.0, 1.0])
    f_b = np.asarray(d["F_est"][k], float)
    fmag = float(np.linalg.norm(f_b))
    fhat = c_nb @ (f_b / max(fmag, 1e-18))
    frac = float(np.sqrt(np.clip(fmag / max(f_ref, 1e-18), 0.0, 1.0)))
    s_f = s_env * (0.48 + 0.62 * frac)
    lw_f = 2.2 + 2.6 * frac
    tau_a = float(np.linalg.norm(d["tau_aero"][k]))
    xv = float(np.dot(x_b, vhat))
    zv = float(np.dot(z_b, vhat))

    ax.plot([-vhat[0] * s_env, vhat[0] * s_env],
            [-vhat[1] * s_env, vhat[1] * s_env],
            [-vhat[2] * s_env, vhat[2] * s_env],
            color="0.72", lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.quiver(0, 0, 0, *(sn * s_env * 0.92), color="#eda100", lw=2.6,
              arrow_length_ratio=0.11, zorder=6)
    ax.quiver(0, 0, 0, *(vhat * s_env), color="#3f3e3b", lw=3.0,
              arrow_length_ratio=0.11, zorder=6)
    ax.quiver(0, 0, 0, *(x_b * s_body), color="#0f766e", lw=3.0,
              arrow_length_ratio=0.13, zorder=7)
    ax.quiver(0, 0, 0, *(z_b * s_body), color="#2a78d6", lw=3.0,
              arrow_length_ratio=0.13, zorder=7)
    ax.quiver(0, 0, 0, *(fhat * s_f), color="#e34948", lw=lw_f,
              arrow_length_ratio=0.15, zorder=8)

    _label3d(ax, sn, s_env * 0.92, "Sun", "#c98500", mesh_r, 0.08)
    if abs(xv) > 0.90:
        ax.text2D(0.62, 0.78, "v ≈ +X", transform=ax.transAxes, fontsize=13.5,
                  fontweight="bold", color="#3f3e3b", ha="left", va="center",
                  bbox=dict(boxstyle="round,pad=0.28", facecolor=SURF,
                            edgecolor="#3f3e3b", linewidth=0.7))
    else:
        _label3d(ax, vhat, s_env, "v", "#3f3e3b", mesh_r, 0.14)
        _label3d(ax, x_b, s_body, "+X", "#0f766e", mesh_r, 0.10)
    _label3d(ax, z_b, s_body, "+Z", "#2a78d6", mesh_r, 0.10)
    _label3d(ax, fhat, s_f, "F", "#c0392b", mesh_r, 0.12)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.62)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=36)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_axis_off()
    ax.set_facecolor(SURF)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor(SURF)
            axis.pane.set_edgecolor(SURF)
            axis.pane.set_alpha(0)
        except Exception:
            pass
    return {
        "t": float(d["t_h"][k]),
        "ap": float(d["ap"][k]),
        "alt": float(d["alt_km"][k]),
        "lit": lit,
        "xv": xv,
        "zv": zv,
        "axv": _ang_deg(x_b, vhat),
        "azv": _ang_deg(z_b, vhat),
        "F": fmag,
        "tau": tau_a,
        "cd": float(d["cd"][k]),
        "B": float(d["deleg_B"][k]),
        "storm": float(d["ap"][k]) > 40.0,
    }


def _attitude_card(ax, heading, info):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tag_c = "#e34948" if info["storm"] else "#0f766e"
    ax.text(0.0, 0.96, heading, fontsize=15.5, fontweight="bold", color=tag_c,
            va="top", ha="left")
    ax.text(0.0, 0.68,
            f"t = {info['t']:.2f} h   Ap {info['ap']:.0f}   {info['alt']:.0f} km   "
            f"{'sunlit' if info['lit'] else 'eclipse'}   "
            f"C_d {info['cd']:.2f}   B {info['B']:+.2f}",
            fontsize=11.5, color=INK2, va="top", ha="left")
    ax.text(0.0, 0.38,
            f"+X to ram   {info['axv']:.0f}°    (+X · v = {info['xv']:+.2f})",
            fontsize=13, color="#0f766e", va="top", ha="left")
    ax.text(0.0, 0.10,
            f"+Z to ram   {info['azv']:.0f}°    (+Z · v = {info['zv']:+.2f})",
            fontsize=13, color="#2a78d6", va="top", ha="left")
    ax.text(0.70, 0.38, f"|F|  {_fmt_force(info['F'])}",
            fontsize=14, fontweight="bold", color="#c0392b", va="top", ha="left")
    ax.text(0.70, 0.10, f"|τ|  {info['tau'] * 1e6:.2f} µN·m",
            fontsize=13, color=INK, va="top", ha="left")


def fig_attitude():
    """07: two large simplified-CAD stills — velocity pointing and aero force."""
    path = V13 / "traces" / "sc_v14d__storm1d.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    d = np.load(path)
    tris, tri_n = _load_simplified()
    k_q, k_s = _pick_attitude_frames(d)
    f_q = float(np.linalg.norm(d["F_est"][k_q]))
    f_s = float(np.linalg.norm(d["F_est"][k_s]))
    f_ref = max(f_q, f_s, 1e-12)
    print(f"=== 07_attitude SC_v14d  quiet k={k_q} t={d['t_h'][k_q]:.2f} h  "
          f"|F|={f_q * 1e6:.1f} µN   storm k={k_s} t={d['t_h'][k_s]:.2f} h  "
          f"|F|={f_s * 1e6:.1f} µN ===", flush=True)

    _style()
    fig = plt.figure(figsize=(16.8, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[6.4, 1.7], hspace=0.02, wspace=0.05,
                          left=0.015, right=0.995, top=0.86, bottom=0.04)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    info_q = _draw_attitude_still(ax0, d, k_q, tris, tri_n, f_ref)
    info_s = _draw_attitude_still(ax1, d, k_s, tris, tri_n, f_ref)
    _attitude_card(fig.add_subplot(gs[1, 0]), "quiet  ·  velocity pointing", info_q)
    _attitude_card(fig.add_subplot(gs[1, 1]), "in storm  ·  aero force", info_s)

    fig.text(0.012, 0.978, "Attitude  —  simplified SolarCat, velocity pointing and forces",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.928,
             "SC_v14d, storm1d, same camera.  Gold Sun, grey ram v, teal body +X (cp–cm), "
             "blue +Z sail/antenna, red onboard force F.  "
             "+X along ram = edge-on (small |F|).  +Z toward ram = sail broadside.  "
             "Red length ∝ √|F|, shared scale.",
             fontsize=11, color=INK2, ha="left", va="top")
    _save(fig, "07_attitude")


def copy_architecture():
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        src = ARCH_SRC / f"key_architecture.{ext}"
        if not src.exists():
            src = ARCH_SRC / f"fig0_architecture.{ext}"
        if not src.exists():
            raise SystemExit(f"no architecture source {ARCH_SRC}/key_architecture.{ext}")
        dest = OUT / f"01_architecture.{ext}"
        shutil.copy2(src, dest)
        print(f"  copied {src.name} -> {dest.name}", flush=True)


def main():
    print(f"[v14fix] out={OUT}", flush=True)
    copy_architecture()
    payload = load_mc()
    ev, talk = load_quiet()
    fig_mc_scorecard(payload)
    fig_quiet_vs_mc(payload, ev, talk)
    fig_mc_simple(payload)
    fig_mc_radar(payload)
    fig_flight_strip()
    fig_attitude()
    print("[v14fix] done", flush=True)


if __name__ == "__main__":
    main()
