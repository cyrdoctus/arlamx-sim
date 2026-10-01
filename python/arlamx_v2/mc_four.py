"""Four-task Monte-Carlo: SC_v13 (± v13b) vs sampling MPC.

Logs longevity, power, downlink, and environmental-torque utilisation on
the *same* 24 dynamic-weather draws. Not a 4-D scatter — four panels.

    PYTHONPATH=python python -m arlamx_v2.mc_four --runs 24 --workers 12
    PYTHONPATH=python python -m arlamx_v2.mc_four --plot
    PYTHONPATH=python python -m arlamx_v2.mc_four --add-v13b --plot

Writes outputs/v13/traces/mc_four.json and outputs/v13/plots/fig8_*.
Level: advanced.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

V13 = OUTPUTS / "v13"
V13_CKPT = V13 / "ckpt"
V13B_CKPT = V13 / "ckpt_v13b"
FLOOR_KM = 295.0
MAX_STEPS = 8000
CAP_DAYS = MAX_STEPS * 300.0 / 86400.0

WRAP_V13 = dict(w_slew=0.5, w_env=0.35, use_cmp=True, clip_wrong_way=True,
                use_pwr_est=True, w_clip=0.08)
WRAP_V13B = dict(w_slew=0.5, w_env=0.35, use_cmp=False, clip_wrong_way=True,
                 use_pwr_est=True, w_clip=0.08)
WRAP_V14 = dict(w_slew=0.5, w_env=0.35, use_cmp=True, clip_wrong_way=True,
                use_pwr_est=True, w_clip=0.08, adapt_mrp=True)
WRAP_MPC = dict(w_slew=0.0, w_env=0.0, use_cmp=False, clip_wrong_way=False,
                use_pwr_est=True, w_clip=0.0)

U575 = {
    "MPC": 6.805,
    "SC_v13": 1.000,
    "SC_v13.5": 1.000,
    "SC_v14": 1.000,
    "SC_v14b": 1.000,
    "SC_v14c": 1.000,
    "SC_v14d": 1.000,
    "SC_v14e": 1.000,
    "SC_v14f": 1.000,
    "SC_v13b": 0.172,
}


def _draw(rng):
    return dict(
        altitude_km=500.0,
        inc_deg=float(rng.uniform(20.0, 40.0)),
        ecc=float(rng.uniform(0.0, 0.01)),
        raan_deg=float(rng.uniform(0.0, 360.0)),
        argp_deg=float(rng.uniform(0.0, 360.0)),
        nu_deg=float(rng.uniform(0.0, 360.0)),
        f107=float(rng.uniform(65.0, 250.0)),
        ap=float(rng.uniform(2.0, 80.0)),
        soc=0.5,
        omega_dps=[0.5, 0.5, 0.2],
    )


def _one(job):
    label, spec, idx, max_steps, floor_km = job
    import warnings
    warnings.filterwarnings("ignore")
    from arlamx_v2.env import ArlamxV2Env
    from arlamx_v2.plot_v12 import load_mpc_predict, load_v12_predict
    from arlamx_v2.train_v12 import V12Wrapper, VARIANT

    rng = np.random.default_rng(10_000 + idx)
    opts = _draw(rng)
    base = ArlamxV2Env(seed=idx, variant=VARIANT)
    if spec == "mpc":
        predict = load_mpc_predict()
        env = V12Wrapper(base, **WRAP_MPC)
    else:
        predict = load_v12_predict(Path(spec))
        if label == "SC_v13b":
            wrap = dict(WRAP_V13B)
        elif label in ("SC_v14", "SC_v14b", "SC_v14c", "SC_v14d", "SC_v14e",
                       "SC_v14f"):
            wrap = dict(WRAP_V14)
        else:
            wrap = dict(WRAP_V13)
        env = V12Wrapper(base, **wrap)
    env.env._max_steps = int(max_steps)
    obs, _ = env.reset(seed=idx, options=opts)
    sma, alt, soc, dl, mtq, B, P = [], [], [], [], [], [], []
    brown, k, done = 0, 0, False
    dep_prev = False
    while not done and k < max_steps:
        a = predict(obs, env.env)
        obs, _r, term, trunc, info = env.step(np.asarray(a, np.float32))
        k += 1
        sma.append(float(info["sma_m"]))
        alt.append(float(info["altitude_km"]))
        s = float(info.get("battery_soc", 0.5))
        soc.append(s)
        vis = float(info.get("gs_visible", 0)) > 0.5
        lit = float(info.get("eclipse", 1)) > 0.5
        dl.append(1.0 if (vis and lit and float(info.get("gs_point_cos", 0)) > 0.7) else 0.0)
        mtq.append(float(info.get("mtq_power_quad_W", info.get("mtq_power_W", 0.0))))
        B.append(float(info.get("deleg_B", 0.0)))
        P.append(float(info.get("deleg_P", 0.0)))
        dep = bool(info.get("battery_depleted"))
        if dep and not dep_prev:
            brown += 1
        dep_prev = dep
        done = term or (alt[-1] < floor_km)
    dt_s = float(env.env._advisor_s)
    env.close()
    alt = np.asarray(alt)
    sma = np.asarray(sma)
    soc = np.asarray(soc)
    days = k * dt_s / 86400.0 if k else 0.0
    hit = np.where(alt <= 300.0)[0]
    out = {
        "idx": idx, "label": label, "steps": k, "days": days,
        "reached_floor": bool(alt[-1] < floor_km + 5),
        "alt_final_km": float(alt[-1]) if len(alt) else None,
        "days_500_to_300": float(hit[0] * dt_s / 86400.0) if len(hit) else None,
        "days_capped": float(hit[0] * dt_s / 86400.0) if len(hit) else float(days),
        "censored": bool(len(hit) == 0),
        "band_pct": float(np.mean((soc >= 0.4) & (soc <= 0.6)) * 100.0) if len(soc) else 0.0,
        "soc_lo_pct": float(np.mean(soc < 0.4) * 100.0) if len(soc) else 0.0,
        "soc_hi_pct": float(np.mean(soc > 0.6) * 100.0) if len(soc) else 0.0,
        "downlink_min_d": float(np.sum(dl) * dt_s / 60.0 / max(days, 1e-9)),
        "mtq_J": float(np.sum(mtq) * dt_s),
        "mtq_J_per_day": float(np.sum(mtq) * dt_s / max(days, 1e-9)),
        "deleg_B": float(np.mean(B)) if B else 0.0,
        "deleg_P": float(np.mean(P)) if P else 0.0,
        "brownouts": int(brown),
        "decay_km_d": float((sma[0] - sma[-1]) / 1e3 / max(days, 1e-9)) if len(sma) > 1 else None,
    }
    return label, out


def _jobs(runs, add_v13b, only_new=False, extra=None):
    jobs = []
    if not only_new:
        jobs += [("MPC", "mpc", i, MAX_STEPS, FLOOR_KM) for i in range(runs)]
        jobs += [("SC_v13", str(V13_CKPT), i, MAX_STEPS, FLOOR_KM)
                 for i in range(runs)]
    if add_v13b and (V13B_CKPT / "model.zip").exists():
        jobs += [("SC_v13b", str(V13B_CKPT), i, MAX_STEPS, FLOOR_KM)
                 for i in range(runs)]
    if extra:
        lab, ckpt = extra
        jobs += [(lab, str(ckpt), i, MAX_STEPS, FLOOR_KM) for i in range(runs)]
    return jobs


def _summarise(rows):
    def nums(key, skip_none=True):
        vals = []
        for r in rows:
            if key not in r or r.get(key) is None:
                if skip_none:
                    continue
                else:
                    continue
            vals.append(float(r[key]))
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "n": len(vals),
        }
    s = {
        "n": len(rows),
        "reached_floor": int(sum(r["reached_floor"] for r in rows)),
        "censored": int(sum(r["censored"] for r in rows)),
        "brownouts_any": int(sum(r["brownouts"] > 0 for r in rows)),
        "days_500_to_300": nums("days_500_to_300"),
        "days_capped": nums("days_capped", skip_none=False),
        "band_pct": nums("band_pct", skip_none=False),
        "soc_lo_pct": nums("soc_lo_pct", skip_none=False),
        "soc_hi_pct": nums("soc_hi_pct", skip_none=False),
        "downlink_min_d": nums("downlink_min_d", skip_none=False),
        "mtq_J_per_day": nums("mtq_J_per_day", skip_none=False),
        "deleg_B": nums("deleg_B", skip_none=False),
        "decay_km_d": nums("decay_km_d"),
    }
    return s


def run_mc(runs, workers, add_v13b, merge_path=None, extra=None):
    only_new = bool((add_v13b or extra) and merge_path and Path(merge_path).exists())
    jobs = _jobs(runs, add_v13b, only_new=only_new, extra=extra)
    print(f"[mc_four] {len(jobs)} jobs, {workers} workers", flush=True)
    t0 = time.time()
    bucket = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        n_done = 0
        for lab, out in ex.map(_one, jobs, chunksize=1):
            bucket.setdefault(lab, []).append(out)
            n_done += 1
            if n_done % 8 == 0:
                print(f"[mc_four] {n_done}/{len(jobs)} ({time.time()-t0:.0f}s)",
                      flush=True)
    if merge_path and Path(merge_path).exists():
        old = json.loads(Path(merge_path).read_text())
        for lab, rows in (old.get("runs") or {}).items():
            if lab not in bucket:
                bucket[lab] = rows
    summary = {lab: _summarise(rows) for lab, rows in bucket.items()}
    for lab, s in summary.items():
        d = s["days_500_to_300"]
        print(f"  {lab:8s}  days={None if not d else round(d['mean'], 2)}  "
              f"n_fall={s['reached_floor']}/{s['n']}  "
              f"band={s['band_pct']['mean']:.1f}  "
              f"dl={s['downlink_min_d']['mean']:.1f}  "
              f"B={s['deleg_B']['mean']:+.2f}", flush=True)
    dest = V13 / "traces" / "mc_four.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary, "runs": bucket,
        "wall_s": round(time.time() - t0, 1),
        "n_draws": runs, "weather": "dynamic F10.7 U[65,250], Ap U[2,80], "
        "plus v10 in-episode storms",
        "cap_days": CAP_DAYS, "u575_ms": U575,
        "ckpt_v13": str(V13_CKPT), "ckpt_v13b": str(V13B_CKPT),
    }
    dest.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[mc_four] wrote {dest} in {time.time()-t0:.0f}s", flush=True)
    return payload


def _style():
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e4e3de", "#fcfcfb"
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
    return INK, INK2, MUTED, GRID, SURF


def plot_four(payload=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, INK2, MUTED, GRID, SURF = _style()
    if payload is None:
        p = V13 / "traces" / "mc_four.json"
        payload = json.loads(p.read_text())
    runs = payload["runs"]
    order = [n for n in ("MPC", "SC_v13", "SC_v13.5", "SC_v14", "SC_v14b",
                         "SC_v14c", "SC_v14d", "SC_v14e", "SC_v14f", "SC_v13b")
             if n in runs]
    cols = {"MPC": "#2a78d6", "SC_v13": "#b45309", "SC_v13.5": "#7c3aed",
            "SC_v14": "#0f766e", "SC_v14b": "#6d28d9", "SC_v14c": "#7c3aed",
            "SC_v14d": "#155e75", "SC_v14e": "#be185d", "SC_v14f": "#9f1239",
            "SC_v13b": "#64748b"}
    panels = [
        ("days_capped", "Longevity\ndays 500→300 km  (higher better)", False,
         "Capped at 28 d if still above 300 km (open = still flying)."),
        ("band_pct", "Power\n% time SoC in 40–60 %  (higher better)", False,
         "Mission power task. Coil J/day is in the table."),
        ("downlink_min_d", "Downlink\ncontact (min/day)  (higher better)", False,
         "Sunlit pass, boresight cosine > 0.7."),
        ("deleg_B", "Environment as actuator\nB  (+ helping, − fighting)", False,
         "MPC has no split, so B = 0 by construction."),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    fig.subplots_adjust(top=0.80, hspace=0.42, wspace=0.28,
                        left=0.07, right=0.98, bottom=0.10)
    rng = np.random.default_rng(0)
    for ax, (key, title, lower, hint) in zip(axes.ravel(), panels):
        data, colors, hats = [], [], []
        for i, name in enumerate(order):
            rows = runs[name]
            vals = np.asarray([float(r[key]) for r in rows], float)
            data.append(vals)
            colors.append(cols[name])
            bp = ax.boxplot(
                [vals], positions=[i], widths=0.46, patch_artist=True,
                showfliers=False, medianprops=dict(color=INK, lw=1.8),
                whiskerprops=dict(color=INK2, lw=1.1),
                capprops=dict(color=INK2, lw=1.1),
                boxprops=dict(facecolor=cols[name], alpha=0.35, edgecolor=cols[name], lw=1.4),
            )
            jitter = rng.normal(0, 0.06, size=len(vals))
            xj = np.full(len(vals), i) + jitter
            cens = np.asarray([bool(r.get("censored")) for r in rows]) if key == "days_capped" else np.zeros(len(vals), bool)
            ax.scatter(xj[~cens], vals[~cens], s=22, c=cols[name],
                       edgecolors=cols[name], linewidths=0.8, alpha=0.85, zorder=4)
            if np.any(cens):
                ax.scatter(xj[cens], vals[cens], s=26, facecolors="white",
                           edgecolors=cols[name], linewidths=1.2, alpha=0.95, zorder=4)
            ax.scatter([i], [np.mean(vals)], marker="D", s=42, color="white",
                       edgecolors=INK, linewidths=1.2, zorder=5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=11)
        ax.set_title(title, pad=8, fontsize=12.5, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
        if key == "deleg_B":
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.text(0.0, -0.18, hint, transform=ax.transAxes, fontsize=8.5, color=MUTED)
    spd13 = U575["MPC"] / U575["SC_v13"]
    spd_pol = U575["MPC"] / U575["SC_v13b"]
    fig.text(0.006, 0.985,
             "Monte-Carlo — four mission tasks, same 24 weather draws",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.006, 0.928,
             "Dynamic space weather: F10.7 U[65, 250] sfu, Ap U[2, 80], plus v10 in-episode storms.  "
             "Start 500 km, 28-day cap.  Diamonds = mean; boxes = median + IQR.",
             fontsize=11, color=INK2, ha="left", va="top")
    fig.text(0.006, 0.878,
             f"U575   MPC {U575['MPC']:.2f} ms    "
             f"SC_v13 policy+comparator {U575['SC_v13']:.2f} ms  ({spd13:.1f}×)    "
             f"policy only {U575['SC_v13b']:.3f} ms  ({spd_pol:.0f}×)    "
             f"budget 15 ms.  Comparator is required to hold band.",
             fontsize=11.5, color=INK, ha="left", va="top", fontweight="bold")
    v14 = any(k in runs for k in
              ("SC_v14f", "SC_v14e", "SC_v14d", "SC_v14b", "SC_v14"))
    PLOTS = (OUTPUTS / "v14" / "plots") if v14 else (V13 / "plots")
    PLOTS.mkdir(parents=True, exist_ok=True)
    note = ("Open circles on longevity = still above 300 km at the 28-day cap (censored).  "
            "Do not mix this MC with the 10-orbit quiet probe.")
    fig.text(0.006, 0.008, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")
    stems = ("fig8_mc_four_tasks", "fig8_mc_four_tasksus")
    if any(k in runs for k in ("SC_v14f", "SC_v14e", "SC_v14d", "SC_v14b", "SC_v14")):
        stems = ("fig9_mc_v14", "fig9_mc_v14us")
    elif "SC_v13.5" in runs:
        stems = ("fig9_mc_v135", "fig9_mc_v135us")
    for stem in stems:
        fig.savefig(PLOTS / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight")
    if stems[0].startswith("fig9_mc"):
        for ext in ("png", "pdf"):
            fig.savefig(PLOTS / f"key_mc_tasks.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stems[-1]}.png/.pdf", flush=True)
    _table_fig(payload, order, cols, INK, INK2, MUTED, SURF, GRID,
               v14=any(k in runs for k in
                       ("SC_v14", "SC_v14b", "SC_v14d", "SC_v14e", "SC_v14f")))
    _write_md(payload, order)


def _table_fig(payload, order, cols, INK, INK2, MUTED, SURF, GRID, v14=False):
    import matplotlib.pyplot as plt
    s = payload["summary"]
    u575 = payload.get("u575_ms", U575)
    rows_h = ["Longevity  days 500→300 (mean)",
              "          still above 300 km / 24",
              "Power      SoC in 40–60 % band (mean)",
              "          coil J/day (mean)",
              "Downlink   min/day (mean)",
              "Environment  B (mean)",
              "Brownouts  runs with any",
              "U575 policy+comparator  ms",
              "Speedup vs MPC  (flight)",
              "U575 policy only  ms  (not flown)"]
    body = [rows_h]
    for name in order:
        sm = s[name]
        days = sm["days_500_to_300"]
        days_s = "—" if not days else f"{days['mean']:.1f}"
        still = sm["n"] - sm["reached_floor"]
        spd = u575["MPC"] / max(u575.get(name, u575["MPC"]), 1e-9)
        pol_ms = "0.172" if name != "MPC" else "—"
        col = [
            days_s,
            f"{still} / {sm['n']}",
            f"{sm['band_pct']['mean']:.1f} %",
            f"{sm['mtq_J_per_day']['mean']:.2f}",
            f"{sm['downlink_min_d']['mean']:.1f}",
            f"{sm['deleg_B']['mean']:+.2f}",
            f"{sm['brownouts_any']}",
            f"{u575.get(name, float('nan')):.3f}",
            "1×" if name == "MPC" else f"{spd:.1f}×",
            pol_ms,
        ]
        body.append(col)
    # transpose to table rows
    header = ["Task"] + order
    table_data = [header]
    for i, h in enumerate(rows_h):
        table_data.append([h] + [body[j + 1][i] for j in range(len(order))])

    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.axis("off")
    fig.patch.set_facecolor(SURF)
    title = "Deep RL vs sampling MPC — Monte-Carlo numbers"
    if v14:
        title = "SC_v14 family vs sampling MPC — Monte-Carlo numbers"
    fig.text(0.02, 0.96, title,
             fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.90,
             "Same 24 dynamic-weather draws. Inference is analytic U575 (0.35 FLOP/cycle), not the MC.",
             fontsize=11, color=INK2, va="top")
    fig.text(0.02, 0.845,
             f"Speed: policy+comparator {u575['SC_v13']:.2f} ms is {u575['MPC']/u575['SC_v13']:.1f}× vs MPC "
             f"({u575['MPC']:.2f} ms). Policy-only {u575['SC_v13b']:.3f} ms is "
             f"{u575['MPC']/u575['SC_v13b']:.0f}×, but flying without the comparator "
             f"collapses band (40 % quiet, 12 % after a 200 k no-cmp train). "
             f"Flight number is 1.00 ms.",
             fontsize=11, color=INK, va="top")
    n_col = 1 + len(order)
    col_w = [0.42] + [0.58 / len(order)] * len(order)
    tab = ax.table(cellText=table_data, loc="upper left", cellLoc="center",
                   colWidths=col_w, bbox=[0.02, 0.08, 0.96, 0.70])
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
                cell.set_text_props(weight="bold", color=cols.get(order[c - 1], INK))
        else:
            cell.set_facecolor(SURF if r % 2 else "#F4F4F4")
            cell.set_text_props(color=INK)
            if c == 0:
                cell.set_text_props(ha="left")
        if r == len(table_data) - 1 and c > 0:
            cell.set_text_props(weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    PLOTS = (OUTPUTS / "v14" / "plots") if v14 else (V13 / "plots")
    PLOTS.mkdir(parents=True, exist_ok=True)
    stems = (("fig9_mc_table", "fig9_mc_tableus") if v14
             else ("fig8_mc_table", "fig8_mc_tableus"))
    for stem in stems:
        fig.savefig(PLOTS / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight")
    if v14:
        for ext in ("png", "pdf"):
            fig.savefig(PLOTS / f"key_mc_table.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stems[-1]}.png/.pdf", flush=True)


def _write_md(payload, order):
    s = payload["summary"]
    u575 = payload.get("u575_ms", U575)
    lines = [
        "# Four-task Monte-Carlo — SC_v13 vs MPC",
        "",
        "Dynamic space weather, 24 draws, 28-day cap, start 500 km.",
        "",
        "| Task | " + " | ".join(order) + " |",
        "|---|" + "|".join(["---"] * len(order)) + "|",
    ]
    def cell(name, key, fmt, inner="mean"):
        sm = s[name][key]
        if not sm:
            return "—"
        return fmt.format(sm[inner])
    rows = [
        ("Longevity days 500→300 (mean)",
         lambda n: cell(n, "days_500_to_300", "{:.1f}")),
        ("Still above 300 km / N",
         lambda n: f"{s[n]['n'] - s[n]['reached_floor']} / {s[n]['n']}"),
        ("Coil J/day (mean)",
         lambda n: cell(n, "mtq_J_per_day", "{:.2f}")),
        ("Band % (mean)",
         lambda n: cell(n, "band_pct", "{:.1f}")),
        ("Downlink min/day (mean)",
         lambda n: cell(n, "downlink_min_d", "{:.1f}")),
        ("B (mean)",
         lambda n: cell(n, "deleg_B", "{:+.2f}")),
        ("Brownout runs / N",
         lambda n: f"{s[n]['brownouts_any']} / {s[n]['n']}"),
        ("U575 ms",
         lambda n: f"{u575.get(n, float('nan')):.3f}"),
        ("Speedup vs MPC",
         lambda n: "1×" if n == "MPC" else f"{u575['MPC']/max(u575.get(n, 1), 1e-9):.1f}×"),
    ]
    for title, fn in rows:
        lines.append("| " + title + " | " + " | ".join(fn(n) for n in order) + " |")
    lines += [
        "",
        "## Inference (the point to highlight)",
        "",
        f"- Sampling MPC: **{u575['MPC']:.2f} ms** on the U575 planning model (budget 15 ms).",
        f"- SC_v13 (4×18 + 2-candidate comparator): **{u575['SC_v13']:.2f} ms** → "
        f"**{u575['MPC']/u575['SC_v13']:.1f}×** faster.",
        f"- Same network, policy only (no comparator): **{u575['SC_v13b']:.3f} ms** → "
        f"**{u575['MPC']/u575['SC_v13b']:.0f}×** faster — **not a flight mode**. "
        "Zero-shot no-cmp drops quiet band 80 % → 41 %; a 200 k resume without "
        "the comparator dropped it to 12–25 %. The comparator is load-bearing.",
        "- The 6×–40× range is those two **cost** numbers on one network, not two shippable controllers.",
        "",
        "Plots: `outputs/v13/plots/fig8_mc_four_tasksus.png`, `fig8_mc_tableus.png`.",
        "",
    ]
    dest = V13 / "MC_FOUR.md"
    dest.write_text("\n".join(lines))
    print(f"  wrote {dest}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=24)
    ap.add_argument("--workers", type=int,
                    default=max(4, (os.cpu_count() or 8) // 4))
    ap.add_argument("--add-v13b", action="store_true")
    ap.add_argument("--add-label", default=None,
                    help="Extra policy label, e.g. SC_v13.5")
    ap.add_argument("--add-ckpt", default=None,
                    help="Checkpoint dir for --add-label")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--mc", action="store_true")
    a = ap.parse_args()
    if not (a.mc or a.plot):
        a.mc = True
        a.plot = True
    payload = None
    if a.mc:
        extra = None
        if a.add_label and a.add_ckpt:
            extra = (a.add_label, a.add_ckpt)
        payload = run_mc(a.runs, a.workers, a.add_v13b,
                         merge_path=str(V13 / "traces" / "mc_four.json")
                         if (a.add_v13b or extra) else None,
                         extra=extra)
    if a.plot:
        plot_four(payload)


if __name__ == "__main__":
    main()
