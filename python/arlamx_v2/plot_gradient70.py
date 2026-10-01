"""Production gradient figures from the 70x70 sweeps (grid70_{sw,alt,ecc}.npz).

One small-multiple figure per (panel, quantity): 2x3 tiles in the fixed policy
order (MPC, heuristic, min-drag / RL PPO, SAC, TD3), shared color scale,
perceptually-uniform ramps (magma for rates, cividis for power, viridis for
Cd, plasma for downlink), the trained envelope drawn as a dashed box, brownout
cells ringed in red on the power maps. PDF (vector) + 300 dpi PNG.

Usage:  PYTHONPATH=python python -m arlamx_v2.plot_gradient70 [--test]
Outputs -> outputs/analysis/gradient70/g70_{panel}_{quantity}.{pdf,png}
Level: advanced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / ".old" / "analysis" / "gradient70"
TEST = "--test" in sys.argv

# Fixed policy identity (order + validated categorical palette).
POLICY_META = [
    ("mpc", "MPC", "#2a78d6"),
    ("heuristic", "Heuristic", "#eb6834"),
    ("min_drag", "Min-drag", "#008300"),
    ("rl_ppo", "RL PPO (IPC)", "#1baf7a"),
    ("rl_sac", "RL SAC (IPC)", "#eda100"),
    ("rl_td3", "RL TD3 (IPC)", "#e87ba4"),
]

PANEL_META = {
    "sw":  dict(xlabel="F10.7 (sfu)", ylabel="Ap index",
                box=(65.0, 250.0, 2.0, 40.0), nom=(150.0, 4.0),
                title="space weather (F10.7 × Ap) at 400 km — trained region dashed"),
    "alt": dict(xlabel="inclination (deg)", ylabel="altitude (km)",
                box=(20.0, 30.0, 300.0, 500.0), nom=(23.0, 400.0),
                title="orbit (inclination × altitude) — trained region dashed"),
    "ecc": dict(xlabel="inclination (deg)", ylabel="eccentricity",
                box=(20.0, 30.0, 0.001, 0.01), nom=(23.0, 0.001),
                title="orbit (inclination × eccentricity) at 500 km — trained region dashed"),
}
QUANT_META = {
    "decay":      dict(label="SMA decay rate (km/day)", cmap="magma", log=True),
    "soc_mean":   dict(label="mean supercap SoC (%)", cmap="cividis", log=False,
                       scale=100.0, brownout=True),
    "lifetime_d": dict(label="lifetime to 250 km (days, capped 3650)",
                       cmap="magma", log=True),
    "downlink":   dict(label="downlink duty cycle (%)", cmap="plasma", log=False,
                       scale=100.0),
    "cd":         dict(label="effective drag coefficient", cmap="viridis", log=False),
}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 8,
    "font.family": "DejaVu Sans", "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
})


def figure(panel, quant, d):
    pm, qm = PANEL_META[panel], QUANT_META[quant]
    slugs = [str(s) for s in d["SLUGS"]]
    sc = qm.get("scale", 1.0)
    Z = {s: d[f"{s}__{quant}"] * sc for s, _l, _c in POLICY_META if s in slugs}
    if not Z:
        return
    vals = np.concatenate([z[np.isfinite(z)] for z in Z.values()])
    pos = vals[vals > 0]
    vmin = (max(pos.min(), 1e-3) if (qm["log"] and len(pos)) else vals.min())
    vmax = max(vals.max(), vmin * 1.0001)
    norm = LogNorm(vmin=vmin, vmax=vmax) if qm["log"] else None
    X, Y = d["X"], d["Y"]
    ext = [X[0], X[-1], Y[0], Y[-1]]

    live = [(s, l) for s, l, _c in POLICY_META if s in Z]
    ncol = 3
    nrow = int(np.ceil(len(live) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow + 0.9),
                             squeeze=False, sharex=True, sharey=True)
    im = None
    for k, (slug, label) in enumerate(live):
        ax = axes[k // ncol][k % ncol]
        z = Z[slug]
        im = ax.imshow(np.where(z > 0, z, np.nan) if qm["log"] else z,
                       origin="lower", aspect="auto", extent=ext,
                       cmap=qm["cmap"], norm=norm,
                       vmin=None if qm["log"] else vmin,
                       vmax=None if qm["log"] else vmax,
                       interpolation="bilinear")
        if qm["log"] and np.any(z <= 0):
            # non-positive cells: decay below the measurement resolution of a
            # 3-orbit episode (residual J2 wobble) — decay is never physically
            # negative here (verified: SRP is a net brake on this sail).
            hold = np.where(z <= 0, 1.0, np.nan)
            ax.imshow(hold, origin="lower", aspect="auto", extent=ext,
                      cmap=matplotlib.colors.ListedColormap(["#bfe0f7"]),
                      vmin=0, vmax=1, interpolation="nearest")
        x0, x1, y0, y1 = pm["box"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="white", lw=0.9, ls="--"))
        ax.plot(*pm["nom"], "o", ms=3.5, mfc="white", mec="black", mew=0.6)
        if qm.get("brownout"):
            bo = d[f"{slug}__soc_min"]
            if np.nanmin(bo) < 0.10 < np.nanmax(bo):
                ax.contour(X, Y, bo, levels=[0.10], colors="#e34948",
                           linewidths=0.9)
        ax.set_title(label, fontsize=9)
        if k // ncol == nrow - 1:
            ax.set_xlabel(pm["xlabel"])
        if k % ncol == 0:
            ax.set_ylabel(pm["ylabel"])
        ax.tick_params(labelsize=7)
    for k in range(len(live), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.subplots_adjust(hspace=0.28, wspace=0.12, top=0.86, bottom=0.16)
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label(qm["label"], fontsize=8)
    cb.ax.tick_params(labelsize=7)

    handles = [Line2D([0], [0], ls="--", color="0.35", lw=1.0,
                      label="deep-RL training envelope"),
               Line2D([0], [0], marker="o", ls="", mfc="white", mec="black",
                      label="nominal mission point")]
    if qm.get("brownout"):
        handles.append(Line2D([0], [0], ls="-", color="#e34948", lw=1.0,
                              label="brown-out region (min SoC < 10%)"))
    if qm["log"] and any(np.any(z <= 0) for z in Z.values()):
        handles.append(Line2D([0], [0], marker="s", ls="", mfc="#bfe0f7",
                              mec="0.6", ms=8,
                              label="below measurement resolution (≈ 0 km/day)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.46, 0.02))
    fig.suptitle(f"{qm['label']} — {pm['title']}", y=0.95, fontsize=10.5)

    name = f"g70_{panel}_{quant}" + ("_test" if TEST else "")
    for ext_ in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext_}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png  [{vmin:.3g} … {vmax:.3g}]")


def main():
    for panel in ("sw", "alt", "ecc"):
        f = OUT / f"grid70_{panel}{'_test' if TEST else ''}.npz"
        if not f.exists():
            print(f"  [skip] {f.name} missing")
            continue
        d = np.load(f, allow_pickle=False)
        for quant in QUANT_META:
            figure(panel, quant, d)
    print("[done]")


if __name__ == "__main__":
    main()
