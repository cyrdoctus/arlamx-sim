"""Bar plot of PPO burn-in board (short eval and robust eval if present)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = ROOT / "outputs" / "tuning" / "tune_ppo"
    board = json.loads((out / "board.json").read_text())
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    xs = [r["i"] for r in board]
    ys = [r["return_mean"] for r in board]
    ax.bar(xs, ys, color="#4c78a8", label="16-step eval")
    best_p = out / "best.json"
    if best_p.exists():
        data = json.loads(best_p.read_text())
        rb = data.get("robust_board")
        if rb:
            rxs = [r["i"] for r in rb]
            rys = [r["return_mean"] for r in rb]
            order = np.argsort(rxs)
            ax.plot(np.array(rxs)[order], np.array(rys)[order], "o-", color="#f58518", label="64-step robust")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("trial")
    ax.set_ylabel("eval return")
    ax.set_title("PPO burn-in on SC_v4a (100k steps)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    png = out / "tune_board.png"
    fig.savefig(png, dpi=130, facecolor="white")
    plt.close(fig)
    print("wrote", png)


if __name__ == "__main__":
    main()
