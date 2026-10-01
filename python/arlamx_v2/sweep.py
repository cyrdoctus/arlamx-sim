"""Sweep = several sessions from base + grid/arms of dotted overrides; CSV ledger."""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

from arlamx_v2 import session as S
from arlamx_v2.paths import OUTPUTS

FIELDS = ("label", "id", "status", "wall_s", "snapshot", "error")


def expand_runs(cfg):
    base = dict(cfg.get("base") or {})
    if cfg.get("arms"):
        runs = []
        for i, arm in enumerate(cfg["arms"]):
            arm = dict(arm)
            label = str(arm.pop("label", f"arm{i}"))
            runs.append((label, {**base, **arm}))
        return runs
    grid = dict(cfg.get("grid") or {})
    keys = list(grid)
    vals = [v if isinstance(v, list) else [v] for v in grid.values()]
    runs = []
    for combo in itertools.product(*vals):
        label = "_".join(f"{k.split('.')[-1]}-{v}" for k, v in zip(keys, combo)) or "base"
        runs.append((label, {**base, **dict(zip(keys, combo))}))
    return runs


def _ledger(path):
    if not path.is_file():
        return {}
    with open(path, newline="") as fh:
        return {r["label"]: r for r in csv.DictReader(fh)}


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_sweep(cfg, force=False, out_root=OUTPUTS, files=None):
    name = str(cfg.get("name", "sweep"))
    ledger = Path(out_root) / "results" / "sweeps" / f"{name}.csv"
    done = {k for k, r in _ledger(ledger).items() if r.get("status") == "done"}
    rows = []
    for label, overrides in expand_runs(cfg):
        if not force and label in done:
            print(f"[sweep] skip {label} (done in {ledger.name})", flush=True)
            continue
        settings = S.load_settings(**(files or {}))
        for k, v in overrides.items():
            S.set_path(settings, k, v)
        row = {"label": label}
        try:
            sid, snap, metrics = S.run(settings, out_root=out_root,
                                       command=f"sweep {name} arm {label}")
            row.update(id=sid, status="done", snapshot=str(snap),
                       wall_s=(metrics or {}).get("wall_s", ""))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[sweep] {label} failed: {row['error']}", flush=True)
        _append(ledger, row)
        rows.append(row)
    return rows
