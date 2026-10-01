"""Validation pass 2 — hostile numerics on the public plant path.

Drive Simulator.reset / set_atmosphere / step only. Record exception, abort,
or a quiet number. Does not redesign handling.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp


def _base():
    p = cpp.SimParams()
    p.dt_s = 2.0
    p.advisor_step_s = 10.0
    p.rk4_step_s = 2.0
    p.use_panel_srp = True
    p.corotating = True
    p.sh_degree = 0
    sim = cpp.Simulator(p)
    n = np.array([[0.0, 0.0, 1.0]])
    a = np.array([0.1])
    c = np.array([[0.0, 0.0, 0.01]])
    sim.set_panels(n, a, c)
    return sim


def run_case(name, fn):
    rec = {"name": name, "status": "ok", "detail": ""}
    try:
        rec["detail"] = str(fn())
    except Exception as exc:
        rec["status"] = "exception"
        rec["detail"] = f"{type(exc).__name__}: {exc}"
    return rec


def main():
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "validate_pass2"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def zero_r():
        sim = _base()
        sim.reset(np.zeros(3), np.array([0.0, 7500.0, 0.0]), np.zeros(3), np.zeros(3))
        return "reset accepted"

    def nan_q():
        sim = _base()
        r = np.array([6778137.0, 0.0, 0.0])
        v = np.array([0.0, 7660.0, 0.0])
        sim.reset(r, v, np.zeros(3), np.zeros(3))
        sim.set_atmosphere(1e-12, 900.0, 2.6e-26)
        o = sim.step(np.array([np.nan, 0.0, 0.0, 0.0]))
        return f"alt={o['altitude_km']} sma={o['sma_m']}"

    def neg_rho():
        sim = _base()
        sim.set_atmosphere(-3e-12, 900.0, 2.6e-26)
        return "atmosphere accepted"

    def zero_T():
        sim = _base()
        sim.set_atmosphere(1e-12, 0.0, 2.6e-26)
        return "atmosphere accepted"

    def huge_dt():
        p = cpp.SimParams()
        p.dt_s = 3600.0
        p.advisor_step_s = 3600.0
        p.rk4_step_s = 3600.0
        p.mode = "prescribed"
        sim = cpp.Simulator(p)
        sim.set_mode("prescribed")
        r = np.array([6778137.0, 0.0, 0.0])
        v = np.array([0.0, 7660.0, 0.0])
        sim.reset(r, v, np.zeros(3), np.zeros(3))
        o = sim.step(np.array([1.0, 0.0, 0.0, 0.0]))
        return f"alt={o['altitude_km']} finite={np.isfinite(o['altitude_km'])}"

    def zero_q():
        sim = _base()
        r = np.array([6778137.0, 0.0, 0.0])
        v = np.array([0.0, 7660.0, 0.0])
        sim.reset(r, v, np.zeros(3), np.zeros(3))
        o = sim.step(np.zeros(4))
        return f"alt={o['altitude_km']}"

    rows.append(run_case("reset_r=0", zero_r))
    rows.append(run_case("step_nan_q", nan_q))
    rows.append(run_case("atm_neg_rho", neg_rho))
    rows.append(run_case("atm_T=0", zero_T))
    rows.append(run_case("rk4_3600s_prescribed", huge_dt))
    rows.append(run_case("step_zero_q", zero_q))

    with open(out_dir / "hostile.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "status", "detail"])
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "hostile.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
