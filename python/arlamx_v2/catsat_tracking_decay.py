"""CatSat decay on the v2.7 plant, from the 1 October 2026 four-day state.

v2.7's own GPS file ends at 2026-09-30 03:50 UTC and has no TLEs, so this
does not refit. The epoch is the v2.6 four-day GPS/TLE state already used
for outputs/analysis/decay/catsat_fmf_20261001 there:

  2026-10-01 04:18:00 UTC
  r = [4183523, -4657480, 2420986] m
  v = [1082.628797702, -2751.93269178855, -7125] m/s

The panel shell is the same 2332-face exterior model. The plant is v2.7:
optical aluminized-Mylar SRP, Earth IR and albedo, pressure scaled inside
the step, Meeus Moon, GGM03S through degree 70, lunisolar on.
srp_scale stays 1.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arlamx_v2 import catsat_fmf_decay as decay

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "analysis" / "decay" / "catsat_fmf_20261001"
STATE_SRC = (
    Path(__file__).resolve().parents[3]
    / "ARLAMX V2.6"
    / "outputs"
    / "analysis"
    / "decay"
    / "catsat_fmf_20261001"
    / "initial_state.json"
)


def ensure_state(out_dir: Path) -> dict:
    dest = out_dir / "initial_state.json"
    if not dest.exists():
        if not STATE_SRC.exists():
            raise SystemExit(f"missing state file {STATE_SRC}")
        dest.write_text(STATE_SRC.read_text())
    return json.loads(dest.read_text())


def apply_epoch(state: dict) -> None:
    epoch = datetime.strptime(state["epoch_utc"].replace(" UTC", ""), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    decay.EPOCH = epoch
    decay.EPOCH_JD = float(state["epoch_jd"])
    decay.R0 = np.array(state["r_m"], float)
    decay.V0 = np.array(state["v_m_s"], float)
    decay.OUT = OUT


def load_panels(out_dir: Path):
    cache = np.load(out_dir / "panels_1pct.npz")
    meta = json.loads((out_dir / "geometry.json").read_text())
    return cache["n"], cache["A"], cache["c"], meta


def direction_for(name: str, meta: dict) -> np.ndarray:
    if name == "nominal":
        return np.array([0.0, -1.0, 0.0])
    if name == "best":
        return np.array(meta["best_ram"]["direction"], float)
    if name == "worst":
        return np.array(meta["worst_ram"]["direction"], float)
    raise SystemExit(f"unknown case {name}")


def run_case(name: str, out_dir: Path) -> None:
    state = ensure_state(out_dir)
    apply_epoch(state)
    n, A, c, meta = load_panels(out_dir)
    direction = direction_for(name, meta)
    print(
        f"{name} epoch {decay.EPOCH.isoformat()}  "
        f"a={state['sma_km']:.2f} km e={state['ecc']:.5f} i={state['inc_deg']:.3f}",
        flush=True,
    )
    # The v2.7 plant integrates the Meeus Moon. These arrays are unused.
    decay.run_case(name, direction, (n, A, c), None, None, out_dir)
    print(f"{name.upper()}_DONE", flush=True)


def plot_all(out_dir: Path) -> dict:
    state = ensure_state(out_dir)
    apply_epoch(state)
    summary = decay.plot_results(out_dir)
    payload = {
        "plant": "arlamx_v2.7",
        "srp": "optical al_mylar ca=0.08 cs=0.88 cd=0.04, srp_scale=1, (AU/d)^2 inside the plant",
        "earth_radiation": "Knocke IR emissivity 0.68 and albedo 0.34, IR ca=0.85 cs=0 cd=0.15",
        "moon": "Meeus analytic, no Horizons table",
        "gravity": "GGM03S degree 70, lunisolar on",
        "initial_state": {k: state[k] for k in state if k not in ("fixes", "tles")},
        "cases": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def smoke(out_dir: Path) -> None:
    state = ensure_state(out_dir)
    apply_epoch(state)
    n, A, c, meta = load_panels(out_dir)
    panels = (n, A, c)
    sim, g = decay.make_sim(panels, None, None)
    p = sim.params()
    flags = {
        "srp_optical": bool(p.srp_optical),
        "earth_rad": bool(p.earth_rad),
        "srp_scale": float(p.srp_scale),
        "srp_ca": float(p.srp_ca),
        "srp_cs": float(p.srp_cs),
        "srp_cd": float(p.srp_cd),
        "ir_ca": float(p.ir_ca),
        "sh_degree": int(p.sh_degree),
        "lunisolar": bool(p.lunisolar),
        "gravity_gradient": bool(p.gravity_gradient),
        "mass": float(p.mass),
        "mode": str(p.mode),
        "max_slew_deg": float(np.degrees(p.max_slew_rad)),
        "ggm_degree": int(g.max_degree()),
        "has_moon_table": hasattr(sim, "set_moon_ephemeris"),
    }
    print("FLAGS", json.dumps(flags), flush=True)
    if not flags["srp_optical"] or not flags["earth_rad"]:
        raise SystemExit("optical SRP or Earth radiation did not stick on the plant")
    if abs(flags["srp_scale"] - 1.0) > 1e-12 or flags["sh_degree"] != 70 or not flags["lunisolar"]:
        raise SystemExit("plant flags are not the decay study settings")
    if flags["ggm_degree"] != 70:
        raise SystemExit(f"GGM loaded at degree {flags['ggm_degree']}")
    if flags["has_moon_table"]:
        raise SystemExit("v2.7 plant still exposes a Moon table")
    sim.reset(decay.R0, decay.V0, np.zeros(3), np.zeros(3))
    lat, lon, alt = decay.geodetic(decay.R0, decay.EPOCH_JD)
    f107, f107a, ap = decay.weather(decay.EPOCH_JD)
    rho, T, mbar = decay._msis(alt / 1e3, np.degrees(lat), np.degrees(lon), decay.EPOCH_JD, f107, f107a, ap)
    sim.set_atmosphere(rho, T, mbar)
    sim.set_srp_scale(1.0)
    v_rel = decay.V0 - np.cross([0.0, 0.0, decay.OMEGA_EARTH], decay.R0)
    q = decay.quat_ram(v_rel, np.array([0.0, -1.0, 0.0]), np.cross(decay.R0, decay.V0))
    out = sim.step(q)
    r = np.asarray(out["r"], float)
    v = np.asarray(out["v"], float)
    jd = decay.EPOCH_JD + float(out["t"]) / 86400.0
    fr = decay.sample_forces(g, panels, None, None, r, v, jd, q, rho, T, mbar, 1.0)
    forces = {k: float(np.linalg.norm(fr[k])) for k in ("F_fmf", "F_srp", "F_erp", "F_harm", "F_moon", "F_sun")}
    print(
        f"SMOKE alt {alt/1e3:.2f} km rho {rho:.3e} eclipse {fr['eclipse']:.0f} "
        + " ".join(f"{k} {forces[k]*1e6:.3f} uN" for k in forces),
        flush=True,
    )
    if not all(np.isfinite(list(forces.values()) + [float(out["altitude_km"])])):
        raise SystemExit("non-finite smoke step")
    if not (1e-6 < forces["F_fmf"] < 1e-3):
        raise SystemExit(f"drag {forces['F_fmf']} N is outside 1 uN..1 mN")
    if not (1e-8 < forces["F_erp"] < 1e-4):
        raise SystemExit(f"Earth radiation {forces['F_erp']} N is outside 0.01 uN..100 uN")
    print("SMOKE_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("nominal", "best", "worst"))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke(out)
        return
    if args.plot_only:
        print(json.dumps(plot_all(out), indent=2), flush=True)
        return
    if args.case:
        run_case(args.case, out)
        return
    for name in ("worst", "nominal", "best"):
        run_case(name, out)
    print(json.dumps(plot_all(out), indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
