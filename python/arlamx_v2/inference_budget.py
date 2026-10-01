"""Onboard inference cost on the STM32U575 — analytic budget, not wishcasting.

The workstation timings mean nothing for flight: a Threadripper core is ~4
orders of magnitude faster than the flight computer. This module counts the
FLOPs each advisor actually executes per 300 s decision and converts them with
a *stated, conservative* throughput model of the target part:

    STM32U575 (Cortex-M33 @ 160 MHz, single-precision FPU)
    - FP32 multiply-accumulate: ~1 FLOP/cycle peak on the M33 FPU, derated to
      SUSTAINED_FLOPS_PER_CYCLE for load/store, loop overhead, and cache-less
      flash access. 0.35 FLOP/cycle is the conservative planning number
      (CMSIS-DSP class hand loops land 0.3-0.6).       [ASSUME - bench on HW]
    - exp/log/sqrt/trig through the FPU + polynomial libs: ~40 cycles each.

Everything else is counting:
    MLP forward:   2 * n_params FLOPs (+ tanh per hidden unit, TRANSCEND each)
    KF step:       fixed 6-state predict/update, counted below
    FP32 RK4 step: 4 accel evals * ~120 FLOPs (two-body + J2 + exp-drag)
    eclipse test:  ~25 FLOPs per sample

Outputs a table for the report: FLOPs, U575 milliseconds, and the duty fraction
of the 300 s decision period. Desktop-measured times are reported next to it
for reference, never as the flight number.

    PYTHONPATH=python python -m arlamx_v2.inference_budget
Level: advanced.
"""
from __future__ import annotations

import json

import numpy as np

from arlamx_v2.paths import OUTPUTS_OLD as OUTPUTS  # pre-v2.7 artifacts

V8 = OUTPUTS / "v8"

CLOCK_HZ = 160e6
SUSTAINED_FLOPS_PER_CYCLE = 0.35        # [ASSUME] derated M33 FPU
TRANSCEND_CYCLES = 40                    # exp/tanh/sqrt/sin class

FLOPS_PER_SEC = CLOCK_HZ * SUSTAINED_FLOPS_PER_CYCLE


def mlp_cost(obs_dim, arch, act_dim):
    """FLOPs and transcendentals for one deterministic forward pass."""
    dims = [obs_dim] + list(arch) + [act_dim]
    macs = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    flops = 2 * macs + sum(arch)             # +bias adds folded into 2*macs; tanh separate
    trans = sum(arch) + act_dim              # tanh on hidden + squash on output
    return flops, trans


def rk4_cost(n_steps):
    """FP32 propagator: 4 accel evals/step, ~120 FLOPs each + 2 transcendentals
    (exp for density, sqrt for radius) per eval; plus state combine ~60."""
    flops = n_steps * (4 * 120 + 60)
    trans = n_steps * 4 * 2
    return flops, trans


def kf_cost():
    """6-state CV filter, dense 6x6 ops per step (predict+update+Joseph)."""
    return 6 * 6 * 6 * 8, 3        # ~1.7 kFLOP, 3 sqrt


def to_ms(flops, trans):
    cycles = flops / SUSTAINED_FLOPS_PER_CYCLE + trans * TRANSCEND_CYCLES
    return cycles / CLOCK_HZ * 1e3


def policy_entry(name, obs_dim, arch, act_dim=7, extra=()):
    f, t = mlp_cost(obs_dim, arch, act_dim)
    f += 1500          # observation assembly / normalisation
    kf_f, kf_t = kf_cost()
    f += kf_f
    t += kf_t
    parts = {"policy_mlp+obs+KF": (f, t)}
    for label, (ef, et) in extra:
        parts[label] = (ef, et)
        f += ef
        t += et
    return {"name": name, "flops": f, "transcend": t, "u575_ms": to_ms(f, t),
            "duty_pct_of_300s": to_ms(f, t) / 300e3 * 100,
            "parts_ms": {k: to_ms(*v) for k, v in parts.items()}}


def v9_future_cost(n_offsets, horizon_s=900.0, dt=30.0):
    """One sweep to the farthest offset + eclipse/soc per sample."""
    f, t = rk4_cost(int(horizon_s / dt))
    f += n_offsets * 60
    t += n_offsets * 1
    return f, t


def duo_extra(horizon_s=21600.0, dt=60.0, arb_s=600.0, arb_dt=20.0,
              hor_obs=18, hor_arch=(20, 20, 20, 20)):
    """Everything the Duo adds on top of its main advisor."""
    prop_f, prop_t = rk4_cost(int(horizon_s / dt))          # 6 h sweep
    ecl_f = int(horizon_s / (dt * 5)) * 25                  # eclipse samples
    hor_f, hor_t = mlp_cost(hor_obs, list(hor_arch), 4)
    arb_f, arb_t = rk4_cost(int(arb_s / arb_dt) * 2)        # both candidates
    bc_f = 2 * 500                                          # projected-area sums
    return [("6h propagation + eclipse", (prop_f + ecl_f, prop_t)),
            ("horizon advisor MLP", (hor_f, hor_t)),
            ("arbiter 2x600s + bc", (arb_f + bc_f, arb_t))]


def mpc_entry():
    """Sampling MPC for contrast: N candidate attitudes, each propagated with
    the same class of model plus a Sentman coefficient sweep (72 panels).

    This is the 24-candidate × 20-step stand-in that produced the published
    8.4 ms number. For a live (H, n_random) config use `mpc_cost`.
    """
    n_cand, n_steps = 24, 20
    sent_f = n_cand * 72 * 60
    sent_t = n_cand * 72 * 2                   # erf/exp per panel
    prop_f, prop_t = rk4_cost(n_cand * n_steps)
    f, t = sent_f + prop_f + 4000, sent_t + prop_t
    return {"name": "Sampling MPC (24 cand)", "flops": f, "transcend": t,
            "u575_ms": to_ms(f, t), "duty_pct_of_300s": to_ms(f, t) / 300e3 * 100,
            "parts_ms": {"sentman sweep": to_ms(sent_f, sent_t),
                         "candidate propagation": to_ms(prop_f, prop_t)}}


def mpc_cost(horizon_steps=6, n_random=8, n_panels=72, advisor_step_s=300.0,
             n_named=5):
    """FLOP model of SamplingMpcPolicy as it actually runs.

    Per candidate: `horizon_steps` two-body RK4 intervals of 60 s, and one
    Sentman coefficient sweep per horizon step (attitude held, v_gas changes).
    Named set is hold/sun/gs/min_drag/nadir (gs is sometimes absent; budget
    uses the worst case).
    """
    n_cand = int(n_named) + int(n_random)
    n_int = max(1, int(round(float(advisor_step_s) / 60.0)))
    n_prop = n_cand * int(horizon_steps) * n_int
    n_sent = n_cand * int(horizon_steps)
    sent_f = n_sent * int(n_panels) * 60
    sent_t = n_sent * int(n_panels) * 2
    prop_f, prop_t = rk4_cost(n_prop)
    f, t = sent_f + prop_f + 4000, sent_t + prop_t
    return {"name": f"MPC H={horizon_steps} R={n_random}",
            "flops": f, "transcend": t, "u575_ms": to_ms(f, t),
            "duty_pct_of_300s": to_ms(f, t) / 300e3 * 100,
            "n_cand": n_cand, "n_prop": n_prop, "n_sent": n_sent,
            "parts_ms": {"sentman sweep": to_ms(sent_f, sent_t),
                         "candidate propagation": to_ms(prop_f, prop_t)}}


def mpc_u575_ms(horizon_steps=6, n_random=8, n_named=5, advisor_step_s=300.0,
                n_panels=72, scale="published"):
    """Milliseconds on the U575 planning model.

    `published` linearly scales the 24×20 stand-in that produced 8.4 ms, so
    the 15 ms freeze gate is in the same units as the thesis number.
    `actual` counts the live algorithm (Sentman every horizon step).
    """
    if scale == "actual":
        return float(mpc_cost(horizon_steps, n_random, n_panels,
                              advisor_step_s, n_named)["u575_ms"])
    n_cand = int(n_named) + int(n_random)
    n_int = max(1, int(round(float(advisor_step_s) / 60.0)))
    n_steps = int(horizon_steps) * n_int
    base = mpc_entry()["u575_ms"]
    return float(base * (n_cand / 24.0) * (n_steps / 20.0))


def main():
    rows = [
        policy_entry("v8a/v8b 4x18", 35, [18] * 4),
        policy_entry("v8b 4x32", 35, [32] * 4),
        policy_entry("v9a 4x24 (+3 future)", 47, [24] * 4,
                     extra=[("future propagation", v9_future_cost(3, 600.0))]),
        policy_entry("v9b 4x32 (+3 past/3 future)", 59, [32] * 4,
                     extra=[("future propagation", v9_future_cost(3, 600.0))]),
        policy_entry("v9c 4x32 (6 past/3 future)", 71, [32] * 4,
                     extra=[("future propagation", v9_future_cost(3, 600.0))]),
        policy_entry("v9d 4x32 (6 past/6 future)", 83, [32] * 4,
                     extra=[("future propagation", v9_future_cost(6, 900.0))]),
        policy_entry("v10 4x40 (v9-class obs)", 83, [40] * 4,
                     extra=[("future propagation", v9_future_cost(6, 900.0))]),
        policy_entry("v9Duo (v9b 4x32 main)", 59, [32] * 4,
                     extra=[("future propagation", v9_future_cost(3, 600.0))]
                     + duo_extra()),
        mpc_entry(),
    ]
    print(f"{'advisor':34s} {'kFLOP':>8s} {'U575 ms':>9s} {'% of 300 s':>11s}")
    print("-" * 68)
    for r in rows:
        print(f"{r['name']:34s} {r['flops']/1e3:8.1f} {r['u575_ms']:9.2f} "
              f"{r['duty_pct_of_300s']:10.4f}%")
    out = V8 / "data" / "inference_u575.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": {"part": "STM32U575", "clock_hz": CLOCK_HZ,
                  "sustained_flops_per_cycle": SUSTAINED_FLOPS_PER_CYCLE,
                  "transcend_cycles": TRANSCEND_CYCLES,
                  "note": "conservative planning numbers; bench on hardware"},
        "rows": rows}, indent=2))
    print(f"\n[inference] wrote {out}")


if __name__ == "__main__":
    main()
