# MPC_tuning.md — 3-hour sprint to freeze `MPC_v2` before SC_v12 training

Runner: `PYTHONPATH=python python -m arlamx_v2.tune_mpc --stage all --workers 20`

Artifacts: `outputs/v8/mpc_tune/` (ledger, per-config json, freeze.json),
`python/configs/mpc_v2.yaml` (written at freeze), `outputs/v8/data/ref_v1.json`
(copy of the historical reference), `ref_v2.json`, `mc_decay_v2.json`.

## Pre-sweep notes (applied)

1. **Sensor noise.** The MPC surrogate now reads GNSS (r, v), dual
   magnetometer (B, averaged unless they disagree by > 5 µT), and SoC with
   0.5 % [ASSUME] white noise. Sun stays ephemeris. MSIS and GS visibility
   are recomputed from the noisy fix. v1 freeze numbers stay truth-state;
   the A–F score anchor is noisy v1 on the 10-orbit protocol so knob
   comparisons are not confounded with the protocol change.
2. **Dipole quantisation.** The C++ plant saturates per-axis torque; there
   is no 8-bit DAC. **Skipped** — recorded in `mpc_v1.yaml` /
   `mpc_v2.yaml` as `dipole_quant_bits: null`. Does not change any number.
3. **Episode terminator.** Sweep stays 10-orbit at 400 km (will not hit
   300 km). Freeze MC uses `--max-steps 20000` (~69 d) so a 5.4 km/d
   min-drag hold still reaches 300 km. Historical `mc_decay.json` is not
   overwritten (`mc_decay_v2.json`).
4. **New metrics.** `slews_per_day` (command change > 2°), `mtq_energy_J_d`,
   `PEI = E_mtq / E_PD-demand` (model-based τ_env-zeroed counterfactual:
   quadratic coil power of `tau_want` vs applied `tau_ctrl`). Logged on
   every v8 probe rollout.

## Sweep

Probe: 3 seeds × 10-orbit at 400 km, i = 23°, F10.7 150/Ap 4 **plus** a
constant-storm probe (F10.7 220, Ap 120). Score =
`min_i(oriented ratio vs noisy v1)` over {decay, band, downlink, slews/day};
any brownout disqualifies. U575 gate: published 24×20 stand-in scaled to
`(n_named+n_random, H×5 RK4 steps)`, budget **≤ 15 ms**.

| Round | Knob | Values |
|---|---|---|
| A | `horizon_steps` | 3, 6, 9, 12 |
| B | `n_random` | 8, 16, 32 (at A winner H) |
| C | `cd` × `slew` | {0.3, 0.5, 0.8} × {0.2, 0.5, 1.0} |
| D | `gs` × `gs_align` | {2, 3, 4} × {1, 2} |
| E | `power_low_pen` × `soc_k` | {4, 8} × {6, 10} |
| F | A–E combo ± 1 grid step | — |

C–E run in one pool from the A/B backbone. Freeze candidate = best score
**inside the 15 ms budget**.

## Freeze

1. `mpc_v2.yaml` with `[TUNE-x]` provenance. Nothing in the MPC changes after
   this file exists.
2. 20-orbit 3-probe suite (same protocol as historical `ref.json`) →
   `ref_v2.json`; `ref.json` updated with `MPC` = v2 and `MPC_v1` nested.
3. Monte Carlo 512 × 500→295 km, 69-day ceiling → `mc_decay_v2.json`.
