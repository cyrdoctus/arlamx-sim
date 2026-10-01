# Note — future auto-optimizer (ARLAMX V2.5 / V3, not V2.0)

**Status:** design note only. Nothing here is wired into SC_v14 training.
V2.0 keeps **hand-listed one-knob arms** (`docs/v12/SC_v14_SWEEP.md`).
This file is the contract so a later version can add a runner without
rewriting the plant.

## Why it is not V2.0

The plant, `env.py`, and the 4×18 advisor must stay freeze-able for the
thesis comparison vs sampling MPC. An 8B–27B loop that rewrites rewards
every night would mix columns and burn the dual-eval meaning. Train SC_v14
by listed sweeps until MC band / brownouts move.

## What it is

A **version-agnostic** Python job (not a new Gym env):

1. Load the **most recent** freeze dir (`outputs/v13/ckpt_*` with a `PICK.json`
   or `KEEP.md`) and the latest `eval_curve.csv`. No hardcoded SC_v14 names
   in the runner — the YAML points at paths.
2. Read a sweep file (`python/configs/sweep_<name>.yaml`): resume path,
   steps, seeds, dual_gate, list of arms (knob → value).
3. For each arm: call the existing `train_v12.train_run(...)` (wrapper
   kwargs only). Do not edit `env.py` / C++ / `mpc_v2.yaml`.
4. Append one row per checkpoint to `outputs/v12/SWEEP_LEDGER.csv`
   (quiet band/dl/B/decay, storm band/dl/B/brownouts, dual_ok, wall_s).
5. Optional **local** LLM (8B, at most 27B, on-box): given the ledger +
   the freeze numbers, emit the **next YAML**. It does not train, does not
   fly, does not touch weights. User sets `--loops N`.
6. Stop if dual_ok and quiet regression holds, or if the step budget is
   gone. Human still runs the 24-draw MC before a freeze rename.

The code path must stay “load latest files + YAML”. New SolarCat versions
drop in a new ckpt dir and a new sweep YAML; the runner does not fork.

## Two timescales (do not mix)

| where | what | compute |
|---|---|---|
| **Train machine** (V2.5) | YAML arms, dual-eval, ledger, optional 8B proposes next YAML | GPU/CPU box, minutes–hours |
| **Onboard U575** (V3) | **Table**, not an LLM: η / SoC / band-dwell → (kd, gate_floor, κ, margin, maybe cmp-on) | few dozen FLOPs, already the v14 REGIME box |

The satellite does **not** run 8B or PPO updates. It interpolates a table
filled on the ground (or slowly updated from logged η, coil J, SoC). That
is the “save values into a big table, infer any time in orbit, minimise
power” path. REGIME in SC_v14 is the first slice of that table.

## What the 8B is allowed to change

Only keys the wrapper already understands: `w_env`, `w_low`, `w_band_storm`,
`w_clip`, `w_slew`, `ppo_kw.learning_rate`, arm `steps`, `seed`.
Never: decay YAML, `gs_weight`, obs dim, dropping the comparator, plant.

## V2.0 action

Use `docs/v12/SC_v14_SWEEP.md` and `train_v12 --round v14d`. Do not implement
this runner until V2.5.
