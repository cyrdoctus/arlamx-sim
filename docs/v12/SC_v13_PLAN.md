# Overnight / tomorrow — get closer to MPC, then maybe name it SC_v13

**Do not rename until actuation improves without a decay regression.**
Runs stay under `outputs/v12/runs/s3_*`. Plant, `env.py`, `mpc_v2.yaml` stay
frozen. Wrapper + configs + new runners only.

## What is already true

- Best SC_v12: s43 ckpt **step_300096**. Downlink matches column A (23.5 vs 23.2).
  Band is close (81.5 vs 86.6). Decay is the hole (15.6 vs 9.3). B is negative.
- Compute is already a win: 0.17 ms policy / 1.00 ms +comparator vs MPC 6.81 ms
  on the U575 analytic budget.
- Comparator cut slews 288 → ~99 / day. Keep it.
- History is already stacked 150–900 s. No LSTM.

## What s3 changes (in the wrapper)

1. **Wrong-way clip.** If τ_env_i · τ_want_i < 0 and s_i > 0.05 → s_i = 0.
   Small `w_clip = 0.08` so PPO unlearns proposing those splits.
2. **Power/geometry τ_env** for r_env and the clip (not a new obs dim).
3. **w_env 0.10 → 0.15.** Actuation signal, not a longevity hammer.
4. Warm-start from the s2 best ckpts (s42 @ 250k, s43 @ 300k, s44 from s43).
5. 800 k additional steps × 3 seeds, ckpt every 50 k, same dual min-score.
6. Every ckpt line prints **infer_ws_ms, infer_total_ms, B, clip_frac, kf_r, pwr_r**.

## What we are *not* doing tonight

- Raising decay weights, brownout weights, or dE_weight. Longevity is not the
  knob. Do not make decay worse to buy B.
- Obs-dim change / LSTM / 50–600 s retune. Needs env.py; that is a real SC_v13
  *after* s3 shows the clip works.
- Editing `gains_mrp.yaml` gate_floor, kp, kd.
- SAC / TD3 / damage / v11.
- Calling anything SC_v13 in filenames.

## Promote to SC_v13 only if

On a 10-orbit quiet eval versus the s43@300k numbers:

| | now (s43@300k) | promote if |
|---|---|---|
| deleg_B | −0.27 | ≥ 0, or ≥ −0.05 with clip_frac < 0.15 |
| decay | 15.61 km/d | ≤ 16.5 (do not give away more than ~1 km/d) |
| band | 81.5 % | ≥ 75 % |
| downlink | 23.5 min/d | ≥ 20 |
| brownouts | 0 | 0 |
| infer_total_ms | ~0.15 + cmp | still ≪ 6.81 ms U575 |

If B flips sign and decay holds, freeze that ckpt, write `SC_v13.md`, copy
weights to `outputs/v13/`. If decay jumps, keep the name SC_v12 and drop w_env
back to 0.10.

## Tomorrow morning

1. Read `outputs/v12/BEST_SO_FAR.md` and each `s3_*/eval_curve.csv`. Pick by
   **checkpoint**, not final step (non-monotone, same as v10/s2).
2. Re-run `python -m arlamx_v2.plot_v12 --all` on the new best if it wins.
3. Optional modest MC (32 runs, 28-day cap) for the days-in-orbit bar.
4. Two-slide deck: architecture (`fig0` / SVG) + `fig1_scorecard`.

## Later SC_v13 (needs env.py — not tonight)

- Pack `τ_env^P` and `|F̂|` into the 4-feature history slots (still not LSTM).
- Power-limited slew / disturbance reserve.
- Train-on-truth vs train-on-sensors as two columns, never mixed.
