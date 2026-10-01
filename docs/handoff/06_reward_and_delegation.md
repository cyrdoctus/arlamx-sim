# 06 — Reward families and the delegation mechanism

Equations are numbered as in `docs/modules/14_reward_v8.md` and
`python/arlamx_v2/reward_v8.py`. Weights live in `python/configs/reward_*.yaml`
with provenance tags [SPEC] [V7] [DERIVED] [TUNED] [ASSUME]; every trained run
froze its resolved config in `<run>/snapshot.json`.

## SC_v3 core (`reward.py`, module 12)

    x = clip((500 − h)/200, 0, 1),   w_long = (e^{3x} − 1)/(e^3 − 1)
    r_ΔE = w_long · clip((ΔE_a − ΔE_b)/max(|ΔE_b|, ε), −1, 1)       (weight 0.75)
    power: m(s) = 0.8 (s ≥ 0.6) else 2.0;  r = m(s)[min(s,0.8) + 0.5 g]
    gs_alignment, shade 0.2, smoothness 0.02, ω 0.002, momentum 0.001, altitude cliff 1000 at 250 km

## v7 composite (`compose_sc_v7`, inherited by v8+)

Terms and v11 weights: `dE_vs_baseline` 2.6 (ΔE_actual vs the min-drag
counterfactual), `power_band` 1.0 (+1 in SoC 0.4–0.6, exp(−6(s−0.6)) above,
−4(0.4−s)/0.4 below, depletion −50), `gs_tiered_pointing` 2.4 (tiers 2°/5°/10°
→ 1/0.5/0.25, penalty 0.5, daylight only), `action_feasibility` 1.0,
`action_smoothness` 0.10, `torque_thrift` 1.5 (linear duty), `omega` 0.002,
`momentum` 0.001, `brownout_recovery` 100.

## v8 stability

    r_margin = −w_m max(0, (h_soft − h)/(h_soft − h_floor))²    w_m 4.0, 320 km / 250 km    (9)
    r_dstab  = −w_s min(max(0, dE_prev − dE)/scale, cap)        w_s 0.6, scale 100 J/kg, cap 3 (10)
    v10 adds power_drop 2.0 (thresh 0.02 SoC/step, scale 0.05, cap 3)

## Delegation (v8b, v9, v10–v14)

Action = [q_target (4), split s (3)], s_i ∈ [0,1].

    gate_i = 1 − s_i (1 − gate_floor),  gate_floor 0.3                             (1)
    A_i = sign(τ_env,i τ_want,i) · min(1, |τ_env,i| / τ_env_ref),  τ_env_ref 1.5e-8 N·m, ε 1e-11   (2)
    B   = Σ s_i A_i / max(Σ s_i, ε)  ∈ [−1, 1]                                    (3)
    E_full,i = min(|τ_want,i|, τ_max,i)/τ_max,i
    E_gate,i = min(|τ_want,i|, gate_i τ_max,i)/τ_max,i
    P   = Σ (E_full,i − E_gate,i) / max(Σ E_full,i, ε)  ∈ [0, 1]                  (4)
    S   = clip((P − p_min)/(p_sig − p_min), 0, 1),  p_min 0.05, p_sig 0.25         (5)
    r_deleg_power = w_power · P · clip(B, 0, 1),   w_power 6.0                     (6)
    r_deleg_acc   = w_accuracy · B,                w_accuracy 2.0 (v8b) → 6.0 (v10+) (7)
    boost = 1 + κ S clip(B,0,1) on dE_vs_baseline, power_band, gs_tiered_pointing, κ 0.5 → 0.7 (8)

τ_env is the Kalman-filtered disturbance torque estimate (`tau_env_est`).
τ_want is, since 2026-09-13, the plant's `tau_demand_mean` / `tau_demand_absmean`
(PD demand *before* the per-axis clip that carries the gate). Eq. (2) grew out
of grok blocker B3 (the earlier |τ_env|/|τ_want| ratio was ~1e-3 during slews).
Eq. (4) is now command-side (grok B2 fixed the "P = 1 − gate" draft; the 2026-09-13
fix removed the measured |τ_ctrl| so B-projection losses are not credited to
the gate).

**The bug that shaped the whole v8–v14 story.** From v2.5 (and, via the
Python replay, plausibly earlier) `tau_want` was the *gated* command, which
can never exceed gate·τ_max; the "binding" condition in (4) never fired and
`deleg_power ≡ 0` in every run. Grok's B1 (2026-08-20) had already found
`deleg_power ≡ 0` once for a different wiring reason; the burn-in gate that
was added to catch it printed `deleg_P` from the same broken input.
Consequence: every "delegation costs mission gap" verdict below was measured
with only eq. (7) (signed B) live. The v8a/v8b ablation has not yet been run
with (6) active.

## v9 trend adaptation (never isolated)

    w_dE   ← w_dE (1 + 0.8 clip(−d_alt_norm, 0, 1)),  d_alt_norm = (h(t+600) − h(t−600))/2 km   (11)
    w_band ← w_band (1 + 1.0 clip(−d_soc/0.2, 0, 1))                                          (12)

## v11 adaptability

`adapt_consistency` 4.0 (penalise tracking error above the pre-strike rolling
median, scale 0.10 rad, cap 3) and `adapt_deleg_hold` 4.0 (keep B while
damaged). Damage: 60 % of episodes, strike at 15–70 %, membrane panel areas
scaled (`damage.py`).

## v12–v14 wrapper terms (`train_v12.py::V12Wrapper`)

r_slew = −w_slew · 1[command change > 2°], w_slew 0.5 (swept 0.25/0.5/1.0);
r_env = w_env (τ̂_env · τ_want), 0.10 → 0.15 → 0.35 (v14); r_clip = −w_clip
n_clip, 0.08; comparator: score σ_prev vs σ_new with `SamplingMpcPolicy._score`
over 2 steps, keep previous unless new wins by 0.05; wrong-way clip: s_i = 0 if
τ_env,i τ_want,i < 0 (soft κ 0.70 quiet → 1.0 storm in v14); v14c–f knobs
w_band_storm, w_low (SoC < 0.25/0.15), w_gen (reward charging below SoC 0.20).

## Tuning verdicts (`docs/v12/00_LESSONS_v7_to_v11.md`, 188 runs + 52 tuning rows)

| term | verdict | evidence |
|---|---|---|
| dE_vs_baseline 2.6 | **HIT, the only decisive knob** | r0: 2.6 → gap 0.92; 3.2 → 1.66; 4.0 → 1.36 |
| power_band 1.0 | target missed | median band 3–4 % in every family; 2.0 → brownouts |
| gs_tiered_pointing 2.0–2.8 | neutral | downlink 10–14 min/d vs MPC 23 |
| smoothness, thrift | wasteful above defaults | 0.2 + thrift 2.0 → band collapse, 15 brownouts |
| brownout 100, altitude cliff | HIT as constraints | brownouts 0 in converged runs |
| deleg_power 6.0 | "wasteful" — **but it was never live** | see bug above |
| deleg_accuracy 2 → 6 | harmful | max-B policies have the worst mission metrics |
| κ boost | dead | B⁺ ≈ 0 on median |
| KF torque estimate | HIT as an observer, not a reward lever | r 0.99 with datasheet noise |
| v9 temporal blocks | HIT best-case, costs variance | v9d best 0.387, median 1.64 |
| adapt_consistency 4.0 | HIT for training stability | 11 → 0 brownouts |
| adapt_deleg_hold 4.0 | wasteful for the mission | post-strike decay worse than v10 |
| comparator (v12) | HIT | slews 288 → 99/day; removing it drops band 80 → 41 % |
