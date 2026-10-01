# 08 — Campaign results: every headline with its caveat

All RL numbers below were produced with `ideal_torque=true` (ideal body
couple) and, for v8+, with `deleg_power ≡ 0` (see `06`). Both caveats apply to
every row unless stated. Paths are the file the number was read from.

## Speed baseline (2026-08-18, `outputs/training/sc_v3_ppo/COMPARISON.md`)

PPO 4×16, 400k steps, 32 envs: 141 s wall (~2830 steps/s) vs hours-class for
V1.7/Basilisk. Rollout reward −280 (still learning) — explicitly *not* a claim
to match V1.7's published +159.

## SC_v4 / PPO tuning (`outputs/tuning/tune_ppo/best.json`, `outputs/analysis/sc_v4_eval`)

Tuning winner `tune_v4a_01` (lr 1e-4, n_steps 128, batch 64, ent 0.003):
return 31.4 in the short eval, **−36.7 ± 45.9 on robust re-eval**, seeds 45–47
at −622 / −230 / −1110 with SoC_min 0. SC_v4a PPO 300k in the v4 eval: −8153
return, 329 brownouts. Lesson: short evals over-select.

## SC_v7 IPC campaign (2026-08-19, `outputs/campaign/LEADERBOARD.md`)

MPC gap 0.000 (decay 9.20, band 84.0 %, downlink 23, 98 J). Best RL
`s4_f0p3_th0p9_obs_sac` gap 0.468 — **invalid**: trained against a magnetic
field pointing the wrong way; re-scored on the corrected plant it is gap 2.914
with 13 brownouts (`outputs/presentation/README.md`).

## Production bake-off (2026-08-20, `outputs/presentation/README.md`, v7 plant, catalogue coils)

| policy | gap | decay nom | decay storm | band % | downlink | actuator J | ms/decision |
|---|---|---|---|---|---|---|---|
| MPC | 0.000 | 9.20 | 17.8 | 84.0 | 23.2 | 98 | 6.16 |
| Heuristic | 0.603 | 16.72 | 34.3 | 4.9 | 15.4 | 250 | 0.05 |
| PPO 50k | 1.097 | 27.0 | 71.9 | 31.0 | 12.0 | 332 | 0.09 |
| TD3 50k | 1.685 | 32.1 | 93.3 | 2.8 | 7.6 | 58 | 0.09 |

More training made things worse (PPO 250k 1.20, SAC 250k 3.18 with 12
brownouts). Network ~65× cheaper per decision. Actuator J here is the old
linear power model.

## v8/v9/v10/v11 family medians (188-run ledger, `outputs/v8/ledger.csv`; table in `docs/v12/00_LESSONS_v7_to_v11.md` §2)

| family | n | gap median | gap best | decay | band % | downlink | MTQ J | B |
|---|---|---|---|---|---|---|---|---|
| v8a (no delegation) | 36 | 1.30 | 0.48 | 26.2 | 3.8 | 11.6 | 97 | — |
| v8b | 36 | 1.51 | 0.86 | 30.2 | 3.5 | 10.3 | 177 | −0.12 |
| v9a | 12 | 1.81 | 0.73 | 30.0 | 4.1 | 12.9 | 173 | −0.08 |
| v9b | 12 | 1.43 | 0.46 | 31.3 | 4.0 | 11.6 | 132 | −0.18 |
| v9c | 12 | **1.20** | 0.52 | 23.8 | 2.9 | 9.6 | 86 | −0.11 |
| v9d | 12 | 1.64 | 0.39 | 26.7 | 3.5 | 10.2 | 105 | −0.22 |
| v10 | 56 | 1.49 | **0.10** | 28.5 | 3.7 | 9.7 | 107 | −0.06 |
| v11 | 12 | 1.12 | 0.38 | 28.4 | 3.8 | 13.8 | 140 | 0.00 |
| MPC | — | 0 | 0 | 9.3 | 86.6 | 23.2 | 3.0 | 0 |

The typical RL policy loses on every mission metric; band is the catastrophe
(3–4 % vs 87 %); actuation 30–60× the MPC. Three headline artifacts:

| | v10_4x18_ppo_1000k_s42 | v11_4x40_ppo_4000k_s42 | v9d_4x32_sac_300k_s42 |
|---|---|---|---|
| gap | **0.096** (seed siblings 0.82, 3.10) | 0.643 | 0.387 |
| decay km/d | **8.78** | 20.07 | 13.24 |
| band % | **92.4** | 5.9 | 35.9 |
| downlink | 7.67 | **25.6** | 15.3 |
| MTQ J | 1.45 | 55.5 | 7.8 |
| B / P | 0.49 / 0.42 | **0.75** / 0.47 | −0.23 / 0.18 |

No model has been MPC-class on all four metrics at once.

## Monte Carlo 512 orbits, 500 → 295 km (`outputs/v8/data/mc_decay.json`, 28-day cap)

| policy | reached 300 km | days (median) | decay @500 | @400 | @300 km/d |
|---|---|---|---|---|---|
| MPC | 252/512 | **17.9** | 5.40 | 14.3 | 88.2 |
| v10 0.096 artifact | 368 | 11.9 | 8.83 | 19.3 | **72.6** |
| heuristic | 388 | 10.6 | 9.00 | 21.3 | 105.6 |

The defensible "RL beats MPC" claim: −18 % decay at the 300 km crossing,
plus 60–80× decision cost. MPC owns total lifetime. (v3 MPC with the 69-day
cap: 29.96 d, 402/512 — the 28-day cap was a terminator artifact,
`outputs/mpc/checks/GATE.md`.)

## v11 damage (`outputs/v8/data/damage_eval.json`)

Adapt reward stabilised training (11 → 0 brownouts in round a0). Holes and
tears absorbed by everyone. **Honest negative:** on multi-hole strikes v11
decay 41 → 94 km/d, not better than v10's 25 → 54. Recovery-time metric
saturates at 1 step (too generous). Duos (v8Duo, v9Duo) equalled their main
advisor exactly — the arbiter never switched.

## SC_v12 → v14 (quiet 10-orbit and 24-draw MC, `outputs/v13/MC_FOUR.md`, `outputs/v13/traces/mc_four.json`)

Quiet freeze numbers (vs MPC_v2 column D 9.19 / 84.1 / 18.6):
SC_v13 decay 12.08, band 80.4 %, downlink 39.1, B +0.45, 1.00 ms.

| 24-draw MC, 28 d cap | MPC | v13 | v14b | v14d | v14e |
|---|---|---|---|---|---|
| days 500→300 | **16.4** | 12.9 | 12.9 | 13.3 | 13.3 |
| still above 300 km | 13/24 | 3 | 4 | — | — |
| coil J/day | **1.83** | 8.48 | 5.84 | 5.72 | 5.51 |
| band % | **76.9** | 44.4 | 42.9 | 41.5 | 38.8 |
| downlink min/d | 17.9 | 22.1 | 19.5 | 20.5 | 20.6 |
| B | 0 | +0.43 | +0.21 | +0.20 | +0.20 |
| brownout runs | **0** | 3 | 4 | 7 | 9 |
| U575 ms | 6.81 | 1.00 | 1.00 | 1.00 | 1.00 |

Precision: fp32 band 77.7 %, fp16 60.3, INT8 47.8 → fly fp32. Plot protocol
(`outputs/v14fix/README.md`): never mix quiet total-J with MC J/day; use
`02_mc_scorecard`; do not show `outputs/v14/plots/key_radar`.

## What is *not* in these tables

- Any result on the closed-loop magnetorquer plant.
- Any v8b/v10+ result with eq. (6) `deleg_power` actually live.
- Any policy trained on sensor-noised observations.
- ONNX/MCU deployment of any policy.
