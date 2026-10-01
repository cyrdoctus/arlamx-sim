time: 2026-08-19T02:00:00Z
agent: grok
style: detailed

# SC_v4 training and advisor bake-off (V2.0 plant)

Same matrix as V1.7 `scripts/train_v4ab.sh`, on the C++ plant. SC_v5 is
not started.

## What was trained

| Name | Variant | Algo | Steps | Envs | Seed |
|---|---|---|---|---|---|
| SC_v4a_4x16_PPO_300k | v4a | PPO | 300000 | 32 | 42 |
| SC_v4b_4x16_PPO_300k | v4b | PPO | 300000 | 32 | 42 |
| SC_v4a_4x16_SAC_300k | v4a | SAC | 300000 | 32 | 42 |
| SC_v4b_4x16_SAC_300k | v4b | SAC | 300000 | 32 | 42 |
| SC_v4a_4x16_PPO_50k | v4a | PPO | 50000 | 32 | 42 |
| SC_v4b_4x16_PPO_50k | v4b | PPO | 50000 | 32 | 42 |

Arch 4×16, lr 3e−4, VecNormalize reward-only, 40-orbit episodes.

## V4 plant extras (ported, not copied)

- `power_band` 40–60 % SoC, −50 if depleted
- `gs_tiered_pointing` 2/5/10 deg per axis, daylight
- `action_feasibility` 40 deg raw-command score
- `dE_vs_baseline` weight 2.0
- F10.7 [65,250], Ap [2,40], 1–4 in-episode weather jumps
- v4b: brownout → B-dot until SoC≥0.15 and |ω|≤0.5 °/s, +100 on hand-back
- Real GS catalogue (in-band stations), 26-dim obs, corotating wind

Heuristic (`mission_v17`) and sampling MPC are evaluated on the same
`v4a` env. Checkpoints: `outputs/SC_v4*/models/`. Bake-off:
`outputs/sc_v4_eval/`.

## Training wall time

All six finished in 447 s (~2750 PPO fps, ~3400 SAC fps).

## Bake-off (3 episodes, seeds 42–44, 40-orbit cap)

| Policy | mean return | min SoC | brownouts | mean Cd | Δh km |
|---|---|---|---|---|---|
| **mpc** | **−109** | 0.30 | 0 | 0.70 | −92 |
| heuristic_mission | −489 | 0.11 | 0 | 0.88 | −47 |
| min_drag | −490 | 0.45 | 0 | 0.75 | −45 |
| SC_v4a SAC 300k | −1209 | 0.47 | 0 | 1.29 | −154 |
| SC_v4b SAC 300k | −1209 | 0.45 | 0 | 1.21 | −154 |
| SC_v4b PPO 300k | −1363 | 0.47 | 0 | 1.25 | −154 |
| SC_v4b PPO 50k | −1668 | 0.00 | 16 | 1.36 | −156 |
| SC_v4a PPO 50k | −4054 | 0.00 | 152 | 1.33 | −155 |
| SC_v4a PPO 300k | −8153 | 0.00 | 329 | 0.96 | −154 |

`survived=false` on several rows is the **250 km altitude terminal** (domain
randomization starts as low as 300 km), not always a brownout. MPC has the
best return; SAC and v4b PPO hold the SoC band; v4a PPO 300k does not.

SC_v5 is not started.

## Commands

```
PYTHONPATH=python python -m arlamx_v2.train_v4
PYTHONPATH=python python -m arlamx_v2.eval_v4
```
