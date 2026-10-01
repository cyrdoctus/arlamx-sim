# 05 — Gym environment and the variant ladder

`python/arlamx_v2/env.py::ArlamxV2Env(geom_path, ggm_path, seed, variant,
gate_floor, kp, kd, obs_extra, reward_w, controller, physics, gsi)`.
Gymnasium API; one `step` = one 300 s advisor step (150 s for v6). The policy
sees plant truth (r, v, ω, σ, B) — sensors feed only the torque KF. This is
the same privilege the MPC has, so advisor-vs-advisor comparisons are fair
but nothing is flight-representative on the input side.

## Observation (base 26, v7+ 35, v9+ up to 83)

Base 26: r/7000 km (3), v/8 km/s (3), ω rad/s (3), B̂_B (3), |B|/50 µT (1),
v̂_B (3), sin ν, cos ν (2), mass (1), log10 ρ (1), SoC (1), P_gen_norm (1),
GS direction + visible (4). Normalisation is ad hoc; SB3 `VecNormalize`
(reward only, `norm_obs=False`) does the rest.

v7 IPC block (+9): control τ/cap (3), disturbance estimate τ/cap clipped ±5
(3), mean gate (1), sun foresight r̂·ŝ, v̂·ŝ (2). `obs_extra ∈ {none, ctrl,
observer, observer_fore}` zero-masks for ablation without changing shape.

v9 temporal blocks: future (FP32 propagator) per offset [Δalt/10, Δsma/10,
SoC_proj, offset/600]; history per offset [alt/7000, SoC, log10 ρ, |ω|].
v9a 3 future (47), v9b 3+3 (59), v9c 6 past + 3 future (71), v9d 6+6 (83).
v10–v14 use the v9d layout (83).

## Action

v3–v5: 4 (quaternion, tanh → normalised; slerp-clipped 45° then 40°).
v6: 5 (+ torque scale 0.15–1). v7+: 7 = q (4) + three per-axis values. In v7
they are authority gates ∈ [gate_floor, 1]; in v8b+ they are the *delegation
split* s ∈ [0,1] mapped through eq. (1) gate = 1 − s(1 − floor); in v8a they
are ignored (control arm, gates ≡ 1).

## Episode envelopes

| variant | alt km | i° | e | F10.7 | Ap | jumps/episode | orbits/episode |
|---|---|---|---|---|---|---|---|
| v3 | U(300,500) | U(20,30) | 0 | U(80,200) | U(2,15) | — | 40 |
| v4–v9 | U(300,500) | U(20,30) | U(0.001,0.01) | U(65,250) | U(2,40) | 1–4 (ΔF ±80, ΔAp −20…+35) | 40 |
| v10/v12+ | U(300,500) | U(20,40) | U(0,0.01) | U(5,250) | U(2,200) | 2–8 (ΔF ±150, ΔAp −60…+150) | 120 |
| v11 | as v10 | | | | | + strike at 15–70 % of episode (p 0.6) | 160 |

v5+ adds 1–3 SRP flashes (×2–6 for 3 steps, off below 450 km). v8+ pins mass
at 0.625 kg and shifts panel centroids by the 1 cm cp offset. Physics YAML
f107/ap only *seed* the env; reset overwrites them unless
`reset(options={f107, ap})` is passed. v14b "climate mix" (40 % quiet /
35 % gradual+spikes / 25 % storm) is a wrapper-level re-draw.

## Power model

Loads: 7 mW housekeeping, +205 mW sunlit avionics, +400 mW during a pointed lit
pass. Generation 0.78 W × 0.85 × |ẑ·ŝ| (two-sided ±Z panel → nadir is a
power-positive comms attitude). Storage 0.53 Wh supercap, SoC₀ 0.5. Actuation:
v3–v7 linear 0.270 W × duty; v8+ quadratic I²R from the applied dipole (249 mW
peak). Brownout (SoC → 0) triggers `detumble` mode and a recovery loop
(v4b, v5b, v6, v7, v8+).

## Variant ladder (what each step changed)

| variant | change relative to its parent | evidence it mattered |
|---|---|---|
| v3 | SC_v3a recipe from V1.7: exponential longevity weight, SoC-modulated power, gs_alignment | speed baseline only |
| v4a/b | band reward (0.4–0.6 SoC), tiered GS pointing, feasibility term; v4b adds brownout recovery | v4 PPO 300k: −8153 return, 329 brownouts in eval (`outputs/analysis/sc_v4_eval`) |
| v5a/b | dE 3.5, lift_work, srp_work, torque_thrift; v5b harder brownout | never isolated |
| v6 | 150 s step, torque scale action, GPS-dropout skill, inference thrift | never isolated |
| v7 | IPC: per-axis authority gates + torque observer + foresight | campaign best RL gap 0.468 (later invalidated by the field-polarity fix → 2.914) |
| v8a/v8b | delegation ablation; custom coils; KF; YAML provenance; snapshot.json | v8a (no delegation) median 1.30 **beats** v8b 1.51; `deleg_power` was ≡ 0 (bug, fixed 2026-09-13) |
| v9a–d | temporal context ablation | v9d best run 0.387, v9c best median 1.20; richer obs → wider spread |
| v10 | perturbation-first weights, widened weather, 120 orbits, power_drop | best artifact 0.096 (seed siblings 0.82, 3.10); MC: −18 % decay at 300 km vs MPC, MPC still wins lifetime |
| v11 | mid-episode membrane damage + adapt terms | training stabilised (11 → 0 brownouts); B record 0.745; post-strike decay *not* better than v10 |
| v12 | wrapper only: slew term, MPC-scored comparator (H = 2), w_env | comparator cut slews 288 → 99/day; U575 1.00 ms |
| v13 | wrong-way clip, power-based τ_env; freeze at s3 s42 @600k | quiet: decay 12.08, band 80.4 %, dl 39.1, B +0.45 |
| v13.5 / v14 | plan/hold terms (dropped); regime estimator schedules kd, floor, κ, margin | dual gate never passed for v13.5 |
| v14b–f | climate mix; one-knob sweeps (w_low, w_env, w_band_storm, w_gen) | 24-draw MC: 12.9–13.3 d vs MPC 16.4; band 39–44 % vs 77; 4–9 brownouts vs 0 |

`train.py --variant` exposes v3–v9b. v10/v11 run through `bench_v8.py
--stage v10|v11` and `v10_tune.py`; v12–v14 only through `train_v12.py
--round`, whose `VARIANT = "v10"` and whose `V12Wrapper` supplies every later
term. Nothing after v11 changes `env.py`, the reward YAMLs or the plant.
