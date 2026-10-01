time: 2026-08-20T00:00:00Z
agent: claude
style: detailed

# v8/v9 fixes applied against the grok triple critique — status before training

**Input:** `2026-08-20_detailed_v8_v9_triple_critique.md` (grok, blockers B1-B9 + fix order 1-14).
**Gate held:** full suite green throughout (100 → **149** tests); the burn-in gate now PASSES.
**User decisions folded in:** mass **0.625 kg confirmed**; custom coils sized for 300 km worst case.

## Blockers — all nine closed

| ID | Fix | Proof |
|---|---|---|
| B1 | `tau_ctrl_mean` (and linear `torque_effort`) now on `info_r` before `compose_sc_v8` | `test_v8b_deleg_terms_fire_with_real_inputs`; burn-in mean P = 0.15 |
| B2 | eq. (4) rewritten around `tau_want`: saved only where the GATE binds (`want > gate*tau_max`); rate-cap savings excluded | `test_power_saved_zero_when_gate_not_binding`, `..._ignores_rate_cap_savings`, `..._positive_when_saturating` |
| B3 | eq. (2) magnitude re-scaled to `tau_env_ref = 1.5e-8 N*m` (the torque that turns the craft ~3 deg per step), not the PD ratio; direction stays signed | `test_alignment_uses_env_scale_not_pd_ratio`; burn-in mean-abs B = 0.61 |
| B4 | `gate_floor` default is now `None`; YAML 0.3 applies for v8+, historical 0.15 kept for v3-v7; explicit arg still wins | `test_gate_floor_comes_from_yaml`, `test_v7_untouched_by_v8_wiring` |
| B5 | brownout condition extended to v8/v9 | `test_brownout_reachable_for_v8` |
| B6 | v8a PINS gates = 1.0 (action[4:7] inert; 7-D action kept so both arms share one architecture). v8a vs v8b is now a one-mechanism ablation | `test_v8a_gates_pinned_open` vs `test_v8b_gates_follow_split` |
| B7 | `train.py` accepts v8a/v8b/v9a/v9b + `--arch`, freezes `snapshot.json`; new `bench_v8.py` runs burnin → ref → sweep → duo → v9 → trace | module exists, burn-in stage executed |
| B8 | returned `info` now carries `deleg_B/P/S/boost/align`, split, gates, `tau_want`, `tau_env_filt/raw`, `tau_aero_mean`, `mtq_power_W`, `tau_shortfall_frac`, `gate_floor`, `mass_kg`, v9 trend | `test_diagnostics_present_in_info` |
| B9 | One vehicle card: doc 14 §6, `power_mtq.yaml`, `env.py` comments all say custom coils 1.473/1.473/0.442 A*m², 38.24/38.24/11.48 µN*m, 249 mW; `torque_max` made EXACTLY `dipole*b_ref` (a rounding mismatch made full-scale requests read as saturated — caught by `test_config_self_consistent`) | tests/power |

## The physics fix that makes the experiment real (grok 2.3)

The symmetric hex geometry has only a tens-of-nN*m shear torque, while the
coils were sized against a 1 cm cp-cm imbalance — the sim would have carried
1000x oversized actuators and no reason to delegate. **v8+ now applies the
1 cm cp offset to the panel centroids** (`power_mtq.yaml: vehicle.cp_offset_m`),
so the simulated disturbance IS the one the actuators were sized against:
µN-class aero torque at 400 km, ~32 µN*m at 300 km in an extreme storm.
v3-v7 keep the symmetric geometry byte-for-byte.

Consequences measured in the burn-in: `tau_want` ~ 3.2 µN*m, `tau_aero`
~ 0.5 µN*m, gate binds on 20 % of slew-heavy steps, P fires at 0.15 mean.

## Other grok items closed

- **KF Joseph form** now uses the same inflated R as the gain (1.6); PSD test pins it.
- **KF retune with a story:** `sigma_jerk` renamed `sigma_drive` (it is the CV
  driving-noise PSD, not jerk — 2.5). 1e-10 → Q≫R (passed noise); 1e-12 →
  over-smoothed the new µN-class signal and DEGRADED raw r 0.98 → 0.62 — the
  burn-in gate caught that live. Final 3e-10 [DERIVED from ~1 µN/step slew
  dynamics]: filtered r = 0.9908 = raw, no degradation; the filter's value is
  the noise-dominated regime (synthetic test: ≥ 20 % RMS gain) and glitch
  rejection. Acceptance re-defined two-sided accordingly.
- **Gain table corrected** (2.4): with σ ≈ Φ/4, ω_n = √(kp/4I), ζ = kd/√(kp·I).
  v8 X/Y: ω_n 0.035 rad/s, ζ 1.85 — overdamped by design, which is the
  "gradual control" brief; the old ζ≈0.9 claim used the wrong linearisation.
- **v9 temporal fixes** (1.3): -150 s history slot was ALWAYS zero
  (`round(0.5)=0` banker's rounding) — now linear interpolation between the
  -300 s snapshot and now; SoC **is** projected forward (eclipse-aware first-order
  power model), so v9a's charge trend is no longer identically zero; trend ends
  match the spec (future = +600 s projection; past = -600 s snapshot for v9b,
  current sample for v9a — the ablation); propagation failures warn once instead
  of silently zeroing; `_fut_block[-4]` index fishing replaced by named fields.
- **Duo library** (1.5): SoC over the horizon now uses the propagated eclipse
  fraction + panel output (was: a rate nobody supplied, so SoC never moved);
  per-candidate drag via `attitude_bc_scale` (projected ram area per attitude —
  was: both candidates propagated with the SAME decay); switch-margin, override
  and feasibility behaviour pinned by tests. Duo is trained/evaluated only
  through `bench_v8.HorizonEnv`, which supplies `q_dot`, `power_norm`,
  `downlink_norm` — the pass-throughs grok flagged as undefined.
- **Flat reward overrides** (`{"dE_weight": x}`) now route into the `v7` section
  instead of being silently ignored (1.1); pinned by test.
- **Mass** fixed at 0.625 for v8+ (no more `uniform(0.5, 0.75)` with fixed
  inertia); v3-v7 keep the randomised path.
- Baselines are **re-run on the v8 plant** (`bench_v8.stage_ref`, v8a env with
  gates pinned): v7 bake-off numbers are kept OUT of v8 tables (3.5) — different
  coils, geometry, power law, brownout wiring.

## Deferred with reasons (per grok's own fix order)

- **Sensors into `_obs` + attitude filter** (fix 14) and **dipole-space
  saturation in C++** (fix 13): both are "separate decision" items in the
  critique; both change the plant for every variant. The observation remains
  truth-state + the measured-gyro KF chain; `power.achievable_torque` shortfall
  is logged per step (`tau_shortfall_frac`) so the along-B story is in the data
  even though it does not yet feed back into dynamics.
- ONNX export: still absent, still flagged. v8 zips remain undeployable until it exists.
- MSIS 300 s hold, GMST-vs-station 1.25°, SRP torque dropped in api.cpp: known,
  unchanged, listed in doc 14.

## Burn-in gate (final)

```
mean_deleg_P            0.150     (> 0.01 required)
frac_steps_P_positive   0.20      (>= 0.04)
mean_abs_B              0.61      (> 0.05)
kf_r_filtered / raw     0.9908 / 0.9908   (no degradation)
gate_floor              0.30      mass 0.625 kg
tau_max                 38.24 / 38.24 / 11.48 µN*m
GATE PASS · KF acceptance PASS
```

Campaign launched after this gate: ref → sweep (6 sizes × 3 algos × 2 budgets ×
v8a/v8b) → duo → v9 (top-2 sizes per algo × 2 budgets × v9a/v9b) → trace.
