# 07 — Baselines: the sampling MPC and the heuristic

Both live in `python/arlamx_v2/advisors/`. They are what every RL number is
measured against; the "gap index" is the mean signed normalised distance to
the MPC over {decay, band, brownouts, downlink, decay_spike} (0 = MPC, negative
beats it). Note the gap index hides per-metric losses (`08`).

## Sampling MPC (`advisors/mpc.py`, `SamplingMpcPolicy`)

Receding-horizon over a cheap surrogate, every 300 s:

| item | value |
|---|---|
| candidates | hold, sun (ephemeris), gs (if station ≥ 10° el.), min_drag (Z ∥ h), nadir (Z ∥ −r), + 8 random body-axis perturbations U[2°, 34°] |
| horizon | 6 steps = 30 min; 60 s internal two-body RK4 |
| surrogate | two-body orbit, first-order SoC (P_gen = 0.78·0.85·|ẑ·ŝ|, loads 7/205/400 mW), Sentman Cd on the real panels, cylindrical eclipse, GS cosine |
| objective | Σ_k [B(SoC_k) + 1[pass, lit](T(θ_gs) + w_align cos θ_gs) − w_cd Cd_k] − w_slew Δcmd/40° |
| weights | power_band 1.0, power_low_pen 4.0, gs 2.0, gs_pen 0.5, gs_align 1.0, cd 0.5, slew 0.2 (v1) → 0.5 (v2/v3) |
| inputs | truth r, v; Sun from ephemeris; MSIS at the current point; SoC; GS direction. v2/v3 add GNSS/mag/SoC noise (0.5 %) |
| output | one q_BN; plant MRP-PD tracks it; no delegation; full authority |
| cost | 8.4 ms (v1 analytic) / 6.81 ms published-scale, 13.4 ms actual-scale (v2) on STM32U575 |

**Tuning.** None until 2026-08-23 (`docs/v12/MPC_tuning.md`): 161 configs,
rounds A–F over horizon {3,6,9,12}, n_random {8,16,32}, cd × slew, gs × align,
power_low × k. Winner kept every v1 knob except slew 0.2 → 0.5 and sensors on.
Frozen as `mpc_v2.yaml` (2026-08-23T16:31Z) and `mpc_v3.yaml` (identical
knobs, 20-orbit × 3 probes × 2 seeds protocol). `outputs/mpc/checks/GATE.md`:
the plant reproduces the published v1 reference exactly.

**Reference numbers (v8+ plant, ideal couple).**

| metric | MPC v1 (`ref.json`) | MPC v2 "column D" | MPC v3 |
|---|---|---|---|
| decay km/d (nominal) | 9.28 | 9.19 | 9.189 |
| decay spike | 17.60 | — | 18.1 |
| band % | 86.6 | 84.1 | 84.14 |
| downlink min/d | 23.2 | 18.6 | 18.64 |
| brownouts | 0 | 0 | 0 |
| MTQ J (3-probe aggregate) | 2.97 | 3.73 | 3.73 |
| MC 500→295 km lifetime (median) | 17.9 d (28 d cap) | — | 29.96 d (69 d cap), 402/512 reach 300 km |
| decay at 300 km crossing | 88.2 km/d | — | 65.9 |

Probes: nominal 400 km / i 23° / F10.7 150 / Ap 4, 120 orbits; spike +100
F10.7 / +36 Ap at 25 %; offnominal 550 km / 51.6° / RAAN 40°. Seeds 1001, 1002.
Initial ω (1, 1, 0.5) °/s.

**Where it is weak (measured).** Deep altitude: at the 300 km crossing the
v10 RL artifact decays 72.6 km/d vs MPC 88.2 (−18 %). Decision cost 60–80×
an RL advisor. Downlink counted only when lit (`daylight_only`), same as the
RL reward, so both may under-count. A stale panel model under damage costs it
nothing on 30-orbit tracking.

## Heuristic (`advisors/heuristic.py`, `HeuristicPowerPolicy`)

The "no model" reference: closes only on SoC and P_gen_norm. Mode ladder on
SoC thresholds (0.60, 0.50, 0.40, 0.30, 0.20): rich → min-drag or station;
mid → station else last-good-sun else search; low → power search (sweep 15°
about Y then X until gen ≥ 0.05, coordinate descent 6° halving, hold at gen ≥
0.70 or step < 1°); eclipse guard holds 6 steps after a fruitless sweep.
Bank of 13 swept once on the 1 % SolarCat mesh (`outputs/analysis/advisors/`);
three members died by latching a dim false sun into umbra. Never re-swept on
the v8+ plant. Gap 0.60 on the v8 plant: band 5 % is its whole deficit
(decay 17.1 km/d, downlink 20.6 min/d, MTQ 32 J).

## Fixed attitudes (`advisors/attitudes.py`)

`MinDragPolicy` (smallest projected-area axis along v_rel), `MaxDragPolicy`,
`NadirPolicy`, `OffNadirPolicy`, `GroundStationPolicy`, `FixedAoAPolicy`
(hold a given angle of attack against the mid-step relative wind, lift aimed
+h or +r). Used by `decay_run`, `aoa_decay`, `mc_decay` and as MPC candidates.
