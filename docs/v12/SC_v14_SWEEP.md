# Focused sweeps — SC_v14c results and what v14d should actually try

This is the regular-tuning plan for **ARLAMX V2.0 / SC_v14**, not the later
onboard auto-tuner (`docs/AUTOOPTIMIZER_NOTE.md`, v2.5/v3). One knob per arm.
Resume **`outputs/v13/ckpt_v14b/`** unless an arm dual-gates and MC-beats it.

Do **not** raise decay / `dE_weight` / `gs_weight`. Do not drop the comparator.
Do not mix quiet 10-orbit with the 24-draw MC. Fly **fp32**.

---

## 1. Where we are

### Dual-gated (the only zip MC was run on)

| | MPC | SC_v13 | **SC_v14b** |
|---|---|---|---|
| Quiet 10-orbit band / dl / B | 84.1 / 18.6 / 0 | 80.4 / 39.1 / +0.45 | **77.7 / 23.5 / +0.11** |
| Quiet decay km/d | 9.19 | 12.08 | 14.09 |
| MC days 500→300 | 16.4 | 12.9 | **12.9** |
| MC still above 300 | 13/24 | 3/24 | **4/24** |
| MC band % | 76.9 | 44.4 | **42.9** |
| MC coil J/day | 1.83 | 8.48 | **5.84** (< 6) |
| MC downlink | 17.9 | 22.1 | 19.5 (≥ MPC) |
| MC B | 0 | +0.43 | **+0.21** |
| MC brownout runs | 0 | 3 | **4** |
| U575 | 6.81 ms | 1.00 | 1.00 |

Quiet v14b **did** dual-gate (storm1d band 68.1, dl 13.9, B +0.14). MC band and
brownouts did not. That is the hole v14d is for.

### SC_v14c (empty dual-gate — do not MC this)

`w_band_storm=0.20` + `w_low=0.15` from v14b. Quiet GS and B went **up**;
quiet band and storm GS still trade.

| ckpt | quiet band / dl / B / decay | storm band / dl / B | why not dual |
|---|---|---|---|
| s42 @ 200 k | **88.0 / 31.3 / +0.13 / 12.40** | 53.2 / 7.6 / +0.08 | storm band |
| s42 @ 250 k | 64.1 / **47.0** / **+0.36** / 11.97 | 56.3 / 21.0 / +0.07 | quiet band |
| s42 final | 72.3 / **47.0** / +0.24 / 12.21 | 59.3 / **30.5** / +0.17 | quiet band 75, storm 62 |
| s43 @ 300 k | 76.6 / **39.1** / **+0.21** / 14.60 | **68.8** / 8.5 / +0.24 | storm dl 12 |

Lesson: stacking band-in-storm **and** low-SoC in one train repeats the
v13.5 failure mode (one metric up, the dual zip empty). **One knob per arm.**

### Precision (v14b weights, not a retrain)

| | quiet band | quiet B | storm band |
|---|---|---|---|
| fp32 | 77.7 | +0.11 | 68.1 |
| fp16 store | 60.3 | −0.02 | 53.8 |
| dynamic INT8 | 47.8 | −0.07 | 77.7 (not a drop-in) |

Fly fp32. U575 is an fp32 FPU. Do not sweep precision until MC band is closed.

---

## 2. What is allowed to move (and what is not)

**Hold (do not sweep this round)**

- Plant, `env.py`, `mpc_v2.yaml`, `gains_mrp.yaml`
- Decay / `dE_weight` / `gs_weight` / brownout_weight in YAML
- Dropping or H=1 the comparator
- Obs dim, LSTM, bigger net
- fp16/INT8 retrain
- Climate-mix percentages (already 40/35/25)
- PPO n_steps/batch (already 256/256/4) — only if a reward arm dual-gates and MC still misses

**Sweep (one at a time, resume v14b, 200 k, seed 42)**

| # | knob | values | why | freeze if |
|---|---|---|---|---|
| D1 | `w_low` | **0.10** (vs 0) | MC 4 brownouts; v14c mixed this with band | 0 brownouts on storm1d, quiet band ≥ 75 |
| D2 | `w_env` | **0.40** (vs 0.35) | MC B +0.21 vs v13 +0.43 | B ≥ v14b, band not < 75 |
| D3 | `w_band_storm` | **0.08** (vs 0 and vs c’s 0.20) | MC band 42.9; 0.20 was too coarse | storm1d band ≥ 68, quiet dl ≥ 22 |

Do **not** run D1+D2+D3 in one train. That was v14c.

**Later (only after an arm dual-gates and we have a new MC)**

| later | knob | note |
|---|---|---|
| L1 | `tau_ref` {3e-6, 5e-6, 8e-6} | η scale; storm1d vs 300 km |
| L2 | `clip_kappa_storm` {0.85, 1.0} | coil J vs B |
| L3 | climate storm fraction 25→35 % | more MC-like episodes, keep 40 % quiet |
| L4 | PPO lr 5e-5 vs 7e-5 | only if weights thrash |
| L5 | gated comparator | speed, not MC band |

**Picker for v14d:** same as v14b (`keep_ok_v14`), not the stricter v14c storm-dl≥12 gate (that empty zip was the gate, not the physics). Ship bar is still the **24-draw MC**: days ≥ 12.9, J ≤ 6, band **up**, B ≥ +0.21, dl ≥ 19.5, **0 brownout runs**. Quiet 10-orbit is the regression gate only.

---

## 3. Automatic methods (train-time first, onboard later)

**Now (V2.0, CPU, no extra model)**

1. **One-knob grid** — what v14d is. Cheap, attributable.
2. **Coordinate descent** — after D1–D3, take the winner and move the next knob (s4 style).
3. **Dual-eval as the constraint**, MC as the objective. Never optimise quiet score_A.

**Do not start with** Bayesian optimisation / PBT / population-based PPO. 24 envs × 200 k is already a minute-scale arm; a 20-trial BO is an afternoon and you will not know *which* knob did it. Save BO for the v2.5 runner.

**Train-time auto (V2.5 sketch, not this week)**

A version-agnostic script loads `configs/sweep_*.yaml` + latest `ckpt_*` + `eval_curve.csv`, runs the listed arms, appends a ledger, stops when dual_ok or budget. A small local LLM (8B, at most 27B) may **write the next YAML**, not the weights. See `docs/AUTOOPTIMIZER_NOTE.md`.

**Onboard auto (V3 sketch, not this week)**

Not PPO on the U575. The flight tuner is the **regime table** already started in v14: η → (kd, gate_floor, κ, margin). Collect (η, SoC, band-dwell, coil J) over orbits, write a lookup table, interpolate. That is the “big table inferred any time in orbit” to keep J low. A second network or an 8B model does **not** fly.

---

## 4. v14d in one paragraph

Resume v14b. Three 200 k arms, seed 42, dual-eval every 50 k, picker = v14 gate.
Arm D1 `w_low=0.10` (brownouts). Arm D2 `w_env=0.40` (B). Arm D3 `w_band_storm=0.08`
(band, lighter than c). Copy the first dual-gated zip to `ckpt_v14d/`. If none,
**keep v14b** and say so. Only then 24-draw MC. fp32. No decay weights.
