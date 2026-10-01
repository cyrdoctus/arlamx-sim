# 01 — Program history and current state

## What ARLAMX is

A reinforcement-learning attitude advisor for a ~0.6–1.8 kg drag-sail
CubeSat ("SolarCat" / hex sail) in very low Earth orbit. The thesis question
as the code frames it: can a small onboard network (PPO, 4×18, FP32, STM32U575
class) choose a 300 s attitude command that manages orbital decay, battery
band, ground-station downlink and actuator energy at least as well as a
sampling model-predictive controller, at 5–60× lower decision cost, and can it
exploit environmental (aerodynamic) torque instead of fighting it ("delegation").

Two spacecraft models: the 72-plate `earthcup_hex_v3.geom` (0.625 kg campaign
vehicle; 0.72 kg in the Sept-2026 decay studies) and the CAD
`SolarCat_Assembly.STL` (4604 triangles, "six-petal", 1.75 kg), simplified to
sealed plates by `geometry.py`.

## Timeline

| Date | Version | What happened | Where recorded |
|---|---|---|---|
| ≤ 2026-08 | V1.7 | Python/Basilisk predecessor. Read-only reference for equations, YAML numbers, `.geom` data, SC_v3a recipe. Never edited. | `ARLAMX_V2.0_OVERVIEW_AND_CPP_PLAN.md` (survey) |
| 2026-08-18 | V2.0 | Build contract approved: new C++ plant (Sentman, panel SRP, GGM03S, SPICE/analytic 3rd body, MRP+RK4, MRP-PD, B-dot, dipole/WMM) + Python Gym/PPO. Basilisk is a referee only. SC_v3a PPO 400k in 141 s wall. | `APPROVAL_PLAN.md`, `claude_handoff.md` |
| 2026-08-18/19 | V2.0 | Equation register; corotating wind default on with lift-work split; optical plate SRP; heuristic + sampling MPC ported; SC_v4/v5/v6 trained; PPO tuning (32 trials); OOD gradient sweeps; SC_v7 IPC campaign (53 runs). | `ACEnv/Reports/2026-08-18..19_*`, `outputs/{training,tuning,campaign,analysis}` |
| 2026-08-20 | V2.0 | Independent audits (grok, claude). Fixes: WMM Schmidt recursion, lunar frame, dipole polarity, J3 fallback, hardcoded WMM year, B-dot reset. Prior campaign winner invalidated (learned against a wrong-way field: gap 0.468 → 2.914). Production bake-off PPO/SAC/TD3 vs MPC. | `ACEnv/Reports/2026-08-20_*`, `outputs/presentation/` |
| 2026-08-20 | v8/v9 | Grok's triple critique: 9 blockers (B1–B9) before any v8 training. Fixed; custom coils sized for 300 km; 1 cm cp–cm offset baked into v8+ geometry; mass 0.625 kg confirmed. 96-run campaign. | `2026-08-20_detailed_v8_v9_triple_critique.md`, `..._v8_fixes_after_grok.md`, `..._v8_v9_campaign_results.md` |
| 2026-08-21 | v10/v11 | v9c/d temporal ablation → v10 (widened weather, 120-orbit episodes) → 512-orbit Monte Carlo → v11 (membrane damage + adapt reward). 188-run ledger. | `2026-08-21_detailed_v10_v11_campaign_results.md`, `HANDOFF_2026-08-21_STATE.md` |
| 2026-08-23 | MPC_v2/v3 | MPC tuned for the first time (3 h sprint, 161 configs), frozen with sensors on. | `outputs/mpc/`, `docs/v12/MPC_tuning.md` |
| 2026-08-23..25 | v12–v14 | Wrapper-only overlay on the v10 env: slew term, comparator (MPC scorer as a safety filter), wrong-way clip, regime estimator, climate mix. SC_v13 freeze; v14b/d/e picks; 24-draw four-task MC; talk figures. | `outputs/v12`, `outputs/v13`, `outputs/v14`, `outputs/v14fix`, `docs/v12/*` |
| 2026-09-04 | v2.1 | Equation + performance audit: full-Sentman wetted faces (min-drag Cd 0.079 → 0.214, changes every baseline), inertial-frame drag baseline (~6 % bias removed), FP32 onboard propagator in C++, portable data paths. 159 tests. | `CHANGELOG_v2.1.md` |
| 2026-09-05..07 | studies | Fixed-attitude decay of hex vs six-petal (500→300 km), 45° AoA lift study, SRP orbit-raising and F10.7 feasibility sweeps. No training. | `outputs/analysis/decay/`, `11_decay_lift_srp_studies.md` |
| 2026-09-10 | v2.5 | Closed-loop magnetorquer path (τ = m×B) becomes the plant default; quaternion law; WMM in the Gym; mamba env `arlamx` (Py 3.12); INT8 post-train quantizer. | `CHANGELOG_v2.5.md` |
| 2026-09-13 | v2.6 | `python -m arlamx_v2` dispatcher, YAML physics presets, Walker–CLL GSI in C++. Review: CLL transcription verified against ADBSat; wiring bugs fixed; then the two v2.5 test failures diagnosed and fixed (observer gyro term; delegation counterfactual), and the closed-loop authority finding recorded. 208 tests green. | `CHANGELOG_v2.6.md`, `docs/modules/15_control_magnetorquer.md` |

## State on 2026-09-13

**Plant.** C++ pybind module `arlamx_cpp` version 2.6. Sentman default GSI,
CLL optional. Gravity GGM03S to degree 70 (presets use 2/4/8). WMM2020 field,
dipole fallback. Panel SRP (cannonball Cr 1.8 or optical). MSIS 2.1 via pymsis
from Python. Magnetorquer through the field by default; `attitude.ideal_torque`
is a declared physics key. Rotational identity exported exactly
(`gyro_mean`, `tau_demand_mean`).

**Training code.** Runs but is *not* what the last month was spent on; the user
had been using the plant for fixed-attitude studies since 2026-09-05. Trainers
exist for v3–v9b (`train.py`), v10/v11 (`bench_v8.py`, `v10_tune.py`), and
v12–v14 (`train_v12.py --round`). No policy is deployable (no ONNX/MCU export).

**Tests.** `PYTHONPATH=python python -m pytest tests -q` → 208 passed, 0 xfail
(Basilisk referee tests skip without Basilisk).

**Best quotable RL artifacts.** SC_v13 freeze `outputs/v13/ckpt/model.zip`;
v14b (`outputs/v13/ckpt_v14b`, fewest MC brownouts) and v14d
(`outputs/v13/ckpt_v14d`, 13.3 d lifetime, 5.72 J/day). None clears the
24-draw Monte Carlo ship bar (MPC: 16.4 d, 76.9 % band, 1.83 J/day, 0 brownouts).

**Immediately open.**
1. Decide the actuator model for any future training (ideal couple vs coils),
   and what a magnetically feasible control architecture is (`04`, `12`).
2. v8b delegation reward has never actually been trained with `deleg_power`
   live; re-running the v8a/v8b ablation is the first experiment that changes.
3. The docs/agent-skills plan (`14`) is agreed in outline, not built.
