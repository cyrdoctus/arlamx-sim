# 04 — Actuators and control: what the inner loop is, and the authority finding

## Inner loop

The advisor emits one quaternion per 300 s. The plant tracks it with a 2 s
MRP-PD (`control/mrp_feedback.cpp`, module 07):

    τ_cmd = −k_p σ_e − k_d ω_e + ω × Iω          (proper MRP error, Schaub & Junkins Ch. 8)
    per-axis clip to max_torque (carries the v7/v8 gate),
    rate limiter: if predicted |ω| > max_body_rate (0.125°/s), replace τ with the
    one-step deceleration torque, clip again.

Quaternion law (`quat_feedback.cpp`, module 16, v2.5) is the same PD on q_v;
since q_v ≈ θ/2 while σ ≈ θ/4 the same k_p is ~2× stiffer. MRP is the default.

B-dot detumble (`bdot.cpp`, module 08) runs in `detumble` mode (entered on
brownout): m = −k Ḃ_B, per-axis coil clip.

## Gains (`python/configs/gains_mrp.yaml`)

| set | k_p | k_d | ω_n (X/Y) | ζ | used by |
|---|---|---|---|---|---|
| v3–v7 constants (`env.py` MTQ_KP/KD) | 4.0e-4 | 8.0e-3 | 0.089 rad/s | 3.6 | v3–v7 |
| v7 stage-S4 winner | 1.0e-4 | 2.0e-3 | 0.045 | 1.79 | SC_v7 campaign |
| **v8+ yaml** | **6.0e-5** | **1.6e-3** | 0.035 (181 s period) | 1.85 | v8–v14 |

Linearisation with σ ≈ Φ/4: I Φ̈ + k_d Φ̇ + (k_p/4) Φ = 0, ω_n = √(k_p/4I),
ζ = k_d/√(k_p I). All overdamped by design (spread a 40° slew across the 300 s
step). The 3×3 sweep grid in the yaml was never run.

## Magnetorquers

**v7 (catalogue):** CubeSpace CR0006 0.6 A·m² X/Y, EXA MT01 0.19 A·m² Z →
18/18/5.7 µN·m at the reference field. Linear power model 0.270 W × duty
(overcharged 180–600× at real duty).

**v8+ (custom wound, `power_mtq.yaml`):** sized for 300 km, F10.7 250, Ap 180,
broadside 0.65 m², Cd 2.2 → 3.19 mN drag, 1 cm cp–cm offset → 31.9 µN·m, ×1.2
margin → 38.2 µN·m at the weakest field 25.96 µT → **1.473 / 1.473 / 0.442 A·m²**,
**38.24 / 38.24 / 11.48 µN·m**, I²R **107 / 107 / 35.6 mW** (249 mW full
3-axis, quadratic P ∝ m²), mass 161 g = 25.7 % of vehicle (flagged). The v14
force-range check found the real design-storm aero force is 2.43 mN / 24.3 µN·m,
so the coils cover it with margin in the *perpendicular* plane.

## Allocation (`control/magnetorquer.cpp`, module 15, v2.5)

    m = (B × τ_cmd)/|B|²,   m_i = clip(m_i, ±m_max,i),   τ = m × B
    shortfall = 1 − |τ|/|τ_cmd|      (includes the unproducible τ ∥ B part)

τ·B̂ = 0 by construction. Per-axis clipping never produces an anti-aligned
torque (τ_clip·τ_cmd ≥ 0). `SimParams.ideal_torque = true` bypasses this and
applies τ_cmd as a body couple — the plant every campaign was trained on.

## The finding (2026-09-13, first-hand measurements)

With `ideal_torque=false` (default since v2.5) the plant **cannot hold
attitude with either gain set**. Hex geometry, 350 km, seed 3, 40 advisor
steps (3.3 h), attitude command held, full authority; mean tracking error over
the last 10 steps:

| plant | gains | i = 23° | i = 60° | i = 90° |
|---|---|---|---|---|
| ideal couple | v7 | 0.02° | 0.03° | 0.02° |
| coils through B | v7 | 74° (peak 174°) | 92° | 70° |
| ideal couple | v8 yaml | 6° | 13° | — |
| coils through B | v8 yaml | 100° (peak 145°) | 120° | — |

It diverges the same way with the atmosphere and SRP removed and zero
initial rate, so it is the actuator, not the disturbance. Mechanism: the
stiff PD (I/k_d ≈ 1.6 s against a 2 s step) removes the B̂-perpendicular rate in
one step and pins ω onto B̂; the attitude drifts about B̂, the proportional term
pushes perpendicular, and the slow rotation of B̂ (1–2 mrad/s) folds that push
back into the uncontrolled axis. A projected PD is only stable when the gains
are slow relative to the field rotation (Lovera & Astolfi 2004). Confirmed:
with k_p 4e-8, k_d 8e-5 (three orders softer), no disturbance, a 20° slew
converges to 10–20° over 30 h; with the aero torque on it does not converge at
23°. Pure rate damping with k_d = 8e-5 behaves textbook-like (|ω| decays
monotonically, delivered torque always opposes rotation); with k_d = 8e-3 the
rate is pinned and never decays.

Consequences a dissertation must state:
- Every v7–v14 number assumes an actuator that can torque about B̂. The
  40°-per-300 s slew architecture is not achievable with magnetorquers alone.
- The aero torque alone (45 nN·m on the symmetric hex; µN·m with the v8 cp
  offset) turns the vehicle by ~100° in the time B̂ takes to rotate, so
  magnetic-only 3-axis hold of this sail is infeasible in any orbit tested.
  Options: momentum device, aerodynamic passive stability (sail as a
  shuttlecock, which is the delegation idea taken seriously), or a control law
  designed for time-varying B (Lovera-type averaging, LQR-periodic).
- Related known gap (`docs/modules/14_reward_v8.md` §6.1): "dipole-space
  saturation in C++" was listed as out of scope through v14; v2.5 added it and
  nobody re-ran the campaigns on it.

## Observer and torque estimate (`estimators.py`, `env.py`)

Flight-knowable disturbance torque: z = I Δω/Δt + ⟨ω × Iω⟩ − τ_ctrl (the
delivered torque), then a constant-velocity Kalman filter (σ_drive 3e-10
N·m/s^1.5, σ_τcmd 2e-9, 5σ innovation gate with R inflation, Joseph form).
Gyro noise from `sensors_solarcat.yaml` (ICM-42688-P ARW 0.0028 dps/√Hz).
Burn-in acceptance 2026-08-20: filtered r = 0.9908 vs truth. Before 2026-09-13
the gyroscopic term was the endpoint ω₁ × Iω₁, which is exact only while rates
stay near zero across the step; under the closed loop it was as large as the
aero torque (cos 0.83, magnitude 1.8×). Now exact via `gyro_mean`.

## Onboard budget (`inference_budget.py`, STM32U575 analytic, 0.35 FLOP/cycle)

Policy 4×18 on 83-D obs: 0.17 ms. Policy + comparator (H = 2): 1.00 ms.
Sampling MPC: 6.81 ms published-scale (13.4 ms actual-scale). Budget 15 ms.
Fly FP32 (INT8 dynamic quantisation dropped quiet band 77.7 → 47.8 %).
