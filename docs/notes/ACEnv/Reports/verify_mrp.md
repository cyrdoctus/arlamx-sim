time: 2026-08-18T22:10:00Z
agent: grok
style: detailed

# Verification — MRP kinematics (`cpp/src/attitude/mrp.cpp`)

## Connection map

- File owns: DCM, shadow set, B(σ) rate, composition, error, angle.
- Called by: plant RK4, MRP-PD, Python `mrp_to_dcm`.
- Calls: `types.hpp` (`skew`, `mul`).
- Boundary: σ is the body-to-inertial MRP (Schaub passive). Angle = 4 arctan|σ|.

## Pass 1 — formatting and simplicity

One function per identity. `mrp_error` is compose(σ, σ_t⁻¹), which is the readable form. No change.

## Pass 2 — assume it is wrong

1. `mrp.cpp:13–16` — DCM `I + 8[σ̃]²/(1+σ²)² − 4(1−σ²)[σ̃]/(1+σ²)²`. Suspect: sign of the linear term (active vs passive). Source: Schaub & Junkins 2018 Ch. 3; Shuster 1993.
2. `mrp.cpp:25–39` — `σ̇ = ¼ B(σ) ω` with B = (1−σ²)I + 2[σ̃] + 2σσᵀ. Suspect: ½ vs ¼, or B vs Bᵀ. Source: Schaub & Junkins Ch. 3.
3. `mrp.cpp:19–22` — shadow when σ²>1, σ ← −σ/σ². Suspect: switch at |σ|=1 exactly (180°). Source: Schaub: switch *outside* the unit sphere so 180° stays unique.
4. `frames.cpp:164–168` — `quat_to_mrp = q_{1:3}/(1+q0)`. Suspect: 180° map (`q0=−1`) returns 0. Source: Schaub; the plant now refuses to integrate a 180° PD target in prescribed mode.

## Pass 3 — adjudicate

1. **Keep** the DCM. The 90°-about-z classroom test was previously failed by a *test* sign, not the DCM.
2. **Keep** the ¼ B(σ) ω kinematics.
3. **Keep** shadow on σ²>1 (not ≥).
4. **Keep** the quaternion map, and **fixed the consumer**: decay and lifetime holds use `mode=prescribed` so a large first command does not integrate MRP-PD at 180 s with unlimited torque (that was the NaN decay). Point-mode training still uses 2 s control steps and a 40° slew cap.

Tests: `tests/attitude/test_mrp.py` pass.
