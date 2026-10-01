time: 2026-08-18T22:10:00Z
agent: grok
style: detailed

# Verification — Sentman GSI (`cpp/src/aero/sentman.cpp`)

## Connection map

- File owns: `sentman(θ, s, Tw/Ti, α_E) → (Cp, Cτ)` and panel sum `spacecraft_aero` / `coefficients_only`.
- Called by: `Simulator::step` (`api.cpp`) every control substep; Python bindings for tests and the MPC Cd path.
- Calls: `constants.hpp` (`K_B`, `PI`).
- Boundary: `v_rel_B` is the **gas** velocity in the body frame (plant supplies `C (v − ω⊕×r)` with a minus sign). `ρ` [kg/m³], `T` [K], `m_bar` [kg], `T_w` [K]. Force in body frame.

## Pass 1 — formatting and simplicity

The kernel is two jobs that are already split: the scalar Sentman pair, and the panel reduction. Names carry units (`Tw_Ti`, `v_rel_B`, `one_sided_ref`). No inline comments were added. The file is well under the 1000-line split. No change in this pass.

## Pass 2 — assume it is wrong

1. `sentman.cpp:23` — `Tr_Ti = α_E Tw_Ti + (1−α_E) s²/2`. Suspect: the older “temperature accommodation” `1 + α(Tw/Ti − 1)`. Source: Moe & Moe (2005) energy-flux definition of α_E; Sentman (1961) re-emission term uses √(Tr/Ti).
2. `sentman.cpp:24–25` — `Cp_r = √(Tr/Ti) (E + √π γ Z) / (2 s²)`. Suspect: missing factor 2, or shear also re-emits. Source: Sentman 1961; Doornbos 2012 eqs. 3.53–3.55 (diffuse re-emission is normal only, so Cτ = Cτ_i).
3. `sentman.cpp` panel loop (A_ref = ½ ∑A when one-sided) — suspect: reference area should be projected ram area. Source: V2.0 contract / Moe & Moe 2005: Cd is reported on a fixed one-sided wetted area so attitudes are comparable. Not a physics error if documented.
4. Hyperthermal identity `Cp cosθ + Cτ sinθ → 2 cosθ` is the **incident** identity. With re-emission the residual is O(1/s). Source: Sentman 1961, high-s limit of incident terms.

## Pass 3 — adjudicate

1. **Keep** M-01. Classroom test `α_E=0` is hotter than `α_E=1` (`test_sentman.py`). The pre-V1.7 formula fails that case.
2. **Keep** normal-only re-emission. Matches Sentman/Doornbos. Residual 0.03 at s=80 is the re-emission floor, not a bug.
3. **Keep** one-sided A_ref. Decay and advisors compare Cd across attitudes on the same reference.
4. **Keep** the hyperthermal test tolerance at 0.03.

No equation change. Tests still pass (`test_sentman.py`, 4 cases).

## Known case

| Input | Expected | Obtained |
|---|---|---|
| s=80, θ=0.4, α=1, Tw=Ti | incident combo = 2 cosθ | residual < 0.03 |
| unit square, face-on, α=1 | Cd ∈ [1.8, 2.4] | pass |
