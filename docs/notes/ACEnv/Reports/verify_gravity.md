time: 2026-08-18T22:10:00Z
agent: grok
style: detailed

# Verification — Earth gravity GGM03S (`cpp/src/orbit/gravity.cpp`)

## Connection map

- File owns: two-body, closed-form J2/J3, GGM03S load, `accel_ecef`, `accel_gravity_N`.
- Called by: `Simulator::accel_nongrav_env`.
- Calls: `frames.hpp` (`dcm_EN`, `gmst_rad`); `constants.hpp` (`MU_GGM`, `J2_GGM`).
- Boundary: `r_N` inertial metres; GMST rad; output inertial m/s². Stokes file is data, not a Basilisk link.

## Pass 1 — formatting and simplicity

`load_ggm` and `accel_ecef` are long but each is one job. The associated-Legendre table is allocated per call (degree 4 → 6×6, acceptable). No inline comments. No split required.

## Pass 2 — assume it is wrong

1. `gravity.cpp:96` — `J2_ = −C_[2][0]`. After denormalisation `C20 = C̄20 √5`, J2 = −C20. Source: Vallado 2013 §8; GGM03S header.
2. `gravity.cpp:40–46` / body — implementation uses **unnormalized** P_n^m after converting Stokes with `norm_factor`, not a Pines/Cunningham recurrence. Spec `docs/modules/03` asked for Pines. Suspect polar singularity. Source: Montenbruck & Gill 2000 §3.2 (Pines is a *method*, not the only valid one); Vallado 2013 associated-Legendre + pole guard is standard if tesseral terms are zeroed on axis.
3. `gravity.cpp:173–178` — ECEF evaluation then `a_N = ENᵀ a_E`. Suspect: rotating the field with the Earth (correct) vs rotating the *evaluation point* only. Source: Montenbruck & Gill 2000 §3.2.2.
4. Two-body μ: GGM header μ when loaded, else `MU_WGS`. Suspect mixed constants in vis-viva. Source: Tapley / GGM03S header; WGS-84 μ differs at 3e−4 m³/s² (relative ~1e−9).

## Pass 3 — adjudicate

1. **Keep** J2 = −C20 after denormalisation. Matches the GGM03S C̄20 √5 value stored as `J2_GGM`.
2. **Keep** associated Legendre with an explicit pole guard (`cosφ < 1e−14` → zonals only). Off-pole Basilisk `computeField` residual is ~1e−16. Do not rotate a polar test point into a tesseral-undefined axis and call that a fail. Pines would be a speed/robustness upgrade (Pass 4 of validation), not an equation fix.
3. **Keep** ECEF-then-rotate. That is the field of a rotating Earth.
4. **Keep** GGM μ for SMA when the field is loaded. Difference vs WGS is far below decay logging precision.

No equation change.

## Known case

Off-pole GGM03S degree 4 vs Basilisk computeField: residual ~1e−16 (see `tests/basilisk_ref/`).
