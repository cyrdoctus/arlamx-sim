# Test register — ARLAMX V2.1

time: 2026-08-18T22:00:00Z
agent: grok

Physics tests use a known input and a known output. Plots and numeric dumps
belong in `tests/outputs/<segment>/`. Program products stay in `outputs/`.

## test_sentman.py

- Source under test: `cpp/src/aero/sentman.cpp`
- Calls / called by: called by `cpp/src/api.cpp` plant; calls `constants.hpp`
- Physics: free-molecular Sentman GSI
- Method: incident Cp/Cτ plus M-01 re-emission
- Known case: s=80, θ=0.4, α_E=1 → Cp cosθ + Cτ sinθ = 2 cosθ ± 0.03; face-on unit square Cd in [1.8, 2.4]
- Citations:
  1. Sentman, L. H. (1961). Free Molecule Flow Theory… LMSC-448514, DTIC AD0265409.
  2. Moe, K. & Moe, M. M. (2005). Planet. Space Sci. 53(8), 793–801. (paywall — user must review)
  3. Doornbos, E. (2012). Thermospheric Density and Wind Determination. Springer.

## test_srp.py

- Source under test: `cpp/src/srp/panel_srp.cpp`
- Calls / called by: called by `api.cpp`; constants `P_SRP_1AU`, `CR_DEFAULT`
- Physics: panel solar radiation pressure
- Method: projected-area cannonball, sunward faces only
- Known case: 1 m² plate, Sun along n, Cr=1, ε=1 → |F| = 4.56e-6 N
- Citations:
  1. Montenbruck, O. & Gill, E. (2000). Satellite Orbits. Springer, §3.4.
  2. Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications, 4th ed., §8.6.4.

## test_gravity.py / test_gravity_vs_basilisk.py

- Source under test: `cpp/src/orbit/gravity.cpp`
- Calls / called by: called by `accel_nongrav_env` in `api.cpp`; uses `frames.cpp` GMST
- Physics: two-body + GGM03S Stokes field
- Method: unnormalized associated Legendre in ECEF, rotate by GMST
- Known case: circular 400 km two-body SMA hold; off-pole GGM03S vs Basilisk computeField ~1e-16
- Citations:
  1. Vallado 2013 §8.
  2. Montenbruck & Gill 2000 §3.2.
  3. Tapley et al. (2005) / CSR GGM03S header (coefficient authority).

## test_third_body.py

- Source under test: `cpp/src/orbit/third_body.cpp`
- Calls / called by: optional plant path `lunisolar`
- Physics: point-mass third body (direct + indirect)
- Method: μ (d/|d|³ − r_b/|r_b|³)
- Known case: analytic pair residual vs closed form
- Citations:
  1. Montenbruck & Gill 2000 §3.3.
  2. Vallado 2013 §8.6.

## test_mrp.py

- Source under test: `cpp/src/attitude/mrp.cpp`
- Calls / called by: plant kinematics; `mrp_feedback.cpp`
- Physics: modified Rodrigues parameters
- Method: Schaub DCM, B(σ) kinematics, shadow set |σ|>1
- Known case: 90° about z maps correctly; |σ|=1 shadow
- Citations:
  1. Schaub & Junkins (2018) Ch. 3–4.
  2. Shuster (1993). J. Astronaut. Sci. 41(4), 439–517.

## test_mrp_feedback.py / test_bdot.py

- Source under test: `cpp/src/control/mrp_feedback.cpp`, `bdot.cpp`
- Calls / called by: `Simulator::step` point / detumble modes
- Physics: MRP-PD + gyroscopic compensation; B-dot
- Method: τ = −kp σ_e − kd ω_e + ω×Iω; m = −k Ḃ
- Known case: error reduces for a rest-to-rest slew; B-dot reduces |ω|
- Citations:
  1. Schaub & Junkins (2018) Ch. 8.
  2. Stickler & Alfriend (1976) / standard B-dot (Wertz 1978).

## test_dipole.py

- Source under test: `cpp/src/mag/field.cpp`
- Physics: tilted dipole / WMM
- Known case: equatorial vs polar magnitude order
- Citations:
  1. Alken et al. (2021) WMM.
  2. Vallado 2013 §8.7.

## test_simplify.py / test_area_quality.py / test_solarcat_attitudes.py

- Source under test: `python/arlamx_v2/geometry.py`
- Physics: sealed-side simplification, ram area
- Known case: SolarCat 1% six-attitude ram-area error ≤ 1%
- Citations:
  1. Cohen-Steiner, Alliez & Desbrun (2004).
  2. Shewchuk constrained triangulation notes.

## test_srp_optical.py

- Source under test: `cpp/src/srp/panel_srp.cpp` (`panel_srp_optical`)
- Physics: plate SRP with ca, cs, cd
- Known case: 1 m² face-on absorber |F|=P; specular |F|=2P; Lambert |F|=P(1+2/3)
- Citations: Montenbruck & Gill 2000 §3.4; Vallado 2013 §8.6.4; McInnes 1999.

## test_lift_work.py

- Source under test: `cpp/src/api.cpp` corotating + energy split
- Physics: v_rel = v − ω⊕×r; dE_lift = F_⊥vrel · v
- Known case: corotation off → dE_lift = 0; on + AoA → nonzero; default corotating is true
- Citations: Vallado 2013 §8.6.2; King-Hele; Doornbos 2012.

## test_plant_step.py / test_prescribed_decay.py / test_env_smoke.py / test_advisors.py

- Source under test: `cpp/src/api.cpp`, `decay_run.py`, `env.py`, `advisors/`
- Known case: vacuum SMA hold; prescribed max-drag loses SMA and stays finite; infeasible ρ/T rejected; heuristic/MPC return unit quaternions
- Citations: Vallado 2013 vis-viva; Hairer / Nørsett / Wanner RK4; Rawlings, Mayne & Diehl 2017 (MPC structure).
