time: 2026-08-18T18:30:00Z
agent: grok
style: version

- AstroChimera ACEnv opened on ARLAMX V2.0 (single-advisor).
- SimParams.corotating bound to Python.
- Decay, heuristic family, sampling MPC, and V&V reports added this session.

time: 2026-08-18T22:45:00Z
agent: grok
style: version

- Bug found: decay NaNs from MRP-PD integrating a large slew over 180 s with unlimited torque.
- Bug fixed: `mode=prescribed` holds σ exactly; dt/rk4/atmosphere/reset gates added.
- Major: min/max drag 500→250 km regenerated (18.81 d / 10.85 d, 1 % SolarCat).
- Major: heuristic bank expanded (13 + search floors + off-nadir + MPC).
- Gym env now defaults `corotating=True`.

time: 2026-08-19T00:20:00Z
agent: grok
style: version

- Bug found: min-drag used the conjugate of q_BN and pointed +Z ~44° from the wind (ram 0.56 m²).
- Bug found: Sentman treated 1e-17-grazing sail faces as front faces (Cd_edge 0.16 instead of 0.07).
- Bug fixed: V1.7 passive quaternion + geometry min-area axis along v; cosθ skip 1e-12.
- Decay regenerated, RK4 5 s, hex: min 103.4 d, max 8.0 d (500→250 km).

time: 2026-08-19T01:05:00Z
agent: grok
style: version

- SimParams.corotating default is now true; step reports dE_drag and dE_lift (F_⊥vrel · v).
- Optical plate SRP added (ca, cs, cd) beside cannonball Cr. Presets in sail_optics.py.
- Equation register written. CLL/CCL FMF is a plan only.

time: 2026-08-19T02:20:00Z
agent: grok
style: version

- SC_v4a/v4b reward + env (band, GS tiers, feasibility, weather jumps, brownout) on the V2.0 plant.
- Trained the V1.7 six-run matrix (PPO/SAC 300k + PPO 50k). Eval vs heuristic and MPC.

time: 2026-08-20T06:58:21Z
agent: grok
style: version

- Independent five-pass audit of V2.0. No code changed.
- Bugs found (not fixed): Gym B-field is untilted +Z dipole, opposite Earth polarity, WMM unused; analytic Moon ecliptic used as equatorial; GGM evaluator is not Pines and misses 1e-8 at the exact pole; MSIS held 300 s; corotating default true vs V1.7 SC_v3 false; altitude_km is |r|-Re_eq while MSIS is geodetic; SRP torque computed and dropped; NaN quaternion silently identity.
- Unproven: Moe & Moe 2005 PSS typeset Tr/Ti (paywall); V1.7 wall-clock speedup factor (V1.7 not timed).
- Tests this audit: 74 passed, including Basilisk GGM referee off-pole ~1e-16.
- Report: ACEnv/Reports/2026-08-20_detailed_independent_v20_physics_audit.md.

