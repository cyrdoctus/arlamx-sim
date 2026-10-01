time: 2026-08-18T22:10:00Z
agent: grok
style: detailed

# Verification — co-rotating relative wind (`cpp/src/api.cpp`)

## Connection map

- Segment: `Simulator::step` lines that form `v_rel` and `v_B_gas`.
- Called by: every plant substep when `SimParams.corotating` is true (env, decay, advisors).
- Calls: `OMEGA_EARTH = 7.2921150e−5 rad/s` (GGM / WGS conventional).
- Boundary: inertial `r, v`; output is the **incident gas** velocity in the body frame.

## Pass 1 — formatting and simplicity

Four lines. The gas-frame minus sign is the only subtle name (`v_B_gas`). No change.

## Pass 2 — assume it is wrong

1. `api.cpp:145` — `v_rel = v − ω⊕ × r` with `ω⊕ = (0,0,Ω)`. Suspect: sign of the cross product, or ECEF vs inertial ω. Source: Vallado 2013 §8.6.2, relative wind for a co-rotating atmosphere.
2. `api.cpp:147` — `v_B_gas = C (v_rel) × (−1)`. Suspect: Sentman wants spacecraft velocity through the gas, not gas onto the craft. Source: Sentman 1961 uses the incident molecular velocity; the plant convention (V1.7 and this file) is gas onto the body, so `v_gas = −v_sc_rel`.
3. `ω⊕` is aligned with inertial +Z, not the CIP / polar motion. Source: Vallado first-order model; polar motion is ~0.3 arcsec and is out of scope.

## Pass 3 — adjudicate

1. **Keep** `v − ω×r`. That is the inertial velocity of the atmosphere at `r` for a z-aligned rotator.
2. **Keep** the minus into Sentman. The classroom face-on test uses `v_gas = (0,0,−7500)` onto +Z.
3. **Keep** z-aligned Ω. Polar motion is not in the source pair for this segment.

Binding: `SimParams.corotating` is exposed and default-on in `env.py` and `decay_run.py`.
