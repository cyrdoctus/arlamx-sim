time: 2026-08-18T22:10:00Z
agent: grok
style: detailed

# Verification — panel SRP (`cpp/src/srp/panel_srp.cpp`)

## Connection map

- File owns: `panel_srp_force` / `panel_srp_force_torque`.
- Called by: `Simulator::step` when `use_panel_srp`.
- Calls: `P_SRP_1AU = 4.56e−6 Pa`, `CR_DEFAULT = 1.8`.
- Boundary: `sun_hat_B` body unit vector, `eclipse` ∈ {0,1}, force in body frame.

## Pass 1 — formatting and simplicity

One loop, sunward faces only. Names are obvious. No change.

## Pass 2 — assume it is wrong

1. `panel_srp.cpp:30` — `fi = P Cr A (n·s) (−s)`. Suspect: should be along `−n` (flat-plate specular) or a mix. Source: Montenbruck & Gill 2000 §3.4 cannonball / absorbed-radiation form used by V1.7; Vallado 2013 §8.6.4 writes both cannonball and plate. This project’s approved law is cannonball with Cr=1.8, not a BRDF.
2. `P_SRP_1AU` is the 1 AU value with no 1/r². Suspect: LEO r_sun varies ±3%. Source: Montenbruck §3.4 — 1 AU constant is the usual first-order LEO model.
3. Eclipse multiplies pressure, cylindrical umbra only (no penumbra). Source: Vallado cylindrical shadow, same as the plant eclipse flag.

## Pass 3 — adjudicate

1. **Keep** cannonball along `−s`. Matches the approved V2.0 module note and V1.7. A specular plate law would be a different model.
2. **Keep** constant 1 AU pressure. Higher-order flux is later work.
3. **Keep** cylindrical eclipse. Penumbra is a validation-pass-4 proposal, not a source mismatch.

Classroom case: 1 m², Cr=1, Sun along n → |F|=4.56e−6 N (`tests/srp/test_srp.py`).
