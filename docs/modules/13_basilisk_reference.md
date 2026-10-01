# 13 — Basilisk as a reference (not a dependency)

**Folder:** `tests/basilisk_ref/`  
**Status:** implemented  
**Plant:** must not `#include` or `import Basilisk`

## What it does

Compares selected V2.0 kernels to an installed Basilisk, when present. This is a **referee**, not the simulator.

## What is compared

| Kernel | Basilisk referee | Gate |
|---|---|---|
| GGM03S field (ECEF) | `SphericalHarmonicsGravityModel.computeField` | 1e-8 relative on \(\mathbf{a}\) |
| Analytic vs SPICE Sun | Basilisk SPICE planet state (optional) | angle documented, not a plant fail |

Aero/Sentman, MRP-PD, B-dot, panel SRP have **no** Basilisk twin. Those stay on the classroom tests in modules 01–08.

## Citations

1. GGM03S / Tapley GRACE models (coefficient authority — same file Basilisk loads).
2. Vallado (2013) §8; Montenbruck & Gill (2000) §3.2 (the field we implement).
3. Basilisk, Autonomous Vehicle Systems Laboratory, University of Colorado Boulder, https://github.com/AVSLab/basilisk (ISC). `SphericalHarmonicsGravityModel` is the reference for `cpp/src/orbit/gravity.cpp`. Cite it with that segment. The plant does not link Basilisk.

## Classroom test

At \(\mathbf{r}_E=(R_E+400\,\mathrm{km},\,0,\,0)\), degree 4: \(\|\mathbf{a}_\mathrm{V2}-\mathbf{a}_\mathrm{BSK}\|/\|\mathbf{a}_\mathrm{BSK}\| < 10^{-8}\). Skip if `import Basilisk` fails.

## Tests run

2026-08-18 — degree 2/4/8 ECEF field vs Basilisk `SphericalHarmonicsGravityModel.computeField` on GGM03S: **PASS** (machine-precision off the polar axis).
