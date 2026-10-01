time: 2026-08-18T18:30:00Z
agent: grok

# Verification plan — ARLAMX V2.0

Segments (small, in this order). Each gets Pass 1–3 in `ACEnv/Reports/`.

1. Sentman GSI + panel force (`cpp/src/aero/sentman.cpp`)
2. Earth gravity GGM03S (`cpp/src/orbit/gravity.cpp`)
3. Panel SRP (`cpp/src/srp/panel_srp.cpp`)
4. Co-rotating relative wind (`cpp/src/api.cpp` v_rel path)
5. MRP kinematics (`cpp/src/attitude/mrp.cpp`)

Sources stay those already listed in `docs/modules/01`–`06` (Sentman 1961 + Moe & Moe 2005; Vallado 2013 + Montenbruck & Gill 2000; Schaub & Junkins 2018 + Shuster 1993). Basilisk is a testing referee only, not a source.

Write-ups (2026-08-18): `ACEnv/Reports/verify_sentman.md`, `verify_gravity.md`, `verify_srp.md`, `verify_corotating.md`, `verify_mrp.md`.
