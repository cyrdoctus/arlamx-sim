time: 2026-08-18T18:30:00Z
agent: grok

# Validation plan — ARLAMX V2.0

Drive the assembled plant from public Python commands (`Simulator.step`, `validate_attitudes`, decay runner). Four passes in `ACEnv/Reports/`.

| Pass | Question |
|---|---|
| 1 | Main path emits state, Cd, eclipse, energy |
| 2 | Hostile numerics (zero q, T≤0, r=0) |
| 3 | Feasibility gates at the Python/C++ boundary |
| 4 | Speed/precision proposal only |

Known system case: circular 400 km two-body SMA hold; GGM03S vs Basilisk `computeField`; SolarCat 1% six-attitude ram area.
