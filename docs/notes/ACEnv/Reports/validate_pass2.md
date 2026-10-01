time: 2026-08-18T22:30:00Z
agent: grok
style: detailed

# Validation pass 2 — hostile numerics

Public path only (`Simulator.reset`, `set_atmosphere`, `step`). Observation,
not redesign. Data: `tests/outputs/validate_pass2/hostile.json`.

| Case | What happened |
|---|---|
| reset r = 0 | exception: `reset \|r\| is inside the Earth` |
| step q = NaN | sanitized to identity; altitude finite (~400 km) |
| ρ < 0 | exception: infeasible atmosphere |
| T = 0 | exception: infeasible atmosphere |
| q = 0 | treated as a zero-norm quaternion (normalize path → identity); finite |
| RK4 h = 3600 s, prescribed | **finite but wrong** (altitude jumped to ~20 000 km) |

The last case is the dangerous one: a huge step does not crash, it lies.
Pass 3 must reject it at the constructor, not after a pretty number comes out.
