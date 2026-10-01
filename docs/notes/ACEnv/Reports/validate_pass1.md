time: 2026-08-18T22:30:00Z
agent: grok
style: detailed

# Validation pass 1 — communication and output

Drive the assembled program from the public commands. This is not a code reread.

## Commands exercised

| Command | Emits | Status |
|---|---|---|
| `python -m arlamx_v2.decay_run --geom stl1pct --kind min/max` | `outputs/decay/*.csv`, `min_max_drag_decay.png` | both holds reach 250 km, finite |
| `python -m arlamx_v2.advisor_run --geom stl1pct --days 2` | `outputs/advisors/*.csv`, `summary.json` | running / completed this session |
| Gym `ArlamxV2Env.reset/step` | 26-dim obs, reward, Cd, altitude | `test_env_smoke.py` pass |
| `Simulator.step` vacuum / drag | SMA, dE, Cd | `test_plant_step.py` pass |

## Connection map (boundary check only)

```
advisor / decay  --q_BN-->  Simulator.step
                     |         |-- GGM03S ECEF accel
                     |         |-- Sentman(v_gas = −C(v−ω⊕×r))
                     |         |-- panel SRP
                     |         |-- prescribed σ or MRP-PD
Python MSIS  --(ρ,T,m̄)-->  set_atmosphere
get_state  -- r,v,C_BN,sun_N,eclipse -->  policies
```

Units at the boundary: r,v in m, m/s; ρ kg/m³; T K; quaternions scalar-first.
Frames: r,v inertial; aero/SRP body; gravity ECEF then rotated by GMST.

## Output the README promised

Decay CSVs and the overlay plot exist under `outputs/decay/`. Advisor summaries
go under `outputs/advisors/`. Test artefacts stay in `tests/outputs/`.

Wiring is intact. Pass 2 may start.
