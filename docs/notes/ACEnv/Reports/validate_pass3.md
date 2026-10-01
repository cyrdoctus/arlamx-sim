time: 2026-08-18T22:35:00Z
agent: grok
style: detailed

# Validation pass 3 — error handling and input feasibility

Using pass 2 as evidence, gates were installed at the plant boundary.

| Gate | Rule | Source of the bound |
|---|---|---|
| `reset` | finite r,v,σ,ω; \|r\| ≥ ½ Re | Vallado: a state inside the Earth is not a LEO IC |
| `set_atmosphere` | ρ ≥ 0, T > 0, m̄ > 0, all finite | Sentman 1961 needs T>0 and m̄>0 for the speed ratio |
| `set_mode` | `point` \| `detumble` \| `prescribed` | API contract |
| `SimParams` | mass>0; dt∈(0,300] s; rk4∈(0,120] s; advisor∈(0,3600] s | RK4 orbit accuracy; 3600 s single-stage was pass 2’s liar |
| `step` q | non-finite → identity | keep the plant running; the *policy* is scored elsewhere |
| Gym `reset(options)` | altitude_km ∈ (250, 2000) if supplied | LEO envelope |

`test_prescribed_decay.py` now asserts the atmosphere, interior-Earth, and
3600 s RK4 rejections.

After this pass, a run that started was judged feasible for the physics in use.
