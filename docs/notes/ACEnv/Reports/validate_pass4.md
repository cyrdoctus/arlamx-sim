time: 2026-08-18T22:35:00Z
agent: grok
style: detailed

# Validation pass 4 — speed and precision (proposal only)

Do not implement these unless asked.

## Multicore

Independent work: MSIS queries, Sentman per panel, GGM03S per RK4 stage,
advisor-campaign policies, decay min vs max.

- Decay min/max already run as two processes.
- Campaign policies are independent — a process pool sized from
  `ACEnv/system/<host>.json` (or 80 % RAM) is the honest next step.
- Do **not** thread the 12-state RK4 of one vehicle. The state is sequential.

## GPU

Not worth it for degree-4 harmonics and O(10²) panels. A GPU Sentman kernel
only pays at CatSat-scale triangle counts, which this project already refuses
to run in the 1 % path.

## Precision

| Quantity | FP32? | Reason |
|---|---|---|
| GGM03S recurrence / vis-viva SMA | **FP64 required** | μ/r and (Re/r)^n lose LEO metres in FP32 |
| Sentman Cp, Cτ | FP32 tolerable | residuals already 1e−3 class |
| MRP kinematics | FP64 preferred | shadow set and 180° composition |
| MSIS ρ | input is ~1e−12; keep FP64 in the plant |

Recommendation: keep the C++ plant in FP64. Python reward/obs may stay FP32
as they already do for Gym.

## Other speed-ups (advice)

1. Persistent associated-Legendre workspace in `GravityHarmonics` (the per-call
   `vector<vector<double>>` is the hottest allocation).
2. Pines / Cunningham if polar orbits become a training envelope.
3. MSIS cache keyed on (lat, lon, alt, 10 min) — already sampled at 600 s in decay.
4. Prescribed mode for any study that does not need ADCS transients.
