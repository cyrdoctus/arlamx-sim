time: 2026-08-18T22:30:00Z
agent: grok
style: detailed

# Min / max drag decay — 500 km → 250 km

## Done

Regenerated lifetime curves on the V2.0 plant after the previous 180-day
run wrote NaNs. The failure was not MSIS or GGM03S. It was MRP-PD integrating
a ~90° first slew for 180 s with unlimited torque (body rate ~29 rad/s, then
σ → NaN). Decay now uses `mode=prescribed`: the commanded quaternion sets
σ exactly and ω=0 each step.

## Updated

- `cpp/src/api.cpp` — prescribed mode, NaN guards, atmosphere / reset / step-size gates
- `python/arlamx_v2/decay_run.py` — 30 s prescribed steps, 10 s RK4, corotating wind, panel SRP, GGM03S degree 4, MSIS 2.1 every 600 s
- `outputs/decay/min_drag_decay.csv`, `max_drag_decay.csv`, `min_max_drag_decay.png`

## Case

| Quantity | Value |
|---|---|
| Geometry | SolarCat 1 % sealed-side mesh |
| Mass | 0.625 kg |
| Elements | a = Re+500 km, e=0.001, i=23°, Ω=ω=ν=0 |
| Epoch | JD 2461060.5 (2026-01-15) |
| Space weather | F10.7 = 150, Ap = 4, MSIS 2.1 |
| Wind | v_rel = v − ω⊕ × r |
| Gravity | GGM03S, n=m=4 |
| SRP | panel cannonball, Cr=1.8, cylindrical eclipse |
| Min-drag hold | body +Z along h = r×v (edge-on) |
| Max-drag hold | body +Z along v (face-on) |

Start |r|−Re = 493.12 km because ν=0 is perigee of the e=0.001 ellipse
(a(1−e)−Re). That is not a 7 km first-step decay.

## Outcome

| Hold | Time to 250 km | End SMA | End Cd | End ρ |
|---|---|---|---|---|
| min-drag | **18.81 d** | 6669.0 km | 0.39 | 4.27e−11 kg/m³ |
| max-drag | **10.85 d** | 6658.2 km | 1.97 | 4.32e−11 kg/m³ |

No NaN samples (904 / 522 rows). The |r| oscillation is the e=0.001 orbit
plus J2, growing as drag pumps eccentricity — the same shape as a Vallado
drag decay, not a plotting artefact.

Lifetime ratio max/min ≈ 1.73. The 1 % SolarCat is not a perfect flat plate,
so the ratio is modest; the *ordering* is the physical check (face-on dies
first). Density at 250 km under F10.7=150 / Ap=4 is ~4.3e−11 kg/m³, in the
MSIS 2.1 LEO band.

## Known case

Vacuum two-body SMA hold (`test_plant_step.py`) still passes. Prescribed
max-drag on a 0.3 m² plate loses SMA and stays finite
(`test_prescribed_decay.py`).
