# 11 — Fixed-attitude decay, lift and SRP studies (September 2026)

No training. The v2.1 plant used as a propagator with prescribed attitude
(`mode = "prescribed"`, no PD, exact hold). Common environment unless stated:
start 500 km, i = 23°, e = 0.001, F10.7 150, Ap 4, MSIS 2.1 co-rotating,
GGM03S degree 4, panel SRP (Cr 1.8), dt 5 s, stop at 300 km.
Runners: `decay_run.py`, `aoa_decay.py`, `lift_drag_study.py`, `srp_assess.py`,
`srp_f107_sweep.py`. Results: `outputs/analysis/decay/`, `outputs/analysis/lift_drag/`.

## Geometries and masses (user-specified 2026-09-05)

| name | file | panels | face-on area | mass |
|---|---|---|---|---|
| "original hexagon" | `data/earthcup_hex_v3.geom` | 72 plates | 0.655 m² | 0.72 kg (campaign value 0.625) |
| "six-petal" | `data/SolarCat_Assembly.STL`, simplified `1pct` | 4580 panels | 0.75 m² | 1.75 kg |

## Lifetimes 500 → 300 km (`compare_hex_vs_stl_500to300_mass.csv`, `aoa45_500to300/summary.json`)

| attitude | hex 0.72 kg | six-petal 1.75 kg |
|---|---|---|
| edge-on, α = 0 (min drag) | 77.1 d (76.84 in the AoA run) | 116.0 d (115.72) |
| α = 45°, lift out-of-plane (+h) | 12.80 d | 26.43 d |
| α = 45°, lift radial (+r) | 13.06 d | 26.88 d |
| face-on, α = 90° (max drag) | 8.97 d | 19.15 d |

Equal-mass comparison (both 0.625 kg, `compare_hex_vs_stl_500to300.csv`):
hex 68.0 / 7.8 d, STL 45.6 / 6.8 d. Older v2.0 runs (250 km stop, pre-v2.1
Sentman fix): 18.81 d / 10.85 d on the 1 % SolarCat — not comparable.

## Free-molecular lift (`aoa45_500to300/README.md`, `lift_drag/stl1pct_i45_m1p75/`)

At 45° AoA the drag is 69 % of face-on; L/D 0.113 (hex) / 0.107 (STL); Cd
1.57 / 1.53, Cl 0.178 / 0.163; forces 22.4 / 2.56 µN at 491 km → 424 / 47 µN
at 300 km (hex). L/D maximum 0.150 at α = 15° (500 km), 0.142 at 300 km.

Findings: lift work is 0.01–0.04 % of drag work (SRP is 15–50× bigger);
radial lift moves the eccentricity vector ≤ 2.4e-4 (drag + diurnal bulge do
4e-3); a fixed-sign out-of-plane lift gives a secular Δi ≈ +0.010° per decay
through the density's first harmonic (sign depends on Sun–node phase), the
same size as the co-rotation drag effect (−0.010°). A half-orbit sign-flip
lift scheme has only been scored analytically (Gauss variational equations in
`lift_drag_study.py`), not simulated.

## SRP feasibility (`aoa45_500to300/{srp_assessment,srp_f107_sweep}/`)

Quasi-static, per-point attitude optimisation over a Fibonacci sphere of
normals (`srp_assess`), and three strategies (edge / max-Sun-push / best) vs
F10.7 20–200 in the 500–450 km band (`srp_f107_sweep`). Units: km/day of
semi-major axis, positive = raised.

- SRP orbit-mean 3.2–4.2 µN (peak 5.3 hex / 6.2 six-petal); net SRP work in
  the decay runs 0.4–0.8 % of drag work.
- Best attitude raises the orbit only above ~620–645 km (hex) / ~650–660 km
  (six-petal) at F10.7 150.
- The user's rule (edge-on in eclipse, max Sun push when lit) holds 500 km
  only for F10.7 ≤ ~60, the whole band for ≤ ~40, and is worse than edge-on
  above F10.7 ~75 (3.5× faster decay at 150).
- The best-net rule holds 500 km up to F10.7 ~80 (hex) / ~70 (six-petal),
  450 km only to ~50 / ~40; at F10.7 150 it extends band time by 15 % / 10 %.
- Observed F10.7 floor is ~65, so realistically only the hexagon can hold
  ~500 km at deep solar minimum.

Caveats: the default SRP model (Cr 1.8 along the Sun line) is optimistic for
edge-on sailing versus a specular membrane; the sail's optical coefficients
are unknown (`sail_optics.py` presets exist); no full-propagation validation
with a strategy policy has been run.

## What these studies are good for in a dissertation

They give clean, actuator-free physics numbers on the two vehicles: decay
envelopes, the smallness of lift and SRP relative to drag, and the F10.7
thresholds at which passive sailing can hold altitude. They are the honest
"what the environment can do" baseline that any active-control claim must be
compared against.
