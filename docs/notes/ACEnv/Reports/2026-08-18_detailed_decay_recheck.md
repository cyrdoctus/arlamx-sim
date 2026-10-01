time: 2026-08-19T00:10:00Z
agent: grok
style: detailed

# Decay recheck — why 19 days was not min-drag

## What was wrong

Two independent bugs made both holds look like a thick body.

1. **Quaternion conjugate.** `quat_from_dcm` used the active/graphics
   antisymmetric order (`C32−C23`). V1.7 already documented that this
   yields the inverse of `q_BN` (Schaub passive). The plant then flew
   +Z about 44° from the wind instead of the thin edge. Ram area at
   “min-drag” was **0.56 m²**, not 0.021 m².

2. **Wrong axis.** V1.7 min-drag puts the *smallest projected-area body
   axis along velocity* (hex: **+Y**, ram **0.0208 m²** ≈ 12 mm beams on
   a ~0.5–0.8 m span, the 36 mm-class side profile). V2 had put +Z along
   the orbit normal.

A third numerical issue: Sentman skipped only `cos θ ≤ 0`. After MRP
round-trip, the sail’s −Z faces had `cos θ ≈ 3e−17` and were treated as
front faces, so the 0.65 m² membrane contributed glancing shear. Skip is
now `cos θ ≤ 1e−12`. Hex edge-on **Cd = 0.070** (was 0.16; thesis v3
edge-on is 0.064). Face-on **Cd = 2.25** (thesis 2.04).

## What “min-drag” is now

| | Hex (EarthCup v3) | 1 % SolarCat STL |
|---|---|---|
| Min axis | +Y | +Y |
| Min ram | **0.0208 m²** | 0.0580 m² |
| Max ram (+Z sail) | 0.655 m² | 0.747 m² |
| Edge-on Cd | 0.070 | 0.24 |
| Face-on Cd | 2.25 | 2.19 |

RK4 / control step is **5 s** as requested, prescribed attitude so the
min-area axis tracks **v** (0.33°/step).

Historical v3 min-drag **107 d** / v4 **59 d** used this hex edge, not
the 1 % STL (the STL edge is ~3× thicker). Regeneration is on **hex**.

## Regenerated 500 → 250 km (hex, RK4 = 5 s, corotating + SRP, F10.7=150, Ap=4)

| Hold | Days | Start/end Cd | Ram |
|---|---|---|---|
| min-drag (+Y along **v**) | **103.4** | 0.070 → 0.137 | 0.0208 m² held |
| max-drag (+Z along **v**) | **7.99** | 2.26 → 2.25 | 0.655 m² held |

103 days sits on the v3 hex figure of **107 days**. The old 18.8 / 10.8 day
pair is obsolete.

Plot: `outputs/decay/min_max_drag_decay.png`.

## Known case after the fix

Hex, 500 km, F10.7=150, Ap=4, one plant step: min Cd 0.070, max Cd 2.25,
min-axis to **v** = 0.00°. Tests in `test_prescribed_decay.py`.
