time: 2026-08-19T01:20:00Z
agent: grok
style: detailed

# Sail optical presets

Plate law (Montenbruck & Gill §3.4), ca+cs+cd=1:

F = −P A cosθ [(ca+cd) s + (2 cs cosθ + 2 cd/3) n]

Cannonball Cr=1.8 is a separate law (force only along −s).

## Face-on 1 m², P = 4.56e−6 Pa

| Preset | ca / cs / cd | |F| face [µN] | |F| 45° [µN] |
|---|---|---|---|
| cannonball_cr18 | (Cr=1.8) | 8.21 | 5.80 |
| absorber | 1 / 0 / 0 | 4.56 | 3.22 |
| specular | 0 / 1 / 0 | 9.12 | 4.56 |
| lambert | 0 / 0 / 1 | 7.60 | 4.83 |
| al_mylar | 0.08 / 0.88 / 0.04 | 8.78 | 4.54 |
| al_kapton | 0.12 / 0.80 / 0.08 | 8.48 | 4.51 |
| black_kapton | 0.92 / 0 / 0.08 | 4.80 | 3.35 |
| white_paint | 0.20 / 0.04 / 0.76 | 7.09 | 4.57 |
| aged_sail | 0.35 / 0.40 / 0.25 | 6.93 | 4.15 |

Specular face-on is 2P (photons reverse). Absorber is P. Aluminized
coatings sit near the old Cr=1.8 cannonball in *magnitude* at face-on,
but the **direction** is mostly along −n, not −s — that is the point
of the optical model for a sail.

Sweep command: `PYTHONPATH=python python -m arlamx_v2.optics_run`

Data: `outputs/optics/optics_forces.csv`, `optics_orbit.csv`, `optics_forces.png`.
