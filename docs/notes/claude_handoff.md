# Claude handoff — ARLAMX V2.0

**Date:** 2026-08-18  
**Tree:** `ARLAMX V2.0/` only. **`ARLAMX-V1.7/` was not edited, copied, or moved.**

Read this first, then `APPROVAL_PLAN.md` and `README.md`.

---

## 1. What was built (the approach)

V2.0 is a **new implementation**, not a Basilisk wrap and not a paste of V1.7 Python.

| Layer | Role |
|---|---|
| **C++ plant** `cpp/` | Sentman FMF (M-01), panel SRP, GGM03S harmonics, optional SPICE 3rd-body, MRP + RK4, MRP-PD, B-dot, dipole/WMM. Folders: `aero / srp / orbit / attitude / control / mag`. |
| **Python** `python/arlamx_v2/` | pybind11 module `arlamx_cpp`, Gym env, SC_v3 reward, PPO trainer, **sealed-side** STL/OBJ simplifier. |
| **Basilisk** | **Referee only.** The plant never imports it. `tests/basilisk_ref/` compares our GGM03S field to Basilisk `computeField` when Basilisk is installed. |
| **V1.7** | Read-only reference (equations, YAML *numbers*, `.geom`/STL *data*, published SC_v3a recipe). |

Training talks to a small C++ command surface: `create / set_panels / reset / step`.

Build: `./build.sh` (ThesisMS Python). Tests: `PYTHONPATH=python python -m pytest tests -q` — **35 passed**.

---

## 2. Plots that already existed vs what is new

**Before this handoff, V2.0 had no PNG figures.** Training only wrote TensorBoard event files:

- `outputs/sc_v3_ppo/logs/PPO_1/events.out.tfevents.*`
- `outputs/sc_v3_ppo_smoke/logs/PPO_1/events.out.tfevents.*`

Those are not pictures. Open with TensorBoard if you want the learning curves.

**New geometry figures** (2026-08-18) live in `outputs/geometry/`:

| File | What it shows |
|---|---|
| `compare_cube.png` | Cube: 12 triangles → **6 sides**, area 6=6, unmatched **0** |
| `compare_hex_prism.png` | Hex prism: 20 triangles → **8 sides**, area conserved, unmatched **0** |
| `compare_solarcat_stl.png` | Real CAD: 4604 triangles → 128 sides / 482 plates. Hex sail shape is kept. Area 1.62→1.36 m², unmatched 1.46 m — this STL is not a closed shell (frame bits), so some cracks remain. |
| `compare_catsat_stl.SKIP.txt` | Full CatSat CAD is ~2.5 M triangles; default skip. `--catsat` to try. |

Each figure is two columns × two viewpoints: **left = input triangle soup**, **right = V2.0 plates** (one colour per side).

Cube and hex are the clean “no holes” proof. SolarCat is the real STL demo (same role as V1.7’s STL-vs-`.geom` card).

Regenerate:

```bash
cd "ARLAMX V2.0"
PYTHONPATH=python python -m arlamx_v2.plot_simplify
```

V1.7’s comparable pictures are still in V1.7 (`output/geometry_viz/`, `scripts/plot_geometry_abstraction.py`). Those draw `.geom` facets as **squares on centroids** (the file has no vertices). V2.0 draws the **actual simplified triangles** the plant would integrate.

---

## 3. How STL is handled (vs V1.7)

**V1.7:** `stl_loader` turns **every CAD triangle** into a Sentman panel. A CatSat STL is millions of facets; production runs used hand `.geom` files instead.

**V2.0:** `python/arlamx_v2/geometry.py` with a **quality toggle**

| `quality=` | Budget | Use |
|---|---|---|
| `1pct` | 1 % wetted **and** ram area | preferred, data generation |
| `5pct` | 5 % | intermediate |
| `10pct` | 10 % | crude smoke only |

Each side keeps the **original triangle area** (no silent membrane loss). Six SolarCat attitudes are checked: side-flat, side-vertex, ±Z max-drag, 45° about X, 45° about Y, 0→1° AoA growth. Runner: `python -m arlamx_v2.validate_attitudes --quality 1pct`. 2026-08-18: 1% is **0.000%** on all six vs the STL.

**V2.0 (cont.):** `python/arlamx_v2/geometry.py`

1. Load STL/OBJ (trimesh), units → metres.  
2. Region-grow triangles whose normals agree (default 12°) → **sides**.  
3. Fit a plane per side.  
4. Snap shared edges onto the two-plane intersection (**no holes** between plates).  
5. Ear-clip each planar outline → a small triangle set.  
6. Push `(n, A, centroid)` into C++.

The SC_v3a bake-off still uses `earthcup_hex_v3.geom` as **numeric panel data** so the vehicle is not silently changed. The simplifier is the path for raw CAD.

---

## 4. SC_v3a training that already ran

`outputs/sc_v3_ppo/metrics.json`

- PPO 4×16, 400k steps, 32 envs, seed 42  
- **141 s wall** (~2830 steps/s)  
- Model: `outputs/sc_v3_ppo/models/ppo_sc_v3a.zip`  
- Notes: `outputs/sc_v3_ppo/COMPARISON.md`

Speed is the fair comparison. The V2.0 env uses the SC_v3 **equations** but not every V1.7 side system (full TinyGS sticky pointing, v17 power). Do **not** quote the 400k rollout reward as matching V1.7’s published +159 eval.

---

## 5. Known gaps (do not hide these)

- Exact Earth **polar axis** in spherical-harmonic gravity still uses a zonal-only guard. Off-axis vs Basilisk is ~1e-16. A Pines evaluator is the proper fix.  
- WMM implementation is “loads + LEO sanity,” not a bit-match to NOAA’s reference code.  
- CatSat full CAD (2.5M triangles) is decimated **for the figure only**; the plant should get a pre-simplified panel set, not 2.5M live facets.  
- Geometry figures are matplotlib 3D (static PNG), not an interactive viewer.

---

## 6. What a reviewer should do

1. Confirm V1.7 is untouched.  
2. `./build.sh` then `pytest tests -q` (35).  
3. Open `outputs/geometry/compare_*.png` — left CAD, right sealed sides.  
4. Treat Basilisk as a referee in `tests/basilisk_ref/` only.  
5. Do not retrain to “fix” a failed plant test.
