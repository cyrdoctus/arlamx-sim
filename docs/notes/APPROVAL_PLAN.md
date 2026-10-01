# ARLAMX V2.0 — Approval Plan

**Status:** approved 2026-08-18. Implementation in progress.
**Date:** 2026-08-18
**Basilisk policy (D11):** the plant and trainer never import or link Basilisk. Optional `tests/basilisk_ref/` may call a local Basilisk install as an independent numeric referee (GGM03S `computeField`, SPICE Sun, WMM). If Basilisk is absent those tests skip.
**Authority:** this document supersedes the earlier `ARLAMX_V2.0_OVERVIEW_AND_CPP_PLAN.md` wherever they disagree. That file stays as the V1.7 survey. This file is the build contract.

**Hard constraints you already set**

- Do **not** touch `../ARLAMX-V1.7/` — review only. No delete, no move, no copy of its source into V2.0.
- Simulation lives in **one C++ library**, organized by function.
- Training / PyTorch / SB3 stay in **Python** and call the library.
- Every module has a short markdown spec, a repeatable analytic test, and **≥2 literature citations**.
- Tests live under `tests/`. Training artifacts live under `outputs/`.
- After the plant is proven, train the **SC_v3 PPO** setup and compare to V1.7.

Nothing below is implemented until you approve the decisions in §11.

---

## 1. What V2.0 is (one sentence)

A new, independently written C++ spacecraft plant that does the V1.7 job — Sentman drag in the force model, panel SRP, Earth gravity with the same spherical-harmonic field Basilisk uses, SPICE Sun/Moon third-body, MRP attitude, MRP-PD and B-dot — called from a Python Gym / PyTorch training loop, plus a watertight CAD simplifier for STL/OBJ.

It is **not** a Basilisk wrapper, **not** a copy of the V1.7 Python tree, and **not** a rewrite of PPO.

---

## 2. Behavioral contract (what “same as V1.7” means)

V2.0 must reproduce the **scientific loop**, not the V1.7 file layout.

```
Python: load config + geometry → Gym step(action)
   │
   ▼
C++ plant, one advisor step of n_substeps:
   action (scalar-first quaternion) → slew-clip → MRP target
   each dt:
      SPICE Earth/Sun/Moon states
      gravity: two-body + GGM03S harmonics (degree/order configurable)
      + third-body Sun/Moon (if enabled)
      + Sentman FMF force/torque from panels and MSIS
      + panel SRP (if enabled)
      + MRP-PD torque  or  B-dot dipole × B
      integrate orbit + attitude (RK4, ZOH body loads, DCM at each stage)
   return state + accumulators (ΔE, Cd, Cl, eclipse, sun_B, B_B, …)
   │
   ▼
Python: sensor noise → observation → composite reward → SB3
```

Units SI. Frames: **N** = GCRS/J2000 inertial, **B** = body, **E** = Earth-fixed (ITRF-class) for the harmonic field. The V1.7 M-01 Sentman closure is the aero law (not the pre-audit formula).

**Same as V1.7 for the SC_v3 comparison run** (flags taken from `config_SC_v3a.yaml`, read-only):

| Flag | Value | Why |
|---|---|---|
| Geometry | `earthcup_hex_v3.geom` panels as **data** | already a clean 72-panel hex; do not re-simplify it for the bake-off |
| `alpha_E` | 0.93 | production GSI |
| `gsi_model` | Sentman, M-01 closure | V1.7 live code |
| `corotating_atmosphere` | false | all V1.7 configs |
| `use_panel_srp` | true | env-injected panel SRP |
| `enable_srp` (cannonball) | false | avoid double-count |
| `spherical_harmonics_degree/order` | 4 / 4 | SC_v3 plant (library still supports 20) |
| `enable_lunisolar` | false | SC_v3 plant (library still supports SPICE 3rd body) |
| `dt_s` / `advisor_step_s` | 2 s / 300 s | 150 substeps |
| PPO | 4×16, 400k steps, 32 envs, seed 42, `ent_coef` 0.003 | SC_v3 |
| Reward | SC_v3 terms (exponential `dE`, SoC-modulated power, `gs_alignment`) | see `docs/modules/12_python_training.md` |

The library is **more capable** than that comparison run. Extra capability stays behind flags so a bake-off is not an accidental physics change.

---

## 3. What is in C++ vs Python

| In C++ (`libarlamx`) | In Python |
|---|---|
| Sentman FMF + whole-body Cd/Cl | YAML / config |
| Panel SRP force (and moment) | STL / OBJ load + **side simplification** |
| GGM03S spherical-harmonic gravity | Gymnasium `ArlamxEnv` wrapper |
| SPICE Earth / Sun / Moon + 3rd-body | Sensor noise |
| Two-body, J2-only fallback, eclipse | Observation registry |
| MRP kinematics, DCM, shadow set | Composite reward (SC_v3 terms) |
| MRP-PD inner controller | SB3 / PyTorch PPO (and SAC/TD3 later) |
| B-dot detumble + dipole × B torque | Provenance, TensorBoard |
| Magnetic field (dipole + WMM file) | ONNX export of the **actor** |
| RK4 12-state fused advisor step | Training CLI, comparison report |
| Atmosphere query **into** the plant (NRLMSIS C, or a Python callback in Phase 1) | Plotting |

Basilisk is **not** a V2.0 dependency. V1.7 remains the independent reference for numeric gates.

---

## 4. C++ tree (function, not a dump)

Implementation, when approved, is written **new** under `cpp/`. Names below are the layout I will use so a reviewer can find one topic in one folder.

```
ARLAMX V2.0/
├── APPROVAL_PLAN.md          ← this file
├── README.md
├── docs/modules/             ← one short spec per module (already drafted)
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/arlamx/
│   │   ├── types.hpp         # Vec3, Mat3, PanelSoA, State12, Atmo
│   │   ├── aero/sentman.hpp
│   │   ├── srp/panel_srp.hpp
│   │   ├── orbit/gravity.hpp        # two-body + harmonics
│   │   ├── orbit/third_body.hpp     # SPICE + Sun/Moon
│   │   ├── orbit/atmosphere.hpp
│   │   ├── attitude/mrp.hpp
│   │   ├── attitude/integrate.hpp   # Euler + RK4
│   │   ├── control/mrp_feedback.hpp
│   │   ├── control/bdot.hpp
│   │   ├── mag/field.hpp            # dipole + WMM
│   │   └── api.hpp                  # fused step + handles
│   └── src/                         # matching .cpp, same split
├── python/
│   ├── arlamx_v2/                   # bindings + Gym env + train
│   └── geometry/                    # STL/OBJ + sealed-side simplifier
├── tests/                           # one subfolder per module
│   ├── aero/
│   ├── srp/
│   ├── orbit/
│   ├── attitude/
│   ├── control/
│   ├── mag/
│   ├── geometry/
│   └── integration/
└── outputs/                         # training only (never tests)
```

Rules for the C++:

- No giant `sim.cpp`. One topic per header/source pair.
- No V1.7 Python pasted into comments as “the code”. Equations + citations live in `docs/modules/`.
- `float64` for all identity tests.
- One `Simulator` handle per env (re-entrant; 32 `SubprocVecEnv` workers).
- Public Python surface is a **small command set** (create / set_geometry / set_params / reset / step / get_state), not 80 free functions.

---

## 5. Force model (how the pieces add)

Inertial acceleration of the spacecraft, Earth-centered:

\[
\mathbf{a}_N
  = \mathbf{a}_{\mathrm{2-body}}
  + \mathbf{a}_{\mathrm{SH}}
  + \mathbf{a}_{\mathrm{3B}}
  + C_{BN}^{\top}(\mathbf{F}_{\mathrm{FMF}}+\mathbf{F}_{\mathrm{SRP}})/m
\]

Body torque:

\[
\boldsymbol{\tau}_B
  = \boldsymbol{\tau}_{\mathrm{FMF}}
  + \boldsymbol{\tau}_{\mathrm{SRP}}
  + \boldsymbol{\tau}_{\mathrm{ctrl}}
  + \boldsymbol{\tau}_{\mathrm{gg}}\ \text{(optional)}
\]

\(\boldsymbol{\tau}_{\mathrm{ctrl}}\) is either MRP-PD, or \(\mathbf{m}\times\mathbf{B}_B\) from B-dot, selected by mode (`detumble` vs `point`).

Sentman is **not** a sidecar. It is a first-class term in \(\mathbf{a}_N\) and \(\boldsymbol{\tau}_B\) every substep, same injection idea as V1.7’s `ext_force_B` but evaluated inside the C++ plant.

Spherical harmonics are evaluated in the Earth-fixed frame (GGM03S is body-fixed to Earth) and rotated to N. That is why SPICE (or an equivalent Earth-orientation path) is required even when third-body gravity is off.

---

## 6. Module list and ownership

Each module already has a short spec in `docs/modules/`:

| # | Module | C++ folder | Spec |
|---|---|---|---|
| 01 | Sentman FMF | `aero/` | `docs/modules/01_aero_sentman.md` |
| 02 | Panel SRP | `srp/` | `docs/modules/02_srp.md` |
| 03 | Gravity + harmonics | `orbit/` | `docs/modules/03_orbit_gravity.md` |
| 04 | SPICE + third body | `orbit/` | `docs/modules/04_orbit_third_body.md` |
| 05 | Atmosphere (MSIS) | `orbit/` | `docs/modules/05_atmosphere.md` |
| 06 | MRP + Euler + RK4 | `attitude/` | `docs/modules/06_attitude_dynamics.md` |
| 07 | MRP-PD | `control/` | `docs/modules/07_control_mrp_feedback.md` |
| 08 | B-dot | `control/` | `docs/modules/08_control_bdot.md` |
| 09 | Magnetic field | `mag/` | `docs/modules/09_magnetic_field.md` |
| 10 | Geometry simplifier | Python `geometry/` | `docs/modules/10_geometry_simplify.md` |
| 11 | Fused plant API | `api` | `docs/modules/11_plant_api.md` |
| 12 | Python training | `python/` | `docs/modules/12_python_training.md` |

Validation standard for every module: `docs/VALIDATION_STANDARD.md`.

---

## 7. Geometry: sides, not raw CAD facets

V1.7 `stl_loader.py` turns **every triangle** into a Sentman panel. A deployed CatSat STL is then hundreds to thousands of facets, many mutually shadowing, and the live RL loop still has **no** self-shadowing.

V2.0 does this in Python **before** the plant sees geometry:

1. Load STL or OBJ (trimesh / numpy-stl). Units → metres.
2. Weld vertices, drop degenerate faces, consistent outward winding.
3. **Region-grow sides:** adjacent triangles whose normals agree within \(\theta_{\mathrm{merge}}\) (default 12°) become one side.
4. Fit a least-squares plane per side (this is variational shape approximation, not mesh decimation).
5. Extract the side’s outer boundary (and holes, if any).
6. **Seal seams:** snap shared-boundary vertices onto the intersection line of the two neighboring planes so adjacent plates meet. No gaps, no T-junctions.
7. Constrained Delaunay triangulation of each planar polygon → a **small** triangle set per side.
8. Emit Panel SoA `(n[3N], A[N], c[3N])` into C++.
9. Report: side count, area change vs raw mesh, maximum gap, watertight/sealed-edge check.

The spacecraft still looks like the spacecraft (hex sail stays a hex, 6U stays a box + wing). It is not a convex hull and not “delete every other triangle.”

**SC_v3 bake-off:** skip this and load `earthcup_hex_v3.geom` as numeric panel data so the plant is not a different vehicle. Simplifier is proven on a cube, a closed hex prism, and one CatSat/SolarCat STL — under `tests/geometry/`, not mixed into the PPO comparison.

---

## 8. Python command surface

Training never pokes C++ internals. The binding is intentionally small:

| Command | Meaning |
|---|---|
| `create(config_dict)` | allocate a `Simulator` |
| `set_panels(n, A, c)` | push SoA geometry |
| `set_mass` / `set_inertia` | domain randomization |
| `set_space_weather(f107, ap)` | per-episode MSIS |
| `reset(orbit, sigma0, omega0)` | new episode |
| `step(q_target4)` | one advisor step; returns state + accumulators |
| `get_state()` | last state dict (Gym keys, same names as V1.7 so reward/obs stay readable) |
| `set_mode("point"\|"detumble")` | MRP-PD vs B-dot |

`step` is **Depth B** (fused). Individual kernels (`sentman`, `gravity`, `mrp_feedback`) are also bound, but only so `tests/` can call them. Training does not use the kernel API.

---

## 9. Tests and outputs

```
tests/<module>/test_*.py     # pytest, calls C++ kernels or Python geometry
tests/<module>/cases/        # tiny analytic inputs (no big CAD)
tests/integration/           # one advisor step vs published numbers / V1.7 oracles

outputs/sc_v3_ppo/           # the comparison training run only
outputs/sc_v3_ppo/models/
outputs/sc_v3_ppo/logs/
outputs/sc_v3_ppo/metrics.json
```

Tests **never** write into `outputs/`. Training **never** writes into `tests/`.

Each module test is a **classroom problem**: one number you can check with a calculator or a cited closed form. No “run a week of LEO and see if it looks ok.”

---

## 10. SC_v3 PPO comparison (after the plant is green)

**Recommended bake-off:** `SC_v3a` — PPO 4×16, 400k steps, 32 envs, seed 42, standard weather box. It is the documented single-seed v3 recipe. `SC_v3b` is the better *campaign* model only because of `env_seed_mode: spawn`; that is a training-RNG difference, not a plant difference. Using v3a keeps the comparison honest.

Protocol (written new in `python/`, not copied from V1.7 `train.py`):

1. Same hyperparameters as `config_SC_v3a.yaml` (read as a **data** file, not imported as code).
2. Same reward equations (re-implemented in V2.0 Python from `docs/modules/12_python_training.md`).
3. Same 22+4 observation list, Gaussian sensor noise with the Earth Cup subsystem numbers.
4. Train V2.0 to 400k steps → `outputs/sc_v3_ppo/`.
5. Compare, on the **same** `scenarios/earthcup_400km.yaml` eval (read as data):

| Metric | Why |
|---|---|
| Wall-clock time / 1k steps | the point of C++ |
| Mean eval reward vs step | did it learn |
| Mean Cd, decay km/day, SoC, P_gen | physics + policy |
| GS alignment while visible | SC_v3 third objective |

We will **not** claim bit-identical training curves. PPO is stochastic. Success is: plant identity tests pass, and the v3a-style run produces a policy in the same performance band as the published V1.7 SC_v3a eval (reward recovery, SoC not brown-out, Cd not max-drag). If the plant gates fail, training is not started.

---

## 11. Decisions for you to approve

These are locked if you accept this plan. Change any of them now.

| ID | Decision | Recommendation |
|---|---|---|
| D1 | V1.7 is read-only. V2.0 is a new implementation from equations + citations. `.geom` / YAML / GGM03S / SPICE kernels / WMM.COF are **data**, not source copies. | Accept |
| D2 | One C++ library, folders `aero / srp / orbit / attitude / control / mag`. Python owns train + geometry simplify + Gym. | Accept |
| D3 | Harmonics = GGM03S, fully normalized, configurable degree/order, default **20** in the library, **4** in the SC_v3 bake-off. Formulation: non-singular Pines/Cunningham, ECEF then rotate to N. | Accept |
| D4 | Third body = point-mass Sun + Moon from CSPICE (`de430.bsp` or the Basilisk-bundled kernel as a **data** path). Off for the SC_v3 bake-off. | Accept |
| D5 | Sentman M-01 is inside the plant force/torque every substep. | Accept |
| D6 | MRP-PD is the pointing controller. B-dot is the detumble controller. Mode switch in C++. B-dot needs a B-field → dipole + optional WMM.COF (Basilisk uses WMM). | Accept |
| D7 | Geometry simplifier = planar-side clustering + sealed seams, not raw facets and not QEM holes. Not used on the hex bake-off. | Accept |
| D8 | Comparison model = **SC_v3a** (single seed 42). | Accept, or switch to SC_v3b |
| D9 | Atmosphere Phase 1 = pymsis callback into C++ (identical MSIS 2.1). Phase 2 = NRLMSIS C linked in-process. Bake-off may use Phase 1 so weather matches V1.7. | Accept |
| D10 | No implementation until you approve this file. | **Accepted 2026-08-18** |
| D11 | **Basilisk is a reference only.** V2.0 does not link, import, or run Basilisk in the plant or the trainer. Optional tests under `tests/basilisk_ref/` compare kernels to a Basilisk install *if present*; if Basilisk is missing those tests skip. V1.7 remains a second independent reference (read-only). | **Accepted 2026-08-18** |

---

## 12. Build order (only after approval)

1. Types + Sentman kernel + aero tests.
2. SRP kernel + tests.
3. Two-body + J2 + harmonics (degree 2 first, then GGM03S) + tests.
4. SPICE load + third-body + eclipse + tests.
5. MRP / Euler / RK4 + tests.
6. MRP-PD + B-dot + dipole/WMM + tests.
7. Fused `step` + Python bindings.
8. Atmosphere hook.
9. Geometry simplifier + tests (cube, hex prism, one STL).
10. Gym env + SC_v3 reward/obs (new Python).
11. SC_v3a training → `outputs/sc_v3_ppo/` + comparison note.

Each step closes its module markdown “Tests run” section with actual numbers. No step is “done” on a green compile alone.

---

## 13. What I will not do

- Edit, delete, move, or copy any file under `ARLAMX-V1.7/`.
- Vendor Basilisk or wrap `BasiliskInterface`.
- Re-derive a new GSI, a new reward, or a new PPO.
- Put training logs in `tests/` or tests in `outputs/`.
- Quietly turn on lunisolar, co-rotation, or CAD simplification during the bake-off.
- Generate C++ in this folder before you approve §11.

---

## 14. How Claude (or anyone) should review this

A reviewer should be able to:

1. Map every C++ folder to a `docs/modules/` file with two citations and a calculator test.
2. Confirm V1.7 is untouched (`git status` on that tree).
3. Confirm the SC_v3 bake-off flags match `config_SC_v3a.yaml` without importing that repo’s Python.
4. Confirm Sentman uses the M-01 closure, not \(1+\alpha_E(T_w/T_i-1)\).
5. Confirm the simplifier is required to report a **zero unmatched-boundary** (no holes between plates).

If any of those five fail, the plan is not being followed.
