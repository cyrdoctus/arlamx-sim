# ARLAMX v2.1 — change notes (2026-09-04)

Scope: an equation and performance audit of the v2.0 plant and trainer. No new
features, no reward or hyper-parameter changes. Every item below is either a
correctness fix or a speed-up that leaves the physics at rounding level.
Test suite: 159 passed, 3 skipped (Basilisk referee, no Basilisk on this
machine). Independent cross-check of every kernel: `tests/validation/verify_physics.py`
(all pass).

Read section 1 before comparing v2.1 numbers with the v2.0 campaign
artifacts under `outputs/`: the two physics fixes change the min-drag
baseline and therefore every "gap vs MPC" and lifetime figure.

---

## 1. Physics corrections (results change)

### 1.1 Sentman panel loop dropped the thermal-spread flux on lee and edge-on faces — CRITICAL

`cpp/src/aero/sentman.cpp`, `sum_panels`.

v2.0 summed only faces with `cos θ > 0`. Sentman's closure is valid for all
θ (the erf/exp terms in γ = s cos θ switch the flux off smoothly on the lee
side) and gives an *exactly edge-on* face a non-zero free-molecular skin
friction C_τ = 1/(s√π) ≈ 0.07 per face at s ≈ 8 (derivable directly from
the Maxwellian number flux n·v̄/4 carrying tangential momentum m·V). The
cut-off therefore

* made the force discontinuous at grazing incidence (0.079 → 0.148 on the
  hex sail between 0° and 0.05° tilt), and
* zeroed **both** faces of the sail membrane whenever it was exactly edge-on,
  which is precisely the plant's own min-drag counterfactual
  (`coefficients_only` with an axis-aligned freestream).

Measured on `earthcup_hex_v3.geom` at s = 7.93 (7670 m/s, 900 K, atomic O):

| flow tilt from the in-plane axis | v2.0 Cd | v2.1 Cd | ratio |
|---|---|---|---|
| 0° (the reward baseline) | 0.079 | 0.214 | 2.70 |
| 0.5° | 0.156 | 0.215 | 1.38 |
| 2° | 0.186 | 0.225 | 1.21 |
| 4° (the co-rotation offset a min-drag policy flies at) | 0.235 | 0.256 | 1.09 |
| 8° | 0.355 | 0.359 | 1.01 |
| ≥ 15° and face-on | unchanged | unchanged | 1.00 |

Fix: sum every face with γ > −4 (beyond that a face contributes < 1e-7 of a
face-on panel). Coincident two-sided membranes do not double count: at
face-on incidence the back face has γ = −8 and is skipped.

Consequences: `dE_baseline` (the min-drag counterfactual in
`dE_vs_baseline`) is larger in magnitude by up to 2.7×; min-drag lifetimes
from `decay_run` / `mc_decay` shorten; MPC's `cd` cost term (it calls the same
kernel) sees the corrected Cd. `tests/integration/test_prescribed_decay.py`
min-drag pin moved from `< 0.12` to `0.15–0.30`.

### 1.2 Drag baseline work was charged in the co-rotating frame — MODERATE

`cpp/src/api.cpp`, energy bookkeeping.

`dE_actual`, `dE_drag`, `dE_lift` are F·v with the *inertial* velocity; the
baseline used `−F_base·|v_rel|`, i.e. the work in the atmosphere's frame.
The two differ by |ω×r|cos i / |v| ≈ 6 % on a prograde LEO, so a vehicle
holding the exact min-drag attitude scored "6 % worse than min-drag" in the
reward core, a bias no policy could remove. Now
`dE_baseline += −F_base·(v̂_rel·v)·Δt/m`. New pin:
`test_baseline_work_is_in_the_inertial_frame` (cube geometry, single
substep, agreement < 1 %).

### 1.3 `accel_gravity_N` fallback rotated a point-mass vector — LATENT

`cpp/src/orbit/gravity.cpp`. With no Stokes file loaded the helper returned
`EN^T · (−μ r_N/|r|³)`, which is not the inertial acceleration. The
`Simulator` never took that branch (it overrode the result with its own
J2+J3 fallback), so no v2.0 number is affected; the helper is now correct
and the Simulator no longer evaluates and discards a field.

## 2. Verified correct (no change)

Cross-checked in `tests/validation/verify_physics.py` against independent
formulations (scipy Legendre functions + finite-difference gradients,
textbook closed forms, direct Maxwellian quadrature, hand-written panel
loops):

| kernel | check | residual |
|---|---|---|
| GGM03S spherical harmonics, degree 2/4/8/12 | ∇U from scipy `lpmv`, fully normalised | 0.7–2.9e-10 (FD floor) |
| J2, J3 closed forms | ∇ of zonal potential | 1e-9 |
| Sentman C_p, C_τ (all θ, s ∈ {3,8,15}, α_E ∈ {0,0.93,1}) | textbook cos θ form; incident C_p by quadrature | 3e-16 / 7e-16 |
| M-01 accommodation T_r/T_i = α T_w/T_i + (1−α) s²/2 | flux-weighted E_w = 2kT (Doornbos 2012) | exact |
| Panel SRP optical law, cannonball law | explicit vector expression, mirror limit 2PA cos²θ | 1e-16 |
| Tilted dipole, WMM2020 (Schmidt recursion, secular terms) | −∇V finite difference | 1e-9 |
| MRP kinematics, DCM, composition, error MRP | existing tests | — |
| Plant RK4, torque-free body at 0.12°/s, 3000 s | |H|, T conserved | 1e-14 |
| Plant RK4, two-body vacuum orbit, 3000 s | SMA drift | 1e-7 m |
| B-dot, MRP-PD, third body, eclipse, Meeus Moon (equatorial) | existing tests | — |

At 5°/s tumbling (detumble regime) the non-symplectic RK4 at dt = 2 s
drifts |H| by 1.3e-5 over 3000 s; acceptable for that mode.

## 3. Performance (physics at rounding level unless stated)

Machine: 16-core desktop, GCC 16, `-O3`. Per 300 s advisor step, 72-panel hex,
SH degree 4, panel SRP, MSIS atmosphere.

| stage | v2.0 | v2.1 | note |
|---|---|---|---|
| C++ plant `Simulator.step` | 0.49 ms | 0.40 ms | while now wetting ~50 % more faces (§1.1); 0.29 ms before that fix |
| `env.step` v3 / v7 | 0.75 / 0.82 ms | 0.60 / 0.69 ms | |
| `env.step` v10 / v11 (trained variants) | 2.57 ms | 1.26 ms | onboard propagator moved to C++ |

What changed:

* **Sentman fast path** (`sentman.cpp`): the flow-only factors (1/s, 1/s²,
  √(T_r/T_i)) are hoisted out of the panel loop; the per-panel `acos`, `cos`,
  `sin`, tangent `norm` and division are gone — C_τ/sin θ multiplies the
  unnormalised tangent whose length is exactly sin θ. Public `sentman(θ, …)`
  unchanged. Residual vs v2.0: 1e-16.
* **Min-drag counterfactual cached per advisor step** (`api.cpp`): v2.0 ran
  the full Sentman panel sum twice per 2 s substep. The coefficient depends
  on the flow only through s, which moves < 0.1 % within 300 s; it is now
  evaluated once and refreshed when |v_rel| drifts > 2e-4 (Cd sensitivity
  ~1e-5 relative).
* **cos(mλ), sin(mλ) tables** in the gravity and WMM syntheses (once per
  order instead of once per (n, m)); bit-identical.
* **Mode string compares hoisted** out of the substep loop; bit-identical.
* **FP32 onboard propagator in C++** (`cpp/src/onboard/propagator_f32.cpp`,
  `cpp.propagate_f32_samples`): operation-for-operation mirror of
  `propagator.py` in IEEE binary32 (the flight-computer fidelity statement is
  preserved). `propagate` / `propagate_samples` dispatch to it
  (`use_cpp=False` selects the numpy reference). The v9+ future block fell
  from 1.3 ms to ~0.1 ms per step. Pinned in `tests/propagator/`: positions
  agree to < 5 m over 900 s, < 30 m over 6 h (float32 rounding-order floor;
  the GNSS the features emulate is 10 m 1-σ).
* **Torch thread knob, opt-in only** (`train.configure_torch_threads`,
  `ARLAMX_TORCH_THREADS`; default leaves torch untouched) in
  `train.build_model` and the `bench_v8` Duo trainer. Measured with 16
  SubprocVecEnv workers, v10 env, 4×18 PPO, 6144 steps: 1261 / 1188 / 1234
  steps/s at 16 / 2 / 1 threads — no effect on this machine, so nothing
  changes by default.

Where PPO time goes now (same measurement): a vectorised step of 16 workers
costs 4.6 ms wall against 1.26 ms of compute per env, i.e. the
SubprocVecEnv pipe traffic (the per-step `info` dict carries ~40 keys of
numpy arrays and nested dicts) is ~3× the physics; and the update phase at
`batch_size=64` (640 gradient steps per 4096-sample rollout, ~3 ms each in
SB3) is ~60 % of wall time. Slimming `info` during training and raising
`batch_size` (the v14 rounds already use 256) are the two levers; both
touch the campaign's analysis contract / optimisation and are left as
recommendations, not audit fixes.

## 4. Portability / housekeeping

* `python/arlamx_v2/paths.py`: data files resolve via `ARLAMX_GGM`,
  `ARLAMX_WMM`, `ARLAMX_HEX_GEOM`, `ARLAMX_SOLARCAT_STL`, `ARLAMX_V17`
  environment variables, then the repo-local `data/`, then the historical
  Basilisk / V1.7 paths. `resolve_wmm()` added. `tests/conftest.py` uses the
  same resolver instead of one hard-coded machine path.
* `data/` added: `GGM03S.txt` (Basilisk format), `WMM.COF` (WMM2020, the
  model the pinned magnetic tests were generated with), `WMM2025.COF`,
  `earthcup_hex_v3.geom`, `SolarCat_Assembly.STL`.
* `build.sh`: `ARLAMX_PYTHON` → ThesisMS env if present → `python3`; finds a
  pip-installed `cmake` next to the interpreter.
* Version strings: package `__version__ = "2.1"`, `cpp.__version__`, module
  docstrings, README, test register. Historical reports and report
  generators (`report_v8.py`, `build_mpc_report.py`, `pack_presentation.py`,
  `ACEnv/`, `outputs/`) keep their v2.0 labels because they describe v2.0
  runs.

## 5. How to verify

```bash
ARLAMX_PYTHON=/path/to/python ./build.sh
PYTHONPATH=python python -m pytest tests -q          # 159 passed, 3 skipped
PYTHONPATH=python python tests/validation/verify_physics.py   # ALL PASS
```

Environment used for this audit: `~/.venvs/arlamx21` (Python 3.11, numpy
2.4, pybind11 3.1, pymsis 0.12, gymnasium 1.3, stable-baselines3 2.9, torch
2.14, scipy 1.17, trimesh 5.1).
