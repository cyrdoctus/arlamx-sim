# ARLAMX V2.0 — System Overview, Equation Map, and C++ Physics-Library Plan

> **Superseded on scope (2026-08-18).** This file is the V1.7 survey (structure, equations, hot spots). The build contract is now [`APPROVAL_PLAN.md`](APPROVAL_PLAN.md): harmonics + SPICE third-body + Sentman + MRP-PD + B-dot live in one C++ plant; Python keeps training and geometry. Do not implement from this file’s older “leave Basilisk as the high-fidelity path / do not reimplement GGM03S” cut.

**Status:** V1.7 survey only. No code is generated from this file.
**Source of truth for this write-up:** live tree `../ARLAMX-V1.7/` as of 2026-08-17.
**Constraint:** V1.7 is not modified, moved, or deleted. Deep RL and AI stay in Python.

---

## 0. Purpose of this document

V1.7 is a working, equation-audited Python simulator that trains a small
attitude policy for Very Low Earth Orbit (VLEO) spacecraft. Training is slow
because the **physics inner loop** (panel aerodynamics, atmosphere lookup,
attitude kinematics, RK4 integration, inner controller) runs in interpreted
Python for every integration substep of every vectorized environment.

V2.0 is **not** a rewrite of ARLAMX. It is a split:

| Stays in Python | Moves into a custom C++ library |
|---|---|
| Gymnasium env orchestration | Sentman free-molecular-flow panel solver |
| Stable-Baselines3 (PPO / SAC / TD3 / RecurrentPPO) | Panel SRP force / torque |
| Observation feature registry | 12-state RK4 orbit + attitude integrator |
| Composite reward terms | Two-body + J2 + J3 gravity |
| Config / YAML / provenance | MRP kinematics, DCM, shadow set |
| Domain randomization policy | MRP-feedback (and other) inner controllers |
| Sensor-noise wrappers | Frames, geodesy, Sun / eclipse |
| ONNX / INT8 export of the actor | Visible-surface / silhouette extract (offline) |
| Basilisk high-fidelity backend (already C++) | Binding surface that the Python env calls |

This document is the overview of how V1.7 works, what equations it applies, and
how those equations import into C++ without changing the scientific contract.

---

## 1. What ARLAMX is

**ARLAMX** = *Attitude Reinforcement Learning Advisor, Mission eXtender*.

It trains a reinforcement-learning **attitude advisor** for spacecraft in VLEO.
The agent does not fire thrusters. It chooses a **target attitude**. Attitude
sets the projected area presented to the oncoming rarefied flow, so aerodynamic
force is the control authority: the policy reorients the vehicle to trade drag
(orbital lifetime / decay) against power, pointing, and safety constraints.

The trained actor is small (typical MLP `4 × 16`) and is exported to INT8 ONNX
plus a C header for an STM32U575-class flight computer. Flight inference is
already a C path. Training physics is not.

### 1.1 The scientific contract

The load-bearing physical claim of ARLAMX is:

> Every integration substep, the **same** Sentman free-molecular-flow force
> computed from the **same** panel geometry and the **same** MSIS atmosphere
> is injected into whichever orbit/attitude propagator is active. Attitude
> therefore changes the orbit, and the reward sees the real energy / altitude
> consequence.

That contract must survive V2.0. The C++ library is a faster evaluation of the
same equations, not a new physics model.

### 1.2 Two spacecraft, one hub

Everything vehicle-specific lives in YAML + geometry files, not in Python
branches.

| | SolarCat / Earth Cup | CatSat V7 |
|---|---|---|
| Vehicle | ~0.625 kg hexagonal membrane sail | ~9 kg 6U CubeSat |
| Geometry | `earthcup_hex_v3.geom` (~72 panels) | `catsat_v7_deployed_rich476.geom` (476 panels) plus STL variants |
| Default propagator | Basilisk (`dynamics.backend: basilisk`) | Python RK4 (`dynamics.backend: rk4`) |
| Orbit seed | Analytic Keplerian elements + domain randomization | Real orbit via `orbit_match` TLE / ephemeris fit |
| Model lines | `SC_v1`, `SC_v2`, `SC_v3*` | `CSat_v1` … `CSat_v4` |
| Typical objective | Power-safe lifetime / downlink | Lifetime + camera-axis safety (reboot cone) |
| Typical RL step | 300 s decision, 2 s physics → **150 substeps** | 60 s decision, 10 s physics → **6 substeps** |

### 1.3 What V1.7 is currently doing (runtime)

V1.7 is the V1.6 tree after the 2026-07-24 whole-repo audit and the 2026-08-03
… 2026-08-05 corrections. Architecture did not change. Two physics defects
that **did** change numbers were fixed:

1. **M-01 — Sentman energy-accommodation closure** in `aero/gas_surface.py`.
   Re-emission temperature now uses the energy-flux definition of \(\alpha_E\)
   (Moe & Moe 2005). Cd rises 6.5–8.5 % at the production value \(\alpha_E = 0.93\).
   Fully accommodated (\(\alpha_E = 1\)) runs are unchanged — which is why the
   old test suite missed the bug.
2. **Shadow-correct visible surface** in `geometry/visible_surface.py`.
   Closed non-convex shells over-state drag 1.2–2.2× if every front-facing
   facet is summed. The depth-buffer extract is used for CAD checks and
   fixed-attitude decay studies; the live RL loop still sums the configured
   `.geom` panel set with **no** per-step self-shadowing.

V1.7 also added CatSat reentry forecast tooling and orbit-variation sweeps.
Those are Python scripts on top of the same physics stack.

**What a training run actually does today:**

1. `main.py train` loads a YAML config.
2. `advisor/train.py` builds `n_envs` (typically 32) `ArlamxEnv` workers in a
   `SubprocVecEnv`, optionally wraps them in `VecNormalize`.
3. Each worker's `ArlamxEnv.step(action)` converts a 4-vector quaternion into
   an MRP target, then runs `n_substeps` of: atmosphere → Sentman FMF →
   optional panel SRP → inner controller → backend integrate.
4. After the substeps, Python composes the observation vector and the
   composite reward (including a min-drag energy counterfactual).
5. Stable-Baselines3 updates PPO / SAC / TD3 / RecurrentPPO.
6. Checkpoints, `model_config.yaml`, and `model_card.yaml` are frozen under
   `output/training/<run>/`.
7. `main.py export` later peels the actor MLP into ONNX / INT8 / C header.

The bottleneck is step 3, not step 5. The policy is a few-thousand-parameter
MLP. The physics loop is tens to hundreds of thousands of Sentman panel
evaluations per vectorized environment step.

---

## 2. V1.7 code structure

### 2.1 Top-level layout

```
ARLAMX-V1.7/
├── main.py                 CLI: train | validate | export | calibrate | baselines
├── advisor/                Deep RL / AI (stays Python in V2.0)
├── aero/                   Sentman FMF + panel SRP          → C++ candidate
├── atmosphere/             pymsis MSIS wrapper              → thin C++ FFI later
├── dynamics/               Gym hub + two backends
│   ├── env.py              ArlamxEnv — THE hub
│   ├── basilisk_interface.py   already C++ under Basilisk
│   └── rk4_backend.py      Python RK4                       → C++ candidate
├── control/                Inner attitude controllers       → C++ candidate
├── geometry/               .geom / .stl → Panel list        → load in Py, arrays in C++
├── orbit_match/            Offline OD + Cowell gravity      → gravity kernel to C++
├── config/                 ~100 YAML scenarios
├── subsystems/             ADCS / sensor hardware YAML
├── scenarios/              Validation scenario YAML
├── space_data/catsat/      Flight telemetry / TLEs
├── scripts/                Analysis, sweeps, plots (stay Python)
└── tests/                  Physics + env gates (stay Python, call C++ later)
```

Core Python (excluding scripts, tests, configs) is about **14 800 lines**.
The hub alone is 1 675 lines (`dynamics/env.py`). Reward and observation
registries are another ~2 700 lines. Those three files are the Python
surface that V2.0 keeps.

### 2.2 Module map with roles

| Module | Lines (approx.) | Role | V2.0 fate |
|---|---:|---|---|
| `dynamics/env.py` | 1 675 | Gymnasium hub: action clip, substep loop, power, GS, reward/obs call | Python orchestrator; inner numerics delegated |
| `advisor/reward.py` | 1 462 | Named, weighted reward-term registry | Stay Python |
| `advisor/observation.py` | 1 222 | Named, sensor-gated observation features | Stay Python |
| `orbit_match/estimator.py` | 695 | Offline fit of epoch state + ballistic coeff | Stay Python; call C++ propagator |
| `dynamics/basilisk_interface.py` | 673 | High-fidelity Basilisk backend | Stay as optional fidelity path |
| `orbit_match/v4.py` | 638 | Sequential-filter OD (EKF over `[r,v,log Bc]`) | Stay Python |
| `dynamics/rk4_backend.py` | 562 | Coupled 12-state RK4 | Port |
| `advisor/baselines.py` | 481 | MinDrag / MaxDrag / Tumble / live LVLH | Stay Python (uses C++ aero) |
| `advisor/train.py` | 444 | SB3 training loop, VecEnv, callbacks | Stay Python |
| `orbit_match/verify.py` | 440 | OD verification | Stay Python |
| `orbit_match/dynamics.py` | 404 | Cowell: 2-body+J2+J3+exp-drag | Port gravity/RK4 kernel |
| `advisor/export.py` | 321 | ONNX / INT8 / C-header actor export | Stay Python |
| `advisor/algorithms.py` | 269 | PPO / SAC / TD3 / RecurrentPPO factory | Stay Python |
| `control/mrp_feedback.py` | 226 | Default inner controller | Port |
| `advisor/evaluate.py` | 224 | Policy vs baseline validation | Stay Python |
| `orbit_match/missions.py` | 218 | Auto-discover `space_data/<mission>` | Stay Python |
| `aero/spacecraft_aero.py` | 190 | Whole-body FMF sum + Cd/Cl | Port (hot) |
| `control/controller_factory.py` | 190 | YAML → controller instance | Stay Python (constructs C++ object) |
| `orbit_match/frames.py` | 187 | COE ↔ (r,v); TEME → GCRS | Port COE; keep astropy TEME in Py |
| `atmosphere/nrlmsise00_wrapper.py` | 167 | pymsis query + mean molecular mass | Keep pymsis first; later C FFI |
| `geometry/visible_surface.py` | 173 | Depth-buffer lit-surface extract | Port as offline tool |
| `aero/gas_surface.py` | 138 | Sentman Cp, Cτ | Port (hottest kernel) |
| `dynamics/sensor_models.py` | 137 | Gaussian sensor noise | Stay Python |
| `aero/srp_panel.py` | 131 | Flat-panel SRP | Port |
| `aero/fmf_panel.py` | 127 | One-panel force/torque | Port (fused into vector kernel) |
| `dynamics/mc_bubble.py` | 125 | Sobol MC position/density deck | Stay Python |
| `control/lqr_controller.py` | 117 | CARE LQR (optional) | Port later |
| `advisor/provenance.py` | 106 | Frozen `model_config` / `model_card` | Stay Python |
| `control/b_dot.py` | 101 | B-dot detumble | Port later |
| `dynamics/environmental_torques.py` | 95 | SRP moment + gravity-gradient | Port |
| `geometry/stl_loader.py` | 82 | STL → Panel via trimesh | Stay Python (load-time) |
| `geometry/geom_loader.py` | 144 | `.geom` parser + `Panel` dataclass | Stay Python (load-time) |

### 2.3 The hub and the pluggable backend

`ArlamxEnv` (`dynamics/env.py`) is the single hub. It is constructed from one
YAML file. On init it:

1. Resolves subsystems (path or inlined snapshot).
2. Selects the backend: `dynamics.backend: basilisk | rk4`, falling back to
   `planner.enabled` for pre-V1.5 configs.
3. Loads geometry via `geometry.loader.load_geometry` (`.geom` or `.stl`).
4. Derives min-drag / max-drag body axes from projected area.
5. Builds the inner controller from `adcs.controller`.
6. Builds `CompositeReward` and pins the observation dimension.
7. Optionally loads ground-station ECEF positions and the v15/v17/v18 power
   model.

**The dynamics backend is the only pluggable part.** Geometry, atmosphere,
aerodynamics, control, observation, and reward are shared. That is the
invariant V2.0 must keep: C++ evaluates the shared physics; Python still owns
the hub.

Backend public surface (both implement it):

```
setup / reset / step(dt, ext_force_B, ext_torque_B)
get_state / get_dcm_BN / get_orbital_energy
get_sun_vec_B / get_srp_force_B / get_mag_field_B / get_rw_momentum_mag
set_mass / set_inertia
has_magnetorquers / has_reaction_wheels
```

State dict keys the rest of the stack consumes:

`r_BN_N`, `v_BN_N`, `v_BN_B`, `sigma_BN`, `omega_BN_B`, `altitude_m/km`,
`sma_m`, `ecc`, `inc_rad`, `raan_rad`, `argp_rad`, `true_anom_rad`,
`eclipse_factor`, `sim_time_s`, `sun_vec_B`, `mag_field_B_nT`.

The env then **augments** this dict with `rho_kg_m3`, `spacecraft_mass_kg`,
`battery_soc`, `power_gen_norm`, `drag_force_N`, `sma_decay_rate_m_min`,
ground-station fields, and planner pointing fields. Observation and reward
read the augmented dict. That augmentation stays in Python.

### 2.4 Dependency graph (internal)

```
main.py
  └─ advisor.{train, evaluate, export, calibrate, baselines}
        └─ dynamics.env.ArlamxEnv
              ├─ geometry.loader ──► geom_loader | stl_loader   → list[Panel]
              ├─ atmosphere.nrlmsise00_wrapper (pymsis)
              ├─ aero.spacecraft_aero ──► fmf_panel ──► gas_surface
              ├─ aero.srp_panel
              ├─ control.controller_factory
              │     └─ mrp_feedback | bang_bang | lqr | b_dot
              ├─ advisor.observation.SensorRealisticObservation
              ├─ advisor.reward.CompositeReward
              ├─ advisor.baselines  (min/max drag axes, live LVLH)
              ├─ dynamics.sensor_models.SensorSuite
              └─ backend
                    ├─ BasiliskInterface          (already C++)
                    └─ RK4DynamicsBackend
                          ├─ orbit_match.dynamics._accel_scalar
                          └─ orbit_match.frames.{rv_to_coe, coe_to_rv}
```

`orbit_match` is also used **offline** by `main.py calibrate` to fit a real
CatSat state + ballistic coefficient. Inside the RL loop the RK4 backend
calls the same gravity kernel with `bc = 0`: drag comes only from the
MSIS-fed Sentman force, never from the exponential atmosphere. That split
is load-bearing and must be preserved.

### 2.5 The per-substep loop (what actually runs)

One advisor decision, from `ArlamxEnv.step`:

```
action ∈ R^4  (raw quaternion, scalar-first)
    │  normalise, q0 ≥ 0
    │  SLERP-clip (isotropic)  OR  anisotropic per-axis clip (SC_v2)
    ▼
held target q  →  σ_target (MRP)
    │  if action.frame = lvlh:  [BN] = [BL][LN]  each substep
    ▼
for s = 1 … n_substeps:                          # 150 (SC) or 6 (CSat)
    1. state = backend.get_state()               # r,v (N); σ,ω (B)
    2. ω_ref = 0  or  ω_orbit = h/|r|²
    3. τ_ctrl = controller.compute(σ, ω, σ_tgt, ω_ref)
    4. v_rel_N = v − (ω_⊕ × r)                   # corotation usually OFF
    5. v_B_gas = −[BN] v_rel_N                   # gas-onto-body
    6. (lat, lon_ECEF, h_geodetic) = Bowring + GMST
    7. (ρ, T, m̄) = MSIS  (cached every N substeps)
    8. (F_B, τ_B, Cd, Cl) = spacecraft_aero(panels, v_B_gas, ρ, T, m̄, T_w, α_E)
    9. F_base = min-drag counterfactual (closed-form Cd path, or a second aero)
   10. F_srp_B = panel_srp_force(...)            # if use_panel_srp
   11. backend.step(dt, F_B+F_srp, τ_B+τ_ctrl)
   12. accumulate  ΔE_actual, ΔE_baseline, drag/lift, power, tracking error
    ▼
power finalize (battery SoC, downlink load)
reward.compute(...)     # Python
obs = observation.build(state)   # Python + sensor noise
return (obs, reward, terminated, truncated, info)
```

Steps 3–11 are the C++ target. Steps around them stay Python.

### 2.6 RL / AI layer (explicitly not moving)

`advisor/` is the AI product. It stays in Python because:

- Stable-Baselines3, PyTorch, Gymnasium, and VecEnv multiprocessing are the
  training runtime. Reimplementing them is months of work for no physics gain.
- The observation and reward registries are **config-composed science**, not
  numerics. Terms are added and reweighted per model line (`SC_v1` vs
  `CSat_v1`). That iteration speed must stay in Python.
- Actor export (`advisor/export.py`) already produces C for the **flight**
  computer. That is a different C artifact (the policy) from the training
  physics library.
- Provenance (`model_config.yaml`, `model_card.yaml`) is a YAML/JSON concern.

Supported algorithms (`advisor/algorithms.py`):

| Algo | Library | Typical use |
|---|---|---|
| PPO | SB3 | Default SolarCat line |
| SAC | SB3 | Off-policy stage-2 |
| TD3 | SB3 | Off-policy stage-2 |
| RecurrentPPO | sb3-contrib | CatSat line (LSTM; **not** ONNX-exportable) |

Typical network: MLP `net_arch = [16,16,16,16]` (~a few thousand parameters).
The actor evaluation is microseconds. The env step is milliseconds to tens of
milliseconds. That is why a C++ physics library is the correct cut, and a C++
policy trainer is not.

Action space: `Box(-1,1,(4,))` — scalar-first quaternion, unit-normalized and
slew-clipped inside the env.

Observation space: composed from named features, intersected with the
subsystem sensor set. SC_v1 is **22-dim** and has **no Sun sensor** — the
policy must infer Sun geometry from `battery_soc` + `power_gen`. CatSat keeps
a sun sensor.

### 2.7 Configuration and provenance

Two tiers:

1. **Global** `config/*.yaml` — editable.
2. **Per-model snapshot** written at train time:
   - `model_config.yaml` — fully resolved, subsystems inlined, `rl` baked.
   - `model_card.yaml` — human digest (arch, obs/action dim, reward terms,
     seed, sweep ranges).

Reloading a checkpoint against its own `model_config.yaml` reproduces the
exact obs/action space. V2.0 bindings must accept the same resolved config
fields (mass, inertia, α_E, T_w, dt, panel arrays) so a V1.7 snapshot can
drive the C++ plant without a schema break.

---

## 3. Equations applied in V1.7

All physics is SI. Frames: **N** = J2000 / GCRS inertial, **B** = spacecraft
body, unless noted. This catalog is taken from the **live V1.7 source**, not
from the older `EQUATIONS_AND_DATAFLOW.md` (that file still prints the
pre-M-01 \(T_r/T_i\) formula).

### 3.1 Free-molecular-flow aerodynamics (the load-bearing model)

**Location:** `aero/gas_surface.py` → `aero/fmf_panel.py` → `aero/spacecraft_aero.py`

Speed ratio

\[
s = \frac{|v|}{\sqrt{2\,k_B T / \bar{m}}}, \qquad
k_B = 1.380649\times 10^{-23}\,\mathrm{J\,K^{-1}}
\]

Incidence \(\theta\) is the angle between the outward panel normal \(\hat{n}\)
and the **gas-onto-panel** velocity. \(\gamma = s\cos\theta\),
\(Z = 1+\mathrm{erf}(\gamma)\), \(E = e^{-\gamma^2}\).

Incident coefficients (Sentman 1961; Doornbos 2012 eqs. 3.53–3.55):

\[
C_{p,i} = \frac{(\gamma^2+\tfrac12)Z + \gamma E/\sqrt{\pi}}{s^2}, \qquad
C_{\tau,i} = \sin\theta\cdot\frac{\gamma Z + E/\sqrt{\pi}}{s}
\]

**V1.7 energy-accommodation closure (M-01, current code):**

\[
\alpha_E = \frac{E_i - E_r}{E_i - E_w}, \quad
E_{r,w} = 2kT_{r,w}, \quad
E_i = \tfrac12 m V^2
\]

which closes as

\[
\frac{T_r}{T_i} = \alpha_E\,\frac{T_w}{T_i} + (1-\alpha_E)\,\frac{s^2}{2}
\]

Do **not** port the older form \(1 + \alpha_E(T_w/T_i - 1)\). That charged
incident molecules only their ambient thermal energy and under-predicted Cd
by 6.5–8.5 % at \(\alpha_E = 0.93\). The two forms coincide only at
\(\alpha_E = 1\).

Diffuse re-emission (no net shear):

\[
C_{p,r} = \sqrt{T_r/T_i}\,\frac{E + \sqrt{\pi}\,\gamma Z}{2s^2}, \qquad
C_p = C_{p,i}+C_{p,r}, \qquad C_\tau = C_{\tau,i}
\]

Hyperthermal identity used as a unit test:

\[
C_p\cos\theta + C_\tau\sin\theta \;\xrightarrow{s\to\infty}\; 2\cos\theta
\]

Panel force / torque (`fmf_panel.panel_force`):

\[
q = \tfrac12\rho |v|^2, \qquad
\mathbf{F} = A\,q\bigl(-C_p\,\hat{n} + C_\tau\,\hat{t}\bigr), \qquad
\boldsymbol{\tau} = \mathbf{r}_c \times \mathbf{F}
\]

\(\hat{t}\) is the tangential **gas-flow** direction in the panel plane
(\(\hat{v} + \cos\theta\,\hat{n}\), then normalized). Pressure pushes along
\(-\hat{n}\); shear drags **with** the flow. A panel with \(\cos\theta\le 0\)
contributes nothing.

Whole-body coefficients:

\[
C_D = \frac{\mathbf{F}\cdot\hat{v}}{q A_\mathrm{ref}}, \qquad
C_L = \frac{|\mathbf{F}-(\mathbf{F}\cdot\hat{v})\hat{v}|}{q A_\mathrm{ref}}
\]

\(A_\mathrm{ref}\) is `one_sided` \(= \tfrac12\sum A_i\) (membrane front+back
counted once) or `total`. Production configs use `one_sided`.

**Implementation note for C++:** V1.7 evaluates this in a Python `for panel`
loop, allocating `ndarray`s and calling `get_model()` per panel. That is the
single largest avoidable cost. The C++ kernel should be one function over
SoA arrays `(n[3*N], A[N], r_c[3*N])`.

### 3.2 Atmosphere

**Live training path** — `atmosphere/nrlmsise00_wrapper.py`

pymsis `calculate(...)` at geodetic \((\phi,\lambda_\mathrm{ECEF},h)\). Model
version is explicit: `nrlmsise00` (0), `msis2.0`, or `msis2.1` (default, the
historical effective behavior). Outputs \(\rho\), \(T\), species number
densities. Mean molecular mass

\[
\bar{m} = \frac{\sum_i n_i m_i}{\sum_i n_i}
\quad\text{over }\{\mathrm{He},\,\mathrm{O},\,\mathrm{N}_2,\,\mathrm{O}_2,\,\mathrm{Ar},\,\mathrm{H},\,\mathrm{N}\}.
\]

Species masses use 2019 SI molar masses / \(N_A\). If \(\sum n_i = 0\), fall
back to atomic oxygen.

The env queries at **geodetic altitude** and **GMST-rotated Earth-fixed
longitude** (`_eci_to_geodetic_llh` + `_gmst_rad`). Caching:
`aero.atmosphere_cache_substeps` (SC_v1 = 5).

**Offline OD path only** — `orbit_match/dynamics.py::density`

Vallado App. B piecewise exponential, 0–1000 km:

\[
\rho(h) = \rho_0\exp\bigl(-(h-h_0)/H\bigr)
\]

plus a co-rotating drag term when the fitted ballistic coefficient \(B_c>0\).
This exponential model is **never** used inside the RL loop.

### 3.3 Gravity and orbital mechanics

**RK4 / orbit_match kernel** — `orbit_match/dynamics.py::_accel_scalar`

\[
\mathbf{a}_{2b} = -\mu\mathbf{r}/r^3
\]

J2:

\[
f_2 = \tfrac32 J_2\mu R_E^2 / r^5,\quad
a_x \mathrel{+}= f_2 x(5z^2/r^2-1),\quad
a_y \mathrel{+}= f_2 y(5z^2/r^2-1),\quad
a_z \mathrel{+}= f_2 z(5z^2/r^2-3)
\]

J3:

\[
f_3 = \tfrac12 J_3\mu R_E^3 / r^7
\]

\[
a_x \mathrel{+}= 5f_3 x(7z^3/r^2-3z),\quad
a_y \mathrel{+}= 5f_3 y(7z^3/r^2-3z),
\]

\[
a_z \mathrel{+}= 3f_3\bigl(35z^4/(3r^2) - 10z^2 + r^2\bigr)
\]

These Cartesian components were re-derived from
\(V=(\mu/r)[1-\sum J_n(R_E/r)^n P_n(z/r)]\) and match \(\nabla V\) to
\(1.15\times 10^{-9}\) relative. Constants from `orbit_match/config.py`
(\(\mu = 3.986004418\times 10^{14}\,\mathrm{m}^3\mathrm{s}^{-2}\),
\(R_E = 6378137\,\mathrm{m}\)).

Co-rotating drag (**OD only**, `bc>0`):

\[
\mathbf{v}_\mathrm{rel} = \mathbf{v}-\boldsymbol{\omega}_\oplus\times\mathbf{r},\qquad
\mathbf{a}_d = -\tfrac12\rho B_c |\mathbf{v}_\mathrm{rel}|\,\mathbf{v}_\mathrm{rel}
\]

In the RL RK4 backend, `bc` is hard-coded **0**. The env injects Sentman
force as \(\mathbf{a}_\mathrm{ext} = [BN]^\top \mathbf{F}_B / m\).

**Basilisk path** (not reimplemented in the custom lib): GGM03S spherical
harmonics (degree configurable; SC_v1 uses 4), optional Sun/Moon third body
via SPICE, WMM magnetic field, cannonball SRP (disabled when the env injects
panel SRP), cylindrical eclipse.

Vis-viva / period:

\[
\varepsilon = v^2/2 - \mu/r = -\mu/(2a),\qquad
T = 2\pi\sqrt{a^3/\mu}
\]

SMA feature uses \(a = 1/(2/r - v^2/\mu)\) with a denominator guard.

Classical elements (`orbit_match/frames.py`): eccentricity vector, node,
quadrant checks — Vallado Alg. 9–10 / Curtis Ch. 4.

### 3.4 Attitude kinematics and control

MRP → DCM (`rk4_backend.mrp_to_dcm`, Schaub & Junkins):

\[
C(\boldsymbol{\sigma}) = I_3 + \frac{8[\tilde{\sigma}]^2 - 4(1-\sigma^2)[\tilde{\sigma}]}{(1+\sigma^2)^2}
\]

Kinematics + shadow set:

\[
\dot{\boldsymbol{\sigma}} = \tfrac14\bigl[(1-\sigma^2)I + 2[\tilde{\sigma}] + 2\boldsymbol{\sigma}\boldsymbol{\sigma}^\top\bigr]\boldsymbol{\omega}
\]

\[
\boldsymbol{\sigma}_\mathrm{shadow} = -\boldsymbol{\sigma}/\sigma^2 \quad\text{when }\sigma^2>1
\]

Euler rotational EOM:

\[
\dot{\boldsymbol{\omega}} = I^{-1}\bigl(\boldsymbol{\tau}_\mathrm{ext} - \boldsymbol{\omega}\times I\boldsymbol{\omega}\bigr)
\]

V1.7 inertia is **diagonal in every config** (principal-axis assumption, CoM
pinned at body origin). The code paths already accept a full \(3\times 3\);
the C++ API should too, even if production YAML still ships a 3-vector.

MRP feedback (default controller):

\[
\boldsymbol{\tau} = -K\boldsymbol{\sigma}_e - P\boldsymbol{\omega}_e + \boldsymbol{\omega}\times I\boldsymbol{\omega}
\]

\(\boldsymbol{\sigma}_e\) is the MRP composition of \(\boldsymbol{\sigma}_{BN}\) with the
inverse of \(\boldsymbol{\sigma}_\mathrm{target}\). Convention `proper` (default) uses
\(-\boldsymbol{\sigma}\); `legacy` uses the shadow set and exists only to replay
pre-V1.5.2 Earth Cup models. Optional per-axis torque saturation and a
predictive body-rate cap (plant-consistent: predict with the PD part, then
re-add the gyroscopic term).

Other controllers (optional, not the production path):

- **Bang-bang:** per-axis time-optimal slew, switch when remaining angle
  equals \(\tfrac12 \omega^2/\alpha_\max\).
- **LQR:** CARE on the 6-state \([\boldsymbol{\sigma}_e,\boldsymbol{\omega}_e]\) linearized
  about rest (the ¼ in \(\dot{\sigma}=\tfrac14\omega\) is a known scaling
  footnote; unused in primary runs).
- **B-dot:** \(\mathbf{m}=-k\,\mathrm{d}\mathbf{B}/\mathrm{d}t\), dipole
  saturated. Detumble only.

Action-side attitude math (env, stays conceptually Python but is cheap
enough to also live in C++ if the whole substep is fused):

- Euler ZYX → quaternion → MRP (legacy).
- Quaternion normalize with \(q_0\ge 0\).
- Shoemake SLERP clip to `max_slew_per_inference_deg`.
- Anisotropic body-axis clip (SC_v2 magnetorquers): scale the rotation
  vector of \(q_\mathrm{held}^{-1}\otimes q_\mathrm{new}\) by the tightest
  per-axis cap.
- LVLH DCM: \(\hat{z}_L=-\hat{r}\), \(\hat{y}_L=-\hat{h}\),
  \(\hat{x}_L=\hat{y}_L\times\hat{z}_L\).

Tracking-error diagnostic: \(\theta = 4\arctan(|\boldsymbol{\sigma}_e|)\).

### 3.5 Environment geometry and optics

Bowring WGS-84 ECEF → geodetic (env):

\[
\theta=\mathrm{atan2}(z a,\, p b),\qquad
\phi=\mathrm{atan2}\bigl(z+e'^2 b\sin^3\theta,\; p-e^2 a\cos^3\theta\bigr)
\]

IAU-1982 linear GMST:

\[
\mathrm{GMST}=280.46061837^\circ + 360.98564736629^\circ\,(\mathrm{JD_{UT1}}-2451545.0)
\]

UTC≈UT1 is accepted (error \(\lt 0.004^\circ\)).

Low-precision Sun (RK4 backend, Montenbruck / Meeus; ~0.01° intrinsic,
~0.36° mean-of-date vs J2000 in 2026):

\[
\lambda = L + 1.915^\circ\sin M + 0.020^\circ\sin 2M,\quad
\varepsilon = 23.439^\circ - 4\times 10^{-7}\,n
\]

\[
\hat{s} = (\cos\lambda,\;\cos\varepsilon\sin\lambda,\;\sin\varepsilon\sin\lambda)
\]

Cylindrical eclipse: lit unless behind Earth along \(-\hat{s}\) and
\(|\mathbf{r}_\perp|<R_E\).

Flat-panel SRP (`aero/srp_panel.py`) — **projected-area cannonball, not sail
optics**:

\[
\mathbf{F}=\sum_{\mathrm{lit}} P_\mathrm{SR}\,C_r\,A\max(0,\hat{n}\cdot\hat{s})\,(-\hat{s})
\]

\(P_\mathrm{SR}=4.56\times 10^{-6}\,\mathrm{Pa}\) at 1 AU, \(C_r=1.8\). No
specular normal term, no diffuse lobe, no per-face optical properties. Force
only in the training env (moment exists in
`dynamics/environmental_torques.py` for the free-attitude study).

Gravity-gradient (same file, not in the main env loop):

\[
\boldsymbol{\tau}_{gg} = 3\frac{\mu}{r^3}\,\hat{o}_B\times(I\hat{o}_B), \qquad
\hat{o}_B = [BN](-\hat{r})
\]

### 3.6 Integrator

Classical RK4 on the 12-state \(\mathbf{y}=[\mathbf{r},\mathbf{v},\boldsymbol{\sigma},\boldsymbol{\omega}]\):

\[
\mathbf{y}_{n+1}=\mathbf{y}_n+\frac{h}{6}(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4)
\]

Body-frame force/torque are held zero-order-hold across the advisor substep,
but the DCM is **re-evaluated at each RK4 stage** because the body rotates.
MRP shadow is applied after each completed sub-step. `planner.rk4_step_s`
(default 15 s) further subdivides `dt_s` when needed.

Orbit-match OD can also use SciPy `solve_ivp` (DOP853) as an independent
cross-check. That stays a Python verification tool.

### 3.7 Power, ground stations, counterfactual energy

Solar generation (env, v15–v18):

\[
P_\mathrm{gen} = P_\mathrm{peak}\,\eta\cdot\mathrm{illum}\cdot\varepsilon_\mathrm{ecl}
\]

`illum` is \(|\hat{s}\cdot\hat{e}_\mathrm{axis}|\) (two-sided) or
\(\max(0,\hat{s}\cdot\hat{e}_\mathrm{axis})\) (one-sided, CatSat +Y wing).
Loads: always-on baseline + optional sun-gated GPS + magnetorquer effort
\(|\boldsymbol{\tau}|/\tau_\max\) + LoRa transmit when a station is visible **and**
the antenna boresight cosine exceeds `downlink_align_min`. Battery / supercap:

\[
E \leftarrow \mathrm{clip}\bigl(E + (P_\mathrm{gen}-P_\mathrm{load})\Delta t,\; 0,\; E_\mathrm{cap}\bigr)
\]

Ground-station visibility: station above the geometric horizon
\(\cos c > R_E/(R_E+h)\). Sticky dwell keeps one station while it stays up.
**Known defect (M-04):** stations are fixed in inertial space (ECI treated as
ECEF). Do not silently “fix” this in C++ or SC_v2 numbers become
incomparable.

Non-gravitational energy used by the longevity reward (not Keplerian
\(\Delta\varepsilon\), which is J2-noise dominated):

\[
\Delta E_\mathrm{actual} = \sum_s \frac{(\mathbf{F}_\mathrm{aero}+\mathbf{F}_\mathrm{SRP})\cdot\mathbf{v}}{m}\,\Delta t
\]

\[
\Delta E_\mathrm{baseline} = -\sum_s \frac{|F_{\mathrm{drag,min}}|\,v_\mathrm{rel}}{m}\,\Delta t
\]

Closed-form baseline (`use_counterfactual_closed_form: true`, SC default)
uses `coefficients_only` at the min-drag freestream and
\(|F|=C_D q A_\mathrm{ref}\), skipping the torque cross product.

### 3.8 Reward (Python, stays Python)

\[
R_t = \sum_{i\in\mathrm{terms}} w_i\, r_i(s_t)
\]

Production terms (see also `../REWARD_EQUATIONS.md`):

| Term | Form (sketch) | Used by |
|---|---|---|
| `dE_vs_baseline` | \(\mathrm{clip}((\Delta E_a-\Delta E_b)/\max(|\Delta E_b|,\varepsilon),-1,1)\cdot w_\mathrm{long}(h)\) | SC, CSat |
| `power_budget` | SoC toward target + charge-when-low − depletion penalty | SC, CSat |
| `shade_conservation` | \(-\overline{\mathrm{effort}\cdot(1-\varepsilon)}\) | SC |
| `action_smoothness` | \(-(1-\langle q_\mathrm{new},q_\mathrm{prev}\rangle)\) | SC, CSat |
| `omega_penalty` | \(-\|\boldsymbol{\omega}\|\) | SC, CSat |
| `momentum_penalty` | \(-\|H\|\) (RW; 0 on RK4) | SC |
| `altitude_cliff` | \(-1[h<h_\mathrm{floor}]\) | SC, CSat |
| `ground_station_pointing` | eclipse-weighted boresight cosine if GS visible | SC_v2 |
| `pointing_exclusion` | \(-v_\mathrm{excl}\) (cone penetration) | CSat |
| `axis_exposure` | \(-(w_s c_\mathrm{sun}+w_r c_\mathrm{ram})\) | CSat |

C++ does **not** own these. It must return the scalars the terms already
consume (`actual_nongrav_dE`, `counterfactual_dE`, mean drag, Cd/Cl, eclipse,
sun/nadir vectors, tracking error, …).

### 3.9 Visible surface (offline / analysis)

`geometry/visible_surface.py` samples the triangle soup, builds a
flow-aligned depth buffer, keeps the **nose** sample (largest \(\mathbf{p}\cdot\hat{d}\))
per pixel, reconstructs area as \(A = \Delta x^2 / \max(\hat{n}\cdot\hat{d},\,\Delta x/L)\),
and bins normals. Self-check:

\[
\sum_i A_i|\hat{n}_i\cdot\hat{d}| = N_\mathrm{pix}\,\Delta x^2 = A_\mathrm{silhouette}
\]

This is a preprocessing / verification kernel, not part of every RL substep.
Worth porting later for CAD-faithful CatSat tables; not Phase 1.

### 3.10 Literature anchors (do not invent new forms)

Sentman 1961 (DTIC AD0265409); Moe & Moe 2005; Doornbos 2012; Sutton 2009;
Mostaza Prieto et al. 2014; Picone et al. 2002; Emmert 2015; Vallado 2013;
Montenbruck & Gill 2000; Curtis 2014; Schaub & Junkins 2018; Shuster 1993;
Tsiotras 1996; Wie 2008; Bowring 1976; Meeus 1998; Hairer / Nørsett / Wanner
1993; Press et al. 2007; Shoemake 1985.

---

## 4. Where V1.7 spends time — and why C++ helps

### 4.1 Work per vectorized training step

Let \(N_\mathrm{env}\), \(N_\mathrm{sub}\), \(N_\mathrm{panel}\) be workers,
physics substeps per advisor step, and panels.

| Line | \(N_\mathrm{env}\) | \(N_\mathrm{sub}\) | \(N_\mathrm{panel}\) | Sentman evals / vec-step (policy aero only) |
|---|---:|---:|---:|---:|
| SC_v1 | 32 | 150 | ~72 | **~345 600** |
| CSat_v1 | 32 | 6 | 476 | **~91 392** |

Plus, per substep, in Python:

- 1–2 `np.linalg` DCM / MRP conversions (Basilisk `rbk` or Python MRP).
- 1 MSIS call every `atmosphere_cache_substeps` (SC: every 5th).
- 1 controller `compute` (MRP compose, optional rate-cap).
- 1 panel-SRP loop (same \(N_\mathrm{panel}\)).
- 1 backend `step` (Basilisk C++ **or** Python RK4 with 4 stages × DCM +
  gravity + Euler, possibly subdivided by `rk4_step_s`).
- Dozens of `ndarray` allocations (`zeros`, `asarray`, `cross`, `norm`).

The policy forward pass is negligible. VecEnv already parallelizes across
processes; each process is still a slow Python physics loop.

### 4.2 Why the Python aero loop is the first target

`spacecraft_aero` is:

```
for panel in panels:
    panel_force(...)          # Python function call
        get_model(gsi_name)   # dict lookup every panel
        sentman(...)          # erf, exp, several Python scalars
        np.cross, np.dot      # 3-vector allocations
```

For a convex 72-panel sail this is 72 interpreter round-trips, 72 times
`arccos` / `clip` / `sqrt`, 72 heap arrays, **per substep**, **per env**.
Nothing about Sentman requires that. The kernel is:

- input: packed `float64` arrays + \((\mathbf{v}_B,\rho,T,\bar{m},T_w,\alpha_E)\)
- output: \(\mathbf{F}_B\), \(\boldsymbol{\tau}_B\), \(C_D\), \(C_L\), \(A_\mathrm{ref}\)

That is a few kilobytes in, 32 bytes out, and is SIMD-friendly (independent
panels, mask \(\cos\theta\le 0\)).

A second aero pass for the min-drag counterfactual is already avoided on
SolarCat via `use_counterfactual_closed_form`. C++ should keep that fast path
and also make the full second pass cheap enough that CatSat can drop the
approximation if wanted.

### 4.3 Why the Python RK4 backend is the second target

`RK4DynamicsBackend.step` allocates a new 12-vector and a new 3×3 DCM **four
times per RK4 stage**, and `_accel_scalar` is already written in scalars
specifically to dodge NumPy. That comment in `orbit_match/dynamics.py` is
the author admitting the hot path wants C.

Basilisk is already C++ and is **not** the thing to replace for fidelity.
It is the thing that makes SolarCat training **heavy in a different way**
(process startup, SPICE, spherical harmonics). V2.0 should give SolarCat
the option of the same fast RK4 plant CatSat uses, with Basilisk retained
as the validation / high-fidelity backend.

### 4.4 What will *not* get faster by moving to C++

- MSIS itself (already Fortran/C inside pymsis). Crossing into Python once
  per cached query is small next to Sentman. A later C FFI to NRLMSIS
  removes the crossing; it is Phase 2.
- SB3 / PyTorch update. Already fine.
- YAML load, plot scripts, ONNX export.
- Visible-surface extraction at 1–2 mm (millions of samples) — worth C++
  but it is offline.
- Basilisk’s own gravity / SPICE.

### 4.5 Expected speedup (order-of-magnitude, not a promise)

These are planning bounds, not measured V2.0 numbers.

| Kernel | V1.7 | After a competent C++ port | Why |
|---|---|---|---|
| Sentman over 72–476 panels | Python loop + allocs | packed SIMD, no per-panel call | 20–100× on that kernel |
| RK4 12-state step | NumPy allocs / stage | in-place scalar / small arrays | 10–30× on that kernel |
| Whole `env.step` (RK4 path) | aero + RK4 + MSIS + Python glue | aero+RK4 fused, MSIS still pymsis | **5–20×** wall-clock per env |
| Whole `env.step` (Basilisk path) | aero Python + Basilisk C++ | aero C++, Basilisk unchanged | **2–5×** (aero-bound at 72+ panels) |
| Training 500k steps × 32 envs | hours | same science, several× fewer hours | more seeds / larger sweeps |

A fused “advance one advisor step” C++ entry point (action in, state/accumulators
out) beats a pile of tiny pybind calls, because 150 substeps × 10 crossings
would otherwise eat the gain.

---

## 5. How the equations import into C++ (no code, just the cut)

### 5.1 Design rule

> One C++ library, one scientific contract, two callers: the Python training
> env and (later) standalone analysis tools. The library does not know what
> PPO is. Python does not re-derive Sentman.

Do **not** generate a second, divergent Python physics stack. V1.7 remains
the reference implementation until numerical gates say the library matches
it. Then Python modules become thin wrappers.

### 5.2 Proposed library shape (conceptual)

Name used here only as a handle: **`libarlamx`**. Not created in this folder.

```
libarlamx
├── types          3-vectors, 3×3, panel SoA, spacecraft parameters
├── aero           sentman(θ,s,Tw/Ti,α_E) + spacecraft_aero(...)
├── srp            panel_srp_force / force+moment
├── gravity        two-body + J2 + J3  (= _accel_scalar)
├── attitude       MRP↔DCM, kinematics, shadow, quat helpers, LVLH
├── control        MRP feedback (+ later LQR / bang-bang / B-dot)
├── frames         rv_to_coe, coe_to_rv, Bowring, GMST, Sun, eclipse
├── integrate      RK4 12-state, ZOH body force, stage-wise DCM
├── atmosphere     optional later: NRLMSIS C API; Phase 1 accepts ρ,T,m̄
└── bindings       pybind11 module `arlamx_cpp`
```

Python V2.0 env (still `ArlamxEnv` conceptually):

```
action  →  (optional) C++ slew-clip + substep loop
                C++ : aero + srp + control + RK4   [or aero only, Basilisk steps]
        ←  state + accumulators
Python  →  SensorSuite noise + observation.build + reward.compute
        →  SB3
```

### 5.3 Two integration depths (do them in this order)

**Depth A — kernels behind the existing Python loop.**
Replace `spacecraft_aero`, `panel_srp_force`, `mrp_to_dcm` / `mrp_rate`,
`_accel_scalar`, and `MRPFeedbackController.compute` with C++ calls, one
substep at a time. Smallest risk. Leaves 150 Python-loop iterations in
`env.step`. Good for proving numeric identity. Modest speedup.

**Depth B — fused advisor step.**
One C++ function:

```
advance_advisor_step(
    y12, panels, mass, I,
    q_target_held,          # already clipped in Python or in C++
    atm_cache,              # ρ,T,m̄ or a callback
    dt_s, n_substeps,
    aero_params, ctrl_params, srp_flag, frame_flag
) → {y12, ΔE_actual, ΔE_baseline, mean_drag, Cd, Cl, eclipse, sun_B, …}
```

This is the training-speed design. Atmosphere can still be a Python callback
in Phase 1 (`std::function` / pybind holder) so pymsis stays. Phase 2 inlines
NRLMSIS.

**Basilisk mode** uses Depth A only: C++ aero + SRP + controller, Basilisk
`step` still owns gravity / third body / WMM. Do not reimplement GGM03S or
SPICE in `libarlamx`.

### 5.4 Data layout and ABI

Port **arrays**, not Python `Panel` objects.

| Quantity | V1.7 | C++ |
|---|---|---|
| Panel set | `list[Panel]` of `ndarray` | SoA: `n[3*N]`, `A[N]`, `c[3*N]` (`float64`) |
| State | dict of `ndarray` | `struct State { r[3], v[3], sigma[3], omega[3]; }` + time |
| Inertia | length-3 or 3×3 | 3×3 always; diagonal is a special case |
| Atmosphere | Python dict | `struct Atmo { rho, T, m_bar; }` |
| Aero result | Python dict | `struct Aero { F[3], tau[3], Cd, Cl, A_ref; }` |
| Config | YAML | POD params copied at `reset` / DR |

`float64` everywhere for identity tests against NumPy. A later `float32`
aero option is a measured optimization, not a default — Sentman has `erf`
and \(s^{-2}\), and Cd identity is a thesis number.

Memory ownership: library copies panel arrays at `set_geometry`. Domain
randomization (scale, appendage flop) rebuilds the arrays in Python
(`ArlamxEnv._randomized_panels`) and pushes them once per episode. Do not
put YAML or `trimesh` in C++.

### 5.5 Equation-by-equation import notes

| Equation group | Import notes |
|---|---|
| **A1 Sentman** | Scalar `sentman(theta,s,TwTi,alphaE) → (Cp,Ctau)`. Use `libm` `erf`/`exp`. Pin M-01 closure. Vectorize over panels in the caller, not inside `sentman`. |
| **A2 Panel force** | Fuse into the vector kernel. Guard `v`, `T`, `m̄`. `t_hat=0` at normal incidence. |
| **A3 Cd/Cl** | Same reduction as Python. Provide `coefficients_only` (no \(\mathbf{r}\times\mathbf{F}\)). |
| **B1 MSIS** | Phase 1: Python pymsis, pass scalars in. Phase 2: link NRLMSIS-00 / MSIS 2.1 C and replicate `SPECIES_MASS_KG` + `m_bar`. Do not write a new atmosphere model. |
| **B2 Exponential ρ** | Port the Vallado table as a static array for OD / `orbit_match`. Keep it out of the RL force model. |
| **C1 J2/J3** | Port `_accel_scalar` literally (it is already scalar C-like Python). Same \(\mu,R_E,J_2,J_3\). |
| **C2 Co-rotating drag** | Implement behind `bc>0` for OD. RL path sets `bc=0` and uses Sentman. Env-level `corotating_atmosphere` (currently `false` in all configs) is a **separate** \(v_\mathrm{rel}\) flag — port it, default false. |
| **C3 COE** | Port `rv_to_coe` / `coe_to_rv`. Keep `teme_to_gcrs` in Python (astropy / IERS). |
| **C4–C5** | Trivial; used in state readout and episode length. |
| **D1–D3 MRP/Euler** | Port as the RK4 RHS. Shadow after each completed step, not mid-stage. |
| **D4 Quat/MRP** | Port if Depth B fuses action handling. |
| **D5 MRP-PD** | Port `proper` and `legacy` inverses, saturation, plant-consistent rate cap. |
| **D6 LVLH** | Port; needed for `action.frame: lvlh` inside a fused step. |
| **E1 Bowring + GMST** | Port; atmosphere query needs it even if MSIS stays in Python (Python can also keep doing this). |
| **E2 Sun + eclipse** | Port the Montenbruck analytic Sun. Basilisk/SPICE Sun stays on the Basilisk path. |
| **E3 SRP** | Port the current cannonball-projected law **as-is** (including \(C_r=1.8\)). Improved sail optics are a later physics change, not a port change. |
| **E4 RK4** | Port the 12-state scheme with stage-wise DCM. Same `h` policy (`ceil(dt/rk4_step)`). |
| **F Reward/obs** | Do not port. Return the scalars they need. |
| **Visible surface** | Phase 3 offline tool. Same silhouette identity test. |
| **Power / GS** | Phase 2 optional. Cheap in Python today; fuse only if Depth B profiling says so. |
| **LQR CARE** | Needs a small dense linear-algebra dependency (`Eigen`) or stay in SciPy at init. Gains are computed once. |
| **B-dot / bang-bang** | Port after MRP-PD. |

### 5.6 Binding strategy

- **pybind11** (recommended): NumPy `dtype=float64` buffers with no copy when
  contiguous; returns tuples / small structs. Matches the rest of the
  scientific Python stack.
- **Do not** use ctypes for the fused step (too many fields, error-prone ABI).
- **Do not** wrap each 3-vector helper as its own Python function for
  training — that is Depth A only, for tests.
- Build a **manylinux / conda** artifact later; for thesis work a local
  `pip install -e` / CMake + scikit-build-core is enough.
- Threading: V1.7 parallelizes with `SubprocVecEnv` (process isolation).
  The C++ library must be **re-entrant per handle** (one `Simulator` object
  per env, no process-global scratch). OpenMP inside one env is optional and
  should default off so 32 processes do not oversubscribe.

### 5.7 What stays a Python responsibility

1. YAML / JSON / geometry file I/O (`geom_loader`, `stl_loader`, trimesh).
2. `ArlamxEnv` Gym API, spaces, termination (`h < alt_floor`), truncation.
3. Domain-randomization **policy** (which knobs, which ranges). The library
   only accepts the drawn numbers.
4. `SensorSuite` noise (and therefore the observation contract).
5. `CompositeReward` and every term in `advisor/reward.py`.
6. `SensorRealisticObservation` and every feature in `advisor/observation.py`.
7. SB3 training, callbacks, VecNormalize, TensorBoard.
8. `advisor/export.py` ONNX / INT8 / STM32 header (policy, not plant).
9. `orbit_match` data ingest (TLE, CSVs), EKF (`v4.py`), reporting, plots.
10. Basilisk construction / SPICE / WMM / RW / MTB effectors.
11. Scripts under `scripts/` and the pytest suite (tests **call** the library).

### 5.8 Numeric identity — the port is wrong until these pass

Bring the V1.7 tests across as oracles. The C++ library must match Python
reference values, not “look reasonable.”

| Gate | Source | Tolerance |
|---|---|---|
| Sentman vs independent Moe/Doornbos oracle | `tests/test_aero.py` | existing pins (post-M-01) |
| Hyperthermal \(C_p\cos+C_\tau\sin\to 2\cos\) | same | ~1e-8 relative at large \(s\) |
| Hex face-on \(C_D\) / peak \(C_L\) | same | V1.7 rebaselined numbers (Cd ~2.31 at \(\alpha_E=0.93\), not the old 2.10) |
| J2/J3 vs \(\nabla V\) | documented 1.15e-9 | ≤ 1e-8 |
| MRP compose / \(C(e)=C_{BN}C_{RN}^\top\) | 1.66e-15 | ≤ 1e-12 |
| Closed-loop MRP regulation to 0 | existing controller tests | 0.001° class |
| `coefficients_only` vs `spacecraft_aero` Cd/Cl | same inputs | 1e-12 relative |
| RK4 one-step vs V1.7 `RK4DynamicsBackend` | shared IC, no aero | 1e-10 class on \(\mathbf{y}\) |
| Fused advisor step vs `ArlamxEnv.step` | frozen config, fixed action | match accumulators to ~1e-9 relative |
| Silhouette identity | `tests/test_visible_surface.py` | by construction |

Do **not** retune rewards or retrain to hide a plant mismatch.

### 5.9 Standing V1.7 caveats — do not bake them in silently

These are live defects / modelling limits. The C++ library should **expose
the same switches** V1.7 has, defaulted the same way, and document them.

1. **`catsat_v7_deployed_rich476.geom` does not represent the CatSat CAD**
   (inverted max-drag axis, fictitious inflatable strip, dropped deployed
   panels). Porting it faithfully preserves a wrong vehicle. V2.0 should
   load **any** panel set and make it easy to swap a shadow-corrected CAD
   extract; it should not hard-code 476.
2. **`corotating_atmosphere: false`** in all configs; OD path uses
   co-rotation (~+11.7 % drag at \(i=23^\circ\)).
3. **Ground stations fixed in inertial space** (M-04) — conditions SC_v2
   downlink results.
4. **SRP is not sail optics** (lumped \(C_r=1.8\), force along \(-\hat{s}\)
   only).
5. **Diagonal inertia, CoM at origin**, tensor not derived from `.geom`
   (`INERTIA_AND_GEOMETRY_NOTE.md`). Accept full \(I\) in the API.
6. **ONNX SAC squash omitted** (M-02) — export-only, not a plant issue.
7. **Self-shadowing is off in the live RL aero sum.** Optional shadowed
   panel set should be an input, not a surprise change of Cd.
8. SC_v1 / SC_v2 **checkpoints were trained before M-01**. Absolute
   drag/decay numbers change in V1.7; policy-vs-policy in one plant is
   still valid. V2.0 must match **V1.7**, not pre-M-01.

---

## 6. Phased plan (still no implementation)

### Phase 0 — Freeze the contract (this folder)

- This document.
- Do not touch V1.7.
- Treat V1.7 tests + M-01 Sentman as the numeric spec.

### Phase 1 — C++ kernels + Depth A bindings

- `sentman`, vector `spacecraft_aero`, `coefficients_only`, panel SRP.
- Gravity `_accel_scalar`, MRP algebra, MRP-PD controller.
- pybind11, pytest calling C++ vs Python on the existing oracles.
- `ArlamxEnv` optionally routes aero/control through the library.
- Success: identical Cd/Cl/F/τ and controller torque; modest speedup.

### Phase 2 — Depth B fused RK4 advisor step

- 12-state RK4 + substep loop in C++.
- Atmosphere still from Python (callback or cached table).
- Python `ArlamxEnv.step` becomes: clip action → `advance_advisor_step` →
  power/GS (still Python) → reward → obs.
- Success: fused step matches V1.7 `ArlamxEnv` on a frozen scenario; RK4
  training (CatSat, and SolarCat-if-switched) is several times faster.

### Phase 3 — Atmosphere and optional extras

- Link NRLMSIS / MSIS 2.1 from C++; drop per-query Python.
- Optional: power and GS inside the fused step (only if profiled).
- Optional: visible-surface extract for CAD-true CatSat panel sets.
- Optional: gravity-gradient / SRP moment for free-attitude studies.
- Basilisk remains the high-fidelity backend with C++ aero injection.

### Phase 4 — Training campaign, not a rewrite

- Same YAML, same reward/obs, same SB3.
- Retrain only if you want numbers against the V1.7 plant at new speed
  (more seeds, longer budgets, honest CatSat geometry).
- Export path unchanged (Python ONNX). Flight C is still the **policy**.

**Out of scope for V2.0 physics library:** new GSI models (Schamberg, CLL)
unless added first in Python and gated; spherical-harmonic gravity;
SPICE; a C++ RL trainer; moving files out of V1.7.

---

## 7. Suggested V2.0 runtime picture

```
                    config YAML  +  .geom/.stl  +  subsystems
                                    │
                                    ▼
                    Python ArlamxEnv  (Gymnasium, DR, sensors)
                       │                         │
          aero+RK4 path│                         │ Basilisk path
                       ▼                         ▼
              libarlamx (C++)              libarlamx aero/control
              fused substep loop           + Basilisk gravity/SPICE/WMM
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                    Python observation + CompositeReward
                                    ▼
                    Stable-Baselines3  (PPO / SAC / TD3 / RecurrentPPO)
                                    ▼
                    advisor.export  →  ONNX / INT8 / STM32 header
```

The scientific identity is the left column of the inner split: **one**
Sentman + MSIS force, injected into **one** of two integrators.

---

## 8. File map: V1.7 source → V2.0 owner

| V1.7 path | Equations | V2.0 owner |
|---|---|---|
| `aero/gas_surface.py` | A1 Sentman + M-01 | `libarlamx` aero |
| `aero/fmf_panel.py` | A2 | fused into aero |
| `aero/spacecraft_aero.py` | A3 | `libarlamx` aero |
| `aero/srp_panel.py` | E3 | `libarlamx` srp |
| `atmosphere/nrlmsise00_wrapper.py` | B1 | Python Phase 1; C FFI Phase 3 |
| `orbit_match/dynamics.py` | B2, C1, C2, E4 | `libarlamx` gravity + integrate |
| `orbit_match/frames.py` | C3 | `libarlamx` frames; TEME stays Py |
| `dynamics/rk4_backend.py` | D1–D3, C4, E2, E4 | `libarlamx` integrate + attitude |
| `dynamics/env.py` helpers | E1, D4, D6, SLERP, power, GS | helpers → lib; hub stays Py |
| `dynamics/environmental_torques.py` | SRP moment, GG | `libarlamx` later |
| `dynamics/basilisk_interface.py` | high-fidelity plant | stay Python/Basilisk |
| `dynamics/sensor_models.py` | noise | Python |
| `control/mrp_feedback.py` | D5 | `libarlamx` control |
| `control/{lqr,bang_bang,b_dot,pd}.py` | optional laws | later |
| `control/controller_factory.py` | YAML wiring | Python |
| `geometry/*` | I/O + visible surface | I/O Python; extract later C++ |
| `advisor/*` | RL, reward, obs, export | **Python only** |
| `main.py` | CLI | Python |

---

## 9. What “done” looks like for this planning step

This folder now holds the overview only. V1.7 is intact.

A later implementation pass (not this document) would be done when:

1. `libarlamx` reproduces V1.7 oracles at the tolerances in §5.8.
2. `ArlamxEnv` can run CatSat training on the fused RK4 path without a
   reward or observation code change.
3. SolarCat can still use Basilisk, with Sentman evaluated in C++.
4. No Deep RL algorithm, network, or export path has been rewritten in C++.
5. Wall-clock env stepping is measurably faster on the same machine / same
   config, with Cd, \(\Delta E\), and closed-loop tracking matching V1.7.

Until then, quote physics numbers from **V1.7**, and treat this file as the
map — not as a new simulator.
