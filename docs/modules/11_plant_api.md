# 11 — Fused plant API

**Folder:** `cpp/` (`api.hpp`) + Python bindings  
**Status:** implemented  
**Tests:** `tests/integration/`

## What it does

One object, one advisor step. Training talks only to this. Kernels stay callable for unit tests.

## Commands (Python names)

```
p = SimParams()            # attitude_law, gsi, ideal_torque, sh_degree, wmm_path, …
sim = Simulator(p)
sim.set_panels(n, A, c)
sim.set_mass(m); sim.set_inertia_diag(I3)
sim.set_atmosphere(rho, T, m_bar, chi=None)   # chi length-7 for CLL
sim.set_dipole_max(m_max)
sim.set_mode("point" | "detumble" | "prescribed")
state = sim.reset(r, v, sigma0, omega0)
out  = sim.step(q4)
```

There is **no** `set_space_weather`. F10.7 / Ap live in Python (`query_msis` / physics YAML).

`step` returns at least:

`r, v, sigma, omega, altitude_km, sma_m, eclipse, sun_B, B_B, Cd, Cl, drag_N, dE_actual, dE_baseline, dE_drag, dE_lift, tracking_err_rad, n_substeps, tau_ctrl_mean, tau_aero_mean, m_mean, tau_cmd_mean, tau_shortfall_frac, tau_demand_mean, tau_demand_absmean, gyro_mean`

v2.6 additions:

* `tau_demand_mean` / `tau_demand_absmean` — the PD demand **before** the per-axis
  `max_torque` clip (signed mean / mean of |.|). `tau_cmd_mean` is the clipped
  (gated) demand. The v8 delegation counterfactual reads the former.
* `gyro_mean` — RK4-consistent mean of \(\boldsymbol{\omega}\times I\boldsymbol{\omega}\)
  over the step (same stage states and Simpson weights as the \(\omega\) update),
  so the flight-side rotational identity is exact to round-off at any rate:

  \[
  \frac{I(\boldsymbol{\omega}_1-\boldsymbol{\omega}_0)}{T}
  = \bar{\boldsymbol{\tau}}_\mathrm{aero}+\bar{\boldsymbol{\tau}}_\mathrm{ctrl}-\overline{\boldsymbol{\omega}\times I\boldsymbol{\omega}},
  \qquad T=n_\mathrm{substeps}\,\Delta t .
  \]

  `tests/integration/test_plant_step.py` pins this on the closed-loop plant.

`ideal_torque=true` applies the PD torque (MRP or quaternion) as a body couple. Default is `apply_magnetorquer`. `gsi` is `"sentman"` (default) or `"cll"`. Unknown `attitude_law` / `gsi` strings throw.

## Integrator contract

- RK4 on 12 states.
- Harmonics + 3rd body + aero + SRP + control evaluated as in `APPROVAL_PLAN.md` §5.
- Aero/SRP/control body loads ZOH across `dt`; gravity/3rd-body at each stage.
- `n_substeps = round(advisor_step_s / dt_s)`.
- **Energy bookkeeping (v2.1):** `dE_actual`, `dE_drag`, `dE_lift` and `dE_baseline` are all work per unit mass on the *inertial* motion, \(\mathbf{F}\cdot\mathbf{v}\,\Delta t/m\). The min-drag counterfactual force is \(-F_\mathrm{base}\hat{v}_\mathrm{rel}\), so its work is \(-F_\mathrm{base}(\hat{v}_\mathrm{rel}\cdot\mathbf{v})\). v2.0 charged \(-F_\mathrm{base}|\mathbf{v}_\mathrm{rel}|\) (co-rotating-frame work), ~6 % low on a prograde orbit. The min-drag coefficient is evaluated once per advisor step and refreshed if \(|\mathbf{v}_\mathrm{rel}|\) drifts by more than 2e-4.

## Citations

The API has no new physics. Integrator citations live in `06_attitude_dynamics.md`. Gymnasium API: Towers et al., *Gymnasium* (Farama, 2023) — Python side only.

## Classroom test

**A.** Zero aero (vacuum), no 3rd body, degree-0 gravity: circular 400 km equatorial, 1 period, SMA change \(\lt 1\,\mathrm{m}\) (RK4 sanity).  
**B.** Face-on plate, dense atmosphere (fixed \(\rho\)), inertial attitude held: energy decreases; \(C_D\) matches `01` kernel.  
**C.** `set_mode("detumble")` with a dipole field reduces \(|\omega|\) from 5 °/s.

## Tests run

2026-08-18 — `tests/integration/test_plant_step.py` **PASS**. Gym `step` ~2 ms / 300 s advisor step (~1.8k PPO fps on 4 envs).
2026-09-04 (v2.1) — `test_baseline_work_is_in_the_inertial_frame` added and **PASS**. Plant 0.40 ms / 300 s advisor step (72 panels, SH4, SRP; v2.0: 0.49 ms with fewer wetted faces); Gym `step` v3 0.60 ms, v10 1.26 ms (v2.0: 0.75 / 2.57 ms).
