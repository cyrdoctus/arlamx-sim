# ARLAMX v2.6 — change notes (2026-09-13)

Version bump from **v2.5 → v2.6**. Review this file plus
[`docs/COMMANDS.md`](docs/COMMANDS.md) in Claude Code / Fable 5.1. Do not
treat [`CHANGELOG_v2.5.md`](CHANGELOG_v2.5.md) as current.

Scope: command surface, YAML physics, and a Walker–CLL free-molecular option
on top of the v2.5 closed-loop plant. Sentman remains the packaged default.
The onboard FP32 propagator is unchanged (no CLL, no harmonics).

---

## CLI and sweeps

`python -m arlamx_v2` dispatches `list`, `train`, `sweep`, `quantize`,
`eval`, `decay`. Other module mains stay as aliases. YAML sweeps
(`python/configs/sweep_example.yaml`) product over algo / controller /
physics / seeds and resume from a CSV ledger. No LLM proposer.

## Physics config

`python/configs/physics.yaml` (presets `fast` / `standard` / `high`) sets
gravity degree, lunisolar, GSI, MSIS version, WMM vs dipole, SRP, and
RK4 step. `ArlamxV2Env(..., physics=..., gsi=...)` and
`python -m arlamx_v2 train --physics high` apply it. Resolved physics is
frozen next to v8+ `snapshot.json`.

## Walker CLL (optional)

C++ `cll()` implements Walker, Mehta & Koller (2014) modified closed-form
coefficients (ADBSat / arXiv:2104.05543 eqs. 11–15), species-mixed from
MSIS mole fractions. `SimParams.gsi = "sentman" | "cll"`. Unknown strings
throw. Same panel sum, same \(A_\mathrm{ref}\), same inertial \(\mathbf{F}\cdot\mathbf{v}\).
Fully accommodated \(\alpha_N=\sigma_T=1\), \(T_w=T_i\) recovers Sentman
\(\alpha_E=1\) on a face-on plate.

Classroom tests for CLL are specified in [`docs/modules/17_aero_cll.md`](docs/modules/17_aero_cll.md)
and [`docs/CLL_TEST_SPEC.md`](docs/CLL_TEST_SPEC.md). Wiring tests live in
`tests/physics/test_physics_wiring.py` and `tests/aero/test_cll.py`.

Review pass (same version): high \(\alpha_N=0.93\) is an isolation choice
(same number as campaign Sentman \(\alpha_E\), not a published Walker
\(\alpha_N\)); `twobody` keeps the GGM, evaluates degree 0, and a missing
GGM file throws instead of falling through to J2+J3; `train_v12` train
**and eval** envs take physics/gsi/controller; v12 sweep resume looks in
`outputs/v12/runs/<name>`; GSI names are canonicalized on a copy (validate
does not mutate its input); mixture lee-skip is per-species; anomalous O
folds into O.

### Closed-loop plant: two fixed bugs and one finding

Both tests that were red after v2.5 (and `xfail` in the first v2.6 review pass)
are fixed and un-marked. Neither was the GSI layer.

| test | cause | fix |
|---|---|---|
| `tests/test_v7_ipc.py::test_observer_recovers_aero_torque` | Observer used the endpoint gyroscopic term \(\boldsymbol{\omega}_1\times I\boldsymbol{\omega}_1\). Under the closed-loop magnetorquer the rates drift inside a step and that term alone is the size of the aero torque (cos 0.83, magnitude 1.8×). | Plant exports the RK4-consistent mean `gyro_mean`; `torque_measurement` / the v7 observer use it. Identity exact to 1e-15 at any rate (`test_plant_step.py`). |
| `tests/integration/test_env_v8.py::test_deleg_P_fires_on_saturating_slew` | `tau_want` was the plant’s **gated** command (`tau_cmd_mean` ≤ gate·τmax by construction), so eq. (4)’s “gate binding” guard could never fire: `deleg_P` was identically 0 and the v8b hypothesis was not in the reward. | Plant exports the full-authority demand (`tau_demand_mean`, `tau_demand_absmean`); eq. (4) is now command-side (`E_full − E_gate`) so the B-projection loss is not credited to the gate. |

Finding, not fixed: with `ideal_torque=false` (the packaged default since
v2.5) the plant **cannot hold attitude with the v7/v8 gains** — 70–120° mean
tracking error at every inclination, also with no disturbance. The actuator has
no authority about \(\hat{\mathbf{B}}\) and the gains are three orders too
stiff for the field-rotation timescale. Every campaign result to date was
produced on `ideal_torque=true`. `attitude.ideal_torque` is now a declared,
snapshotted physics key (default `false`). Numbers and mechanism:
[`docs/modules/15_control_magnetorquer.md`](docs/modules/15_control_magnetorquer.md).
Gains were **not** retuned; that is a research decision.

Full `tests/` after this pass: **208 passed, 0 xfailed** (the earlier
"200 passed, 1 skipped, 2 xfailed" line was the pre-fix state).

---

## Version strings

* `python/arlamx_v2/__init__.py`: `__version__ = "2.6"`
* `cpp/src/bindings.cpp`: `m.attr("__version__") = "2.6"`

```bash
PYTHONPATH=python python -c "from arlamx_v2 import cpp, __version__; print(__version__, cpp.__version__)"
```

---

## Files added or rewritten for this bump

* `python/arlamx_v2/__main__.py`, `cli.py`, `sweep.py`, `physics.py`
* `python/configs/physics.yaml`, `physics_fast.yaml`, `physics_high.yaml`, `sweep_example.yaml`
* `cpp/include/arlamx/aero/cll.hpp`, `cpp/src/aero/cll.cpp`
* `docs/COMMANDS.md`, `docs/CLL_TEST_SPEC.md`
* `docs/modules/15_control_magnetorquer.md`, `16_control_quat_feedback.md`, `17_aero_cll.md`

Plant / Gym wiring: `sentman.cpp` `GsiParams` switch, `SimParams.gsi`,
`Atmo.chi`, `env.py` / `decay_run.py` / `atmosphere.py` / `train.py`.
