# 02 — Repository map and toolchain

Root: `ARLAMX V2.1/` (the folder name still says 2.1; package version is 2.6).
Not a git repository as of 2026-09-13. `ARLAMX-V1.7/` (sibling, read-only) is
the predecessor and must never be edited.

## Layout

```
cpp/                     C++17 plant (pybind11 module arlamx_cpp)
  include/arlamx/        types.hpp (Vec3/Mat3/PanelSoA/State12/Atmo/StepOut), constants.hpp
    aero/                sentman.hpp, cll.hpp
    srp/                 panel_srp.hpp
    orbit/               gravity.hpp (GGM03S), third_body.hpp (Sun/Moon, optional CSPICE), frames.hpp
    attitude/            mrp.hpp, integrate.hpp (RK4)
    control/             mrp_feedback, quat_feedback, bdot, magnetorquer
    mag/                 field.hpp (tilted dipole + WMM)
    onboard/             propagator_f32.hpp (flight-computer two-body+J2+drag, binary32)
    api.hpp              SimParams, Simulator (fused advisor step)
  src/                   matching .cpp; bindings.cpp
  CMakeLists.txt
python/
  arlamx_v2/             package (72 modules) — see 05/10 for the map
    arlamx_cpp*.so       built plant (python/arlamx_v2/, cpython-312)
  configs/               19 YAML: physics*.yaml, reward_v8a/b, v9, v10, v11, v12, gains_mrp,
                         power_mtq, estimator_kf, sensors_solarcat, duo, mpc_v1/v2/v3, sweep_example
data/                    GGM03S.txt, WMM.COF (2020), WMM2025.COF, earthcup_hex_v3.geom,
                         SolarCat_Assembly.STL, SolarCat_6Petals.stl
tests/                   one folder per module + integration, physics, reward, estimators,
                         validation/verify_physics.py, basilisk_ref (skips w/o Basilisk)
docs/                    modules/01–17 (equation notes), COMMANDS.md, VALIDATION_STANDARD.md,
                         CLL_TEST_SPEC.md, v12/ (campaign plans), handoff/ (this bundle)
outputs/                 every artifact — map in 09
ACEnv/                   audit/critique/campaign reports (Reports/*.md), agent use log
build/                   CMake build tree (disposable)
```

Top-level documents: `README.md`, `APPROVAL_PLAN.md` (build contract),
`ARLAMX_V2.0_OVERVIEW_AND_CPP_PLAN.md` (V1.7 survey and equation map),
`claude_handoff.md`, `HANDOFF_2026-08-21_STATE.md`, `CHANGELOG_v2.1/2.5/2.6.md`.

## Environment

Canonical interpreter: mamba env **`arlamx`**, Python **3.12**, at
`~/.local/share/mamba/envs/arlamx/bin/python`. `environment.yml` is the
intended install (conda-forge CPU pytorch + stable-baselines3; `pymsis` via
pip). Do not `pip install -r requirements.txt` into that env.

```bash
mamba env update -n arlamx -f environment.yml
mamba activate arlamx
./build.sh                 # refuses any non-3.12 interpreter; drops stale CMake cache
PYTHONPATH=python python -c "from arlamx_v2 import cpp, __version__; print(__version__, cpp.__version__)"   # 2.6 2.6
PYTHONPATH=python python -m pytest tests -q            # 208 passed
PYTHONPATH=python python tests/validation/verify_physics.py   # independent kernel cross-check
```

Optional: CSPICE (Conan path hard-coded in `cpp/CMakeLists.txt`); analytic
Sun/Moon otherwise. Basilisk only for `tests/basilisk_ref/`.

Historical interpreters seen in reports: `ThesisMS` (miniforge, 3.11) and
`~/.venvs/arlamx21` (3.11). Neither builds v2.5+; the 3.11 `.so` was removed.

## Data resolution (`python/arlamx_v2/paths.py`)

Order: environment variables `ARLAMX_GGM`, `ARLAMX_WMM`, `ARLAMX_HEX_GEOM`,
`ARLAMX_SOLARCAT_STL`, `ARLAMX_V17` → repo-local `data/` → historical
Basilisk/V1.7 absolute paths. `resolve_ggm()` / `resolve_wmm()` return `""`
when nothing is found; the env warns and falls back to the tilted dipole for
magnetics, and (since 2026-09-13) throws if a *requested* GGM path fails to load.

## Physics presets (`python/configs/physics*.yaml`)

| Preset | Gravity | Atmosphere | GSI | Magnetics | Actuator |
|---|---|---|---|---|---|
| `fast` | GGM degree 2 | MSIS 2.1, F10.7 150, Ap 4, corotating | Sentman α_E 0.93 | tilted dipole | coils (`ideal_torque: false`) |
| `standard` | GGM degree 4 | same | Sentman α_E 0.93 | WMM | coils |
| `high` | GGM degree 8 + lunisolar | same | CLL α_N 0.93, σ_T 1 + species mix | WMM | coils |

`standard` reproduces the v2.5 campaign plant *except* for the actuator: the
campaigns ran with an ideal couple. Override per run with a YAML path or
`physics.load(name, overrides={...})`.

## Validation standard (`docs/VALIDATION_STANDARD.md`)

A module is done when it has: a spec in `docs/modules/`, two literature
citations, one classroom test (fixed inputs, checkable by hand), pytest code
under `tests/<module>/` that writes nothing to `outputs/`, and a recorded
result. Tolerances: aero identities 1e-8 rel, gravity 1e-8 rel, MRP algebra
1e-12 abs, closed-loop angle 0.01°, geometry seam 1e-9 m. Rule: never retune a
reward to hide a failed plant test.

## Performance reference (v2.1 audit, 16-core desktop)

Plant `Simulator.step` 0.40 ms per 300 s advisor step (72 panels, degree 4,
SRP, MSIS held); Gym `env.step` v3 0.60 ms, v10 1.26 ms; PPO with 16
SubprocVecEnv workers ~1200 steps/s, where pipe traffic (40-key `info`) and
the SB3 update phase dominate, not physics.
