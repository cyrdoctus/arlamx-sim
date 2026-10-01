# ARLAMX v2.5 — change notes (2026-09-10)

Scope: control-path, Gym, toolchain, and binding updates on top of the
audited v2.1 plant. Sentman / energy bookkeeping / gravity kernels are
unchanged from v2.1. Campaign READMEs, output plot titles, and audit
reports that quote v2.1 numbers are left as historical artifacts.

v2.1 equation history: [`CHANGELOG_v2.1.md`](CHANGELOG_v2.1.md).

---

## Closed-loop MTQ

The plant applies magnetorquer dipole command through the local magnetic
field (`τ = m × B`) rather than treating MRP-PD demand as an ideal body
torque (`ideal_torque` remains available for bake-offs). Per-axis coil
saturation (`m_max`, torque caps) and B-dot detumble sit on the same
closed-loop path the Gym env uses.

## Quaternion law (MRP default)

`attitude_law` / Gym `controller` accept `"mrp"` or `"quaternion"`.
**MRP stays the default.** Both laws use `τ = -kp·err - kd·ωerr + gyro`,
but `qv ≈ θ/2` while `σ ≈ θ/4`, so the same yaml `kp` is ~2× stiffer on
the quaternion path (`quat_feedback.hpp`). That is a unit difference, not
an equivalent PD rewrite. It does not change aero, SRP, or orbit physics.

## WMM-in-Gym

The Gym env resolves `WMM.COF` through `paths.resolve_wmm()` and loads it
into the plant. A requested COF that fails to load is a hard error (no
silent tilted-dipole fallback). An empty resolve path still warns and
falls back to the dipole, so training without a COF is explicit.

## Mamba env `arlamx` / Python 3.12

Canonical interpreter is the local mamba env named `arlamx` (Python 3.12).
`./build.sh` refuses any other minor version and prefers that env’s
`cmake` / `c++`. Intended install is `environment.yml` (conda-forge CPU
`pytorch` + `stable-baselines3`; `pymsis` via pip).

## FP32 train + `quantize_int8`

PPO/SAC/TD3 training stays FP32. INT8 is a separately callable post-train
step (`python/arlamx_v2/quantize.py`, `quantize_int8`): dynamic quant of
actor `nn.Linear` only. Not imported by `train_one` or `train_v12.train_run`.

```bash
PYTHONPATH=python python -m arlamx_v2.quantize --algo ppo --in PATH --out PATH
```

## Binding fixes

pybind11 module `arlamx_cpp` version and docstring track the package
(`cpp.__version__ == "2.5"`). Array-length checks and exposed plant
controls (MRP feedback, WMM, onboard FP32 propagator) stay aligned with
the C++ API. The Python 3.11 leftover `.so` (stamped v2.1) is removed so
an ABI-matching 3.11 import cannot report package 2.5 vs cpp 2.1.

## Version stamps

* `python/arlamx_v2/__init__.py`: `__version__ = "2.5"`
* `cpp/src/bindings.cpp`: `m.attr("__version__") = "2.5"`, docstring ARLAMX v2.5

Confirm after `./build.sh`:

```bash
PYTHONPATH=python ~/.local/share/mamba/envs/arlamx/bin/python -c \
  "from arlamx_v2 import cpp, __version__; print(__version__, cpp.__version__)"
```
