# ARLAMX v2.6 — command catalog

Prefix every example with `PYTHONPATH=python` from the repo root, in the
`arlamx` mamba env (Python 3.12). After `./build.sh`:

```bash
PYTHONPATH=python python -m arlamx_v2 list
```

`python -m arlamx_v2.<mod>` still works. The dispatcher is the documented
front door. Per-command flags: `python -m arlamx_v2 <cmd> -h`.

This file is the review surface for Fable 5.1 (commands + flags + physics).
What v2.6 changed: [`CHANGELOG_v2.6.md`](../CHANGELOG_v2.6.md).

---

## 1. First-class dispatcher commands

### `list` / `help`

Print the in-process catalog (same groups as this file, shorter).
`help` does **not** dump argparse. Per-command flags: `python -m arlamx_v2 <cmd> -h`.

```bash
python -m arlamx_v2 list
python -m arlamx_v2 help
```

### `train`

FP32 PPO / SAC / TD3 via `train.train_one`. Controller defaults to the
physics YAML (`attitude.law`). Quantization is **not** applied here.

| Flag | Default | Meaning |
|---|---|---|
| `--variant` | `v4a` | `v3` `v4a/b` `v5a/b` `v6` `v7` `v8a/b` `v9a/b` |
| `--algo` | `ppo` | `ppo` `sac` `td3` |
| `--timesteps` | `300000` | env steps |
| `--n-envs` | `32` | vectorized envs |
| `--seed` | `42` | |
| `--arch` | `4x16` | `LAYERSxWIDTH` → `[width]*layers` |
| `--controller` | unset | `mrp` `quaternion` (overrides YAML `attitude.law`) |
| `--physics` | `standard` | `fast` `standard` `high` or a YAML path |
| `--gsi` | unset | `sentman` `cll` (overrides physics YAML) |
| `--out` | `outputs/training` | run root |
| `--name` | auto | `SC_{variant}_{arch}_{ALGO}_{k}k` |

```bash
python -m arlamx_v2 train --variant v8a --algo ppo --physics high
python -m arlamx_v2 train --physics python/configs/physics.yaml --gsi cll --controller mrp
```

v8+ writes `snapshot.json` (reward, gains, power, estimator, **resolved physics**).

### `sweep`

YAML product or explicit `arms`. Calls `train.train_one` or
`train_v12.train_run`. Skips a name if `metrics.json` exists or it is
already in the ledger. `--force` redoes.

| Flag | Meaning |
|---|---|
| `--config` | required YAML (see `python/configs/sweep_example.yaml`) |
| `--force` | ignore ledger / metrics.json |

Sweep YAML keys: `name`, `backend` (`train`\|`train_v12`), `variant`,
`algo` (scalar or list), `controller`, `physics`, `gsi`, `seeds`,
`timesteps`, `n_envs`, `arch`, `out`, optional `arms`, `ppo_kw`, `v12_kw`.
Ledger: `outputs/training/SWEEP_LEDGER.csv` (or `ledger:` in the YAML).

```bash
python -m arlamx_v2 sweep --config python/configs/sweep_example.yaml
```

### `quantize`

Dynamic INT8 of actor `nn.Linear` only (PPO / SAC / TD3). Training stays
FP32. Refuses to overwrite the source zip.

| Flag | Meaning |
|---|---|
| `--in` | zip or dir with `model.zip` |
| `--out` | distinct destination zip |
| `--algo` | optional `ppo` `sac` `td3` (else inferred from path / class) |

```bash
python -m arlamx_v2 quantize --in PATH --out PATH_INT8
```

### `eval`

Thin wrappers. Remaining flags go to the existing module main.

```bash
python -m arlamx_v2 eval mc_decay --runs 24
python -m arlamx_v2 eval mc_v12 --runs 24
```

### `decay`

Prescribed min/max-drag orbital decay (`decay_run.py`). Same `--physics`
/ `--gsi` as train.

| Flag | Default | Meaning |
|---|---|---|
| `--geom` | `hex` | `hex` `stl1pct` |
| `--kind` | `both` | `min` `max` `both` |
| `--alt0` / `--alt-stop` | `500` / `250` | km |
| `--inc` / `--ecc` | `23` / `0.001` | |
| `--dt` | `5` | s |
| `--max-days` | `800` | |
| `--f107` / `--ap` | `150` / `4` | |
| `--mass` | `0.625` | kg |
| `--physics` | `standard` | preset or path |
| `--gsi` | unset | `sentman` `cll` |
| `--out` | `outputs/analysis/decay` | |

```bash
python -m arlamx_v2 decay --physics standard --kind both
```

---

## 2. Physics presets (`--physics`)

Loaded by `arlamx_v2.physics.load`. Applied to `SimParams` in the Gym env
and in `decay_run`. Unknown model names raise.

| Preset | File | Gravity | GSI | Magnetics |
|---|---|---|---|---|
| `fast` | `python/configs/physics_fast.yaml` | GGM degree 2 | Sentman | tilted dipole |
| `standard` | `python/configs/physics.yaml` | GGM degree 4 | Sentman | WMM |
| `high` | `python/configs/physics_high.yaml` | GGM degree 8 + lunisolar | Walker CLL at \(\alpha_N=0.93\) + MSIS species mix | WMM |

A YAML path is also accepted. Keys: `gravity` (`ggm03s`\|`twobody`,
degree 0–24, `lunisolar`), `atmosphere` (`msis21`\|`nrlmsise00`\|`constant`,
`f107`, `ap`, `corotating`), `aero` (`sentman`\|`cll`, `alpha_E`, `alpha_n`,
`alpha_t`, `T_w`), `srp`, `magnetics` (`wmm`\|`dipole`), `attitude.law`
(`mrp`\|`quaternion`), `attitude.ideal_torque` (`false` = coils through the
field, `true` = ideal body couple), `integrator.rk4_step_s`.

Notes:

* `gravity.model: twobody` is spherical-harmonic degree 0 (point mass). The
  GGM file stays loaded so the plant does **not** fall through to the no-file
  two-body+J2+J3 path.
* `alpha_E` is Sentman-only. Under `gsi: cll` the live knobs are `alpha_n`
  and `alpha_t`. \(\alpha_N\) is Walker’s normal-energy accommodation, not
  Sentman \(\alpha_E\) (Moe & Moe 2005). Walker 2014 Table 2 / ADBSat take
  \(\alpha_N\) as a free input. High’s \(0.93\) is an isolation choice (same
  number as the campaign Sentman \(\alpha_E\)) so standard-vs-high isolates
  the kernel + species mix. Face-on plate at \(s=8\), \(T_w/T_i=1/3\):
  Sentman \(0.93\) Cd \(\approx 2.37\); CLL \(\alpha_N=0.93\) \(\approx 2.56\)
  (\(+8\%\)); CLL \(\alpha_N=1\) \(\approx 2.15\) (\(-9\%\)). Sweep
  \(\alpha_N\) if you need that sensitivity. \(\alpha_N=1\) is
  Schaaf–Chambre, not the Walker fit.
* `atmosphere.f107` / `ap` **seed** the env. v4+ episode reset overwrites
  them from the training envelope unless `reset(options={f107, ap})` is
  passed. Decay CLI `--f107` / `--ap` default to the file.

CLI `--gsi` and `--controller` override the file when set. Omit them to
keep the YAML. Sentman remains the packaged default. The onboard FP32
propagator does not read this file.

---

## 3. Registered aliases

`python -m arlamx_v2 <alias>` execs that module’s `main`. Flags are the
module’s own argparse (use `-h`).

| Alias | Module | Purpose |
|---|---|---|
| `train_v12` | `arlamx_v2.train_v12` | SC_v12–v14 campaign trainer |
| `train_v4` | `arlamx_v2.train_v4` | Sequential SC_v4a/v4b matrix |
| `train_v56` | `arlamx_v2.train_v56` | SC_v5 then SC_v6 schedule |
| `aoa_decay` | `arlamx_v2.aoa_decay` | Fixed-AoA decay + lift/drag log |
| `lift_drag_study` | `arlamx_v2.lift_drag_study` | Panel Cd/Cl vs AoA |
| `srp_assess` | `arlamx_v2.srp_assess` | SRP vs drag along-track power |
| `validate_attitudes` | `arlamx_v2.validate_attitudes` | STL simplify 1% area check |
| `bench_v8` | `arlamx_v2.bench_v8` | SC_v8 / Duo / v9 campaign |
| `mc_decay` | `arlamx_v2.mc_decay` | Monte-Carlo 500→300 km decay |
| `mc_v12` | `arlamx_v2.mc_v12` | SC_v12 lifetime vs MPC |
| `mc_four` | `arlamx_v2.mc_four` | Four-task MC (SC_v13 vs MPC) |
| `plot_v8` | `arlamx_v2.plot_v8` | v8 conference figures |
| `plot_v12` | `arlamx_v2.plot_v12` | v12 figures + ADCS animation |
| `campaign` | `arlamx_v2.campaign` | SC_v7 IPC tuning + ledger |
| `decay_run` | `arlamx_v2.decay_run` | same as first-class `decay` |
| `quantize` | `arlamx_v2.quantize` | same as first-class `quantize` |

---

## 4. Other module mains (not rewritten)

Still `python -m arlamx_v2.<mod>`. Not first-class; use when you already
know the campaign script.

| Module | Purpose |
|---|---|
| `srp_f107_sweep` | F10.7 sweep of an SRP-aware strategy |
| `tune_ppo` | PPO burn-in hyperparameter trials |
| `tune_mpc` / `tune_mpc_v3` | Sampling-MPC gain sprints |
| `eval_v4` | SC_v4 bake-off vs heuristic / MPC |
| `analyze_ood` | OOD / weather-spike / brownout |
| `gradient70` / `plot_gradient70` | 70×70 space-weather grids |
| `advisor_run` | Heuristic / MPC on the 1% STL plant |
| `optics_run` | Sail optical-preset forces |
| `damage_eval` | SC_v11 hole / tear / multi |
| `v10_tune` | SC_v10 reward burn-in |
| `report_v8` / `build_mpc_report` | Campaign PDF assembly |
| `plot_production` / `anim_production` | SC_v7 presentation figures |
| `showcase_runs` / `showcase_plots` | Conference time-series + figures |
| `ipc_animation` | 3D IPC disturbance response |
| `plot_simplify` | Mesh vs sealed-side plates |
| `inference_budget` | STM32U575 FLOP budget |
| `check_mpc_ref` | MPC_v1 vs v2 integrity |
| `run_v5_then_v6` / `run_v5v6_pipeline` | v5→v6 train + OOD |
| `pack_presentation` / `slides` | Talk bundle |

---

## 5. Build (not a Python command)

```bash
mamba activate arlamx
./build.sh
```

Refuses a non-3.12 interpreter. Writes
`python/arlamx_v2/arlamx_cpp.cpython-312-*.so`. Check:

```bash
PYTHONPATH=python python -c "from arlamx_v2 import cpp, __version__; print(__version__, cpp.__version__)"
```

Expect `2.6 2.6`.
