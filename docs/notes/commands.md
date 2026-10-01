# ARLAMX v2.7 — commands

The short version of this file is the user guide. Open [`../guide/index.html`](../guide/index.html) in a browser.

Everything goes through `main.py` in the repo root. It adds `python/` to the path itself, so you do not need
`PYTHONPATH`.

```bash
python main.py list        # every command and tool
```

## 0. Interpreter

Use a Python 3.12 environment that has numpy, torch, stable-baselines3, gymnasium, pymsis, trimesh and pyyaml.
The `arlamx` mamba environment that `environment.yml` describes is **not** on this machine right now.
`~/miniforge3/envs/ThesisMS` works with the C++ library that is already built:

```bash
PY=~/miniforge3/envs/ThesisMS/bin/python
$PY main.py test                  # 259 tests
$PY main.py verify                # independent physics cross-check
```

Rebuild the C++ plant (only after C++ edits): `ARLAMX_PYTHON=$PY ./build.sh`.

## 1. Where the settings are

| File | What it holds |
|---|---|
| `config/network.yaml` | neural net: `algo` (ppo/sac/td3), `layers`, `width`, `timesteps`, `n_envs`, `seed`, `torch_threads`, one hyperparameter block per algo, `vecnormalize` |
| `config/network_quick.yaml` | same, sized for a smoke run (20k steps, 4 envs) |
| `config/orbit.yaml` | `training:` orbit/weather envelope drawn at every episode reset · `simulation:` initial orbit for `decay` |
| `config/train.yaml` | session: `backend` (train/v12), `variant`, `physics`, `gsi`, `controller`, `reward_w`, `v12:` wrapper knobs |
| `config/sweep.yaml` | example sweep (`base` + `grid` or `arms`) |
| `config/plant/physics*.yaml` | gravity, atmosphere, GSI, SRP, magnetics, attitude law, integrator (`--physics fast/standard/high`) |
| `config/plant/reward_*.yaml` | reward weights per variant (v8a, v8b, v9, v10, v11, v12) |
| `config/plant/gains_mrp.yaml`, `power_mtq.yaml`, `estimator_kf.yaml`, `sensors_solarcat.yaml` | controller gains, coils + vehicle card, torque KF, sensor noise |
| `config/plant/mpc_v*.yaml`, `duo.yaml` | MPC baselines, Duo arbiter |
| `config/plant/vehicle.yaml` | six-rod hex vehicle for `v10r6`: mass budget, rod dipole/power, cp offset |

How values are read in `orbit.yaml → training`:

- `[lo, hi]`: drawn uniformly at every episode.
- a number: fixed.
- `null`: the variant's built-in default.

The shipped file holds the v10 envelope. Set `training: {}` to get every variant's own historical envelope.

## 2. Train

```bash
python main.py train                                        # config/*.yaml as they are
python main.py train --network config/network_quick.yaml    # small smoke run
python main.py train --algo sac --arch 3x32 --timesteps 500000 --n-envs 16 --seed 7
python main.py train --variant v11 --physics high --gsi cll --controller quaternion
python main.py train --name storm --set orbit.training.ap=[100,200] --set network.ppo.learning_rate=1e-4
python main.py train --set 'train.reward_w={dE_weight: 3.0}'
python main.py train --backend v12 --arch 4x18 --timesteps 600000 --set train.v12.dual_eval=true
python main.py train --variant v10r6                      # six beam rods, 0.750 kg vehicle, 6 on/off gates
```

- Precedence: flags and `--set` override the YAML files. `--network/--orbit/--train FILE` swap in a whole file.
- `--set` takes `section.key=value`, with the value read as YAML. Sections: `network`, `orbit`, `train`.
- Backends:
  - `train`: plain SB3 on the env reward.
  - `v12`: PPO only, env v10 + V12Wrapper, a checkpoint and a 10-orbit eval every 50k steps, and `best/`.
- Training is FP32. INT8 is a separate step: `python main.py quantize --in outputs/models/<id>/models/ppo_<id>.zip --out /path/int8.zip`.

## 3. Snapshots (every training session)

A session with id `MM-DD-HH-MM_<ALGO>_<LAYERS>x<WIDTH>[_name]` writes two things:

- `outputs/snapshots/<id>.yaml`:
  - `session`: id, time, status (running/done/failed/interrupted), error, command, replay command, versions, wall time
  - `network`, `orbit`, `train`: the resolved settings
  - `plant`: the resolved physics, reward, gains, power, estimator and sensors actually used
  - `result`: the final metrics
- `outputs/models/<id>/`: `models/<algo>_<id>.zip`, `models/vecnormalize.pkl`, `logs/` (TensorBoard), `metrics.json`.
  The v12 backend also writes `best/`, `ckpts/`, `eval_curve.csv` and `BEST.md`.

The snapshot is written **before** training starts and updated when training ends, so crashed runs are recorded too.
The plant values are pinned for the whole run, so editing the YAMLs mid-run has no effect on it.

```bash
python main.py snapshots                                                 # list them
python main.py train --from-snapshot outputs/snapshots/09-25-15-42_PPO_4x16.yaml            # exact replay
python main.py train --from-snapshot outputs/snapshots/09-25-15-42_PPO_4x16.yaml --timesteps 1000000 --seed 43
```

When replaying, network or orbit changes keep the snapshot's plant values. Changing `train.variant`, `backend`,
`physics`, `gsi` or `reward_w` re-reads the plant from `config/plant/`.

## 4. Sweeps

```yaml
# config/sweep.yaml
name: example
base: {network.timesteps: 100000, network.n_envs: 8, train.variant: v8a}
grid: {network.algo: [ppo, sac], train.controller: [mrp, quaternion]}   # 4 sessions
# arms: [{train.physics: high}, {network.width: 32}]                     # or an explicit list
```

```bash
python main.py sweep --config config/sweep.yaml          # arms marked done are skipped
python main.py sweep --config config/sweep.yaml --force
```

Each arm is a normal session with its own snapshot. The ledger is `outputs/results/sweeps/<name>.csv`.

## 4b. Physics switches (config/plant/physics*.yaml)

`srp.optical` + `srp.optics` (plate law, default `al_mylar`), `srp.earth` (Earth IR + albedo), `srp.ir`
(thermal partition), `gravity.gradient_torque`, `atmosphere.interp` (2 MSIS samples per step). A scalar
`ap` is a steady-Ap scenario; `query_msis` also accepts a 7-element Ap history (storm-time mode).
Details and sources: `../modules/02_srp.md`, `05_atmosphere.md`, `06_attitude_dynamics.md`,
`15_control_magnetorquer.md`.

## 5. Simulations

```bash
python main.py decay                                   # config/orbit.yaml simulation:, min + max drag
python main.py decay --kind min --physics high --set orbit.simulation.alt0_km=450
python main.py sim aoa_decay -h
python main.py sim lift_drag_study -h
python main.py sim srp_assess -h
python main.py sim srp_f107_sweep -h
python main.py sim optics_run -h
python main.py sim advisor_run -h
python main.py sim validate_attitudes
python main.py sim plot_simplify
python main.py sim geometry --in part.step          # .stl/.obj/.step -> simplified .geom (STEP needs cascadio)
```

Output goes to `outputs/results/<tool>/`. `decay` writes `outputs/results/decay/<stamp>_<kind>_<physics>/` together
with the `settings.yaml` it used.

## 6. Evaluations, figures, old campaigns (work on `outputs/.old`)

```bash
python main.py eval mc_decay --runs 24
python main.py eval mc_v12 --runs 24
python main.py eval damage_eval --stage eval
python main.py plot plot_v8 --set 3
python main.py legacy bench_v8 --stage ref
python main.py run <any_module> [flags]
```

These tools read the pre-v2.7 campaign artifacts (v8, v12–v14, campaign, presentation), and write back into
`outputs/.old/...`, as they did before.

Some tools **take no flags and run as soon as you call them**: `inference_budget`, `plot_v14fix`,
`plot_production`, `plot_gradient70`, `showcase_plots`, `slides`, `ipc_animation`, `report_v8`,
`build_mpc_report`, `gradient70`. Each one regenerates its figures or report inside `outputs/.old/`.
`main.py <group> <tool> -h` only describes these tools; it does not run them. `build_mpc_report` needs `reportlab`.

## 7. Folders

```
main.py            front door
commands.md        this file
config/            network.yaml orbit.yaml train.yaml sweep.yaml network_quick.yaml, plant/ (model YAMLs)
python/arlamx_v2/  library (session.py = snapshots, train.py, train_v12.py, env.py, ...)
python/arlamx_v2/.old/   archived precoded sessions (train_v4, train_v56, v10_tune, tune_ppo, train_v12 rounds, ...)
outputs/snapshots  outputs/models  outputs/results  outputs/.old (everything before v2.7)
tests/  cpp/  data/  docs/
```

## 8. SC_v1_6mq (SolarCat_v3, six beam rods)

```bash
python -m arlamx_v2.sc6mq --minutes 25 --n-envs 44          # time-boxed PPO 4x18, settings config/sc_v1_6mq.yaml
cd outputs/SC_v1_6mq/analysis
python cluster_v3.py            # SolarCat_v3.stl -> 176 plates (1 % area)
python decay_att.py --att min   # max | min | aoa45, 500 -> 300 km, --f107 --ap
python sizing.py && python sizing2.py   # disturbance, B_z statistics, rod design + power
python tune_mrp3.py             # MRP gain sweep on the rod plant
python eval_6mq.py              # policy vs rods-always-on vs MPC vs min drag
python plots.py && python report.py     # figures + outputs/SC_v1_6mq/REPORT.md
```

Vehicle cards: `config/plant/vehicle_v3.yaml` (SolarCat_v3), `config/plant/vehicle.yaml` (hex). Gaussian sensors:
`config/plant/sensors_gaussian.yaml`. Magnetometer gap: env `mtq_duty` (plant `SimParams.mtq_duty`).

## 9. SC_v2_6mq (min drag held in eclipse)

```bash
python -m arlamx_v2.sc6mq --config config/sc_v2_6mq.yaml --steps 500000 --tag SC_v2_6mq --out outputs/SC_v2_6mq
cd outputs/SC_v2_6mq/analysis
python burnin.py --tag X --set '{"weights": {"downlink": 3.0}}'   # 50k burn-in + evaluation -> burnin_ledger.jsonl
python run_all.py && python plots_v2.py && python report_v2.py      # comparison, maps, tumble, figures, REPORT.md
python sweeps.py                                                    # advisor 100-800 s, inner loop 4 s .. 1 kHz
```

## 10. SC_v3_6mq campaign (outputs/SC_v3_6mq/analysis, PY=~/miniforge3/envs/ThesisMS/bin/python)
```
$PY hold_ref.py                         # hold reference on the burn-in set (GPS, nGPS)
./round.sh 0 TAG '{json override}' ...  # parallel GPS burn-ins (1 = nGPS), ledger burnin_ledger*.jsonl
$PY mc_mpc.py --procs 14                # MPC v3 Monte Carlo tuning -> mpc3_best.json
$PY full3.py --steps 3000000            # 4x16/18/20, GPS + nGPS -> models_v3.json
$PY final_eval.py && $PY compute3.py && $PY tables3.py && $PY plots3.py
$PY nav_study.py ; $PY nav_stress.py    # nGPS filter gains, robustness
```
