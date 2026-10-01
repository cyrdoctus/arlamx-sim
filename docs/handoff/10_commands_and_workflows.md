# 10 — Commands and reproducible workflows

Prefix everything with `PYTHONPATH=python` from the repo root, in the `arlamx`
mamba env. Full catalog with flags: `docs/COMMANDS.md`; in-process:
`python -m arlamx_v2 list`. Per-command flags: `python -m arlamx_v2 <cmd> -h`.

## Build and verify

```bash
mamba activate arlamx
./build.sh
PYTHONPATH=python python -c "from arlamx_v2 import cpp, __version__; print(__version__, cpp.__version__)"
PYTHONPATH=python python -m pytest tests -q                       # 208 passed
PYTHONPATH=python python tests/validation/verify_physics.py       # ALL PASS
```

## First-class dispatcher (`python -m arlamx_v2 <cmd>`, v2.6)

| cmd | what | key flags |
|---|---|---|
| `list` / `help` | catalog | |
| `train` | PPO/SAC/TD3 via `train.train_one`, variants v3–v9b | `--variant v8a --algo ppo --timesteps 300000 --n-envs 32 --seed 42 --arch 4x16 --controller {mrp,quaternion} --physics {fast,standard,high,PATH} --gsi {sentman,cll} --out --name` |
| `sweep` | YAML product with CSV ledger resume | `--config python/configs/sweep_example.yaml --force`; keys name, backend (train\|train_v12), variant, algo, controller, physics, gsi, seeds, timesteps, n_envs, arch, arms, ppo_kw, v12_kw |
| `quantize` | dynamic INT8 of actor Linear layers, post-train | `--in PATH --out PATH_INT8 --algo` |
| `eval mc_decay` / `eval mc_v12` | forward to those mains | |
| `decay` | prescribed min/max-drag decay | `--geom {hex,stl1pct} --kind {min,max,both} --alt0 500 --alt-stop 250 --inc 23 --ecc 0.001 --dt 5 --f107 --ap --mass 0.625 --physics --gsi --out` |

Aliases forward to module mains: `train_v12`, `train_v4`, `train_v56`,
`aoa_decay`, `lift_drag_study`, `srp_assess`, `validate_attitudes`,
`bench_v8`, `mc_decay`, `mc_v12`, `mc_four`, `plot_v8`, `plot_v12`,
`campaign`, `decay_run`, `quantize`.

## Recipes that reproduce recorded results

**SC_v3a speed baseline**
```bash
python -m arlamx_v2 train --variant v3 --timesteps 400000 --n-envs 32 --seed 42 --arch 4x16
```

**v8/v9/v10/v11 campaign stages (2026-08-20/21)**
```bash
python -m arlamx_v2.bench_v8 --stage burnin      # PRINTS deleg_P; fails if ≈ 0
python -m arlamx_v2.bench_v8 --stage ref         # MPC + heuristic reference on this plant
python -m arlamx_v2.bench_v8 --stage sweep|duo|v9duo|v9|v10|v11|trace
python -m arlamx_v2.v10_tune --round r0|r1|r2|a0|a1 [--variant v11]
python -m arlamx_v2.mc_decay --runs 512 --policies mpc,heuristic,v10:1
python -m arlamx_v2.damage_eval --stage eval|anim
python -m arlamx_v2.report_v8 ; python -m arlamx_v2.plot_v8 --set 3
```

**MPC tuning / freeze (2026-08-23)**
```bash
python -m arlamx_v2.tune_mpc --stage all --workers 20
python -m arlamx_v2.tune_mpc_v3 --stage all
python -m arlamx_v2.check_mpc_ref --stage all
```

**SC_v12 → v14 rounds (wrapper trainer)**
```bash
python -m arlamx_v2.train_v12 --round r0|long|s2|s3|s4|v13b|v135|v14|v14b|v14c|v14d|v14e|v14f|prec|infer
python -m arlamx_v2.mc_four --runs 24 --workers 8 --plot --mc
python -m arlamx_v2.plot_v12 --all --v14 ; python -m arlamx_v2.plot_v14fix
```
`train_run(...)` now forwards `physics`, `controller`, `gsi`; outputs land in
`outputs/v12/runs/<name>`.

**Fixed-attitude decay and lift studies (September 2026)**
```bash
python -m arlamx_v2 decay --geom stl1pct --alt-stop 300 --inc 23 --ecc 0.001 --mass 1.75
python -m arlamx_v2.aoa_decay                     # both geometries, 45°, +h and +r, refs 0/90
python -m arlamx_v2.aoa_decay --plot-only
python -m arlamx_v2.lift_drag_study --geom stl1pct --mass 1.75 --inc 45 --out outputs/analysis/lift_drag/stl1pct_i45_m1p75
python -m arlamx_v2.srp_assess --geoms hex stl1pct
python -m arlamx_v2.srp_f107_sweep --plot-only
python -m arlamx_v2.validate_attitudes --quality 1pct
```

**Sweep over physics / controller (v2.6)**
```yaml
# python/configs/sweep_example.yaml
name: v8a_controller_grid
backend: train
variant: v8a
algo: [ppo, sac]
controller: [mrp, quaternion]
physics: standard
seeds: [42]
timesteps: 100000
n_envs: 8
arch: 4x16
```
```bash
python -m arlamx_v2 sweep --config python/configs/sweep_example.yaml
```

**Ideal-couple actuator for a run (declared, snapshotted)**
```python
from arlamx_v2.physics import load
cfg = load("standard", overrides={"attitude": {"ideal_torque": True}})
# write cfg to a YAML and pass --physics path/to/that.yaml
```

## Process rules the project learned the hard way

- Run the burn-in gate that *prints* term activity before any long run
  (caught `decay_stab_scale` four orders off and `deleg_power ≡ 0`).
- ≥ 3 seeds per cell; report the median; the best seed is an artifact.
- Pick checkpoints by eval curve, not final step (non-monotone budgets).
- One knob per arm (v14c stacked two and produced an empty dual gate).
- Never compare quiet 10-orbit totals with Monte Carlo per-day rates.
- Report the *full* test-suite count, never a subset, and never leave a
  failing test as `xfail` without the diagnosis next to it.
