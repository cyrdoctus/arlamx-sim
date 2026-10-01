# temp.md — v2.6 → v2.7 refactor log

Working log. Each segment: **BEFORE** (current functionality, written before the edit),
**CHANGE**, **CHECK 1** (functionality vs BEFORE), **CHECK 2** (independent re-check).

## 0. Baseline (taken before any edit)

- Interpreter: `~/miniforge3/envs/ThesisMS/bin/python` (3.12). The `arlamx` env named in
  build.sh / README no longer exists on this machine.
- Code backup: `/mnt/storage/ARLAMX_Dev/ARLAMX_v2.6_code_backup_pre_v2.7.tar.gz` (no outputs/, no build/).
- `pytest tests`: **211 passed**.
- Behaviour fingerprints (`scratchpad/fingerprint.py`, run twice → identical, i.e. deterministic):
  - `cfg:*`: gains_mrp, power_mtq, estimator_kf, duo, mpc_v2, sensors_solarcat
  - `reward:*`: load_reward for v8a v8b v9a-d v10 v11 (with a flat `dE_weight` override)
  - `physics:*`: fast / standard / high resolved dicts
  - `env:*`: 12-step trajectories (obs + reward hashes, max_steps, mass, f107, ap) for
    v3 v4a v6 v7 v8a v9d v10 v11, v10 with reset options, v8a with physics=high + quaternion
  - `train:*`: trained policy weights after train_one for PPO(256) / SAC(160) / TD3(160) + snapshot.json bytes
  - `v12:*`: train_v12.train_run 128 steps: policy weights + final eval metrics
  - `decay:min`: decay_run.run_one min-drag CSV bytes (0.05 d)

---

## 1. config.py  (python/configs → config/plant)

**BEFORE.** `CONFIG_DIR = python/configs`. `config_path(name)` resolves a bare name or a
.yaml path; `load(name)` returns a deep-copied cached dict; `merge` recursive override;
`load_reward(variant, override)` picks reward_v8a/v8b/v9(+obs per v9a-d)/v10/v11 and routes
flat overrides into the `v7` section; `snapshot(cfgs, path)` writes sorted JSON.
Users of CONFIG_DIR: env.py (sensors yaml), tune_mpc, tune_mpc_v3, check_mpc_ref; test_sensors
hard-codes python/configs/sensors_solarcat.yaml.

**CHANGE.** All 17 plant YAMLs moved to `config/plant/` unchanged (byte copies; provenance
comments kept). `sweep_example.yaml` moved to `config/sweep_example_v2.6.yaml` (replaced later).
`CONFIG_DIR = config/plant`, new `SETTINGS_DIR = config`. New `pin(cfgs)`: when a name is
pinned, `load(name)` returns the pinned dict and `load_reward` starts from the pinned `reward`
(overrides still merged, idempotent). Nothing is pinned unless `pin()` is called.

## 2. physics.py / sensors.py

**BEFORE.** `physics.load(preset|path, overrides)` → validated dict. `SensorConfig.from_yaml(path)`
→ strict dataclass build.
**CHANGE.** `physics.load` also accepts a dict (deep-copied, merged, validated) — used for
snapshot replay. `SensorConfig.from_dict(d)` split out of `from_yaml` (from_yaml now calls it).

## 3. env.py

**BEFORE.** `ArlamxV2Env(geom_path, ggm_path, seed, variant, gate_floor, kp, kd, obs_extra,
reward_w, controller, physics, gsi)`. Sensors from `CONFIG_DIR/sensors_solarcat.yaml`.
Episode length: 160 orbits (v11) / 120 (v10) / 40 (others) at 400 km period.
`reset()` per-variant envelopes, each draw **eager** (`opt.get(k, rng.uniform(..))` draws even
when the option is given):
- v10/v11: alt U(300,500), inc U(20,40), ecc U(0,0.01), f107 U(5,250), ap U(2,200), raan/argp/nu U(0,2π) unless option
- v4..v9: alt U(300,500), inc U(20,30), ecc U(0.001,0.01), f107 U(65,250), ap U(2,40), angles as above
- v3: alt U(300,500), inc U(20,30), ecc 0, f107 U(80,200), ap U(2,15), raan=argp=0, nu 0 unless option
- mass: v8+ vehicle card (power_mtq.yaml 0.625) unless option; else U(0.5,0.75) eager
- omega U(-5,5)°/s ×3 unless option; soc 0.5 unless option

**CHANGE.** New kwargs `orbit=None, frozen=None`. `frozen` → `config.pin()` before any YAML load.
`orbit` (config/orbit.yaml): list = draw range, scalar = fixed, missing/None = variant default;
unknown keys raise; `omega_dps` must be `[lo, hi]`. Reset draws go through `_pick` (always one
draw, option wins — same eager semantics), `_pick_fixed` (no draw unless a range is set),
`_pick_angle` (option > orbit > U(0,2π) or 0 for v3). v3 still ignores raan/argp **options**
(as before). `episode_orbits` replaces the 160/120/40 default. Sensors now via
`SensorConfig.from_dict(load_cfg("sensors_solarcat"))` (same YAML, pinnable).

**CHECK 1 (segments 1-3).** Fingerprints cfg:* reward:* physics:* env:* (27 keys) vs baseline:
**0 mismatches**.
**CHECK 2.** Old tree (pristine copy) vs new tree with reset options (raan/argp/nu/mass/soc/ecc/
omega) for v3 v4a v8a v10: **identical**. New paths: orbit = v10 ranges → identical to no orbit;
fixed altitude changes trajectory but keeps every other draw; episode_orbits=5 → 92 steps;
frozen configs + physics dict → identical; a changed frozen kd is actually used; bad keys raise.

## 4. paths.py + outputs/ layout

**BEFORE.** `OUTPUTS = outputs/`; grouped layout `OUT_TRAINING/TUNING/ANALYSIS/LOGS`
(outputs/training, tuning, analysis, logs); `run_dir(name)` searches training, tuning,
campaign/runs, outputs/ and defaults to outputs/training/<name>. Campaign modules hard-code
`OUTPUTS / "v8" | "v12" | "v13" | "v14" | "v14fix" | "mpc" | "presentation" | "campaign"`,
several use literal `ROOT / "outputs" / ...`. Data paths (GGM/WMM/geom/STL) unaffected.

**CHANGE.** Everything in outputs/ moved to `outputs/.old/` (same relative tree; a same-filesystem
rename, no copy). New: `outputs/snapshots/` (one YAML per training session),
`outputs/models/` (one folder per session: weights, vecnormalize, tensorboard logs, metrics),
`outputs/results/` (simulations, evaluations, sweep ledger). paths.py: `OUTPUTS_OLD`,
`OUT_SNAPSHOTS`, `OUT_MODELS`, `OUT_RESULTS`; the old grouped names now point inside `.old`;
`run_dir` searches outputs/models first, then the archive, default outputs/models/<name>.
Campaign/analysis modules that read or extend historical artifacts import
`OUTPUTS_OLD as OUTPUTS` (their relative paths unchanged → they keep working on the archive).
Plain simulations write to `outputs/results/<same sub-path without "analysis/">`.

**CHECK (segment 4 edits, textual).** 13 modules `OUTPUTS_OLD as OUTPUTS` (1 line each), 9 modules
literal `ROOT/"outputs"` → `ROOT/"outputs"/".old"` (16 sites), 7 simulations + plot_simplify default
`--out` → outputs/results/... . Remaining `"outputs"` literal: only train.py (rewritten in §6).
All 16 archived top-level entries present in outputs/.old (545 MB before and after).

## 5. Archived precoded sessions → python/arlamx_v2/.old/

Moved unchanged (not importable from the package any more; kept for reference):
train_v4, train_v56, run_v5_then_v6, run_v5v6_pipeline (v4-v6 precoded trainings/pipelines),
tune_ppo, reeval_tune, plot_tune (v4a PPO tuning board), v10_tune (v10/v11 reward rounds r0..a1),
bench_production, anim_production, pack_presentation, showcase_runs (v7 bake-off / presentation
runs), tune_mpc_v3 (MPC_v3 sweep), and a full copy of the original train_v12.py (all
`round_*` sessions r0 … v14f, prec, infer). No live module or test imports any of them
(checked with grep; the two remaining mentions are prose).

## 6. train.py

**BEFORE.** `make_env`, `configure_torch_threads`, `build_model(algo, env, arch, seed, log_dir,
ppo_kw)` — PPO defaults lr 3e-4, n_steps 128, batch 64, epochs 10, gamma .99, ent .003, clip .2,
gae .95 (+ppo_kw); SAC lr 3e-4 buffer 1e5 batch 256 gamma .99 tau .005 verbose=1; TD3 same
numbers verbose=0; net_arch pi/vf = arch for PPO, arch for SAC/TD3.
`train_one(variant, algo, timesteps, n_envs, seed, name, out_root, ppo_kw, env_kw, arch)` →
`out_root/name/` {models/<algo>_<name>.zip, models/vecnormalize.pkl, logs/, metrics.json,
snapshot.json (v8*/v9* only)}; SubprocVecEnv if n_envs>1; VecNormalize(norm_obs=False,
norm_reward=True, clip_reward=10, gamma=.99). `main()` argparse CLI (default out outputs/training).

**CHANGE.** `PPO_DEFAULTS / SAC_DEFAULTS / TD3_DEFAULTS / VECNORM_DEFAULTS` hold the exact old numbers
(incl. SAC verbose=1, TD3 verbose=0). `build_model(..., algo_kw=None)`: defaults ← algo_kw ←
ppo_kw. `train_one(..., run_dir=None, algo_kw=None, vecnorm=None)`: run_dir replaces
out_root/name. metrics.json gains `algo_kw`/`vecnorm` only when they are passed; `frozen` is
not copied into metrics (it lives in the snapshot). `main()` forwards to `main.py train`.
**CHECK 1.** Fingerprints train:ppo/sac/td3 weights + legacy snapshot.json bytes: **0 mismatches**.
**CHECK 2.** Explicit PPO_DEFAULTS/VECNORM_DEFAULTS as algo_kw/vecnorm → identical weights to the
implicit call; run_dir honoured; lr=1e-3 → different weights (value is live); SAC explicit ok.

## 7. train_v12.py

**BEFORE.** Library: constants (OUT=outputs/v12, SEEDS, ARCH 4x18, VARIANT v10, CKPT_EVERY 50k,
REF_A/REF_D, EVAL_OPT/STORM_OPT), V12Wrapper (r_slew, comparator, climate mix, regime …),
`min_score`, `_eval_wrapped` (10-orbit eval, fixed EVAL_OPT, optional storm), `make_env`,
`infer_costs`, `_write_best`, `_campaign_row` (OUT/BEST_SO_FAR.md), CURVE_FIELDS, `_row_from_agg`,
`_wrap_kw`, `_log_line`, `_dual_ok` + keep_ok* gates, `_apply_ppo_kw`,
`train_run(name, w_slew, steps, n_envs, seed, …27 knobs…, physics, controller, gsi)` →
OUT/runs/<name>/ {snapshot.json, models/, best/, ckpts/step_N/, eval_curve.csv, BEST.md,
metrics.json}; skips if metrics.json + eval_curve.csv exist.
Precoded sessions: round_r0/long/s2/s3/s4/v13b/v135/v14/v14b-f, _s3_resume, _pick_keep_ckpt,
_copy_dual_ckpt(_many), eval_prec, PPO_STABLE(_C), V13/V14 ckpt constants, `main()` (--round …).
External users: tests (OUT, _eval_wrapped, _wrap_kw, make_env, train_run), sweep (OUT, train_run),
plot_v12 (EVAL_OPT, N_ORBITS_EVAL, REF_A, REF_D, VARIANT, V12Wrapper, _eval_wrapped, min_score),
mc_v12/mc_four (V12Wrapper, VARIANT).

**CHANGE.** Removed (archived in .old/train_v12.py): round_r0 … round_v14f, _s3_resume, V13_CKPT,
round_s4, V13_FREEZE, _pick_keep_ckpt, _copy_dual_ckpt(_many), V14_KEEP, PPO_STABLE(_C), V14B/D/E
constants, eval_prec, main. Kept: everything tests/plot_v12/mc_v12/mc_four/sweep import, plus
keep_ok* gates (used by `_dual_ok`) and infer_costs. `OUT` → outputs/.old/v12.
`train_run(..., run_dir=None, arch=None, algo_kw=None, vecnorm=None, orbit=None, frozen=None)`;
`make_env(..., orbit, frozen)`; `_eval_wrapped(..., frozen)` and `_wrap_kw(..., frozen)` (eval gets
frozen configs but NOT orbit: the eval protocol stays EVAL_OPT). `_campaign_row(..., board)`:
sessions write their board to outputs/results/v12_best_so_far.md, legacy calls to OUT.
1928 → 1178 lines.
**CHECK 1.** Fingerprints v12:params + v12:eval (128-step run + 10-orbit eval): **0 mismatches**.
**CHECK 2.** pyflakes on all edited modules: no undefined names. plot_v12 / mc_v12 / mc_four import.
train_run(run_dir, arch=[8,8], orbit) → files in run_dir (BEST.md best eval_curve.csv logs
metrics.json models snapshot.json), arch [8,8] in snapshot + metrics, board in results, archive
untouched.

## 8. New config files (config/)

network.yaml (algo, layers, width, timesteps, n_envs, seed, torch_threads, ppo/sac/td3 blocks,
vecnormalize — every number equals the v2.6 hard-coded default), network_quick.yaml (same, 20k
steps × 4 envs for smoke runs), orbit.yaml (`training:` = env reset envelope, the v10 ranges;
`simulation:` = decay_run defaults 500→250 km, i 23, e 0.001, F10.7 150, Ap 4, 0.625 kg, dt 5,
800 d, hex), train.yaml (name, backend train|v12, variant v10, physics, gsi, controller, reward_w,
v12 block = train_run defaults), sweep.yaml (base + grid/arms of dotted keys).

## 9. session.py (new)

settings → id `MM-DD-HH-MM_ALGO_LxW[_name][_n]` → outputs/snapshots/<id>.yaml written BEFORE
training (status running) and rewritten after (done / failed + error / interrupted, wall_s,
result = metrics). Snapshot sections: session (id, created, command, replay command, versions,
host), network, orbit, train, plant (resolved physics, reward, gains_mrp, power_mtq, estimator_kf,
sensors_solarcat — v8+; v3-v7 rewards are code, noted as such). Plant dicts are pinned for the
run (main process + every env via `frozen`), so the YAMLs can change mid-run without effect.
`--from-snapshot` reloads network/orbit/train + plant → exact replay.

## 10. sweep.py

**BEFORE.** YAML `{name, backend train|train_v12, variant, algo, controller, physics, gsi, seeds,
timesteps, n_envs, arch, out, arms, ppo_kw, v12_kw}` → product jobs → `train_one` /
`train_v12.train_run`; skip if metrics.json in dest or name in out/SWEEP_LEDGER.csv; `--force`.
Tests: expand_jobs (omitted controller stays None), run_job forwards physics/controller/gsi to
train_run, job_dest = outputs/v12/runs/<name> + skip on metrics.json.

**CHANGE.** Old sweep.py archived in .old/sweep.py (its YAML in config/.old/sweep_example_v2.6.yaml).
New sweep: `name`, `base`, `grid` (product) or `arms`, all dotted keys over config/*.yaml. Each arm
is a full session (own snapshot + model folder). Ledger outputs/results/sweeps/<name>.csv
(label, id, status, wall_s, snapshot, error); `done` labels skipped unless --force; a failing arm
is recorded and the sweep continues (Ctrl-C still stops it).
**CHECK 1.** Rewritten wiring tests keep the three v2.6 properties: omitted controller stays None;
v12 backend receives physics (name high, gsi cll), controller quaternion, w_slew 0.1, run_dir in
models/; completed arms are skipped (replaces the metrics.json-in-dest test, since session folders
are time-stamped). **CHECK 2.** Real `main.py sweep` (ppo+td3, 256 steps): both done, second call
skips both, ledger written; PPO arm weights == baseline train:ppo.

## 11. main.py (new) / cli.py

**BEFORE (cli.py).** `python -m arlamx_v2 {list,help,train,sweep,quantize,decay,eval mc_decay|mc_v12}`
+ aliases (train_v12, train_v4, train_v56, aoa_decay, lift_drag_study, srp_assess,
validate_attitudes, bench_v8, mc_decay, mc_v12, mc_four, plot_v8, plot_v12, campaign, quantize,
decay_run) forwarded with runpy.
**CHANGE.** main.py: train | sweep | decay | snapshots | quantize | sim/eval/plot/legacy <tool> |
run <module> | test | verify. Every old alias except the archived train_v4/train_v56/train_v12
rounds is reachable (grouped). `train` flags are a superset of the old train CLI (+ --set, --network,
--orbit, --train, --from-snapshot, --backend, --out). `decay` builds decay_run flags from
orbit.yaml `simulation:` and writes outputs/results/decay/<stamp>_<kind>_<physics>/ + settings.yaml.
cli.py → 10-line shim (old one in .old/cli.py); `python -m arlamx_v2 …` and
`python -m arlamx_v2.train …` both land in main.py. `__version__` = "2.7".
**CHECK 1 (end-to-end vs baseline, run into a scratch --out).**
- `main.py train --variant v8a --timesteps 256 --n-envs 1 --set orbit.training={}` → weights == baseline train:ppo
- `main.py train --from-snapshot <that snapshot>` → identical weights (exact replay)
- `python -m arlamx_v2.train --backend v12 --arch 4x18 --timesteps 128 …` → weights == v12:params, eval == v12:eval
- `main.py train --algo td3 … 160 steps` → == train:td3
- `main.py decay --kind min --set orbit.simulation.max_days=0.05` → CSV bytes == decay:min
**CHECK 2.** Every sim/eval/plot/legacy tool imports after the moves (31/31; build_mpc_report needs
`reportlab`, not installed in ThesisMS — pre-existing). Full fingerprint suite (36 keys) on the final
tree: 0 mismatches. pytest 213 passed (211 original − 1 replaced + 1 + 2 new session tests).

Bugs found by the checks and fixed: (a) v12 leaderboard ignored `--out` → now `<root>/results/`;
(b) it crashed when `<root>/results/` did not exist → mkdir (the crash was correctly recorded as
`status: failed` in the snapshot). tests/geometry wrote to outputs/geometry → now
outputs/results/geometry.

**Incident (reported to the user).** Import-checking tools with `-h` ran 8 tools that ignore `-h`
(plot_v14fix, plot_production, plot_gradient70, showcase_plots, slides, ipc_animation, report_v8,
inference_budget); they regenerated 71 derived files in outputs/.old (figures, SC_v8_v9_report.md,
one mp4, 3 JSON summaries) from the archive's own data. Restored all 71 byte-exact from
Thesis/ARLAMX V2.0/outputs (verified mirror: all 188 untouched neighbours byte-identical), mtimes
preserved; afterwards no archive file is newer than this log.

## 12. Tests changed

tests/physics/test_physics_wiring.py (3 sweep tests → new API, same properties),
tests/sensors/test_sensors.py (config/plant path), tests/geometry/test_solarcat_attitudes.py
(output → outputs/results/geometry), new tests/integration/test_session.py (id format, snapshot
sections, pinning, orbit/network passed through, replay uses snapshot values, failure recorded,
pins cleared).

## 13. Docs, version, rename

- `commands.md` (new, repo root): interpreter, settings table, train / snapshot / replay / sweep /
  simulation / eval examples, folder map. Examples were executed (tiny runs, scratch --out):
  overrides + reward_w + orbit range → snapshot records them and plant reward has dE_weight 3.0;
  quantize ok; replay with 2 subprocess envs → identical weights.
  Found + fixed: PyYAML reads `1e-4` as a string → `session.coerce()` turns sci-notation strings
  into floats for --set values and loaded settings files.
- main.py: tools that take no flags (10) are marked in `list`; `<tool> -h` describes them
  instead of running them.
- README.md (v2.7 header, Test/Train/Layout via main.py), CHANGELOG_v2.7.md (new),
  docs/COMMANDS.md → docs/.old/COMMANDS_v2.6.md, docstrings in reward_v8.py / duo.py.
  Dated reports and handoff docs that mention python/configs are left as historical records.
- `__init__.__version__` = "2.7". The compiled C++ module still reports 2.6 in its own
  `__version__` until the next `./build.sh` (C++ source not edited).
- build.sh drops a CMake cache from another source dir (the existing cache pointed at
  `.../ARLAMX V2.1/cpp` on an external drive).
- Folder renamed ARLAMX_v2.6 → ARLAMX_v2.7 (no code path contains the folder name; all paths
  derive from `__file__`).

---

# v2.7 critic pass (criticv1.md, 2026-09-25)

User decisions: F4 → **both membrane faces carry cells** (keep |ŝ_B,z|, now a stated fact);
style pass scope → **live code** (plant C++, env, physics/power/sensors/estimators/reward,
trainers, session, main); legacy campaign/plot tools only get import renames.

Build: `ARLAMX_PYTHON=~/miniforge3/envs/ThesisMS/bin/python ./build.sh` works (system cmake/g++,
pybind11 3.0.4). The unmodified tree rebuilt from source reproduces all 36 fingerprints, so later
differences come from the edits. Prebuilt v2.6 .so kept in scratchpad.

Order: Phase A = physics F1-F11 + the two sanctioned speed items (behaviour changes, each with the
critic's classroom check as a test). New fingerprint baseline after Phase A. Phase B = style pass,
which must reproduce the Phase A fingerprints bit for bit.

## A-BEFORE (plant as of v2.7 restructure)
- api.cpp step: F_B = aero.F + F_srp; tau_B = aero.tau + tau_ctrl (SRP moment dropped, no gravity
  gradient, no Earth radiation). Binding panel_srp_optical returns force only.
- SRP pressure P_SRP_1AU*srp_scale; analytic Sun = unit vector * 1 AU (no distance variation).
- Presets: srp.optical false, Cr 1.8 (cannonball). apply_to_params never sets srp_ca/cs/cd.
- v_rel = v − ω⊕×r (no wind hook).
- drag_N = substep mean of |F_aero|. Onboard propagator: two-body + J2 + exp drag, spherical
  altitude |r|−RE, no SRP. env anchors it at the end-of-step spherical altitude with rho from the
  start-of-step geodetic MSIS query.
- altitude_km (step + get_state) and the 80 km stop are spherical; 450 km flash cutoff reads it.
- env._update_power: illum 0 in eclipse → gen_n 0 → _future_block forecasts 0 generation for every
  future sunlit slot when the step ends in shadow.
- lunisolar: Sun/Moon ephemeris rebuilt on every RK4 stage.
- Sentman/CLL evaluate erf/exp for every windward γ.
- query_msis fills the 7 Ap slots with one scalar.

## A-CHANGE (criticv1 F1-F11 + speed)
| item | change | files |
|---|---|---|
| F1 | SRP moment applied (cannonball + optical); binding `panel_srp_optical` returns (F, τ), new `panel_srp_force_torque`; new step outputs `tau_srp_mean`, `tau_erp_mean`, `tau_gg_mean`, `tau_env_mean` (= aero+srp+erp+gg); observer truth / test identity use `tau_env_mean` | api.cpp, bindings.cpp, types.hpp, env.py (info), train_v12 (kf truth), tests |
| F2 | presets `srp.optical: true, optics: al_mylar`; physics.py now passes ca/cs/cd (it never did); P = P_SR (AU/d)² · scale; analytic Sun distance series `sun_dist_au` (Astronomical Almanac / Vallado Alg. 29) | physics*.yaml, physics.py, frames.cpp, third_body.cpp, api.cpp |
| F3 | `_update_power` keeps geometric illumination `_illum_geo`; future sunlit slots use it (eclipse now ≠ zero forecast) | env.py |
| F4 | user: cells on both faces → |ŝ_B,z| kept, documented | env.py |
| F5 | gravity-gradient torque `gg_torque` per substep, μ from GGM header; switch `gravity.gradient_torque` | integrate.cpp/.hpp, api.cpp |
| F6 | Earth IR (e0/4 (R/r)², on in eclipse, thermal partition 0.85/0/0.15 [ASSUME]) + albedo (a0 (R/r)² max(0, r̂·ŝ)), Knocke 1988 a0 0.34 e0 0.68; switch `srp.earth` | panel_srp.cpp/.hpp, constants.hpp, api.cpp |
| F7 | wind hook `v_rel = v − ω×r − v_wind`, `Simulator.set_wind`, default 0 | api.cpp, bindings |
| F8 | `drag_N` = along-track drag; plant exports `a_srp_sun`/`n_sunlit`; env holds it; onboard propagator (C++ + numpy) adds held along-track SRP gated by cylindrical shadow | api.cpp, propagator_f32.cpp, propagator.py, env.py |
| F9 | `altitude_km` (step, get_state), 80 km stop → Bowring geodetic; onboard exponential uses ellipsoid height r − a(1 − f sin²φ) anchored at the geodetic height of the MSIS query; future-block altitudes geodetic | api.cpp, bindings, propagator*, env.py |
| F10 | CLL paragraph + header comment corrected with measured table; 11 stale "spec only" status lines → implemented; reward_v8 status | docs/modules/*, cll.hpp |
| F11 | `query_msis` accepts length-7 Ap history (storm-time mode); `atmosphere.interp` (standard/high on): 2nd MSIS sample at predicted step end, plant interpolates ρ,T,m̄,χ over substeps; min-drag baseline refresh keyed on speed ratio s | atmosphere.py, env.py (`_atmo_at`), api.cpp, physics*.yaml |
| speed | γ>6 → erf=1, E=0 in Sentman and CLL; lunisolar Sun/Moon computed once per advisor step | sentman.cpp, cll.cpp, api.cpp |

Numbers note: a verify_physics torque-free block now sets `gravity_gradient=False` explicitly (it is
torque-free by construction; no tolerance changed, no "leave alone" expression touched).

## A-CHECK 1
- pytest: 237 passed (213 + 24 new in tests/physics/test_disturbances.py, one per critic check:
  F1 r×F / 0.01 P / centred 0 / plant = kernel at 1/d²; F2 hand-summed mylar hex 5.692 µN, 90° → 0,
  cannonball Cr=1 = P, perihelion 0.9833 / aphelion 1.0167 AU, presets; F3 shadow-now → positive
  future generation; F5 axis → 0, 45° → 3μ/r³·½|Izz−Ixx| (1e-12); F6 IR outward in eclipse, deep-space
  plate 0, albedo day side only, switch leaves solar term; F7 zero wind bit-identical, opposing wind ↑
  drag; F8 ΔE = a_srp v T (0.5 %), a_srp=0 bit-identical (numpy + C++); F9 +Z axis 421.4 km =
  Bowring; F11 Ap history + interpolation; γ-limit residual ≤ 1e-15 for γ = 0.5…8).
- verify_physics.py: ALL PASS.
## A-CHECK 2
- Old observer identity I Δω/T = τ_env + τ_ctrl − ⟨ω×Iω⟩ still exact (1e-12) with the new moments.
- Fingerprints: configs + rewards unchanged; physics/env/train/decay changed (intended). Phase A
  fingerprints recorded twice → deterministic (baseline for Phase B).
- Timing, v10 env step: 1.05 → 1.10 ms (new moments), 1.23 ms with interpolation; high 2.60 → 2.94.
- Smoke `main.py train` (network_quick, 2048 steps, 4 envs): done.
- CLL numbers in the doc were re-measured, not copied.

# Phase C — six beam rods, 0.750 kg vehicle, STEP import (user request mid-pass)

User decisions: rods **along each beam** (the hex beams run along the hexagon SIDES, so rod axes
are at 30/90/150° and the dipole is in-plane only); total **0.750 kg** (80 g PCB+battery centre,
170 g rods on the beams, 500 g structure); **new variant with 6 gates** (v10r6), old variants intact.

## C-BEFORE
- Actuator: 3-axis coils, per-axis dipole clip (power_mtq.yaml 1.473/1.473/0.442 A m²), v8+ split →
  per-axis gates; vehicle card 0.625 kg, inertia diag(0.0125, 0.0125, 0.025) hard-coded (INERTIA_DIAG).
- geometry.load_mesh: STL/OBJ via trimesh, `units="mm"` scale; no STEP; no .geom writer / CLI.

## C-CHANGE
- config/plant/vehicle.yaml + python/arlamx_v2/vehicle.py: mass budget, beam share by area
  (10.4 % → beams 52 g, membrane 448 g) [ASSUME], lamina + thin-rod + point-mass inertia →
  diag(0.0436, 0.0436, 0.0872) kg m², CoM 0; rod axes/positions; rod sizing scaled by coil mass
  from the v8 design (0.596 A m², 0.0439 W per rod) [ASSUME]; torque caps; axis_authority().
- C++: RodArray, `rods_allocate` (d = Uᵀ(UUᵀ)⁺m over enabled rods, Jacobi pinv, uniform scale),
  `apply_rods`; Simulator.set_rods / set_rod_gates / rod_gates; B-dot and PD both allocate onto
  rods when present; StepOut.rod_duty2_mean; binding `rods_allocate`.
- env: variant `v10r6` (= v10 + vehicle.yaml): action 10 (quat + 6 gates, a>0 on, brownout all on),
  obs +6 rod states, mass/inertia/cp offset/KF inertia/torque caps/peak power from the vehicle,
  effort = Σ P_k ⟨duty²⟩ / Σ P_k, delegation split = 1 − rod authority per torque axis.
  INERTIA_DIAG uses → self._I (same value for v3–v11). train_v12 uses env._I.
- config.load_reward("v10r6") = reward_v10; session records `vehicle` in the snapshot plant.
- geometry: `.step/.stp` via trimesh + cascadio (installed into ThesisMS: `pip install cascadio`,
  added to environment.yml / requirements.txt), file units → metres, merged vertices;
  `write_geom`, `python main.py sim geometry --in X.step`. Test fixture
  tests/geometry/data/box_100x200x20mm.step written by tests/geometry/make_box_step.py (no STEP
  file existed on the machine; gmsh wheel could not load its native lib).
- Docs: 15_control_magnetorquer.md (rods section), 10_geometry_simplify.md (STEP), train.yaml.

## C-CHECK 1
- tests/control/test_rods.py (21): allocation == numpy pinv (1e-12) for 5 gate patterns × 3 requests,
  |d| ≤ dmax, off rods 0, m_z = 0, small in-plane request met exactly, saturation keeps direction,
  mass 0.750, CoM 0, I diagonal, I_z = I_x + I_y (planar), I_x = I_y (6-fold), authority 1/0,
  plant gating (all off → τ_ctrl = 0, duty 0), env v10r6 sizes / gates / obs / split / power.
- tests/geometry: STEP box == STL box (6 sides, per-normal areas 1e-6 — float32 GLB), .geom round
  trip, STEP with units="mm" refused.
- Full suite 259 passed; verify_physics ALL PASS.
## C-CHECK 2
- Fingerprints vs Phase A: every existing key identical (v3–v11 env, trainers, decay unchanged);
  new keys env:v10r6, veh:I, rods:alloc, geom:step; recorded twice → deterministic.
- `main.py train --variant v10r6` (2048 steps, 4 envs): done; snapshot plant.vehicle present.
- v10r6 step 1.19 ms (v10 1.22 ms).

# Phase B — style pass (live code)

## B-BEFORE
Live Python: 8396 lines, ~330 comment-only lines + trailing comments, ~440 docstring lines; 8
names > 20 chars. C++: 33 files, 331 comment lines, no file headers; long names only the two
Python-facing bindings (kept). Reference behaviour = Phase C fingerprints (all keys) + 259 tests.

## B-CHANGE (method)
1. scratchpad/strip.py: module docstring → one-line description of the file; function/class
   docstrings and comments removed, EXCEPT lines carrying a source/equation citation (regex over
   author names, "eq.", years, Alg., [SPEC], …), which are kept as `#` comments in place.
   Python check built in: AST (minus docstrings) of new == old. C++ check built in: comment-free
   preprocessed token stream of new == old (`g++ -fpreprocessed -E -P`).
2. Manual citation pass on every equation-bearing file (citations only where the source is known).
3. Rename map (9 names), token-level, callers in legacy tools and tests updated.

## B-CHECK 1
- Python: 32 live files stripped, AST (minus docstrings) identical for every file (built-in check);
  every later citation insertion re-checked the same way (cite.py). C++: 33 files, comment-free
  preprocessed token stream identical for every file (cll.cpp differs only by the space left where
  `/* atomic O */` was — tokens equal); every citation insertion re-checked (cite_cpp.py).
- Rebuilt plant; pytest 259 passed; verify_physics ALL PASS.
- Fingerprints vs Phase C (40 keys: configs, rewards, physics, 11 env trajectories incl. v10r6,
  PPO/SAC/TD3 weights, v12 weights + eval, decay CSV, vehicle inertia, rod allocation, STEP
  simplification): **0 mismatches** → the style pass changed no behaviour.
## B-CHECK 2
- Fragment review: the regex pass kept single lines of multi-line comments that happened to name
  an author; all such fragments were dropped and replaced by complete citation lines by hand
  (cll.cpp's "published kink" line — contradicted by F10 — removed). Citation section numbers I
  could not verify were widened to the chapter (Vallado ch. 3/4/5), not guessed.
- Renames (9): estimate_environment_torque→env_torque, configure_torch_threads→torch_threads,
  _eclipse_fraction_from→_ecl_frac, decay_rate_km_per_day→decay_rate,
  _shared_features_extractor→_features, _infer_algo_from_path→_algo_of_path,
  adapt_delegation_hold→hold_term, compose_duo_secondary→duo_reward, bc_inverse_from_drag→bc_from_drag;
  31 sites incl. bench_v8 (legacy) and 2 tests; no string keys touched; no name collisions.
  Other names were already short; sensor-config field names mirror YAML keys with units and stay.
- pyflakes: no undefined names. Import-only check of every python/arlamx_v2 module: all import
  (build_mpc_report needs reportlab — pre-existing). No tool was *run* in this pass.

# SC_v1_6mq (user request, 2026-09-25 evening)
- Geometry: SolarCat_v3.stl → 176 plates by normal clustering from raw triangles (1 % area / 0.1 % force);
  simplify_sides bug found (sliver centroids up to 660 km) — not used, logged in CHANGELOG.
- Sizing: 53 µN·m design torque (max drag, storm, 300 km), |B|min 19.7 µT; in-plane rods cannot hold max drag
  (needs B·v); min drag / 45° covered by 6 × 0.8 A m². Rod: 2.25 mm × 0.48 m permalloy, 28.3 g, ~2 mW I²R.
- MRP: no PD gain tracks with magnetic-only in-plane rods (torque request ∥ B, aero moment > authority);
  chose wn 0.02, ζ 1.2.
- New: mtq_duty (C++), sensors_gaussian.yaml, vehicle beam_angle0_deg, env kwargs, sc6mq wrapper/trainer,
  3 tests (262 passed). Old variants unaffected (mtq_duty default 1.0; defaults unchanged).

# SC_v3_6mq campaign (user request, 2026-09-26) — BEFORE notes
Goal: full optimisation of RL (v3 vehicle) and a re-designed MPC; nGPS variant; FP32 vs INT8; 4x16/4x18/4x20.
Plan (edits, each checked twice):
1. C++ `set_nav(r, v)` / `clear_nav()`: flow-frame target from an onboard nav state (two-body RK4 per
   substep) instead of truth. Default off → every existing variant unchanged (fingerprint check).
2. env.py: `_gps_W` (0.205 W receiver load, sunlit), `_tx_fn` hook (tx gate from onboard knowledge),
   `_term_km` (default 250). Defaults keep old behaviour.
3. New python/arlamx_v2/nav.py: fp32 J2+drag RK4 propagator from the known launch state, eclipse-timing
   fixes from the solar panels (alpha-beta phase + drag-scale filter), onboard station table.
4. New python/arlamx_v2/sc3.py: SC3 wrapper (10 d or 300 km episodes, data buffer, altitude/data
   scheduled reward weights, nGPS obs patching), trainer with width 16/18/20.
5. New advisors/mpc_v3.py: flow-frame, actuator-aware sampling MPC (same objectives/schedule as RL).
6. New python/arlamx_v2/int8.py: integer-only (CMSIS-NN style) actor inference, calibrated scales.
7. outputs/SC_v3_6mq/analysis: ev3, burn-in driver + ledger, MPC Monte Carlo, final eval, plots, report.
## CHECK 1 (SC_v3)
- C++ set_nav/clear_nav default off, cleared on reset; env `_gps_W` / `_tx_fn` / `_term_km` defaults reproduce the old
  power and termination paths exactly; pytest 268 passed after the plant rebuild and again at the end.
- Nav: stress test 36 cases (10 d, random/zero/tumble, storms) all finite; fixes gated (120 s), sanity fallback.
## CHECK 2 (SC_v3)
- Found and fixed during the campaign: negative per-step reward made early re-entry pay (alive term added, round 1 rerun);
  evaluator now loads each model's settings.json (foresight changes obs size); two nav divergence modes (large-gap
  velocity correction, drag-scale runaway near 300 km) fixed; nGPS burn-ins N03-N05/N07/N08/N10 rerun.
