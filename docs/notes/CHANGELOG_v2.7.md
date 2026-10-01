# ARLAMX v2.7.5 — gravity degree (2026-10-01)

`SH_MAX_DEGREE` is 70. The evaluator stores the GGM03S coefficients fully
normalised and recurses fully normalised associated Legendre functions, so a
70×70 field stays finite. Degrees 2, 8 and 24 match the previous unnormalised
evaluator to roundoff. Training presets stay at degree 2, 4 and 8. The Python
and compiled module version is 2.7.5.

# ARLAMX v2.7 — change notes (2026-09-25)

Restructure only. The plant, the reward and every default number are unchanged. The checks are recorded in `temp.md`:
36 behaviour fingerprints (configs, 10 env trajectories, PPO/SAC/TD3 weights, v12 weights + eval, decay CSV)
are bit-identical to v2.6, and `pytest` gives 213 passed.

- **Front door:** `main.py` (train, sweep, decay, snapshots, quantize, sim/eval/plot/legacy tools, run, test, verify).
  `python -m arlamx_v2 …` forwards to it. Guide: `commands.md`.
- **Settings:** `config/network.yaml` (neural net), `config/orbit.yaml` (training envelope + decay orbit),
  `config/train.yaml` (session), `config/sweep.yaml`. The plant YAMLs moved from `python/configs/` to `config/plant/`,
  byte-for-byte.
- **Snapshots:** each training session writes `outputs/snapshots/<MM-DD-HH-MM>_<ALGO>_<LxW>.yaml`. It holds the
  resolved network, orbit, train and plant settings, status, versions, the command and the result.
  `--from-snapshot` replays it exactly. The plant dicts are pinned for the run.
- **Env:** `ArlamxV2Env(orbit=…, frozen=…)`. Both are optional; omitting them gives v2.6 behaviour.
- **Trainers:** `train_one`/`build_model` take `run_dir`, `algo_kw`, `vecnorm`. `train_v12.train_run` also takes `arch`,
  `orbit` and `frozen`. The defaults are the old hard-coded values.
- **Sweeps:** dotted overrides (`base` + `grid`/`arms`). Each arm is a session. The ledger is
  `outputs/results/sweeps/<name>.csv`.
- **Outputs:** everything from before v2.7 is in `outputs/.old/`, and campaign/plot tools read and write there.
  New work goes to `outputs/snapshots`, `outputs/models` and `outputs/results`.
- **Archived** to `python/arlamx_v2/.old/`: train_v4, train_v56, run_v5_then_v6, run_v5v6_pipeline, tune_ppo,
  reeval_tune, plot_tune, v10_tune, bench_production, anim_production, pack_presentation, showcase_runs, tune_mpc_v3,
  the v2.6 cli.py and sweep.py, and the full v2.6 train_v12.py with its `round_*` sessions.
- `build.sh` drops a CMake cache that belongs to another source directory (for example after a rename).

## Critic pass (criticv1.md) — plant physics changes

**This changes the science plant relative to every campaign trained before v2.7** (all v8–v14
results in `outputs/.old` were produced on the old plant; their `snapshot.json` files were not
rewritten).

- **F1** SRP moment Σ r_c × F applied (was dropped); `panel_srp_optical` returns (F, τ);
  new step outputs `tau_srp_mean`, `tau_erp_mean`, `tau_gg_mean`, `tau_env_mean`.
- **F2** presets default to the **optical plate law** (`al_mylar`) instead of cannonball C_r = 1.8;
  pressure scales with heliocentric distance (AU/d)² (analytic Sun distance or CSPICE).
- **F3** power forecast no longer zeroes future sunlit generation when the step ends in eclipse.
- **F4** solar cells on both membrane faces (user) → |ŝ_B,z| kept, documented.
- **F5** gravity-gradient torque 3μ/r³ û × Iû (switch `gravity.gradient_torque`).
- **F6** Earth IR (on in eclipse) + albedo (day side) as plate sums, Knocke et al. 1988
  (switch `srp.earth`, thermal partition `srp.ir`).
- **F7** wind hook `Simulator.set_wind` (zero until a cited wind model is supplied).
- **F8** `drag_N` is the along-track drag; the onboard forecast carries the held along-track SRP.
- **F9** `altitude_km`, the 80 km stop, the 450 km flash cutoff and the forecast anchor use the
  geodetic height MSIS uses (features shift by up to ~21 km at high latitude; rewards not rescaled).
- **F10** CLL near α_N = 1 documented with measured numbers; stale "spec only" statuses fixed.
- **F11** optional Ap history (storm-time MSIS); `atmosphere.interp` (standard/high): 2 MSIS samples
  per step, linear in time across the substeps.
- Speed: γ > 6 limit in Sentman/CLL (≤ 1e-15), lunisolar ephemeris once per step.
  v10 env step 1.05 → 1.23 ms (new physics + interpolation).

## Vehicle and geometry

- **New variant `v10r6`**: 0.750 kg six-beam hex (80 g PCB + battery, 6 × 28.3 g torque rods along
  the beams, 500 g structure) from `config/plant/vehicle.yaml`; inertia computed
  (diag 0.0436/0.0436/0.0872 kg m²); action = quaternion + 6 rod on/off gates; minimum-norm rod
  allocation over the rods that are on. Rod dipole/power and the beam/membrane mass split are
  stated assumptions. v3–v11 keep their original vehicle cards.
- **STEP import**: `geometry.load_mesh` reads `.step/.stp` (trimesh + cascadio / OpenCASCADE);
  `python main.py sim geometry --in part.step` writes the simplified `.geom`.

## Style

Live code: one-line description at the top of each file, equation citations inline, other comments
and docstrings removed; 9 long names shortened. Verified behaviour-neutral (AST / preprocessed-token
identity per file, 40 fingerprints bit-identical, 259 tests).

## SC_v1_6mq (2026-09-25)

SolarCat_v3 176-plate model (`data/SolarCat_v3_1pct_plates.geom`), `vehicle_v3.yaml` (0.750 kg, six 0.8 A m^2
rods on the hexagon sides, `beam_angle0_deg`), `mtq_duty` (rods on-fraction; magnetometer reads in the gap),
`sensors_gaussian.yaml`, env kwargs `vehicle`, `sensors_cfg`, `mtq_duty`, training module `arlamx_v2.sc6mq`,
analysis + report in `outputs/SC_v1_6mq/`. Found: `geometry.simplify_sides` emits far-away sliver centroids on
SolarCat_v3 (not used; to fix).

## SC_v2_6mq (2026-09-25)

- Plant: `cmd_frame` = inertial | flow | flowB | flowS — the advisor command is held in the flow frame (e1 = v_rel)
  and re-evaluated every inner step with orbit-rate feed-forward; roll reference orbit normal / field / Sun;
  `Simulator.set_cmd_frame` for the mode manager. `dcm_to_quat` (Sheppard).
- Plant: torque-space least-squares allocation for rod arrays (`apply_rods`), replacing the projection of the 3-D
  minimum-norm dipole. `Simulator.set_inertia` (full tensor; env uses it when products of inertia are non-zero).
- Vehicle: `vehicle_v3c.yaml` — 7 cm static margin on the min-drag axis (avionics at the leading vertex + trim),
  `pcb_pos_m`, `trim`, optional `normal_coil`; `vehicle.py` places them.
- Env: `advisor_s`, `inner_dt` (cadence), brownout sun-search commands converted into the command frame.
- `sc6mq.py`: SC6mqV2 wrapper (mode manager: eclipse -> min drag + B-aligned roll + all rods; sunlit -> advisor +
  Sun roll; zero action = min drag, rods on), configurable PPO, step-limited training.
- Outputs: `outputs/SC_v2_6mq/` (REPORT.md, 13 figures, burn-in ledgers, cadence sweeps).

## SC_v3_6mq campaign (2026-09-26)
- C++: `Simulator.set_nav(r, v)` / `clear_nav()` — flow-frame ZOH from an onboard nav state (two-body RK4), off by default.
- env.py: `_gps_W` (receiver load), `_tx_fn` (onboard transmitter gate), `_term_km`; info `tx_on`, `dl_on`.
- New: sc3.py (10 d / 300 km episodes, data buffer, scheduled weights, survival term, foresight obs, nGPS),
  nav.py (fp32 J2+drag propagation, panel eclipse-timing filter), int8.py (integer-only actor), advisors/mpc_v3.py.
- Configs: sc_v3_6mq.yaml, sc_v3_6mq_ngps.yaml. Results: outputs/SC_v3_6mq/REPORT.md.
