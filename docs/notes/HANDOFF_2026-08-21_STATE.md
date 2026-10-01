# ARLAMX V2.0 — full state handoff (2026-08-21, pre-reboot)

Written so work resumes cold from any machine or session. **No background jobs
were running at write time — safe to reboot.** Everything below is on disk and
verified; nothing lives only in a session's memory.

---

## 1. Where the program stands, in one paragraph each

**Plant (C++).** Audited and fixed (WMM Schmidt recursion, lunar ecliptic→equatorial
frame, tilted dipole polarity, J3 fallback, stack Legendre workspace — 17 % faster,
physics bit-identical). 149 pytest gates green, Basilisk gravity referee at 1e-8.
The v8+ vehicle: 0.625 kg (confirmed), custom wound coils sized for 300 km worst
case (1.473/1.473/0.442 A·m² → 38.24/38.24/11.48 µN·m at the weakest field,
249 mW full 3-axis, 161 g = 25.7 % of vehicle — flagged), 1 cm cp–cm offset baked
into the v8+ panel geometry so the simulated disturbance matches the coil sizing.

**Model generations.** v7 (catalogue coils, historical) → v8a/v8b (delegation
ablation; gates pinned vs live split) → v9a/b/c/d (temporal context: future-only /
3+3 / 6 past / 6+6) → v9Duo → v10 (perturbation-first reward, widened weather
F10.7 5–250 & Ap 2–200, random storms, 120-orbit episodes, v9d observation layout,
budgets to 1M) → v11 (v10 + mid-episode membrane damage, adapt reward, 160-orbit
episodes, budgets to 4M). All reward/gain/filter/power values live in
`python/configs/*.yaml` with provenance tags; **every trained run froze its
resolved config in `<run>/snapshot.json`**.

**Headline results (all on the v8+ plant; v7 numbers are a different vehicle):**

| claim | number | caveat |
|---|---|---|
| v10 meets "within 10 % of MPC" | v10_4x18_ppo_1000k_s42: **gap 0.096, decay 8.78 vs MPC 9.28 km/d** | seed replication 0.096/0.82/3.10 → cell median 0.82; the 0.096 model is a saved artifact, not the typical outcome |
| Where RL beats MPC | Monte Carlo 512 orbits: v10 artifact **~18 % lower decay at the 300 km crossing** (72.6 vs 88.2 km/d) | MPC still owns total lifetime 500→300 (17.9 d vs 11.9 median) |
| Decision cost (STM32U575, analytic 0.35 FLOP/cyc) | advisors 0.14–0.76 ms · Duo 5.5 ms · MPC 8.4 ms | bench on hardware; workstation times are not flight numbers |
| Best temporal layout | v9d (6 past/6 future) best run 0.387; v9c best median 1.20 | v10 uses v9d; variance managed by multi-seed burn-ins |
| v11 delegation | v11_4x40_ppo_4M: gap 0.643, **B = 0.745 program record**, holds ~0.4 post-strike | — |
| v11 honest negative | multi-hole strikes: v11 decay 41→94 km/d, **not better than v10's** 25→54 | adapt reward stabilised TRAINING (11 brownouts → 0 in round a0) and preserved delegation, not post-strike orbit-holding |
| Duos | v8Duo and v9Duo both == their main advisor exactly | arbiter never switched; weak horizon advisors |

**Reward tuning history** (all in `outputs/v8/v10_tune.csv`, `v11_tune.csv`, and the
`[TUNE-n]` comments in the YAMLs): v10 r0–r2 = 13 candidates × 3 seeds, gap
1.664 → 0.921, only dE 2.6 helped; v11 a0 = adapt weights 4.0/4.0 decisive.

---

## 2. Artifact map (all under `ARLAMX V2.0/`)

```
outputs/v8/ledger.csv                 188 runs: name,variant,algo,arch,steps,seed + probe metrics + gap
outputs/v8/runs/<name>/               model zip, vecnormalize, snapshot.json, probe_metrics.json
outputs/v8/data/                      ref.json (MPC/heuristic on THIS plant), burnin.json,
                                      mc_decay.json (512x5), damage_eval.json, inference_u575.json,
                                      traces.json, duo_/v9duo_{pick,eval}.json
outputs/v8/*.log                      every stage's stdout (sweep, v9cd, tune rounds, matrices, MC, damage)
outputs/presentation/plots_v8/        13 figures f1..f13, PNG+PDF
outputs/presentation/diagrams/        v8_onboard / v8duo_onboard / v9_onboard (+ DIAGRAMS.md walkthrough)
outputs/presentation/animations_v11/  damage_{mpc,v10,v11}_tear.mp4/.gif (strike on camera)
outputs/presentation/SC_v8_v9_report.md   auto-generated master report (report_v8.py — regenerate, don't edit)
ACEnv/Reports/2026-08-2*_detailed_*.md    audit → fixes → v8/v9 campaign → grok critique → fixes → v10/v11 results
docs/modules/14_reward_v8.md          every equation, numbered, with provenance
python/configs/*.yaml                 the single source of truth for all values
```

Older-but-kept: `outputs/campaign/` (v7, pre-mag-fix — provenance only),
`outputs/presentation/{plots,runs,data}` (v7 bake-off), `outputs/ARLAMX_V2_presentation.zip`.

---

## 3. How to resume (exact commands)

Environment: **Linux only as-is** (hard paths `/mnt/storage/...`, `/home/nekolny/miniforge3/envs/ThesisMS`,
Basilisk data paths in `python/arlamx_v2/paths.py` + `tests/conftest.py`). On Windows
use WSL or re-point those paths; the C++ lib rebuilds with `./build.sh`.

```bash
cd "/mnt/storage/ARLAMX_Dev/Thesis/ARLAMX V2.0"
PY=/home/nekolny/miniforge3/envs/ThesisMS/bin/python
./build.sh                                   # rebuild plant if needed
PYTHONPATH=python $PY -m pytest tests -q     # must be 149 passed
PYTHONPATH=python $PY -m arlamx_v2.report_v8 # regenerate master report from artifacts
PYTHONPATH=python $PY -m arlamx_v2.plot_v8 --set 3          # all 13 figures
PYTHONPATH=python $PY -m arlamx_v2.bench_v8 --stage <...>   # burnin|ref|sweep|duo|v9duo|v9|v10|v11|trace
PYTHONPATH=python $PY -m arlamx_v2.v10_tune --round <r0..a1> [--variant v11]
PYTHONPATH=python $PY -m arlamx_v2.mc_decay --runs 512 --policies mpc,heuristic,v10:1
PYTHONPATH=python $PY -m arlamx_v2.damage_eval --stage eval|anim
```

Key models by ledger name: `v10_4x18_ppo_1000k_s42` (the 0.096 artifact),
`v11_4x40_ppo_4000k_s42` (B-record), `v9d_4x32_sac_300k_s42`, `v9b_4x32_ppo_300k_s42`.

---

## 4. Open decisions / queued next steps (none started)

1. **v11 multi-hole gap**: adapt reward needs a decay-holding term while damaged
   (e.g. scale `decay_stab_weight` up under `damage_active`), not only
   consistency + delegation-hold. Then re-run `damage_eval`.
2. **Duo on a strong main**: both Duos equalled their mains; re-assemble on the
   0.096 artifact (`bench_v8.stage_duo(main_name=..., base_variant="v10")`).
3. **Seed-replicate** `v11_4x40_ppo_4000k` before any claim rests on it.
4. User's queued ideas, explicitly deferred until this data was in:
   **v10Duo, v10Tri, v10Phy** (physics-informed network for ADCS/orbital dynamics).
5. Plant items deliberately out of scope pending separate decisions: dipole-space
   torque saturation in C++, sensors into `_obs` + attitude filter, ONNX/MCU
   export (still absent — no policy is deployable), MSIS 300 s hold, actuator
   mass 25.7 % review, damage-eval recovery metric too generous (saturates at
   1 step; use post-strike decay and B as discriminators).
6. Animations for the conference: user picks 2–3 plots first; strike-on-camera
   trio exists; MPC/v11 "response" animations can reuse `damage_eval.stage_anim`
   or the v7 `ipc_animation` machinery on new recordings.
```
