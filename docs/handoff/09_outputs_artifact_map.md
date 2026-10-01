# 09 — Artifact map: where everything lives

Root `outputs/`. Sizes and dates from the 2026-09-13 inventory. ~880 model
zips in total. `outputs/README.md` (2026-08-19) describes only the first five
groups; the v-series folders came later.

| folder | files / size | dates | what |
|---|---|---|---|
| `training/` | 59 / 2.7 M | 08-18/19 | SC_v3–v6 runs (`models/*.zip`, `metrics.json`, `logs/`), `sc_v3_ppo/COMPARISON.md` |
| `tuning/` | 156 / 4.4 M | 08-18/19 | `tune_ppo/{best,board}.json`, 32 PPO trials, consistency/robust reruns |
| `campaign/` | 272 / 8.2 M | 08-19 | SC_v7 IPC: `LEADERBOARD.md`, `ledger.csv`, 53 runs (pre-field-fix; provenance only) |
| `analysis/` | 275 / 194 M | 08-18 → 09-07 | advisors, sc_v4_eval, ood*, gradient70, showcase, slides, geometry, optics, **decay/** (Sept studies), lift_drag |
| `presentation/` | 142 / 35 M | 08-20 → 08-24 | production bake-off (18 runs, `README.md`), `SC_v8_v9_report.md` (auto-generated), `plots_v8/` f1–f13, diagrams, animations_v11 |
| `conference_package/` | 31 / 25 M | 08-19 | curated 8-minute talk set, rev 2 |
| `v8/` | 1396 / 79 M | 08-20 → 08-23 | **`ledger.csv` (188 runs)**, `v10_tune.csv`, `v11_tune.csv`, 241 run dirs, `data/{ref,ref_v1,ref_v2,mc_decay,mc_decay_v2,damage_eval,duo_eval,inference_u575,traces}.json`, `mpc_tune/` |
| `mpc/` | 216 / 2.4 M | 08-23 | MPC_v2/v3 freeze: `REPORT.md`, `ledger.csv`, `freeze.json`, `ref_v3.json`, `mc_decay_v3.json`, `checks/GATE.md` |
| `v12/` | 1751 / 117 M | 08-23 → 08-25 | 42 runs with `ckpts/step_*/` (543 zips), `BEST_SO_FAR.md`, `S3_RESULT.md`, `STATUS_NIGHT.md`, `infer.json`, `traces/mc_lifetime.json`, slides |
| `v13/` | 159 / 44 M | 08-24/25 | **`ckpt/model.zip` = SC_v13 freeze**, `ckpt_v14{_keep,b,d,e}/`, `SC_v13*.md`, `SC_v14*.md`, `MC_FOUR.md`, `traces/mc_four.json`, `PREC_v14b`, `MTQ_AUTHORITY.md`, `ONBOARD_LOOP_U575.md`, PDFs |
| `v14/` | 86 / 19 M | 08-25 | `FORCE_RANGES.{md,json}`, `plots/` (82; `key_*` set), `PLOT_GUIDE.pdf` |
| `v14fix/` | 16 / 2.7 M | 08-25 | corrected talk figures 01–07, `README.md`, `SYSTEM.md` — the set to show |
| `geometry/` | 2 | 09-13 | `attitude_check_1pct.{csv,png}` (rewritten by the test suite) |
| `logs/` | 5 / 1.1 M | 08-18/19 | tune_ppo, v4ab_seq, v5v6_pipeline, ood_v4_long |
| `ARLAMX_V2_presentation.zip` | 14.5 MB | 08-20 | bundled talk pack |

## Which model to quote

| purpose | path | numbers |
|---|---|---|
| Current SC freeze for a single quote | `outputs/v13/ckpt/model.zip` (+ `vecnormalize.pkl`) = `s4_env0.35_…_s43/ckpts/step_150048` | quiet decay 12.08, band 80.4, dl 39.1, B +0.45; MC 12.9 d, 8.48 J/day |
| Best MC brownout count | `outputs/v13/ckpt_v14b/` | MC 12.9 d, 4/24 brownouts, 5.84 J/day |
| Best MC lifetime among v14 | `outputs/v13/ckpt_v14d/` | 13.3 d, 5.72 J/day, quiet 10.03 km/d, 85.9 % band |
| The 0.096 artifact | `outputs/v8/runs/v10_4x18_ppo_1000k_s42/` | gap 0.096; siblings 0.82 / 3.10 |
| B record | `outputs/v8/runs/v11_4x40_ppo_4000k_s42/` | B 0.745, decay 20 km/d |
| MPC reference (v1) | `python/configs/mpc_v1.yaml`, `outputs/v8/data/ref.json` | 9.28 / 86.6 / 23.2 / 2.97 J |
| MPC frozen (v2) | `python/configs/mpc_v2.yaml`, `outputs/v8/data/ref_v2.json` | 9.19 / 84.1 / 18.6 / 3.73 J |

Ship bar (`outputs/v14fix/README.md`): the 24-draw MC. No RL zip clears it.

## Newest artifacts (September 2026 studies, no training)

`outputs/analysis/decay/`:
`compare_hex_vs_stl_500to300_mass.csv`, `{hex,stl1pct}_500to300_m*/`,
`aoa45_500to300/` (README.md, summary.json, `srp_assessment/`,
`srp_f107_sweep/`), and `outputs/analysis/lift_drag/stl1pct_i45_m1p75/`.
Details in `11_decay_lift_srp_studies.md`. The older v2.0 CSVs at the top of
`analysis/decay/` (250 km stop, pre-v2.1 physics) are kept untouched.

## Reports (`ACEnv/Reports/`)

Chronological: 2026-08-18 decay min/max, decay recheck, mpc/heuristic
campaign, sc_v4 plan · 08-19 equation register, OOD gradients, sail optics,
sc_v4 training, v2 triple audit, v5/v6 plan · 08-20 independent v2.0 physics
audit, production bake-off, v2 fixes applied, v8 fixes after grok, v8/v9
campaign results, v8/v9 triple critique · 08-21 v10/v11 campaign results ·
plus `validate_pass1–4.md`, `verify_{corotating,gravity,mrp,sentman,srp}.md`.
`ACEnv/env.md` is the agent use log (grok and claude sessions, dated).

## Reproducibility contract

Every v8+ run: `snapshot.json` (resolved reward/gains/power/estimator/physics
YAML + arch/algo/seed), `metrics.json`, `models/` or `ckpts/`,
`vecnormalize.pkl`. Ledgers: `outputs/v8/ledger.csv`,
`outputs/training/SWEEP_LEDGER.csv` (v2.6 sweep runner),
`outputs/v12/runs/<name>` for `train_v12` jobs. Report generators
(`report_v8.py`, `build_mpc_report.py`, `plot_v8/v12/v14fix.py`) read every
number from these files; regenerate, do not hand-edit.
