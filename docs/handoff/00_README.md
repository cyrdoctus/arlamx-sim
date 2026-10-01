# ARLAMX handoff bundle — read me first

**Written:** 2026-09-13, from the ARLAMX V2.1 tree at package version 2.6.
**Purpose:** seed a PhD dissertation project folder with everything the repo,
its reports, and the September 2026 review sessions established. These files
are a snapshot, not living documentation. Copy them out, then let the repo's
own `docs/`, `CHANGELOG_*.md`, and `outputs/**/README.md` take over.

**Audience:** the author, and any LLM agent (Claude Code, Codex, Cursor/Grok)
that is handed the dissertation folder cold. Every number here is traceable
to a file path; if a number has no path next to it, it came from a measurement
made during the 2026-09-13 review session and is labelled as such.

## Reading order

| # | File | What it gives you |
|---|---|---|
| 01 | `01_program_history_and_state.md` | Timeline V1.7 → V2.0 → 2.1 → 2.5 → 2.6, and where things stand today |
| 02 | `02_repo_map_and_toolchain.md` | Layout, mamba env, build, test, data files, how paths resolve |
| 03 | `03_plant_physics.md` | The C++ plant: every force/torque model, its equation, defaults, tests, citations |
| 04 | `04_actuator_and_control.md` | MRP-PD, magnetorquer allocation, gains, and the closed-loop authority finding |
| 05 | `05_gym_env_and_variants.md` | Observation/action contract and the SC_v3 → v14 variant ladder |
| 06 | `06_reward_and_delegation.md` | Reward families, the numbered delegation equations, tuning verdicts |
| 07 | `07_baselines_mpc_heuristic.md` | The sampling MPC (v1/v2/v3) and the heuristic: what they are and how they score |
| 08 | `08_campaign_results_scoreboard.md` | Every headline result with its caveat and file of origin |
| 09 | `09_outputs_artifact_map.md` | Where every model, ledger, trace and figure lives; which freeze to quote |
| 10 | `10_commands_and_workflows.md` | The command catalog and the exact recipes that reproduce things |
| 11 | `11_decay_lift_srp_studies.md` | The September 2026 fixed-attitude decay, lift and SRP studies (six-petal sail) |
| 12 | `12_open_problems_and_dissertation_threads.md` | Honest negatives, unresolved decisions, and what a dissertation can defensibly claim |
| 13 | `13_glossary_and_references.md` | Terms, symbols, and the literature the code cites |
| 14 | `14_llm_agent_context.md` | How to put an LLM to work on this repo without it corrupting results; skill plan |

## Three facts that override everything else

1. **Every campaign number (v7–v14) was produced on `ideal_torque=true`**, the
   v2.1 plant where the PD torque is applied as an ideal body couple. Since v2.5
   the packaged default is `ideal_torque=false` (coils through the local field),
   and on that plant the campaign gains cannot hold attitude at all
   (`04_actuator_and_control.md`). State the actuator model in every quoted number.
2. **The v8b delegation reward term `deleg_power` was identically zero in every
   run before 2026-09-13** because the reward was fed the gated torque command.
   The delegation *mechanism* was live; its *reward* was not (`06_reward_and_delegation.md`).
3. **Seed variance exceeds most design deltas.** The one artifact that beat the
   MPC on decay (`v10_4x18_ppo_1000k_s42`, gap 0.096) has seed siblings at 0.82
   and 3.10. Medians over ≥3 seeds are the reportable numbers.

## Provenance of this bundle

Sources read in full: `README.md`, `HANDOFF_2026-08-21_STATE.md`,
`claude_handoff.md`, `APPROVAL_PLAN.md`, `CHANGELOG_v2.1/2.5/2.6.md`,
`docs/COMMANDS.md`, `docs/VALIDATION_STANDARD.md`, `docs/modules/01–17`,
`docs/v12/*`, `outputs/v14fix/{README,SYSTEM}.md`, the `ACEnv/Reports/*`
headers and the 2026-08-19 equation register, the 2026-08-20 triple critique,
the 2026-08-21 v10/v11 results; plus an inventory of `outputs/`,
`python/arlamx_v2/`, and `python/configs/`. The 2026-09-13 review session
(CLL review, closed-loop diagnosis, observer and delegation fixes) is
first-hand.
