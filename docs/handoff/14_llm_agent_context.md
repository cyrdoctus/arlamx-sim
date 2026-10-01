# 14 — Working with LLM agents on this repo

This project was built almost entirely through LLM agents (grok, claude,
Cursor) between 2026-08-18 and 2026-09-13, with the author directing and
reviewing. The pattern that worked: one agent implements, a *different* agent
audits against sources before compute is spent, the author decides. The
pattern that failed: an agent reporting a subset of tests as the whole suite,
and a physics change (v2.5 closed loop) shipped without checking that the
downstream premise (attitude hold) still held.

## What an agent must be told before touching anything

1. Interpreter: only the `arlamx` mamba env (Python 3.12). `./build.sh`
   refuses others. `PYTHONPATH=python` for every command.
2. `../ARLAMX-V1.7/` is read-only. Never edit, copy, or move it.
3. Outputs only under `outputs/`; tests write nothing there
   (`outputs/geometry/attitude_check_1pct.*` is the known exception).
4. Report the full `pytest tests -q` count. Never mark a failing test
   `xfail` without the diagnosis next to it and a changelog line.
5. Every quoted number states its actuator model (`ideal_torque`), plant
   version, and file of origin.
6. No edits to `gains_mrp.yaml`, `power_mtq.yaml`, the MPC freeze YAMLs, or
   `env.py` without a named decision from the author.
7. A new physics knob touches five places: the YAML preset, `physics.validate`,
   `physics.apply_to_params`, `SimParams` (+ bindings), and a test in
   `tests/physics/test_physics_wiring.py`. Missing any one is a silent default.
8. A new reward term must pass the burn-in gate that prints its activity
   (`bench_v8 --stage burnin`) before any run longer than 40k steps.
9. ≥ 3 seeds per cell; medians; checkpoints by eval curve.
10. Check equations against the cited source PDF, not against the docstring
    (the CLL review found a bug-free transcription and a wrong preset by
    doing exactly this).

## Planned structure (agreed in outline 2026-09-13, not built)

```
AGENTS.md                      canonical rules (the list above) + repo map + how to run
CLAUDE.md                      one line: @AGENTS.md
.claude/skills/<name>/SKILL.md the skills, plain Markdown with name/description/when-to-use
.cursor/rules/*.mdc            one paragraph each pointing at the matching SKILL.md
docs/agents/                   the same skills rendered into the MkDocs site
```

Skills worth writing first (each with the exact commands inside):
`build-plant`, `run-suite`, `physics-preset` (the five-file rule),
`run-campaign` (train/sweep/decay + ledgers), `review-physics` (checklist:
source PDF, regime stated, actuator stated, full-suite count),
`changelog-and-docs` (module note + COMMANDS.md + changelog + test count).

## Documentation toolchain (recommended, not built)

MkDocs + Material (Markdown you already have; MathJax for the `\(...\)`
equations in `docs/modules/`), mkdocstrings for the Python API (620 public
defs with docstrings), Doxygen + doxygen-awesome for `cpp/include`, and a
small script that dumps each module's argparse into the command catalog.
`mkdocs serve` for students, `mkdocs build` for a static site.

## How the September 2026 review sessions were run (for reproducibility)

Read every file in the change batch by mtime; verify equations against the
primary source (ADBSat PDF fetched and read page by page); run the built
module numerically on the spec's own test cases; run the full suite; probe
any failing test under alternative regimes (ideal vs closed loop, presets,
inclinations, gains) before diagnosing; fix the root cause in the plant, not
the test; then re-derive the test's premise from the physics. The findings and
numbers in `04` came from that procedure and are reproducible with the
snippets in `docs/modules/15_control_magnetorquer.md`.

## Agent-visible memory that already exists

`ACEnv/env.md` (agent use log, dated entries per session), `ACEnv/Reports/`
(every audit and campaign report), `CHANGELOG_v2.*.md`, and per-run
`snapshot.json`. Point a new agent at `00_README.md` of this bundle first,
then `docs/COMMANDS.md`, then the changelog for the version it is touching.
