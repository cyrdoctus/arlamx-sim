time: 2026-08-20T00:00:00Z
agent: claude
style: detailed

# ARLAMX V2.0 — three-pass audit (equations, structure vs V1.7, speed/numerics)

**Requested by:** user, 2026-08-19. "Go through V2.0 three times: check accuracy of all equations and structure, confirm it fixes what V1.7 did wrong and builds on it with faster code without missing anything critical. Report only — no code changes."

**Scope:** `ARLAMX V2.0/` (cpp, python, tests, docs, outputs) read-only; `ARLAMX-V1.7/` read-only reference. Nothing was edited, deleted, or moved in either tree. Numeric claims were verified by independent computation against the compiled `arlamx_cpp` module and the shipped data files (`GGM03S.txt`, `WMM.COF`).

**Method:** three independent passes, run in parallel with separate contexts, then cross-checked:

- **Pass 1 — equation accuracy.** Every C++ physics module re-derived from primary sources (Sentman 1961 / Doornbos 2012, Montenbruck & Gill, Vallado, Schaub & Junkins, Moe & Moe) and checked line-by-line against code, `docs/modules/` specs, and the V1.7 equivalents.
- **Pass 2 — structure and V1.7 parity.** Checklist of V1.7 defects (M-01…M-04 + survey items) verified against V2.0; full capability sweep for silent drops; Python-layer structural review; test/doc claims audited.
- **Pass 3 — speed and numerics.** Measured where the ~2830 steps/s comes from and what fidelity paid for it; hot-loop and pybind boundary audit; independent numeric spot-checks of the five most error-prone equations; robustness and test-tolerance review.

---

## Executive verdict

**The core force model is right and the speed is real.** Sentman FMF (M-01 closure), panel SRP (cannonball + new optical plate), GGM03S spherical harmonics, frames/GMST/Bowring/Sun, third-body form, MRP kinematics + RK4, MRP-PD, and B-dot were all independently re-derived and verified — several to machine precision (Sentman ≤5e-16 vs an independent Doornbos implementation; SH gravity 2e-9 vs an independent ∇U; MRP identities 4.4e-16; two-body RK4 SMA drift 1.6e-7 m over 3.6 orbits). The measured throughput (400k PPO steps in 141 s ≈ 2831 steps/s) is reproducible from artifacts and is overwhelmingly an honest architecture win: V1.7's Python aero kernel alone costs ~212 ms per advisor step; the entire V2.0 C++ plant does that step in 0.54 ms.

**But three classes of problems were found:**

1. **Two genuine equation defects**, both in environment models that are new or newly in-housed in V2.0: the WMM evaluator's Legendre normalization is mathematically wrong (1–34 % field error), and the analytic Moon position is returned in ecliptic coordinates but consumed as equatorial (12–20° direction error).
2. **The SC_v3a bake-off environment is off its own approved comparison contract** (corotating ON, no sensor noise, no rate/torque caps, 6-station GS, 300 s atmosphere hold, reduced domain randomization) — the physics of each change is fine or better, but "V2.0 vs V1.7" comparative claims are not currently supported, and no test pins V2.0 against a V1.7 numeric oracle despite both plans requiring it.
3. **Four contract-scoped capabilities were dropped without acknowledgment**: sensor noise, config/provenance snapshots, ONNX/actor export (leaving every V2.0-trained policy undeployable), and the entire CatSat line.

---

## Consolidated findings, ranked

### CRITICAL

**C-1. WMM Legendre recursion is mathematically wrong** — `cpp/src/mag/field.cpp:69-94`. Found independently by Pass 1 and Pass 3 with matching numbers. The code applies a per-element Schmidt scale factor √((2n−1)(2n+1)/((n−m)(n+m))) on top of a recursion whose inputs are already scaled — a double normalization the code's own comment half-admits ("this recursion is approximate"). Verified vs exact Schmidt semi-normalized functions on the same `WMM.COF` coefficients: P₂₀ inflated by exactly √(5/3)=1.291 (+29 %); P₆₁ has the wrong sign; end-to-end field error at 400–450 km is **434–14 827 nT, i.e. ~1–34 % of |B|**, worst near the poles. Only n=1 and sectoral n=m terms are right. The sanity test (`tests/mag/test_dipole.py:26`, 10–80 µT band) cannot catch it. Spec `09_magnetic_field.md` explicitly demands the official WMM report algorithm — violated. **Regression vs V1.7**, which took B from Basilisk's correct WMM. Mitigating: the training env never sets `wmm_path`, so the default path uses the dipole; the defect corrupts any run that loads WMM.

**C-2. The SC_v3a comparison runs violate the approved bake-off contract** — `APPROVAL_PLAN.md` §2/§13 pins `corotating_atmosphere: false` for the comparison and promises not to "quietly turn on … co-rotation … during the bake-off." The shipped env sets corotating ON for every variant including v3 (`python/arlamx_v2/env.py:103`; C++ default true, `api.hpp:27`; V1.7 changelog quantifies +11.7 % drag at i=23°). The change is verified good physics (`verify_corotating.md`) but is documented only in ACEnv side-logs, not in the plan, handoff, or `COMPARISON.md`. Together with M-1 below, **any "V2.0 vs V1.7 SC_v3a" physics/decay/reward comparison is off-contract** until either the contract flags are restored for the bake-off run or the deviation is promoted into the top-level docs and the comparison re-framed.

### MAJOR — equations / physics

**M-1. Bake-off parity deviations beyond the acknowledged ones** (Pass 2, line-by-line vs `config_SC_v3a.yaml`):
- **Sensor noise: none anywhere.** `_obs()` is noiseless truth (`env.py:177-242`); the plan's own pipeline diagram and §10.3 require Gaussian sensor noise with Earth Cup numbers. Unacknowledged.
- **Rate/torque caps absent for v3/v4.** V1.7 caps body rate at 0.125°/s per config; V2.0 v3 never calls `set_controller`, C++ defaults are uncapped (`mrp_feedback.hpp:11-12`), so v3 slews essentially instantly where V1.7 takes ~6 min for 45°. Caps exist only for v5+ (`env.py:114-117`).
- **MSIS cadence cut 30×**: one query per 300 s advisor step, held over 150 substeps (`env.py:391-408`) vs V1.7's every-10 s refresh (`atmosphere_cache_substeps: 5`). Near the terminator, per-substep drag can be off by O(2×) locally. Pass 3 measured the buy-back cost at ~0.6 ms/step — nearly free, since training is SB3-bound, not plant-bound.
- **Ground stations: 392 SatNOGS → 6 hand-picked cities**, highest-elevation re-pick, no sticky dwell (`advisors/stations.py:15-22,51-71`). Only the stickiness loss is admitted in the handoff.
- **Domain randomization reduced** for v3: ecc=0.0, RAAN/AoP/TA=0 (`env.py:315-322`) vs V1.7's full sweeps.
- **`shade_conservation` semantics changed**: constant 0.1 draw when in eclipse (`reward.py:37-38`) vs V1.7's actuation-effort-weighted term (`advisor/reward.py:905-917`) — the term can no longer teach "go quiet in shadow."
- **PPO `n_steps` 128 vs V1.7's 256** (`train.py:35`), undocumented. `momentum_penalty` omitted from `compose_sc_v3` though `docs/modules/12` lists it (numerically ~0 without wheels — doc/code mismatch).
- What **does** match, verified term-by-term: the 26-dim observation (features and normalizations), reward weights dE_vs_baseline/power_budget/gs_alignment/action_smoothness/omega_penalty/altitude_cliff, the power constants, mass/inertia/α_E/T_w/dt/substeps/45° slew clip/VecNormalize/lr/arch/seed.

**M-2. Analytic Moon in the wrong frame** — `cpp/src/orbit/frames.cpp:81-82`. The Meeus series coefficients are correct, but the returned unit vector is built directly from ecliptic (λ, β) with no R₁(−ε) obliquity rotation, while the SPICE path and `accel_third_body` consume it as equatorial J2000. Verified 12–20° direction error over Jan 2026; the code's Moon declination can never exceed ±5.3° (real: ±28°). The Sun function applies the rotation; the Moon one doesn't. Latent today (`lunisolar` defaults false; SPICE bypasses it) — but any lunisolar run without kernels silently mis-points the Moon. No Moon test exists.

**M-3. Dipole field polarity inverted (and untilted, contra spec)** — `cpp/src/mag/field.cpp:16-18`. m = +7.94e22 ẑ gives southward equatorial B; the real Earth's equatorial field points north. `tests/mag/test_dipole.py:12` (`assert B[2] < 0`) enshrines the wrong sign. Spec 09 promises a *tilted* dipole (g₁⁰ g₁¹ h₁¹); the implementation is untilted, the header comment "tilted" is false, and the calibration comment (30 103 nT) doesn't match the constants (30 601 nT). B-dot detumbling is polarity-invariant, so training is unharmed, but the observation feature `bhat` is globally flipped and any magnetometer-realism or tilted-dipole upgrade inherits the error. Since training never loads WMM, **the field the policies actually see is this untilted, sign-inverted dipole** — a fidelity step below V1.7's Basilisk WMM.

**M-4. No V1.7 numeric-identity gates exist**, though both planning documents made them the definition of done (Overview §5.8 "the port is wrong until these pass"; APPROVAL_PLAN §9 "V1.7 oracles"). Zero tests reference a V1.7 number; the closest are loose classroom bands (1.8 < Cd < 2.4). Gravity is genuinely refereed vs Basilisk (and independently confirmed in this audit), but aero, SRP, controllers, and the fused step are refereed against nothing external. `COMPARISON.md`'s claim that "plant identity is the 35 pytest gates + the Basilisk gravity referee" overstates what the gates prove. Note: Pass 3's independent verification substantially *de-risks* this gap for the equations themselves — but the promised oracle tests still don't exist as regression protection.

### MAJOR — scope / structure

**M-5. Dropped without acknowledgment** (each contract-scoped or fleet-critical):
- **Actor export (ONNX/INT8/STM32)** — APPROVAL_PLAN §3 lists it as V2.0 scope; nothing exists. All SC_v4–v7 winners trained on the V2.0 plant have **no deployment path**, and V1.7's exporter cannot load them. M-02 (tanh squash omission) therefore also remains unfixed anywhere.
- **Config/provenance layer** — V1.7 froze `model_config.yaml` + `model_card.yaml` per run; V2.0 hard-codes every physics/power number in `env.py:23-31,95-108,129-131` and records only `metrics.json`. Reproducing a run requires the git state of `env.py`.
- **Sensor models** (see M-1, first bullet).
- **CatSat line** — no CatSat env, RecurrentPPO, pointing-exclusion rewards, orbit_match/OD/EKF, or TLE ingest. Defensible scope, but no document declares it, and the plan's "does the V1.7 job" claim currently covers half the fleet.

### MAJOR — performance

**M-6. Per-call heap allocation of the Legendre table** — `cpp/src/orbit/gravity.cpp:118` allocates a nested `std::vector` on every `accel_ecef` call = ~4 200 allocations per advisor step (4 stages × 150 substeps × 7 allocs). Measured at ~20 % of plant time (0.11 of 0.54 ms). A fixed workspace (as `WMM::field_ecef` already does) reclaims most of it. This was already flagged in `validate_pass4.md`; the measurement confirms it as the top plant-level win.

### MINOR (selected; full lists in the pass reports)

- **Perf:** second full Sentman pass for the baseline Cd every 2 s substep with near-invariant inputs (~⅓ of plant time; cacheable per advisor step, `api.cpp:181-186`); B-field chain computed per substep but unused in point mode (`api.cpp:203-208`); acos→cos/sin round-trip per panel + √π and ΣA recomputed per call (`sentman.cpp:13-14,36-40,56`); inertia inverted per controller call (`mrp_feedback.cpp:30`); `step` holds the GIL for the whole 150-substep advance (moot under SubprocVecEnv, bites any threaded harness); no LTO/`-march` in CMake.
- **Numerics/robustness:** B-dot differentiator not reset on mid-episode detumble re-entry → one stale clipped Ḃ sample (`api.cpp:116` / `env.py:490`); no-GGM fallback drops J3 (V1.7's minimum model was J2+J3; `accel_j3` exists but is never called by the Simulator, `api.cpp:123`); WMM epoch year hard-coded 2026.0 (`api.cpp:206,305`); episode length computed from a fixed 400 km / R=6371 km orbit (`env.py:138`); GMST computed at step start but stations placed against the post-step position (~1.25° error, `env.py:394-396,465`); bare `except Exception` silently substitutes ρ=3e-12, T=900 on any MSIS failure (`env.py:406-407`) — silent constant-atmosphere training if pymsis breaks; power billed as a single 300 s lump using end-of-step eclipse (`env.py:244-265`); `min_drag_axis` scans only +X/+Y/+Z — exact for the symmetric hex, wrong in general (`api.cpp:20-37`).
- **Tests/docs:** README's "35 classroom tests" is stale (72 test functions in 19 files); `test_circular_sma_stable_one_period` tolerance 50 m vs measured 1.6e-7 m capability (1e8× margin); Basilisk referee test silently skips where Basilisk is absent; module 05 (atmosphere) has no dedicated test; spec 03 mandates Pines but code uses guarded classic Legendre (deviation acknowledged, and the guard is validated); spec 04's "reject r=0" classroom case untested and unimplemented; handoff figure paths stale after the `outputs/analysis/` restructure; hard-coded machine-absolute paths in `paths.py:7-14`, `conftest.py:7-9`, and the CSPICE conan path in CMake; `hostile.json` artifact stored inside `tests/`.
- **Code hygiene:** `accel_nongrav_env` is misnamed (it returns gravity + third-body, `api.cpp:120`); exported `rk4_step` (`integrate.cpp:34`) is dead code whose frozen-gravity semantics differ from the live loop — a maintenance trap; `_unmatched_length` in `geometry.py:229-262` is dead code with a no-op loop; `mpc.py` duplicates the power-constant block and MU/RE instead of sharing a source; env imports a private helper from `advisors.mpc` and `coe_to_rv` from run-script `decay_run.py` (layering); pybind exposes only `set_inertia_diag` though C++ accepts a full tensor; spec 11 says `step` returns `sim_time_s` but bindings return `"t"`.
- **Constants:** all verified correct (K_B/N_A 2019 SI exact; MU/RE/J2 match the GGM03S file to all printed digits; WGS-84 set exact). P_SRP = 4.56e-6 is the 1367 W/m²-era value (modern TSI 1361 → 4.54e-6, −0.45 %) — same as V1.7, internally consistent; cite the constant used.

---

## Verified correct (independently re-derived in this audit)

| Module | Result | vs V1.7 |
|---|---|---|
| `aero/sentman.cpp` | Exact Sentman/Schaaf–Chambré; M-01 closure `Tr/Ti = α_E·Tw/Ti + (1−α_E)s²/2` confirmed from the energy-flux derivation; ≤5e-16 vs independent Doornbos implementation across θ=0–89°, s=7.5; Cd(s=8, face-on, Tw/Ti=0.3) = 2.1370 (α=1) / 2.3673 (α=0.93), bit-identical to V1.7 post-M-01 | SAME (faithful port of the audited law) |
| `srp/panel_srp.cpp` | Cannonball and optical plate both canonical; absorber/specular/Lambert limits exact (P / 2P / (1+⅔)P); 1 AU-fixed pressure and cylindrical umbra documented | SAME + optical model added |
| `orbit/gravity.cpp` | SH gradient matches independent normalized-∇U to 2e-9 (FD-limited) and a line-by-line re-implementation to 2e-16; J2/J3 closed forms match Vallado exactly; J2 = −√5·C̄20 to all digits vs the shipped file; unnormalized approach safe to the stated N≤20; pole guard validated | IMPROVED (GGM03S vs J2+J3) |
| `orbit/frames.cpp` | GMST reproduces Vallado's worked example; Bowring exact on Vallado's example; Sun (Montenbruck) correct; LVLH/quat/slerp-clip/quat→MRP all correct | Sun SAME; **Moon new but buggy (M-2)** |
| `orbit/third_body.cpp` | Standard direct+indirect form; constants standard; double-precision cancellation acceptable at this fidelity | NEW capability |
| `attitude/mrp.cpp`, RK4 in `api.cpp` | All MRP identities verified numerically (B·Bᵀ=(1+σ²)²I to 4.4e-16; C(σ_err)=C_BN·C_RNᵀ to 8e-16); proper composition (not naive subtraction); classical RK4 with per-stage gravity and per-stage DCM; shadow only after completed steps; torque-free |H| and E_rot conserved to double precision; two-body SMA drift 1.6e-7 m / 3.6 orbits | SAME (equations); per-stage gravity better than the dead library `rk4_step` |
| `control/mrp_feedback.cpp` | Schaub & Junkins PD + gyroscopic feedforward; rate-cap logic correctly excludes the feedforward from the ω prediction | SAME |
| `control/bdot.cpp` | m = −k·Ḃ, τ = m×B, saturation; energy-decrease test is a real functional test | IMPROVED (now a live plant mode) |
| Plant assembly `api.cpp` | v_rel = v − ω⊕×r sign/frame correct; gas convention verified end-to-end (drag does negative work); B/N frame rotations correct; no unit mixing found; dE_drag + dE_lift = dE_actual closes at 1e-14; feasibility gates real and tested | IMPROVED overall |
| Python `atmosphere.py` | pymsis argument order, species indices, m̄, kg/m³, geodetic alt + GMST lon all correct | SAME |
| Python `geometry.py` | Area conservation enforced both ways; normals hint-aligned, right-handed basis, CCW verified; edge-seal with honest unmatched-edge audit | FIXES the facet-dump STL loader and rich476 misrepresentation |
| Advisors | Faithful conceptual ports; baselines carry the passive-quaternion fix | SAME/IMPROVED |

## Speed claim adjudication (Pass 3, measured)

- 2 831 steps/s (400k/141.28 s) confirmed from `outputs/training/sc_v3_ppo/metrics.json`; the 4M-step v6 run sustains 2 824.
- Source of the speedup: the fused `Simulator::step` doing all 150 substeps per Python crossing. V1.7's Python aero kernel alone: ~212 ms per advisor step (measured 0.707 ms/call × 150 substeps × 2 passes). Entire V2.0 plant: 0.536 ms. `env.step` total: 0.71 ms (plant 0.54, Python glue+reward 0.15, MSIS 0.02).
- **The plant is no longer the bottleneck**: 32 envs could feed ~45k steps/s; training achieves 2.8k, so SB3's PPO update + SubprocVecEnv IPC dominate. Plant micro-optimizations won't move training wall time; conversely, restoring V1.7's 10 s MSIS cadence would be nearly free.
- Fidelity that part-funded the number and must be disclosed next to it: 300 s MSIS ZOH (30× coarser than V1.7 training) and the untilted dipole replacing Basilisk WMM. Not dropped: dt=2 s RK4 (tighter than V1.7's 15 s backend default), SH degree 4, per-substep Sentman + SRP + baseline pass, α_E=0.93; corotating wind is a fidelity *gain* over V1.7 training (but see C-2).

## Cross-check against the earlier ACEnv verification (grok session)

- `verify_sentman.md`, `verify_gravity.md`, `verify_srp.md`, `verify_mrp.md`, `verify_corotating.md`: **all confirmed** by independent re-derivation; every "keep" decision in those files was correct.
- `validate_pass4.md`'s top recommendation (persistent Legendre workspace) is confirmed by measurement as the top plant-level optimization (M-6).
- The equation register (`2026-08-19_detailed_equation_register.md`) §9 states the WMM path uses "standard Gauss coefficients from a .COF file" — the coefficients are standard but the evaluator is not (C-1). The register also doesn't cover `moon_analytic` (M-2). These are the two spots the earlier verification missed; everything else it asserts held up.

## Bottom line

**Fixes what V1.7 did wrong:** yes for the core physics and the two headline defects that motivated the rewrite — the speed problem (genuinely solved, ~300× on the env-step path) and the geometry pipeline (sealed-side simplifier with validated 0.000 % attitude error replaces the facet dump / rich476 set). M-01 is correctly baked in; M-03 is fixed-by-default (but off-contract for the bake-off, C-2); M-04 is fixed. Basilisk is properly demoted to referee. Documentation discipline exceeds V1.7.

**Builds on it without missing anything critical:** not yet defensible as stated. The WMM evaluator is broken (C-1), the training magnetic field is a sign-inverted untilted dipole (M-3), the Moon is in the wrong frame (M-2), sensor noise / export / provenance / CatSat are silently absent (M-1, M-5), and no V1.7 oracle test exists (M-4). The safe thesis posture is the one `COMPARISON.md` half-takes already: V2.0 supports **speed** claims and **within-V2.0** policy comparisons; it does not currently support "same plant as V1.7" or policy-performance-vs-V1.7 claims without restoring the contract flags or adding the promised oracle gates.

**Priority order if fixes are later authorized:** C-1 (WMM recursion + a pinned NOAA test value), M-2 (one rotation + a declination test), M-3 (sign/tilt or amend spec 09), C-2/M-1 (either run the bake-off on contract flags or promote the deviations into the top-level docs), M-5 (export path — without it every V2.0-trained policy is undeployable), M-6 (Legendre workspace). No fixes were applied in this audit.
