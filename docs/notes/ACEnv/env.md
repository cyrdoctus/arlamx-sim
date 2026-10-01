time: 2026-08-18T22:45:00Z
agent: grok

# AstroChimera use log

**Mode:** single advisor (Grok as main LLM). Advanced generation. Detailed reporting.

**Start:** 2026-08-18
**End:** (open)

**Update 2026-08-19:** Full equation register; corotating default on with lift-work split; optical plate SRP presets; CLL/CCL FMF plan only.

**Update 2026-08-20:** Independent five-pass physics/code audit of V2.0 (no prior LLM V2.0 review read; no source edits). Report: `ACEnv/Reports/2026-08-20_detailed_independent_v20_physics_audit.md`. pytest 74 passed. High flags: untilted opposite-polarity dipole in Gym, 300 s frozen MSIS, corotation default vs V1.7 bake-off, spherical vs geodetic altitude, polar SH vs Basilisk, Moon ecliptic-as-equatorial (lunisolar off).

Session work:

- Opened ACEnv on ARLAMX V2.0. V1.7 untouched.
- Three-pass verification of Sentman, GGM03S, panel SRP, co-rotating wind, MRP.
- Four-pass validation of the assembled plant (main commands, hostile numerics, feasibility gates, speed/precision proposal).
- Diagnosed NaN decay as MRP-PD at 180 s; added `prescribed` attitude mode.
- Regenerated min/max-drag 500→250 km at i=23°, e=0.001, F10.7=150, Ap=4, corotating + SRP, 1 % SolarCat: 18.81 d / 10.85 d.
- Ported a heuristic family and sampling MPC; campaign on the 1 % plant.
- SC_v4 training written as a plan only.

**Update 2026-08-20 (agent: claude).** Three-pass audit, then fixes, then a production bake-off.

- Three independent passes over V2.0 (equations / structure vs V1.7 / speed + numerics) →
  `Reports/2026-08-19_detailed_v2_triple_audit.md`.
- Fixed the two real equation defects (WMM Schmidt recursion, lunar ecliptic→equatorial frame), the
  inverted+untilted dipole, J3 in the no-GGM fallback, the hardcoded WMM year, and the B-dot
  differentiator reset. Plant is 17 % faster (0.536 → 0.443 ms/step) with physics bit-identical.
  Tests 74 → 83. → `Reports/2026-08-20_detailed_v2_fixes_applied.md`.
- Measured the invalidation: the MPC reference reproduces exactly on the fixed plant, but the previous
  campaign winner degrades 0.468 → 2.914 because it learned against a field pointing the wrong way.
  All pre-fix trained models are stale; `outputs/campaign/` kept for provenance.
- Retrained PPO/SAC/TD3 at 50k and 250k, three seeds each (18 runs, 14 min wall), scored against the
  MPC and heuristic on the campaign probe suite. MPC still wins on mission quality; the network wins
  65× on cost per decision. → `Reports/2026-08-20_detailed_production_bakeoff.md`.
- Presentation pack (6 figures, 2 system diagrams, 2 animations) in `outputs/presentation/`.
- V1.7 untouched (verified by mtime scan).

**Update 2026-08-20 late (agent: claude).** grok's v8/v9 triple critique -> fixes -> full campaign.

- All nine blockers closed (B1-B9), 149 tests green, burn-in gate PASS
  (`Reports/2026-08-20_detailed_v8_fixes_after_grok.md`).
- Custom coils sized for 300 km worst case (1.473/1.473/0.442 A*m^2 -> 38.2/38.2/11.5 uN*m,
  249 mW full 3-axis, 161 g); 1 cm cp-cm offset now IN the v8+ geometry so the sim disturbance
  matches the sizing. Mass 0.625 kg confirmed.
- Campaign: 96 runs + Duo + baselines re-scored on the v8 plant.
  Best model: v9b_4x32_ppo_300k gap 0.463 (positive delegation benefit).
  v8a/v8b ablation: delegation saves actuation (P to 0.56) but does not yet pay on the gap
  (w_accuracy too weak vs w_power — recommended sweep recorded, not run).
  `Reports/2026-08-20_detailed_v8_v9_campaign_results.md`.
- Deliverables: 9 figures (plots_v8, 3 conference sets), 3 system diagrams + DIAGRAMS.md,
  auto-generated SC_v8_v9_report.md. No animations — user selects from the plots first.

**Update 2026-08-21 (agent: claude).** v9c/d -> v10 -> Monte Carlo -> v11.

- v9c (6 past) best MEDIAN 1.20; v9d (6past/6future) best RUN 0.387 -> v10 takes the v9d layout.
- MSIS returned finite garbage (T ~ 1e26 K) at the widened F10.7=5 tail and detonated the
  integrator at step 12 — output now plausibility-bounded. No completed runs affected.
- v10 (widened weather 5-250/2-200, random storms, 120-orbit episodes, tuned reward over
  3 burn-in rounds x 3 seeds): best v10_4x18_ppo_1000k gap 0.096, decay 8.78 km/d — beats
  the MPC's 9.28. Seed replication 0.096/0.82/3.10 -> cell median 0.82. Both numbers stated.
- Monte Carlo 512 orbits x 5 policies, 500->295 km: MPC owns lifetime (17.9 d median) but the
  v10 artifact out-holds it ~18 % at the 300 km deep end.
- v11 (membrane damage: hole/tear/multi mid-episode, 160-orbit episodes, adapt reward tuned
  in round a0 — 4.0/4.0 cut 11 brownouts to 0): best v11_4x40_ppo_4M gap 0.643, B 0.745
  (program record). Honest negative: multi-hole decay NOT better than v10's post-strike.
- v9Duo = its main advisor exactly (arbiter never switched). U575 inference budget:
  advisors 0.14-0.76 ms, Duo 5.5 ms, MPC 8.4 ms.
- 13 figures, strike-on-camera animations, report + this file's Reports/ entries.
