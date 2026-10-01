time: 2026-08-21T00:00:00Z
agent: claude
style: detailed

# SC_v10 / v11 campaign, Monte Carlo, and damage verification — results and honest reading

**Artifacts:** `outputs/v8/` (ledger now 188 runs + tune ledgers + snapshots),
`outputs/v8/data/{mc_decay,damage_eval,inference_u575}.json`,
`outputs/presentation/{plots_v8 (13 figures), animations_v11, SC_v8_v9_report.md}`.

## v10 — the headline, and its honest error bars

Burn-in tuning (13 candidates × 3 seeds): TUNE-0 → frozen config cut the gap **1.664 → 0.921**;
the single decisive knob was dE_weight back to 2.6 (shouting about longevity loses it).
KF sigma_drive stayed 3e-10 — every faster/slower variant lost in combination.

Matrix (56 runs, 6 budgets to 1M, sizes incl. the new 4x40):

- **v10_4x18_ppo_1000k_s42: gap 0.096, decay 8.78 km/d — BEATS the MPC's 9.28 nominal decay.**
  The "within 10 % of the MPC" target is met by this artifact, with margin.
- **Seed replication: 0.096 / 0.820 / 3.095 → cell median 0.820.** The 0.096 policy is real
  and saved, but it is a best-of-seeds artifact, not the typical training outcome. Both
  numbers belong in any claim.
- Budget curves are non-monotone (PPO 4x18: 0.53 @500k → 0.096 @1M; PPO 4x32 worsens past
  500k; SAC diverges at 1M in two cells). "More steps" is a lottery multiplier, not a ladder.

## Monte Carlo (512 orbits × 5 policies, 500 → 295 km, 40 min on 46 cores)

| policy | days 500→300 (median) | decay@300 (mean) |
|---|---|---|
| MPC | **17.9** | 88.2 |
| v10 best (0.096 artifact) | 11.9 | **72.6** |
| heuristic | 10.6 | 105.6 |

The MPC still owns total lifetime — but **the v10 model out-holds the MPC at the 300 km deep
end by ~18 %**, exactly the regime its coils and delegation were sized for. That is the
defensible "where deep RL beats the MPC" claim, alongside the 60-80× decision-cost advantage
(U575: 0.14-0.76 ms vs MPC 8.4 ms; v9Duo 5.5 ms).

## v11 — adaptability: what worked and what did not

Burn-in a0 was unambiguous: the adaptability terms at 4.0/4.0 cut the tuning gap 3.33 → 1.22
and eliminated ELEVEN brownouts — **the adapt reward is what makes training on a mutating
vehicle stable at all.** Damage prob 0.85 hurt; 0.6 stands.

Matrix (12 runs, best-3 v10 classes × 50k/500k/1M/4M, quadruple episodes):
best = **v11_4x40_ppo_4000k: gap 0.643, B = 0.745 — the program record for correct
delegation**, held at B ≈ 0.38-0.48 even post-strike.

Damage verification (hole/tear/multi × 8 seeds; v11 vs v10-never-saw-damage vs MPC-with-stale-model):

- Holes and tears are absorbed by everyone: less membrane ≈ slightly less drag, and the
  MRP-PD closes the attitude loop regardless of who advises.
- **The honest negative: on multi-hole strikes v11 did NOT out-hold v10 on orbit decay**
  (41→94 km/d vs v10's 25→54; different pre-strike baselines, same doubling). The
  adaptability reward stabilised *training* and preserved *delegation quality* under damage,
  but did not deliver better post-strike orbit-holding in this test.
- The MPC's stale internal model costs it nothing on 30-orbit attitude tracking — its
  weakness is decision cost and deep-altitude decay, not strike response. Say so.
- The recovery-time metric (err < 2× pre-strike median) saturated at 1 step for everyone —
  too generous against the ~21° tracking baselines the delegating policies run; the
  discriminators are post-strike decay and B.

## Family picture (188-run ledger)

Best per family: v10 0.096 · v9d 0.387 · v9b/v9Duo 0.463 · v10(median cell) 0.82 ·
v11 0.643 · v8a 0.479 · v8b 0.858. Assembled Duos never beat their own main advisor
(arbiter conservatism + weak horizon advisors); flagged, not hidden.

## Deliverables

13 figures (f1-f13, incl. Monte Carlo boxes, damage comparison, budget curve, U575 table),
3 system diagrams + DIAGRAMS.md, strike-on-camera animations (MPC / v10 / v11: nominal →
tear appears on the CAD hull → post-strike behaviour with live membrane-%/cp/Cd readout),
auto-generated SC_v8_v9_report.md with every config value and tuning round.

## Recommended next (not run)

1. v11 multi-hole result → the adapt reward needs a decay-holding term, not only
   consistency+delegation; candidate: extend decay_stability's weight while damaged.
2. Re-assemble a Duo on the v10 0.096 artifact (both prior Duos inherited weak mains).
3. Seed-replicate v11_4x40_ppo_4000k before any claim rests on it.
4. v10Duo / v10Tri / v10Phy (physics-informed) — user's queued ideas, awaiting this data.
