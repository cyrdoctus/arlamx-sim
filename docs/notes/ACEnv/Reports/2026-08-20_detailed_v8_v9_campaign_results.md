time: 2026-08-20T00:00:00Z
agent: claude
style: detailed

# SC_v8 / v8Duo / v9 campaign — results and honest reading

**Precedes:** `2026-08-20_detailed_v8_fixes_after_grok.md` (all grok blockers closed, burn-in gate PASS).
**Artifacts:** `outputs/v8/` (ledger.csv 96 runs + snapshots), `outputs/presentation/{plots_v8, SC_v8_v9_report.md, diagrams}`.
**Everything below is on the v8 plant** (1 cm cp offset, custom 38 µN·m coils, quadratic coil power,
sensor-fed torque KF, brownout live). v7 numbers are deliberately absent — different vehicle.

## The campaign

96 training runs, one seed each (seed replication deferred — see caveats):
- v8a / v8b: 6 widths (4x14…4x32) × PPO/SAC/TD3 × 50k/300k = 72
- v9a / v9b: top-2 widths per algo from the v8b ledger × 2 budgets = 24
- v8Duo: horizon advisor (PPO 4x20 class) trained 50k on the wrapper env; main = v8b_4x18_ppo_300k
- Baselines re-scored on this plant: MPC decay 9.28 km/d (86.6 % band, 23.2 min/d), heuristic 17.13 (5.0 %, 20.6)

## Headline results

| family | best run | gap | decay | band % | B | P |
|---|---|---|---|---|---|---|
| **v9b** | **ppo 4x32 300k** | **0.463** | 13.92 | 25.8 | **+0.039** | 0.245 |
| v8a | sac 4x24 50k | 0.479 | 10.73 | 18.6 | — | — |
| v9a | sac 4x24 50k | 0.726 | 12.53 | 22.7 | −0.033 | 0.099 |
| v8b | ppo 4x18 50k | 0.858 | 22.98 | 6.6 | −0.129 | 0.408 |
| Duo | assembled | 0.931 | 20.19 | 5.0 | — | — |

1. **v9b is the best model of the campaign** (gap 0.463), and the only family whose best run has
   POSITIVE delegation benefit. Past+future temporal context beats future-only (v9b median 1.43 vs
   v9a 1.81) and beats its own v8b parent (1.51). The v9 design earns its complexity.
2. **The v8a/v8b ablation answers honestly: delegation as configured does NOT pay on the mission gap.**
   v8a (gates pinned) medians 1.30 vs v8b 1.51. v8b policies save real actuation (P up to 0.56;
   median 0.28 at 300k) but delegate in the wrong direction on median (B ≤ 0 in 3 of 4 cells).
   The mechanism *works* — the policies have not learned to use it selectively. The one lever the
   data points at: w_accuracy (2.0) is too weak against w_power (6.0) — policies farm the saving and
   eat the signed accuracy penalty. Recommended next experiment: w_accuracy 4–6 with w_power unchanged.
   NOT changed mid-campaign, to keep v9-vs-v8b comparable.
3. **Every learned run still sits above the MPC** (best 0.46 vs 0.0) and the best four now beat the
   heuristic (0.60). The compute story is unchanged: ~0.1 ms/decision vs the MPC's ~6 ms.
4. **Size structure is real but learner-dependent** (fig v8_f2): PPO improves monotonically to 4x24
   then collapses at 4x32 in v8a; SAC's best sits at 4x18–4x24; several cells diverge outright
   (worst 9.6). One seed per cell — treat cell-level ordering as indicative, not proven.
5. **Duo underperformed its own main advisor on gap** (0.931 vs 0.903) though it trimmed decay
   (20.2 vs 20.5). The arbiter chose "main" overwhelmingly; the horizon advisor rarely offered a
   candidate that beat the 6 h counterfactual. With the main advisor itself mediocre (v8b arm), the
   Duo inherited its ceiling. Re-assembling Duo on the v9b winner is the obvious follow-up.
6. **The torque story is the strongest physical result** (fig v8_f4): onboard estimate vs plant truth
   r = 0.988 with datasheet sensor noise, and the along-B authority shortfall routinely 20–100 % —
   the physical argument for why environmental torque is worth delegating to at all.

## Caveats that must ride with any use of these numbers

- **One seed per cell.** The v7 study measured seed spread larger than many cell differences.
  Family-level medians (n = 12–36) are meaningful; single-cell comparisons are not.
- v8b's B medians are negative: "the policy delegates" ≠ "the policy delegates well".
- Duo's horizon advisor trained only 50k on a wrapper whose reward is the propagator's opinion.
- Plant simplifications still open: per-axis (not dipole-space) torque saturation, MSIS 300 s hold,
  truth-state observation outside the KF chain, no ONNX export.

## Recommended next steps (not run)

1. Seed replication (×3) of the four family winners + the w_accuracy sweep above.
2. Re-assemble Duo with v9b_4x32_ppo_300k as main.
3. Animations: user selects from the plot set first (per instruction, none rendered).
