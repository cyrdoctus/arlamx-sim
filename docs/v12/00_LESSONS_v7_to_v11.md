# SC_v12 — Step 0: what worked and what did not (v7 → v11), and what the MPC actually is

**Purpose.** First document of the v12 design series. Nothing here is new work —
it is the evidence from 188 ledger runs, 52 tuning rows, two campaign reports,
the grok critique and the configs, sorted into *hit / neutral / wasteful /
harmful* so v12 starts from what was measured rather than from what was hoped.
Every number is traceable: `outputs/v8/ledger.csv`, `outputs/v8/{v10,v11}_tune.csv`,
`outputs/v8/data/ref.json`, `python/configs/reward_v11.yaml`,
`ACEnv/Reports/2026-08-2{0,1}_detailed_*.md`.

Read order: §1 the target (MPC) → §2 the scoreboard → §3 reward verdicts →
§4 non-reward lessons → §5 what this implies for v12 → §6 decisions for you.

---

## 1. The target: what the MPC is, how it scored, and what tuning it got

### 1.1 What it is (`python/arlamx_v2/advisors/mpc.py`, report `2026-08-18_detailed_mpc_heuristic.md`)

Sampling receding-horizon MPC on a cheap surrogate, run every 300 s:

| item | value |
|---|---|
| candidates per step | `hold`, `sun` (analytic ephemeris), `gs` (if station ≥ 10° el.), `min_drag` (Z ∥ h), `nadir` (Z ∥ −r), + **8 random** body-axis perturbations, angle U[2°, 34°] |
| horizon | 6 advisor steps = **30 min**, 60 s internal two-body RK4 |
| surrogate | two-body orbit, first-order SoC (`P_gen = 0.78·0.85·|ẑ·ŝ|`, loads 7 mW base + 205 mW GPS sunlit + 400 mW TX), Sentman `Cd` on the real panels, GS cosine |
| objective | `Σ_k [ B(SoC_k) + 1[pass,lit]·(T(θ_gs) + w_align·cosθ_gs) − w_cd·Cd_k ] − w_slew·Δcmd/40°` |
| weights | power_band **1.0**, power_low_pen **4.0**, gs **2.0**, gs_pen **0.5**, gs_align **1.0**, cd **0.5**, slew **0.2** |
| band | +1 in SoC 0.4–0.6, `exp(−6(s−0.6))` above, `−4(0.4−s)/0.4` below |
| GS tiers | full / ½ / ¼ / −0.5 at 2° / 5° / 10° / beyond — **identical tiers to the RL `gs_tiered_pointing`** |
| output | one `q_BN`; the plant's MRP-PD tracks it at 2 s, **no delegation, full torque authority** |
| cost | 8.4 ms per decision on STM32U575 (analytic), vs 0.14–0.76 ms for an RL advisor |

What it is allowed to know: truth `r, v`, sun from ephemeris (no sun sensor),
`rho/T/m_bar` from MSIS at the current point, SoC, `gs_dir_N`. It never sees
sensor noise — the same privilege every RL policy currently has (§4.1).

### 1.2 Tuning history: **none**

Grep of every report and the code: the MPC runs `SamplingMpcPolicy(seed=0)` with
`DEFAULT_W` in all of `bench_v8`, `campaign`, `gradient70`, `eval_v4`. The only
variants ever instantiated are `mpc_no_sun_eph` (drops the ephemeris candidate)
and a cheaper `horizon 4 / n_random 4` for the OOD sweep. No weight sweep, no
horizon sweep, no candidate-set sweep was ever run. **The reference you are
chasing is an untuned default.** (Two consequences: it is beatable, and a tuned
MPC would be the honest comparator for the thesis — see §6.)

### 1.3 How it scores on the v8+ plant (`ref.json`, probes nominal/spike/offnominal × seeds 1001,1002)

| metric | MPC agg | nominal | spike | offnominal |
|---|---|---|---|---|
| decay km/d | **9.28** | 9.28 | 17.60 | −0.42 |
| band % (SoC in 0.4–0.6) | **86.6** | 84.4 | 92.5 | 82.9 |
| brownouts | 0 | 0 | 0 | 0 |
| downlink min/d | **23.2** | 35.1 | 19.5 | 15.1 |
| MTQ energy J | **2.97** | 0.64 | 2.29 | 0.04 |
| observer r (kf_r) | 0.80 | 0.96 | 0.99 | 0.44 |

Note the MTQ energy: the MPC spends ~3 J because its candidate set is mostly
"hold" and physically meaningful fixed attitudes. It does not slew for fun.

### 1.3b The heuristic planner (`advisors/heuristic.py`, `HeuristicPowerPolicy`)

The second scripted baseline. It is the "no model at all" reference: it closes
only on SoC and `power_gen_norm` (no sun vector, no ephemeris, no surrogate).

**Parts**

| part | what it does | defaults (the bench uses these, `bench_v8.py:227`) |
|---|---|---|
| mode ladder | picks a *mode* from SoC alone, thresholds descending `t0…t4` | (0.60, 0.50, 0.40, 0.30, 0.20) |
| ≥ t0 "rich" | +Z at a visible station, else `rich_mode` attitude | `rich_mode = min_drag`, `allow_gs = True` |
| t1…t0 | station if visible, else last-good-sun quaternion, else search | |
| < t1 | power search, no station | |
| search step scaling | sweep/climb steps ×1 (≥t2), ×1.5 (t3…t2), ×2 (<t3); below t4 hold on *any* sensed power | `sweep_step 15°`, `climb_step 6°` |
| power search | state machine `sweep → climb → hold`: rotate about body Y then X by the sweep step until `gen ≥ sense_floor`, then coordinate descent on ±X/±Y keeping a step if gen rose > 1e−3, halving after a failed cycle, hold at `gen ≥ gen_target` or step < 1° | `sense_floor 0.05`, `gen_target 0.70` |
| eclipse guard | fruitless 360° sweep or gen → 0 ⇒ freeze attitude for N steps | `eclipse_hold_steps 6` (30 min) |
| memory | stores a "last-good sun" quaternion whenever gen ≥ 0.5 | |
| frames it may build | min-drag (Z ∥ h), nadir (Z ∥ −r), GS (from r, v, station) | same privilege as MPC |

**Tuning history.** A 13-member bank was swept once on the 1 % SolarCat mesh
(2-day campaign, report `2026-08-18_detailed_mpc_heuristic.md`, data
`outputs/analysis/advisors/summary.csv`): thresholds 0.45…0.99 / 0.15…0.35,
sense floor 0.02–0.20, gen target 0.15–0.85, step sizes 8°/3° to 25°/10°,
eclipse hold 6–24 steps, rich mode min-drag vs nadir. Lesson: the three that
died (`conserve`, `any_power_hold`, `search_f02_t30`, 450–540 brownouts each)
all *latched a dim false sun* and rode it into umbra. Floor 0.05–0.20 with a
real climb target survives. The bench uses the `mission_v17` defaults — the
sweep was never re-run on the v8+ plant.

### 1.3c Every performance record of the two baselines

**(a) 2-day SolarCat campaign, 1 % mesh, 500 km, i = 23°, F10.7 150 / Ap 4 (v7-era plant)**

| policy | min SoC | brownouts | gen (lit) | GS lock | survived |
|---|---|---|---|---|---|
| mpc | 0.361 | 0 | 0.336 | 0.021 | yes |
| mpc_no_sun_eph | 0.370 | 0 | 0.335 | 0.010 | yes |
| heuristic mission_v17 | 0.322 | 0 | 0.472 | 0.049 | yes |
| heuristic ground_first (nadir rich) | 0.528 | 0 | 0.802 | 0.052 | yes |
| fixed nadir | 0.528 | 0 | 0.811 | 0.028 | yes |
| fixed min_drag | 0.425 | 0 | 0.449 | 0.035 | yes |
| fixed off_nadir_60 | 0.504 | 0 | 0.713 | 0.016 | yes |

Reading: on this bus the **two-sided ±Z panel makes nadir a power-positive
comms attitude** (0.81 of peak sunlit). The MPC's generation is *lowest* of the
survivors (0.34) because it trades power for Cd once the band is satisfied —
it does not maximise power, it holds the band. GS lock ≈ 5 % ≈ every visible
pass; stations are only up for minutes.

**(b) v7 production bake-off (fixed v7 plant, catalogue coils 18/18/5.7 µN·m, 3 probes × 2 seeds × 20 orbits)**

| policy | gap | decay nom | decay storm | band % | brownouts | downlink min/d | actuator J | ms/decision |
|---|---|---|---|---|---|---|---|---|
| MPC | 0.000 | 9.20 | 17.8 | 84.0 | 0 | 23.2 | 98 | 6.16 |
| Heuristic | 0.603 | 16.72 | 34.3 | 4.9 | 0 | 15.4 | 250 | 0.05 |
| best RL (PPO 50k) | 1.097 | 27.0 | 71.9 | 31.0 | 0 | 12.0 | 332 | 0.09 |
| TD3 50k (thriftiest) | 1.685 | 32.1 | 93.3 | 2.8 | 0 | 7.6 | 58 | 0.09 |

(Actuator J here is under the old *linear* power model and v7 coils — not
comparable to the v8+ joules below.)

**(c) v8+ plant probes (`ref.json`; custom coils, quadratic power, 1 cm cp offset) — the reference every v8–v11 gap is measured against**

Probes: `nominal` 400 km / i 23° / F10.7 150 / Ap 4, 120 orbits; `spike` same
with a +100 F10.7 / +36 Ap storm at 25 % and removal at 60 %; `offnominal`
550 km / i 51.6° / RAAN 40°. Seeds 1001, 1002. Initial ω = (1, 1, 0.5) °/s.

| metric | MPC agg | nom | spike | offnom | Heur agg | nom | spike | offnom |
|---|---|---|---|---|---|---|---|---|
| decay km/d | **9.28** | 9.28 | 17.60 | −0.42 | 17.13 | 17.13 | 34.33 | 0.15 |
| band % | **86.6** | 84.4 | 92.5 | 82.9 | 5.0 | 4.9 | 4.9 | 5.4 |
| brownouts | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| downlink min/d | **23.2** | 35.1 | 19.5 | 15.1 | 20.6 | 25.4 | 15.6 | 20.7 |
| MTQ energy J | **2.97** | 0.64 | 2.29 | 0.04 | 32.0 | 5.7 | 26.3 | 0.05 |
| observer r | 0.80 | 0.96 | 0.99 | 0.44 | 0.79 | 0.99 | 1.00 | 0.37 |

Heuristic gap on this plant: **0.60** (band is its whole deficit — it is within
10 % of the MPC on downlink and ~2× on decay). Note the heuristic's 5 % band
is the *same* 3–5 % the RL families sit at: a policy that searches for power
instead of planning it ends up riding a full battery, which the band term
penalises. That is the shared failure mode.

**(d) Monte Carlo decay, 512 random orbits each, 500 → 295 km (`mc_decay.json`, v10 plant)**

| policy | reached 300 km | decay@500 mean | decay@400 | decay@300 | days 500→300 (mean) |
|---|---|---|---|---|---|
| MPC | 252 / 512 | 5.40 | 14.3 | **88.2** | **17.97** |
| heuristic | 388 | 9.00 | 21.3 | 105.6 | 12.72 |
| v10_4x18_ppo_1000k (0.096) | 368 | 8.83 | 19.3 | **72.6** | 13.51 |
| v10_4x18_ppo_300k | 418 | 10.17 | 23.8 | 110.2 | 11.48 |
| v10_4x40_td3_1000k | 401 | 9.91 | 24.5 | 109.1 | 11.91 |

Reading: the MPC's advantage is at the *top* (5.4 vs 8.8 km/d at 500 km — it
spends the high-altitude phase closer to min-drag); the RL artifact's
advantage is at the *bottom* (72.6 vs 88.2 at 300 km). Only half the MPC
orbits even reach 300 km inside the run window.

**(e) OOD gradient sweep (v4-era, `2026-08-19_detailed_ood_gradients.md`)**

MPC is the only advisor that is power-stable everywhere: return rises with
altitude (+25 at 400 km, +41 at 500, +39 at 650), flat across inclination
(24–33), min SoC 0.36–0.50, almost insensitive to an Ap spike (density lag ≫
its 30-min horizon), geodetic |Δh| −17 km quiet → −23 km superstorm. The
heuristic and most RL models were "deeply negative" on the same sweep.

**(f) Decision cost (STM32U575 analytic; `inference_u575.json`)**: MPC 8.4 ms
(6.16 ms on the v7 bench incl. state assembly), heuristic 0.05 ms, RL advisors
0.14–0.76 ms, Duo 5.5 ms.

### 1.4 Where the MPC is weak (measured, not asserted)

1. **Deep altitude.** Monte Carlo 512 orbits 500→295 km: MPC decay at the
   300 km crossing 88.2 km/d vs the v10 artifact 72.6 (−18 %). Its 30-min
   horizon and `w_cd = 0.5` do not value drag enough when drag is the whole game.
2. **Decision cost**: 8.4 ms vs <1 ms (60–80×).
3. **Stale model under damage** costs it nothing on 30-orbit tracking — *not* a
   weakness; say so in the thesis.
4. **Downlink only counted when lit** (`tx_align_min = 0.7`, `lit` required in
   its surrogate, and `gs_tiered_pointing` is `daylight_only=True` in the RL
   reward as well). If the bench's `dl` counter also requires sunlight, both sides
   are leaving eclipse passes on the table — worth checking before v12 (§6).

---

## 2. The scoreboard: every RL family against the MPC, per metric

Family **medians** from the ledger (n runs), MPC reference in the last row.
Gap index = mean over (decay, band, brownouts, downlink, decay_spike) of the
signed normalised distance to the MPC; 0 = MPC, negative = beats it.

| family | n | gap med | gap best | decay km/d | band % | downlink min/d | MTQ J | deleg B | deleg P |
|---|---|---|---|---|---|---|---|---|---|
| v8a (no delegation) | 36 | 1.30 | 0.48 | 26.2 | 3.8 | 11.6 | 97 | — | — |
| v8b (delegation) | 36 | 1.51 | 0.86 | 30.2 | 3.5 | 10.3 | 177 | −0.12 | 0.14 |
| v9a (future obs) | 12 | 1.81 | 0.73 | 30.0 | 4.1 | 12.9 | 173 | −0.08 | 0.20 |
| v9b (3+3) | 12 | 1.43 | 0.46 | 31.3 | 4.0 | 11.6 | 132 | −0.18 | 0.02 |
| v9c (6 past) | 12 | **1.20** | 0.52 | 23.8 | 2.9 | 9.6 | 86 | −0.11 | 0.05 |
| v9d (6+6) | 12 | 1.64 | 0.39 | 26.7 | 3.5 | 10.2 | 105 | −0.22 | 0.02 |
| v10 | 56 | 1.49 | **0.10** | 28.5 | 3.7 | 9.7 | 107 | −0.06 | 0.11 |
| v11 | 12 | 1.12 | 0.38 | 28.4 | 3.8 | 13.8 | 140 | 0.00 | 0.24 |
| **MPC** | — | 0 | 0 | **9.3** | **86.6** | **23.2** | **3.0** | 0 | 0 |

Three facts that were under-emphasised in the campaign reports:

1. **The typical RL policy loses on *every* mission metric, and band is the
   catastrophe**: 3–4 % of the time in the 0.4–0.6 band vs the MPC's 87 %.
   Decay is 3× the MPC. Downlink is half. None of the five families moved the
   medians materially — v7→v11 reshaped the *tail*, not the distribution.
2. **RL policies burn 30–60× the actuation energy of the MPC.** They slew
   constantly. Brownouts stay at 0 only because `brownout_weight = 100`, not
   because the policies manage power.
3. Delegation benefit `B` is **≤ 0 on median in every family that has it**.

### 2.1 The three headline artifacts, decomposed

| | v10_4x18_ppo_1000k_s42 (gap 0.096) | v11_4x40_ppo_4000k_s42 ("B record") | v9d_4x32_sac_300k_s42 |
|---|---|---|---|
| decay km/d | **8.78** (−0.054) | 20.07 (+1.16) | 13.24 (+0.43) |
| band % | **92.4** (−0.067) | 5.9 (+0.93) | 35.9 (+0.59) |
| downlink min/d | **7.67 (+0.670)** | **25.6 (−0.10)** | 15.3 (+0.34) |
| decay spike | 16.4 (−0.067) | 39.2 (+1.22) | 27.9 (+0.58) |
| MTQ J | **1.45** | 55.5 | 7.8 |
| B / P | 0.49 / 0.42 | 0.75 / 0.47 | −0.23 / 0.18 |

So: the 0.096 artifact is **an MPC-class longevity+power policy that gave up
two thirds of the downlink**; the gap index let it, because the downlink term
is one of five. The v11 B-record is the only model that beats the MPC on
downlink — and it does so with 20 km/d decay, 6 % band and 55 J of actuation.
**No model has ever been MPC-class on all four metrics at once.** That is the
v12 target, stated precisely.

---

## 3. Reward terms: verdict table

Legend — **HIT**: measured positive effect on mission metrics. **NEUTRAL**: never
isolated / no evidence either way. **WASTEFUL**: costs tuning effort or
capacity, no measured gain. **HARMFUL**: measured to make things worse.

### 3.1 Inherited v7 core (`compose_sc_v7`)

| term | weight (v11) | verdict | evidence |
|---|---|---|---|
| `dE_vs_baseline` | 2.6 | **HIT — the only decisive knob** | r0: 2.6 → gap 0.92, 3.2 → 1.66, 4.0 → 1.36 (3 seeds × 40k). Monotone in the *wrong* direction above 2.6: "shouting about longevity trades band for drag and loses overall". Same lesson already in v7. |
| `power_band` (SoC 0.4–0.6) | 1.0 | **NEUTRAL as designed, target missed** | Median band 3–4 % in every family. `band_weight 2.0` (r1 kf9_band) → 12 brownouts on one seed, no band gain on the others. The term is too weak to matter at 1.0 and destabilises at 2.0 — i.e. shaped wrong, not weighted wrong (see §3.4). |
| `gs_tiered_pointing` | 2.4 | **NEUTRAL, target missed** | 2.0→2.4 (TUNE-0) then 2.8 (r2 gs28: gaps 1.15–1.50) — no downlink improvement; median downlink 10–14 min/d vs MPC 23. Tier geometry is *identical* to the MPC's, so the difference is not the term, it is that the MPC has a "gs" candidate in hand and RL must discover the attitude from a quaternion. Also `daylight_only=True`. |
| `action_smoothness` | 0.10 | **WASTEFUL at higher values** | 0.02→0.10 (TUNE-0) unisolated; 0.2 (+thrift 2.0, r2 soft) collapsed the band and produced 15 brownouts on one seed. |
| `torque_thrift` (linear duty) | 1.5 | **WASTEFUL / inconsistent** | Linear in duty although the plant's power is quadratic in dipole (`14_reward_v8 §6`); 0.9→1.5 unisolated; 2.0 hurt (r2 soft). Median MTQ energy still 30–60× the MPC — the term does not buy stillness. The 0.096 artifact's 1.45 J came from the dE term, not from thrift. |
| `brownout_recovery` | 100 | **HIT as a constraint** | Brownouts = 0 in essentially every converged run; removing it is not on the table. But it is a cliff, not shaping: it keeps SoC > 0, it does not keep SoC in band. |
| `altitude_cliff` (250 km) | hard | HIT as a constraint | Never contested. |
| `action_feasibility` (40° cap) | 1.0 | NEUTRAL | Never isolated; the plant slerps to 40° anyway. |
| `omega_penalty`, `momentum_penalty` | 0.002 / 0.001 | NEUTRAL | Three orders of magnitude below the other terms — effectively dead weight in the composite. |
| `shade_conservation` | (v7 default) | NEUTRAL | Never isolated. |

### 3.2 v8 stability terms

| term | weight | verdict | evidence |
|---|---|---|---|
| `altitude_margin` (soft barrier 320 km) | 4.0 | NEUTRAL | Never isolated. Nominal probes start well above 320 km and 120-orbit episodes at 9–30 km/d rarely reach it; it is live mainly on the deep-altitude MC. Keep for v12 but treat as a constraint, not shaping. |
| `decay_stability` (one-sided, scale 100 J/kg) | 0.6 | NEUTRAL | Never isolated. Design is sound (one-sided, measured scale); value unproven. |
| `power_drop` (v10) | 2.0 | **HARMFUL at 4.0, unproven at 2.0** | r2 drop4: 9 brownouts on seed 42, band 3 %. It double-penalises what `power_band` should already shape. |

### 3.3 Delegation (v8b → v11) — the program's largest investment

| term | weight | verdict | evidence |
|---|---|---|---|
| `deleg_power` = w·P·clip(B,0,1) | 6.0 | **WASTEFUL** | Structurally sparse (B>0 rare → product ≈ 0). v8a (gates pinned) median 1.30 **beats** v8b 1.51 — delegation as rewarded *costs* mission gap. |
| `deleg_accuracy` = w·B (signed) | 2.0 → 6.0 | **WASTEFUL → HARMFUL** | Raising to 6.0 (TUNE-0, the "v8 campaign lesson") did not lift median B above 0 in v10 (−0.06). The policies that maximise B (v11 4x40: 0.75) are the ones with the worst mission metrics (§2.1). The term pays for *agreeing with the environment*, which correlates with *not fighting drag* — i.e. with decaying. |
| `kappa` boost on mission terms | 0.5 → 0.7 | **DEAD** | Multiplies positive terms by `1+κ·S·B⁺`; with B⁺≈0 on median it is identity. Never isolated. |
| delegation **mechanism** (split → gate floor 0.3) | — | physically real, reward-wise unproven | The mechanism works (P up to 0.56, gate floor enforced) — what failed is paying for it. The 0.096 artifact has B 0.49 / P 0.42, so a good policy *can* delegate well; it was not the delegation reward that made it good. |
| KF torque estimate feeding B | `σ_drive 3e-10` | **HIT as an observer, NOT a reward lever** | r = 0.99 vs truth with datasheet noise. Every σ_drive variant (1e-9, 1e-10, "kf9" combos) lost in combination — tuning the filter through the reward was noise. |

**Net**: ~6 tuned numbers, two campaigns, a B-metric, a Duo arbiter — and the
only family without delegation (v8a) has the better median. For v12 the
delegation *mechanism* (gate floor, torque observer) is worth keeping as an
actuator-side feature; the delegation *reward* should not return in its current
form.

### 3.4 v9 trend adaptation and temporal observation

| item | verdict | evidence |
|---|---|---|
| trend re-weighting (eq. 11–12) | NEUTRAL | Always co-varied with the observation change; never isolated. Weights 0.8/1.0 are [ASSUME]. |
| v9 temporal blocks (past/future Δalt, Δsma, SoC, offset) | **HIT on best-case, costs variance** | v9b median 1.43 < v8b 1.51 (parent); v9d best 0.387 but median 1.64; v9c best *median* 1.20. Richer obs → wider spread. v10/v11 inherit v9d. The *future* block (FP32 onboard propagation, flight-computable) is the defensible part; the past block is cheap. |

### 3.5 v11 adaptability (you have scrapped it; recorded for completeness)

| term | verdict | evidence |
|---|---|---|
| `adapt_consistency` 4.0 | HIT for *training stability only* | a0: 11 brownouts → 0, gap 3.33 → 1.22. |
| `adapt_deleg_hold` 4.0 | WASTEFUL for the mission | B-record 0.745 with decay 20 km/d; multi-hole post-strike decay 41→94 km/d, worse than v10's 25→54. |
| damage prob 0.6 | — | 0.85 hurt (1.50). |
| 160-orbit episodes | NEUTRAL | Needed for the strike; costs 4× wall time per episode. |

One transferable lesson from v11: **a term that penalises deviation from the
policy's own recent behaviour stabilised training** (0 brownouts). That is a
trust-region-like effect on the *attitude*, and it is the same thing the MPC's
`w_slew` and "hold" candidate do. v12 can get this without the damage machinery.

---

## 4. Non-reward lessons

### 4.1 Inputs — the policy sees truth

`env._obs` (`env.py:289–364`) builds the 35-dim core from **plant truth**: `r, v,
ω, σ` (→ `v̂_B`, `B̂_B`, |B|), true anomaly, mass, log ρ, SoC, gen, GS
direction+visibility. Sensors (`sensors_solarcat.yaml`, Allan-validated) feed
**only** the torque KF, whose output enters the 9-dim IPC block as
`tau_dist_est` and the reward as `tau_env_est`. The grok critique §1.4 ("sensors
are a sidecar, not the observation") is still open. So:

- no policy to date has been trained on noisy attitude/rate/field inputs;
- the MPC has the same privilege, so the comparison is fair *between advisors*
  but neither is flight-representative;
- observation normalisation is ad hoc (`r/7e6`, `v/8e3`, raw `ω` in rad/s,
  `|B|/50 µT`, `log10 ρ` unscaled ≈ −12…−10) and relies on SB3 `VecNormalize`.

### 4.2 Action space — raw quaternion vs the MPC's menu

RL: 7 outputs, `tanh` → `[q(4) → normalised, s(3) → gates]`, slerp-capped at 40°
per 300 s step. MPC: picks from 5 *named* attitudes + random perturbations.
Everything the MPC does well (band 87 %, downlink 23 min/d, 3 J actuation) comes
from having `sun`, `gs`, `min_drag`, `nadir`, `hold` available as single
choices. The RL policy has to discover these four attitudes inside SO(3) from a
composite reward, every episode, through a 300 s-step actuator. That is the
single biggest structural reason for the scoreboard in §2, and it is a design
choice, not a capacity problem (4x18 beat 4x40).

### 4.3 Controller

`kp 6e-5, kd 1.6e-3`, overdamped (ζ≈1.85 X/Y, 1.31 Z), 181 s period on X/Y vs
300 s advisor step — chosen so a 40° slew spreads over the step rather than
burning in the first 50 s. 3×3 sweep grid exists in `gains_mrp.yaml` and **was
never run**. Per-axis torque saturation only; **dipole-space saturation
(`m ⊥ B`) is not in the plant** — attitude authority is optimistic by 20–100 %
whenever B is unfavourable (`14_reward_v8 §6.1`). Power is physical
(quadratic, custom coils 38/38/11.5 µN·m, 249 mW full 3-axis).

### 4.4 Learner, size, budget, seeds

- **PPO is the only reliable learner** on this env: v10 PPO median 1.08 (2/20
  diverged), SAC 1.60 (6/18 diverged), TD3 1.65 (1/18). SAC diverges at 1M.
- **Small wins**: best-ever is 4x18; PPO 4x32 worsens past 500k; 4x40 only
  helped v11's 4M run. Capacity is not the bottleneck.
- **Budget curves are non-monotone**: 4x18 PPO 0.53 @500k → 0.096 @1M, but
  seed replication of that cell gives 0.096 / 0.82 / 3.10. "More steps" is a
  lottery multiplier. **Seed variance is larger than most design deltas** — any
  v12 claim needs ≥ 3 seeds per cell, and the median is the number.
- 40k-step burn-ins predicted the 300k–1M ranking reasonably for reward
  *rejection* (finding harmful terms) but not for *selection* of the winner.

### 4.5 Episode horizon

40 orbits (v8/v9) → 120 (v10) → 160 (v11). v10's jump coincided with the
0.096 artifact but was bundled with widened weather and storms, so the effect
of horizon alone is unmeasured. Reasoning stands: the longevity term has to
*see* the decay it shapes.

### 4.6 Duos

Both Duos equalled their main advisor exactly — the arbiter never switched.
Weak horizon advisors (50k on a propagator-opinion reward) and a conservative
arbiter. 5.5 ms decision cost for no gain. Do not carry into v12 as-is.

### 4.7 Process lessons worth keeping

- Burn-in gate that *prints* term activity caught two silent failures
  (`decay_stab_scale` 4 orders off; `deleg_power ≡ 0`). Keep it mandatory.
- Frozen `snapshot.json` per run, provenance tags in YAML, honest medians.
- The external critique (grok) found 8 blockers before any compute was spent.

---

## 5. What this implies for v12 (preview — each becomes its own step)

1. **Target = all four metrics at once, each at MPC level or better, with
   downlink the one to beat.** The gap index hid the downlink loss; v12
   reporting needs a per-metric table and a "worst metric" scalar next to the
   mean.
2. **Give the policy the MPC's menu.** Action space as a *structured* choice
   (sun / gs / min-drag / nadir / hold) plus a bounded residual rotation, or as
   mixing weights over those basis attitudes. This is the one change the
   evidence most strongly supports: it attacks band, downlink and actuation
   energy simultaneously and is flight-computable (the MPC already builds the
   frames onboard).
3. **Reward: keep `dE_vs_baseline` (2.6-class), `brownout`, `altitude_cliff`;
   rebuild band and downlink as *time-in-state* terms** the way the MPC scores
   them (per-step +1 in band, per-step downlink credit when locked *and* a
   pass is up), add a quadratic actuation-energy term (physical), and add a
   slew/hold term (the transferable v11 lesson). Drop `deleg_power`,
   `deleg_accuracy`, `kappa`, `power_drop`, and the trend re-weighting.
4. **Inputs through sensors.** Attitude/rate/field through
   `sensors_solarcat.yaml` noise → an attitude filter → the observation; keep
   the torque KF; keep the FP32 future block; fix the normalisation. Train
   with noise from day one.
5. **Controller**: keep MRP-PD as the inner loop (it is what the MPC uses and
   what the plant validates), run the 3×3 gain sweep once, and decide on
   dipole-space saturation in C++ (it changes authority; it is the honest
   plant).
6. **Learner**: PPO, 4x18–4x24, ≥ 3 seeds, 120-orbit episodes, 40k burn-in
   gate with printed term activity before any long run.
7. **Comparator**: a *tuned* MPC (weights, horizon, candidate count) so the
   claim survives review.

---

## 6. Decisions for you (airplane list)

1. Do you want v12 to keep the delegation **mechanism** (split → gate floor) as
   an output at all, or go full-authority like the MPC and treat
   environmental torque purely as an *input*? (My recommendation: input only
   for v12a, re-add as an ablation v12b.)
2. Structured action space: discrete-choice + residual (hybrid policy) vs
   continuous mixing weights over basis attitudes (simpler with PPO)?
3. Is downlink legitimately sunlit-only? (`tx_align_min 0.7`, `daylight_only`
   in both the MPC surrogate and the reward.) If the radio can transmit in
   eclipse from the battery, both sides are under-counting and v12 can win
   downlink by design.
4. Dipole-space saturation in the plant before v12, or after (it alters all
   prior numbers)?
5. Should the comparator MPC be tuned first (fair fight) or frozen as the
   historical reference (continuity with the thesis numbers so far)? Both can
   be reported.
6. Episode length: 120 orbits (v10) or longer, given no damage events?

Next document: `01_INPUTS_AND_FILTERING.md` — the observation vector v12 gets,
sensor by sensor, with the noise model and the filter between each sensor and
the policy.
