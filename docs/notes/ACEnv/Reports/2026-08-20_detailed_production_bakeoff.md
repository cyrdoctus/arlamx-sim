time: 2026-08-20T00:00:00Z
agent: claude
style: detailed

# Production bake-off on the fixed plant — PPO / SAC / TD3 vs the MPC and heuristic advisors

**Follows:** `2026-08-20_detailed_v2_fixes_applied.md`. Every run below is on the post-fix plant.
**Driver:** `python/arlamx_v2/bench_production.py` · **Figures:** `python/arlamx_v2/plot_production.py`
**Artefacts:** `outputs/presentation/` (runs, data, plots, diagrams, animations)

## Setup

Eighteen training runs: PPO, SAC and TD3 × {50 000, 250 000} steps × seeds {42, 43, 44}, all on the
identical SC_v7 stage-S4 configuration (`kp=1e-4`, `kd=2e-3`, `obs_extra=observer`, `gate_floor=0.3`,
reward `band_above_mode=exp`, `dE_weight=3.5`, `thrift_weight=0.9`), 32 parallel environments.
Only the learner and the seed change, so the comparison isolates the algorithm.

Scoring reuses the tuning campaign's harness unchanged (`campaign.probe_suite` / `gap_index`):
3 probe scenarios × 2 seeds × 20 orbits, with the sampling MPC as the zero reference. A heuristic
advisor predict-function was added (`bench_production.heuristic_predict_fn`) so the scripted baseline
is scored on exactly the same probes.

Training cost: **14.1 min total wall** for all 18 runs. Throughput by learner:
PPO 2 619–2 688 steps/s, SAC 3 329–3 495, TD3 4 512–4 723.

## Results

| Policy | gap (median) | gap spread | decay nom (km/d) | decay storm | band % | brownouts | downlink (min/d) | actuator (J) | ms/decision |
|---|---|---|---|---|---|---|---|---|---|
| **MPC** | **0.000** | — | **9.20** | **17.8** | **84.0** | 0 | **23.2** | 98 | 6.16 |
| **Heuristic** | **0.603** | — | 16.72 | 34.3 | 4.9 | 0 | 15.4 | 250 | 0.05 |
| PPO 50k | 1.097 | 1.04 – 1.38 | 27.01 | 71.9 | 31.0 | 0 | 12.0 | 332 | 0.09 |
| PPO 250k | 1.202 | 1.08 – 3.03 | 25.96 | 65.8 | 8.8 | 0 | 12.8 | 368 | 0.10 |
| SAC 50k | 1.501 | 0.96 – 2.49 | 19.51 | 61.2 | 3.7 | 0 | 11.5 | 131 | 0.13 |
| SAC 250k | 3.179 | 0.95 – **10.07** | 21.56 | 48.3 | 3.6 | 12 | 8.4 | 88 | 0.13 |
| TD3 50k | 1.685 | 1.33 – 1.92 | 32.14 | 93.3 | 2.8 | 0 | 7.6 | **58** | 0.09 |
| TD3 250k | 1.899 | 1.64 – 2.10 | 34.16 | 112.6 | 3.0 | 0 | 12.1 | 266 | 0.09 |

## What the data says

**1. The sampling MPC is still the best controller, and the heuristic is second.** No learned policy at
either budget beat either scripted advisor on the gap index. The MPC's margin is not marginal: 9.2 km/d
decay against 27 km/d for the best learner, and 84 % battery-in-band against 31 %.

**2. Seed variance is large enough that single-seed results are not evidence.** PPO 50k battery-in-band
across three seeds: 11.6 %, 31.0 %, 61.7 %. SAC 250k gap across three seeds: 0.95, 3.18, 10.07. The
previous campaign leaderboard's headline (gap 0.468) was the best of 53 trials — a best-of-N statistic
that a fresh single run cannot be compared against, and it was also trained on the pre-fix magnetic field.

**3. More training made things worse.** Every learner's median gap degraded from 50k to 250k. PPO and
SAC begin browning the battery out at 250k. The extra budget is spent driving drag harder than the power
system can support — a reward-shaping problem (the `dE_weight=3.5` term outrunning the power terms),
not a training-length problem.

**4. Where the network genuinely wins is cost per decision.** One MPC decision takes 6.16 ms because it
rolls the dynamics forward repeatedly; the network answers in 0.09 ms at fixed cost — **65× cheaper**, with
no onboard propagator and no MSIS query in the loop. That is the difference between a controller that
fits a CubeSat duty cycle and one that does not. It is the only dimension on which the learned policies
clearly beat the MPC, and it is a real result worth stating plainly.

**5. TD3 is by far the thriftiest actuator user** — 58 J against the MPC's 98 J and PPO's 332 J, holding
attitude on a mean commanded torque of 17 nN·m against PPO's 58 nN·m. It pays for that in orbit-energy
performance (worst decay of any policy). This is a usable trade for a power-starved vehicle, not a win.

**6. The v7 authority-gate action is not doing what it was designed to do.** The learners' mean ceilings
barely differ (PPO 0.60, SAC 0.67, TD3 0.76) and — critically — the ordering is *inverted* against energy
spend: TD3 sets the highest ceiling and spends the least. The gate is a cap, and what separates the
policies is how much torque they actually command underneath it. Whatever the gates are learning, it is
not "release an axis and let the atmosphere turn the craft." Worth an ablation before the mechanism is
claimed as a contribution.

**7. The disturbance observer works, and its accuracy tracks how quiet the vehicle is.** Correlation
against the plant's true aerodynamic torque, excluding the first hour of post-deployment detumble:
heuristic r = 0.80, MPC r = 0.72, PPO r = 0.47. The estimator is
`τ_dist = I·ω̇ + ω×Iω − τ_cmd`, so hard slewing corrupts the finite-difference term — the policies that
push their actuators hardest get the worst estimate of the environment they are pushing against.

## Caveats that belong with any use of these numbers

- The env still carries the audit's unfixed scope items: **no sensor noise**, MSIS held for 300 s,
  6 ground stations rather than 392. They apply equally to every policy here, so the *comparison* is fair,
  but absolute numbers are optimistic.
- The advisors are scored with the IPC observation block masked (`obs_extra="none"`), matching the
  campaign reference. They do not consume the observer features.
- Decision-cost timings include, for the MPC and the heuristic, the onboard state assembly they need
  (density lookup ~0.02 ms, station lookup); for the network it is the forward pass. Both would need a
  stored density model in flight.
- `outputs/campaign/` remains on disk for provenance but is superseded: its models were trained against a
  magnetic field pointing the wrong way.

## Recommendation

The honest framing for the thesis is that V2.0 delivers **a validated plant and a deployable-cost
controller**, not a controller that beats model-predictive control. If the goal is to close the quality
gap, the evidence points at the reward, not the learner: the `dE` term dominates the power terms at longer
budgets, and every learner converges on trading battery margin for drag. A reward sweep on that balance,
with three seeds per point, is the next experiment — not more steps.
