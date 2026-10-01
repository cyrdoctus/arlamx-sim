time: 2026-08-19T08:20:00Z
agent: grok
style: detailed

# SC_v5 / SC_v6 status

## PPO burn-in (v4a plant, 32 × 100k)

Knobs only. Best short eval was trial 1: lr=1e-4, n_steps=128, batch=64,
ent=0.003, epochs=10, clip=0.2, GAE λ=0.95, return **+31.4**.

That +31 is a 16-step fluke. Seed reruns 42/43/44: −2.7 / +6.1 / −6.7.
A 4×64-step re-eval of all 32 trials still ranks trial 1 first, and it is
the **only** recipe that does not empty the bus (return −37 ± 46, min SoC
0.11). Extra consistency seeds 45/46/47 of the same knobs all died
(−231 / −623 / −1110). 100k PPO on this plant is high-variance; the
trial-1 *checkpoint* is the useful artifact, the knobs are the least-bad
recipe. SC_v5/v6 train with those knobs.

Board: `outputs/tune_ppo/{board,best}.json`, `tune_board.png`.

## What changed in the plant for v5/v6

- MRP-PD retuned to the Earth Cup magnetorquers: τ_max = (1.8, 1.8, 0.57)×10⁻⁵ N·m
  (0.6 / 0.19 A·m² × 3×10⁻⁵ T), kp=4×10⁻⁴, kd=8×10⁻³, rate 0.125 deg/s.
  A 2° error is unsaturated; a 40° error saturates. Granular near target.
- Power integrates `advisor_step_s` (300 s v5, 150 s v6) and real MTQ effort.
- Brownout: B-dot until |ω|<2 deg/s, then +Z sun-search, hand-back at
  SoC≥0.15 and |ω|≤0.5 deg/s. Recovery is now possible and ν-dependent.
- v5 flashes (2–6× SRP, 3 steps) apply only above 450 km.
- v6: 5th action is MTQ scale 0.15–1.0; GPS dropout 35% uses a two-body
  hold from the last fix over the elapsed time; inference holds until a
  slew longer than 150 s has finished (or one extra step if the turn < 8°).

Rewards: v5 longevity weight 3.5, lift_work (stronger below 450 km, no
lift in the observation), srp_work (stronger above 420 km). v5b adds
recovery 180 + speed term. v6 keeps that and raises torque_thrift to 0.9
plus GPS/inference thrift.

## SC_v5 trained (100k PPO, 32 env, ~37 s each)

Checkpoints: `outputs/SC_v5a_4x16_PPO_tuned/`, `outputs/SC_v5b_4x16_PPO_tuned/`.
OOD: `outputs/ood_v5/`.

**v5a is the first PPO that keeps the bus alive outside the 16-step
eval.** Min SoC ≈ 0.44 from 300–650 km, including OOD 550/650. Returns
sit next to the SAC/MPC cluster (400 km −32, 500 km −1, 550 km +0.8)
versus v4a PPO 300k at −1100 / −900 / −850 with SoC = 0. Inclination
sweep is flat (SoC 0.42–0.46). 500 km SRP flash return +11.9, SoC 0.41.
SMA decay at 400 km is −3.8 km / 4 h, worse than min-drag (−1.3 km):
100k bought power, not lifetime.

**v5b** learns the recover bonus. Forced-brownout time vs ν matches the
plant curve (25 min sunlit, 250 min if you start into eclipse). After
hand-back it holds SoC 0.13–0.43 (v4b PPO often re-browns to 0). On
nominal episodes it still browns (soc_min often 0) and the 180-point
bonus makes storm/inclination returns jump around (+187 in one storm).
That is the heavy recovery reward doing what it was asked.

## SC_v6 trained (100k / 250k / 500k / 4M)

| run | wall | steps/s |
|-----|------|---------|
| 100k | 36 s | 2763 |
| 250k | 89 s | 2794 |
| 500k | 179 s | 2794 |
| 4M | 1416 s | 2824 |

Checkpoints: `outputs/SC_v6_4x16_PPO_{100,250,500,4000}k/`.
OOD: `outputs/ood_v56/` (13 policies).

**4M** is the PPO that keeps SoC 0.48 from 300–650 km, returns +24 at
500 km (MPC +41, v5a −1, v4a PPO −900), holds +22 through a superstorm,
and +43.7 on the 500 km SRP flash. After a sunlit brownout it hands
back in 20–35 min and ends at SoC 1.0. Eclipse-entry (ν=60°, 240°)
still needs more than 3.3 h at the 150 s step.

100k is already power-stable. 250k flattens SoC. 500k dips. 4M is the
one to keep. Lifetime vs min-drag is still open: SMA −4.6 km / 2 h at
400 km vs min-drag −1.3 km / 4 h. The extra capacity went to power,
GPS dropout, and MTQ thrift, not yet to edge-on drag.
