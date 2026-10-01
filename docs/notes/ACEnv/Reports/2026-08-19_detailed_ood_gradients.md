time: 2026-08-19T08:00:00Z
agent: grok
style: detailed

# OOD / gradient / solar-weather / brownout analysis (v4 family)

Train box: altitude 300–500 km, inclination 20–30°, F10.7 65–250, Ap 2–40.
Rollouts here are **48 advisor steps** (~4.0 h at 300 s) with e = 0.001 and
true anomaly 90° so the start radius is the requested altitude, not a random
point on an e = 0.01 ellipse. Plots: `outputs/ood/grad_*.png`.

Geodetic Δh over a few orbits is dominated by J2 + argument of latitude.
Use it as a shape diagnostic, not a lifetime number. Lifetime belongs on
the vis-viva SMA (added as `dsma_km` for the v5/v6 rerun).

## Altitude, including OOD 260 / 550 / 650 km

Every policy hits the −1000 altitude cliff at **260 km** (7 steps, ~35 min).
That is the hard floor.

From 300 km up, two clusters:

- **Power-stable:** MPC, v4a SAC, v4b SAC. Returns rise with altitude
  (MPC 400 km +25, 500 km +41, 650 km +39). Min SoC stays 0.36–0.50 even
  at 650 km. These policies sun-point enough that the 4 h bus does not die.
- **Power-dead:** v4a PPO 300k, heuristic, min-drag. Min SoC = 0 at every
  altitude ≥ 300 km. Returns −500 to −1700. Min-drag does not point the
  membrane at the Sun, so a 4 h hold with GPS + MTQ load empties 0.53 W·h.
  v4a PPO matches that failure mode: the 16-step bake-off hid it.

v4b PPO sits in between: it browns, then the new sun-search recovery
keeps SoC at 0.08–0.14 instead of a hard zero, and return is only −50 to
−170 above 300 km.

**OOD above the train box (550 / 650 km) does not collapse the SAC/MPC
SoC.** The cliff is below 300 km, not above 500 km.

## Inclination (OOD: 0°, 35°, 51.6°, 80°, 97.4°)

MPC is flat across i (return 24–33, SoC ~0.50). Both SAC models are
similarly flat (SoC 0.36–0.46). v4a PPO is the worst at every i
(−960 to −1900) and dies on power. Heuristic is unstable in i (best at
equator, worst at 35°). Min-drag is smooth and second-tier once power
death is accepted (−15 to −67).

## F10.7 spike at step 6 (65 → 300) and Ap storms

MPC / SAC min SoC is almost insensitive to the spike (density lag ≫ 4 h
for the bus, and they already hold SoC). Superstorm (F10.7=300, Ap=80)
does increase geodetic |Δh| (MPC −17 km quiet → −23 km superstorm) —
the atmosphere is doing work, the battery is not the first casualty.

Heuristic is the weather-sensitive policy: SoC 0.26 at F10.7=65, 0 at
100–150, 0.29 at 200, 0 at 300. v4a PPO is dead at every spike. v4b PPO
holds SoC ~0.26–0.30.

Storm sweep at 400 km (quiet / moderate / storm / superstorm) ranks the
same: MPC ~+27, v4b SAC +12 to +16, v4a SAC ~−6, everyone else deeply
negative. Superstorm |Δh| is 4–10 km worse than quiet.

## 500 km SRP flash (5× pressure, 6 steps)

MPC +45.8, v4b SAC +25.0, v4a SAC +17.5, v4b PPO −3.2. v4a PPO / heuristic /
min-drag still empty the bus. The flash is visible in return for policies
that are already alive; it does not rescue a dead bus.

## Forced brownout vs true anomaly (v4b only)

The flight computer now **detumbles with B-dot, then sun-searches +Z to
the Sun** once |ω| < 2 deg/s, and hands back when SoC ≥ 0.15 and
|ω| ≤ 0.5 deg/s. That sequence is policy-independent, so PPO and SAC
share the same recover-time curve:

| ν [deg] | steps | minutes |
|--------:|------:|--------:|
| 0, 300, 330 | 5 | 25 |
| 180 | 7 | 35 |
| 150 | 9 | 45 |
| 120 | 10 | 50 |
| 210 | 13 | 65 |
| 30 | 15 | 75 |
| 90 | 33 | 165 |
| 270 | 40 | 200 |
| 240 | 47 | 235 |
| 60 | 50 | 250 |

Peaks at ν ≈ 60° and 240° are eclipse-entry: the bus must wait for
daylight before the sun-search can charge. Post-recovery **policy**
shows up in SoC at 80 steps: SAC ends 0.44–1.0 on the sunlit side and
still 0.15–0.55 after the long waits; PPO often re-browns (SoC 0–0.18).

v5b is trained with a 180-point recover bonus plus a speed term so the
policy, once it has the bus back, is paid to keep it.

## PPO burn-in (v4a plant, 32 × 100k, then robust eval)

Short 16-step eval crowned trial 1 (lr=1e-4, n_steps=128, batch=64,
ent=0.003) at **+31.4**. Three seed reruns of those knobs did **not**
reproduce it (−2.7, +6.1, −6.7). A 4×64-step re-eval of all 32 trials
keeps trial 1 as the only recipe that does not empty the battery
(return −37 ± 46, min SoC 0.11). Next best is trial 20 (−130). Those
knobs are what SC_v5 / SC_v6 train with; extra consistency seeds
45/46/47 of the same knobs all emptied the bus (−231 / −623 / −1110).
100k PPO is high-variance; trial 1 is the least-bad recipe, not a
stable attractor.

## SC_v5 (100k, same knobs) — `outputs/ood_v5/`

**v5a** is the first PPO that holds SoC outside a 16-step eval. Min SoC
≈ 0.44 from 300–650 km (flat, including OOD 550/650). Returns join the
SAC/MPC band (400 km −32, 500 km −1, 550 km +0.8) instead of v4a PPO’s
−1100/−900/−850 with a dead bus. i-sweep is flat. 500 km SRP flash:
+11.9, SoC 0.41. SMA loss at 400 km is −3.8 km / 4 h vs min-drag −1.3 km
— power is solved, lifetime is not, at 100k.

**v5b** shares the plant recover-vs-ν curve (25 min sunlit, 250 min
eclipse-entry). After hand-back it keeps SoC 0.13–0.43 where v4b PPO
re-browns. On free-running episodes it still hits SoC = 0 and the 180
recovery bonus makes some storm/inclination returns large and noisy.

## SC_v6 (100k / 250k / 500k / 4M) — `outputs/ood_v56/`

v6 advisor step is 150 s, so 48 steps ≈ 2.0 h (v4/v5 48 steps ≈ 4.0 h).
Compare SoC and return, not raw Δh.

**4M is the PPO that matches SAC/MPC on power and sits next to MPC on
return.** Min SoC = 0.479 from 300–650 km (flat, including OOD). Returns:
400 km −2.8, 500 km +24.5, 550 km +25.6, 650 km +21.5. Storm sweep is
+22 at every Ap, SoC 0.39. 500 km SRP flash +43.7 (MPC +45.8). i-sweep
SoC 0.49, return −9 to +4.

100k is already alive (SoC 0.34–0.45). 250k locks SoC at 0.456 but
gives back some return. 500k is a mid-training dip (storm SoC 0.33).
4M takes the return back and holds the highest PPO SoC.

Forced brownout at 150 s/step: sunlit ν recovers in 20–35 min and 4M
ends at SoC 1.0. ν = 60° and 240° do **not** recover in 80 steps
(3.3 h) — v5 had 6.7 h in the same step budget. The recover-vs-ν shape
is still eclipse-driven.

SMA decay at 400 km is −4.6 km / 2 h. Lifetime vs min-drag is not won
yet; the 4M policy spent its capacity on power, GPS-dropout robustness,
and staying in the SAC/MPC return band.
