time: 2026-08-18T22:45:00Z
agent: grok
style: detailed

# Sampling MPC and heuristic advisors on ARLAMX V2.0

This report is the user-facing mathematics and the 1 % SolarCat campaign.
A typeset PDF of the same material is
`ACEnv/Reports/2026-08-18_detailed_mpc_heuristic.pdf`.

The spacecraft has **no sun sensor**. Orbit and time are known (so nadir,
ground-station line-of-sight, and an analytic sun *ephemeris* are legal).
The sun unit vector is **not** a measurement the heuristic is allowed to
close on. The heuristic closes only on `power_gen_norm`. The MPC may use
the ephemeris the same way a flight computer uses a clock + orbit; a
second MPC variant (`mpc_no_sun_eph`) does not.

Geometry for every number below: SolarCat 1 % sealed-side mesh, 0.625 kg,
i = 23°, e = 0.001, a = Re+500 km, F10.7 = 150, Ap = 4, MSIS 2.1,
GGM03S degree 4, co-rotating atmosphere, panel SRP.

---

## 1. What the advisors adjust

Both families output one scalar-first unit quaternion `q_BN` (`q0 ≥ 0`)
every advisor step (300 s). The plant tracks it with MRP-PD at 2 s
(or, in decay, holds it exactly). That quaternion is the only actuator
command. It orients body +Z, which is simultaneously:

- the sail normal (drag / SRP)
- the two-sided solar-panel axis (power)
- the antenna boresight (ground station)

There is no independent solar-array gimbal.

---

## 2. Heuristic family

### 2.1 Mode ladder (SoC only)

Let the descending thresholds be
`t = (t0, t1, t2, t3, t4)`, default `(0.60, 0.50, 0.40, 0.30, 0.20)`.

| SoC | Command |
|---|---|
| ≥ t0 | +Z at a visible station, else a “rich” hold (min-drag, nadir, or last-good sun) |
| t1 … t0 | station if visible, else last-good sun, else search |
| < t1 | power search (no station) |

Search step size grows as SoC falls:

- ≥ t2: base sweep / climb
- t3 … t2: ×1.5
- t4 … t3: ×2
- < t4: ×2 and hold on *any* sensed power (`gen_target ← sense_floor`)

Source for mode switching: Wertz (ed.), *Spacecraft Attitude Determination
and Control*, 1978. Thresholds are mission parameters, not physics.

### 2.2 Power search (no sun vector)

State machine: `sweep → climb → hold`, with `eclipse` as a timeout.

1. **Sweep.** Rotate the commanded `q_BN` about body Y, then X, by
   `sweep_step_deg` per advisor step (left multiplication
   `q' = q_δ ⊗ q_BN`, a body-frame delta under the passive convention).
   Transition to climb when `power_gen_norm ≥ sense_floor`.
2. **Climb.** Coordinate descent on ±X, ±Y (Nocedal & Wright, *Numerical
   Optimization*, 2006). Keep a step if generation rose by more than 1e−3,
   else revert. Halve the step after a failed cycle; hold when
   `power_gen_norm ≥ gen_target` or the step is < 1°.
3. **Eclipse guard.** A fruitless 360° sweep on both axes, or generation
   dropping to ~0 in climb/hold, means umbra. Freeze the attitude for
   `eclipse_hold_steps` (default 6 = 30 min) so the ADCS does not spend
   the night hunting.

A “last-good sun” quaternion is stored whenever generation ≥ 0.5. That
is still not a sun sensor: it is a remembered command that once made
power.

### 2.3 Bank

| Name | Idea | t0…t4 | sense / target | rich mode |
|---|---|---|---|---|
| mission_v17 | V1.7 defaults | 0.60…0.20 | 0.05 / 0.70 | min-drag |
| conserve | stay high | 0.75…0.35 | 0.08 / 0.80 | min-drag |
| eclipse_hold | long umbra freeze | 0.70…0.30 | 0.10 / 0.75 | min-drag |
| eclipse_deep | 24-step freeze | 0.80…0.25 | 0.10 / 0.80 | min-drag |
| power_first | search early | 0.85…0.25 | 0.04 / 0.85 | min-drag |
| ground_first | nadir whenever rich | 0.45…0.15 | 0.05 / 0.55 | nadir |
| ground_rich_nadir | milder ground | 0.65…0.20 | 0.05 / 0.70 | nadir |
| low_threshold_search | hear a faint sun | 0.55…0.15 | 0.03 / 0.50 | min-drag |
| fine_search | 8° / 3° steps | 0.60…0.20 | 0.05 / 0.75 | min-drag |
| coarse_search | 25° / 10° | 0.60…0.20 | 0.04 / 0.60 | min-drag |
| high_floor | ignore dim glints | 0.70…0.25 | 0.20 / 0.80 | min-drag |
| any_power_hold | latch the first watt | 0.60…0.20 | 0.02 / 0.15 | min-drag |
| search_only | never GS | 0.99…0.15 | 0.05 / 0.70 | min-drag |

Fixed attitudes run in the same campaign: min-drag, max-drag, nadir,
ground-station, and off-nadir 0/15/30/45/60° toward velocity.
Search-floor sweeps: (0.02, 0.30), (0.05, 0.50), (0.10, 0.70), (0.20, 0.85).

---

## 3. Sampling MPC

### 3.1 Why sampling, not a QP

The map (attitude → Cd, power, GS cosine) is non-convex. Sentman FMF has
no cheap derivative. A receding-horizon *sampler* (Rawlings, Mayne & Diehl,
*Model Predictive Control*, 2017; Camacho & Bordons, 2007) is the honest
controller: propose a short list, roll a cheap surrogate, take the first
move of the winner, repeat.

### 3.2 What it adjusts

Same `q_BN` as the heuristic. Candidates each step:

- hold the incumbent
- sun-track from the analytic ephemeris (`mpc` only)
- ground-station track, if a station is above 10° elevation
- min-drag (Z along h)
- nadir (Z along −r)
- K = 8 random body-axis perturbations, angles uniform in [2°, 0.85 × 40°]

Every candidate is SLERP-clipped to 40° from the incumbent, matching the
SC_v4 feasibility band:

\[
q(\tau)=\frac{\sin((1-\tau)\Omega/2)}{\sin(\Omega/2)}\,q_a
+\frac{\sin(\tau\Omega/2)}{\sin(\Omega/2)}\,q_b,\quad
\Omega=2\arccos|\langle q_a,q_b\rangle|,\quad
\tau=\min\bigl(1,\,40°/\Omega\bigr).
\]

### 3.3 Surrogate (H = 6 steps = 30 min)

The candidate quaternion is held **inertially** (zero-order hold), which
is what the plant does between inferences.

- **Orbit.** Two-body RK4, 60 s:
  \(\ddot{\mathbf r}=-\mu_\oplus\mathbf r/\|\mathbf r\|^3\),
  \(\mu_\oplus=3.986004418\times10^{14}\,\mathrm{m}^3\mathrm{s}^{-2}\).
  J2, drag and SRP are omitted — they do not change *ranking* over 30 min
  (Vallado 2013 §1).
- **Eclipse.** Cylindrical: sunlit iff
  \(\mathbf r\cdot\hat s\ge 0\) or
  \(\|\mathbf r-(\mathbf r\cdot\hat s)\hat s\|\ge R_\oplus\).
- **Power.** Two-sided cosine
  \(P_\mathrm{gen}=P_\mathrm{peak}\,\eta\,|\hat z_N\cdot\hat s|\,\mathbf 1[\mathrm{lit}]\),
  \(P_\mathrm{peak}=0.78\,\mathrm{W}\), \(\eta=0.85\).
  Loads: 7 mW always, 205 mW GPS when lit, 400 mW TX if
  \(\hat z_N\cdot\hat g > 0.7\) and lit.
- **Supercap.** \(E=0.53\,\mathrm{Wh}\).
  \(\mathrm{SoC}\leftarrow\mathrm{clip}_{[0,1]}(\mathrm{SoC}+(P_g-P_l)\Delta t/E)\).
- **Drag.** Sentman `coefficients_only` on the real 1 % panels,
  \(\mathbf v_B=-R_{BN}\mathbf v\), latest MSIS \((\rho,T,\bar m)\).

### 3.4 Objective

\[
J(q)=\sum_{k=1}^{H}\Big[
B(\mathrm{SoC}_k)
+\mathbf 1[\mathrm{pass,lit}]\big(T(\theta_{gs,k})+w_\mathrm{align}\cos\theta_{gs,k}\big)
-w_{cd}\,C_{d,k}
\Big]
-w_\mathrm{slew}\frac{\Delta_\mathrm{cmd}}{40°}
\]

Power band \(B\):

\[
B(s)=\begin{cases}
-4\,(0.4-s)/0.4 & s<0.4\\
+1 & 0.4\le s\le 0.6\\
\exp(-6(s-0.6)) & s>0.6
\end{cases}
\]

GS tier \(T\): full / half / quarter / −0.5 at 2° / 5° / 10° / beyond.

Default weights: power_band 1, power_low_pen 4, gs 2, gs_pen 0.5,
gs_align 1, cd 0.5, slew 0.2.

The *physical* plant still runs GGM03S + Sentman + SRP + MSIS. The
surrogate is only inside the planner’s head.

### 3.5 What the MPC cannot know

It does not get a sun-sensor measurement. Variant `mpc_no_sun_eph` also
drops the ephemeris sun-track candidate, so sun pointing has to appear
from random slerps scored by the power band — the same information the
heuristic has.

---

## 4. Power and ground-pointing limits (what the campaign is for)

Without a sun sensor the craft cannot put +Z on the sun except by
climbing the power reading. Two-sided ±Z panels change the nadir story:

- **Nadir** (Z ‖ −r). At local noon \(\mathbf r\) is nearly sunward, so
  the anti-sun face still sees the Sun and \(|\hat z\cdot\hat s|\approx 1\).
  The 2-day run measured mean sunlit generation **0.81** of peak — nadir
  is a power-positive communications attitude on this bus.
- **Off-nadir.** Tilting +Z from nadir toward velocity by 0°/15°/30°/45°/60°
  keeps min SoC at 0.528 / 0.523 / 0.517 / 0.511 / 0.504 and sunlit
  generation at 0.81 / 0.79 / 0.77 / 0.74 / 0.71. A 60° ground-looking
  tilt is still eclipse-stable.
- **Eclipse.** ~35 min of umbra per ~95 min orbit. A 0.53 Wh supercap at
  the GPS+sensing load drops about 0.23 SoC in one eclipse. Heuristics
  that *latch a dim false sun* (sense_floor 0.02, or hold-on-any-power)
  enter umbra too low and brown out. Floor 0.05–0.20 with a real climb
  target survives.

Stations used for i=23°: Honolulu, Mexico City, Mumbai, Singapore, Quito,
Darwin (all \(|\phi|\le 22°\)). Tucson / Prague / Santiago are outside
the ground track and were not used.

Campaign length: 2 days, 300 s advisor, 2 s MRP-PD, 1 % SolarCat.
Results land in `outputs/advisors/summary.csv` and are copied into
§5 when the run finishes.

---

## 5. Campaign results (2 days, 1 % SolarCat)

| Policy | min SoC | brown | gen (lit) | GS lock | nadir 30° | survived |
|---|---|---|---|---|---|---|
| mission_v17 | 0.322 | 0 | 0.472 | 0.049 | — | yes |
| eclipse_hold | 0.362 | 0 | 0.474 | 0.049 | — | yes |
| power_first | 0.382 | 0 | 0.472 | 0.045 | — | yes |
| ground_first | 0.528 | 0 | 0.802 | 0.052 | high | yes |
| nadir | 0.528 | 0 | 0.811 | 0.028 | 1.00 | yes |
| off_nadir_60 | 0.504 | 0 | 0.713 | 0.016 | tilt | yes |
| min_drag | 0.425 | 0 | 0.449 | 0.035 | — | yes |
| mpc | 0.361 | 0 | 0.336 | 0.021 | — | yes |
| mpc_no_sun_eph | 0.370 | 0 | 0.335 | 0.010 | — | yes |
| search_f05_t50 | 0.325 | 0 | 0.421 | 0.035 | — | yes |
| search_f20_t85 | 0.375 | 0 | 0.402 | 0.031 | — | yes |
| conserve | 0.000 | 542 | 0.189 | 0.023 | — | **no** |
| any_power_hold | 0.000 | 527 | 0.195 | 0.035 | — | **no** |
| search_f02_t30 | 0.000 | 452 | 0.306 | 0.036 | — | **no** |

GS lock ≈ 5 % is essentially every visible pass (stations are only up for
minutes). The three failures all *stopped searching too early* and rode a
dim attitude through eclipse.

Full table, SoC histories and the off-nadir curve:
`outputs/advisors/summary.csv`, `soc_timeseries.png`, `off_nadir_budget.png`.

---

## 6. Connection map

```
HeuristicPowerPolicy / SamplingMpcPolicy
    -- q_BN -->  Simulator.step (point mode, 2 s × 150)
                     |-- GGM03S n=4
                     |-- Sentman, v_gas = −C(v−ω⊕×r)
                     |-- panel SRP
    <-- r, v, C_BN, sun_N, eclipse -- get_state
    <-- power_gen_norm, battery_soc -- Python power integrator
    <-- gs_dir_N, gs_visible      -- stations.py (elevation ≥ 10°)
```

The heuristic reads SoC and `power_gen_norm` only, plus r,v to build
min-drag / nadir / GS frames (the same privilege as the analytic
baselines). The MPC additionally rolls the surrogate.

---

## Citations

1. Rawlings, J. B., Mayne, D. Q. & Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design*, 2nd ed.
2. Camacho, E. F. & Bordons, C. (2007). *Model Predictive Control*, 2nd ed.
3. Wertz, J. R. (ed.) (1978). *Spacecraft Attitude Determination and Control.*
4. Nocedal, J. & Wright, S. (2006). *Numerical Optimization*, 2nd ed.
5. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4th ed.
6. Sentman, L. H. (1961). Free Molecule Flow Theory… LMSC-448514.
7. Schaub, H. & Junkins, J. L. (2018). *Analytical Mechanics of Space Systems*, 4th ed.
