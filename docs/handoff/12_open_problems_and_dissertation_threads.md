# 12 — Open problems and dissertation threads

Ordered by how much they change what can be claimed.

## A. The actuator question (blocks every control claim)

The magnetorquer-only plant cannot hold attitude with the campaign gains, in
any orbit tested, even with no disturbance (`04`). Every RL and MPC result to
date assumed an ideal body couple. Decide one of:

1. **Keep the ideal couple as an explicit modelling assumption** ("an ADCS
   with 3-axis authority of ≥ 38 µN·m"), state it in every table, and treat
   the coil sizing as a power/mass budget only. Cheapest; weakest.
2. **Add a momentum device** (one wheel + magnetic desaturation) to the plant
   and re-run the reference MPC and one RL family. Changes power and mass
   budgets (the 25.7 % actuator mass is already flagged).
3. **Design for the actuator you have**: slow projected-PD or periodic-LQR
   gains (Lovera & Astolfi 2004), advisor horizon ≥ field-rotation timescale,
   attitude targets restricted to what magnetics can reach, and aerodynamic
   passive stability (cp behind cm) doing the rest. This is the version in
   which "delegation to environmental torque" is a necessity, not a reward
   term — arguably the strongest thesis narrative, and the least built.

Whichever you choose, `attitude.ideal_torque` in the physics YAML and the
frozen `snapshot.json` make the choice auditable per run.

## B. The delegation ablation has not actually been run

`deleg_power` (eq. 6) was identically zero in every v8b–v14 run. The recorded
verdict "delegation as rewarded costs mission gap" was measured with only the
signed accuracy term live. First experiment after A: v8a vs v8b, 3 seeds,
burn-in gate, on the chosen actuator model. Also revisit whether P should be
credited on the command side only (current) or whether a physically motivated
saving (coil I²R actually avoided) is the better signal.

## C. The comparison is between advisors that both see truth

No policy or MPC has run on sensor-noised attitude/rate/field inputs; sensors
feed only the torque KF. Flight-representative results need the attitude
filter and sensor path in `_obs` (grok critique §1.4, still open). Train with
noise from day one in any new family.

## D. Structural reasons the RL loses

`docs/v12/00_LESSONS_v7_to_v11.md` §4.2: the MPC has five named attitudes
(hold/sun/gs/min-drag/nadir) as single choices; the policy must rediscover
them inside SO(3) through a 300 s actuator every episode. A structured action
space (choice + bounded residual, or mixing weights over basis attitudes) is
the one change the evidence most supports and it is flight-computable. Band
(SoC 0.4–0.6) is the shared failure of every learned and heuristic policy;
time-in-state shaping the way the MPC scores it is the proposed fix.

## E. Honest negatives already in hand

- Median RL loses to the MPC on every metric; only tails moved v7 → v11.
- The 0.096 artifact is a best-of-seeds outcome (siblings 0.82, 3.10).
- v11 adaptability did not improve post-strike orbit holding.
- Duos never switched.
- The comparator (MPC scorer as a filter) is 83 % of the 1.00 ms flight cost
  and removing it collapses band 80 → 41 %; the "policy-only 0.17 ms" number
  is not a flight mode.
- INT8 breaks the policy; fp16 storage hurts; fly fp32.
- No policy is deployable (no export path to the MCU).

## F. Plant fidelity items still open

MSIS held for 300 s; spherical vs geodetic altitude in a few reporting paths;
SH pole guard is zonal-only (not Pines); WMM not bit-matched to NOAA; SRP
torque dropped; unknown sail optical coefficients; Walker He/H fit rows not
checked against the primary paper; CLL α_N is a free input with a published
kink at 1. None of these changes a conclusion; all should be listed.

## G. Decisions the author has to make (from the v12 "airplane list")

1. Delegation mechanism as an output, or environmental torque as an input only?
2. Structured action space: discrete choice + residual vs continuous mixing?
3. Is downlink legitimately sunlit-only? Both sides currently under-count.
4. Comparator MPC: tuned (fair fight) or frozen (continuity)? Both can be reported.
5. Actuator model (A above) — before any new compute.

## H. What a dissertation can defensibly claim today

- A validated, fast (0.4 ms/300 s) C++ free-molecular plant with audited
  equations, independent cross-checks, and a documented equation history
  (`CHANGELOG_v2.1.md`), including an optional CLL kernel.
- A sampling MPC reference and a heuristic, tuned and frozen with provenance.
- RL advisors 6–60× cheaper per decision that match or beat the MPC on
  *individual* metrics (deep-altitude decay −18 %; downlink; B > 0) but not on
  all four at once, with seed-honest medians.
- The actuator-authority result: magnetic-only 3-axis hold of a drag sail
  against its own aero torque is infeasible with fast PD gains; this
  reframes "delegation" from a reward bonus into a control necessity.
- Passive-physics envelopes for two sail geometries (decay, lift, SRP
  thresholds).

## I. Queued ideas explicitly deferred (from the 08-21 handoff)

v10Duo, v10Tri, v10Phy (physics-informed network), Duo on a strong main,
seed-replicate v11_4x40_ppo_4M, a decay-holding adapt term, ONNX/MCU export,
the 3×3 gain sweep (never run), MSIS 300 s hold.
