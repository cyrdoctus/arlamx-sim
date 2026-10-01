time: 2026-08-18T22:40:00Z
agent: grok
style: detailed

# SC_v4 training plan (detailed, no training this session)

Plan only. V1.7 is the specification source (`SC_V4_ADVISOR_STUDY_REPORT.md`);
this file is how that study is rebuilt on the V2.0 library.

## Intent

SC_v4 is a *reward and envelope* generation, not a new spacecraft. The C++
plant already has GGM03S, Sentman M-01, panel SRP, co-rotating wind, MRP-PD
and B-dot. What is missing for a v4 *training* run is:

1. The four v4 reward terms in `python/arlamx_v2/reward.py`
2. Space-weather jumps and v4b brownout in `env.py`
3. Configs + a `train.py` matrix
4. A real body-frame magnetic observation (V1.7 shipped zeros)

## Connection map for a future trainer

```
train.py (SB3 PPO/SAC)
    → ArlamxV2Env.step(q4)
        → Simulator.step            # C++ plant, 2 s × 150
        → pymsis (ρ,T,m̄)
        → compose_sc_v4(info)       # not yet written
    → outputs/sc_v4_*/
```

Heuristic and sampling-MPC advisors already consume the same `state` dict
and are the baselines the trained policy must beat.

## Reward (copy the maths, not the V1.7 file)

Power band:

- depleted → −50
- SoC < 0.4 → −4 (0.4 − SoC)/0.4
- 0.4…0.6 → +1
- SoC > 0.6 → +exp(−6 (SoC − 0.6))

GS tier (daylight, visible, per body axis of the error rotation):
2° / 5° / 10° → +w / +w/2 / +w/4, else −0.5, w=2.

Action feasibility on the *raw* quaternion, 40° band.

Longevity: `dE_vs_baseline` weight 2.0, exponential altitude ramp 300–500 km.

v4b: +100 at B-dot hand-back when SoC≥0.15 and |ω|≤0.5 °/s.

## Envelope

F10.7 [65, 250], Ap [2, 40], 1–4 storm steps per episode. Same 500 km / 23° /
e=0.001 start as the decay and advisor campaigns so the numbers compare.

## Matrix

Six runs, seed 42, 32 envs, 4×16 MLP: PPO/SAC 300k and PPO 50k, each × {v4a, v4b}.

## Bake-off

After those checkpoints exist, rerun `advisor_run.py` with the RL policy as
one more name in the bank. Compare to this session’s heuristic/MPC table.

## Explicitly not done now

No 300k PPO. No SAC. No new Gym wrappers beyond corotating + feasibility
altitude check. The user asked for the plan only.
