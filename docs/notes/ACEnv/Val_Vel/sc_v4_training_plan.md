time: 2026-08-18T22:20:00Z
agent: grok

# SC_v4 training plan (plan only — do not train in this session)

Port of V1.7 `config_SC_v4a.yaml` / `config_SC_v4b.yaml` onto the V2.0 C++ plant.
V1.7 remains read-only. This is the recipe, not a run.

## Why v4 exists

SC_v3a–d hold the supercap near full and barely downlink. V4 changes the
*objective*, not the plant: stay in a 40–60 % SoC band, point +Z at a visible
TinyGS station in daylight, and pay for infeasible commands. Longevity weight
on `dE_vs_baseline` goes 0.75 → 2.0.

## Plant (already in V2.0)

| Item | V4 value |
|---|---|
| Geometry | 1 % sealed-side SolarCat (or hex for smoke) |
| Mass / inertia | 0.625 kg, diag(0.0125, 0.0125, 0.025) |
| Orbit start | 500 km, i=23°, e=0.001, epoch 2026-01-15 |
| Atmosphere | MSIS 2.1, F10.7∈[65,250], Ap∈[2,40], **corotating** |
| Gravity | GGM03S degree 4 |
| SRP | panel cannonball, Cr=1.8 |
| Control | MRP-PD, dt=2 s, advisor=300 s, slew cap 40° |
| Action | scalar-first q_BN, q0≥0 |
| Obs | 26-dim SC_v3a vector, **but** body magnetic field must be real (V1.7 silently zeroed `magField_B`) |

## Reward terms to add in `reward.py` (not present yet)

1. `power_band` — full +w in [0.4, 0.6], −4×deficit/0.4 below, +w e^{−6(SoC−0.6)} above, −50 if depleted.
2. `gs_tiered_pointing` — per-axis |φ| tiers 2°/5°/10°, daylight and visible only.
3. `action_feasibility` — score the *raw* command: +0.1 inside 40°, −Δ/40 beyond, −5 if invalid.
4. `brownout_recovery` (v4b only) — +100 on hand-back, +gen while B-dot is flying the bus.
5. Keep `dE_vs_baseline` at weight 2.0 with exponential altitude progression.

Exact formulae: V1.7 `SC_V4_ADVISOR_STUDY_REPORT.md` §4 (do not copy V1.7 code).

## Environment extensions (config-gated)

- Space-weather jumps: 1–4 piecewise steps per episode, ΔF10.7∈[−80,80], ΔAp∈[−20,35], clipped to the envelope.
- v4b brownout: SoC≤1e−6 → ignore the policy, B-dot detumble, shed GPS/TX; exit at SoC≥0.15 and |ω|≤0.5 °/s.

## Training matrix (when the user asks to train)

| Run | Algo | Steps | Envs | Seed | Notes |
|---|---|---|---|---|---|
| SC_v4a PPO 300k | PPO | 300000 | 32 | 42 | band + GS + feasibility |
| SC_v4a SAC 300k | SAC | 300000 | 32 | 42 | same reward |
| SC_v4a PPO 50k | PPO | 50000 | 32 | 42 | quick look |
| SC_v4b PPO 300k | PPO | 300000 | 32 | 42 | + brownout recovery |
| SC_v4b SAC 300k | SAC | 300000 | 32 | 42 | + brownout recovery |
| SC_v4b PPO 50k | PPO | 50000 | 32 | 42 | quick look |

Arch 4×16 MLP, lr 3e−4, VecNormalize on reward only, 40-orbit episodes.
Write models to `outputs/sc_v4_*`. Never into `tests/outputs/`.

## Bake-off after training

Same 7-day and 40-orbit comparison already used for the V2.0 heuristic/MPC:
min-drag, the heuristic bank, sampling MPC, SC_v3a checkpoint, new v4a/v4b.
Report SoC band fraction, brownouts, downlink minutes/day, mean Cd, km/day.

## Not in this session

No PPO/SAC launch. Advisors and decay use the plant as it stands.
