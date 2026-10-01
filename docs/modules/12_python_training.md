# 12 — Python training (PyTorch / SB3) and SC_v3 bake-off

**Folder:** `python/arlamx_v2/`  
**Status:** implemented  
**Tests:** `tests/integration/` (env smoke). Training artifacts: `outputs/sc_v3_ppo/`

## What it does

Re-implements the **training job** V1.7 does for SolarCat SC_v3: Gym env, VecEnv, PPO, VecNormalize, callbacks. The plant underneath is `libarlamx`, not Basilisk and not copied `dynamics/env.py`.

Do **not** copy `advisor/train.py`. Write a small new trainer that reads the same *numbers* from `config_SC_v3a.yaml` as a data file.

## Reward (SC_v3 — must match the published equations)

\[
R=\sum_i w_i r_i
\]

Longevity (`dE_vs_baseline`, \(w=0.75\)):

\[
x=\mathrm{clip}\bigl((500-h)/200,0,1\bigr),\quad
w_\mathrm{long}=(e^{3x}-1)/(e^{3}-1)
\]

\[
r_{\Delta E}=w_\mathrm{long}\cdot\mathrm{clip}\bigl((\Delta E_a-\Delta E_b)/\max(|\Delta E_b|,\varepsilon),-1,1\bigr)
\]

Power (`power_budget`, \(w=1\)): depletion \(=-50\); else

\[
m(s)=\begin{cases}0.8&s\ge0.6\\2.0&s<0.6\end{cases},\quad
r=m(s)\bigl[\min(s,0.8)+0.5\,g\bigr]
\]

GS (`gs_alignment`, \(w=1\)): \(1[s>0.5]\cdot f(C_D)\cdot\max(0,\hat{z}_B\cdot\hat{g}_B)\cdot 1[\mathrm{visible}]\),  
with \(C_{D,\mathrm{ref}}=1.05\), gate 10–20 % above that Cd.

Also: shade 0.2, smoothness 0.02, omega 0.002, momentum 0.001, altitude cliff 1000 at 250 km.

Observation: the SC_v3a 26-vector (GPS r/v, gyro, mag unit+mag, v_body unit, ν sin/cos, mass, log ρ, SoC, power_gen, gs_dir_B). No Sun sensor.

PPO: arch `[16,16,16,16]`, lr \(3\times10^{-4}\), 400k steps, 32 envs, seed 42, `ent_coef` 0.003, reward-normalize clip 10, `advisor_step_s=300`, `dt_s=2`.

## Citations

1. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
2. Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR* 22(268), 1–8.

Reward physics basis (energy vs drag): Vallado (2013) §8; the SC_v3 term shapes are specified in `../REWARD_EQUATIONS_SC_v3.md` (thesis tree, read as documentation, not imported as code).

## Classroom test (trainer, not a 400k run)

**A.** `env.reset` + 3 `env.step` of a zero quaternion: obs finite, dim 26, reward finite, no NaN.  
**B.** Reward unit: feed a fake info dict with \(\Delta E_a=\Delta E_b\), SoC=0.8, no GS → longevity term 0, power term \(=0.8\cdot(0.8+0)\) before weights.  
**C.** After plant gates are green: full SC_v3a train writes **only** under `outputs/sc_v3_ppo/` and a `metrics.json` with wall-clock, eval reward, mean Cd, SoC, P_gen.

## Tests run

2026-08-18 — `tests/integration/test_env_smoke.py` **PASS** (26-dim obs, finite reward).  
Smoke PPO 4096 steps / 4 envs → `outputs/sc_v3_ppo_smoke/` (~1740 steps/s).  
Full SC_v3a 400k / 32 envs writes `outputs/sc_v3_ppo/`.
