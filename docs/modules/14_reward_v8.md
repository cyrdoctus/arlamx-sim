# 14 — SC_v8 / v8Duo / v9: equations, values, and where each number came from

**Status: implemented and trained** (v8–v11 campaigns, outputs/.old/v8).

Code: `python/arlamx_v2/{reward_v8,estimators,propagator,power,sensors,duo,config}.py`
Configs: `python/configs/{reward_v8a,reward_v8b,reward_v9,gains_mrp,estimator_kf,power_mtq,duo,sensors_solarcat}.yaml`

Provenance tags used throughout: **[SPEC]** datasheet · **[V7]** inherited
unchanged · **[DERIVED]** computed from other quantities · **[TUNED]** changed
with a stated reason · **[ASSUME]** engineering estimate, no measurement yet.

---

## 1. What changed conceptually

In SC_v7 the policy emitted a target quaternion plus three *authority gates*, and
the reward paid a flat `torque_thrift` penalty on applied torque. Two things were
wrong with that framing:

1. **The policy looked like it was commanding actuators.** It is not, and should
   not. It advises an attitude; MRP-PD (default) or quaternion PD closes the loop.
2. **The gate was rewarded for being used, not for being right.** The v7
   bake-off showed the consequence: the learners' mean ceilings barely differed
   (PPO 0.60, SAC 0.67, TD3 0.76) and the ordering was *inverted* against energy
   spend. Whatever the gates learned, it was not "release an axis and let the
   atmosphere turn the craft."

v8 reframes the same three outputs as a **delegation split** and pays for the
outcome: power actually saved, in a direction the environment was actually
pushing.

---

## 2. Delegation (v8b only)

Action: `a = [q_target (4), s (3)]`, `s_i ∈ [0,1]` after the tanh squash is
mapped from `[-1,1]`.

### 2.1 Split to gate

```
gate_i = 1 − s_i · (1 − gate_floor)                                    (1)
```

`s_i = 0` → full magnetorquer authority. `s_i = 1` → the axis drops to the floor
and environmental torque does the turning. `gate_floor = 0.3` **[V7]** — an axis
is never fully released, so the turn stays damped.

### 2.2 Was the environment actually helping?

Per axis, with `τ_want` the torque the MRP-PD demanded **before** the gate clipped
it, and `τ_env` the Kalman-filtered environmental torque estimate:

```
A_i = sign(τ_env,i · τ_want,i) · min(1, |τ_env,i| / max(|τ_want,i|, ε))   (2)
```

`A_i = +1`: the environment is already pushing the way the controller wanted,
hard enough to matter. `A_i < 0`: it is fighting, and delegating to it is a
mistake. `ε = 1e-11 N·m` **[ASSUME]** — 0.06 % of X/Y peak torque, below which
"what the controller wanted" is numerical noise.

`τ_want` is the plant’s `tau_demand_mean`: the slew-clipped PD demand **at
full authority**, i.e. before the per-axis torque clip that carries the gate.
(v2.5 used `tau_cmd_mean`, which is that clip’s output and so can never exceed
`gate_i · τ_max,i` — the gate could never “bind” and (4) was identically zero.
Fixed in v2.6.) Not an unslewed Python replay of the advisor quaternion.

### 2.3 Benefit and power saved

```
B = Σ_i s_i A_i / max(Σ_i s_i, ε)                    ∈ [−1, 1]         (3)

E_full_i = min(|τ_want,i|, τ_max,i) / τ_max,i             (full authority)
E_gate_i = min(|τ_want,i|, gate_i · τ_max,i) / τ_max,i    (what the gated loop commands)
P        = Σ_i (E_full_i − E_gate_i) / max(Σ_i E_full_i, ε)  ∈ [0, 1]   (4)
```

(4) is a *model-based* counterfactual, not a second simulation, and it is
measured on the **command** side. `|τ_want,i|` is the plant’s per-axis mean of
the absolute full-authority demand (`tau_demand_absmean`), so a demand that
flips sign inside one advisor step still counts where it saturated.

v2.6 change: earlier drafts subtracted the *measured* `|τ_ctrl,i| / τ_max,i`.
With the closed-loop magnetorquer the delivered torque is the B-perpendicular
projection of the command and sits well below `gate_i · τ_max,i` whenever the
demand has a component along B. Paying that projection loss (or a rate-cap
reduction) as a “gate saving” would let the policy farm `w_power` by delegating
exactly when B happens to align with the demand, which is not the hypothesis.
`tests/reward/test_v8.py::test_power_saved_ignores_projection_loss` pins it.

### 2.4 Significance gate and reward terms

```
S = clip((P − p_min)/(p_sig − p_min), 0, 1)                            (5)
r_deleg_power = w_power · P · clip(B, 0, 1)                            (6)
r_deleg_acc   = w_accuracy · B                                         (7)
boost         = 1 + κ · S · clip(B, 0, 1)                              (8)
```

(8) multiplies `dE_vs_baseline`, `power_band`, `gs_tiered_pointing` — **positive
values only**, never penalties.

| symbol | value | tag | why |
|---|---|---|---|
| `w_power` | 6.0 | [ASSUME] | Deliberately the largest weight in the file. If it is not large enough to move the policy, the v8a/v8b experiment answers nothing. |
| `w_accuracy` | 2.0 | [ASSUME] | Smaller but **signed**, so the policy cannot farm `w_power` by delegating everything into a hostile environment. |
| `κ` | 0.5 | [ASSUME] | At most +50 % on the mission terms delegation helped produce. |
| `p_min` | 0.05 | [ASSUME] | Encodes "a 4 % improvement is not worth a mission penalty" — below this, no boost at all. |
| `p_sig` | 0.25 | [ASSUME] | Full boost at a 25 % actuation saving. |

**Known sparsity risk (must be checked in burn-in):** when the controller demands
essentially zero torque, `E_act ≈ 0`, so `P = 0` and (6) never fires. In a smoke
run with a near-tracked attitude, `deleg_power` was identically zero while
`deleg_accuracy` was active. If `deleg_power` stays at zero across a burn-in
episode, `w_power` is irrelevant and the mechanism reduces to (7) alone.

---

## 3. Stability and storm resilience (v8a **and** v8b)

```
r_margin = −w_m · max(0, (h_soft − h)/(h_soft − h_floor))²             (9)
r_dstab  = −w_s · min(max(0, dE_prev − dE)/scale, cap)                (10)
```

| symbol | value | tag | why |
|---|---|---|---|
| `w_m` | 4.0 | [ASSUME] | Soft barrier weight. |
| `h_soft` | 320 km | [ASSUME] | 70 km of margin above the 250 km cliff, so altitude is defended *before* a storm eats it rather than after. |
| `h_floor` | 250 km | [V7] | Same floor as `altitude_cliff`. |
| `w_s` | 0.6 | [ASSUME] | |
| `scale` | 100 J/kg | **[DERIVED]** | Measured on the reference orbit: `dE ≈ −500 J/kg` per 300 s step, median step-to-step change **39 J/kg**, p90 **478**. This scale puts the median penalty at −0.23 and p90 at −2.9 — the same order as every other term. |
| `cap` | 3.0 | [ASSUME] | One storm step cannot swamp an episode. |

(10) is **one-sided** on purpose. A two-sided `|dE − dE_prev|` also punishes the
normal perigee/apogee and day/night density swing, which the policy cannot and
should not fight, and punishes *recovery* as hard as degradation.

> The first draft used `scale = 0.02`, which was wrong by four orders of
> magnitude and produced a −3231 reward term that buried everything else. It was
> caught by running the env, not by reading it. The value above is measured.

---

## 4. Controller gains — gradual response

Law (unchanged, `cpp/src/control/mrp_feedback.cpp`):

```
τ = −kp·σ_err − kd·ω_err + ω × Iω                                     (C1)
ω_n = √(kp/I)      ζ = kd / (2√(kp·I))      t_settle ≈ 4/(ζ·ω_n)      (C2-C4)
```

`I = diag(0.0125, 0.0125, 0.025) kg·m²`. **Corrected linearisation** (grok 2.4):
with σ ≈ Φ/4 the Euler-angle loop is `I·Φ̈ + kd·Φ̇ + (kp/4)·Φ = 0`, so
`ω_n = √(kp/(4I))` and `ζ = kd/√(kp·I)` — an earlier draft treated σ as the
angle and claimed ζ ≈ 0.9; the settling time was accidentally right because the
factors cancel in C4.

| | kp | kd | ω_n (X/Y) | period | ζ (X/Y) | regime |
|---|---|---|---|---|---|---|
| v7 | 1.0e-4 | 2.0e-3 | 0.045 rad/s | 141 s | 1.79 | overdamped |
| **v8** | **6.0e-5** | **1.6e-3** | 0.035 rad/s | 181 s | 1.85 | overdamped |
| v8 Z | 6.0e-5 | 1.6e-3 | 0.024 rad/s | 256 s | 1.31 | overdamped |

**[TUNED]** — the v7 loop burns its slew at the start of each 300 s step and
then sits, which is the behaviour that ran PPO's actuator duty to 332 J in the
v7 bake-off. v8 lowers both gains so the torque is spread across the step; all
axes are overdamped by design — no overshoot, no ringing. A 3×3 sweep grid
(`gains_mrp.yaml: sweep`) is provided so this is confirmed rather than asserted.

---

## 5. Disturbance-torque Kalman filter

Measurement (the v7 observer, now fed **measured** rates):

```
z = I·(ω₁ − ω₀)/Δt + ω₁ × Iω₁ − τ_cmd                                 (E1)
```

State and model — constant-velocity, because aerodynamic torque varies smoothly
on the orbital timescale (~90 min) while the measurement noise is white:

```
x = [τ, τ̇]
F = [[I₃, Δt·I₃], [0, I₃]]
Q = σ_jerk² · [[Δt³/3, Δt²/2], [Δt²/2, Δt]] ⊗ I₃                      (E2)
H = [I₃, 0]
σ_z,i = √( (I_i·σ_ω·√2/Δt)² + (2|Iω|_i·σ_ω)² + σ_τcmd² )              (E3)
```

`σ_ω` comes from the **gyro datasheet** through the sensor module's bandwidth
convention `σ = D/√(2Δt)`, so `R` is derived rather than tuned.

| symbol | value | tag |
|---|---|---|
| `σ_drive` | 3.0e-10 N·m/s^1.5 | [DERIVED] — white PSD of the CV driving noise (τ̈; "jerk" in the first draft was a misnomer — grok 2.5). Sized from the v8 plant's real dynamics: with the cp-offset the aero torque changes ~1 µN·m within a slewing step, `σ_drive·√(dt³/3) = 1e-6`. Two earlier values failed: 1e-10 made Q≫R (filter passed noise through); 1e-12 over-smoothed and DEGRADED the raw observer r 0.98→0.62 — the burn-in gate caught it. |
| `σ_τcmd` | 2.0e-9 N·m | [ASSUME] — ~10 % of a typical applied torque (dipole commanded open-loop against an estimated field). The first draft's 1e-8 dominated R and drowned the gyro term. |
| innovation gate | 5σ | [ASSUME] — beyond this, `R` is **inflated**, not the sample rejected; a hard reject deadlocks the filter exactly when the true torque steps. Joseph form now uses the SAME inflated R the gain was computed from (grok 1.6). |

**Honest scope note:** on the v8 plant the raw 300 s observer already reaches
r ≈ 0.99 against truth (the µN-class signal is far above the noise floor), so
the filter's measurable value there is protection, not improvement — it earns
its place in the noise-dominated regime (synthetic CV test: ≥ 20 % RMS gain)
and in glitch rejection. Acceptance is therefore two-sided: never degrade the
raw observer by more than 0.02, and beat it by ≥ 20 % RMS on the synthetic.

Joseph-form covariance update, so the inflated-R gate cannot break positive
definiteness.

**Acceptance (burn-in 2026-08-20, PASSED):** filtered r = 0.9908 vs raw 0.9908
on the slew-heavy driver — no degradation; synthetic noise-regime gain covered
by `tests/estimators/test_kf.py`.

---

## 6. Magnetorquer power — corrected

**This is a physics fix, and it changes the power budget relative to v7.**

Up to v7: `P = 0.270 W · effort`, **linear** in torque. A coil dissipates `I²R`
and its dipole is `m = N·I·A`, so

```
P ∝ m² ∝ τ²                                                           (P1)
m_cmd = (B × τ_cmd)/|B|²                                              (P2)
τ_achievable = m_sat × B                                              (P3)
P = Σ_i P_peak,i · (m_sat,i/m_max,i)²
```

### 6.0 Custom actuators, sized for 300 km

**SC_v7's catalogue parts cannot control the vehicle at its own lower bound.**
CR0006/MT01 give 18/18/5.7 µN·m; the worst-case aerodynamic torque at 300 km is
32–64 µN·m. v8 flies coils wound in house.

Requirement chain — 300 km, MSIS at F10.7 = 250 / Ap = 180 (ρ = 7.47e-11), sail
broadside 0.65 m², Cd 2.2, v = 7726 m/s → drag **3.19 mN**; at a 1 cm cp–cm
offset that is **31.9 µN·m**; ×1.20 margin → **38.2 µN·m**; at the *weakest*
field the vehicle sees (25.96 µT, magnetic equator at 300 km) → **1.473 A·m²**.

> **Torque is linear in the cp–cm offset, so coil power goes as its square.**
> Halving the offset quarters the power. The 1 cm figure is a *balance
> requirement on the bus*, not an observation — at 2 cm every dipole doubles and
> every power number quadruples. Mass balance buys more than copper does.

Per-axis split: broadside drag acts along the sail normal (body Z), so
`τ = r_cp × F` lands in the **X–Y plane**. X and Y carry the load; Z sees only
the tangential/shear term (bounded at 30 %).

| | as-built | P (80 °C) | mass |
|---|---|---|---|
| **X, Y** | ferrite rod 20 cm × 4 mm, L/D = 50, **µ_eff = 648**, 2800 t × 0.30 mm | 107 mW | 67.8 g ea |
| **Z** | air-core loop on the sail perimeter, 0.60 m² enclosed, 30 t × 0.20 mm | 35.6 mW | 25 g |

`µ_eff = µ_r/(1 + N_d(µ_r−1))` with the cylinder demagnetisation factor — this
is what makes rod torquers hard: at L/D = 10, µ_eff collapses to 49, so a stubby
rod is useless. Resistances are quoted **hot** (80 °C) so power is worst case;
current density stays under 1 A/mm² (conservative for vacuum, no convection).

**Power: 249 mW at full 3-axis dipole against 451 mW available sunlit → 1.8×
headroom.** Because power goes as dipole², partial commands are nearly free:
70 % dipole = 122 mW, 50 % = 62 mW, 30 % = 22 mW.

**Mass is the binding constraint, and it is flagged.** 161 g = **25.7 % of the 625 g vehicle**. For a fixed rod geometry the product
(power × copper mass) is invariant, so this can only be traded:

| P_full | mass | % of vehicle |
|---|---|---|
| 150 mW | 252 g | 40.3 % |
| 200 mW | 195 g | 31.1 % |
| **250 mW** | **161 g** | **25.7 %** ← selected |
| 300 mW | 138 g | 22.1 % |

Resulting plant values:

| | value | tag |
|---|---|---|
| `m_max` | (1.473, 1.473, 0.442) A·m² | [DERIVED] from the requirement above |
| `P_peak` | (107, 107, 35.6) mW | [DERIVED] I²R at 80 °C, as-built |
| `B_ref` | 25.96 µT | [DERIVED] — the **weakest** field in the operating band, so the plant never assumes authority the coils cannot deliver. Over the poles the same dipoles give 2× these torques. |
| `τ_max` | (38.24, 38.24, 11.47) µN·m | [DERIVED] = m_max × B_ref — **2.12× v7 on X/Y** |

Re-costing the v7 policies at the duty they actually ran:

| policy | mean \|τ\| | legacy linear | quadratic | ratio |
|---|---|---|---|---|
| TD3 | 17.1 nN·m | 0.257 mW | 0.00018 mW | 611× |
| MPC | 24.6 | 0.369 | 0.00037 | 425× |
| PPO | 58.0 | 0.870 | 0.00208 | 180× |

So actuation energy was overcharged by two to three orders of magnitude, which
is why it looked like a first-order term in the v7 bake-off when physically it
is not.

### 6.1 The bigger effect, and an honest limitation

Only `m ⊥ B` produces torque. The consequence is **not** that badly-aligned
torque costs more power — it is that it **cannot be produced at all**:

| request | field | achievable | shortfall |
|---|---|---|---|
| 18 µN·m about X | B ∥ Z (strong Y coil) | 18.0 µN·m | 0 % |
| 18 µN·m about X | B 45° in YZ | 13.0 | 28 % |
| 18 µN·m about X | B ∥ Y (weak Z coil) | **5.7** | **68 %** |
| 18 µN·m about X | B ∥ X | 0 | 100 % |

**The v2.5+ C++ plant does model this.** `apply_magnetorquer` builds
\(\mathbf{m}=(\mathbf{B}\times\boldsymbol{\tau})/|\mathbf{B}|^2\), clips per axis, and
applies \(\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}\). Gym logs plant
`tau_shortfall_frac`, `tau_cmd_mean`, `m_mean`. I²R uses
`effort_from_dipole(m_mean)`. `power.achievable_torque` remains an analysis
helper, not the dynamics.

---

## 7. Sensor error injection

One pure function per sensor, truth → measured, no filtering or clipping to
"reasonable" values. Full parameter table in `python/configs/sensors_solarcat.yaml`,
every field tagged.

Bandwidth convention (stated once, applied everywhere): a read over `dt` is one
sample band-limited to that interval's Nyquist frequency, so

```
σ(dt) = D·√BW = D/√(2·dt)        ADEV(τ) = D/√(2τ)        σ_rw = K·√dt
```

Validated by an Allan-deviation test that recovers the injected ARW to **+0.67 %**
(0.002819 vs 0.0028 dps/√Hz, log-log slope −0.4997) and the rate-random-walk
coefficient to **−3.8 %**. The two magnetometers draw from independent RNG
streams; a mutation test confirms the independence check fails (Pearson r = 0.995)
if a naive implementation shares one draw.

---

## 8. v9 trend adaptation

```
d_alt_norm = (h(t+600) − h(t−600)) / h_ref
d_soc      = soc(t) − soc(t−600)
w_dE   ← w_dE   · (1 + gain_alt · clip(−d_alt_norm, 0, 1))            (11)
w_band ← w_band · (1 + gain_soc · clip(−d_soc/soc_ref, 0, 1))         (12)
```

`gain_alt = 0.8`, `gain_soc = 1.0`, `h_ref = 2 km`, `soc_ref = 0.2` — all
**[ASSUME]**. Bounded so one transient cannot dominate the composite.

For **v9a** the past end of each difference is the current sample, so the trend
is forward-looking only. That is the entire ablation between v9a and v9b.

Observation widths: v8 **35** · v9a **47** (+3×4 future) · v9b **59** (+3×4 past).
Both blocks carry `[Δaltitude/10, Δsma/10, soc, offset/600]` per sample.

---

## 9. v8Duo

Two policies and a **deterministic** arbiter — no learning in the arbiter.

The horizon advisor is **blind to the sensors** by design: its job is not to
re-estimate the present (the main advisor did that) but to ask what the present
command costs over 6 h. Giving it the same observation would make it a second
copy of the main advisor.

Secondary reward:

```
r = −w_decay·clip(decay_prop/ref, 0, 3)
    −w_power·clip((soc_ref − soc_end)/soc_ref, 0, 1)
    +w_improve·clip((decay_main − decay_prop)/ref, −1, 1)
    −w_slew·(1 − q_prop·q_main)
```

`w_improve = 6.0` is the largest term: the advisor is paid for **beating the main
advisor's propagated decay**, not for low decay in absolute terms.

Arbiter order — feasibility, then hard overrides, then weights:

1. reject any candidate beyond the 40° slew cap (it cannot be reached in one step)
2. `SoC < 40 %` → take the higher-power candidate
3. decay gap > 6 km/d → take the lower-decay candidate
4. `SoC > 60 %` **and** a pass is due → take the downlink candidate
5. else weighted 0.45 decay / 0.35 power / 0.20 downlink, with a 0.05 switch
   margin so the pair cannot oscillate between two near-identical attitudes

Model selection is **rule-based, not hand-picked** (`duo.pick_duo_models`): main
slot = lowest median `decay_nominal` among v8b runs with zero brownouts;
secondary slot = highest median delegation benefit `B`.

### 9.1 Onboard propagator

FP32 throughout — a fidelity statement about the flight computer, not an
optimisation. Two-body + J2 + exponential-atmosphere drag referenced to the last
measured density, attitude frozen.

Step size **[DERIVED]** by measurement against a `dt = 10 s` reference over 6 h:

| dt | decay error | cost |
|---|---|---|
| 20 s | 34 m | 46 ms |
| **60 s** | **46 m** (1.2 % of the 3.7 km horizon decay) | **15 ms** |
| 120 s | 341 m | 8 ms |

---

## 10. Open questions requiring a decision before training

1. ~~Vehicle mass~~ — **RESOLVED: 0.625 kg**, confirmed as the current figure.
   All sizing above uses it.
2. **Power-model switch.** v8 defaults to the physical quadratic model. This
   makes v8 power/energy numbers **not** directly comparable with the v7
   bake-off. `power_mtq.yaml: model: legacy_linear` reproduces v7 exactly if a
   like-for-like comparison is wanted instead.
3. **Actuator mass, 161 g = 25.7 % of the vehicle** (§6.0). High for a 625 g
   craft. It is the price of full 3-axis authority against a broadside sail at
   300 km in an extreme storm. Two levers: tighten the cp–cm balance below 1 cm
   (quadratic saving), or accept reduced authority in the worst case. Needs a
   call against the real mass budget.
4. **Dipole-space saturation** (§6.1) — plant change, currently out of scope.
5. **Delegation sparsity** (§2.4) — if `deleg_power` never fires in burn-in, the
   mechanism collapses to the accuracy term and `w_power` needs rethinking.
