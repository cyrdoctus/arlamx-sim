time: 2026-08-20T00:00:00Z
agent: grok
style: detailed

# ARLAMX V2.0 — SC_v8 / v8Duo / v9 design critique (three-pass)

**Plant:** `ARLAMX V2.0/` C++ library + Python Gym / training face.  
**Reference (read-only):** `ARLAMX-V1.7/` (not modified).  
**Code changed in this review:** none. Nothing was implemented, trained, or deleted.  
**Authority:** hand this file to Claude Code as the fix list. Do not train or plot until the blockers in §0 are closed.

Three independent passes over the v8 / v8Duo / v9 design:

1. **Engineer** — will it run, and will the artifacts be real data?
2. **Physicist** — are the equations the right physics?
3. **Researcher** — patent claims and a physical prototype.

**Sources opened:** `docs/modules/14_reward_v8.md`; YAML under `python/configs/` (`reward_v8a`, `reward_v8b`, `reward_v9`, `gains_mrp`, `estimator_kf`, `power_mtq`, `duo`, `sensors_solarcat`); `python/arlamx_v2/{env,reward,reward_v8,estimators,power,sensors,propagator,duo,config,train,campaign,plot_production}.py`; C++ `api.cpp`, `mrp_feedback.cpp`, `mrp.cpp`, `frames.cpp`; `tests/` (including `test_v7_ipc.py`, `tests/sensors/`); `docs/VALIDATION_STANDARD.md`; `APPROVAL_PLAN.md`; reports `2026-08-20_detailed_production_bakeoff.md`, `2026-08-20_detailed_v2_fixes_applied.md`. pytest was **not** re-run; this is a read-only review.

**Verdict:** the surrounding plant can step. **v8 / v9 as wired cannot produce scientifically usable data.** Several failures are silent (reward terms stuck at 0, YAML ignored, ablation not an ablation). Training now would waste compute and create a false “it ran” corpus.

Fix the blockers first, add the missing tests, then a short burn-in with logged internals — not a 300k-step campaign.

---

## 0. Blockers — do not train until these are fixed

| ID | Severity | Where | What happens if you train anyway |
|---|---|---|---|
| **B1** | Critical | `env.py` ~752–830, `reward_v8.py` 244–247 | `tau_ctrl_mean` is never put on `info_r`. `power_saved_fraction` defaults to zeros → **`P ≡ 0` → `deleg_power ≡ 0` and `S ≡ 0`**. The v8b hypothesis is not in the reward. |
| **B2** | Critical | `reward_v8.py` 138–148 | Even after B1, **`P = 1 − gate` whenever the loop is not saturating.** Ungated demand is not `τ_ctrl / gate`. The policy can be paid for closing gates that did not change torque. |
| **B3** | Critical | `reward_v8.py` 110–125 + `env.py` 809–821 | `A_i = sign · min(1, \|τ_env\|/\|τ_want\|)`. Plant aero is **~17–58 nN·m** (v7 bake-off). PD demand for a few-degree error is **µN·m**. `A_i ~ 10⁻³` during slews. `B` and `P` have **almost disjoint support.** |
| **B4** | Critical | `env.py` 73–133 | `gate_floor` default is `0.15`, and YAML is loaded only `if gate_floor is None`. Callers never pass `None`. **`gains_mrp.yaml` `gate_floor: 0.3` is dead.** v8 runs at 0.15, not the S4/v7 floor. |
| **B5** | Critical | `env.py` 605, 671 | Brownout is `variant in ("v4b","v5b","v6","v7")`. **v8/v9 never brown out.** Battery can sit at 0 with no detumble. `brownout_recovery` is dead. Not comparable to v7. |
| **B6** | Critical | `env.py` 533–551 vs `reward_v8a.yaml` | v8a turns off the **reward**, not the **gates**. 7-D action still scales `max_torque`. v8a is not a control arm; it is “same actuators, no deleg terms.” |
| **B7** | Critical | `train.py` 126 | CLI variants: `v3,v4a,v4b,v5a,v5b,v6`. **No v8a/v8b/v9a/v9b, no v8Duo, no size sweep, no `config.snapshot()`.** `campaign.py` is v7-only. There is no way to run the planned matrix. |
| **B8** | Critical | missing tests + missing logs | No pytest for reward_v8 / KF / propagator / power / duo / env-v8. Returned `info` drops `deleg_B/P/S`, `tau_want`, split. **You cannot tell from artifacts whether delegation worked.** |
| **B9** | Critical | `docs/14` vs `power_mtq.yaml` vs `power.py` | Spec still has CR0006 / 18 µN·m / 465 mW. YAML has custom coils 1.473 A·m² / 38.2 µN·m / 249 mW. `env.py` comment still says 465 mW. **Three disagreeing vehicle definitions.** |

---

# Pass 1 — Engineer (functionality, data integrity)

## 1.1 Integration holes

**`compose_sc_v8` is wired; the inputs are not.**

`info_r` is built with `torque_effort` and later receives `tau_want`, `tau_env_est`, `deleg_split`, `gates`, `tau_max`. It never receives `tau_ctrl_mean`. That quantity is computed ~100 lines earlier and stuffed into `extra` / returned `info` **after** the reward call.

`power_saved_fraction(info.get("tau_ctrl_mean", zeros), …)` therefore always sees zeros. This is exactly the “sparsity” §2.4 of spec 14 already worried about — except it is a wiring bug, not physics.

**YAML provenance is incomplete.** `config.snapshot()` exists and is never called from `train.py`. The audit item this was meant to close (`2026-08-19_detailed_v2_triple_audit.md` F-5) is still open for any real run.

**`load_reward(variant, reward_w)`** merges overrides at the YAML root. `compose_sc_v8` reads `cfg["v7"]`. A campaign-style `reward_w={"dE_weight": ...}` is silently ignored.

**v8/v9 inherit `_v4plus` and `_v5plus`**, so weather jumps and SRP flashes are on, but brownout is off. Storms without a recovery mode.

**`_update_power` comment is stale:** “real 465 mW” while YAML sums to **0.1069 + 0.1069 + 0.0356 = 0.2494 W**.

**`env.py` module docstring** still lists variants v3–v6.

## 1.2 Training / eval / plots — the planned experiment does not exist

| Planned | Actual |
|---|---|
| v8a vs v8b isolation | Env accepts the names; trainer does not |
| Size sweep for Duo slot pick | `duo.pick_duo_models` exists; no sweep, no ledger columns `decay_nominal` / `deleg_benefit` |
| v9a / v9b obs 47 / 59 | Env can build those shapes; nothing trains them |
| v8Duo | `duo.py` is library code. Env never calls `horizon_observation`, `arbitrate`, or `compose_duo_secondary` |
| Burn-in KF acceptance (`estimator_kf.yaml`) | No runner |
| Gain sweep (`gains_mrp.yaml: sweep`) | No runner (v7 `campaign --stage s1` is a different stack) |
| Plots of B, P, gates, KF vs `tau_aero` | `plot_production.py` is the **v7 bake-off**. No v8 figure list, no writer |

`train.py` still hardcodes `arch = [16,16,16,16]`, `n_steps=128`, no ONNX export. APPROVAL_PLAN still requires actor export; still absent. Every v8 zip would be undeployable.

**What a run would actually write today:** TensorBoard scalar reward + `metrics.json` + SB3 zip. That cannot answer “did IPC / delegation work?”

`compose_sc_v8` writes `_diag` (`deleg_B`, `deleg_P`, `deleg_S`, `deleg_boost`, `deleg_align`) onto `info_r`. The Gym `info` returned to the caller is a **different dict**. Those diagnostics are computed and thrown away. `reward_parts` does contain `deleg_power` / `deleg_accuracy` (both ~0 because of B1), which is not enough to diagnose why.

## 1.3 v9 observation contract is internally inconsistent

Three texts, three layouts:

| Source | Future 4-vector | Past 4-vector |
|---|---|---|
| `docs/modules/14_reward_v8.md` §8 | `[Δh/10, Δa/10, soc, offset/600]` | same |
| `reward_v9.yaml` comments | `(d_alt, d_sma, eclipse_frac, d_soc_est)` | `(alt/7000, soc, log10 ρ, \|ω\|)` |
| `env.py` `_future_block` / `_snapshot_features` | `(Δh/10, Δa/10, soc, offset/600)` | `(alt/7000, soc, log10 ρ, \|ω\|)` |

Code matches neither the spec nor the YAML comments.

**150 s history is always zero.** Offsets `[600, 300, 150]`, advisor step 300 s:

```python
back = int(round(off / step_s))  # round(0.5) → 0 in Python 3 (banker's rounding)
if 0 < back <= len(self._hist):  # 150 s never writes
```

**v9a SoC trend is identically zero.** YAML: `d_soc = soc(t+600) − soc(t−600)`. Code: `d_soc = soc − soc_past`. v9a has no past → `d_soc = 0` → `w_band` never moves. Future SoC is not propagated (`_future_block` reuses current `soc`).

Trend uses `_fut_block[-4]` as “+600 s Δh/10”. That is only true if the last future group is 600 s **and** the 4-vector starts with Δh/10. Fragile; no test.

`propagate_samples` exceptions are swallowed → future block of zeros, policy sees “no change.”

## 1.4 Sensors are a sidecar, not the observation

`SensorSuite` is constructed for v8+. In `step()`, only `read_gyro` is called, and only for the torque KF.

`_obs()` still uses **plant truth** for `r, v, ω, B_B`. GNSS / mag / accel are never read. TTFF, dual-mag vote, hard-iron, dipole coupling (all zeros in YAML) do not exist in the loop.

The controller still runs on **true MRP** (`sim.step(q)` → C++ `quat_to_mrp`). There is no attitude estimator.

So: “sensor-realistic v8” is **true-state RL + a noisy 300 s gyro difference into a KF**. That is not a flight software stack and not what `sensors_solarcat.yaml` describes.

Gyro ZRO **0.5 dps** vs rate cap **0.125 dps**: unusable as a rate measurement. It only “works” because the KF differences two samples and the policy sees truth `ω`.

## 1.5 Duo cannot be trained or evaluated

- Horizon obs claims 18 dims; `soc_end` uses `state.get("soc_rate_per_s", 0.0)` → **SoC does not move**.
- `ecl_frac` is computed and then ignored for charge.
- `_score_candidate` uses `bc_scale` from the caller. Env never supplies a per-attitude Cd. **Both candidates get the same decay.**
- `compose_duo_secondary` uses `info.get("q_dot", 1.0)` — never set → slew penalty always 0.
- Arbiter `power_norm` / `downlink_norm` are pass-throughs of whatever the caller stuffed in. No definition in env.
- `pick_duo_models` requires a v8b ledger that nothing writes.

## 1.6 Kalman filter implementation bugs (software, not just tuning)

**Joseph form vs inflated R** (`estimators.py` 155–168): `S` is inflated; `K` uses inflated `S`; Joseph still uses **original `R`**. Posterior `P` is optimistic after a gate.

**First measurement** seeds `x` and returns raw `z` with no `τ̇` init from the model. Fine, but untested.

**No unit tests** despite VALIDATION_STANDARD (`docs/VALIDATION_STANDARD.md`: spec, 2 citations, classroom test, pytest, recorded result). Module 14 has **no citations block, no classroom numeric, Tests run = not run**. Only `tests/sensors/test_sensors.py` exists among the new modules.

## 1.7 Other functional defects

- **Mass every episode:** `self._rng.uniform(0.5, 0.75)` even for v8. YAML `vehicle.mass_kg: 0.625` unused. Inertia stays `diag(0.0125, 0.0125, 0.025)` while mass moves — drag accel and rotational inertia disagree.
- **Coil mass 161 g** is in the YAML essay, not in `p.mass` or inertia.
- **WMM never loaded** in the Gym path (`wmm_path` empty → tilted dipole only). After the 2026-08-20 mag fix the dipole is at least the right polarity; it is still not WMM.
- **MSIS still one query per 300 s**, held over 150 substeps. Already flagged; still true.
- **GMST from pre-step, stations from post-step** (~1.25°). Old bug, still there.
- **SRP flashes** last 3 advisor steps at 300 s = 15 min of 2–6× SRP, then forced off below 450 km. Easy to misread as physics.
- **`_v7` is true for v8+**, so v7 IPC obs block (35-D) is live, including **mean** gate rather than the 3-axis split the policy actually commands.

## 1.8 What would actually crash vs silently lie

Likely **runs:** `ArlamxV2Env(variant="v8b"); reset(); step(7-vector)` should not throw.  
Likely **CLI failure:** `python -m arlamx_v2.train --variant v8b`.  
Likely **silent lie:** reward looks finite; `deleg_power` is 0; gates wander; brownouts never happen; plots from v7 tools will mislabel the run.

---

# Pass 2 — Physicist (equations)

## 2.1 Delegation alignment (eq. 2) is the wrong quantity

Spec 14 / `reward_v8.py`:

```
A_i = sign(τ_env,i · τ_want,i) · min(1, |τ_env,i| / max(|τ_want,i|, ε))
```

`τ_want` is the **unsaturated PD law on the pre-step state**, including gyroscopic feedforward, **without** the C++ rate cap or torque clip, **without** the 40° slerp the plant applies.

Numbers that actually exist in this tree:

- v7 bake-off mean `|τ_ctrl|`: **17 nN·m (TD3) – 58 nN·m (PPO)** (`2026-08-20_detailed_production_bakeoff.md`).
- KF note: **~30 nN·m** aero swing per orbit (`14_reward_v8.md` / `estimator_kf.yaml`).
- v8 PD at 40°: `σ ≈ tan(10°)` ≈ 0.176, `kp = 6e-5` → `τ_want ≈ 10.6 µN·m` on X/Y.
- Aero at 6°: still `τ_want ~ 1.6 µN·m` vs `τ_env ~ 30 nN·m` → `A_i ~ 0.02`.
- `A_i = O(1)` only for `σ ≲ 30e-9 / 6e-5 = 5×10⁻⁴` → **~0.1°** of tracking error.

So B is only live when already pointed. P (if wired and saturating) is live during slews. **The product `P · clip(B,0,1)` is structurally ~0.** That matches the smoke-run note in spec 14 §2.4, and it is not a tuning issue.

**What the equation should be if the science is “is the atmosphere helping this turn?”**  
A direction cosine (or a signed projection of `τ_env` onto the **error axis** / desired `ω̇`), **not** a magnitude ratio against PD demand. Aero cannot match a PD that was sized to burn a 40° slew in one step; it can accumulate `½ α t²` with `α = τ/I ≈ 2.4×10⁻⁶ rad/s²` → **~6° per 300 s**. The policy should be paid for commanding inside that envelope, not for `τ_env ≈ τ_PD`.

## 2.2 Power-saved (eq. 4) is not a counterfactual

```python
e_act  = |τ_ctrl| / τ_max
e_full = min(1, e_act / gate)     # WRONG unless the gate is binding
```

| Regime | True ungated torque | True P | Formula P |
|---|---|---|---|
| PD below `gate · τ_max` | `τ_ctrl` | **0** | `1 − gate` |
| `gate · τ_max < τ_d < τ_max` | `τ_d` | `(τ_d − gate τ_max) / τ_d` | `1 − gate` (over) |
| Both saturate | `τ_max` | `1 − gate` | `1 − gate` |

They already compute `τ_want` and then **do not use it** in `P`. Correct sketch:

- if `|τ_want| ≤ gate · τ_max`: `P = 0`
- else: `P` from `|min(τ_want, τ_max)|` vs `|τ_ctrl|`

And P is still **linear torque effort**, after the whole module argues `P_elec ∝ m² ∝ τ²`. Battery uses quadratic `power.py`; the deleg reward does not. Those two “power saved” numbers will not match.

## 2.3 Magnetorquer physics vs the plant

`power.py` (P2)–(P3) is the right dipole inverse:

```
m = (B × τ) / |B|²
τ_ach = m_sat × B
```

The C++ plant **does not use this**. `mrp_feedback.cpp` clips **per-axis body torque**. Along-B torque is applied anyway. Power is billed for a dipole the dynamics did not need (or could not produce). Spec 14 §6.1 admits this; v8 then **resizes the coils** as if authority were the binding constraint. In the live plant it is not.

**Custom-coil requirement is not the simulated vehicle.** `power_mtq.yaml`:

- ρ = 7.47×10⁻¹¹, A = 0.65 m², Cd = 2.2, v = 7726 m/s → F = 3.19 mN (arithmetic OK).
- **cp–cm = 1.0 cm [ASSUME / layout requirement]** → 31.9 µN·m → 38.2 µN·m with 1.2×.

Default geom `earthcup_hex_v3.geom` (`ARLAMX-V1.7/geometry/models/`): membrane at **z = 0**, beams ±6 mm, hex **symmetric about origin**. Broadside drag through a planar sail with CM at the origin has **essentially zero normal-force moment**. Aero torque in this plant is Sentman **shear / beam thickness / small asymmetry** — consistent with **tens of nN·m**, not 32 µN·m.

They sized 38 µN·m coils for a 1 cm imbalance **the training geometry does not have**. Coils are ~10³× the disturbance the policy will see. “Actuators lose at 300 km” is then false **in this sim**, and v8b has no physical reason to delegate.

Rod design (20 cm × 4 mm, L/D = 50, μ_eff = 648, 2800 turns): dipole `N I A μ_eff` is internally consistent with 1.473 A·m² if you grant μ_r ~ 10⁴ and that demag formula. Z loop: 30 turns × 0.60 m² × 24.6 mA = 0.443 A·m², and 30 × hex perimeter ~3 m = 90 m of wire — also consistent on paper.

Physics issues that remain:

- μ_r / ferrite grade / temperature / remanence **unspecified**. Residual dipole into MMC5983MA is modeled as **0**.
- 20 cm rods on a 625 g ChipSat+sail: packing, boom vs bus, inertia contribution **not in `INERTIA_DIAG`**.
- Air-core loop **on the deployable sail perimeter**: EMI, deployment, and mag coupling not in the plant.
- `b_ref = 25.96 µT` is “weakest equator @ 300 km” from the **tilted dipole**, not WMM, and not the field used at 400 km training altitude.

## 2.4 MRP gain table (C2–C4) is the wrong linearization

Law is `τ = −kp σ − kd ω + ω × Iω` (`mrp_feedback.cpp`). MRP: `σ ≈ Φ/4`, `σ̇ ≈ ω/4`.

Euler-angle loop:

```
I Φ̈ + kd Φ̇ + (kp/4) Φ = 0
  ⇒  ω_n = √(kp / (4I)),   ζ = kd / √(kp I)
```

YAML uses `ω_n = √(kp/I)`, `ζ = kd / (2 √(kp I))` (treating σ as an angle). The same file also writes `I θ̈ + kd θ̇ + (kp/2) θ = 0`. **kp/2, kp/4, and kp/I cannot all be right.**

| | Their table (X/Y) | Euler-angle linearization |
|---|---|---|
| v8 ω_n | 0.069 rad/s (91 s) | 0.035 rad/s (~181 s) |
| v8 ζ | 0.92 | **~1.85 overdamped** |
| Z ζ | 0.65 “underdamped-ish” | **~1.31 overdamped** |

Settling time `4/(ζ ω_n)` accidentally matches because the factors of 2 cancel. The story “gradual but still ζ ≈ 0.9” is false. Z is not left underdamped.

Python `τ_want` also **omits the rate cap**. Cap is 0.125 dps; 40° takes ~320 s. The cap often binds; `τ_want` overstates demand.

C++ `quat_to_mrp` (`q_vec / (1+q0)`) matches the Python reconstruction; `mrp_error` composition has the identity `s ⊕ (−s) = 0`. The law match is fine; the gain table is not.

## 2.5 Disturbance KF (E1–E3)

Measurement E1 is the right Euler residual if `τ_cmd` is **applied** torque (they pass `tau_ctrl_mean` — good) and `Δω/Δt` is the right derivative (300 s endpoint difference is a **poor** estimate of mean `ω̇` over a slew that finishes in the first tens of seconds). v7 already only got r = 0.47 on a slewing policy. A constant-velocity KF on that measurement cannot recover frequencies the measurement destroyed.

Q is the **standard white-acceleration** Q for `[τ, τ̇]`, units N·m s⁻¹·⁵. They named it **jerk**. If they meant white `τ̈`, Q would be dt⁵/20, dt⁴/8, dt³/3. The math is CV; the name is wrong.

`σ_jerk = 1e-10`, dt = 300 s → `√Q_ττ ~ 3×10⁻⁷ N·m`. Gyro-derived `R` at 300 s average is **~10⁻¹⁰ N·m**. **Q ≫ R** → filter hugs the (bad) measurement. Unlikely to beat the raw observer by 0.10 correlation, which is the YAML acceptance test — and that test is not coded.

Gyroscopic noise term `2|Iω|σω` is a crude bound, not `∂(ω × Iω)/∂ω`. Minor.

KF uses **measured** ω; `τ_want` uses **true** ω. Reward alignment is privileged.

Gyro scale factor 0.5 % and misalignment 0.5° do not difference out, but at this cadence they contribute ~4×10⁻¹⁰ N·m of false torque — small next to 30 nN·m.

## 2.6 Onboard propagator

Two-body + J2 + exponential drag + co-rotating wind, float32. The J2 form matches Vallado 8-30 (`1.5 J2 μ Re² / r⁴` times `(5z²−1)` / `(5z²−3)`). Wind `v − ω × r` sign is correct:

```
ω × r = (−Ω ry, Ω rx, 0)
v_rel = (vx + Ω ry, vy − Ω rx, vz)
```

Issues:

- `bc_inverse_from_drag` uses **`drag_N = mean |F_aero|`** (`api.cpp`: `drag_sum += norm(aero.F)`), not the drag component along `v_rel`. Lift inflates Cd A/m.
- Env passes `|v|` not `|v_rel|` (~5 % on LEO).
- No SRP, no J3/GGM, attitude frozen — acceptable as a **flight** model only if errors are reported against the plant. The 46 m / 6 h figure is a self-test against dt = 10 s of the **same** model, not against `libarlamx`. That is integrator convergence, not fidelity.
- Scale height 60 km fixed; storm onsets will be wrong on 6 h.
- `propagate_samples` keeps the original `(ρ0, h0)` across segments, which is the correct exponential reference.

## 2.7 Power / energy bookkeeping

Quadratic coil model is the right exponent. Effort is then ~10⁻⁶ at nN·m duty, so `torque_thrift` (weight 0.9) is **numerically dead**. v8 “economy” is only `deleg_power`, which is also dead (B1–B3).

Battery still lumps 300 s at **end-of-step** eclipse and **end-of-step** B. Fine for a first order; not for eclipse-edge or polar B swings.

`loads_W` still 0.53 Wh supercap, 0.78 × 0.85 panel — unchanged, and **not** reduced by 161 g of copper.

Eclipse convention: `eclipse_cylindrical` returns `1` sunlit, `0` cylindrical umbra. `_update_power` treats `eclipse > 0.5` as sunlit — consistent.

## 2.8 Remaining plant physics (unchanged, still in the loop)

- SRP **torque computed and dropped** (`api.cpp`). A 1 cm-cp sail would have SRP moment; they use that 1 cm to size MTQs and then omit SRP τ.
- No self-shadowing.
- Harmonics: guarded Legendre, degree 4, not Pines.
- `dE_actual` is `∫ (F_aero + F_srp) · v dt / m` over the advisor step (J/kg). `decay_stability` scale 100 J/kg is at least the right **unit**; the 39 / 478 J/kg stats were not remeasured in this review.
- One-sided `decay_stability` (do not punish recovery / orbit-periodic dE) is the right **shape**.
- Boost only on **positive** mission terms is the right sign discipline.

## 2.9 v9 trend (eq. 11–12)

`d_alt_norm = (h_fut − h_past) / h_ref` with `h_ref = 2 km` over ±600 s. A 2 km drop in 20 min is a **violent** decay (order 140 km/day). Nominal is ~10 km/day → ~0.14 km in 20 min → clip input ~0.07 → +6 % on `w_dE`. The gain barely moves except in storms / cliff. That may be intended; it is not documented as such.

`d_soc` as implemented cannot see a future brownout (no SoC propagator). Spec, YAML, and code disagree on whether the past end is `t−600` or `t` and whether the future end is `t+600` or `t`.

---

# Pass 3 — Researcher (patent + prototype)

## 3.1 What is actually new vs what is a training trick

**Possibly claim-worthy (if reduced to practice on honest physics):**

1. **Per-axis authority split** between a weak magnetic actuator and environmental (aero / SRP) torque, with an onboard disturbance estimate, as a **closed-loop advisor** — not B-dot, not “aero-stable CubeSat,” not differential drag.
2. **Dual advisor + deterministic arbiter** (short-horizon RL + long-horizon frozen-attitude propagator, no learned mixer).
3. Flight-knowable torque observer from gyro + own `τ_cmd` (the residual itself is textbook; **using it to gate magnetorquers** is the twist).

**Not patentable as stated, and dangerous in a filing:**

- Reward weights, PPO arch, YAML numbers, “boost κ.”
- Trend-adaptive **reward** weights (training-only; not onboard unless frozen into the policy).
- “Intelligent perturbation control” as a slogan without a plant that can **refuse** along-B torque.

Prior art a reviewer will cite: B-dot; aero-stabilized CubeSats; underactuated magnetic ADCS; disturbance-utilizing / “environment-aware” attitude papers; solar-sail reflectivity control; differential drag. Claims must be **narrow**: estimated environmental torque → per-axis magnetic authority floor → sailcraft in VLEO.

**Do not file on the current implementation.** A competent examiner (or a later invalidity search) will match sim vs hardware and find: true-state controller, dipole saturation absent, 1 cm torque used for sizing but not in the geom, sensors not in the loop, no ONNX / MCU path.

## 3.2 v8a / v8b is not the experiment the paper thinks it is

Intended: one-factor isolation of delegation.  
Actual differences vs v7 (both arms): new gains, new coils, quadratic power, KF, sensors object, stability terms, `dE_weight` 3.5 → 2.6, gate floor accidentally 0.15, no brownout.  
v8a vs v8b: **only** `delegation.enabled` in the **reward**. Action still gates. If v8b’s extra terms are ~0 (B1–B3), the two arms are the **same noisy 7-D policy class**. A null result is uninformative.

To isolate: v8a must **pin gates = 1** (4-D action or frozen last 3). Same plant, same gains, same power model. That is the only honest ablation.

v8 vs v7 is a **new spacecraft**, not a new reward.

## 3.3 Prototype gaps (why this would not fly)

| Flight need | V2.0 v8 |
|---|---|
| Attitude estimate from IMU + mag + sun | Controller on true MRP |
| Mag dipole `m × B` saturation | Per-axis τ clip |
| Mag contamination of MMC5983MA | Coupling matrix = 0; ferrite remanence omitted |
| Dual-mag vote | `read_mag` never called |
| GNSS + TTFF | Truth `r, v` in obs |
| Rate / bias Kalman | 300 s gyro sample, 0.5 dps ZRO |
| MTQ driver I²R + PWM | Mean power from end-of-step B |
| Mass / inertia / cg | 0.625 vs 0.750 open; 161 g coils not in inertia; mass randomized in training |
| 20 cm rods + 30-turn sail loop | Paper design, no CAD / fit / deployment |
| Actor on MCU | No ONNX / INT8 path (APPROVAL_PLAN §3) |
| WMM / IGRF onboard | Gym dipole only |
| Safe mode | Brownout code not entered for v8 |
| TVAC / Allan / coil dipole | YAML `[MEASURE]` / `[ASSUME]` not a test plan with hardware IDs |

**Mass is a prototype stopper, not a comment.** YAML still says 0.625 vs 750 g is an open question, then adds 161 g of coils (25.7 % of 625 g). Inertia is still the old 0.625 kg tensor. At 750 g they already noted catalogue dipoles miss the torque; custom dipoles were sized at 625 g and **1 cm cp**. None of this is a frozen flight mass budget.

**1 cm cp–cm as a “balance requirement”** is the right **engineering** instinct (torque ∝ offset, power ∝ offset²) and the wrong **simulation** input. For a prototype: measure cp–cm on the real stack (or CAD), put **that** offset in the geom, **then** size coils. Doing it in reverse invented a 38 µN·m actuator against a ~30 nN·m plant.

**Z sail loop:** 90 m of 0.2 mm Cu around a 0.65 m² deployable membrane is a mechanism program, not a YAML block. If the prototype is Earth Cup / SolarCat, this has to be a separate deployable harness with EMI and mag-calibration procedures.

**Ferrite in LEO:** temperature, shock, residual moment. CubeSpace-class rods exist because they are **small**; 20 cm × 2 on a 625 g vehicle is a different spacecraft.

## 3.4 Usable data — what “usable” has to mean

For a thesis / paper / filing, a v8 run is usable only if you can show, on the **same** geom and mass:

1. Plant `τ_aero` time series vs KF `τ_env` (r, lag, storm step).
2. Per-axis `s`, `gate`, `τ_want`, `τ_ctrl`, `τ_achievable` (from `power.achievable_torque`).
3. Electrical W from `power.py` vs legacy linear (and vs battery ΔE).
4. v8a **gates pinned** vs v8b, same seeds, MPC / heuristic baselines on the **fixed** plant.
5. Whether `|B|` and `m × B` shortfall correlate with delegation (the actual physical story).
6. Frozen `snapshot.json` of every YAML.

None of (1)–(6) are produced by the current trainer or `info` dict.

Until B1–B6 and that log are done, **do not spend 32-env 300k steps.** A 1-episode burn-in with printed `reward_parts` will already show `deleg_power == 0`.

## 3.5 Patent / prototype recommendation

- Treat v8 as **research software for a mechanism**, not a vehicle PDR.
- Freeze **one** mass, **one** geom (with measured or CAD cp–cm), **one** actuator (catalogue **or** custom, not both in different files).
- Put **dipole-space saturation in C++** before any “we harvest aero torque because MTQs cannot” claim. That is the physical content. Reward-only gating of an optimistic plant is not.
- Put **sensors in the observation and an attitude filter in the loop** before any “onboard” language.
- Export ONNX. A patent that cannot run on the MCU is a paper.
- Keep v7 bake-off numbers **out** of v8 tables unless the plant, mass, coils, and power law are identical.

---

# What is actually in good shape (do not rip out)

- Delegation **split → gate** map (eq. 1) is clear and invertible; opposite sense vs v7 is documented.
- `config.py` load / merge / snapshot design is the right provenance fix — **if training calls it**.
- `sensors.py` + `tests/sensors/test_sensors.py` (Allan, dual-mag independence, digitize) is the only new module that meets VALIDATION_STANDARD.
- Propagator J2 + co-rotating wind signs look correct; float32 as a fidelity statement is honest.
- `power.dipole_for_torque` / `achievable_torque` are the right diagnostics — they must **drive the plant**, or at least be logged, not sit unused.
- One-sided `decay_stability` (don’t punish recovery / orbit-periodic dE) is the right shape.
- Boost only on **positive** mission terms is the right sign discipline.
- C++ `quat_to_mrp` matches Python `q_vec / (1+q0)`; `mrp_error` composition has the identity `s ⊕ (−s) = 0`.
- v8+ imports fail at import time — good. v3–v7 paths are intended to stay byte-identical.

---

# Fix order for Claude Code (do not train until 1–8 are green)

1. **Pass `tau_ctrl_mean` (and `tau_want`) into `compose_sc_v8`.** Add a unit test: with gates = 0.3 and saturating PD, `P > 0`; with unsaturated PD, `P == 0`.
2. **Rewrite eq. (4)** using `τ_want` vs applied vs `gate · τ_max`. Stop using `e_act / gate`.
3. **Rewrite eq. (2)** as a **direction** alignment (and/or compare `τ_env` to the **error-axis** unit vector), with a floor in **angle**, not in `|τ_env| / |τ_PD|`. Document the ~6°/step aero envelope.
4. **`gate_floor` default `None` for v8+** so YAML 0.3 applies; keep 0.15 only if that is a deliberate v7 default.
5. **v8a: pin gates at 1.0** (drop or ignore action[4:7]). v8b: live split. Same everything else.
6. **Include `v8a/v8b/v9a/v9b` in brownout** the same way as v7, or explicitly document “no brownout” and drop those reward terms.
7. **One vehicle card:** mass, inertia, geom, cp–cm, dipoles, `P_peak`, `τ_max`. Make `docs/14`, `power_mtq.yaml`, `power.py`, `env.py` comments, and `gains_mrp.yaml` match. Resolve 0.625 vs 0.750 **before** any run. Do not randomize mass for the bake-off.
8. **`train.py` + `config.snapshot` + eval / probe reuse from `campaign.py`.** Log `deleg_B/P/S`, split, gates, `tau_want`, `tau_env`, `tau_aero`, `tau_ctrl`, MTQ watts, shortfall from `achievable_torque`.
9. Tests per VALIDATION_STANDARD: `tests/reward/test_v8.py`, `tests/estimators/`, `tests/power/`, `tests/propagator/`, `tests/duo/`, `tests/integration/test_env_v8.py` (obs dims 35/47/59, yaml `gate_floor`, brownout, P not identically 0). Classroom numerics in spec 14. **Citations** (Schaub MRP, IEEE 952 already in sensors, Vallado J2, a mag-coil / `m = NIA` reference, a disturbance-observer paper).
10. **v9:** one 4-vector layout; 150 s history that is actually sampled (store at 150 s or drop the offset); SoC future from a real power step; stop using `round(0.5) → 0`.
11. **v8Duo:** do not train until env calls it, `bc_scale` comes from plant Cd(q), SoC / eclipse used, and the arbiter sees two different decays.
12. **Plots (minimum set, after a burn-in, not after 300k):** (a) KF vs `tau_aero`; (b) `s` / gates / B / P; (c) v8a-pinned vs v8b vs MPC on decay / SoC / downlink / MTQ joules; (d) `τ_achievable / τ_cmd` vs true anomaly; (e) learning curves with `reward_parts` stacked. Reuse `plot_production.py` structure, new `outputs/analysis/v8/` stem.
13. Only then: dipole-space saturation in C++ (separate decision — invalidates v7 comparison; required for prototype claims).
14. Only then: sensors into `_obs` + attitude filter (separate decision).

**Burn-in gate (must print, not “look at TensorBoard”):** one episode v8b, report mean `deleg_P`, `deleg_B`, `tau_aero` vs `tau_want` ratio, brownout count, `gate_floor` actually used, mass actually used. If `deleg_P` is still ~0, **stop**.

---

## Files this review is about (not an edit list)

```
docs/modules/14_reward_v8.md
python/configs/{reward_v8a,reward_v8b,reward_v9,gains_mrp,estimator_kf,power_mtq,duo,sensors_solarcat}.yaml
python/arlamx_v2/{env,reward,reward_v8,estimators,power,sensors,propagator,duo,config,train,campaign,plot_production}.py
cpp/src/{api.cpp,control/mrp_feedback.cpp,attitude/mrp.cpp,orbit/frames.cpp}
tests/sensors/test_sensors.py
tests/test_v7_ipc.py
```

`ARLAMX-V1.7/` was not touched and should not be touched by the follow-up fix pass.
