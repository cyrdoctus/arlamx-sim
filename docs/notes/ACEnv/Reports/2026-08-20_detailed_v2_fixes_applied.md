time: 2026-08-20T00:00:00Z
agent: claude
style: detailed

# V2.0 fixes applied — what changed, what it invalidates, what was deliberately left alone

**Follows:** `2026-08-19_detailed_v2_triple_audit.md` (three-pass audit) and the earlier grok verification set
(`verify_*.md`, `validate_pass*.md`, `2026-08-19_detailed_equation_register.md`).

**Brief:** fix the findings that are cheap and safe, without invalidating more data than necessary, without
altering functionality or the cleanliness of the tree. Everything below was applied to `ARLAMX V2.0/` only.
`ARLAMX-V1.7/` was not touched.

**Gate held throughout:** the full suite passed before (74) and after (83) every change, and the Basilisk
gravity referee (`tests/basilisk_ref/`, rel < 1e-8) passed at every step.

---

## 1. Equation fixes

### F-1 · WMM Schmidt recursion — CRITICAL, now correct to 0.0000 %

`cpp/src/mag/field.cpp:60-95`. The old code ran a Gauss-normalised recursion and then applied a per-element
scale factor to values that had already been scaled — a double normalisation its own comment half-admitted
("this recursion is approximate"). Replaced with the WMM/IGRF technical-report recursion:

```
P_n^n = sqrt((2n-1)/(2n)) sin(theta) P_{n-1}^{n-1}                              (n >= 2)
P_n^m = [(2n-1) cos(theta) P_{n-1}^m - sqrt((n-1)^2-m^2) P_{n-2}^m] / sqrt(n^2-m^2)
```

with the Schmidt `(2 - delta_m0)` factor carried entirely by the seeds, so no extra scaling is applied
anywhere. The derivative recursion follows by differentiating the same relation.

| | before | after |
|---|---|---|
| vs independent scipy-`lpmv` Schmidt synthesis, 8 random points at 450 km | 434 – 14 827 nT error (1.1 – 32.6 %) | **0.0000 %** |
| P_2^0 at 30 deg colatitude | +29 % (inflated by exactly sqrt(5/3)) | exact |
| P_6^1 | wrong sign | exact |

### F-2 · Tilted dipole — MAJOR, polarity was inverted and the tilt was missing

`cpp/src/mag/field.cpp:11-28`. The old model was `m = +7.94e22 z_hat` — an **untilted** dipole pointing the
wrong way, while `field.hpp` promised "tilted" and spec 09 asked for the IGRF g10/g11/h11 form. Replaced with
the true degree-1 truncation of IGRF-13 / WMM2020:

```
B = (a^3/r^3) [3 (d . r_hat) r_hat - d],   d = (g11, h11, g10),  g10 < 0
```

`g10 < 0` is the physical content: Earth's dipole moment points toward geographic **south**, which is why the
field at the magnetic equator points north. Verified after the change:

- equatorial field now points north (`B_z > 0`); field dips **inward** over the north pole (inclination +90 deg)
- dipole axis tilt **9.41 deg** (IGRF ~9.4 deg); was 0 deg
- `|B|` = 29 806 nT at the magnetic equator on r = a, 24.8 uT at 400 km, 23.8 uT at 500 km
- against the corrected full WMM over 200 random LEO points: median direction error 8 deg, median `|B|` ratio 1.01

### F-3 · Lunar ephemeris frame — MAJOR, ecliptic returned where equatorial was consumed

`cpp/src/orbit/frames.cpp:80-92`. `moon_analytic` built its unit vector straight from Meeus's **ecliptic**
(lon, lat) while `Ephemeris::moon_pos` and `accel_third_body` consume equatorial J2000 — and
`sun_unit_analytic` right above it *does* apply the obliquity rotation. Added the missing `R1(-eps)` with the
same obliquity expression the Sun uses.

| | before | after | truth |
|---|---|---|---|
| lunar declination range over 100 days | capped at +-5.3 deg | **-28.03 to +28.55 deg** | +-18 to 28 deg |
| geocentric distance range | — | 364 278 – 405 110 km | 363 300 – 405 500 km |

Latent in the shipped configuration (`lunisolar` defaults false, and CSPICE bypasses the analytic path), but any
lunisolar run without kernels was silently mis-pointing the Moon by 12 – 20 deg.

### F-4 · J3 restored in the no-GGM fallback — MINOR

`cpp/src/api.cpp:130-143`, `constants.hpp`. With no Stokes file the plant fell back to two-body + J2, while
V1.7's minimum force model was two-body + J2 + **J3**. `accel_j3` existed but was never called. Now wired, using
`J3_EGM = -2.53265649e-6` — the same constant V1.7 used (`orbit_match/config.py:52`). No effect on normal runs,
which load GGM03S (whose C30 already contains J3).

### F-5 · WMM epoch year no longer hardcoded — MINOR

`cpp/src/api.cpp`. Both field call sites passed a literal `2026.0` for the secular-variation year. Now derived
from `epoch_jd + t` via `decimal_year_from_jd`, so a run at any epoch gets the right secular terms.

### F-6 · B-dot differentiator reset on mode change — MINOR

`cpp/src/api.cpp:95-103`. `bdot_.reset()` only happened in `Simulator::reset`, so re-entering detumble mid-episode
(brownout) differenced against a stale `B_prev` from the previous detumble epoch and emitted one spurious,
saturation-clipped dipole command. `set_mode` now resets the differentiator whenever the mode actually changes.

---

## 2. Performance — 17 % faster plant, bit-identical physics

Measured on the 400 km hex, 300 s advisor step, 150 x 2 s substeps, degree-4 gravity:

| | before | after |
|---|---|---|
| plant advisor step, point mode | 0.536 ms | **0.443 ms** (-17.4 %) |
| detumble mode (B-dot live) | — | 0.493 ms |

Two changes, both of which leave every reported number bit-identical (Basilisk referee unchanged at 1e-8):

- **F-7 · Legendre workspace on the stack** (`cpp/src/orbit/gravity.cpp:116-131`, `gravity.hpp`). `accel_ecef`
  heap-allocated a nested `std::vector` on every call — ~4 200 allocations per advisor step, the single hottest
  allocation in the plant. Now a fixed stack buffer bounded by a new `SH_MAX_DEGREE = 24` (the unnormalised
  recursion is not trustworthy past ~20-25 anyway, and the plant uses degree 4). This was `validate_pass4.md`'s
  own top recommendation.
- **F-8 · Skip the magnetic chain when nothing reads it** (`cpp/src/api.cpp`). The per-substep `dcm_EN` + field +
  two rotations were consumed **only** by the detumble branch — the reported `B_B` is recomputed once after the
  substep loop. Now computed only in detumble mode. Pure deletion of dead work.
- **F-9 · `sqrt(PI)` hoisted** out of `sentman()` (`cpp/src/aero/sentman.cpp`), which runs ~20 000 times per
  advisor step. Bit-identical.

Deliberately **not** done: replacing the `acos -> cos/sin` round-trip in the Sentman panel loop. It would be
slightly faster and slightly more accurate, but it perturbs results at the last ulp, and the plant is not the
training bottleneck (SB3 and the vec-env IPC are — the plant is ~6 % of training wall time). Not worth spending
bit-reproducibility on.

---

## 3. Tests — the gates that missed these bugs are replaced

`tests/` went from 74 to 83 passing.

| test | what it now pins |
|---|---|
| `test_wmm_matches_independent_synthesis` | rebuilds the field from `scipy.special.lpmv` — a genuinely independent normalisation path — and requires < 1 nT. A repeat of the double-normalisation bug cannot pass. |
| `test_wmm_reference_points` | 5 pinned vectors at 450 km, so the gate survives without scipy |
| `test_wmm_exact_pole_is_finite` | the `Bphi / sin(theta)` 0/0 case on the spin axis |
| `test_wmm_and_dipole_agree_in_bulk` | statistical, over 200 points — deliberately not pointwise, because the South Atlantic Anomaly really is ~40 % off the dipole |
| `test_dipole_points_north_at_equator` | replaces the old `assert B[2] < 0`, which **enshrined the inverted sign** |
| `test_dipole_axis_tilt_matches_igrf` | that the dipole is actually tilted (the old one was not) |
| `test_dipole_inclination_sign_at_poles`, `test_dipole_magnitude_bands` | field dips inward at the north pole; 1/r^3 falloff |
| `test_moon_is_equatorial_not_ecliptic` | declination must exceed +-18 deg — impossible in the ecliptic frame |
| `test_moon_distance_range` | perigee/apogee brackets |

The old WMM gate was `10 uT < |B| < 80 uT` — an 8x band that could not detect a 33 % error, which is precisely
the error it was sitting next to.

Also tightened: `test_circular_sma_stable_one_period` from 50 m to **0.05 m** (measured capability is ~1.6e-7 m;
a 50 m band would pass a badly broken integrator).

---

## 4. Hygiene (no behaviour change)

- `accel_nongrav_env` renamed **`accel_gravity_env`** — it returns gravity + third body, so the old name was
  actively misleading in a physics file. Contained to `api.hpp` / `api.cpp`.
- `field.hpp` header comments corrected (the dipole really is tilted now; the WMM comment names the report
  recursion). The stale "|B|_eq = 30 103 nT" calibration note is gone — the constants give 29 806 nT.
- `env.py` MSIS failure path no longer swallows the exception silently: it still falls back to a fixed
  atmosphere so a run does not die, but warns once per process. Training on a silently-constant atmosphere would
  otherwise look like a healthy run.
- `README.md` test count corrected (was "35", actually 83).
- `gradient70._build_policy` accepts an optional `spec["root"]` so a caller can point at a run tree other than
  the tuning campaign's. Default path unchanged.
- New Python binding `cpp.moon_analytic(jd) -> (unit vector, distance)`, symmetric with the existing
  `sun_unit_analytic`, so the lunar path is testable from Python at all.

---

## 5. What this invalidates — measured, not guessed

The magnetic-field fixes change three of the 35 observation features (`bhat`), so **every policy trained before
this date is invalid**. That is not a theoretical concern; it was measured directly.

Re-scoring the previous campaign winner `s4_f0p3_th0p9_obs_sac` on the fixed plant:

| | recorded pre-fix | re-scored post-fix |
|---|---|---|
| gap index | 0.468 | **2.914** |
| decay, nominal | 12.85 km/d | 8.85 km/d |
| battery in band | 14.7 % | 4.6 % |
| brownouts | 0 | **13** |

The policy had learned against a field vector pointing the wrong way; inverting three of its inputs breaks it.
(Its decay actually improved — it is the power behaviour that collapses.)

**The plant itself is unchanged for anything that does not read the magnetic field.** Re-running the MPC
reference on the fixed plant reproduces the recorded values exactly:

| | recorded | post-fix |
|---|---|---|
| decay, nominal | 9.20 km/d | **9.20** |
| battery in band | 84.0 % | **84.0** |
| downlink | 23 min/d | **23.2** |

So the invalidation is precisely scoped: gravity, aero, SRP, attitude and control are untouched; only
magnetic-field-dependent artefacts are stale.

**Stale after this change:** `outputs/campaign/` (LEADERBOARD.md, ledger.csv, runs/) and every trained model
under `outputs/training/`, for any claim involving the magnetic observation. `outputs/campaign/mpc_reference.json`
is still valid. Nothing was deleted — the artefacts remain for provenance.

**Superseding runs:** `outputs/presentation/runs/` — PPO, SAC and TD3 retrained on the fixed plant at 50k and
250k steps, three seeds each (18 runs), on the identical stage-S4 configuration.

---

## 6. Deliberately NOT fixed — these are decisions, not defects

Each of these was flagged in the audit; each would change the science rather than correct an error, so they are
left for an explicit call.

| Finding | Why left alone |
|---|---|
| **MSIS cadence** — one query per 300 s advisor step vs V1.7's every 10 s | A real fidelity gap, and cheap to close (~0.6 ms/step, and the plant is not the bottleneck). But it changes drag physics for every run and would invalidate the entire training corpus a second time. This is the highest-value remaining fidelity buy-back and should be a deliberate, one-time decision. |
| **Sensor noise absent** | `APPROVAL_PLAN.md` §2/§10.3 require Gaussian sensor noise with the Earth Cup numbers; `_obs()` builds noiseless truth. Adding it is a research change (it will lower every score), not a bug fix. |
| **Bake-off contract deviations** (corotating ON, no rate caps on v3, 6 ground stations vs 392, reduced randomisation) | Either restore the contract flags for a bake-off run, or promote the deviations into `APPROVAL_PLAN.md` and re-frame the comparison. A documentation/scope decision. |
| **ONNX / actor export missing** | Contract-scoped and absent, so every trained policy is currently undeployable. Needs building, not fixing. |
| **Config/provenance layer** | Physics constants are hard-coded in `env.py` with no per-run snapshot. Restoring V1.7's frozen `model_config.yaml` is new work. |
| **Pines/Cunningham gravity** | Spec 03 asks for it; the guarded classic Legendre is validated against Basilisk at 1e-8 and the training envelope is 20-30 deg inclination. A speed/robustness upgrade, not an equation fix. |
| **`min_drag_axis` scans only +X/+Y/+Z** | Exact for the symmetric hex in use; wrong in general. Fixing it changes the counterfactual baseline and therefore the reward, so it must wait for a geometry that needs it. |
| **CatSat line, orbit_match/OD, LQR/bang-bang** | Out of scope by plan, or never declared. Scope decisions. |

---

## 7. Files touched

```
cpp/include/arlamx/constants.hpp        J3_EGM
cpp/include/arlamx/mag/field.hpp        header comments
cpp/include/arlamx/orbit/gravity.hpp    SH_MAX_DEGREE
cpp/include/arlamx/api.hpp              method rename
cpp/src/mag/field.cpp                   WMM recursion, tilted dipole
cpp/src/orbit/frames.cpp                lunar obliquity rotation
cpp/src/orbit/gravity.cpp               stack Legendre workspace, degree clamp
cpp/src/aero/sentman.cpp                sqrt(PI) hoist
cpp/src/api.cpp                         J3 fallback, WMM year, B-chain skip, bdot reset, rename
cpp/src/bindings.cpp                    moon_analytic binding
python/arlamx_v2/env.py                 MSIS fallback warns once
python/arlamx_v2/gradient70.py          optional spec["root"]
tests/mag/test_dipole.py                rewritten (8 tests)
tests/orbit/test_third_body.py          +3 lunar tests
tests/integration/test_plant_step.py    SMA tolerance 50 m -> 0.05 m
README.md                               test count
```

New, additive only: `python/arlamx_v2/bench_production.py`, `plot_production.py`, `anim_production.py`.
