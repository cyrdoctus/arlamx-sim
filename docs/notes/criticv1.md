# Critic v1 — plant physics audit (2026-09-25)

Audience: the next agent. Read this file and fix the items in **Fix these**.
Do not retune rewards, do not rewrite the training loop, and do not "clean up"
code that the **Leave these alone** section says is already correct.

Mesh used for every number below: `data/earthcup_hex_v3.geom`, 72 panels,
1.38144 m² total, normals are unit length, 94.8 % of the area has `|n_z| > 0.9`.
The v8+ plant shifts centroids by `cp_offset_m = [0.01, 0, 0]`
(`config/plant/power_mtq.yaml`). Inertia stays
`diag(0.0125, 0.0125, 0.025)` about the origin. That shift is a pure
center-of-pressure offset: face-on torque equals `|F| × 1 cm`. Keep it.

A 300 s advisor step is 150 substeps of `dt_s = 2`. Measured on this tree
(Sentman, GGM loaded, hex, 500 km):

| Configuration | Time per advisor step |
|---|---|
| Gravity degree 2 | 0.48 ms |
| Degree 4 (default in `config/plant/physics.yaml`) | 0.57 ms |
| Degree 8 | 0.56 ms |
| Degree 20 | 0.82 ms |
| 150 `spacecraft_aero` calls from Python | 0.65 ms |

The panel loop is the plant. Gravity degree is not.

---

## Fix these

Do them in this order. Each item has a classroom check. Run that check.
Update the module doc in the same change as the code. A doc that still
describes the old equation is a failed fix.

### F1 — Apply the solar-pressure torque the kernel already computes

`cpp/src/srp/panel_srp.cpp` returns `(F, τ)` with `τ = Σ r_c × F_i`.
`Simulator::step` in `cpp/src/api.cpp` (about lines 274–285 and 321–322)
keeps `.first` and sets

```text
F_B   = aero.F + Fsrp
tau_B = aero.tau + tau_ctrl
```

The Python binding `panel_srp_optical` in `cpp/src/bindings.cpp` also
returns `.first` only.

With the 1 cm offset, face-on hex cannonball force is 5.374 μN, so the
dropped moment is 54 nN·m. Documented quiet-500 km broadside aero is
15.7 μN, which is 157 nN·m about the same offset. The missing moment is
about a third of the disturbance the coils are sized for, and it remains
when the sail is edge-on to the air (measured aero torque on this hex at
a pure +X freestream is 0).

Fix:

- Add `τ_srp` into `tau_B` on both the cannonball and the optical branch.
- Expose torque from the optical binding the same way the force is exposed,
  or stop dropping it. Callers that only want the force can ignore the
  second return. Do not leave the binding as a silent truncation.
- Prescribed-attitude mode still zeros `ω` and holds `σ`. SRP torque must
  not be allowed to integrate the attitude in that mode. Apply it on the
  `point` and `detumble` paths, where aero torque is already applied.
- `tau_aero_mean` currently accumulates `aero.tau` only. Add the SRP moment
  into the disturbance accumulator the observer and the reward read, or add
  a separate `tau_srp_mean` and thread it through. Do not hide SRP torque
  inside `tau_ctrl`.

Check: one plate, area 1 m², centroid `(0.01, 0, 0)`, Sun along `+z`,
normal `+z`, `C_r = 1`. Force magnitude `P_SR`, torque magnitude
`0.01 × P_SR`, direction `r × F`. A centered plate (centroid 0) has torque 0.
Existing `tests/srp/test_srp.py` must still pass.

### F2 — Default the sail to the optical plate law

`config/plant/physics.yaml` (and `physics_fast.yaml`, `physics_high.yaml`)
set `srp.optical: false` and `Cr: 1.8`. The cannonball force is entirely
along `-ŝ`. A reflective sail's force is along the plate normal. The
optical law in `panel_srp.cpp` already matches Montenbruck & Gill §3.4:

```text
F_i = -P A cosθ [ (c_a + c_d) ŝ + (2 c_s cosθ + (2/3) c_d) n̂ ]
```

with `cosθ = n̂·ŝ > 0`. Verified by expanding that expression by hand
against the code at 45°.

Measured, `P = 4.56e-6`, aluminized-mylar partition from
`python/arlamx_v2/sail_optics.py` (`ca, cs, cd = 0.08, 0.88, 0.04`):

| Case | Cannonball | Optical | Angle between them |
|---|---|---|---|
| 1 m² plate, Sun 45° off the normal | 5.80 μN along `-ŝ` | 4.38 μN | 41.4° |
| Hex, same geometry | 3.94 μN | 2.88 μN | 40.1° |
| Hex, face-on | 5.374 μN | 5.692 μN | 0° |

Face-on magnitude was never the bug. The normal component is the
orbit-raising term, and cannonball deletes it at every attitude except
face-on.

Fix:

- Set the three physics presets to `optical: true` with the `al_mylar`
  partition, unless a cited coating for this vehicle says otherwise.
  `sail_optics.py` is the partition table. Do not invent a new partition.
- Keep the cannonball function. Tests that pin `C_r = 1` face-on
  (`|F| = P_SR`) stay on that function.
- Scale the pressure by heliocentric distance. `Ephemeris::sun_pos`
  (`cpp/src/orbit/third_body.cpp`) returns `sun_unit_analytic(jd) * AU`
  for the analytic path, so `|r_sun|` is identically 1 AU and the ±1.7 %
  annual distance (about ±3.4 % in flux) never appears. `P_SRP_1AU` in
  `cpp/include/arlamx/constants.hpp` is `4.56e-6` (the 1367 W/m² scale).
  Use `P = P_SRP_1AU * (AU / |r_sun|)² * srp_scale`. When CSPICE is
  linked, `|r_sun|` is already real. For the analytic ephemeris, put a
  one-line Earth–Sun radius on `sun_unit_analytic` (Vallado low-precision
  radius, the same family as the longitude already in `frames.cpp`) and
  use it. Do not leave the unit vector and a hard-coded 1 AU.
- Update `docs/modules/02_srp.md` and `docs/handoff/03_plant_physics.md`.
  They currently say cannonball is the default and that the plant drops
  the moment.

Check: face-on hex, optical mylar, eclipse 1, `srp_scale = 1`, Sun at
exactly 1 AU, reproduces 5.692 μN within 1e-6 relative. The same case at
`|r_sun| = AU * 0.983` scales the force by `1/0.983²`. A 90° plate
(Sun in the plane) is identically 0. Cannonball `C_r = 1`, face-on,
1 m², stays at `P_SRP_1AU`.

This changes the science plant relative to every campaign trained on
cannonball. Say so in the changelog. Do not rewrite old `snapshot.json`
files.

### F3 — Eclipse must not zero the sunlit power used by the forecast

`ArlamxV2Env._update_power` (`python/arlamx_v2/env.py`, about 597–622)
sets illumination to 0 whenever `eclipse <= 0.5`, then returns
`gen_n = p_gen / panel_peak`, which is 0 in shadow. `_future_block`
(about 548–558) does

```text
p_gen_now = gen_n * panel_peak
gen = p_gen_now if future_point_sunlit else 0
```

A step that starts in eclipse therefore forecasts zero generation for
every later sunlit point. `sun_B` is still computed in eclipse
(`StepOut.sun_B`). The body–Sun angle of an inertially held attitude does
not depend on the shadow flag.

Fix:

- Bill the battery with eclipse-gated power, as now.
- Pass the geometric illumination `abs(sun_B[2])` (or the corrected
  one-sided value from F4) into `_future_block` separately from `gen_n`.
- Future sunlit slots use that geometric illumination. Future shadowed
  slots stay 0. A step that is in eclipse now and sunlit 600 s later must
  show positive generation if `|sun_B[2]| > 0`.

Check: construct a state in cylindrical eclipse with `sun_B = (0, 0, 1)`.
Instantaneous generation is 0. The +600 s sample, placed on the sunlit
side of the same Sun vector, has generation `panel_peak * 1`.

### F4 — Solar-array illumination is `|n̂_z · ŝ|` on both faces

`_update_power` uses `illum = abs(sun_b[2])` for a sunlit vehicle and
`0.25` while tumbling. The force model uses the panel mesh. Power does
not. `abs` makes the back of the membrane a full array (peak
`0.78 * 0.85 W` from `env.py` line 290; the yaml is
`config/plant/power_mtq.yaml` `generation`).

`ARLAMX_V2.0_OVERVIEW_AND_CPP_PLAN.md` describes an older CatSat one-sided
+Y wing. This vehicle is the hex. Do not copy that sign.

Fix:

- Find which membrane face carries cells (assembly notes, STL name, or
  the user). If the cells are on one side, use `max(± sun_B[2], 0)` for
  that side, not `abs`.
- If nothing in the repo states the face, stop and ask. Do not guess.
  Leaving `abs` in place is acceptable until the face is known. F3 does
  not depend on this choice.
- Do not build a second mesh illumination model that disagrees with the
  panel normals. If you sum panel contributions, use the same normals and
  areas the plant uses, and only the panels that actually carry cells.

### F5 — Gravity-gradient torque

Not in `tau_B`. For a diagonal inertia and a unit nadir vector `û` in the
body frame (`û = -C r̂_N`),

```text
τ_gg = (3 μ / r³) (û × I û)
```

`μ` is the same gravitational parameter the translational step uses
(GGM header `μ` when the file is loaded, else `MU_WGS`). Peak magnitude
on this inertia at 400–500 km is about 24 nN·m
(`3μ/r³ ≈ 3.8e-6 s⁻²`, `|û × I û|_max = ½|I_zz − I_xx| = 0.00625`).
That is ~15 % of the quiet broadside aero moment (157 nN·m) and it is the
largest steady moment in the edge-on attitude.

Fix: add `τ_gg` inside the substep, in the body frame, next to aero and
SRP torque. Recompute it when `C` or `r` changes (once per substep is
enough; it does not need a per-stage panel loop). Include it in the
disturbance accumulator, same rule as F1. Prescribed mode holds attitude;
do not let `τ_gg` spin the body there.

Check: principal-axis inertia, `r` along body X, `τ_gg = 0` by symmetry.
`r` at 45° in the X–Z plane, `|τ| = (3μ/r³)(½)|I_zz − I_xx|` to 1e-12.
Torque-free translation (no aero, no SRP) still conserves orbital angular
momentum direction to the existing gravity tolerance. Do not add this
torque to the onboard FP32 propagator.

### F6 — Earth infrared and albedo, as a plate sum, not a second cannonball

Cylindrical eclipse currently zeros all radiation pressure
(`eclipse_cylindrical` in `cpp/src/orbit/frames.cpp`, consumed in
`api.cpp`). Earth IR does not turn off in eclipse. Order of magnitude:
outgoing longwave ~240 W/m² at the surface, `F/c` at 400 km on a
nadir-facing plate is on the order of 15 % of `P_SRP_1AU`. Albedo is a
smaller day-side term and the photons come from the Earth, not from the
Sun.

Fix, in the SRP module, using the optical plate law already in the code
(same `c_a, c_s, c_d` as solar, unless a cited IR emissivity/reflectivity
is different — solar-band `c_s = 0.88` is wrong in the thermal IR; use a
stated thermal absorptance/emittance and document the number):

- Earth IR source direction from the spacecraft toward the Earth is
  `-r̂`. Flux `(R_E / r)² σ T_E⁴ / c` with `T_E` and `R_E` named constants
  and a citation (Vallado Earth-radiation section or Knocke). Sum over
  panels with `n̂ · (-r̂) > 0`. This term is independent of solar eclipse.
- Albedo: same geometry, flux `a_E (R_E / r)² P_solar` times a day-side
  visibility factor (zero in the cylindrical umbra, full when the
  spacecraft is over the sunlit hemisphere and the Earth disk is lit).
  Cite the factor. Do not turn albedo on in eclipse.
- Both forces and both torques enter `F_B` and `tau_B` under the same
  rules as F1.
- A switch, default on for the science presets, so a test can run the
  pure-solar case.

Check: eclipse, nadir-facing 1 m² plate, IR force is nonzero and along
the outward radial direction (the plate is pushed away from the Earth).
Sunlit face-on solar case of F2 is unchanged when the Earth terms are
subtracted back out. A plate facing deep space (`n̂` along `+r̂`, Sun
elsewhere) gets zero Earth IR.

If the thermal optical coefficients are not in the repo, pick a cited
pair (for example a high-IR-emittance polymer back, `ε ≈ 0.7` to `0.9`,
specular fraction ~0 in the thermal band), write the citation next to the
constant, and use it. Do not reuse the solar mylar `(0.08, 0.88, 0.04)`
for IR.

### F7 — Thermospheric wind is a velocity, not a new density

Co-rotation is already correct:

```text
v_rel = v − ω_⊕ × r
ω_⊕ = (0, 0, 7.2921150e-5)
```

At 500 km that cross product is 502 m/s. Horizontal winds of a few
hundred m/s are another few percent of orbital speed and about 5–10 % on
dynamic pressure, and they rotate lift.

Do not invent a climatology and do not call MSIS more often to fake wind.
MSIS has no wind.

Fix:

- Write the relative velocity as `v_rel = v − ω_⊕ × r − v_wind_N`.
- Default `v_wind_N = 0`, so every current test is unchanged.
- If a horizontal-wind model is added, it has to be a cited model (HWM14
  or a successor) evaluated in ECEF and rotated with the same `dcm_EN`
  the magnetic field uses. One sample per advisor step, held like density.
- No HWM dependency in the tree means the fix is the hook plus a test, and
  a sentence in `docs/modules/05_atmosphere.md` that the wind is zero
  until a model is supplied. Do not ship a constant "200 m/s eastward".

Check: `v_wind = 0` reproduces today's `v_rel` to 1e-12. A uniform
`v_wind_N = (100, 0, 0)` changes `v_rel` by exactly that vector and
changes the body-frame gas direction `v_B_gas = −C v_rel` by the same
rotation. Drag on a face-on plate increases when the wind opposes `v`.

### F8 — Onboard forecast is missing the non-gravitational force it just measured

`python/arlamx_v2/propagator.py` and
`cpp/src/onboard/propagator_f32.cpp` propagate two-body + J2 + exponential
drag. The ballistic coefficient comes from `drag_N`, and `drag_N` in
`api.cpp` is the substep average of `|F_aero|`, not the along-track
component (`drag_sum += norm(aero.F)`).

On this hex the `|F| / F_along` inflation peaks at 1.4 % (80° pitch) and
is 1.000 at face-on and edge-on. Do not spend the fix on that 1.4 %. Use
the along-track component anyway (`|F| * |C_D| / hypot(C_D, C_L)`, which
the step output already supports via `Cd` and `Cl`), so a later geometry
with real lift does not inherit the shortcut.

The force that matters is SRP. At quiet 500 km the plant's own numbers are
5.4 μN of SRP against 15.7 μN of aero, and SRP can raise the orbit. The
forecast has none.

The FP32 propagator is the flight-computer model. Do not port panel
Sentman, CLL, or degree-20 gravity into it.

Fix:

- Recover the along-track aero acceleration from the measured along-track
  aero force, as `bc_inverse_from_drag` already does, using `F_along`
  rather than `|F|`.
- Recover an along-track solar acceleration the same way from the SRP
  force the plant just applied (`F_srp · v̂`), and add it as a separate
  held acceleration. Eclipse along the propagated arc still uses
  `eclipse_cylindrical` with the Sun direction held for the horizon (the
  Sun moves ~0.01° in 600 s). Do not fold SRP into the ballistic
  coefficient: SRP does not scale with `ρ v²`.
- Keep the exponential atmosphere and the 60 km scale height. Those are
  the flight model on purpose.
- The "< 30 m over 6 h" test is agreement with this same model. Update it
  if the SRP term changes the float32 reference. Do not claim that number
  is agreement with the C++ plant.

Check: zero density, known SRP force along `-v`, propagated Δv over `T`
seconds equals `(F_srp / m) T` to float32 accuracy. Zero SRP reproduces
the previous drag-only trajectory.

### F9 — Geodetic height is what the atmosphere is, and several consumers still use `|r| − R_eq`

`query_msis` is called with Bowring height (`env.py` around 878–890:
`ecef_to_geodetic` on the inertial vector, which is valid because latitude
and height are invariant under the GMST rotation about Z, then
`lon_ecef = lon − gmst`). That part is correct. Keep it.

These use spherical altitude `( |r| − R_WGS ) / 1e3` instead:

- `StepOut.altitude_km` and the 80 km break in `Simulator::step`
- the 450 km solar-pressure flash cutoff in `env.py`
- `_future_block`, which anchors the exponential at `alt_km * 1e3` while
  `rho0` came from the geodetic query

The ellipsoid is ~21 km below `R_eq` at the pole. Against a 60 km scale
height that is a factor `exp(21/60) ≈ 1.4` in the forecast whenever the
label and the density disagree.

Fix:

- Report geodetic altitude from the same Bowring call the atmosphere uses.
  Keep `|r|` for gravity and for SMA.
- Anchor the onboard exponential at that geodetic height.
- The 80 km "inside the atmosphere" stop and the 450 km flash cutoff
  should use the same altitude the atmosphere uses. State the unit in the
  field name or the doc so the next reader does not add a second
  definition.
- Observation features that were trained on spherical altitude will shift
  by up to ~20 km. Note it. Do not silently rescale the reward weights to
  hide the shift.

Check: a point on the +Z axis at `|r| = R_eq + 400 km` has Bowring height
about `400 km + 21.4 km` (the polar flattening). `ecef_to_geodetic` already
implements this; the test is that `altitude_km` matches it, not a new
geodesy formula.

### F10 — Correct the CLL kink paragraph

`docs/modules/17_aero_cll.md` and the matching comment in
`cpp/include/arlamx/aero/cll.hpp` say the Walker branch does not approach
Schaaf–Chambre, that the limit is about 2.3× the diffuse term, and that a
sweep through `α_N = 1` jumps.

The 2.3× is real and it is only the re-emission prefactor
`(T_w/T_i)^δ ζ / s` for atomic oxygen. On the plate coefficient, face-on,
`s = 8`, `T_w/T_i = 1/3`, atomic O, measured from `cpp.cll`:

| α_N | C_p |
|---|---|
| 0.99 | 2.246 |
| 0.999 | 2.147 |
| 1 (Schaaf branch) | 2.144 |
| Walker limit as α_N → 1⁻ | 2.306 |

`α_N = 1` is bit-identical to Sentman `α_E = 1` (residual 0 at face-on,
5×10⁻¹⁷ at 69°). The 7.6 % gap between 2.306 and 2.144 opens only for
`1 − α_N ≲ 1e-5`. A sweep that hits 0.999 and then 1.0 does not jump.

Fix the paragraph so it states those three facts. Do not blend the two
branches and do not change `walker_fit`. He and H mid-band coefficients
are still only the ADBSat transcription; leave the "confirm against
Walker 2014 Table 2" note in place.

Also fix the Sentman module status line. `docs/modules/01_aero_sentman.md`
says **Status: spec only** while the code and the tests exist. Same for
any other module doc whose status line is stale and would send the next
reader looking for a missing implementation. Status only. Do not edit
equations in those files unless a fix above changed the equation.

### F11 — Ap history and the 300 s density hold

Leave the 300 s MSIS hold. Calling MSIS every 2 s substep would dominate
the 0.6 ms plant and the density does not need that cadence. The hold is
wrong across the terminator (tens of percent inside one advisor step).
The acceptable fix is one extra sample at the end of the step and a linear
interpolation in time of `ρ, T, m̄, χ` across the 150 substeps, still two
MSIS calls per decision, not 150. Do this only if F1–F9 are done. It is
not free and it is not the largest error.

`query_msis` fills all 7 Ap slots with one number
(`python/arlamx_v2/atmosphere.py`). That is the right behavior when the
scenario knob is "Ap = 180, steady". It is the wrong storm: MSIS wants
daily Ap, current 3-hour Ap, and the lagged averages. Fix by accepting an
optional length-7 history. When the caller passes a scalar, keep today's
fill and document it as a steady-Ap scenario. Do not change the meaning of
the existing `ap:` yaml key.

---

## Speed, after the physics

Do not do these instead of F1–F10. Do not do them in a way that changes
the equations above.

Measured fact: going from gravity degree 2 to degree 8 costs ~0.1 ms on a
0.57 ms step. Caching spherical harmonics, or evaluating degree ≥ 3 once
per substep, is not the win at the degrees the presets use. Degree 20
costs ~0.3 ms. Only touch the gravity schedule if a preset actually runs
at degree 20 and a profile says so.

Aero is already zero-order-held across the four Runge–Kutta stages.
Recomputing Sentman on every stage multiplies the plant by about four
(600 panel evaluations measured 2.6 ms from Python). Leave the hold.
During a 40° slew spread over 300 s the body moves ~0.3° per substep.
The hold is the coarse approximation only in detumble, at ~5°/s (10° of
attitude inside one 2 s step).

The one kernel change that preserves every sentman digit: for `γ > 6`,
`erf(γ) = 1` and `exp(−γ²) < 2e-16` in IEEE double. Substitute the limit
inside `sentman_cs` and inside the CLL `γ` loop. Lee faces are already
skipped at `γ ≤ −4`. Most windward faces of this sail at `s ∼ 8` qualify.
Check bit-identity against the current function for `γ ≤ 6`, and a
relative residual below 1e-15 for `γ > 6` on `C_p` and `C_τ`.

The 2 s inner step exists because `gains_mrp.yaml` is that fast
(`I / k_d` is a couple of seconds). A longer inner step needs softer
gains. The magnetorquer note in `docs/modules/15_control_magnetorquer.md`
already shows those stiff gains do not hold attitude once torque is
`m × B`. Softening the gains is a control change, not a plant-speed
patch. Do not lengthen `dt_s` in this pass.

When `gravity.lunisolar` is true (`physics_high.yaml`), Sun and Moon
positions are rebuilt on every Runge–Kutta stage inside
`accel_gravity_env`. Over 2 s neither body moves at the level this
ephemeris resolves. Compute them once per advisor step and reuse the
vectors. Analytic-only; do not change the third-body formula

```text
a = μ_k [ (r_k − r) / |r_k − r|³ − r_k / |r_k|³ ]
```

which matches a hand evaluation to 3×10⁻¹².

`pymsis` was not importable in the audit environment, so the MSIS call
was not timed. It is already once per advisor step. Leave that cadence
(see F11).

---

## Leave these alone

The following were re-derived or measured. A "cleanup" that changes them
is a regression. If a test fails after F1–F10, fix the caller, not these
expressions.

### Sentman (`cpp/src/aero/sentman.cpp`)

`γ = s cosθ`, `Z = 1 + erf γ`, `E = exp(−γ²)`, `s = |v| / sqrt(2 kT / m̄)`.

```text
C_p,i = [(γ² + 1/2) Z + γ E / √π] / s²
C_τ   = sinθ (γ Z + E / √π) / s
T_r / T_i = α_E (T_w / T_i) + (1 − α_E) s² / 2
C_p,r = √(T_r/T_i) (E + √π γ Z) / (2 s²)
```

This is the Moe & Moe energy-flux closure. Do not replace it with
`T_r/T_i = 1 + α_E (T_w/T_i − 1)`.

The plant passes the incoming gas velocity, `v_B_gas = −C v_rel`, not the
spacecraft velocity. Pressure `−C_p n̂` then points backward along the
flight direction. The unnormalized tangent `t = v̂_gas + cosθ n̂` has
length `sinθ`. Both signs are required.

Face-on, `s = 8`, `T_w/T_i = 1/3`, `α_E = 0.93`: `C_p = 2.369423`.
Edge-on shear at `s = 8`: `C_τ = 1/(s √π) = 0.07052`.
Hex, `v = 7500`, `T = 900`, `m̄ = 2.656e-26`, `ρ = 1.5e-12`: face-on
`C_D = 2.256`, edge-on `C_D = 0.217` (the write-up's 0.214 is the same
quantity at a slightly different `s`).

`A_ref = ½ Σ A` when `one_sided_ref` is true. It cancels in the min-drag
baseline `F_base = C_D,base q A_ref`, because `C_D` was divided by the
same `A_ref`. Do not "fix" the reference area.

The hyperthermal incident energy `s² kT` drops the thermal part of the
incident flux. At `s ≈ 8` and `α_E = 0.93` that is a fraction of a percent
on `C_D`. Leave it.

Lee-side cutoff `γ ≤ −4` on the Sentman path only. CLL cuts per species.
Leave that split: a mixture cutoff zeros atomic hydrogen while it still
carries flux.

### CLL (`cpp/src/aero/cll.cpp`)

Schaaf–Chambre at `α_N = 1` matches Sentman `α_E = 1` to roundoff.
Walker `α_N = 0.93`, same plate, gives `C_p = 2.558` against Sentman's
2.369. That gap is the kernel, and it is what the `high` preset is for.
`α_N = 0.93` is an isolation choice, not a published LEO accommodation.
Do not retune it.

Mass-fraction mix `Σ χ_j m_j C_j / Σ χ_j m_j` and per-species
`s_j = |v| / sqrt(2 kT / m_j)` are the ADBSat eq. 15 transcription.
Species order He, O, N2, O2, Ar, H, N. Argon reuses the N2 row. Masses
match `atmosphere.py` (2019 SI molar mass / `N_A`). Anomalous O is folded
into O. NO is omitted from `χ` and from `m̄`. Leave that until there is a
Walker row for NO.

### Gravity, frames, third body

Unnormalized associated Legendre in geocentric latitude, Kaula factor
`√[2(2n+1)(n−m)!/(n+m)!]`, acceleration `∇U` with the two-body term
separate. J2 and J3 closed forms match a finite-difference gradient of
`U = (μ/r) [−J_n (R/r)^n P_n(sinφ)]` to ~1e-9 (the finite-difference
floor) at `r = (4.5e6, 1.0e6, 5.0e6)`. Do not replace the evaluator with
Pines in this pass.

`accel_j3` is only on the no-file fallback. Campaigns load `GGM03S.txt`.
Do not remove the fallback and do not add J3 a second time on top of a
loaded field.

Third body as cited above, residual 3×10⁻¹². GMST-only Earth orientation
(no precession, nutation, or polar motion) is the documented frame. At
degree 4 it is below the drag uncertainty. Do not bolt on a full IAU
rotation in this pass.

Cylindrical eclipse geometry is fine for the solar shadow at LEO (the
umbra cone differs from a cylinder by a fraction of a kilometre). F6 adds
Earth IR beside it. Do not replace the solar eclipse test with a penumbra
model unless F6 needs a day-side factor, and then cite it.

### Attitude and magnetic actuation

MRP kinematics `σ̇ = (1/4) B(σ) ω`, shadow set `−σ/|σ|²` when `|σ| > 1`,
DCM as Schaub, Euler `ω̇ = I⁻¹ (τ − ω × Iω)`. Leave them.

Magnetorquer `m = (B × τ) / |B|²`, then `τ = m × B`, per-axis clip. This
is the minimum-norm dipole and it has no component along `B`. B-dot is
`m = −k dB/dt` in the body frame, first sample zero, spherical cap forced
to `1e300` so only `dipole_max_` binds.

Quaternion feedback uses the same numeric `k_p` as the MRP law. Because
`q_v ≈ θ/2` and `σ ≈ θ/4`, that gain is twice as stiff on the quaternion
path. `quat_feedback.hpp` already says so. Do not "correct" it by
inserting a factor of 1/2 without a gain retune, and this pass does not
retune gains.

`ideal_torque: false` is the packaged default and the campaign gains were
not tuned on it (`docs/modules/15_control_magnetorquer.md`). Do not flip
that flag to make F1 or F5 look smaller.

### Energy bookkeeping

`dE_*` are `F·v Δt / m` on the inertial velocity, evaluated at the start
of the substep. The baseline is `−F_base (v̂_rel · v)`, not `−F_base |v_rel|`.
The v2.0 form had a ~6 % bias. Do not put it back. The along-track / lift
split is a partition of `F_aero` parallel and perpendicular to `v_rel`,
then each piece is dotted with inertial `v`. The sum equals `F_aero · v`.

### WMM and the dipole

`cpp/src/mag/field.cpp`: Schmidt semi-normalized recursion, `B = −∇V`,
`nT → T` at the end, reference radius 6371.2 km. Dipole
`(g11, h11, g10) = (−1450.9, 4652.5, −29404.8) nT` at epoch 2020, and
`B = (a/r)³ [3 (d·r̂) r̂ − d]`. The sign (field into the Earth at the
north geographic pole) is required. The 2026-08-20 audit fixed an inverted
dipole; do not invert it again. Decimal year tracks `epoch_jd`. Leave the
secular terms.

### Min-drag axis

`min_drag_axis` scans +X, +Y, +Z projected area only, then
`coefficients_only` runs real Sentman on that axis. For this hex, +X is
the minimum (`C_D` 0.217 versus 2.26 face-on). A general mesh can have its
minimum off the body axes. Do not replace it with a numeric attitude
search in this pass.

---

## Files the fixes touch

| Item | Files |
|---|---|
| F1 torque | `cpp/src/api.cpp`, `cpp/src/bindings.cpp`, `cpp/include/arlamx/types.hpp` if a new accumulator is added, `tests/srp/`, `docs/modules/02_srp.md`, `docs/handoff/03_plant_physics.md` |
| F2 optical default and `1/r²` | `config/plant/physics.yaml`, `physics_fast.yaml`, `physics_high.yaml`, `cpp/src/orbit/frames.cpp` (`sun_unit_analytic` radius), `cpp/src/api.cpp` (pressure), `docs/modules/02_srp.md` |
| F3 forecast eclipse | `python/arlamx_v2/env.py` `_update_power`, `_future_block` |
| F4 cell face | `python/arlamx_v2/env.py` only after the face is known |
| F5 gravity gradient | `cpp/src/api.cpp`, `cpp/src/attitude/integrate.cpp` only if the torque is easier to put next to `omega_dot`; a free function next to the attitude code is fine. Test under `tests/attitude/` |
| F6 Earth radiation | `cpp/src/srp/panel_srp.cpp`, `cpp/include/arlamx/srp/panel_srp.hpp`, `cpp/src/api.cpp`, `docs/modules/02_srp.md` |
| F7 wind hook | `cpp/src/api.cpp` (the `v_rel` block), `docs/modules/05_atmosphere.md`, `python/arlamx_v2/atmosphere.py` only if a real model is wired |
| F8 forecast force | `python/arlamx_v2/propagator.py`, `cpp/src/onboard/propagator_f32.cpp`, `python/arlamx_v2/env.py` `_future_block`, `tests/propagator/` |
| F9 geodetic altitude | `cpp/src/api.cpp` `altitude_km`, `python/arlamx_v2/env.py` |
| F10 docs | `docs/modules/17_aero_cll.md`, `cpp/include/arlamx/aero/cll.hpp` comment, stale "spec only" status lines |
| F11 optional | `python/arlamx_v2/atmosphere.py`, `python/arlamx_v2/env.py` |

Rebuild the extension after any `cpp/` change (`build.sh` or the existing
CMake target). `python/arlamx_v2/arlamx_cpp*.so` is the module the tests
import. Run, at minimum:

```text
tests/srp/
tests/aero/
tests/attitude/
tests/orbit/
tests/propagator/
tests/integration/test_plant_step.py
tests/integration/test_lift_work.py
tests/physics/test_physics_wiring.py
tests/validation/verify_physics.py
```

`verify_physics.py` is the equation register (Sentman closed form,
Maxwellian quadrature, J2/J3 gradients). A failure there means an
expression in **Leave these alone** was edited. Revert that edit.

Do not retune `gains_mrp.yaml`, the reward yaml files, or
`ideal_torque`. Do not change Sentman, CLL, the Legendre recursion, the
MRP map, or the magnetorquer cross product as part of making a new test
pass.
