# 03 — The C++ plant: physics models

One fused advisor step (`Simulator::step(q_cmd)`, `cpp/src/api.cpp`):
slerp-clip the commanded quaternion (≤ `max_slew_rad`, 40° for v4+) → MRP
target → `n_substeps = advisor_step_s/dt_s` (300/2 = 150) of: MSIS state held,
Sentman/CLL panel force+torque, panel SRP + Earth-radiation force and torque, gravity-gradient torque, gravity at
each RK4 stage, control torque, RK4 on the 12-state
[r, v, σ, ω]. Body loads are zero-order-hold across a substep. Returns the
accumulators listed in `docs/modules/11_plant_api.md`.

Frames: N = J2000 inertial, B = body, E = Earth-fixed (GMST rotation only, no
precession/nutation). Units SI. Detail per module lives in `docs/modules/`
with citations; this file is the condensed, verified version.

## Free-molecular aero — Sentman with M-01 closure (`aero/sentman.cpp`, module 01)

s = |v_rel| / √(2 k_B T / m̄); γ = s cos θ, Z = 1 + erf γ, E = e^{−γ²}.

    C_p,i = [(γ² + ½) Z + γE/√π] / s²
    C_τ,i = sin θ (γZ + E/√π) / s
    T_r/T_i = α_E T_w/T_i + (1 − α_E) s²/2          (M-01, Moe & Moe 2005; NOT 1+α(Tw/Ti−1))
    C_p,r = √(T_r/T_i) (E + √π γ Z) / (2 s²)
    F = A q (−C_p n̂ + C_τ t̂),  q = ½ ρ |v_rel|²,  τ = r_c × F

Defaults: α_E 0.93, T_w 300 K, A_ref = half the total panel area
(`one_sided_ref`). Every face with γ > −4 is summed (v2.1 fix): the exactly
edge-on face keeps C_τ = 1/(s√π) ≈ 0.07 at s ≈ 8. Face-on plate Cd ≈ 2.1–2.4;
hex sail edge-on Cd 0.214 (v2.0 had 0.079). Relative wind v_rel = v − ω⊕×r
(corotating, default on); energy split F_drag ∥ v_rel, F_lift ⊥ v_rel, each
dotted with inertial v.

Verified: `tests/aero/test_sentman.py`; `verify_physics.py` residual 3e-16
against textbook cos θ form and direct Maxwellian quadrature.

## Walker–CLL option (`aero/cll.cpp`, module 17, v2.6)

ADBSat (Sinpetru et al. 2021, arXiv:2104.05543) transcription of Walker,
Mehta & Koller 2014 eqs. 11–15, species-mixed with MSIS mole fractions
χ_j (He, O, N2, O2, Ar, H, N), per-species s_j, Cp mixed by χ_j m_j / M.
Live knobs α_N (normal energy accommodation) and σ_T. α_N = 1 takes the
Schaaf–Chambre branch = Sentman with α_E = 1. Known: the published closed form
has a kink as α_N → 1 (limit ≈ 2.3× the diffuse term); do not blend.
Face-on plate, s = 8, T_w/T_i = 1/3: Sentman 0.93 → Cd 2.37; CLL 0.93 → 2.56;
CLL 1.0 → 2.15. `high` preset's α_N = 0.93 is an *isolation choice* (same
numeral as α_E), not a literature value. Verified 2026-09-13 line by line
against the ADBSat PDF; Walker He/H rows not independently checked.

## Panel SRP (`srp/panel_srp.cpp`, module 02)

Default since v2.7: optical plate law (`al_mylar`), F_i = −P A cosθ[(c_a+c_d)ŝ + (2c_s cosθ + ⅔c_d)n̂],
P = P_SR (1 AU/d)² · srp_scale. Cannonball (C_r, force along −ŝ) remains selectable. Force **and**
moment Σ r_c × F_i enter the plant (`tau_srp_mean`). Earth IR (e₀/4 (R/r)², on in eclipse) and albedo
(a₀ (R/r)² max(0, r̂·ŝ), day side) are summed per plate with the same law (`tau_erp_mean`). Gravity
gradient τ = 3μ/r³ û × Iû (`tau_gg_mean`); `tau_env_mean` = aero + SRP + Earth radiation + gravity
gradient. Face-on hex, mylar, 1 AU: 5.69 µN. At 500 km quiet, SRP ~ aero (5.4 vs 15.7 µN); at
300 km storm, aero ≈ 450× SRP (`outputs/.old/v14/FORCE_RANGES.md`). Pre-v2.7 campaigns: cannonball, no
moment, no Earth radiation, no gravity gradient.

## Gravity (`orbit/gravity.cpp`, module 03)

GGM03S fully-normalised Stokes coefficients, fully-normalised Legendre recursion
with a pole guard (zonal-only derivative within cos φ < 1e-14); not Pines.
`SH_MAX_DEGREE` 70; degree < 2 → point mass with the GGM μ. If no file is
loaded, the fallback is two-body + J2 + J3 (V1.7 minimum). Verified against
scipy `lpmv` gradients to 1e-10 and against Basilisk `computeField` to ~1e-16
off-axis. Lunisolar third body from analytic Sun/Moon (Montenbruck & Gill,
Meeus) or CSPICE when linked; SC campaigns leave it off.

## Atmosphere (`python/arlamx_v2/atmosphere.py`, module 05)

pymsis (MSIS 2.1 or NRLMSISE-00) queried at geodetic height and Earth-fixed
longitude, once per advisor step (300 s hold — a known simplification). Returns
ρ, T, m̄ = Σ n_i m_i / Σ n_i over seven species (anomalous O folded into O, NO
dropped), and χ for CLL. F10.7 = F10.7a = daily value; Ap constant across the
7-slot array. Guard: MSIS returns finite garbage at F10.7 ≈ 5 (T ~ 1e26 K);
the env bounds ρ < 1e-8, 150 < T < 4000 K, 8e-28 < m̄ < 1e-25 or falls back
to ρ = 1e-13, T = 600 K.

## Attitude dynamics (`attitude/`, module 06)

MRP σ with shadow switch at |σ| > 1; σ̇ = ¼ B(σ) ω; ω̇ = I⁻¹(τ − ω × Iω).
I = diag(0.0125, 0.0125, 0.025) kg m². RK4 at h = min(rk4_step_s, dt_s) = 2 s.
Torque-free body conserves |H| and T to 1e-14 over 3000 s at 0.12°/s; at
5°/s (detumble) |H| drifts 1.3e-5.

## Magnetic field (`mag/field.cpp`, module 09)

Tilted dipole (IGRF/WMM epoch dipole terms) or WMM2020 from `WMM.COF`
(Schmidt recursion, secular terms, decimal year tracked from `epoch_jd`).
|B| 20–50 µT at 400 km. Verified by −∇V finite difference to 1e-9. Not a
bit-match to NOAA reference code.

## Onboard FP32 propagator (`onboard/propagator_f32.cpp`)

Two-body + J2 + exponential drag in IEEE binary32, operation-for-operation
mirror of `propagator.py`. Feeds the v9+ "future" observation block. < 5 m
over 900 s, < 30 m over 6 h vs the numpy reference.

## Energy bookkeeping (v2.1)

`dE_actual`, `dE_drag`, `dE_lift`, `dE_baseline` are all F·v Δt/m on the
inertial velocity. The min-drag counterfactual (`coefficients_only` with an
axis-aligned freestream on the smallest-projected-area body axis) is evaluated
once per advisor step; its work is −F_base (v̂_rel·v). v2.0 charged
−F_base |v_rel| (6 % bias against the true min-drag attitude).

## Rotational identity (v2.6)

    I(ω₁ − ω₀)/T = τ̄_aero + τ̄_ctrl − ⟨ω × Iω⟩,   T = n_substeps Δt

`gyro_mean` is accumulated with the same RK4 stage states and Simpson
weights as the ω update, so the identity holds to 1e-15 at any rate. The
flight-side observer uses it (`tests/integration/test_plant_step.py`).

## Numbers a reviewer will ask for (`outputs/v14/FORCE_RANGES.md`)

Hex geometry: 72 plates, 1.38144 m² total, 0.65472 m² face-on, cp–cm 1 cm
(v8+ only), 0.625 kg. Aero broadside |F|: 15.7 µN (500 km quiet) → 694 µN
(300 km quiet) → 2.43 mN (300 km, F10.7 250, Ap 180); |τ| up to 24.3 µN·m.
Edge-on quiet 500 km: 0.49 µN.
