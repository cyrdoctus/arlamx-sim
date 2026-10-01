time: 2026-08-19T01:00:00Z
agent: grok
style: detailed

# V2.0 equation register — sources, what is in the code, lift work

This is the full-plant reread the user asked for. Each block is one
physics segment. Sources are 2–4 independent. Basilisk is not a source.
Moe & Moe (2005) and Walker (2014) are **paywall** — review those by hand.

Default `SimParams.corotating` is now **true**. Energy is split into
drag work and lift work. Optical plate SRP is available beside Cr=1.8.

---

## 1. Sentman GSI — `cpp/src/aero/sentman.cpp`

**Code.** s = |v|/√(2 kT/m̄), γ = s cosθ, Z = 1+erf(γ), E = e^{−γ²}.

Cp_i = [(γ²+1/2)Z + γ E/√π] / s²  
Cτ_i = sinθ (γ Z + E/√π) / s  
Tr/Ti = α_E Tw/Ti + (1−α_E) s²/2   (M-01, not 1+α(Tw/Ti−1))  
Cp_r = √(Tr/Ti) (E + √π γ Z) / (2 s²)  
F = A q (−Cp n + Cτ t), q = ½ ρ |v|²  
Skip panel if cosθ ≤ 1e−12.

**Sources.** Sentman 1961 (DTIC AD0265409); Moe & Moe 2005 (*Planet. Space
Sci.*, paywall); Doornbos 2012 eqs. 3.53–3.55.

**Pass 3.** Keep M-01 and /s on Cτ (not the V1.3 /s² slip). Keep the
1e−12 skip so a 0.65 m² sail at numerical 90° is not a front face.

**Known case.** s=80, θ=0.4, α=1: combo = 2 cosθ ± 0.03. Face-on Cd ∈ [1.8, 2.4].
Hex edge-on Cd = 0.070 (thesis v3 0.064).

---

## 2. Co-rotating wind and lift work — `cpp/src/api.cpp`

**Code.** v_rel = v − ω⊕ × r, ω⊕ = (0,0,7.2921150×10⁻⁵).  
Gas in body: v_B_gas = −C_BN v_rel.  
Inertial aero force Fa_N = C_BNᵀ F_B.  
dE = (Fa_N · v) Δt / m   (spacecraft mechanical energy).  
Split: F_drag ∥ v_rel, F_lift ⊥ v_rel, then each dotted with **inertial v**.

When the atmosphere rotates, v_rel is not parallel to v, so
F_lift · v ≠ 0. That is lift doing work on the orbit. With corotation
off, F_lift · v = 0 by construction.

**Sources.** Vallado 2013 §8.6.2 (co-rotating atmosphere, relative wind);
King-Hele, *Satellite Orbits in an Atmosphere* (lift from a rotating
atmosphere); Doornbos 2012 Ch. 3 (v_rel and energy).

**Pass 3.** Keep v − ω×r (not +). Keep F·v (not F·v_rel) for vis-viva.
dE_baseline still uses min-axis |F| v_rel (counterfactual drag only).

**Known case.** `test_lift_work.py`: corotation off → dE_lift = 0; on +
finite AoA → |dE_lift| > 0; dE_drag + dE_lift = dE_actual (no SRP).

---

## 3. Panel SRP — `cpp/src/srp/panel_srp.cpp`

**Cannonball (default).** F_i = P Cr A (n·s) (−s), n·s > 0.  
Cr = 1.8, P = 4.56×10⁻⁶ Pa.

**Optical plate (new).** ca+cs+cd = 1.

F_i = −P A cosθ [(ca+cd) s + (2 cs cosθ + 2 cd/3) n]

**Sources.** Montenbruck & Gill 2000 §3.4; Vallado 2013 §8.6.4;
McInnes 1999 *Solar Sailing* (coating split).

**Pass 3.** Keep cannonball as the bit-compatible default. Optical is a
switch (`srp_optical`). Pure absorber ≡ cannonball Cr=1. Perfect
specular is 2 P A along −n, not along −s.

**Known case.** Face-on 1 m²: absorber |F|=P; specular |F|=2P; Lambert
|F|=P(1+2/3). `test_srp.py`, `test_srp_optical.py`.

---

## 4. Two-body, J2, GGM03S — `cpp/src/orbit/gravity.cpp`

**Code.** a = −μ r / |r|³.  
J2: f2 = (3/2) J2 μ Re² / r⁵, ax = f2 x (5 z²/r² − 1), …  
GGM03S: fully-normalized Stokes → unnormalized via N_nm, associated
Legendre in ECEF, a_N = ENᵀ a_E. J2 = −C20 after denormalisation.

**Sources.** Vallado 2013 §8; Montenbruck & Gill 2000 §3.2; Tapley /
CSR GGM03S header (coefficient authority).

**Pass 3.** Keep associated Legendre + pole guard (Pines is a later
speed-up, not an equation fix). Off-pole vs Basilisk computeField ~1e−16.

---

## 5. Frames, GMST, eclipse, SMA — `cpp/src/orbit/frames.cpp`

**GMST.** 280.46061837 + 360.98564736629 (JD−2451545) deg, IAU-1982 linear.  
**Bowring** WGS-84 lat/height.  
**Sun.** Montenbruck low-precision mean equator.  
**Eclipse.** Cylindrical: sunlit if r·s ≥ 0 or |r−(r·s)s| ≥ Re.  
**SMA.** a = −μ / (2ε), ε = v²/2 − μ/|r|.

**Sources.** Vallado 2013 App. / §3 (GMST), §4 (Bowring); Montenbruck &
Gill 2000 App. (Sun); Vallado §5 (cylindrical shadow) and vis-viva.

**Pass 3.** Keep cylindrical (no penumbra). Keep GGM μ for SMA when
harmonics are loaded.

---

## 6. Third body — `cpp/src/orbit/third_body.cpp`

**Code.** a = μ (d/|d|³ − r_b/|r_b|³), d = r_b − r.  
Optional SPICE; else analytic Sun/Moon.

**Sources.** Montenbruck & Gill 2000 §3.3; Vallado 2013 §8.6.

**Pass 3.** Keep direct+indirect. Lunisolar is off in SC_v3/v4 configs.

---

## 7. MRP kinematics — `cpp/src/attitude/mrp.cpp`

**Code.** C(σ) = I + [8[σ̃]² − 4(1−σ²)[σ̃]] / (1+σ²)²  
σ̇ = (1/4) B(σ) ω, B = (1−σ²)I + 2[σ̃] + 2σσᵀ  
shadow if σ² > 1. Angle = 4 arctan|σ|.  
q_BN → σ = q_vec/(1+q0) (Schaub). Python quat extraction is now
**passive** (C23−C32), matching V1.7.

**Sources.** Schaub & Junkins 2018 Ch. 3–4; Shuster 1993 *J. Astronaut. Sci.*

**Pass 3.** Keep shadow on σ²>1. The conjugate quat bug is closed.

---

## 8. MRP-PD and B-dot — `control/`

**MRP-PD.** τ = −kp σ_e − kd ω_e + ω×Iω, optional torque/rate caps.  
**B-dot.** m = −k Ḃ, τ = m × B.

**Sources.** Schaub & Junkins 2018 Ch. 8; Wertz 1978 (B-dot);
Stickler & Alfriend 1976.

**Pass 3.** Keep gyroscopic compensation. Prescribed mode zeros τ.

---

## 9. Magnetic field — `cpp/src/mag/field.cpp`

**Dipole.** B = (μ0/4π r³) [3(m·r̂)r̂ − m], m along +Z, |Beq|~30103 nT.  
**WMM.** Standard Gauss coefficients from a .COF file.

**Sources.** Vallado 2013 §8.7; Alken et al. 2021 WMM.

**Pass 3.** Keep until WMM is loaded. V1.7’s zero body-field observation
is **not** repeated: get_state / step expose B_B from this dipole/WMM.

---

## 10. Atmosphere query — `python/arlamx_v2/atmosphere.py`

**Code.** pymsis 2.1, F10.7 and Ap passed in. m̄ from species number
densities and 2019 SI masses. Geodetic height + Earth-fixed lon
(Bowring + GMST).

**Sources.** Picone et al. 2002 NRLMSISE-00; Emmert 2015 thermospheric
density review (MSIS 2.x lineage).

**Pass 3.** Keep MSIS 2.1. Density is an input to Sentman, not a second
drag model.

---

## 11. Integrator — `api.cpp` RK4

**Code.** Classical RK4 on [r, v, σ, ω]. Gravity at each stage;
aero/SRP/control ZOH across dt. Prescribed: σ held, ω = 0.

**Sources.** Hairer / Nørsett / Wanner, *Solving ODE I*, §II.1;
Schaub kinematics as above.

**Pass 3.** rk4_step_s ∈ (0, 120], decay uses 5 s.

---

## Connection map

```
advisor q_BN
    → slerp_clip → σ_tgt
    → v_rel = v − ω⊕×r          # corotating default on
    → Sentman(v_gas = −C v_rel)
    → SRP cannonball or optical plate
    → Fa_N · v , F_lift · v     # lift work
    → RK4 (GGM03S ECEF)
```

## What was not re-derived from memory

Every equation above is the one in the named sources. If a source is
paywalled, the user must open it. CLL/CCL is **not** in the plant; the
plan is `ACEnv/Val_Vel/ccl_fmf_plan.md`.
