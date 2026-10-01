time: 2026-08-20T06:58:21Z
agent: grok
style: detailed

# Independent physics and code audit of ARLAMX V2.0

**Plant:** `ARLAMX V2.0/` C++ library + Python Gym/training face.  
**Reference (read-only):** `ARLAMX-V1.7/` (not modified).  
**Code changed in this audit:** none. Nothing was deleted.  
**Prior LLM audits of V2.0:** not read.

This write-up is a new, independent audit. Findings come from reading the V2.0 C++/Python source, the V1.7 physics kernels, the V2.0 module specs, a full pytest run, independent numeric checks on this machine, and literature that was actually opened. If a formula could not be confirmed from an opened source, that is listed as uncertainty rather than treated as proven.

---

## 0. Mandate and method

The request was: go through ARLAMX 2.0, check that V1.7 defects are fixed and that the rewrite is faster and free of code/equation errors; do three or more (fewer than six) detailed passes; verify equations against online sources with citations; write a new report under `ACEnv/`; flag everything without editing the tree; record anything that could not be proved.

Five passes were run, in this order:

| Pass | Question |
|---|---|
| 1 | What is the plant, who calls whom, and which V1.7 defects were even in scope? |
| 2 | Do the implemented equations match opened textbooks/papers, or only the in-tree comments? |
| 3 | Do independent numeric oracles and the test suite agree with the kernels? |
| 4 | Where do the module specs, the C++ plant, and the Python env disagree? Frames, units, defaults. |
| 5 | Training env, reward, geometry, advisors: remaining defects and unproven claims. |

**Source rule used here.** A citation is marked **opened** only if the page or PDF was fetched and the relevant equation was read in this session. Paywalled or blocked documents are marked **not opened**. Wikipedia and Q&A sites are not used as authorities. Basilisk `computeField` is an implementation referee, not a literature source.

**Validation data this turn.** `PYTHONPATH=python python -m pytest tests -q` with `/home/nekolny/miniforge3/envs/ThesisMS/bin/python`: **74 passed** in 2.11 s, including `tests/basilisk_ref/test_gravity_vs_basilisk.py` at degrees 2/4/8. Additional numeric scripts in Pass 3.

---

## 1. What was reviewed

V2.0 C++ (all headers and sources under `cpp/`), Python plant face (`python/arlamx_v2/env.py`, `atmosphere.py`, `geometry.py`, `reward.py`, `sail_optics.py`, `decay_run.py`, `advisors/*`, `bindings.cpp` surface), tests under `tests/`, module specs `docs/modules/01`–`13`, `APPROVAL_PLAN.md`, V1.7 `CHANGELOG_V1.7.md` plus V1.7 `aero/gas_surface.py`, `aero/fmf_panel.py`, `aero/srp_panel.py`, `orbit_match/dynamics.py` J2/J3, `control/mrp_feedback.py`. V1.7 was not copied into V2.0.

Not reviewed as an authority: any file under `ACEnv/Reports/` that predates this audit, `claude_handoff.md`, and any other LLM-authored V2.0 review.

---

## 2. Connection map

```
Python ArlamxV2Env.step(action q4 [+gates])
  │  MSIS (pymsis) once per advisor step at Bowring(lat) + (lon_inertial − GMST)
  │  set_atmosphere(ρ, T, m̄)          — frozen for the whole C++ loop
  ▼
C++ Simulator::step(q_cmd)
  slerp_clip → σ_tgt
  for nsub = round(advisor_step_s / dt_s):          # default 150 × 2 s
      C = mrp_to_dcm(σ)                             # C_BN, v_B = C v_N
      v_rel = v − ω_⊕ × r     if corotating else v
      v_B_gas = −C v_rel
      Sentman F_B, τ_B, Cd, Cl
      SRP force only (cannonball or optical plate)
      B: WMM if loaded else untilted +Z dipole, ECEF → N → B
      τ_ctrl: prescribed 0 | detumble m×B | MRP-PD
      dE from F_N · v  (aero+SRP); dE_drag / dE_lift along/across v_rel
      RK4 on [r,v,σ,ω]; gravity+3rd-body at each stage; aero/SRP/τ ZOH
  return r,v,σ,ω, altitude_km=|r|−R_eq, sma, eclipse, sun_B, B_B, Cd, Cl, dE_*
  │
  ▼
Python: GS in ECEF rotated by GMST, battery SoC, reward, 26- or 35-vector obs
```

**What crosses the C++/Python boundary.** Packed `float64` panel SoA `(n, A, c)`; mass; diagonal inertia; `(ρ, T, m̄)`; quaternion command; returned state and accumulators. Reward, observation, MSIS, STL load, and PPO stay in Python.

**What does not cross.** Self-shadowing, gravity-gradient torque, SRP torque, species-resolved aero, NRLMSIS in-process, WMM (unless `wmm_path` is set; the Gym env never sets it).

---

## 3. Pass 1 — architecture and V1.7 issue mapping

V2.0 is a new C++ plant with a Python Gym face. It is not a Basilisk wrapper. Basilisk is optional in `tests/basilisk_ref/` only. That matches `APPROVAL_PLAN.md` D11.

The scientific loop is the V1.7 loop: attitude → Sentman force from the same panels and an MSIS density → Cowell integration. Training still does not fire thrusters.

### 3.1 V1.7 defects vs this tree

Taken from `ARLAMX-V1.7/CHANGELOG_V1.7.md` (read) and the live V1.7 aero source.

| V1.7 item | What V1.7 said | V2.0 as read | Verdict |
|---|---|---|---|
| **M-01** Sentman `T_r/T_i = 1 + α_E(T_w/T_i − 1)` | Fixed in V1.7 to energy-flux closure `T_r/T_i = α_E T_w/T_i + (1−α_E) s²/2` | Same formula in `cpp/src/aero/sentman.cpp` | **Ported.** Independent algebra of incident `C_p, C_τ` matches opened Sentman-family sources. The `s²/2` piece itself is **not** certified against the paywalled Moe & Moe 2005 typesetting (Pass 2, F-11). |
| Shadow-correct visible surface | Offline CAD check; live RL still sums all front faces | Live plant still sums all front-facing panels. Simplifier reduces CAD to plates; it does not ray-occlude. | **Not in the live loop.** Same limitation as V1.7 training. Material for non-convex CAD. |
| Test oracles that cloned the bug | V1.7 `test_aero.py` had hardcoded the bad `T_r` | V2.0 tests call C++ and check hyperthermal / α_E = 0 vs 1; they do not import V1.7 | **Not repeated.** The hyperthermal gate is still loose (F-10). |
| **M-02** SAC ONNX omits tanh squash | Deployment-only | No ONNX/export path exists under `python/arlamx_v2/` | **Not in scope of the plant; not re-implemented.** |
| **M-03** co-rotating atmosphere `false` in all V1.7 configs | OD path used it; RL did not | `SimParams.corotating = true`; Gym env leaves it true. `APPROVAL_PLAN.md` bake-off table still says `false` for SC_v3 | **Changed, not matched.** Physically better; **not** a same-as-V1.7 bake-off (F-06). |
| **M-04** ground stations fixed in inertial space | SC_v2 downlink null result | `advisors/stations.py` stores ECEF and rotates with GMST | **Fixed** in the V2.0 env. Residual: GS uses pre-step GMST (F-15). |
| Cannonball SRP, `C_r = 1.8`, force only | No specular/diffuse plate law | Default still cannonball force-only. Optional optical plate (`ca, cs, cd`) exists; Gym default is cannonball. Torque from SRP is computed in the kernel and **dropped** in `Simulator::step` | **Same as V1.7 training on the default path.** Optical law is extra. Missing SRP torque is F-08. |
| `catsat_v7_deployed_rich476.geom` inverted ram axis | Not a CatSat CAD | Default Gym geometry is V1.7 `earthcup_hex_v3.geom`. STL path uses `simplify_sides` | **Default training does not use that geom.** CAD path is a different model, not a silent reuse of the bad file. |
| Python inner-loop cost | Sentman + RK4 in interpreted Python | Fused C++ advisor step, `-O3`, SoA panels | **Addressed as a design.** Measured ~2.1 ms per 300 s advisor step on this machine (Pass 3). V1.7 wall-clock was **not** re-timed in this audit, so no speedup factor is claimed (F-19). |

### 3.2 Readability / layout (Pass 1 only)

The C++ split matches the approval plan: one topic per pair (`sentman`, `gravity`, `frames`, `mrp`, `mrp_feedback`, `bdot`, `field`, `panel_srp`, `api`). The plant object is re-entrant per `Simulator` handle. `float64` is used throughout.

Spec gaps, not style nits: `docs/modules/03` requires a Pines/Cunningham non-singular gravity evaluator; the code is naive associated Legendre with a pole guard. `docs/modules/05` describes `cpp/orbit/atmosphere.hpp` and a C++ MSIS callback; that header does not exist. `docs/modules/09` describes a **tilted** dipole; the implementation is **untilted** `+Z`. `docs/modules/11` lists `set_space_weather`; that command does not exist (Python owns F10.7/Ap).

---

## 4. Pass 2 — equations versus opened sources

Each block states the code formula, what was opened, and whether they match. Line numbers refer to the tree as read.

### 4.1 Sentman free-molecular panel (`cpp/src/aero/sentman.cpp`)

Speed ratio

\[
s = \frac{|v|}{\sqrt{2 k_B T / \bar m}}, \qquad
k_B = 1.380649\times 10^{-23}\,\mathrm{J\,K^{-1}}.
\]

\(k_B\) matches NIST CODATA 2018/2022, exact (**opened**: NIST Boltzmann constant page). \(s\) is the ratio of bulk speed to most-probable thermal speed; Walker 2014 eq. (3.26) is the same definition (**opened**).

Incident coefficients, with \(\gamma = s\cos\theta\), \(Z=1+\mathrm{erf}(\gamma)\), \(E=e^{-\gamma^2}\):

\[
C_{p,i}=\frac{(\gamma^2+\tfrac12)Z + \gamma E/\sqrt{\pi}}{s^2},\qquad
C_{\tau,i}=\sin\theta\cdot\frac{\gamma Z + E/\sqrt{\pi}}{s}.
\]

**Opened match.** Pilinski (Colorado thesis, eqs. 4.7–4.8) writes the same \(C_n\) and \(C_t\) for Sentman diffuse. Walker 2014 eqs. (3.23)–(3.28) are the \(\Gamma_1,\Gamma_2\) grouping of the same incident terms. Algebra: Pilinski \(C_n\) incident = code \(C_{p,i}\); Pilinski \(C_t\) = code \(C_{\tau,i}\). Re-emission pressure in both is \(\sqrt{T_r/T_i}\,(E+\sqrt{\pi}\,\gamma Z)/(2s^2)\), which is code `Cp_r`.

Force on a panel (`sum_panels`):

\[
\mathbf{F}=A q(-C_p\hat n + C_\tau\hat t),\quad
q=\tfrac12\rho |v|^2,\quad
\hat t \parallel (\hat v + \cos\theta\,\hat n).
\]

Front face: \(\cos\theta = -\hat v\cdot\hat n > 0\) with \(\hat v\) the **gas-onto-body** direction. That is the V1.7 `fmf_panel.py` convention and is consistent with pressure along \(-\hat n\) (into the surface). Grazing faces with \(\cos\theta\le 10^{-12}\) are skipped (after a V2.0 fix recorded in `version.md`).

**Hyperthermal identity.** Incident-only, \(C_p\cos\theta + C_\tau\sin\theta \to 2\cos\theta\) as \(s\to\infty\). With diffuse re-emission the extra term is \(O(1/s)\). Code at \(\theta=0.4\), \(\alpha_E=1\): residual \(0.202\) at \(s=8\), \(0.019\) at \(s=80\). The module spec’s “\(10^{-8}\) at \(s=80\)” is therefore **false** for the implemented (incident+re-emission) law. The pytest gate is \(0.03\), which the residual meets. **F-10.**

**Energy accommodation (M-01).** Code:

\[
\frac{T_r}{T_i}=\alpha_E\frac{T_w}{T_i}+(1-\alpha_E)\frac{s^2}{2}.
\]

This is the energy-flux closure: \(\alpha_E=(E_i-E_r)/(E_i-E_w)\), \(E_{r,w}=2kT_{r,w}\) (Maxwellian effusion energy flux), \(E_i=\tfrac12 m V^2\) (hyperthermal beam, thermal energy of the incident gas dropped). Then \(m V^2/(4k)=(s^2/2)T_i\).

**Opened on \(\alpha_E\):** Moe, AAS 05-258 (**opened**), eq. (1): \(\alpha=(E_i-E_r)/(E_i-E_w)\) with \(E_i\) “kinetic energy of an incoming molecule”. That paper does **not** write the \(s^2/2\) algebra.

**Opened on a different closure:** Walker 2014 eqs. (3.29)–(3.32) uses the same \(\alpha\) definition but sets \(T_{k,i}=V_i^2/(3R)=(2/3)s^2 T_\infty\). That is **not** \(s^2/2\). At \(s=8\), \(\alpha_E=0.93\), \(T_w/T_i=0.3\): flux closure gives \(T_r/T_i=2.519\); Walker’s \(2/3\) gives \(3.266\). Walker’s own Figure 3.4 at \(\alpha=1\) (where both closures coincide) is \(C_p\approx 2.14\); code gives \(2.137\). At \(\alpha=0.8\) the figure is \(\approx 2.35\) and code is \(2.587\) — the difference is the \(T_r\) convention, not a coding error in \(C_{p,i}\).

**Not opened:** Moe & Moe, *Planet. Space Sci.* 53 (2005) 793–801 (paywall). That is the paper the in-tree comments treat as the typeset source of \(s^2/2\). **F-11.** Until that PDF is read, the incident Sentman integrals are considered verified; the M-01 \(T_r\) algebra is considered **consistent with energy-flux 2kT + hyperthermal \(E_i\)**, **inconsistent with Walker’s \(T_k=V^2/(3R)\)**, and **unverified against the 2005 PSS typesetting**.

At \(\alpha_E=1\), \(T_r=T_w\) in every convention. Face-on two-sided unit square, \(v=7500\,\mathrm{m\,s^{-1}}\), \(T=T_w=1000\,\mathrm{K}\), atomic-O mass: \(C_D(\alpha_E=1)=2.259\), \(C_D(\alpha_E=0.93)=2.423\) (**+7.3 %**). V1.7 changelog claimed +6.5–8.5 % at \(\alpha_E=0.93\). That band is reproduced.

Whole-body \(C_D=\mathbf{F}\cdot\hat v/(q A_\mathrm{ref})\), \(A_\mathrm{ref}=\tfrac12\sum A_i\) when `one_sided_ref` (default). `one_sided_ref` is **not** bound to Python; it is stuck at `true`. Harmless if that is intended.

### 4.2 Two-body, J2, J3 (`cpp/src/orbit/gravity.cpp`)

\[
\mathbf{a}_{2b}=-\mu\mathbf{r}/r^3.
\]

J2 (code `accel_j2`):

\[
f_2=\frac32 J_2\mu R_E^2/r^5,\quad
a_x=f_2 x(5z^2/r^2-1),\ \text{same in }y,\quad
a_z=f_2 z(5z^2/r^2-3).
\]

**Opened match.** Niemeyer, *Orbit Perturbations* (opened), and poliastro docs quoting Curtis (12.30) (opened). Independent finite-difference of

\[
U=\frac{\mu}{r}\Bigl[1-J_2(R_E/r)^2 P_2(z/r)\Bigr],\quad P_2=\tfrac12(3\sin^2\phi-1)
\]

at \(\mathbf{r}=(5,1.2,4)\times 10^6\,\mathrm{m}\) gives relative error \(5.1\times 10^{-10}\) against `accel_twobody+accel_j2`. **J2 is correct.**

J3 (code `accel_j3`) matches Niemeyer’s Cartesian J3 after multiplying through by \(r^2/r^2\). **Opened match.** The plant fallback (`gravity` file missing) adds two-body+J2 only; J3 is unused there. When GGM03S is loaded (Gym default), degree ≥ 3 includes \(C_{30}\) as part of the harmonic sum.

Zonal J2 evaluated in inertial coordinates with a shared \(z\)-axis is equivalent to ECEF-then-rotate, because J2 is axisymmetric about \(z\). Tesseral terms are not. The fallback path is therefore legitimate for J2 and **not** for \(C_{22}\).

### 4.3 GGM03S harmonics (`GravityHarmonics`)

File header (opened, `ARLAMX-V1.7/support/GGM03S.txt` and the Basilisk copy used at runtime):

```
0.6378136300E+07, 0.3986004415E+15, 7.2921150E-5, 180, 180, ...
2, 0, -4.841692638330E-04, ...
```

Code `MU_GGM`, `RE_GGM`, `OMEGA_EARTH` copy that header. Fully-normalized \(\bar C_{20}\) converts as \(J_2=-\bar C_{20}\sqrt{5}\). Computed \(J_2=1.0826353865466185\times 10^{-3}\), identical to `constants.hpp`. **Coefficient authority is the file, not a second-hand quote.**

Potential (unnormalized associated Legendre, geocentric latitude \(\phi\), no Condon–Shortley):

\[
U_\mathrm{pert}=\frac{\mu}{r}\sum_{n=2}^{N}\sum_{m=0}^{n}
\Bigl(\frac{R_E}{r}\Bigr)^n P_n^m(\sin\phi)
\bigl[C_{nm}\cos m\lambda+S_{nm}\sin m\lambda\bigr],
\]

then \(\mathbf{a}=\nabla U\) in spherical basis, two-body added separately. Normalization \(C_\mathrm{unnorm}=\bar C\,N_{nm}\) with \(N_{n0}=\sqrt{2n+1}\) and \(N_{nm}=\sqrt{2(2n+1)(n-m)!/(n+m)!}\) for \(m>0\) is the standard fully-normalized → conventional map.

**Numeric referee (not literature):** Basilisk `SphericalHarmonicsGravityModel.computeField` on the same file.

| degree | off-pole relative \(\|\mathbf{a}-\mathbf{a}_\mathrm{BSK}\|/\|\mathbf{a}_\mathrm{BSK}\|\) | exact pole |
|---|---|---|
| 2 | \(2\times 10^{-16}\) | \(5\times 10^{-9}\) |
| 4 | \(4\times 10^{-16}\) | \(7.3\times 10^{-6}\) |
| 8 | \(6\times 10^{-16}\) | \(9.7\times 10^{-6}\) |
| 20 | \(6\times 10^{-16}\) | \(1.2\times 10^{-5}\) |

Off the polar axis the field is correct to machine precision against Basilisk. On the axis it is finite (the classroom “must not NaN” test) but **not** at the 1e-8 spec, and it is **not** Pines/Cunningham as `docs/modules/03` required. Default SolarCat \(i=23^\circ\) never hits the pole. **F-03.**

Degree-2 full field vs two-body+J2 differs at \(10^{-5}\) off-equator because \(C_{22},C_{21}\) are present. That is expected, not a bug.

Earth orientation for the field is GMST-only `dcm_EN` (rotation about \(+z\)). No polar motion, no nutation. Consistent with the rest of the plant.

### 4.4 GMST, Bowring, co-rotation

IAU 1982 linear GMST (**opened**: Meeus form quoted by astrogreg, which is Aoki et al. 1982 without \(T^2,T^3\)):

\[
\theta_0=280.46061837^\circ+360.98564736629^\circ\,(\mathrm{JD}-2451545.0).
\]

Code implements exactly that and drops \(0.000387933 T^2-T^3/38710000\). At 2026, \(T\approx 0.26\), the quadratic is \(\sim 0.09''\). Acceptable at this plant’s accuracy class; document the truncation.

Bowring WGS-84 (`ecef_to_geodetic`): \(\theta=\mathrm{atan2}(z a, p b)\), \(\phi=\mathrm{atan2}(z+e'^2 b\sin^3\theta,\, p-e^2 a\cos^3\theta)\). **Original Bowring 1976 PDF was not opened.** The implemented formula is the standard one-iteration form. Round-trip at geodetic \(23^\circ\), \(h=400\,\mathrm{km}\): height error \(0.20\,\mathrm{mm}\), latitude error \(4\times 10^{-9}\,\mathrm{deg}\). **Algorithm behaves as Bowring; the 1976 paper itself is unverified in this session.** Flattening \(f=1/298.257223563\) is the WGS-84 value; TR8350.2 was not opened.

Earth rate \(\omega_\oplus=7.2921150\times 10^{-5}\,\mathrm{rad\,s^{-1}}\) matches IERS numerical standards as listed by Paris Observatory EOP-PC (**opened**: \(\Omega=7.292\,115\,0(1)\times 10^{-5}\,\mathrm{rad\,s^{-1}}\)). Co-rotating wind \(\mathbf{v}_\mathrm{rel}=\mathbf{v}-\boldsymbol{\omega}_\oplus\times\mathbf{r}\) is Vallado’s drag relative velocity (poliastro drag notes, opened, write \(\mathbf{v}_\mathrm{rel}\) the same way). **F-06** is about the *default flag*, not the formula.

### 4.5 Third body, Sun, Moon, eclipse

Point-mass Earth-centred third body (**opened**: poliastro quoting Curtis 12.10):

\[
\mathbf{a}_{3B}=\mu_k\Bigl(\frac{\mathbf{r}_k-\mathbf{r}}{|\mathbf{r}_k-\mathbf{r}|^3}-\frac{\mathbf{r}_k}{|\mathbf{r}_k|^3}\Bigr).
\]

Code matches. Hand-eval test is identity. Default `lunisolar=false`. Lunar term at LEO is \(\sim 10^{-6}\,\mathrm{m\,s^{-2}}\) vs two-body \(\sim 8.7\,\mathrm{m\,s^{-2}}\).

Analytic Sun (`sun_unit_analytic`) is the low-precision Meeus/Montenbruck mean-longitude form. Unit-norm at JD 2461060.5. Not compared to SPICE in this audit (CSPICE path exists but is not the Gym default). Expected error of this class is tenths of a degree; that is enough for LEO SRP at the cannonball level and **not** enough for milliarcsecond work.

Analytic Moon (`moon_analytic`) uses Meeus truncated lunar **ecliptic** longitude and latitude, then builds

\[
\hat r=(\cos\beta\cos\lambda,\;\cos\beta\sin\lambda,\;\sin\beta)
\]

**with no ecliptic-to-equatorial rotation.** Independent reconstruction at JD 2461060.5: ecliptic \(\lambda\approx 313.47^\circ\), \(\beta\approx -2.21^\circ\); angle between that vector and the same angles rotated by the true obliquity is **\(16.97^\circ\)**. If `lunisolar` is ever enabled without SPICE, the Moon’s acceleration is pointed the wrong way by that amount. **F-02.** Default Gym has lunisolar off, so the training path is not hit.

Cylindrical eclipse: lit if \(\mathbf{r}\cdot\hat s\ge 0\) or impact parameter \(\ge R_E\). Matches the V1.7 RK4 model. No penumbra, no flattening. \(R_E\) is WGS-84 equatorial, so polar umbra is slightly oversized.

SPICE, when linked: `et=(jd-2451545)\times 86400`. If `epoch_jd` is UTC, that is not TDB/ET (leap seconds + 32.184 s). Solar angular error from 69 s is negligible; this is still the wrong time scale if someone treats SPICE as high-fidelity. **F-16.**

### 4.6 MRP kinematics and DCM (`cpp/src/attitude/mrp.cpp`)

**Opened:** Crassidis & Markley, NASA 1996 (NTRS 19960035754), eqs. (7), (8), (11).

\[
\boldsymbol{\sigma}=\frac{\mathbf{q}_{1:3}}{1+q_0}=\hat n\tan(\Phi/4).
\]

Code `quat_to_mrp` is that, after \(q_0\ge 0\).

\[
\dot{\boldsymbol{\sigma}}=\frac14\bigl[(1-\sigma^2)I+2[\tilde\sigma]+2\boldsymbol{\sigma}\boldsymbol{\sigma}^\top\bigr]\boldsymbol{\omega}.
\]

Code `mrp_rate` builds that \(B\) matrix and multiplies by \(1/4\). Matches Crassidis (8)/(9).

DCM (Crassidis eq. 11; code `mrp_to_dcm`):

\[
C=I+\frac{8[\tilde\sigma]^2-4(1-\sigma^2)[\tilde\sigma]}{(1+\sigma^2)^2}.
\]

90° about \(+z\): \(\sigma_z=\tan 22.5^\circ\),

\[
C=\begin{pmatrix}0&1&0\\-1&0&0\\0&0&1\end{pmatrix},
\]

\(\det C=1\), \(C C^\top=I\) to \(3\times 10^{-16}\). Inertial \(+\hat x\) maps to body \(-\hat y\). That is \(C_{BN}\) for a right-hand \(+90^\circ\) body rotation about \(z\). Shadow set \(\boldsymbol{\sigma}\leftarrow -\boldsymbol{\sigma}/\sigma^2\) when \(\sigma^2>1\). Angle \(4\arctan|\boldsymbol{\sigma}|\). Compose/inverse with `proper` \(=-\boldsymbol{\sigma}\) is the physically correct inverse (V1.7 `proper` default). **Schaub & Junkins textbook was not opened**; Crassidis is an independent opened source for the same algebra. **Tsiotras 1996 (MRP-PD stability) was not opened.**

Euler rotational equation \(\dot{\boldsymbol{\omega}}=I^{-1}(\boldsymbol{\tau}-\boldsymbol{\omega}\times I\boldsymbol{\omega})\) is standard rigid-body mechanics (Wertz / any dynamics text). `inverse_sym` is a full 3×3 inverse; if \(\det I\approx 0\) it returns the zero matrix and \(\dot{\boldsymbol{\omega}}\) silently vanishes. **F-18.**

RK4 is the classical four-stage rule with stage-wise DCM and stage-wise gravity inside `Simulator::step`. The free function `rk4_step` **freezes** gravity at the value passed in; the Simulator does **not** call it. Dead/divergent helper, not a live-path bug. **Hairer / Numerical Recipes were not opened**; the Butcher coefficients in source are the textbook RK4 weights.

### 4.7 MRP-PD and B-dot

\[
\boldsymbol{\tau}=-K\boldsymbol{\sigma}_e-P\boldsymbol{\omega}_e+\boldsymbol{\omega}\times I\boldsymbol{\omega}.
\]

Code does that, then optional per-axis torque clip, then a predictive rate cap that rebuilds \(\boldsymbol{\tau}\) from the PD part only (gyroscopic feedforward cancelled in the prediction). That matches the module spec and V1.7 `mrp_feedback.py`. **Tsiotras 1996 not opened**; the *formula* is the standard globally-stabilizing MRP PD, the *stability proof* is not re-derived here. Closed-loop test `test_regulation_closes` drives angle below \(0.05^\circ\). Unlimited torque plus a 180 s hold was a known NaN path; `prescribed` mode and, in v5+/v7, magnetorquer saturation address it in the Gym. Bare `Simulator` still defaults to unlimited torque.

B-dot: first sample \(\mathbf{m}=\mathbf{0}\); then \(\mathbf{m}=-k\dot{\mathbf{B}}_B\), saturated; \(\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}\). **Opened:** comparison papers quoting Avanzini & Giulietti 2012 and Stickler & Alfriend 1976 as \(\mathbf{m}=-k\dot{\mathbf{b}}\). **Stickler & Alfriend 1976 PDF not opened.** The discrete difference \(\dot{\mathbf{B}}\approx(\mathbf{B}_k-\mathbf{B}_{k-1})/\Delta t\) is the usual flight approximation. Tested: first call zero, saturation, spin energy falls on a toy Euler plant.

### 4.8 Magnetic field

Untilted dipole, tesla:

\[
\mathbf{B}=\frac{\mu_0}{4\pi r^3}\bigl(3(\mathbf{m}\cdot\hat r)\hat r-\mathbf{m}\bigr),\quad
\mathbf{m}=7.94\times 10^{22}\,\mathrm{A\,m^2}\,\hat z.
\]

\(\mu_0/4\pi=10^{-7}\). Equator \(|B|=30601\,\mathrm{nT}\), pole/eq = 2 exactly. Pulecchi, Lovera & Varga (**opened**) use \(\mu_m=7.943\times 10^{15}\,\mathrm{Wb\,m}=(\mu_0/4\pi)m\), i.e. the **same magnitude**, and a **tilt \(\gamma=11.44^\circ\)**. Spec `docs/modules/09` says tilted. Code comment says untilted. Earth’s real dipole points toward geographic **south**, so \(B_z\) at the geographic north pole is **inward**. WMM at \(r=(0,0,R_E+400\,\mathrm{km})\), year 2025: \(B_z=-31608\,\mathrm{nT}\). Code dipole: \(B_z=+50994\,\mathrm{nT}\). **Opposite polarity and no tilt. F-01.**

B-dot damping is invariant under a global reversal of \(\mathbf{B}\) (\(\mathbf{m}\) and \(\mathbf{B}\) both flip). Magnetometer **observations** used by the policy are not. The Gym env never sets `wmm_path`, so training \(\mathbf{B}\) is this dipole.

WMM: file loads `g_n^m, h_n^m` and secular terms. Evaluation uses Schmidt-quasi-normalized associated Legendre in geocentric spherical coordinates with reference radius \(6371.2\,\mathrm{km}\), then nT→T, then spherical→ECEF. That **outline** is the WMM/IGRF algorithm. The recurrence in `field.cpp` is commented “approximate”. The WMM 2020 technical report PDF was too large to fetch in this session (**not opened**). Tests only check “10–80 µT in LEO”. Surface equator \(|B|\approx 29.1\,\mu\mathrm{T}\) is plausible but is **not** a coefficient-level proof. **F-04.** Training does not call WMM.

### 4.9 SRP

Cannonball (default, V1.7 law):

\[
\mathbf{F}=\sum_{\hat n\cdot\hat s>0} P\,C_r A\,(\hat n\cdot\hat s)\,(-\hat s)\,\varepsilon.
\]

\(P=4.56\times 10^{-6}\,\mathrm{Pa}\). Face-on \(C_r=1\), \(A=1\,\mathrm{m^2}\) gives \(|F|=P\) exactly. **Montenbruck §3.4 and Vallado §8.6.4 were not opened.** The special cases absorber \(|F|=P\), perfect specular along the normal \(|F|=2P\), Lambert \(|F|=P(1+2/3)\) are the standard plate identities and the optical tests pass at \(10^{-14}\). I am **not** willing to cite Montenbruck’s page number as if I had the book. **F-12** (constant 1 AU pressure, no \((a_\odot/r)^2\)).

Optical plate as coded:

\[
\mathbf{F}_i=-P A\cos\theta\bigl[(c_a+c_d)\hat s+(2 c_s\cos\theta+\tfrac23 c_d)\hat n\bigr].
\]

Coefficients are renormalized to sum to 1. Gym default `srp_optical=false`. **F-08:** `Simulator::step` adds SRP **force** and discards SRP **torque** even though `panel_srp_force_torque` / `panel_srp_optical` return both.

### 4.10 Atmosphere (Python)

`query_msis`: `pymsis.calculate(dates, lons, lats, alts, f107s=..., f107as=..., aps=..., version=2.1)`. **Opened:** pymsis 0.12 documentation. Argument order and names match. Output columns: \(\rho\), n(N2), n(O2), n(O), n(He), n(H), n(Ar), n(N), anomalous O, NO, \(T\). Mean mass sums the seven named species (not anomalous O, not NO), molar masses / \(N_A\) with 2019 SI. Fallback \(\bar m=m_O\).

Independent call at 400 km, 0°N, 0°E, 2026-01-15, F10.7=150, Ap=4, MSIS 2.1: \(\rho=2.35\times 10^{-12}\,\mathrm{kg\,m^{-3}}\), \(T=936\,\mathrm{K}\). That is a normal LEO number. **Picone et al. 2002 (NRLMSISE-00 paper) was not opened**; the wrapper talks to the NRLMSIS implementation inside pymsis, not a re-derived atmosphere.

Gym queries MSIS **once per 300 s advisor step** and holds \((\rho,T,\bar m)\) for all 150 substeps. V1.7 cached every 5 substeps (10 s). **F-05.** On exception the env silently substitutes \((3\times 10^{-12}, 900\,\mathrm{K}, m_O)\).

Longitude: inertial position is passed to `ecef_to_geodetic` (latitude and height are invariant under a \(z\)-rotation), then `lon_ecef = lon - GMST`. That is the correct GMST-only ECEF longitude.

**Altitude used by MSIS is Bowring geodetic. Altitude used for termination, reward, and `altitude_km` is \((|r|-R_{eq})/10^3\).** At 23° geodetic 400 km, spherical \(|r|-R_{eq}=396.76\,\mathrm{km}\). **F-07.**

---

## 5. Pass 3 — independent numeric checks and tests

### 5.1 Tests run

Command: `PYTHONPATH=python python -m pytest tests -q`  
Interpreter: `/home/nekolny/miniforge3/envs/ThesisMS/bin/python`  
Result: **74 passed**, 13 warnings (Basilisk SWIG deprecations), 2.11 s.  
19 `test_*.py` files. Basilisk referee ran (degrees 2/4/8).

Classroom coverage vs `docs/VALIDATION_STANDARD.md`:

| Spec gate | What the test actually does |
|---|---|
| Sentman hyperthermal \(10^{-8}\) at \(s=80\) | Residual 0.019; assert `< 0.03` |
| Face-on \(C_D\) in 1.8–2.4 | Passes (2.26 at \(\alpha_E=1\)) |
| J2 vs GGM degree 2 | Relative \(< 5\times 10^{-3}\) (tesseral leftover), not 1e-8 zonal-only |
| Polar field finite | Passes; accuracy vs Basilisk is 7e-6 at degree 4 on the axis |
| Vacuum SMA \(< 1\,\mathrm{m}\) over 1 period | ~0.1 orbit, 50 m allowance |
| WMM LEO sanity | `return` if file missing, not `pytest.skip` |
| MSIS dedicated unit test | Still absent; only exercised through the env |

Hostile script `tests/validation/hostile_inputs.py` is a runner, not part of pytest. Live gates: negative \(\rho\) and \(T=0\) now throw; \(r=0\) reset throws; `rk4_step_s=3600` throws. `step` with a NaN quaternion **does not throw**: it substitutes identity. **F-14.**

### 5.2 Independent oracles (this machine)

| Check | Result |
|---|---|
| Python scalar Sentman vs `cpp.sentman` | \(0\) difference on four \((\theta,s,\alpha_E)\) points |
| J2 potential FD vs `accel_j2` | \(5.1\times 10^{-10}\) relative |
| GGM vs Basilisk off-pole | \(\sim 10^{-16}\) |
| GGM vs Basilisk exact pole, degree 4 | \(7.3\times 10^{-6}\) |
| \(J_2\) from \(\bar C_{20}\sqrt{5}\) vs `J2_GGM` | exact |
| Optical plate identities | absorber \(-P\), specular \(-2P\), Lambert \(-P(1+2/3)\), all at \(\le 10^{-15}\) |
| Bowring round-trip 23°, 400 km | \(0.20\,\mathrm{mm}\) |
| Dipole pole/eq | exactly 2 |
| MSIS 400 km sanity | \(2.35\times 10^{-12}\,\mathrm{kg\,m^{-3}}\) |
| Moon ecliptic-as-equatorial angle | \(16.97^\circ\) |
| Gym `env.step` | \(2.13\,\mathrm{ms}\) per 300 s advisor step (6-step average, seed 0) |
| C++ 72-panel `spacecraft_aero` | \(3.6\,\mu\mathrm{s}\) |
| Python scalar Sentman vs C++ scalar | \(10.6\times\) (not the fused-step speedup) |

### 5.3 Speed versus V1.7

V1.7’s cost was a Python panel loop plus a Python RK4 (or Basilisk) inside 150 substeps. V2.0 fuses that loop in C++ (`-O3`). The fused advisor step is **2.1 ms** here, which is the number `docs/modules/11` already advertised.

**This audit did not time V1.7 on the same machine**, so a “\(N\times\) faster than V1.7” claim would be invented. Kernel-level evidence: 72-panel aero is microseconds; a Python loop of 72 `erf`/`exp` plus NumPy temporaries is not. That is a real improvement. The factor is **not quantified against V1.7 wall-clock in this report. F-19.**

MSIS is still pymsis in Python once per advisor step. That is no longer the inner-loop cost.

---

## 6. Pass 4 — spec versus code, frames, units

| Spec / contract | Code | Flag |
|---|---|---|
| Pines/Cunningham gravity | Naive \(P_n^m(\sin\phi)\) + pole guard | F-03 |
| Tilted dipole | Untilted \(+\hat z\), opposite Earth polarity | F-01 |
| WMM official algorithm | Recurrence marked approximate; unused in Gym | F-04 |
| `atmosphere.hpp` / C++ MSIS callback | Does not exist; Python sets scalars | documented Phase 1, still a spec lie |
| `set_space_weather` | Does not exist | spec-only |
| SC_v3 bake-off `corotating_atmosphere: false` | Default `true` | F-06 |
| `altitude_km` as geodetic | \(|r|-R_{eq}\) | F-07 |
| Optional \(\boldsymbol{\tau}_{gg}\) | Not implemented | F-09 |
| SRP moment in the force model | Force only | F-08 |
| Constrained Delaunay of each side | Ear-clip of a single outer loop | F-17 |
| Hyperthermal \(10^{-8}\) at \(s=80\) | Residual \(0.019\) | F-10 |

Frames, as coded: **N** = GCRS/J2000 with GMST-only Earth rotation (no polar motion). **B** = body, \(C_{BN}=\mathrm{DCM}(\boldsymbol{\sigma})\). **E** = z-rotation of N. Units SI. Harmonic \(\mu\) is GGM’s \(3.986004415\times 10^{14}\); vis-viva fallback and COE use \(3.986004418\times 10^{14}\). Difference 0.75 ppm. Osculating SMA from mixed \(\mu\) is a small inconsistency when GGM is loaded vs when it is not.

LVLH DCM `dcm_LN` exists (nadir, anti-orbit-normal, completing triad) and is **not** used inside `Simulator::step`. Commands are inertial quaternions. V1.7 `action.frame: lvlh` is not a V2.0 plant feature.

---

## 7. Pass 5 — env, reward, geometry, advisors

### 7.1 Gym env

Default: hex `.geom`, GGM degree 4, lunisolar off, panel SRP cannonball, corotating on, \(\alpha_E=0.93\), \(T_w=300\,\mathrm{K}\), 2 s / 300 s, mass 0.625 kg, inertia \((0.0125,0.0125,0.025)\). Magnetorquer saturation is installed for v5+/v7; v3 uses unlimited MRP-PD plus a 45° slew clip.

WMM is never loaded. Magnetic observations are the untilted dipole. **F-01.**

Atmosphere held 300 s. **F-05.**

`altitude_km` spherical. Episode ends at 250 km spherical (~253 km geodetic at 23°). **F-07.**

Ground stations: ECEF catalogue rotated by GMST (M-04 fixed), using GMST from **before** the 300 s step. Earth turns \(\approx 1.25^\circ\) in 300 s. **F-15.**

Power: peak \(0.78\times 0.85\,\mathrm{W}\), loads 7 mW + 205 mW sunlit + 400 mW downlink + 270 mW \(\times\) torque effort, capacity \(0.53\,\mathrm{W\,h}\). These numbers were not traced to a hardware datasheet in this audit. **F-20 (unproven).**

v6 GPS dropout integrates **two-body only** (`_two_body_rk4` in `mpc.py` / env). No drag, no J2. That is a training nuisance, not a plant error.

v7 disturbance observer: \(\boldsymbol{\tau}_\mathrm{dist}=I\dot{\boldsymbol{\omega}}+\boldsymbol{\omega}\times I\boldsymbol{\omega}-\boldsymbol{\tau}_\mathrm{ctrl}\) from gyro endpoints over 300 s. Valid in principle; 300 s finite difference is crude relative to the 2 s physics. Not independently validated against `tau_aero_mean` in this audit beyond noting the test file `test_v7_ipc.py` exists and is in the 74 that passed.

### 7.2 Reward

SC_v3 terms in `reward.py` match `docs/modules/12` algebra: exponential \(w_\mathrm{long}\), \(\Delta E\) clip, SoC power, GS Cd gate, shade, smoothness, omega, altitude cliff. `compose_sc_v3` **omits** `momentum_penalty` even though the spec lists it; v4+ include it. **F-21.** Weights and shaping (v4–v7) are design choices, not physics. They are not “wrong equations”; they are not validated against flight.

`dE_actual` is specific-energy work \((\mathbf{F}_\mathrm{aero}+\mathbf{F}_\mathrm{SRP})\cdot\mathbf{v}\,\Delta t/m\). That is the inertial mechanical power of the non-gravitational force, which is the right scalar for orbital energy, not \(\mathbf{F}\cdot\mathbf{v}_\mathrm{rel}\). Split `dE_drag` / `dE_lift` projects aero onto \(\hat v_\mathrm{rel}\) then dots with \(\mathbf{v}\). With corotation off and SRP off, `dE_lift=0` and `dE_drag+dE_lift=dE_actual` (tested).

Min-drag counterfactual: body axis of smallest \(+\hat e_k\) projected area, then `coefficients_only` with gas along \(+\hat e_k\). For a two-sided sail \(+\hat z\) and \(-\hat z\) areas match, so ram of \(-\hat z\) equals projected \(+\hat z\). For an asymmetric body they need not. **F-22.**

### 7.3 Geometry simplifier

Region-grow by normal angle, fit plane, project shared edges onto the intersection line, ear-clip outline if area stays inside the quality budget, else keep source triangles with area preserved. Cube: 6 sides, area 6, sealed. Hex prism: sealed, area within 2 % in the older test / 1 % in the quality test.

Spec promised constrained Delaunay that preserves holes. Implementation is ear-clip of one outer loop (`_earclip`). Interior holes (panel cutouts) are **not** a first-class path. **F-17.** Cohen-Steiner et al. 2004 and Shewchuk 1996 were **not opened**; they are cited as method ancestors, not as equation checks.

No live self-shadowing. A convex hex does not need it. A deployed CatSat CAD does. V1.7 measured 1.2–2.2× drag over-statement on that CAD without a depth buffer. V2.0’s simplifier reduces facet count; it does not replace occlusion. **F-13.**

### 7.4 Advisors / MPC

Sampling MPC rolls a **two-body** surrogate with a Sentman \(C_D\), not the GGM+MSIS plant. Documented in `mpc.py`. Scores are therefore not the plant’s \(\Delta E\). Heuristic bank returns unit quaternions (tested).

No ONNX export. V1.7 M-02 is untouched because there is nothing to export to.

---

## 8. Findings ledger

Severity: **H** = changes default training numbers or contradicts a stated contract; **M** = real defect, default path partly shielded; **L** = spec/test/docs; **U** = cannot prove from opened sources.

| ID | Sev | Where | What |
|---|---|---|---|
| F-01 | H | `mag/field.cpp`, `env.py` | Training \(\mathbf{B}\) is an untilted \(+\hat z\) dipole of magnitude \(7.94\times 10^{22}\,\mathrm{A\,m^2}\). Spec asked for tilt. Polarity is opposite Earth’s (north-pole \(B_z\) positive vs WMM negative). Gym never sets `wmm_path`. B-dot still damps; mag features in the observation are not Earth-like. |
| F-02 | H | `orbit/frames.cpp` `moon_analytic` | Meeus ecliptic \((\lambda,\beta)\) used as equatorial. ~17° error at the test epoch. Default `lunisolar=false` hides it. |
| F-03 | H | `orbit/gravity.cpp`; spec 03 | Not Pines. Exact-pole degree-4 field is \(7\times 10^{-6}\) relative vs Basilisk (spec 1e-8). Off-pole is machine-precise. Default \(i=23^\circ\) is safe; polar orbits are not. |
| F-04 | M | `mag/field.cpp` | WMM recurrence is commented approximate. No coefficient-level test. Unused in Gym. |
| F-05 | H | `env.py` + `Simulator::step` | MSIS sampled once per 300 s and held. V1.7 updated ~every 10 s. Density at VLEO can move several percent in 300 s. |
| F-06 | H | `api.hpp`, `env.py`, `APPROVAL_PLAN.md` | Co-rotation default **on**. V1.7 SC_v3 YAML and the bake-off table say **off**. Drag ~10 % class effect at \(i=23^\circ\) (V1.7 changelog: +11.7 %). |
| F-07 | H | `api.cpp` altitude; `env.py` MSIS | Two altitudes. Reward/termination use spherical \(|r|-R_{eq}\) (3.2 km low at 23°). MSIS uses Bowring geodetic. |
| F-08 | M | `api.cpp` ~221 | SRP torque computed, not applied. Sail with offset centre of pressure has no SRP moment. V1.7 training was also force-only. |
| F-09 | M | plant | Gravity-gradient \(\boldsymbol{\tau}_{gg}=3\mu r^{-3}\,\hat o\times I\hat o\) not present. Optional in the overview; listed in the force-model section of the approval plan. |
| F-10 | L | spec 01 vs `test_sentman.py` | Spec claims hyperthermal residual \(10^{-8}\) at \(s=80\). Implemented residual 0.019 with re-emission. Test allows 0.03. |
| F-11 | U | `sentman.cpp` M-01 | \(T_r\) uses \(s^2/2\). Incident \(C_p,C_\tau\) verified. Moe & Moe 2005 PSS **not opened**. Walker 2014 uses \(2s^2/3\). |
| F-12 | M | `panel_srp.cpp` | \(P=4.56\times 10^{-6}\,\mathrm{Pa}\) constant. No \((1\,\mathrm{AU}/r_\odot)^2\). Few-percent annual envelope. Montenbruck/Vallado pages not opened. |
| F-13 | H | live aero | No self-shadowing. Convex hex: OK. Non-convex CAD: V1.7 measured 1.2–2.2× over-drag. |
| F-14 | M | `api.cpp` `step` | Non-finite quaternion → identity, no exception. |
| F-15 | M | `env.py` GS | GMST taken before the 300 s step (~1.25° Earth turn). |
| F-16 | L | `third_body.cpp` SPICE | `et` from JD as if TDB. Gym does not use SPICE. |
| F-17 | L | `geometry.py` vs spec 10 | Ear-clip, not constrained Delaunay; holes not preserved by construction. |
| F-18 | M | `types.hpp` `inverse_sym` | Singular inertia → zero inverse → \(\dot{\boldsymbol{\omega}}=\mathbf{0}\) with no error. |
| F-19 | U | speed | V2.0 fused step ~2.1 ms. V1.7 not timed here. Do not quote a factor. |
| F-20 | U | `env.py` power | 0.53 W h, 0.78×0.85 W peak, load table: not traced to a datasheet in this audit. |
| F-21 | L | `reward.py` vs spec 12 | SC_v3 compose drops `momentum_penalty` that the spec lists. |
| F-22 | M | `api.cpp` min-drag axis | \(+\hat e_k\) projected area may not equal ram of a flow along \(+\hat e_k\) on an asymmetric body. |
| F-23 | L | `accel_gravity_N` | If `!loaded()`, two-body is computed in N then treated as ECEF. Caller overwrites this; latent if the helper is reused. |
| F-24 | L | tests | WMM missing-file path `return`s; SMA vacuum test is not the spec’s 1-period 1 m gate; no dedicated MSIS unit test. |
| F-25 | M | `env.py` | pymsis failure → silent \((3\times 10^{-12}, 900\,\mathrm{K}, m_O)\). |

No finding in this list was “fixed” in this audit. The code is as found.

---

## 9. Uncertainties (could not prove)

These are **not** declared wrong. They are declared unproven.

1. **Moe & Moe 2005 PSS typeset \(T_r/T_i\).** Paywall. AAS 05-258 (opened) defines \(\alpha=(E_i-E_r)/(E_i-E_w)\) only. Walker 2014 (opened) converts \(\alpha\) with \(T_k=V^2/(3R)\). Code uses \(s^2/2\). I will not claim the 2005 journal equation was verified.
2. **Sentman 1961 LMSC-448514.** DTIC PDF was blocked this session. Incident integrals were verified via Walker and Pilinski, who attribute them to Sentman.
3. **Schaub & Junkins 2018, Vallado 2013, Montenbruck & Gill 2000, Tsiotras 1996, Shuster 1993, Stickler & Alfriend 1976, Doornbos 2012 (book), Picone 2002, Tapley GGM03 notes, WMM 2020 report, Bowring 1976 original, Hairer 1993, NIMA TR8350.2.** Cited in-tree; **not opened here.** Where an opened secondary source restated the same formula, that is what was used.
4. **WMM coefficient-level correctness.** Sanity-only. Recurrence labelled approximate in source.
5. **V1.7 wall-clock speedup factor.** Not measured.
6. **Power-model watts and watt-hours.** Not traced to hardware documents.
7. **Optical-coefficient presets** (`al_mylar` etc. in `sail_optics.py`). Partition \(c_a+c_s+c_d=1\) is imposed; the specific triples were not checked against a materials table.
8. **MPC scores.** Two-body surrogate; not a claim about the plant.
9. **Species-resolved Sentman.** Single \(\bar m\). Literature often sums species. Not implemented; not claimed.
10. **Pines evaluator “queued”.** Spec says queued. Still queued.

---

## 10. Equations that **did** check out (opened source + numeric)

Use these as the clean set.

- \(k_B\) exact (NIST).
- \(\omega_\oplus\) (IERS / EOP-PC).
- GGM03S header \(\mu, R_E, \bar C_{20}\to J_2\).
- Two-body and Cartesian J2 (Curtis/Niemeyer + FD).
- Cartesian J3 vs Niemeyer (unused on the no-GGM fallback).
- GGM03S acceleration vs Basilisk off the pole, degrees 2/4/8/20, \(\sim 10^{-16}\).
- Third-body (Curtis 12.10 / poliastro).
- Sentman incident \(C_p, C_\tau\) and re-emission \(\sqrt{T_r/T_i}\) structure (Walker, Pilinski).
- \(\alpha_E\) *definition* (Moe AAS 05-258 eq. 1).
- Face-on \(C_D(\alpha_E=0.93)/C_D(\alpha_E=1)\approx 1.073\) (V1.7’s 6.5–8.5 % band).
- MRP \(\boldsymbol{\sigma}=\hat n\tan(\Phi/4)\), \(\dot{\boldsymbol{\sigma}}\), DCM, 90° map (Crassidis 1996).
- RK4 weights as written; stage-wise gravity in the live `Simulator` path.
- Co-rotating \(\mathbf{v}_\mathrm{rel}\) formula.
- Bowring *behaviour* (0.2 mm), original paper not opened.
- IAU 1982 linear GMST terms.
- Optical-plate special cases \(P\), \(2P\), \(P(1+2/3)\).
- Cylindrical eclipse classroom points.
- pymsis calling convention and output layout.
- MSIS 400 km density in the expected \(10^{-12}\) band.
- Ground-station ECEF→ECI rotation consistent with `dcm_EN`.

---

## 11. Speed (what can be said)

On this machine, one Gym advisor step (150 × 2 s physics, hex panels, GGM degree 4, one pymsis call) is **2.1 ms**. C++ 72-panel Sentman is **3.6 µs**. A Python scalar Sentman is about **11×** slower than the C++ scalar, which is *not* the fused-step comparison.

That is enough to say the inner loop is no longer interpreted Python. It is not enough to print “5–20× versus V1.7” without a paired V1.7 timing.

---

## 12. Citations

### Opened in this session (used as authorities)

1. Crassidis, J. L. & Markley, F. L. (1996). Attitude estimation using modified Rodrigues parameters. NASA NTRS 19960035754. Eqs. (7), (8), (11). https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf
2. Mostaza Prieto, D. (2014). PhD thesis, University of Manchester. Chapter 3, eqs. (3.23)–(3.32), Fig. 3.4. https://pure.manchester.ac.uk/ws/files/60830422/FULL_TEXT.PDF
3. Moe, K. & Bowman, B. R. (2005). The effects of surface composition and treatment on drag coefficients of spherical satellites. AAS 05-258. Eq. (1). https://wpdev.spacewx.com/wp-content/uploads/2024/02/AAS_2005_258_Surface_Effects_on_CD_of_Spheres.pdf
4. NIST. CODATA Boltzmann constant \(k=1.380649\times 10^{-23}\,\mathrm{J\,K^{-1}}\) (exact). https://physics.nist.gov/cgi-bin/cuu/Value?k
5. IAU 1982 GMST linear terms as given by Meeus, quoted at https://astrogreg.com/snippets/greenwichMeanSiderealTime1982.html (Aoki et al., *A&A* 105, 359–366, 1982, is the underlying paper; the \(T^2,T^3\) terms are omitted in code).
6. Niemeyer, K. Orbit perturbations (Cartesian J2, J3). https://kyleniemeyer.github.io/space-systems-notes/orbital-mechanics/orbit-perturbations.html
7. poliastro perturbation docs (Curtis 12.30 J2; Curtis 12.10 third body; drag \(\tfrac12\rho v_\mathrm{rel}(C_D A/m)\mathbf{v}_\mathrm{rel}\)). https://docs.poliastro.space/en/stable/autoapi/poliastro/core/perturbations/index.html
8. GGM03S coefficient file header (runtime path `/home/nekolny/Thesis/basilisk/dist3/Basilisk/supportData/LocalGravData/GGM03S.txt`; same bytes in `ARLAMX-V1.7/support/GGM03S.txt`).
9. pymsis 0.12 `calculate` reference. https://swxtrec.github.io/pymsis/reference/generated/pymsis.msis.run.html
10. IERS / Paris Observatory EOP-PC, mean Earth rotation \(\Omega=7.2921150\times 10^{-5}\,\mathrm{rad\,s^{-1}}\). https://archive.fo/2024.11.16-024853/https://hpiers.obspm.fr/eop-pc/models/constants.html
11. Pulecchi, T., Lovera, M. & Varga, A. Classical vs modern magnetic attitude control. Dipole magnitude \(7.943\times 10^{15}\,\mathrm{Wb\,m}\), tilt \(11.44^\circ\). https://elib.dlr.de/55648/1/varga-mag_control_appl_art2.pdf
12. Pilinski, M. (PhD, Colorado). Sentman \(C_n, C_t\) eqs. 4.7–4.8, as retrieved from https://scholar.colorado.edu/downloads/cf95jb65j
13. Tapley, B. et al. GGM03S (2007 AGU abstract) as the *name* of the model; coefficients taken from the file in (8), not from the CSR notes PDF (fetch failed).

### Named but not opened (do not treat as verified page citations)

Sentman (1961) DTIC AD0265409; Moe & Moe (2005) *Planet. Space Sci.* 53:793; Doornbos (2012) Springer; Vallado (2013) 4th ed.; Montenbruck & Gill (2000); Schaub & Junkins (2018); Shuster (1993) *J. Astronaut. Sci.* 41:439; Tsiotras (1996) *JGCD* 19:772; Stickler & Alfriend (1976) *JSR* 13:282; Avanzini & Giulietti (2012) *JGCD* 35:1326 (quoted by opened secondaries); Bowring (1976) *Survey Review* 23:323; Picone et al. (2002) *JGR*; Chulliat et al. WMM 2020 report; Hairer, Nørsett & Wanner (1993); Pines (1973) *AIAA J.* 11:1508; McInnes (1999); NIMA TR8350.2.

### Implementation referee (not literature)

Basilisk `SphericalHarmonicsGravityModel.computeField` on GGM03S, degrees 2/4/8/20.

---

## 13. Pass summary

**Pass 1.** The C++/Python cut is the one the approval plan described. M-01 incident+re-emission structure is in the plant. M-04 stations rotate. M-03 was inverted, not copied. Visible-surface, ONNX, and SRP moment were not brought into the live loop.

**Pass 2.** J2, two-body, third-body, MRP DCM/kinematics, GMST linear terms, \(k_B\), \(\omega_\oplus\), GGM header, Sentman *incident* integrals, and \(\alpha_E\)’s energy definition all match opened sources. The M-01 \(s^2/2\) closure, WMM recurrence, Pines gravity, tilted dipole, and optical-plate textbook pages do not have opened-primary confirmation.

**Pass 3.** 74 tests pass. Off-pole gravity is machine-precise vs Basilisk. J2 matches a potential finite difference to \(5\times 10^{-10}\). Fused step is 2.1 ms. Polar gravity, WMM, and V1.7 timing are the holes in the test net.

**Pass 4.** Spec and code disagree on Pines, dipole tilt, atmosphere-in-C++, SC_v3 co-rotation, geodetic vs spherical altitude, and the hyperthermal numeric gate.

**Pass 5.** Gym defaults hide F-02 (Moon) and WMM, and expose F-01, F-05, F-06, F-07. Reward SC_v3 is mostly as specified (momentum term missing). Geometry simplifier is area-preserving on the cube/hex tests and is not an occlusion model.

---

## 14. What this audit did not do

- No source file was edited or deleted.
- V1.7 was not modified.
- V1.7 was not re-timed.
- Moe & Moe 2005 PSS, Sentman 1961, Vallado, Montenbruck, Schaub, and the WMM 2020 report were not read as PDFs.
- No new tests were added.
- No training run was launched.

If a later pass is used to *change* the plant, the high-severity items to treat first are F-01 (dipole vs WMM/tilt), F-06 (co-rotation default vs bake-off), F-05 (300 s atmosphere hold), F-07 (two altitudes), F-03 (polar harmonics if polar orbits matter), and F-11 (read Moe & Moe 2005 before treating \(s^2/2\) as journal-certified).
