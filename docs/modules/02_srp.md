# 02 — Panel solar radiation pressure

**Folder:** `cpp/srp/`  
**Status:** implemented  
**Tests:** `tests/srp/`

## What it does

Attitude-dependent radiation pressure on the same panel set as aero. Two solar laws:

- **optical plate** (default since v2.7, `srp.optical: true`, `optics: al_mylar`): absorption \(c_a\), specular \(c_s\), diffuse \(c_d\), \(c_a+c_s+c_d=1\). The normal component is the orbit-raising term.
- **cannonball** (`srp.optical: false`, V1.7 law): lumped \(C_r\), force only along \(-\hat s\). Kept for the \(C_r=1\) classroom checks and for old-campaign comparisons.

Plus Earth radiation (`srp.earth: true`), summed per plate with the same optical law, source direction nadir:

- **Earth IR**, on in eclipse: \(P_\mathrm{IR}=P_\mathrm{SR}\,\tfrac{e_0}{4}\,(R_E/r)^2\), thermal partition `srp.ir` (\(c_a=0.85, c_s=0, c_d=0.15\) [ASSUME]).
- **Albedo**, day side only: \(P_\mathrm{alb}=P_\odot\,a_0\,(R_E/r)^2\max(0,\hat r\cdot\hat s)\), solar partition.

\(a_0=0.34\), \(e_0=0.68\) (Knocke, Ries & Tapley 1988). \((R_E/r)^2\) is the configuration factor of a plate facing the Earth's centre; tilted plates use the same point-source direction.

v2.7 (criticv1 F1/F2/F6): every force **and its moment** enters the plant (`tau_srp_mean`, `tau_erp_mean`, summed in `tau_env_mean`). The pressure scales with heliocentric distance, \(P_\odot=P_\mathrm{SR}(1\,\mathrm{AU}/d)^2\), where \(d\) comes from the Astronomical Almanac low-precision series (Vallado 2013 Alg. 29) or CSPICE. Every campaign before v2.7 flew cannonball with no moment, no \(1/d^2\), and no Earth radiation.

## Equations

Cannonball:

\[
\mathbf{F}=\sum_{\hat{n}\cdot\hat{s}>0}
P_\mathrm{SR}\,C_r\,A\,(\hat{n}\cdot\hat{s})\,(-\hat{s})\,\varepsilon_\mathrm{ecl}
\]

Optical plate (Montenbruck & Gill §3.4):

\[
\mathbf{F}_i=-P A\cos\theta\Bigl[(c_a+c_d)\hat s+\bigl(2c_s\cos\theta+\tfrac23 c_d\bigr)\hat n\Bigr]
\]

\(P_\mathrm{SR}=4.56\times10^{-6}\,\mathrm{Pa}\) at 1 AU. \(\varepsilon_\mathrm{ecl}\in[0,1]\).  
Moment: \(\boldsymbol{\tau}=\sum\mathbf{r}_c\times\mathbf{F}_i\). Back faces contribute 0.

Presets: `python/arlamx_v2/sail_optics.py`. Sweep: `python -m arlamx_v2.optics_run`.

## Citations

1. Montenbruck, O. & Gill, E. (2000). *Satellite Orbits.* Springer, §3.4.
2. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications,* 4th ed., §8.6.4; Alg. 29 (Sun).
3. McInnes, C. R. (1999). *Solar Sailing.* Springer (coating partition).
4. Knocke, P. C., Ries, J. C. & Tapley, B. D. (1988). Earth radiation pressure effects on satellites. AIAA 88-4292.
5. Howell, J. R. *A Catalog of Radiation Heat Transfer Configuration Factors* (plate facing a sphere).

## Classroom test

One 1 m² plate, Sun along \(-\hat{n}\), \(\varepsilon=1\), \(C_r=1\):  
\(|\mathbf{F}|=P_\mathrm{SR}=4.56\times10^{-6}\,\mathrm{N}\), direction \(-\hat{s}\).  
Rotate 90°: \(\mathbf{F}=\mathbf{0}\).  
Two opposite plates of a closed box: only the sunward face contributes; moment about a centered CoM is zero if the face is symmetric.

## Tests run

2026-08-18 — `tests/srp/test_srp.py` **PASS**.
2026-09-25 — `tests/physics/test_disturbances.py` (F1 moment = r × F, 1 cm offset → 0.01 P; F2 face-on hex mylar 5.692 µN, 90° plate 0, cannonball \(C_r=1\) = \(P_\mathrm{SR}\), 1/d² series; F6 IR pushes a nadir plate outward in eclipse, deep-space plate gets 0, albedo day side only, switch leaves the solar term untouched) **PASS**.
