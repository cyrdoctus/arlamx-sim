# 03 — Two-body gravity and Earth spherical harmonics

**Folder:** `cpp/orbit/` (`gravity.hpp`)  
**Status:** implemented  
**Tests:** `tests/orbit/`

## What it does

Earth-centered acceleration: point mass plus the **same field Basilisk uses** — GGM03S Stokes coefficients, fully normalized, truncated at a configurable degree and order \(N\) (library default 20; SC_v3 bake-off uses 4). Evaluated in Earth-fixed coordinates, then rotated to inertial.

This is **not** wrapping Basilisk. It is the same *model* (same coefficient file, same potential).

## Equations

Geopotential (Montenbruck / Vallado; \(n\) from 2 to \(N\)):

\[
U=\frac{\mu}{r}\sum_{n=2}^{N}\sum_{m=0}^{n}
\left(\frac{R_E}{r}\right)^n \bar{P}_{nm}(\sin\phi)
\bigl[\bar{C}_{nm}\cos m\lambda+\bar{S}_{nm}\sin m\lambda\bigr]
\]

Acceleration \(\mathbf{a}_E=\nabla U\) in ECEF, then \(\mathbf{a}_N=C_{NE}\mathbf{a}_E\).

Zonal-only check (degree 2, only \(J_2=-\bar{C}_{20}\sqrt{5}\)):

\[
f_2=\tfrac32 J_2\mu R_E^2/r^5,\quad
a_x=f_2 x(5z^2/r^2-1),\ \ldots
\]

Implementation (`gravity.cpp`): fully normalised associated Legendre \(\bar{P}_n^m(\sin\phi)\) in geocentric latitude / longitude, with a **pole guard**. When \(\cos\phi < 10^{-14}\), \(\lambda\) is set to 0 and \(\partial\bar{P}/\partial\phi\) keeps only zonals. This is not Pines or Cunningham; the polar axis stays finite by that guard. Degree is capped at `SH_MAX_DEGREE = 70`. `sh_degree < 2` (physics `twobody`) returns the two-body term only.

A non-empty `ggm_path` that fails to load throws at `Simulator` construction (same honesty as WMM). Empty path + `sh_degree < 2` is analytic point mass. Empty path + `sh_degree >= 2` is the historical two-body+J2+J3 fallback and is warned from the Gym env.

`GGM03S.txt` is loaded as **data** (path configurable). Do not copy Basilisk C++.

## Citations

1. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications,* 4th ed., §8 (spherical harmonics / J2).
2. Montenbruck, O. & Gill, E. (2000). *Satellite Orbits.* Springer, §3.2.
3. (coefficients) Tapley, B. D. et al. (2005). GGM02 — GRACE gravity models. *J. Geodesy* / CSR GGM03S notes. Use the header of the GGM03S file as the coefficient authority.
4. Vallado / Montenbruck associated-Legendre recursion; pole guard as implemented (not Pines). Fully normalised recursion: Holmes, S. A. & Featherstone, W. E. (2002). *Journal of Geodesy* 76, 279–299.
5. Reference implementation for this field: `SphericalHarmonicsGravityModel` in Basilisk, Autonomous Vehicle Systems Laboratory, University of Colorado Boulder, https://github.com/AVSLab/basilisk (ISC license). The ARLAMX evaluator is not a copy of that source and does not link it. Reuse of `cpp/src/orbit/gravity.cpp` should cite Basilisk alongside ARLAMX.

## Classroom test

**A.** Point mass at \(r=(R_E+400\,\mathrm{km},0,0)\): \(a=-\mu/r^2\) along \(\hat{r}\), error ≤ 1e-14 relative.  
**B.** J2-only vs the closed Cartesian J2 formula at a non-equatorial LEO point: ≤ 1e-8 relative.  
**C.** Degree-2 GGM03S vs J2 closed form (after the \(\bar{C}_{20}\leftrightarrow J_2\) conversion printed in the GGM03S header): ≤ 1e-7 relative.  
**D.** A polar point (\(\phi=90^\circ\)) must stay finite.

## Tests run

2026-08-18 — `tests/orbit/test_gravity.py` **PASS**.  
`tests/basilisk_ref/test_gravity_vs_basilisk.py` **PASS** at degree 2/4/8 vs Basilisk `computeField` (relative error ~1e-16 off-axis). Exact polar axis uses the zonal-only guard above. Plant does **not** import Basilisk.
