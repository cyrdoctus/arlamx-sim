# 04 — SPICE ephemeris and third-body gravity

**Folder:** `cpp/orbit/` (`third_body.hpp`)  
**Status:** implemented  
**Tests:** `tests/orbit/`

## What it does

Loads NASA SPICE kernels and returns Earth-centered inertial positions of the **Sun** and **Moon** (and Earth orientation for the harmonic frame). Optional point-mass third-body acceleration on the spacecraft.

SC_v3 bake-off leaves third-body gravity **off**, but still uses Sun position for SRP and eclipse.

## Equations

Third-body (point mass \(k\), barycentric-consistent Earth-centered form):

\[
\mathbf{a}_{3B}=\mu_k\left(
\frac{\mathbf{r}_k-\mathbf{r}}{|\mathbf{r}_k-\mathbf{r}|^3}
-\frac{\mathbf{r}_k}{|\mathbf{r}_k|^3}
\right)
\]

Cylindrical eclipse (same idea as V1.7 RK4):

- sunward hemisphere: \(\varepsilon=1\)
- else if \(|\mathbf{r}-(\mathbf{r}\cdot\hat{s})\hat{s}|<R_E\): \(\varepsilon=0\)
- else \(\varepsilon=1\)

Kernels (data, not source): `naif0012.tls` (or current LSK), `de430.bsp` (or Basilisk’s bundled ephemeris as a path), Earth PCK for ECEF.

Library: **CSPICE** (C). Python does not parse SPK.

## Citations

1. Montenbruck, O. & Gill, E. (2000). *Satellite Orbits.* Springer, §3.3 (third body, solar ephemeris).
2. Acton, C. H. (1996). Ancillary data services of NASA’s Navigation and Ancillary Information Facility. *Planet. Space Sci.* 44(1), 65–70. (SPICE)
3. (supporting) Vallado, D. A. (2013), §8 (third-body perturbation).

## Classroom test

**A.** At a documented JPL horizon time (e.g. 2026-01-15T00:00:00 TDB), Sun unit vector from CSPICE vs Montenbruck low-precision analytic: angle \(\lt 0.5^\circ\) (analytic is coarse; this only checks we loaded the kernel and the frame).  
**B.** Third-body formula at \(\mathbf{r}=\mathbf{0}\) is undefined — API must reject. At a circular 400 km point with a fake Sun on \(+X\) at 1 AU, \(\mathbf{a}_{3B}\) matches a hand-evaluated scalar to 1e-12.  
**C.** Eclipse: spacecraft at \(r=-2R_E\hat{s}\) → 0; at \(+2R_E\hat{s}\) → 1.

## Tests run

2026-08-18 — `tests/orbit/test_third_body.py` **PASS** (hand-eval 3rd body, eclipse, analytic Sun). CSPICE is linked when found; analytic fallback remains.
