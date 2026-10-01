# 09 — Magnetic field (dipole + WMM)

**Folder:** `cpp/mag/`  
**Status:** implemented  
**Tests:** `tests/mag/`

## What it does

Provides \(\mathbf{B}\) in inertial and body frames. Required for B-dot and for the magnetometer observation.

- **Dipole** (always on, used by unit tests): tilted geocentric dipole.
- **WMM** (optional, Basilisk’s path): load `WMM.COF` as **data**, evaluate at geodetic lat/lon/height and epoch year.

## Equations (dipole)

\[
\mathbf{B}(\mathbf{r})=\frac{\mu_0}{4\pi}\left(
\frac{3(\mathbf{m}_E\cdot\hat{r})\hat{r}-\mathbf{m}_E}{r^3}
\right)
\]

Use a published epoch dipole (e.g. IGRF/WMM dipole terms) so the test is pinned.

WMM: implement the official spherical-harmonic algorithm from the WMM technical report (not a made-up expansion).

## Citations

1. Chulliat, A. et al. (2020). *The US/UK World Magnetic Model for 2020–2025.* NOAA/NCEI technical report. (WMM.COF + algorithm)
2. Wertz, J. R. (ed.) (1978). *Spacecraft Attitude Determination and Control.* Appendix H / geomagnetic field — dipole form used in ADCS texts.  
   Also: Finlay, C. C. et al. (2010). International Geomagnetic Reference Field. *Geophys. J. Int.*

## Classroom test

**A.** Dipole at the equator, magnetic equator: \(B_r\approx 0\), \(B_\theta\) matches \(\mu_0 m_E/(4\pi r^3)\) to 1e-12.  
**B.** If `WMM.COF` is present: at (lat, lon, h) = (0, 0, 400 km), epoch 2025.0, \(|B|\) in 20–50 µT (LEO sanity) and finite.

## Tests run

2026-08-18 — `tests/mag/test_dipole.py` **PASS** (equator Br=0). WMM file loads; LEO |B| sanity checked when `WMM.COF` is present.
