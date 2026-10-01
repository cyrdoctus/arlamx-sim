# 05 — Atmosphere (MSIS)

**Folder:** `python/arlamx_v2/atmosphere.py` (no C++ MSIS header)  
**Status:** implemented  
**Tests:** exercised through `ArlamxV2Env`

## What it does

Python `query_msis` (pymsis) returns \(\rho\), \(T\), \(\bar{m}\) and, when requested, mole fractions \(\chi_j\) for \(\{\mathrm{He},\mathrm{O},\mathrm{N}_2,\mathrm{O}_2,\mathrm{Ar},\mathrm{H},\mathrm{N}\}\). Anomalous O (pymsis index 8) is folded into atomic O (same mass, same Walker O row). NO (index 9) has no Walker row and is omitted, so \(\chi\) is renormalized over the seven Walker species, not the full MSIS number density. The Gym env / decay runner call `Simulator.set_atmosphere`. Sentman uses \((\rho,T,\bar{m})\) only. Walker CLL uses \(\chi_j\) for the mass-weighted mix.

Physics YAML `atmosphere.model`: `msis21` (pymsis version 2.1), `nrlmsise00` (version 0), or `constant`. There is no C++ atmosphere header and no `set_space_weather` on the plant.

## Equations

\[
\bar{m}=\frac{\sum_i n_i m_i}{\sum_i n_i},\quad
i\in\{\mathrm{He},\mathrm{O},\mathrm{N}_2,\mathrm{O}_2,\mathrm{Ar},\mathrm{H},\mathrm{N}\}
\]

Query at **geodetic** height and **Earth-fixed** longitude (Bowring + GMST), not \(|r|-R_E\) and inertial lon. Since v2.7 the plant's `altitude_km`, the 80 km stop, the 450 km flash cutoff and the onboard-forecast anchor all use that same geodetic height (criticv1 F9). Observation features built on altitude therefore shift by up to ~21 km at high latitude relative to pre-v2.7 runs. The reward weights were **not** rescaled to hide this.

**Time interpolation (F11).** With `atmosphere.interp: true` (standard, high), the env queries MSIS again at the predicted end of the 300 s step. The plant interpolates \(\rho, T, \bar m, \chi\) linearly over the substeps, so there are 2 MSIS calls per decision, not 150. `fast` keeps the single held sample.

**Ap.** A scalar `ap` fills all 7 MSIS slots (steady-Ap scenario, daily-Ap mode, unchanged meaning of the yaml key). A length-7 history (daily, 3 h now, −3, −6, −9 h, mean −12…−33 h, mean −36…−57 h) switches MSIS to storm-time mode (`geomagnetic_activity = −1`).

**Wind.** The relative velocity is \(\mathbf v_\mathrm{rel}=\mathbf v-\boldsymbol\omega_\oplus\times\mathbf r-\mathbf v_\mathrm{wind}\) (`Simulator.set_wind`, inertial axes). MSIS has no wind. The wind is **zero** until a cited model (HWM14 or successor) is supplied; no constant wind is shipped.

Species molar masses / \(N_A\): 2019 SI (same constants V1.7 documents). Fallback \(\bar{m}=m_O\) if \(\sum n_i=0\).

## Citations

1. Picone, J. M., Hedin, A. E., Drob, D. P. & Aikin, A. C. (2002). NRLMSISE-00. *JGR Space Physics* 107(A12), 1468. doi:10.1029/2002JA009430.
2. Emmert, J. T. (2015). Thermospheric mass density: a review. *Adv. Space Res.* 56(5), 773–824.

## Classroom test

**A.** Fixed point: 400 km, lat 0°, lon 0°, 2026-01-15T00:00Z, F10.7=150, Ap=4, MSIS 2.1. \(\rho\) must be in \(10^{-12}\)–\(10^{-11}\,\mathrm{kg\,m^{-3}}\) (LEO sanity) and match a second independent pymsis call to 1e-12 relative.  
**B.** \(\bar{m}\) for a one-species gas of pure O is \(m_O\) exactly.  
**C.** Geodetic vs geocentric altitude: at lat 90°, using \(|r|-R_{eq}\) is ~21 km low; the wrapper must use Bowring height (documented difference, not a free parameter).

## Tests run

2026-08-18 — exercised through `ArlamxV2Env` (pymsis callback). Dedicated MSIS unit test still to add; plant accepts (ρ, T, m̄) from Python.
2026-09-25 — `tests/physics/test_disturbances.py`: Ap history (scalar = legacy fill, storm history raises ρ, wrong length raises), wind hook (zero wind bit-identical, opposing wind raises drag), end-sample interpolation (mean over substeps, end sample held) **PASS**.
