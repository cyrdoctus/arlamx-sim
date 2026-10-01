# 17 — Walker–CLL free-molecular flow

**Folder:** `cpp/src/aero/cll.cpp` (header `cpp/include/arlamx/aero/cll.hpp`)  
**Status:** implemented (optional GSI)  
**Tests:** [`docs/CLL_TEST_SPEC.md`](../CLL_TEST_SPEC.md); wiring in `tests/aero/test_cll.py` and `tests/physics/test_physics_wiring.py`.

## What it does

Named alternative to Sentman (`SimParams.gsi = "cll"`). Same `sum_panels`, same \(A_\mathrm{ref}\), same inertial \(\mathbf{F}\cdot\mathbf{v}\). **Sentman stays the default.** The onboard FP32 propagator does **not** use CLL.

This is **not** a Monte-Carlo sample of the Cercignani–Lampis kernel. It is Walker, Mehta & Koller (2014) modified closed-form plate coefficients as transcribed by Sinpetru / ADBSat (arXiv:2104.05543 eqs. 11–15).

`alpha_E` is Sentman-only. Under CLL the live knobs are \(\alpha_N\) (Walker normal-energy accommodation) and \(\sigma_T\). These are not Sentman \(\alpha_E\) (Moe & Moe 2005 energy accommodation). Walker 2014 Table 2 / ADBSat treat \(\alpha_N\) as a free input the species fits are functions of; there is no published “LEO \(\alpha_N\)”.

The `high` preset uses \(\alpha_N=0.93\) as an **isolation choice**: the same number as the campaign Sentman \(\alpha_E\), so standard-vs-high isolates the kernel and the species mix. It is not a Walker-derived value. Face-on plate at \(s=8\), \(T_w/T_i=1/3\): Sentman \(0.93\) \(C_D\approx 2.37\); CLL \(\alpha_N=0.93\) \(\approx 2.56\) (\(+8\%\)); CLL \(\alpha_N=1\) (Schaaf–Chambre = Sentman \(\alpha_E=1\)) \(\approx 2.15\) (\(-9\%\)). Sweep \(\alpha_N\) if you need that sensitivity.

## Equations

\(\gamma=s\cos\theta\), \(Z=1+\mathrm{erf}(\gamma)\), \(E=e^{-\gamma^2}\).

\[
\Gamma_1=\frac{\gamma E}{\sqrt{\pi}}+\tfrac12(1+2\gamma^2)Z,\qquad
\Gamma_2=\frac{E}{\sqrt{\pi}}+\gamma Z
\]

If \(\alpha_N<1\) (species \(j\), Walker fits \(\beta_j,\gamma_j,\delta_j,\zeta_j\)):

\[
C_{p,j}=\frac{1}{s^2}\Biggl[\bigl(1+\sqrt{1-\alpha_N}\bigr)\Gamma_1
+\tfrac12 e^{-\beta_j(1-\alpha_N)^{\gamma_j}}\Bigl(\frac{T_w}{T}\Bigr)^{\delta_j}\frac{\zeta_j}{s}
\sqrt{\frac{T_w}{T}}\,\sqrt{\pi}\,\Gamma_2\Biggr]
\]

\[
C_{\tau,j}=\frac{\sigma_T\sin\theta}{s}\Gamma_2
\]

If \(\alpha_N=1\): \(C_p\) is Schaaf–Chambre (ADBSat eq. 9) with \(\sigma_N=\alpha_N\); \(C_\tau\) unchanged.

**Behaviour near \(\alpha_N=1\)** (face-on plate, \(s=8\), \(T_w/T_i=1/3\), atomic O, measured from `cpp.cll`, 2026-09-25):

| \(\alpha_N\) | \(C_p\) |
|---|---|
| 0.99 | 2.246 |
| 0.999 | 2.147 |
| 0.9999 | 2.152 |
| 0.99999 | 2.185 |
| \(1-10^{-9}\) | 2.282 |
| Walker limit \(\alpha_N\to1^-\) | ≈ 2.31 |
| 1 (Schaaf–Chambre branch) | 2.144 |

1. The ≈2.3× factor is only the re-emission prefactor \((T_w/T_i)^\delta\zeta/s\) for atomic O, not the plate coefficient.
2. \(\alpha_N=1\) is identical to Sentman \(\alpha_E=1\) (residual 0 face-on and at 69°).
3. The gap between the Walker limit and the Schaaf branch (≈7.6 %) opens only for \(1-\alpha_N\lesssim10^{-4}\); a sweep that steps 0.999 → 1.0 does not jump.

We keep the ADBSat split and do not blend the branches. The high preset sits at 0.93.

Per-species speed ratio \(s_j=|v|/\sqrt{2k_BT/m_j}\). Mixture (eq. 15):

\[
C_p=\frac{1}{M}\sum_j\chi_j m_j C_{p,j},\qquad M=\sum_j\chi_j m_j
\]

Species order: \(\mathrm{He},\mathrm{O},\mathrm{N}_2,\mathrm{O}_2,\mathrm{Ar},\mathrm{H},\mathrm{N}\). Ar reuses the N2 Walker row (ADBSat omits Ar). He and H mid-band (\(0.50<\alpha_N\le 0.90\)) are byte-identical in ADBSat `Fitted_Parameters` (`{3.45, 0.52, 2.4, 0.93}`) — that is the source table, not an ARLAMX copy error. Confirm vs Walker 2014 Table 2 if the JSR PDF is in hand. Without \(\chi_j\), the kernel uses atomic-O fits and the mixture \(s\) from \(\bar{m}\).

Lee-side skip is **per species** (\(s_j\cos\theta\le-4\)). The Sentman path still uses the mixture \(s\). A mixture cut would zero H at \(\gamma_H\approx-1\) (still ~16 % of face-on flux) whenever the O-dominated mix sits at \(\gamma=-4\).

Walker prefactors (\(\exp\), two \(\mathrm{pow}\), \(\sqrt{}\), per-species \(s_j\)) are hoisted once per `sum_panels` call.

## Citations

1. Cercignani, C. & Lampis, M. (1971). Kinetic models for gas–surface interactions. *TTSP* 1, 101–114.
2. Lord, R. G. (1991). Some extensions to the Cercignani–Lampis kernel. *Phys. Fluids A* 3, 706–710.
3. Walker, A., Mehta, P. & Koller, J. (2014). Drag coefficient model using the CLL GSI. *J. Spacecraft Rockets* 51(5), 1544–1563.
4. (open transcription) Sinpetru et al., ADBSat, arXiv:2104.05543, eqs. 11–15.

Do not implement a DSMC kernel sample in the plant loop.
