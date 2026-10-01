# 01 — Sentman free-molecular flow

**Folder:** `cpp/aero/`  
**Status:** implemented  
**Tests:** `tests/aero/`

## What it does

Maps one flat panel + flow state to pressure \(C_p\) and shear \(C_\tau\), then sums force and torque on the spacecraft. This term is added **inside** the plant force model every substep. **Default GSI** (`SimParams.gsi = "sentman"`). The optional Walker–CLL path is [`17_aero_cll.md`](17_aero_cll.md); both share `sum_panels`.

## Equations

Speed ratio \(s = |v| / \sqrt{2 k_B T / \bar{m}}\).  
\(\gamma = s\cos\theta\), \(Z = 1+\mathrm{erf}(\gamma)\), \(E=e^{-\gamma^2}\).

\[
C_{p,i}=\frac{(\gamma^2+\tfrac12)Z+\gamma E/\sqrt{\pi}}{s^2},\quad
C_{\tau,i}=\sin\theta\cdot\frac{\gamma Z+E/\sqrt{\pi}}{s}
\]

Energy accommodation (V1.7 M-01 / Moe & Moe energy-flux definition):

\[
\frac{T_r}{T_i}=\alpha_E\frac{T_w}{T_i}+(1-\alpha_E)\frac{s^2}{2}
\]

\[
C_{p,r}=\sqrt{T_r/T_i}\,\frac{E+\sqrt{\pi}\,\gamma Z}{2s^2},\quad
C_p=C_{p,i}+C_{p,r},\quad C_\tau=C_{\tau,i}
\]

\[
\mathbf{F}=A q(-C_p\hat{n}+C_\tau\hat{t}),\quad
\boldsymbol{\tau}=\mathbf{r}_c\times\mathbf{F},\quad
q=\tfrac12\rho|v|^2
\]

**Wetted faces (v2.1).** Every panel with \(\gamma = s\cos\theta > -4\) is summed, back faces included; the erf/exp terms switch the flux off on the lee side. An exactly edge-on face keeps its free-molecular skin friction \(C_\tau = 1/(s\sqrt{\pi})\) (≈0.07 per face at \(s\approx8\)). v2.0 summed only \(\cos\theta>0\), which made the force discontinuous at grazing incidence and dropped both faces of an edge-on membrane: on the hex sail the min-drag \(C_D\) was 0.079 instead of 0.214.

\[
C_D=\mathbf{F}\cdot\hat{v}/(q A_\mathrm{ref}),\quad
C_L=|\mathbf{F}_\perp|/(q A_\mathrm{ref})
\]

Hyperthermal check: \(C_p\cos\theta+C_\tau\sin\theta\to 2\cos\theta\) as \(s\to\infty\).

**Do not implement** \(T_r/T_i=1+\alpha_E(T_w/T_i-1)\). That is the pre-V1.7 bug.

## Citations

1. Sentman, L. H. (1961). *Free Molecule Flow Theory…* LMSC-448514, DTIC AD0265409.
2. Moe, K. & Moe, M. M. (2005). Gas–surface interactions and satellite drag coefficients. *Planet. Space Sci.* 53(8), 793–801. doi:10.1016/j.pss.2005.03.005.
3. (supporting) Doornbos, E. (2012). *Thermospheric Density and Wind Determination…* Springer, eqs. 3.53–3.55.

## Classroom test

**Plate, face-on, \(\alpha_E=1\), \(s=8\), \(T_w=T_i\).**  
Hyperthermal identity must hold to 1e-3 at \(s=8\) and to 1e-8 at \(s=80\).  
At \(\alpha_E=1\), \(T_r/T_i\) must equal \(T_w/T_i\) exactly (M-01 and the old formula agree here).  
At \(\alpha_E=0\), \(T_r/T_i\) must equal \(s^2/2\) (old formula fails this).

Second case: two-sided unit square, flow along \(-\hat{n}\), \(A_\mathrm{ref}\) one-sided → \(C_D\) in the published fully-diffuse face-on band (~2.1 at \(\alpha_E=1\), higher at 0.93).

## Tests run

2026-08-18 — `tests/aero/test_sentman.py` **PASS** (hyperthermal residual < 0.03 at s=80 with re-emission; M-01 α=0 hotter than α=1; face-on Cd in 1.8–2.4).
2026-09-04 (v2.1) — `tests/validation/verify_physics.py` **PASS**: \(C_p, C_\tau\) vs the textbook \(\cos\theta\) form to 3e-16, incident \(C_p\) vs direct Maxwellian quadrature to 7e-16, panel sums vs a hand loop to 1e-16. Edge-on faces now wetted; `test_prescribed_decay` min-drag pin moved from <0.12 to 0.15–0.30.
