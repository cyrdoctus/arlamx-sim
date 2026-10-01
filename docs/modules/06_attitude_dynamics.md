# 06 — Attitude kinematics, Euler equation, RK4

**Folder:** `cpp/attitude/`  
**Status:** implemented  
**Tests:** `tests/attitude/`

## What it does

12-state plant: \(\mathbf{y}=[\mathbf{r},\mathbf{v},\boldsymbol{\sigma},\boldsymbol{\omega}]\). Classical RK4. Body force/torque held ZOH; DCM rebuilt at every RK4 stage. MRP shadow after each completed step.

## Equations

MRP → DCM (Schaub):

\[
C(\boldsymbol{\sigma})=I+\frac{8[\tilde{\sigma}]^2-4(1-\sigma^2)[\tilde{\sigma}]}{(1+\sigma^2)^2}
\]

\[
\dot{\boldsymbol{\sigma}}=\tfrac14\bigl[(1-\sigma^2)I+2[\tilde{\sigma}]+2\boldsymbol{\sigma}\boldsymbol{\sigma}^\top\bigr]\boldsymbol{\omega}
\]

\[
\boldsymbol{\sigma}\leftarrow-\boldsymbol{\sigma}/\sigma^2\quad\text{when }\sigma^2>1
\]

\[
\dot{\boldsymbol{\omega}}=I^{-1}(\boldsymbol{\tau}-\boldsymbol{\omega}\times I\boldsymbol{\omega})
\]

RK4: \(\mathbf{y}_{n+1}=\mathbf{y}_n+(h/6)(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4)\).

## Citations

1. Schaub, H. & Junkins, J. L. (2018). *Analytical Mechanics of Space Systems,* 4th ed., AIAA, Ch. 3–4.
2. Shuster, M. D. (1993). A survey of attitude representations. *J. Astronaut. Sci.* 41(4), 439–517.
3. (RK4) Hairer, E., Nørsett, S. P. & Wanner, G. (1993). *Solving Ordinary Differential Equations I,* 2nd ed., §II.1.

## Classroom test

**A.** \(\boldsymbol{\sigma}=\mathbf{0}\Rightarrow C=I\).  
**B.** 90° about \(\hat{z}\): MRP \(\sigma_z=\tan(22.5^\circ)\); \(C\) maps \(\hat{x}\to\hat{y}\) to 1e-12.  
**C.** Torque-free principal spin: \(\boldsymbol{\omega}=(0,0,\omega_0)\), diagonal \(I\), \(\boldsymbol{\omega}\) constant, \(\boldsymbol{\sigma}\) integrates to the analytic rotation.  
**D.** Shadow: start with \(\sigma^2>1\), one step later \(\sigma^2\le 1\).

## Tests run

2026-08-18 — `tests/attitude/test_mrp.py` **PASS**. `tests/integration/test_plant_step.py` **PASS** (vacuum SMA, drag removes energy).

## Environmental torques (v2.7)

\(\boldsymbol\tau_B=\boldsymbol\tau_\mathrm{aero}+\boldsymbol\tau_\mathrm{SRP}+\boldsymbol\tau_\mathrm{Earth\,rad}+\boldsymbol\tau_\mathrm{gg}+\boldsymbol\tau_\mathrm{ctrl}\), zero-order held over each 2 s substep. Gravity gradient (`physics gravity.gradient_torque`):

\[
\boldsymbol\tau_\mathrm{gg}=\frac{3\mu}{r^3}\,\hat u\times I\hat u,\qquad \hat u=-C\hat r_N
\]

(Markley & Crassidis 2014, ch. 3). Peak on this inertia at 400–500 km is ≈ 24 nN·m. Prescribed mode still holds σ and zeros ω, so no disturbance moment integrates there. Check: `tests/physics/test_disturbances.py::test_f5_gravity_gradient` (r along a principal axis gives 0; r at 45° in X–Z gives \(3\mu/r^3\cdot\tfrac12|I_{zz}-I_{xx}|\) to 1e-12).

