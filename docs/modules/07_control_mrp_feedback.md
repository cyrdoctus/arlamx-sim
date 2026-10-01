# 07 — MRP feedback pointing controller

**Folder:** `cpp/control/` (`mrp_feedback.hpp`)  
**Status:** implemented  
**Tests:** `tests/control/`

## What it does

Inner-loop pointing law used by the advisor (`attitude_law = "mrp"`, default). Production convention is **proper** inverse (\(-\boldsymbol{\sigma}\)). Optional torque saturation and plant-consistent body-rate cap.

**v2.5+ plant path:** `compute` returns \(\boldsymbol{\tau}_\mathrm{cmd}\). Unless `ideal_torque=true`, [`apply_magnetorquer`](15_control_magnetorquer.md) maps that demand to a dipole and applies \(\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}\). The quaternion law is the same outer path with a different error vector ([`16_control_quat_feedback.md`](16_control_quat_feedback.md)).

## Equations

\[
\boldsymbol{\tau}=-K\boldsymbol{\sigma}_e-P\boldsymbol{\omega}_e+\boldsymbol{\omega}\times I\boldsymbol{\omega}
\]

\(\boldsymbol{\sigma}_e\) = MRP composition of \(\boldsymbol{\sigma}_{BN}\) with inverse(\(\boldsymbol{\sigma}_\mathrm{cmd}\)).

Proper inverse: \(-\boldsymbol{\sigma}\).  
Legacy inverse (replay only): \(-\boldsymbol{\sigma}/\sigma^2\). Default is proper.

Rate cap: predict the next \(\boldsymbol{\omega}\) with the **PD part only** (gyroscopic feedforward cancels the plant term), then rebuild \(\boldsymbol{\tau}\) if the cap is exceeded.

## Citations

1. Schaub, H. & Junkins, J. L. (2018). *Analytical Mechanics of Space Systems,* 4th ed., Ch. 8 (MRP PD).
2. Tsiotras, P. (1996). Stabilization and optimality results for the attitude control problem. *JGCD* 19(4), 772–779. doi:10.2514/3.21698.

## Classroom test

**A.** \(\boldsymbol{\sigma}=\boldsymbol{\sigma}_\mathrm{cmd}=\mathbf{0}\), \(\boldsymbol{\omega}=\mathbf{0}\) → \(\boldsymbol{\tau}=\mathbf{0}\).  
**B.** Rest-to-rest: non-collinear initial MRP, closed loop on the 12-state plant with this torque, 200 s, angle \(4\arctan|\boldsymbol{\sigma}| < 0.01^\circ\).  
**C.** Composition identity: \(C(\boldsymbol{\sigma}_e)=C_{BN}C_{RN}^{\top}\) to 1e-12.

## Tests run

2026-08-18 — `tests/control/test_mrp_feedback.py` **PASS** (zero-error torque, closed-loop regulation).
