# 16 — Quaternion feedback pointing

**Folder:** `cpp/control/` (`quat_feedback.hpp`)  
**Status:** implemented  
**Tests:** later-agent classroom (not written in v2.6)

## What it does

Optional inner-loop law (`SimParams.attitude_law = "quaternion"`, Gym `controller="quaternion"`). **MRP remains the default.** Same magnetorquer path as MRP-PD unless `ideal_torque=true`.

## Equations

Schaub error quaternion \(q_{BR}=q_{BN}\otimes q_{RN}^*\) (`quat_mul(q, quat_conj(q_tgt))`), shortest-path flip of the vector part, then

\[
\boldsymbol{\tau}=-k_p\,\mathbf{q}_v-k_d(\boldsymbol{\omega}-\boldsymbol{\omega}_\mathrm{tgt})+\boldsymbol{\omega}\times I\boldsymbol{\omega}
\]

Same yaml `kp`/`kd` as MRP-PD. Because \(q_v\approx\theta/2\) while \(\sigma\approx\theta/4\), that numerical `kp` is about **2× stiffer** at small angle. That is a unit difference, not a bug.

Rate cap and optional `max_torque` clip match the MRP controller.

## Citations

1. Wie, B. (2008). *Space Vehicle Dynamics and Control* — quaternion PD.
2. Schaub & Junkins (2018) Ch. 8 — \(q_{BR}\) product order.
