# 15 — Magnetorquer allocation

**Folder:** `cpp/control/` (`magnetorquer.hpp`)  
**Status:** implemented  
**Tests:** later-agent classroom (not written in v2.6)

## What it does

Maps a PD torque demand to coil dipoles and the torque the field can actually produce. Default plant path (`ideal_torque=false`).

## Equations

\[
\mathbf{m}=\frac{\mathbf{B}\times\boldsymbol{\tau}_\mathrm{cmd}}{|\mathbf{B}|^2},\qquad
m_i=\mathrm{clip}(m_i,\pm m_{\max,i}),\qquad
\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}
\]

Zero dipole if \(|\mathbf{B}|\) vanishes. \(\boldsymbol{\tau}\cdot\hat{\mathbf{B}}=0\) by construction.

\[
\mathrm{shortfall}=1-\frac{|\boldsymbol{\tau}|}{|\boldsymbol{\tau}_\mathrm{cmd}|}
\]

includes both the unproducible \(\boldsymbol{\tau}\parallel\mathbf{B}\) part and per-axis clip. `BDot.max_dipole` is a legacy sphere; the plant disables it so only `dipole_max_` binds.

## Closed-loop authority — measured (v2.6 review)

`ideal_torque=false` is the packaged default since v2.5, but **no campaign gain
was ever tuned on it**, and with the v7/v8 gains the default plant does not hold
attitude. Measured on the hex geometry, 350 km, seed 3, 40 advisor steps
(3.3 h), commanded attitude held, full authority:

| plant | gains | i = 23° | i = 60° | i = 90° |
|---|---|---|---|---|
| ideal couple | v7 (kp 4e-4, kd 8e-3) | 0.02° | 0.03° | 0.02° |
| coils through B | v7 | 74° (peak 174°) | 92° | 70° |
| ideal couple | v8 yaml (kp 6e-5, kd 1.6e-3) | 6° | 13° | — |
| coils through B | v8 yaml | 100° (peak 145°) | 120° | — |

(mean tracking error over the last 10 steps.) The same divergence occurs with
the atmosphere and SRP removed and zero initial rate, so it is not the
disturbance: it is the actuator. \(\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}\)
has no component along \(\hat{\mathbf{B}}\). A stiff PD (loop time constant
\(I/k_d\approx 1.6\) s against a 2 s control step) removes the
\(\hat{\mathbf{B}}\)-perpendicular rate in one step and pins
\(\boldsymbol{\omega}\) onto \(\hat{\mathbf{B}}\); the attitude then drifts about
\(\hat{\mathbf{B}}\), the proportional term pushes perpendicular, and the slow
rotation of \(\hat{\mathbf{B}}\) (1–2 mrad/s) folds that push back into the
uncontrolled axis. The projected PD is only stable when the gains are slow
relative to the field rotation (Lovera & Astolfi 2004). Confirmed here: with
\(k_p=4\times10^{-8},\;k_d=8\times10^{-5}\) (three orders softer), no
disturbance, a 20° slew converges to 10–20° over 30 h; with the aero torque on
it does not converge at all at i = 23°. With the v8 cp offset the aero torque
alone turns the vehicle by radians in one quarter orbit, before \(\hat{\mathbf{B}}\)
has moved.

Consequences:

* Every v7–v14 result was produced on `ideal_torque=true` (v2.1 plant) and
  assumes an actuator that can torque about \(\hat{\mathbf{B}}\). The 300 s
  advisor-step slews (40°) are not achievable magnetically.
* Training on the v2.5/v2.6 default (`false`) with the campaign gains trains a
  tumbling vehicle. `attitude.ideal_torque` in the physics YAML makes the choice
  explicit and is frozen in `snapshot.json`; pick it deliberately.
* Magnetic-only 3-axis hold of this sail against its own aero torque needs either
  a momentum device, aerodynamic passive stability (the sail as a shuttlecock),
  or a control law designed for the time-varying \(\mathbf{B}\) — a research
  decision, not a gain sweep. The v2.6 review fixed the two observer/reward
  bugs that this exposed (`gyro_mean`, `tau_demand_mean`) but did **not**
  retune gains.

## Citations

1. Stickler & Alfriend (1976) — magnetic actuation \(\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}\).
2. Lovera, M. & Astolfi, A. (2004). Spacecraft attitude control using magnetic actuators. *Automatica* 40, 1405–1414 — averaging condition for the projected PD.
3. Plant implementation: `cpp/src/control/magnetorquer.cpp`.

## Six beam torque rods (v2.7, variant `v10r6`)

Vehicle card `config/plant/vehicle.yaml` (user, 2026-09-25): 0.750 kg = 80 g PCB + battery at the centre, 6 × 28.3 g rods (one per beam), and 500 g structure. The structure is split between membrane and beams by area share (10.4 % beams) [ASSUME]. Rod k lies along beam k, which runs along a hexagon side: \(\hat u_k=(-\sin\theta_k,\cos\theta_k,0)\), \(\theta_k=60^\circ k\), at the side midpoint (0.433 m). Opposite beams are parallel, so the dipole is **in the membrane plane only**. \(\tau_z\) needs in-plane B; \(\tau_x,\tau_y\) need \(B_z\).

Mass properties (`python/arlamx_v2/vehicle.py`): hexagonal lamina \(I_z=\tfrac{5}{12}mR^2\) (Roark Table A.1), beams as thin rods with the parallel-axis theorem, rods and PCB as point masses. Result: \(I=\mathrm{diag}(0.0436, 0.0436, 0.0872)\,\mathrm{kg\,m^2}\), CoM at the origin, \(I_z=I_x+I_y\) (planar body).

Rod sizing [ASSUME, until the rod design exists]: the v8 air-coil design scaled by coil mass, 0.596 A·m² and 0.0439 W per rod. Torque caps at \(b_\mathrm{ref}\): \(\tau_x\) from \(m_y\), \(\tau_y\) from \(m_x\).

Allocation per substep, over the rods that are **on**:

\[
\mathbf m_\mathrm{des}=\frac{\mathbf B\times\boldsymbol\tau_\mathrm{cmd}}{|\mathbf B|^2},\quad
\mathbf d=U^\top(UU^\top)^+\mathbf m_\mathrm{des},\quad
\mathbf d\leftarrow\mathbf d\,\min\Bigl(1,\min_k\frac{d_{\max,k}}{|d_k|}\Bigr),\quad
\boldsymbol\tau=(U\mathbf d)\times\mathbf B
\]

(Moore–Penrose minimum-norm, Penrose 1955; direction-preserving scaling, Bodson 2002). \((UU^\top)^+\) is taken by Jacobi eigendecomposition. B-dot allocates its raw dipole the same way. Power is \(\sum_k P_k\langle(d_k/d_{\max,k})^2\rangle\) (`rod_duty2_mean`).

**Control.** Action = quaternion (4) + one gate per rod (6): rod k is on when \(a_{4+k}>0\), and all rods are on in brownout. The observation carries the six on/off states. For the v10 reward, the delegated fraction per torque axis is the rod authority switched off (`vehicle.axis_authority`). The v8–v11 variants keep the 3-axis coil card (0.625 kg) for reproducibility.

Tests: `tests/control/test_rods.py` covers the pseudo-inverse match for every gate pattern, zero out-of-plane dipole, exact small request, direction-preserving saturation, mass properties, plant gating and duty, and the env variant.

