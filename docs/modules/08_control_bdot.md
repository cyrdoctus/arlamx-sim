# 08 — B-dot detumble

**Folder:** `cpp/control/` (`bdot.hpp`)  
**Status:** implemented  
**Tests:** `tests/control/`

## What it does

Detumble mode. Commands a magnetic dipole from the finite-differenced body field, saturates to the magnetorquer limit, applies \(\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}\).

V1.7 *configures* B-dot (`adcs.detumble: b_dot`) but the live `ArlamxEnv.step` path uses MRP-PD every substep. V2.0 still implements B-dot **in the plant** so detumble is a real mode (`set_mode("detumble")`), not a dead YAML key.

## Equations

\[
\dot{\mathbf{B}}_B\approx(\mathbf{B}_k-\mathbf{B}_{k-1})/\Delta t,\qquad
\mathbf{m}=-k\,\dot{\mathbf{B}}_B
\]

`BDot::dipole` still applies a **vector-norm** cap (`|m| > BDot.max_dipole`). The plant constructor / `set_bdot` sets that spherical cap to \(10^{300}\) so it does not bind. Live saturation is **per-axis** `saturate_dipole(m, dipole_max_)` inside `apply_magnetorquer` (same coils as pointing). First sample: \(\mathbf{m}=\mathbf{0}\).

\[
\boldsymbol{\tau}=\mathbf{m}\times\mathbf{B}_B
\]

Detumbled when \(|\boldsymbol{\omega}|<\omega_\mathrm{th}\) (default 0.01 rad/s).

## Citations

1. Stickler, A. C. & Alfriend, K. T. (1976). Elementary magnetic attitude control system. *J. Spacecraft & Rockets* 13(5), 282–287.
2. Lovera, M. (2001). Magnetic satellite detumbling: the b-dot algorithm revisited. *Proc. ACC* / see also Avanzini & Giulietti (2012), *JGCD* 35(4), 1326–1334, “Magnetic detumbling of a rigid spacecraft.”

## Classroom test

**A.** First call returns \(\mathbf{m}=\mathbf{0}\).  
**B.** Constant inertial dipole field, spacecraft spinning about \(\hat{z}\): \(\mathbf{B}_B\) rotates, \(\mathbf{m}\) opposes \(\dot{\mathbf{B}}\), and \(|\boldsymbol{\omega}|\) must decrease over 500 s on the Euler plant (energy \(\tfrac12\boldsymbol{\omega}\cdot I\boldsymbol{\omega}\) strictly falling after the first sample).  
**C.** Saturation: huge \(k\) still gives \(|\mathbf{m}|=m_\max\).

## Tests run

2026-08-18 — `tests/control/test_bdot.py` **PASS** (first sample zero, saturation, spin energy falls).
