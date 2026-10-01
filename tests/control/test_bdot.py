"""08 — B-dot. Citations: Stickler & Alfriend 1976; Avanzini & Giulietti 2012."""
import numpy as np
from arlamx_v2 import cpp


def test_first_call_zero():
    b = cpp.BDot()
    b.dt = 0.1
    m = b.dipole(np.array([2e-5, 0.0, 0.0]))
    assert np.linalg.norm(m) == 0.0


def test_saturates():
    b = cpp.BDot()
    b.gain = 1e20
    b.max_dipole = 0.19
    b.dt = 0.1
    b.dipole(np.zeros(3))
    m = b.dipole(np.array([1e-5, 0.0, 0.0]))
    assert abs(np.linalg.norm(m) - 0.19) < 1e-12


def test_spin_energy_falls():
    b = cpp.BDot()
    b.gain = 1e6
    b.max_dipole = 0.6
    b.dt = 0.5
    I = np.array([0.0125, 0.0125, 0.025])
    omega = np.array([0.0, 0.0, 0.1])
    B0 = np.array([3e-5, 0.0, 0.0])  # inertial dipole along X
    sigma = np.zeros(3)
    E0 = None
    E_last = None
    for k in range(400):
        C = cpp.mrp_to_dcm(sigma)
        B_B = C @ B0
        m = b.dipole(B_B)
        tau = cpp.BDot.torque(m, B_B)
        Iw = I * omega
        wdot = (tau - np.cross(omega, Iw)) / I
        omega = omega + wdot * b.dt
        sigma = cpp.mrp_shadow(sigma + cpp.mrp_rate(sigma, omega) * b.dt)
        E = 0.5 * np.dot(omega, I * omega)
        if k == 2:
            E0 = E
        E_last = E
    assert E_last < E0
