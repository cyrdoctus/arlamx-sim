"""07 — MRP-PD. Citations: Schaub & Junkins 2018 Ch.8; Tsiotras 1996."""
import numpy as np
from arlamx_v2 import cpp


def test_zero_error_zero_torque():
    c = cpp.MRPFeedback()
    c.set_inertia_diag(np.array([0.0125, 0.0125, 0.025]))
    tau = c.compute(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 1.0)
    assert np.linalg.norm(tau) < 1e-15


def test_regulation_closes():
    # Integrate Euler + MRP kinematics with the PD law toward 0.
    c = cpp.MRPFeedback()
    c.kp = 0.05
    c.kd = 0.02
    I = np.array([0.0125, 0.0125, 0.025])
    c.set_inertia_diag(I)
    sigma = np.array([0.2, -0.15, 0.05])
    omega = np.zeros(3)
    dt = 0.2
    for _ in range(800):
        tau = c.compute(sigma, omega, np.zeros(3), np.zeros(3), dt)
        # Euler: I wdot = tau - w x Iw  (gyro cancelled by feedforward)
        wdot = tau / I  # because feedforward cancels, leftover is PD / I ... 
        # Use plant I wdot = tau - w×Iw
        Iw = I * omega
        wdot = (tau - np.cross(omega, Iw)) / I
        omega = omega + wdot * dt
        B = cpp.mrp_rate(sigma, omega)
        sigma = cpp.mrp_shadow(sigma + B * dt)
    ang = cpp.mrp_angle(sigma)
    assert ang < math_deg(0.05)


def math_deg(d):
    return d * 3.141592653589793 / 180.0
