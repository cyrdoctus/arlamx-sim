"""Disturbance-torque Kalman filter. Citations: Bar-Shalom, Li & Kirubarajan
2001 (CV model, Joseph form); Gelb 1974 (innovation gating)."""
import numpy as np
import pytest

from arlamx_v2.estimators import (TorqueKF, TorqueKFConfig, measurement_sigma,
                                  torque_measurement)

I_DIAG = (0.0125, 0.0125, 0.025)


def test_measurement_identity():
    """E1 must equal I*(w1-w0)/dt + w1 x I*w1 - tau_cmd, written out
    independently here — a formula pin, robust to any trajectory choice."""
    I = np.asarray(I_DIAG)
    dt = 300.0
    w0 = np.array([1e-3, -2e-3, 5e-4])
    w1 = np.array([-4e-4, 1.5e-3, 2e-4])
    tau_cmd = np.array([1e-6, -2e-6, 3e-7])
    z = torque_measurement(w0, w1, dt, I, tau_cmd)
    expect = I * (w1 - w0) / dt + np.cross(w1, I * w1) - tau_cmd
    assert np.allclose(z, expect, rtol=0, atol=1e-18)


def test_measurement_uses_gyro_mean_when_given():
    """v2.6: with the plant's RK4-consistent <omega x I omega> the endpoint
    gyroscopic term is replaced, nothing else changes."""
    I = np.asarray(I_DIAG)
    dt = 300.0
    w0 = np.array([1e-3, -2e-3, 5e-4])
    w1 = np.array([-4e-4, 1.5e-3, 2e-4])
    tau_cmd = np.array([1e-6, -2e-6, 3e-7])
    gm = np.array([2e-9, -1e-9, 5e-10])
    z = torque_measurement(w0, w1, dt, I, tau_cmd, gyro_mean=gm)
    assert np.allclose(z, I * (w1 - w0) / dt + gm - tau_cmd, rtol=0, atol=1e-18)


def test_kf_converges_on_cv_signal():
    """Track a slowly ramping torque through noisy measurements: the filtered
    RMS error must beat the raw measurement RMS error."""
    rng = np.random.default_rng(3)
    cfg = TorqueKFConfig()
    kf = TorqueKF(cfg)
    dt = 300.0
    # Noise comparable to the raw observer's real error budget: with sigma at
    # the signal scale the smoothing is what carries the test.
    sigma_z = 1.2e-8
    errs_f, errs_raw = [], []
    for k in range(120):
        t = k * dt
        tau_true = np.array([2e-8 + 1e-11 * t, -1e-8, 5e-9]) \
            + 1e-8 * np.sin(2 * np.pi * t / 5400.0) * np.array([1.0, 0.5, 0.0])
        z = tau_true + rng.normal(0.0, sigma_z, 3)
        est = kf.step(z, np.full(3, sigma_z ** 2), dt)
        if k > 10:
            errs_f.append(np.linalg.norm(est - tau_true))
            errs_raw.append(np.linalg.norm(z - tau_true))
    assert np.mean(errs_f) < 0.8 * np.mean(errs_raw)


def test_joseph_covariance_stays_psd_under_gate():
    """A wild outlier triggers the innovation gate; the posterior covariance
    must stay symmetric positive-definite (the pre-fix code fed the un-inflated
    R to the Joseph form — grok 1.6)."""
    kf = TorqueKF(TorqueKFConfig())
    dt = 300.0
    kf.step(np.array([2e-8, 0, 0]), np.full(3, 1e-18), dt)     # seed
    kf.step(np.array([2e-8, 0, 0]), np.full(3, 1e-18), dt)
    kf.step(np.array([5e-4, 0, 0]), np.full(3, 1e-18), dt)     # absurd outlier
    P = kf.P
    assert np.allclose(P, P.T, atol=1e-25)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_gate_limits_outlier_influence():
    kf_g = TorqueKF(TorqueKFConfig(innovation_gate_sigma=5.0))
    kf_n = TorqueKF(TorqueKFConfig(innovation_gate_sigma=0.0))
    dt, Rd = 300.0, np.full(3, 1e-18)
    for kf in (kf_g, kf_n):
        for _ in range(5):
            kf.step(np.array([2e-8, 0, 0]), Rd, dt)
    out = np.array([5e-4, 0, 0])
    est_g = kf_g.step(out, Rd, dt)
    est_n = kf_n.step(out, Rd, dt)
    assert abs(est_g[0] - 2e-8) < abs(est_n[0] - 2e-8)


def test_disabled_passthrough():
    kf = TorqueKF(TorqueKFConfig(enabled=False))
    z = np.array([1e-7, -2e-8, 3e-9])
    assert np.array_equal(kf.step(z, np.full(3, 1e-18), 300.0), z)


def test_measurement_sigma_scales_with_dt():
    s_short = measurement_sigma(np.zeros(3), 10.0, I_DIAG, 1e-6)
    s_long = measurement_sigma(np.zeros(3), 300.0, I_DIAG, 1e-6)
    assert np.all(s_short > s_long)          # differencing term ~ 1/dt
