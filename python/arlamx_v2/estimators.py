"""Environmental-torque Kalman filter and the rigid-body torque measurement."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Euler's equation solved for the disturbance: tau = I w_dot + w x I w - tau_ctrl
# (Markley & Crassidis 2014, ch. 3).
def torque_measurement(omega_prev, omega_now, dt_s, inertia_diag, tau_cmd,
                       gyro_mean=None):
    omega_prev = np.asarray(omega_prev, float).reshape(3)
    omega_now = np.asarray(omega_now, float).reshape(3)
    I = np.asarray(inertia_diag, float).reshape(3)
    tau_cmd = np.asarray(tau_cmd, float).reshape(3)
    dt = max(float(dt_s), 1e-9)
    omega_dot = (omega_now - omega_prev) / dt
    if gyro_mean is None:
        gyroscopic = np.cross(omega_now, I * omega_now)
    else:
        gyroscopic = np.asarray(gyro_mean, float).reshape(3)
    return I * omega_dot + gyroscopic - tau_cmd


# First-order error propagation of torque_measurement: finite difference sqrt(2)/dt,
# gyroscopic term 2|I w| sigma_w, command noise.
def measurement_sigma(omega_now, dt_s, inertia_diag, sigma_omega, sigma_tau_cmd=0.0):
    omega_now = np.asarray(omega_now, float).reshape(3)
    I = np.asarray(inertia_diag, float).reshape(3)
    dt = max(float(dt_s), 1e-9)
    diff_term = I * sigma_omega * np.sqrt(2.0) / dt
    cross_term = 2.0 * np.abs(I * omega_now) * sigma_omega
    return np.sqrt(diff_term ** 2 + cross_term ** 2 + float(sigma_tau_cmd) ** 2)


@dataclass
class TorqueKFConfig:
    enabled: bool = True
    sigma_drive: float = 1.0e-12
    sigma_tau_cmd: float = 2.0e-9
    p0_tau: float = 1.0e-6
    p0_taudot: float = 1.0e-12
    innovation_gate_sigma: float = 5.0
    inertia_diag: tuple = (0.0125, 0.0125, 0.025)


@dataclass
# State [tau, tau_dot], discretized continuous white-noise-acceleration model
# (Bar-Shalom, Li & Kirubarajan 2001, Estimation with Applications to Tracking and Navigation, ch. 6).
class TorqueKF:
    cfg: TorqueKFConfig = field(default_factory=TorqueKFConfig)
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    P: np.ndarray = field(default_factory=lambda: np.eye(6))
    initialized: bool = False

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.diag(np.concatenate([
            np.full(3, self.cfg.p0_tau), np.full(3, self.cfg.p0_taudot)]))
        self.initialized = False

    def _FQ(self, dt):
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        s2 = self.cfg.sigma_drive ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * s2 * dt ** 3 / 3.0
        Q[:3, 3:] = np.eye(3) * s2 * dt ** 2 / 2.0
        Q[3:, :3] = Q[:3, 3:]
        Q[3:, 3:] = np.eye(3) * s2 * dt
        return F, Q

    def step(self, z, R_diag, dt_s):
        z = np.asarray(z, float).reshape(3)
        R_diag = np.asarray(R_diag, float).reshape(3)
        dt = max(float(dt_s), 1e-9)
        if not self.cfg.enabled:
            return z.copy()
        if not self.initialized:
            self.x[:3] = z
            self.P[:3, :3] = np.diag(np.maximum(R_diag, 1e-24))
            self.initialized = True
            return z.copy()

        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = np.diag(np.maximum(R_diag, 1e-30))
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        g = float(self.cfg.innovation_gate_sigma)
        R_used = R
        if g > 0.0:
            sd = np.sqrt(np.maximum(np.diag(S), 1e-30))
            over = np.abs(y) > g * sd
            if np.any(over):
                scale = np.where(over, (np.abs(y) / (g * sd)) ** 2, 1.0)
                R_used = R + np.diag(np.maximum(np.diag(R) * (scale - 1.0), 0.0))
                S = H @ self.P @ H.T + R_used
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        A = np.eye(6) - K @ H
        self.P = A @ self.P @ A.T + K @ R_used @ K.T
        return self.x[:3].copy()

    @property
    def tau(self):
        return self.x[:3].copy()

    @property
    def tau_dot(self):
        return self.x[3:].copy()

    def sigma(self):
        return np.sqrt(np.maximum(np.diag(self.P)[:3], 0.0))


def env_torque(kf, omega_prev, omega_now, dt_s, tau_cmd,
                                sigma_omega, cfg=None, gyro_mean=None):
    cfg = cfg or kf.cfg
    I = np.asarray(cfg.inertia_diag, float).reshape(3)
    z = torque_measurement(omega_prev, omega_now, dt_s, I, tau_cmd,
                           gyro_mean=gyro_mean)
    R = measurement_sigma(omega_now, dt_s, I, sigma_omega,
                          cfg.sigma_tau_cmd) ** 2
    return kf.step(z, R, dt_s), z
