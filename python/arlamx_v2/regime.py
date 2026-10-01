"""Onboard regime estimator: schedules gains, gate floor and margins from the torque estimate."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeConfig:
    tau_ref: float = 5.0e-6
    tau_dot_ref: float = 1.67e-8
    ewma_s: float = 5400.0
    dt_s: float = 300.0
    shock_deta: float = 0.30
    shock_tdot: float = 0.30
    kd_quiet: float = 1.0e-3
    kd_ease: float = 0.40
    kp: float = 6.0e-5
    gate_floor_quiet: float = 0.30
    gate_ease: float = 0.50
    clip_kappa_quiet: float = 0.70
    clip_kappa_storm: float = 1.00
    margin0: float = 0.05
    margin_gain: float = 2.0
    m_max_Am2: float = 1.473


@dataclass
class RegimeEstimator:
    cfg: RegimeConfig = field(default_factory=RegimeConfig)
    eta: float = 0.0
    eta_slow: float = 0.0
    shock: float = 0.0
    authority: float = 0.0
    n: int = 0

    def reset(self):
        self.eta = 0.0
        self.eta_slow = 0.0
        self.shock = 0.0
        self.authority = 0.0
        self.n = 0

    @property
    def alpha(self):
        tau = max(float(self.cfg.ewma_s), 1.0)
        return 1.0 - math.exp(-float(self.cfg.dt_s) / tau)

    @property
    def eta_eff(self):
        return float(max(self.eta, self.shock))

    def update(self, tau_env, tau_dot=None, B=None):
        te = float(np.linalg.norm(np.asarray(tau_env, float).reshape(-1)))
        self.eta = float(np.clip(te / max(self.cfg.tau_ref, 1e-18), 0.0, 1.0))
        a = self.alpha
        if self.n <= 0:
            self.eta_slow = self.eta
        else:
            self.eta_slow = (1.0 - a) * self.eta_slow + a * self.eta
        self.n += 1
        deta = abs(self.eta - self.eta_slow)
        td = 0.0
        if tau_dot is not None:
            td = float(np.linalg.norm(np.asarray(tau_dot, float).reshape(-1)))
        tdn = td / max(self.cfg.tau_dot_ref, 1e-18)
        self.shock = 1.0 if (deta > self.cfg.shock_deta or tdn > self.cfg.shock_tdot) else 0.0
        if B is not None:
            b = float(np.linalg.norm(np.asarray(B, float).reshape(-1)))
            self.authority = te / max(self.cfg.m_max_Am2 * b, 1e-16)
        else:
            self.authority = 0.0
        return self.peek()

    def peek(self):
        ee = self.eta_eff
        c = self.cfg
        return {
            "eta": float(self.eta),
            "eta_slow": float(self.eta_slow),
            "shock": float(self.shock),
            "eta_eff": float(ee),
            "authority": float(self.authority),
            "kd": float(c.kd_quiet * (1.0 - c.kd_ease * ee)),
            "kp": float(c.kp),
            "gate_floor": float(c.gate_floor_quiet * (1.0 - c.gate_ease * ee)),
            "margin": float(c.margin0 * (1.0 + c.margin_gain * ee)),
            "clip_kappa": float(
                c.clip_kappa_quiet
                + (c.clip_kappa_storm - c.clip_kappa_quiet) * ee),
        }

    def as_info(self):
        sch = self.peek()
        return {
            "eta": sch["eta"],
            "eta_slow": sch["eta_slow"],
            "shock": sch["shock"],
            "eta_eff": sch["eta_eff"],
            "authority": sch["authority"],
            "kd_eff": sch["kd"],
            "gate_floor_eff": sch["gate_floor"],
            "margin_eff": sch["margin"],
            "clip_kappa": sch["clip_kappa"],
        }
