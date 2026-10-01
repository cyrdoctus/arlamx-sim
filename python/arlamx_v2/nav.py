"""No-GNSS onboard navigation: fp32 J2 + drag propagation from the launch state, eclipse-timing fixes from the solar panels."""
from __future__ import annotations

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.stations import nearest_visible

F = np.float32
MU, RE, J2, WE = F(3.986004418e14), F(6378137.0), F(1.08262668e-3), F(7.2921150e-5)
# Exponential atmosphere, base altitude [km], rho0 [kg/m^3], scale height [km] (Vallado 2013, Table 8-4).
ATM = np.array([(250, 7.248e-11, 45.546), (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298),
                (400, 3.725e-12, 58.515), (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822),
                (600, 1.454e-13, 71.835)])


def rho_exp(h_km):
    h_km = min(max(h_km, 150.0), 1000.0)
    i = int(np.clip(np.searchsorted(ATM[:, 0], h_km) - 1, 0, len(ATM) - 1))
    return F(ATM[i, 1] * np.exp(-(h_km - ATM[i, 0]) / ATM[i, 2]))


def acc(r, v, bc):
    """fp32 a = two-body + J2 (Vallado 2013, eq. 8-30) - 0.5 rho |v_rel| v_rel Cd A / m."""
    x, y, z = r
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    k = MU / (r2 * rn)
    f = F(1.5) * J2 * (RE * RE) / r2
    zz = F(5.0) * z * z / r2
    a = np.array([-k * x * (F(1.0) + f * (F(1.0) - zz)),
                  -k * y * (F(1.0) + f * (F(1.0) - zz)),
                  -k * z * (F(1.0) + f * (F(3.0) - zz))], F)
    vr = v - np.array([-WE * y, WE * x, F(0.0)], F)
    vn = np.sqrt(vr @ vr)
    return a - F(0.5) * rho_exp(float(rn - RE) / 1e3) * bc * vn * vr


def rk4(r, v, h, bc):
    h = F(h)
    k1r, k1v = v, acc(r, v, bc)
    k2r, k2v = v + F(0.5) * h * k1v, acc(r + F(0.5) * h * k1r, v + F(0.5) * h * k1v, bc)
    k3r, k3v = v + F(0.5) * h * k2v, acc(r + F(0.5) * h * k2r, v + F(0.5) * h * k2v, bc)
    k4r, k4v = v + h * k3v, acc(r + h * k3r, v + h * k3v, bc)
    s = h / F(6.0)
    return r + s * (k1r + F(2) * k2r + F(2) * k3r + k4r), v + s * (k1v + F(2) * k2v + F(2) * k3v + k4v)


def fly(r, v, dt, bc, h=30.0):
    n = max(1, int(np.ceil(abs(dt) / h)))
    for _ in range(n):
        r, v = rk4(r, v, dt / n, bc)
    return r, v


def lit(r, sun):
    """Cylindrical shadow margin, > 0 sunlit (Vallado 2013, ch. 5)."""
    along = float(r @ sun)
    return 1.0 if along >= 0.0 else float(np.linalg.norm(r - along * sun)) / float(RE) - 1.0


def crossing(r, v, t, sun, t_lo, t_hi, entering, bc, h=30.0):
    """Shadow crossings of the propagated (r, v; t) inside [t_lo, t_hi] -> list of times (linear in the margin)."""
    r, v = fly(r, v, t_lo - t, bc)
    out, tt, g = [], t_lo, lit(r, sun)
    while tt < t_hi:
        r, v = rk4(r, v, h, bc)
        g1 = lit(r, sun)
        if (g > 0 >= g1 and entering) or (g <= 0 < g1 and not entering):
            out.append(tt + h * g / (g - g1))
        tt, g = tt + h, g1
    return out


class Nav:
    """fp32 state from the deployer hand-over, alpha-beta phase / drag-scale filter on eclipse entry and exit times."""

    def __init__(self, r, v, t, cfg, rng):
        self.c = cfg
        self.rng = rng
        self.r = (np.asarray(r, float) + rng.normal(0.0, cfg["sig_r0_m"], 3)).astype(F)
        self.v = (np.asarray(v, float) + rng.normal(0.0, cfg["sig_v0_ms"], 3)).astype(F)
        self.t = float(t)
        self.k = F(1.0)
        self.t_fix = float(t)
        self.fixes = []

    @property
    def bc(self):
        return self.k * F(self.c["bc0"])

    def to(self, t):
        if t > self.t:
            r0, v0 = self.r, self.v
            with np.errstate(all="ignore"):
                self.r, self.v = fly(self.r, self.v, t - self.t, self.bc, self.c["h_s"])
            if not self.sane():
                # sanity fallback: keep the last good state, drag scale back to 1, coast on two-body + J2 only
                self.k = F(1.0)
                self.faults = getattr(self, "faults", 0) + 1
                self.r, self.v = fly(r0, v0, t - self.t, F(0.0), self.c["h_s"])
            self.t = t

    def sane(self):
        rn = float(np.linalg.norm(self.r))
        return np.all(np.isfinite(self.r)) and np.all(np.isfinite(self.v)) and 6578e3 < rn < 8378e3

    def fix(self, t_meas, entering, sun):
        """Residual dt = t_meas - t_pred -> phase shift -alpha dt and drag scale k exp(-beta dt / T)."""
        sun = np.asarray(sun, F)
        for w in (900.0, 2700.0):
            cands = crossing(self.r, self.v, self.t, sun, t_meas - w, t_meas + w, entering, self.bc)
            if cands:
                break
        if not cands:
            return None
        dt = t_meas - min(cands, key=lambda x: abs(x - t_meas))
        stale = self.t - self.t_fix > 3.0 * 5550.0
        if abs(dt) > self.c.get("gate_s", 120.0) and not stale:
            self.rejected = getattr(self, "rejected", 0) + 1                 # outlier gate
            return None
        r0, v0, k0 = self.r, self.v, self.k
        self.r, self.v = fly(self.r, self.v, -self.c["alpha"] * dt, self.bc, self.c["h_s"])
        gap = self.t - self.t_fix
        if 1500.0 < gap < 12000.0 and not stale:
            # Along-track drift e_dot = v dt / gap = -1.5 n da, da = 2 dv / n -> dv = e_dot / 3 (Vallado 2013, ch. 6, CW secular term).
            self.v = self.v * F(1.0 + np.clip(self.c.get("gamma", 0.0) * dt / gap / 3.0, -2e-4, 2e-4))
        self.k = F(np.clip(self.k * np.exp(-self.c["beta"] * np.clip(dt, -30.0, 30.0) / 60.0), 0.25, 4.0))
        if not self.sane():
            self.r, self.v, self.k = r0, v0, F(1.0)
            return None
        self.t_fix = self.t
        self.fixes.append((self.t, dt))
        return dt

    def gs(self, c_bn, gmst):
        d, vis, _ = nearest_visible(self.r.astype(float), gmst)
        return vis, float((c_bn.T @ np.array([0.0, 0.0, 1.0])) @ d) if vis > 0.5 else 0.0


def true_crossing(r, v, sun, dt, entering, h=10.0):
    """Truth crossing time inside [0, dt] of the step: 10 s two-body arc, linear in the shadow margin."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    t, g = 0.0, lit(r, sun)
    while t < dt:
        r, v = rk4(r, v, h, 0.0)
        g1 = lit(r, sun)
        if (g > 0 >= g1 and entering) or (g <= 0 < g1 and not entering):
            return t + h * g / (g - g1)
        t, g = t + h, g1
    return None
