"""Onboard FP32 propagator (two-body + J2 + exponential drag + held along-track SRP), numpy reference of the C++ mirror."""
from __future__ import annotations

import numpy as np

try:
    from arlamx_v2 import cpp as _cpp
    _CPP_SAMPLES = getattr(_cpp, "propagate_f32_samples", None)
except Exception:  # pragma: no cover - only when the extension is missing
    _CPP_SAMPLES = None

MU = np.float32(3.986004418e14)
RE = np.float32(6378137.0)
J2 = np.float32(1.0826353865e-3)
OMEGA_E = np.float32(7.2921150e-5)
FLAT = np.float32(1.0 / 298.257223563)
H_SCALE_DEFAULT = np.float32(60.0e3)


# a = -mu r/|r|^3 + a_J2 - 1/2 rho(h) (Cd A/m) |v_rel| v_rel + a_srp v_hat,
# rho(h) = rho0 exp(-(h - h0)/H), v_rel = v - w_E x r (Montenbruck & Gill 2000, Sec. 3.5).
def _accel(r, v, bc_inv, rho0, h0, h_scale, a_srp=0.0, sun=None):
    r2 = np.float32(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    rn = np.float32(np.sqrt(r2))
    if rn < np.float32(1.0):
        return np.zeros(3, dtype=np.float32)
    inv_r3 = np.float32(1.0) / (r2 * rn)

    a = -MU * inv_r3 * r

    # J2 (Vallado eq. 8-30), z-axis aligned with the pole.
    zr = np.float32(r[2] / rn)
    f = np.float32(1.5) * J2 * MU * RE * RE / (r2 * r2)
    five_z2 = np.float32(5.0) * zr * zr
    a = a + np.array([
        f * (five_z2 - np.float32(1.0)) * np.float32(r[0] / rn),
        f * (five_z2 - np.float32(1.0)) * np.float32(r[1] / rn),
        f * (five_z2 - np.float32(3.0)) * np.float32(r[2] / rn),
    ], dtype=np.float32)

    # Height above the ellipsoid, first order: r_E = a (1 - f sin^2 phi) (Vallado 2013, ch. 3).
    alt = np.float32(rn - RE * (np.float32(1.0) - FLAT * zr * zr))
    rho = np.float32(rho0 * np.exp(-(alt - h0) / h_scale))
    v_rel = np.array([v[0] + OMEGA_E * r[1], v[1] - OMEGA_E * r[0], v[2]],
                     dtype=np.float32)
    vr = np.float32(np.sqrt(v_rel[0] ** 2 + v_rel[1] ** 2 + v_rel[2] ** 2))
    a = a - np.float32(0.5) * rho * np.float32(bc_inv) * vr * v_rel

    if sun is not None and np.float32(a_srp) != np.float32(0.0):
        rs = np.float32(r[0] * sun[0] + r[1] * sun[1] + r[2] * sun[2])
        lit = rs >= np.float32(0.0) or np.float32(np.linalg.norm(r - rs * sun)) >= RE
        vn = np.float32(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
        if lit and vn > np.float32(0.0):
            a = a + (np.float32(a_srp) / vn) * v
    return a.astype(np.float32)


# Classical fourth-order Runge-Kutta (Montenbruck & Gill 2000, Sec. 4.1).
def _rk4(r, v, dt, bc_inv, rho0, h0, h_scale, a_srp=0.0, sun=None):
    dt = np.float32(dt)
    ex = (a_srp, sun)
    k1v = _accel(r, v, bc_inv, rho0, h0, h_scale, *ex)
    k1r = v
    k2v = _accel(r + np.float32(0.5) * dt * k1r, v + np.float32(0.5) * dt * k1v,
                 bc_inv, rho0, h0, h_scale, *ex)
    k2r = v + np.float32(0.5) * dt * k1v
    k3v = _accel(r + np.float32(0.5) * dt * k2r, v + np.float32(0.5) * dt * k2v,
                 bc_inv, rho0, h0, h_scale, *ex)
    k3r = v + np.float32(0.5) * dt * k2v
    k4v = _accel(r + dt * k3r, v + dt * k3v, bc_inv, rho0, h0, h_scale, *ex)
    k4r = v + dt * k3v
    six = np.float32(6.0)
    r_n = r + dt / six * (k1r + np.float32(2.0) * k2r + np.float32(2.0) * k3r + k4r)
    v_n = v + dt / six * (k1v + np.float32(2.0) * k2v + np.float32(2.0) * k3v + k4v)
    return r_n.astype(np.float32), v_n.astype(np.float32)


# F = 1/2 rho Cd A v^2  =>  Cd A/m = 2F / (rho v^2 m) (Montenbruck & Gill 2000, Sec. 3.5).
def bc_from_drag(drag_N, rho, v_rel, mass):
    den = 0.5 * float(rho) * float(v_rel) ** 2 * max(float(mass), 1e-9)
    if den <= 1e-30 or not np.isfinite(den):
        return 0.0
    return float(np.clip(float(drag_N) / den, 0.0, 10.0))


def _sun32(sun):
    if sun is None:
        return None
    s = np.asarray(sun, float).reshape(3)
    return (s / max(np.linalg.norm(s), 1e-15)).astype(np.float32)


def _cpp_samples(r0, v0, offsets, dt_s, bc_inv, rho0, alt0_m, h_scale, a_srp=0.0, sun=None):
    out = _CPP_SAMPLES(np.asarray(r0, dtype=np.float64).reshape(3),
                       np.asarray(v0, dtype=np.float64).reshape(3),
                       [float(o) for o in offsets], float(dt_s), float(bc_inv),
                       float(rho0), float(alt0_m), float(h_scale), float(a_srp),
                       None if sun is None else np.asarray(sun, float).reshape(3))
    return [(np.asarray(r, dtype=np.float32), np.asarray(v, dtype=np.float32))
            for r, v in out]


def propagate(r0, v0, horizon_s, dt_s=30.0, bc_inv=0.0, rho0=3.0e-12,
              alt0_m=400e3, h_scale=H_SCALE_DEFAULT, use_cpp=True, a_srp=0.0, sun=None):
    if use_cpp and _CPP_SAMPLES is not None and float(horizon_s) > 0.0:
        return _cpp_samples(r0, v0, [float(horizon_s)], dt_s, bc_inv, rho0,
                            alt0_m, h_scale, a_srp, sun)[0]
    return propagate_numpy(r0, v0, horizon_s, dt_s=dt_s, bc_inv=bc_inv,
                           rho0=rho0, alt0_m=alt0_m, h_scale=h_scale, a_srp=a_srp, sun=sun)


def propagate_numpy(r0, v0, horizon_s, dt_s=30.0, bc_inv=0.0, rho0=3.0e-12,
                    alt0_m=400e3, h_scale=H_SCALE_DEFAULT, a_srp=0.0, sun=None):
    r = np.asarray(r0, dtype=np.float32).reshape(3).copy()
    v = np.asarray(v0, dtype=np.float32).reshape(3).copy()
    n = max(1, int(round(float(horizon_s) / max(float(dt_s), 1e-3))))
    dt = np.float32(float(horizon_s) / n)
    rho0 = np.float32(rho0)
    h0 = np.float32(alt0_m)
    hs = np.float32(h_scale)
    s32 = _sun32(sun)
    for _ in range(n):
        r, v = _rk4(r, v, dt, bc_inv, rho0, h0, hs, a_srp, s32)
        if not np.isfinite(r).all() or not np.isfinite(v).all():
            break
        if np.linalg.norm(r) < RE * np.float32(0.9):
            break
    return r, v


def propagate_samples(r0, v0, offsets_s, dt_s=30.0, bc_inv=0.0, rho0=3.0e-12,
                      alt0_m=400e3, h_scale=H_SCALE_DEFAULT, use_cpp=True, a_srp=0.0, sun=None):
    offs = [float(o) for o in offsets_s]
    if use_cpp and _CPP_SAMPLES is not None:
        return _cpp_samples(r0, v0, offs, dt_s, bc_inv, rho0, alt0_m, h_scale, a_srp, sun)
    r = np.asarray(r0, dtype=np.float32).reshape(3).copy()
    v = np.asarray(v0, dtype=np.float32).reshape(3).copy()
    out, t = [], 0.0
    for target in offs:
        seg = max(0.0, target - t)
        if seg > 0.0:
            r, v = propagate_numpy(r, v, seg, dt_s=dt_s, bc_inv=bc_inv, rho0=rho0,
                                   alt0_m=alt0_m, h_scale=h_scale, a_srp=a_srp, sun=sun)
            t = target
        out.append((r.copy(), v.copy()))
    return out


# Vis-viva: a = -mu / (2 (v^2/2 - mu/r)) (Vallado 2013, ch. 1).
def sma_m(r, v, mu=float(MU)):
    rn = float(np.linalg.norm(np.asarray(r, float)))
    vn = float(np.linalg.norm(np.asarray(v, float)))
    if rn < 1.0:
        return float("nan")
    energy = 0.5 * vn * vn - mu / rn
    if energy >= -1e-9:
        return float("nan")
    return float(-mu / (2.0 * energy))


def decay_rate(r0, v0, horizon_s, **kw):
    a0 = sma_m(r0, v0)
    r1, v1 = propagate(r0, v0, horizon_s, **kw)
    a1 = sma_m(r1, v1)
    if not (np.isfinite(a0) and np.isfinite(a1)) or horizon_s <= 0:
        return float("nan")
    return float((a0 - a1) / 1e3 / (float(horizon_s) / 86400.0))
