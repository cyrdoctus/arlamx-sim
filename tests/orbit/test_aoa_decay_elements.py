"""Classroom tests for the aoa_decay post-processing: rv_to_coe round-trips
coe_to_rv, and the Gauss lift-only integration is zero for zero lift and
matches the closed-form secular rate for a constant normal acceleration
flipped every half orbit."""
import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.aoa_decay import MU, RE, gauss_integrate, orbit_mean, rv_to_coe
from arlamx_v2.decay_run import coe_to_rv


def test_rv_to_coe_roundtrip():
    a, e, inc, raan, argp, nu = RE + 500e3, 0.001, np.radians(23.0), 0.7, 1.9, 2.4
    r, v = coe_to_rv(a, e, inc, raan, argp, nu)
    c = rv_to_coe(r, v)
    assert np.isclose(c["a"], a, rtol=1e-12)
    assert np.isclose(c["e"], e, atol=1e-12)
    assert np.isclose(c["inc"], inc, atol=1e-12)
    assert np.isclose((c["raan"] - raan + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=1e-10)
    assert np.isclose((c["argp"] - argp + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=1e-8)
    assert np.isclose((c["nu"] - nu + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=1e-8)
    assert np.isclose((c["u"] - argp - nu + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=1e-8)


def _synthetic(a_N_func, n_orb=20, dt=60.0):
    a, e, inc = RE + 400e3, 0.0, np.radians(23.0)
    n = np.sqrt(MU / a**3)
    t = np.arange(0.0, n_orb * 2 * np.pi / n, dt)
    u = (n * t) % (2 * np.pi)
    m = 1.0
    d = {
        "t_s": t, "t_day": t / 86400.0, "sma_km": np.full_like(t, a / 1e3), "ecc": np.full_like(t, e),
        "inc_deg": np.full_like(t, np.degrees(inc)), "raan_deg": np.zeros_like(t), "nu_deg": np.degrees(u),
        "u_deg": np.degrees(u),
        "lift_R_N": np.zeros_like(t), "lift_T_N": np.zeros_like(t), "lift_N_N": a_N_func(u) * m,
    }
    return d, a, n, m


def test_gauss_zero_for_zero_lift():
    d, _a, _n, m = _synthetic(lambda u: np.zeros_like(u))
    g = gauss_integrate(d, m, "lift")
    assert np.all(g["di_deg"] == 0.0) and np.all(g["de_vec_mag"] == 0.0)


def test_gauss_constant_normal_accel_is_periodic_only():
    d, a, n, m = _synthetic(lambda u: np.full_like(u, 1e-6))
    g = gauss_integrate(d, m, "lift")
    # di/dt = r cos(u) a_N / h -> amplitude a_N / (n v) ; no secular growth
    v = np.sqrt(MU / a)
    amp = np.degrees(1e-6 / (n * v))
    assert np.max(np.abs(g["di_deg"])) < 1.2 * amp
    assert abs(g["di_deg"][-1]) < 0.1 * amp


def test_gauss_half_orbit_flip_matches_two_over_pi():
    d, a, n, m = _synthetic(lambda u: 1e-6 * np.sign(np.cos(u)))
    g = gauss_integrate(d, m, "lift")
    v = np.sqrt(MU / a)
    t_end = d["t_s"][-1]
    expect = np.degrees((2.0 / np.pi) * 1e-6 / v * t_end)
    assert np.isclose(g["di_deg"][-1], expect, rtol=0.02)


def test_orbit_mean_removes_one_per_rev():
    a = RE + 400e3
    n = np.sqrt(MU / a**3)
    t = np.arange(0.0, 30 * 2 * np.pi / n, 120.0)
    y = 5.0 + np.cos(n * t) + 0.3 * np.cos(2 * n * t)
    ym = orbit_mean(t, y, np.full_like(t, a))
    assert np.max(np.abs(ym - 5.0)) < 0.03
