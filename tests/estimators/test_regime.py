"""SC_v14 onboard regime scheduler — numbers pin, no plant."""
import numpy as np

from arlamx_v2.regime import RegimeConfig, RegimeEstimator


def test_quiet_defaults():
    r = RegimeEstimator()
    sch = r.peek()
    assert sch["kd"] == 1.0e-3
    assert sch["gate_floor"] == 0.30
    assert sch["margin"] == 0.05
    assert abs(sch["clip_kappa"] - 0.70) < 1e-12
    assert sch["eta_eff"] == 0.0
    assert sch["kp"] == 6.0e-5


def test_full_storm_schedules():
    r = RegimeEstimator()
    sch = r.update(np.array([3.0e-5, 0.0, 0.0]))
    assert abs(sch["eta"] - 1.0) < 1e-12
    assert abs(sch["kd"] - 6.0e-4) < 1e-12
    assert abs(sch["gate_floor"] - 0.15) < 1e-12
    assert abs(sch["margin"] - 0.15) < 1e-12
    assert abs(sch["clip_kappa"] - 1.0) < 1e-12


def test_eta_clipped():
    r = RegimeEstimator()
    sch = r.update(np.array([1.0e-3, 0.0, 0.0]))
    assert sch["eta"] == 1.0


def test_ewma_lags_step():
    r = RegimeEstimator()
    r.update(np.zeros(3))
    r.update(np.array([3.0e-5, 0.0, 0.0]))
    # first update set eta_slow = 0; second mixes in alpha ~ 0.054
    a = r.alpha
    assert 0.04 < a < 0.07
    assert r.eta == 1.0
    assert abs(r.eta_slow - a) < 1e-9
    assert r.eta_slow < 0.2


def test_shock_on_jump():
    r = RegimeEstimator()
    r.update(np.zeros(3))
    r.update(np.array([3.0e-5, 0.0, 0.0]))
    assert r.shock == 1.0
    assert r.eta_eff == 1.0


def test_shock_on_tau_dot():
    r = RegimeEstimator()
    r.update(np.array([1e-10, 0.0, 0.0]), tau_dot=np.array([4e-8, 0.0, 0.0]))
    # |tau_dot| / 1e-7 = 0.4 > 0.30
    assert r.shock == 1.0


def test_no_shock_on_slow_climb():
    """~8 orbits of linear climb (gradual climate, not a storm flash)."""
    r = RegimeEstimator()
    n = 150
    for k in range(n):
        tau = np.array([(k / float(n)) * 3e-5, 0.0, 0.0])
        r.update(tau)
    assert r.shock == 0.0
    assert r.eta > 0.9
    assert abs(r.eta - r.eta_slow) < r.cfg.shock_deta


def test_authority_ratio():
    r = RegimeEstimator()
    B = np.array([0.0, 0.0, 2.5963e-5])
    tau = np.array([3.824350e-5, 0.0, 0.0])  # dipole * B at ref
    r.update(tau, B=B)
    # |tau| / (1.473 * |B|) ≈ 1
    assert abs(r.authority - 1.0) < 1e-3
