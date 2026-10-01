"""15 — Magnetorquer power and authority. Citations: Wertz 1978 Sec. 6.5
(m = NIA, tau = m x B); Sidi 1997 Sec. 7.5 (magnetic control limits)."""
import numpy as np
import pytest

from arlamx_v2.config import load
from arlamx_v2.power import (achievable_torque, dipole_for_torque, mtq_power_w)

CFG = load("power_mtq")
M_MAX = np.asarray(CFG["magnetorquers"]["dipole_max_Am2"])
P_PEAK = np.asarray(CFG["magnetorquers"]["power_peak_W"])
B_REF = float(CFG["magnetorquers"]["b_ref_T"])
TAU_MAX = np.asarray(CFG["magnetorquers"]["torque_max_Nm"])


def test_config_self_consistent():
    """tau_max = dipole_max * b_ref must hold or the vehicle card is split."""
    assert np.allclose(TAU_MAX, M_MAX * B_REF, rtol=1e-3)


def test_dipole_inverse_roundtrip():
    """m = (B x tau)/|B|^2 must reproduce tau for a perpendicular request."""
    B = np.array([0.0, 0.0, B_REF])
    tau = np.array([1e-5, 0.0, 0.0])
    m, unach = dipole_for_torque(tau, B)
    assert unach == pytest.approx(0.0)
    assert np.allclose(np.cross(m, B), tau, rtol=1e-12)


def test_power_quadratic_in_torque():
    B = np.array([0.0, 0.0, B_REF])
    p1, _ = mtq_power_w(np.array([1e-6, 0, 0]), B, CFG)
    p2, _ = mtq_power_w(np.array([2e-6, 0, 0]), B, CFG)
    assert p2 == pytest.approx(4.0 * p1, rel=1e-9)


def test_full_scale_power_matches_peak():
    """Full X torque with B along Z engages the Y coil at its full dipole."""
    B = np.array([0.0, 0.0, B_REF])
    p, d = mtq_power_w(np.array([TAU_MAX[0], 0, 0]), B, CFG)
    assert p == pytest.approx(P_PEAK[1], rel=1e-3)
    assert not d["saturated"]


def test_along_B_torque_unachievable():
    B = np.array([B_REF, 0.0, 0.0])
    tau = np.array([1e-5, 0.0, 0.0])          # parallel to B
    ta, short = achievable_torque(tau, B, CFG)
    assert short == pytest.approx(1.0)
    assert np.linalg.norm(ta) < 1e-12


def test_weak_axis_routing_shortfall():
    """X torque with B along Y must route through the weak Z coil: the
    achievable torque collapses to the Z coil's share."""
    B = np.array([0.0, B_REF, 0.0])
    tau = np.array([TAU_MAX[0], 0.0, 0.0])
    ta, short = achievable_torque(tau, B, CFG)
    assert np.linalg.norm(ta) == pytest.approx(M_MAX[2] * B_REF, rel=1e-3)
    assert 0.5 < short < 0.9


def test_legacy_mode_reproduces_v7():
    cfg = load("power_mtq")
    cfg["magnetorquers"]["model"] = "legacy_linear"
    tau = np.asarray(cfg["magnetorquers"]["torque_max_Nm"])
    p, d = mtq_power_w(tau, np.array([0.0, 0.0, B_REF]), cfg)
    assert d["model"] == "legacy_linear"
    assert p == pytest.approx(cfg["magnetorquers"]["legacy_peak_W"], rel=1e-9)
