"""Classroom test for FixedAoAPolicy: the wind sits at the requested angle
of attack in the body frame, the DCM is proper, and the Sentman lift points
where ``lift_toward`` says."""
import numpy as np
import pytest

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import OMEGA_EARTH, FixedAoAPolicy, drag_extreme_axes
from arlamx_v2.decay_run import coe_to_rv, load_panels

RE = cpp.RE_WGS


def _state(nu_deg):
    return coe_to_rv(RE + 500e3, 0.001, np.radians(23.0), 0.3, 0.1, np.radians(nu_deg))


@pytest.mark.parametrize("alpha", [0.0, 15.0, 45.0, 90.0])
@pytest.mark.parametrize("lift_toward", ["normal", "anti_normal", "zenith", "nadir"])
def test_aoa_and_frame(alpha, lift_toward):
    pol = FixedAoAPolicy(alpha, "y", "z", lift_toward=lift_toward)
    for nu in (0.0, 47.0, 200.0):
        r, v = _state(nu)
        c = pol.dcm_bn(r, v)
        assert np.allclose(c @ c.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(c), 1.0, atol=1e-12)
        w_b = c @ pol.wind_dir(r, v)
        assert np.isclose(np.degrees(np.arcsin(np.clip(w_b[2], -1, 1))), alpha, atol=1e-9)
        assert np.isclose(w_b[1], np.cos(np.radians(alpha)), atol=1e-12)
        assert abs(w_b[0]) < 1e-12
        q, _ = pol.predict(None, {"r": r, "v": v})
        c_q = cpp.mrp_to_dcm(cpp.mrp_shadow(_q_to_mrp(q)))
        assert np.allclose(c_q, c, atol=1e-9)


def _q_to_mrp(q):
    q = np.asarray(q, float)
    return q[1:] / (1.0 + q[0])


@pytest.mark.parametrize("lift_toward", ["normal", "anti_normal", "zenith", "nadir"])
def test_lift_direction_hex(lift_toward):
    nrm, area, cen = load_panels("hex")
    mn, mx, _ = drag_extreme_axes(nrm, area)
    pol = FixedAoAPolicy(45.0, mn, mx, lift_toward=lift_toward)
    r, v = _state(80.0)
    c = pol.dcm_bn(r, v)
    v_rel = v - np.cross([0.0, 0.0, OMEGA_EARTH], r)
    v_b_gas = -(c @ v_rel)
    a = cpp.spacecraft_aero(nrm, area, cen, v_b_gas, 1e-12, 900.0, 2.656e-26, 300.0, 0.93, True)
    f_n = c.T @ np.asarray(a["force"], float)
    w = v_rel / np.linalg.norm(v_rel)
    lift = f_n - float(f_n @ w) * w
    h = np.cross(r, v)
    h /= np.linalg.norm(h)
    rhat = r / np.linalg.norm(r)
    target = {"normal": h, "anti_normal": -h, "zenith": rhat, "nadir": -rhat}[lift_toward]
    assert float(f_n @ w) < 0.0  # drag opposes the wind
    assert np.linalg.norm(lift) > 0.05 * abs(float(f_n @ w))  # L/D ~ 0.1 at 45 deg
    cosang = float(lift @ target) / np.linalg.norm(lift)
    assert cosang > 0.99
