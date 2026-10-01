"""SC_v1_6mq pieces: magnetorquer duty gap, v3 vehicle card, wrapper observation and reward terms."""
import numpy as np
import pytest

from arlamx_v2 import cpp
from arlamx_v2.config import load
from arlamx_v2.vehicle import build


def _step(duty):
    p = cpp.SimParams()
    p.use_panel_srp = False
    p.gravity_gradient = False
    p.advisor_step_s = 2.0
    p.mtq_duty = duty
    sim = cpp.Simulator(p)
    v = build(load("vehicle_v3"))
    sim.set_rods(v.rod_axis, v.rod_dmax)
    r = np.array([cpp.RE_WGS + 450e3, 0, 0.0])
    sim.reset(r, np.array([0, np.sqrt(cpp.MU_WGS / np.linalg.norm(r)), 0.0]), np.array([0.1, 0.0, 0.05]), np.zeros(3))
    return sim.step(np.array([1.0, 0, 0, 0]))


def test_mtq_duty_scales_torque_and_energy():
    a, b = _step(1.0), _step(0.5)
    assert np.allclose(np.asarray(b["tau_ctrl_mean"]), 0.5 * np.asarray(a["tau_ctrl_mean"]), rtol=1e-12)
    assert np.allclose(b["rod_duty2_mean"], 0.5 * np.asarray(a["rod_duty2_mean"]), rtol=1e-12)
    p = cpp.SimParams()
    p.mtq_duty = 0.0
    with pytest.raises(ValueError):
        cpp.Simulator(p)


def test_v3_vehicle():
    v = build(load("vehicle_v3"))
    assert v.mass == pytest.approx(0.750)
    ang = np.degrees(np.arctan2(v.rod_pos[:, 1], v.rod_pos[:, 0])) % 360
    assert np.allclose(np.sort(ang), [30, 90, 150, 210, 270, 330])
    assert np.allclose(np.abs(v.rod_axis @ (v.rod_pos.T / np.linalg.norm(v.rod_pos, axis=1))).diagonal(), 0, atol=1e-12)
    assert v.I[2, 2] == pytest.approx(v.I[0, 0] + v.I[1, 1], rel=1e-12)


def test_wrapper():
    from arlamx_v2.sc6mq import CFG, make_env
    from arlamx_v2.session import read_yaml
    env = make_env(read_yaml(CFG), 0, 3)()
    o, _ = env.reset(seed=3)
    assert o.shape == env.observation_space.shape
    o, r, te, tr, info = env.step(env.action_space.sample())
    parts = info["sc6mq_parts"]
    assert np.isfinite(r) and r == pytest.approx(sum(parts.values()))
    assert {"base", "slew", "env", "rest", "downlink", "band", "low", "brownout"} <= set(parts)
    if info["eclipse_est"] > 0.5:
        assert np.all(o[-4:-1] == 0.0)
    else:
        assert np.linalg.norm(o[-4:-1]) == pytest.approx(1.0, rel=1e-5)
