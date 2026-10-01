"""Flow-frame zero-order hold: target follows the flow; frame switch; dcm_to_quat; env cadence options."""
import numpy as np
import pytest

from arlamx_v2 import cpp


def test_dcm_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.normal(size=4); q /= np.linalg.norm(q); q *= np.sign(q[0])
        C = cpp.mrp_to_dcm(cpp.quat_to_mrp(q))
        q2 = cpp.dcm_to_quat(C)
        assert np.allclose(q2, q, atol=1e-12)


def _sim(frame):
    p = cpp.SimParams()
    p.use_panel_srp = False
    p.gravity_gradient = False
    p.cmd_frame = frame
    sim = cpp.Simulator(p)
    r = np.array([cpp.RE_WGS + 400e3, 0, 0.0])
    v = np.array([0, np.sqrt(cpp.MU_WGS / np.linalg.norm(r)), 0.0])
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    sim.set_mode("prescribed")
    return sim


def test_prescribed_flow_frame_keeps_body_x_on_ram():
    sim = _sim("flow")
    we = np.array([0, 0, cpp.OMEGA_EARTH])
    for _ in range(6):
        sim.step(np.array([1.0, 0, 0, 0]))            # identity q_BF: body x on v_rel
        st = sim.get_state()
        vr = np.asarray(st["v"]) - np.cross(we, np.asarray(st["r"]))
        vb = np.asarray(st["C_BN"]) @ vr
        assert np.degrees(np.arccos(vb[0] / np.linalg.norm(vb))) < 1.0   # within one 2 s substep of flow rotation


def test_frame_validation_and_switch():
    p = cpp.SimParams(); p.cmd_frame = "bogus"
    with pytest.raises(ValueError):
        cpp.Simulator(p)
    sim = _sim("flowS")
    sim.set_cmd_frame("flowB")
    assert sim.params().cmd_frame == "flowB"
    with pytest.raises(ValueError):
        sim.set_cmd_frame("nope")


def test_env_cadence_options():
    from arlamx_v2.env import ArlamxV2Env
    e = ArlamxV2Env(seed=0, variant="v10", advisor_s=600.0, inner_dt=0.5)
    assert e.sim.params().advisor_step_s == 600.0 and e.sim.params().dt_s == 0.5 and e.sim.params().rk4_step_s <= 0.5
    e.reset(seed=0)
    o, r, te, tr, info = e.step(np.r_[1.0, 0, 0, 0, np.zeros(3)].astype(np.float32))
    assert info["tracking_err_rad"] >= 0 and np.isfinite(r)
