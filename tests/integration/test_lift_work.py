"""Corotating wind: lift can change inertial energy. Vallado 2013 §8.6.2."""

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import quat_body_axis_along
from arlamx_v2.decay_run import coe_to_rv


def _plate_sim(corotating):
    p = cpp.SimParams()
    p.dt_s = 5.0
    p.advisor_step_s = 5.0
    p.rk4_step_s = 5.0
    p.use_panel_srp = False
    p.corotating = corotating
    p.mode = "prescribed"
    p.max_slew_rad = np.pi
    p.sh_degree = 0
    sim = cpp.Simulator(p)
    sim.set_mode("prescribed")
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    a = np.array([0.65, 0.65])
    c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sim.set_panels(n, a, c)
    sim.set_atmosphere(3e-12, 900.0, 2.656e-26)
    return sim


def test_no_corotation_lift_work_vanishes():
    sim = _plate_sim(False)
    r, v = coe_to_rv(cpp.RE_WGS + 400e3, 0.0, np.radians(23.0), 0.0, 0.0, 0.0)
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    q = quat_body_axis_along(v, "z")
    out = sim.step(q)
    assert abs(out["dE_lift"]) < 1e-14


def test_corotation_lift_work_nonzero_at_aoa():
    sim = _plate_sim(True)
    r, v = coe_to_rv(cpp.RE_WGS + 400e3, 0.0, np.radians(51.6), 0.0, 0.0, 0.0)
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    h = np.cross(r, v)
    mid = v / np.linalg.norm(v) + 0.4 * h / np.linalg.norm(h)
    q = quat_body_axis_along(mid, "z", helper_n=h)
    out = sim.step(q)
    assert np.isfinite(out["dE_lift"])
    assert abs(out["dE_lift"]) > 1e-12
    assert abs(out["dE_drag"] + out["dE_lift"] - out["dE_actual"]) < 1e-14


def test_corotating_default_on():
    p = cpp.SimParams()
    assert p.corotating is True
