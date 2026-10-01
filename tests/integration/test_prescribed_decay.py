"""Prescribed-attitude hold must stay finite and lose energy under drag.

Sources: Vallado 2013 §8 (drag energy); Hairer / Nørsett / Wanner (RK4).
"""

import numpy as np
import pytest

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import MaxDragPolicy, MinDragPolicy, drag_extreme_axes
from arlamx_v2.decay_run import coe_to_rv, load_panels
from conftest import ggm_path


def _sim():
    p = cpp.SimParams()
    p.dt_s = 30.0
    p.advisor_step_s = 30.0
    p.rk4_step_s = 10.0
    p.use_panel_srp = True
    p.corotating = True
    p.sh_degree = 4
    path = ggm_path()
    if path:
        p.ggm_path = path
    p.mode = "prescribed"
    p.max_slew_rad = np.pi
    sim = cpp.Simulator(p)
    sim.set_mode("prescribed")
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    a = np.array([0.3, 0.3])
    c = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]])
    sim.set_panels(n, a, c)
    sim.set_atmosphere(3e-12, 900.0, 2.656e-26)
    return sim


def test_prescribed_max_drag_finite_one_orbit():
    sim = _sim()
    r0, v0 = coe_to_rv(cpp.RE_WGS + 500e3, 0.001, np.radians(23.0), 0.0, 0.0, 0.0)
    sim.reset(r0, v0, np.zeros(3), np.zeros(3))
    pol = MaxDragPolicy()
    sma0 = None
    for _ in range(20):
        st = sim.get_state()
        q, _ = pol.predict(None, st)
        out = sim.step(q)
        assert np.isfinite(out["altitude_km"])
        assert np.isfinite(out["sma_m"])
        assert np.isfinite(out["omega"]).all()
        assert abs(out["omega"][0]) < 1e-12
        if sma0 is None:
            sma0 = out["sma_m"]
    assert out["sma_m"] < sma0
    assert out["Cd"] > 0.3


def test_prescribed_min_drag_lower_cd_than_max():
    n, a, c = load_panels("hex")
    min_ax, max_ax, proj = drag_extreme_axes(n, a)
    r0, v0 = coe_to_rv(cpp.RE_WGS + 500e3, 0.001, np.radians(23.0), 0.0, 0.0, 0.0)
    cds = {}
    for name, pol in (
        ("min", MinDragPolicy(body_axis=min_ax)),
        ("max", MaxDragPolicy(body_axis=max_ax)),
    ):
        sim = _sim()
        sim.set_panels(n, a, c)
        sim.reset(r0, v0, np.zeros(3), np.zeros(3))
        st = sim.get_state()
        st["min_drag_axis"] = min_ax
        st["max_drag_axis"] = max_ax
        q, _ = pol.predict(None, st)
        out = sim.step(q)
        cds[name] = float(out["Cd"])
    # v2.1: the edge-on membrane now carries its free-molecular skin friction
    # (C_tau = 1/(s sqrt(pi)) per face, Sentman 1961), so min-drag Cd on the
    # hex is ~0.21-0.26 at the 4 deg co-rotation offset, not the 0.08-0.12 the
    # v2.0 cos(theta) > 0 shortcut gave. Face-on is unchanged.
    assert 0.15 < cds["min"] < 0.30
    assert cds["max"] > 1.5
    assert cds["max"] > 6.0 * cds["min"]
    assert proj[min_ax] < 0.05
    assert proj[max_ax] > 0.4


def test_infeasible_atmosphere_rejected():
    sim = _sim()
    with pytest.raises(Exception):
        sim.set_atmosphere(-1.0, 900.0, 2.6e-26)
    with pytest.raises(Exception):
        sim.set_atmosphere(1e-12, 0.0, 2.6e-26)


def test_reset_inside_earth_rejected():
    sim = _sim()
    with pytest.raises(Exception):
        sim.reset(np.zeros(3), np.array([0.0, 7e3, 0.0]), np.zeros(3), np.zeros(3))


def test_huge_rk4_step_rejected():
    p = cpp.SimParams()
    p.rk4_step_s = 3600.0
    p.dt_s = 3600.0
    p.advisor_step_s = 3600.0
    with pytest.raises(Exception):
        cpp.Simulator(p)
