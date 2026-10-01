"""Six beam torque rods: allocation, gating, saturation, mass properties."""
import numpy as np
import pytest

from arlamx_v2 import cpp
from arlamx_v2.vehicle import axis_authority, build

VEH = build()


def ref_alloc(m, U, dmax, on):
    d = np.zeros(len(U))
    if on.any():
        d[on] = np.linalg.pinv(U[on].T) @ m        # minimum-norm solution
    s = min([1.0] + [dmax[k] / abs(d[k]) for k in range(len(d)) if abs(d[k]) > dmax[k]])
    d *= s
    return U.T @ d, d


@pytest.mark.parametrize("on", [[1] * 6, [1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0] * 6])
@pytest.mark.parametrize("m", [[0.3, -0.2, 0.5], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
def test_allocation_matches_pseudoinverse(on, m):
    on = np.array(on, bool)
    m = np.array(m, float)
    got_m, got_d = cpp.rods_allocate(m, VEH.rod_axis, VEH.rod_dmax, list(on))
    ref_m, ref_d = ref_alloc(m, VEH.rod_axis, VEH.rod_dmax, on)
    assert np.allclose(got_d, ref_d, atol=1e-12)
    assert np.allclose(got_m, ref_m, atol=1e-12)
    assert np.all(np.abs(got_d) <= VEH.rod_dmax + 1e-15)
    assert np.all(np.asarray(got_d)[~on] == 0.0)
    assert abs(got_m[2]) < 1e-15                   # rods lie in the membrane plane


def test_small_request_is_met_exactly_in_plane():
    m = np.array([0.2, -0.1, 0.4])
    got, _ = cpp.rods_allocate(m, VEH.rod_axis, VEH.rod_dmax, [True] * 6)
    assert np.allclose(got, [0.2, -0.1, 0.0], atol=1e-14)


def test_saturation_preserves_direction():
    m = np.array([30.0, 10.0, 0.0])
    got, d = cpp.rods_allocate(m, VEH.rod_axis, VEH.rod_dmax, [True] * 6)
    assert np.isclose(np.max(np.abs(d)), VEH.rod_dmax[0])
    assert np.allclose(np.cross(got, m), 0.0, atol=1e-12)


def test_mass_properties():
    assert VEH.mass == pytest.approx(0.750, abs=1e-12)
    assert np.allclose(VEH.com, 0.0, atol=1e-15)
    I = VEH.I
    assert np.allclose(I, np.diag(np.diag(I)), atol=1e-15)
    assert I[2, 2] == pytest.approx(I[0, 0] + I[1, 1], rel=1e-12)   # planar body
    assert I[0, 0] == pytest.approx(I[1, 1], rel=1e-12)              # 6-fold symmetry


def test_axis_authority():
    assert np.allclose(axis_authority(np.ones(6, bool), VEH), 1.0)
    assert np.allclose(axis_authority(np.zeros(6, bool), VEH), 0.0)


def test_plant_rods_gate_and_duty():
    p = cpp.SimParams()
    p.use_panel_srp = False
    p.gravity_gradient = False
    p.advisor_step_s = 20.0
    sim = cpp.Simulator(p)
    sim.set_rods(VEH.rod_axis, VEH.rod_dmax)
    r = np.array([cpp.RE_WGS + 450e3, 0, 0.0])
    v = np.array([0, np.sqrt(cpp.MU_WGS / np.linalg.norm(r)), 0.0])
    sim.reset(r, v, np.array([0.1, 0.0, 0.05]), np.zeros(3))
    o = sim.step(np.array([1.0, 0, 0, 0]))
    assert len(o["rod_duty2_mean"]) == 6 and max(o["rod_duty2_mean"]) > 0.0
    sim.set_rod_gates([False] * 6)
    o = sim.step(np.array([1.0, 0, 0, 0]))
    assert np.all(np.asarray(o["tau_ctrl_mean"]) == 0.0)
    assert np.all(np.asarray(o["rod_duty2_mean"]) == 0.0)


def test_env_v10r6():
    from arlamx_v2.env import ArlamxV2Env
    base = ArlamxV2Env(seed=1, variant="v10")
    env = ArlamxV2Env(seed=1, variant="v10r6")
    assert env.action_space.shape == (10,)
    assert env.observation_space.shape[0] == base.observation_space.shape[0] + 6
    o, _ = env.reset(seed=1)
    assert np.all(o[-6:] == 1.0)
    assert env._mass == pytest.approx(0.750)
    assert np.allclose(env.sim.params().inertia if hasattr(env.sim.params(), "inertia") else env._I, VEH.I_diag)
    a = np.array([1, 0, 0, 0, 1, -1, 1, -1, 1, -1], np.float32)
    o, r, term, trunc, info = env.step(a)
    assert np.isfinite(r)
    assert list(env.sim.rod_gates()) == [True, False, True, False, True, False]
    assert np.all(o[-6:] == [1, 0, 1, 0, 1, 0])
    assert np.all((0.0 <= info["deleg_split"]) & (info["deleg_split"] <= 1.0))
    off = np.array([1, 0, 0, 0, -1, -1, -1, -1, -1, -1], np.float32)
    _, _, _, _, info = env.step(off)
    assert np.allclose(info["deleg_split"], 1.0)
    assert np.all(np.asarray(info["tau_ctrl_mean"]) == 0.0)
    assert env._pdiag["power_W"] == 0.0


def test_apply_rods_is_torque_least_squares():
    rng = np.random.default_rng(0)
    for _ in range(50):
        B = rng.normal(size=3) * 3e-5
        tau = rng.normal(size=3) * 1e-6
        on = rng.random(6) > 0.3
        got, m, d, sf = cpp.apply_rods(tau, B, VEH.rod_axis, VEH.rod_dmax * 100, list(on))
        A = np.array([np.cross(u, B) for u in VEH.rod_axis]).T[:, on]
        dref = np.zeros(6)
        if on.any():
            dref[on] = np.linalg.pinv(A) @ tau
        assert np.allclose(d, dref, atol=1e-9 * max(1.0, np.abs(dref).max()))
        best = A @ dref[on] if on.any() else np.zeros(3)
        assert np.allclose(got, best, atol=1e-18 + 1e-9 * np.linalg.norm(best))
        assert abs(m[2]) < 1e-15


def test_apply_rods_beats_dipole_projection():
    B = np.array([9.27e-6, 36.32e-6, 11.38e-6]); tau = np.array([1.51e-6, -18.61e-6, 0.85e-6])
    got, m, d, sf = cpp.apply_rods(tau, B, VEH.rod_axis, VEH.rod_dmax, [True] * 6)
    m_old, _ = cpp.rods_allocate(np.cross(B, tau) / (B @ B), VEH.rod_axis, VEH.rod_dmax, [True] * 6)
    err_new = np.linalg.norm(got - tau); err_old = np.linalg.norm(np.cross(m_old, B) - tau)
    assert err_new <= err_old * (1 + 1e-12)
