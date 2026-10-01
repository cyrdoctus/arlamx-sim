"""11 — Fused plant smoke. Integrator: Hairer / Nørsett / Wanner; vis-viva Vallado."""
import numpy as np
from arlamx_v2 import cpp
from conftest import ggm_path


def circular_rv(alt_m, mu, Re):
    r = Re + alt_m
    v = np.sqrt(mu / r)
    return np.array([r, 0.0, 0.0]), np.array([0.0, v, 0.0])


def test_circular_sma_stable_one_period():
    p = cpp.SimParams()
    p.dt_s = 10.0
    p.advisor_step_s = 10.0
    p.rk4_step_s = 10.0
    p.use_panel_srp = False
    p.sh_degree = 0
    path = ggm_path()
    if path:
        p.ggm_path = path
        p.sh_degree = 0  # two-body if we force degree < 2
    sim = cpp.Simulator(p)
    mu = cpp.MU_WGS
    r, v = circular_rv(400e3, mu, cpp.RE_WGS)
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    sim.set_atmosphere(0.0, 1000.0, 2.6e-26)
    sma0 = cpp.sma_from_rv(r, v, mu)
    # One period ~ 5554 s → 556 steps of 10 s
    q = np.array([1.0, 0.0, 0.0, 0.0])
    p.advisor_step_s = 10.0
    for _ in range(56):  # ~560 s smoke (full period is slow for CI; check 0.1 orbit)
        o = sim.step(q)
    sma1 = o["sma_m"]
    assert abs(sma1 - sma0) < 0.05  # metres over ~0.1 orbit, vacuum, no SH
    # Measured capability is ~1e-7 m; a 50 m band would pass a badly broken RK4.


def test_drag_removes_energy():
    p = cpp.SimParams()
    p.dt_s = 2.0
    p.advisor_step_s = 20.0
    p.use_panel_srp = False
    sim = cpp.Simulator(p)
    r, v = circular_rv(400e3, cpp.MU_WGS, cpp.RE_WGS)
    # Single plate face-on to velocity (gas onto +Y if v is +Y)
    n = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    A = np.array([1.0, 1.0])
    c = np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]])
    sim.set_panels(n, A, c)
    sim.set_atmosphere(3e-12, 900.0, 2.656e-26)
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    o = sim.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert o["dE_actual"] < 0.0
    assert o["Cd"] > 0.5


def test_baseline_work_is_in_the_inertial_frame():
    """v2.1 fix. dE_baseline is the work the min-drag counterfactual would do
    on the INERTIAL motion, the same frame as dE_actual / dE_drag. With the
    vehicle held exactly at its min-drag attitude the two must agree; v2.0
    charged the baseline with |v_rel| instead of vhat_rel . v and so sat ~6 %
    low on an equatorial prograde orbit."""
    p = cpp.SimParams()
    p.dt_s = 2.0
    p.advisor_step_s = 2.0     # one substep: the held attitude stays aligned
    p.rk4_step_s = 2.0
    p.use_panel_srp = False
    p.corotating = True
    p.mode = "prescribed"
    p.max_slew_rad = np.pi
    path = ggm_path()
    if path:
        p.ggm_path = path
    sim = cpp.Simulator(p)
    sim.set_mode("prescribed")
    # Cube: every axis-aligned orientation is the same wetted geometry, so the
    # only difference between the live force and the counterfactual is the
    # frame the work is charged in.
    n = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    A = np.full(6, 0.25)
    c = 0.25 * n
    sim.set_panels(n, A, c)
    sim.set_atmosphere(5e-12, 900.0, 2.656e-26)
    mu = cpp.MU_WGS
    r, v = circular_rv(400e3, mu, cpp.RE_WGS)
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    # Body +X along the AIRSPEED (v - omega x r), which is what min-drag means.
    we = np.array([0.0, 0.0, 7.2921150e-5])
    from arlamx_v2.advisors.attitudes import quat_body_axis_along
    q = quat_body_axis_along(v - np.cross(we, r), "x", helper_n=np.cross(r, v))
    o = sim.step(q)
    assert o["dE_actual"] < 0.0 and o["dE_baseline"] < 0.0
    rel = abs(o["dE_actual"] - o["dE_baseline"]) / abs(o["dE_baseline"])
    assert rel < 1e-2, f"min-drag attitude scores {rel*100:.2f} % off its own baseline"


def test_step_exports_exact_rotational_identity_closed_loop():
    """v2.6: I*(omega1-omega0)/T == tau_aero_mean + tau_ctrl_mean - gyro_mean
    to round-off on the default closed-loop magnetorquer plant, with the PD
    fighting a plate's aero torque. gyro_mean is the RK4-consistent mean of
    omega x I omega (same stage states and Simpson weights as the omega
    update), which is what makes the observer identity exact at any rate."""
    p = cpp.SimParams()
    p.dt_s = 2.0
    p.advisor_step_s = 60.0
    p.use_panel_srp = False
    assert p.ideal_torque is False
    sim = cpp.Simulator(p)
    r, v = circular_rv(400e3, cpp.MU_WGS, cpp.RE_WGS)
    n = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    A = np.array([0.5, 0.5])
    c = np.array([[0.02, 0.05, 0.01], [0.02, -0.05, 0.01]])   # cp offset -> aero torque
    sim.set_panels(n, A, c)
    sim.set_atmosphere(3e-12, 900.0, 2.656e-26)
    ctrl = cpp.MRPFeedback()
    ctrl.kp, ctrl.kd = 4e-4, 8e-3
    ctrl.set_inertia_diag(np.array([0.0125, 0.0125, 0.025]))
    ctrl.set_max_torque(np.array([1.8e-5, 1.8e-5, 5.7e-6]))
    sim.set_controller(ctrl)
    sim.reset(r, v, np.array([0.05, -0.02, 0.03]), np.array([1e-3, -5e-4, 2e-4]))
    I = np.array([0.0125, 0.0125, 0.025])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        om0 = np.asarray(sim.get_state()["omega"], float)
        o = sim.step(q)
        om1 = np.asarray(o["omega"], float)
        T = o["n_substeps"] * p.dt_s
        lhs = I * (om1 - om0) / T
        rhs = (np.asarray(o["tau_env_mean"]) + np.asarray(o["tau_ctrl_mean"])
               - np.asarray(o["gyro_mean"]))
        assert np.linalg.norm(o["tau_aero_mean"]) > 0.0
        assert np.linalg.norm(lhs - rhs) <= 1e-12 * np.linalg.norm(rhs)
        # the demand at full authority can exceed the gated command, never the reverse
        assert np.all(np.abs(o["tau_demand_absmean"]) + 1e-18 >= np.abs(o["tau_cmd_mean"]))
        assert np.all(o["tau_demand_absmean"] >= np.abs(o["tau_demand_mean"]) - 1e-18)
