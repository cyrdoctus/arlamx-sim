"""Gym env smoke: 26-dim obs, finite reward. SC_v3 / SC_v4 / v5 / v6 unit tests."""
import numpy as np
from arlamx_v2.env import ArlamxV2Env, MTQ_KD, MTQ_KP, MTQ_TAU_NM
from arlamx_v2.reward import (
    action_feasibility,
    compose_sc_v3,
    compose_sc_v4,
    compose_sc_v5,
    compose_sc_v6,
    dE_vs_baseline,
    lift_work,
    power_band,
    power_budget,
    srp_work,
)
from arlamx_v2 import cpp


def test_reward_units():
    assert dE_vs_baseline(-1.0, -1.0, 400.0) == 0.0
    r = power_budget(0.8, 0.0, False)
    assert abs(r - 0.8 * 0.8) < 1e-12


def test_power_band_edges():
    assert power_band(0.5, False) == 1.0
    assert power_band(0.0, True) == -50.0
    assert power_band(0.2, False) < 0.0
    assert 0.0 < power_band(1.0, False) < 0.2


def test_action_feasibility_band():
    assert action_feasibility(10.0, False) == 0.1
    assert action_feasibility(80.0, False) == -1.0
    assert action_feasibility(0.0, True) == -5.0


def test_compose_v4_has_band():
    info = {
        "dE_actual": -1.0,
        "dE_baseline": -1.0,
        "altitude_km": 400.0,
        "battery_soc": 0.5,
        "battery_depleted": False,
        "power_gen_norm": 0.4,
        "Cd": 1.0,
        "gs_visible": 0.0,
        "gs_err_axes_deg": np.zeros(3),
        "eclipse": 1.0,
        "shade_draw": 0.0,
        "q_new": np.array([1.0, 0, 0, 0]),
        "q_prev": np.array([1.0, 0, 0, 0]),
        "omega": np.zeros(3),
        "cmd_angle_deg": 5.0,
        "action_invalid": False,
    }
    rew, parts = compose_sc_v4(info)
    assert "power_band" in parts
    assert parts["power_band"] == 1.0
    assert np.isfinite(rew)


def test_env_reset_step():
    env = ArlamxV2Env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (26,)
    assert np.all(np.isfinite(obs))
    for _ in range(2):
        obs, rew, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
        assert obs.shape == (26,)
        assert np.isfinite(rew)
        assert "Cd" in info
    env.close()


def test_env_v4a_step():
    env = ArlamxV2Env(seed=1, variant="v4a")
    obs, _ = env.reset(seed=1)
    obs, rew, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.isfinite(rew)
    assert "power_band" in info["reward_parts"]
    env.close()


def _base_info(**kw):
    info = {
        "dE_actual": -1.0,
        "dE_baseline": -1.2,
        "altitude_km": 380.0,
        "battery_soc": 0.5,
        "battery_depleted": False,
        "power_gen_norm": 0.4,
        "Cd": 1.0,
        "gs_visible": 0.0,
        "gs_err_axes_deg": np.zeros(3),
        "eclipse": 1.0,
        "shade_draw": 0.0,
        "q_new": np.array([1.0, 0, 0, 0]),
        "q_prev": np.array([1.0, 0, 0, 0]),
        "omega": np.zeros(3),
        "cmd_angle_deg": 5.0,
        "action_invalid": False,
        "dE_lift": 2.0,
        "dE_srp": 1.0,
        "srp_scale": 1.0,
        "torque_effort": 0.2,
        "used_propagator": False,
        "held_inference": False,
        "brownout_recovered": False,
        "brownout_active": False,
    }
    info.update(kw)
    return info


def test_v5_reward_has_lift_srp_longevity():
    rew, parts = compose_sc_v5(_base_info())
    assert "lift_work" in parts
    assert "srp_work" in parts
    assert parts["dE_vs_baseline"] > 0.0
    assert lift_work(2.0, 300.0) > lift_work(2.0, 500.0)
    assert srp_work(4.0, 500.0, 4.0) > srp_work(4.0, 300.0, 1.0)
    assert np.isfinite(rew)


def test_v5b_recovery_bonus():
    rew, parts = compose_sc_v5(
        _base_info(brownout_recovered=True, recover_steps=3, brownout_active=False),
        brownout=True,
    )
    assert parts["brownout_recovery"] >= 180.0
    assert parts["recover_speed"] > 0.0
    assert np.isfinite(rew)


def test_v6_thrift_and_gps():
    rew, parts = compose_sc_v6(
        _base_info(used_propagator=True, held_inference=True, torque_effort=0.2),
        brownout=True,
    )
    assert parts["gps_dropout_skill"] > 0.0
    assert parts["inference_thrift"] == 0.15
    assert parts["torque_thrift"] < 0.0
    assert np.isfinite(rew)


def test_env_v5a_step():
    env = ArlamxV2Env(seed=2, variant="v5a")
    obs, _ = env.reset(seed=2, options={"altitude_km": 400.0, "inc_deg": 23.0, "ecc": 0.001, "soc": 0.5})
    assert env.action_space.shape == (4,)
    obs, rew, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.isfinite(rew)
    assert "lift_work" in info["reward_parts"]
    assert "srp_work" in info["reward_parts"]
    env.close()


def test_env_v6_action_and_gps():
    env = ArlamxV2Env(seed=3, variant="v6")
    obs, _ = env.reset(seed=3, options={"altitude_km": 400.0, "inc_deg": 23.0, "ecc": 0.001, "soc": 0.5})
    assert env.action_space.shape == (5,)
    assert env._advisor_s == 150.0
    used = False
    for _ in range(12):
        obs, rew, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0, 0.0, -0.5]))
        assert np.isfinite(rew)
        used = used or bool(info.get("used_propagator", False))
        if term or trunc:
            break
    env.close()
    assert used


def test_mtq_gains_granular_at_2deg():
    c = cpp.MRPFeedback()
    c.kp = MTQ_KP
    c.kd = MTQ_KD
    c.set_inertia_diag(np.array([0.0125, 0.0125, 0.025]))
    c.set_max_torque(MTQ_TAU_NM)
    sig = np.array([np.tan(np.radians(2.0) / 4.0), 0.0, 0.0])
    tau = c.compute(sig, np.zeros(3), np.zeros(3), np.zeros(3), 2.0)
    assert abs(float(tau[0])) < 0.95 * MTQ_TAU_NM[0]
    sig40 = np.array([np.tan(np.radians(40.0) / 4.0), 0.0, 0.0])
    tau40 = c.compute(sig40, np.zeros(3), np.zeros(3), np.zeros(3), 2.0)
    assert abs(float(tau40[0])) <= MTQ_TAU_NM[0] + 1e-12


def test_brownout_can_recover():
    recovered = False
    for nu in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0):
        env = ArlamxV2Env(seed=5, variant="v5b")
        obs, _ = env.reset(
            seed=5,
            options={
                "altitude_km": 400.0,
                "inc_deg": 23.0,
                "ecc": 0.001,
                "nu_deg": nu,
                "soc": 0.0,
                "force_brownout": True,
                "f107": 150.0,
                "ap": 4.0,
                "omega_dps": [4.0, -3.5, 3.0],
                "mass_kg": 0.625,
            },
        )
        for _ in range(40):
            obs, rew, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
            if info.get("brownout_recovered") or not info.get("brownout_active", True):
                recovered = True
                break
            if term or trunc:
                break
        env.close()
        if recovered:
            break
    assert recovered, "sun-search brownout recovery never handed back in 40 steps at any nu"
