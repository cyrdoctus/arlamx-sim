"""Classroom tests for the SC_v7 IPC variant (per docs/VALIDATION_STANDARD.md).

Covers: action/observation contract, per-axis authority gates reaching the
plant, the disturbance-torque observer against the plant's true aero torque,
and the compose_sc_v7 weight knobs.
"""
import numpy as np
import pytest

from arlamx_v2.env import ArlamxV2Env, MTQ_TAU_NM
from arlamx_v2.reward import V7_DEFAULT_W, compose_sc_v7, power_band_v7

OPT = dict(altitude_km=350.0, inc_deg=23.0, ecc=0.001, f107=180.0, ap=10.0,
           raan_deg=0.0, argp_deg=0.0, nu_deg=0.0, mass_kg=0.625, soc=0.5,
           omega_dps=[0.05, 0.05, 0.02])


@pytest.fixture(scope="module")
def env():
    e = ArlamxV2Env(seed=3, variant="v7")
    yield e
    e.close()


def test_spaces(env):
    assert env.action_space.shape == (7,)
    assert env.observation_space.shape == (35,)


def test_gate_floor_honoured(env):
    env.reset(seed=3, options=dict(OPT))
    a = np.array([1, 0, 0, 0, -1, -1, -1], dtype=np.float32)  # release all axes
    env.step(a)
    assert np.allclose(env._gates, env._gate_floor)
    a = np.array([1, 0, 0, 0, 1, 1, 1], dtype=np.float32)     # full authority
    env.step(a)
    assert np.allclose(env._gates, 1.0)


def test_observer_recovers_aero_torque(env):
    """tau_dist_est must equal the plant's true mean aero torque (the only
    disturbance modelled) on the DEFAULT closed-loop magnetorquer plant.

    v2.6: the observer uses the plant's RK4-consistent <omega x I omega>
    (gyro_mean) instead of the endpoint omega1 x I omega1. Under the closed
    loop the body rates drift within a 300 s step and the endpoint term alone
    was the size of the aero torque (cos 0.83, |est| 1.8x truth). The
    identity now holds to round-off at ANY rate, so no near-zero-rate premise
    is needed; the tolerance below is the KF-free v7 path's numeric floor."""
    env.reset(seed=3, options=dict(OPT))
    a = np.array([1, 0, 0, 0, 1, 1, 1], dtype=np.float32)
    info = None
    for _ in range(6):
        _o, _r, _t, _tr, info = env.step(a)
        truth = np.asarray(info["tau_env_mean"], float)   # aero + SRP + Earth rad + gravity gradient
        est = np.asarray(info["tau_dist_est"], float)
        assert np.linalg.norm(truth) > 0.0
        assert np.linalg.norm(est - truth) < 1e-6 * np.linalg.norm(truth)
    assert env.sim.params().ideal_torque is False


def test_effort_is_real_torque(env):
    env.reset(seed=3, options=dict(OPT))
    a = np.array([1, 0, 0, 0, 1, 1, 1], dtype=np.float32)
    _o, _r, _t, _tr, info = env.step(a)
    expect = float(np.clip(np.mean(np.abs(np.asarray(info["tau_ctrl_mean"]))
                                   / MTQ_TAU_NM), 0.0, 1.0))
    assert info["torque_effort"] == pytest.approx(expect, rel=1e-9)


def test_obs_ablation_masks():
    for mode, live in (("none", 0), ("ctrl", 4), ("observer", 7)):
        e = ArlamxV2Env(seed=1, variant="v7", obs_extra=mode)
        obs, _ = e.reset(seed=1, options=dict(OPT))
        e.step(np.array([1, 0, 0, 0, 1, 1, 1], dtype=np.float32))
        obs, *_ = e.step(np.array([1, 0, 0, 0, 1, 1, 1], dtype=np.float32))
        block = obs[26:]
        assert block.shape == (9,)
        assert int(np.sum(np.abs(block) > 0)) <= live
        e.close()


def test_power_band_v7_modes():
    w = dict(V7_DEFAULT_W)
    assert power_band_v7(0.5, False, w) == pytest.approx(1.0)
    assert power_band_v7(0.2, False, w) < 0.0
    assert power_band_v7(0.0, True, w) == pytest.approx(-50.0)
    # exp mode stays positive above the band; linear_penalty crosses zero
    assert 0.0 < power_band_v7(0.95, False, w) < 1.0
    w2 = dict(V7_DEFAULT_W, band_above_mode="linear_penalty", band_above_slope=2.0)
    assert power_band_v7(1.0, False, w2) == pytest.approx(-2.0)
    assert power_band_v7(0.6, False, w2) == pytest.approx(1.0)


def test_compose_sc_v7_weight_overrides():
    info = dict(battery_soc=0.5, battery_depleted=False, shade_draw=0.0,
                dE_actual=-1.0, dE_baseline=-1.0, altitude_km=400.0,
                gs_err_axes_deg=np.array([1.0, 1.0, 1.0]), gs_visible=1.0,
                eclipse=1.0, cmd_angle_deg=10.0, action_invalid=False,
                q_new=np.array([1.0, 0, 0, 0]), q_prev=np.array([1.0, 0, 0, 0]),
                omega=np.zeros(3), torque_effort=0.5,
                brownout_recovered=False, brownout_active=False,
                power_gen_norm=0.0)
    _r1, p1 = compose_sc_v7(info)
    assert p1["gs_tiered_pointing"] == pytest.approx(6.0)   # 3 axes x weight 2
    assert p1["torque_thrift"] == pytest.approx(-0.25 * 0.5)
    _r2, p2 = compose_sc_v7(info, w={"gs_weight": 4.0, "thrift_weight": 0.9})
    assert p2["gs_tiered_pointing"] == pytest.approx(12.0)
    assert p2["torque_thrift"] == pytest.approx(-0.45)
