"""Advisor smoke: min/max drag quaternions and heuristic/MPC finite output."""

import numpy as np

from arlamx_v2.advisors import (
    HEURISTIC_BANK,
    GroundStationPolicy,
    MaxDragPolicy,
    MinDragPolicy,
    NadirPolicy,
    SamplingMpcPolicy,
)


def _state():
    r = np.array([6878137.0, 0.0, 0.0])
    v = np.array([0.0, 7612.0, 0.0])
    return {
        "r": r,
        "v": v,
        "battery_soc": 0.55,
        "power_gen_norm": 0.0,
        "gs_visible": 0.0,
        "gs_dir_B": None,
        "sun_N": np.array([1.0, 0.0, 0.0]),
        "rho": 3e-12,
        "T": 900.0,
        "m_bar": 2.656e-26,
    }


def test_min_max_unit_quat():
    st = _state()
    st["min_drag_axis"] = "y"
    st["max_drag_axis"] = "z"
    qn, _ = MinDragPolicy(body_axis="y").predict(None, st)
    qx, _ = MaxDragPolicy(body_axis="z").predict(None, st)
    assert abs(np.linalg.norm(qn) - 1.0) < 1e-12
    assert abs(np.linalg.norm(qx) - 1.0) < 1e-12


def test_heuristic_bank_steps():
    st = _state()
    for _name, pol in HEURISTIC_BANK.items():
        pol.reset()
        q, info = pol.predict(None, st)
        assert q.shape == (4,)
        assert np.isfinite(q).all()
        assert info["name"]


def test_mpc_picks_a_quat():
    st = _state()
    mpc = SamplingMpcPolicy(horizon_steps=2, n_random=2, seed=0)
    q, info = mpc.predict(None, st)
    assert q.shape == (4,)
    assert np.isfinite(float(info["score"]))


def test_nadir_and_gs_finite():
    st = _state()
    qn, _ = NadirPolicy().predict(None, st)
    st["gs_visible"] = 1.0
    st["gs_dir_N"] = -st["r"] / np.linalg.norm(st["r"])
    qg, info = GroundStationPolicy().predict(None, st)
    assert abs(np.linalg.norm(qn) - 1.0) < 1e-12
    assert abs(np.linalg.norm(qg) - 1.0) < 1e-12
    assert info["mode"] == "gs"
