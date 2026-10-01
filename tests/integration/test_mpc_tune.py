"""MPC_v2 tuner: config round-trip, U575 gate, score, noisy state."""
import numpy as np

from arlamx_v2.advisors.mpc import SamplingMpcPolicy
from arlamx_v2.bench_v8 import _advisor_state, mpc_fn
from arlamx_v2.config import load as load_cfg
from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.inference_budget import mpc_u575_ms
from arlamx_v2.tune_mpc import BUDGET_MS, score_vs


OPT = dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001, soc=0.5,
           omega_dps=[0.0, 0.0, 0.0], f107=150.0, ap=4.0)


def test_from_config_matches_v1_yaml():
    cfg = load_cfg("mpc_v1")
    pol = SamplingMpcPolicy.from_config(cfg, seed=0)
    assert pol.horizon_steps == 6
    assert pol.n_random == 8
    assert pol.SOC_K == 6.0
    assert pol.w["cd"] == 0.5
    assert SamplingMpcPolicy.DIPOLE_QUANT_BITS is None
    q, info = pol.predict(None, {
        "r": np.array([6878137.0, 0.0, 0.0]),
        "v": np.array([0.0, 7612.0, 0.0]),
        "battery_soc": 0.55,
        "sun_N": np.array([1.0, 0.0, 0.0]),
        "rho": 3e-12, "T": 900.0, "m_bar": 2.656e-26,
        "gs_visible": 0.0,
    })
    assert q.shape == (4,)
    assert np.isfinite(float(info["score"]))


def test_u575_v1_under_budget():
    # Published-scale: v1 must stay under 15 ms and near the 8.4 ms thesis number.
    ms = mpc_u575_ms(6, 8, scale="published")
    assert ms < BUDGET_MS
    assert 5.0 < ms < 12.0
    assert mpc_u575_ms(12, 32, scale="published") > BUDGET_MS


def test_score_disqualifies_brownout_and_rewards_all_four():
    anchor = dict(decay_km_d=9.0, band_pct=80.0, downlink_min_d=20.0,
                  slews_per_day=10.0, brownouts=0)
    better = dict(decay_km_d=8.0, band_pct=90.0, downlink_min_d=22.0,
                  slews_per_day=8.0, brownouts=0)
    worse_dl = dict(decay_km_d=8.0, band_pct=90.0, downlink_min_d=10.0,
                    slews_per_day=8.0, brownouts=0)
    brn = dict(better, brownouts=1)
    assert score_vs(better, anchor) > 1.0
    assert score_vs(worse_dl, anchor) < 1.0
    assert score_vs(brn, anchor) is None


def test_noisy_advisor_state_finite():
    env = ArlamxV2Env(seed=3, variant="v8a")
    env.reset(seed=3, options=dict(OPT))
    st = _advisor_state(env, noisy=True)
    assert np.all(np.isfinite(st["r"]))
    assert np.all(np.isfinite(st["v"]))
    assert 0.0 <= st["battery_soc"] <= 1.0
    q = mpc_fn(noisy=True, seed=3)(np.zeros(35, np.float32), env)
    assert q.shape == (7,)
    assert np.isfinite(q).all()
    env.close()
