"""SC_v8Duo library. Citations: docs/modules/14 Sec. 9; duo.yaml."""
import numpy as np
import pytest

from arlamx_v2 import duo
from arlamx_v2.config import load
from arlamx_v2.geometry import load_geom
from arlamx_v2.paths import HEX_GEOM
from arlamx_v2.reward_v8 import duo_reward

MU = 3.986004418e14
RE = 6378137.0
CFG = load("duo")


def _state(alt=400e3, soc=0.5):
    r = np.array([RE + alt, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(MU / (RE + alt)), 0.0])
    return dict(r=r, v=v, soc=soc, rho=3e-12, alt_km=alt / 1e3,
                sun_N=np.array([1.0, 0.0, 0.0]), f107=150.0, bc_inv=1.0,
                p_gen_sunlit_W=0.78 * 0.85 * 0.8, batt_cap_J=0.53 * 3600.0)


def test_horizon_observation_shape_and_soc_moves():
    obs, meta = duo.horizon_observation(_state(), np.array([1.0, 0, 0, 0]), CFG)
    assert obs.shape == (int(CFG["horizon"]["obs_dim"]),)
    assert np.isfinite(obs).all()
    # grok 1.5: SoC must actually be projected, and eclipse must matter.
    assert meta["soc_end_prop"] != pytest.approx(0.5)
    assert 0.0 < meta["ecl_frac"] < 1.0


def test_attitude_bc_scale_separates_candidates():
    """Broadside vs edge-on must give very different drag factors — without
    this the arbiter compared identical decays (grok 1.5)."""
    n, A, c = load_geom(str(HEX_GEOM))
    v_n = np.array([0.0, 1.0, 0.0])
    q_edge = np.array([1.0, 0.0, 0.0, 0.0])
    # 90 deg about X turns the sail normal into the flow
    q_broad = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    s = duo.attitude_bc_scale(n, A, q_broad, q_edge, v_n)
    assert s > 3.0


def test_arbiter_overrides():
    qm = np.array([1.0, 0, 0, 0])
    qh = np.array([np.cos(np.radians(15)), np.sin(np.radians(15)), 0, 0])
    sm = dict(_state(), bc_scale=1.0, power_norm=0.5, downlink_norm=0.2)
    sh = dict(_state(), bc_scale=0.6, power_norm=0.7, downlink_norm=0.1)
    # critical battery -> higher power wins regardless
    q, info = duo.arbitrate(qm, qh, dict(sm, soc=0.30), sh, qm, CFG)
    assert info["reason"] == "soc_critical" and info["choice"] == "horizon"
    # comfortable battery + pass due -> downlink candidate
    q, info = duo.arbitrate(qm, qh, dict(sm, soc=0.70, downlink_soon=True),
                            sh, qm, CFG)
    assert info["reason"] == "downlink_priority" and info["choice"] == "main"
    # infeasible horizon candidate (beyond the slew cap) is rejected
    q_far = np.array([np.cos(np.radians(80)), np.sin(np.radians(80)), 0, 0])
    q, info = duo.arbitrate(qm, q_far, sm, sh, qm, CFG)
    assert info["choice"] == "main"


def test_arbiter_switch_margin_keeps_main():
    """Near-identical candidates must not oscillate: main wins the tie."""
    qm = np.array([1.0, 0, 0, 0])
    qh = np.array([np.cos(0.01), np.sin(0.01), 0, 0])
    sm = dict(_state(), bc_scale=1.0, power_norm=0.5, downlink_norm=0.5)
    sh = dict(_state(), bc_scale=0.999, power_norm=0.5, downlink_norm=0.5)
    _, info = duo.arbitrate(qm, qh, sm, sh, qm, CFG)
    assert info["choice"] == "main"


def test_secondary_reward_pays_for_beating_main():
    good = duo_reward({"decay_prop_km_d": 5.0, "decay_main_km_d": 15.0,
                                  "soc_end_prop": 0.5, "q_dot": 0.95}, CFG)[0]
    bad = duo_reward({"decay_prop_km_d": 15.0, "decay_main_km_d": 5.0,
                                 "soc_end_prop": 0.5, "q_dot": 0.95}, CFG)[0]
    assert good > bad


def test_pick_duo_models_rules():
    rows = [
        {"name": "a", "variant": "v8b", "algo": "ppo", "arch": "4x16",
         "decay_nominal": 8.0, "brownouts": 0, "deleg_benefit": 0.2},
        {"name": "b", "variant": "v8b", "algo": "td3", "arch": "4x20",
         "decay_nominal": 12.0, "brownouts": 0, "deleg_benefit": 0.6},
        {"name": "c", "variant": "v8b", "algo": "sac", "arch": "4x24",
         "decay_nominal": 5.0, "brownouts": 3, "deleg_benefit": 0.1},
        {"name": "d", "variant": "v8a", "algo": "ppo", "arch": "4x16",
         "decay_nominal": 1.0, "brownouts": 0, "deleg_benefit": 0.9},
    ]
    pick = duo.pick_duo_models(rows, CFG)
    assert pick["main"]["name"] == "a"        # best decay among zero-brownout v8b
    assert pick["secondary"]["name"] == "b"   # best delegation benefit (v8b only)
