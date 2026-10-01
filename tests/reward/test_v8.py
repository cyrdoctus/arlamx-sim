"""14 — SC_v8 delegation reward. Citations: Schaub & Junkins 2018 Ch. 8 (MRP-PD
demand); grok critique 2026-08-20 (B1-B3: the counterfactual and alignment
failure modes these tests pin)."""
import numpy as np
import pytest

from arlamx_v2.config import load_reward
from arlamx_v2.reward_v8 import (altitude_margin, compose_sc_v8, decay_stability,
                                 delegation_alignment, delegation_benefit,
                                 power_saved_fraction, split_to_gates)

TAU_MAX = np.array([3.8244e-5, 3.8244e-5, 1.1473e-5])


def test_split_to_gates_endpoints():
    g = split_to_gates([0.0, 0.5, 1.0], 0.3)
    assert np.allclose(g, [1.0, 0.65, 0.3])


# --- eq. (4): the grok B2 regimes ------------------------------------------

def test_power_saved_zero_when_gate_not_binding():
    """PD below gate*tau_max: gating changed nothing, so P must be 0 — the old
    e_act/gate formula paid 1-gate here."""
    gates = np.full(3, 0.5)
    tau_want = 0.2 * gates * TAU_MAX          # well below the gate ceiling
    tau_ctrl = tau_want.copy()                # controller applies the demand
    assert power_saved_fraction(tau_ctrl, tau_want, TAU_MAX, gates) == 0.0


def test_power_saved_positive_when_saturating():
    """Demand beyond the gated ceiling: the gate genuinely withheld torque."""
    gates = np.full(3, 0.3)
    tau_want = 2.0 * TAU_MAX                  # saturating demand
    tau_ctrl = gates * TAU_MAX                # what the gated controller applies
    P = power_saved_fraction(tau_ctrl, tau_want, TAU_MAX, gates)
    assert 0.5 < P <= 1.0                     # e_full=1, e_act=0.3 -> P=0.7


def test_power_saved_ignores_rate_cap_savings():
    """e_act below demand WITHOUT the gate binding (rate cap) earns nothing."""
    gates = np.ones(3)                        # gate wide open
    tau_want = 0.8 * TAU_MAX
    tau_ctrl = 0.1 * TAU_MAX                  # rate cap held it back, not the gate
    assert power_saved_fraction(tau_ctrl, tau_want, TAU_MAX, gates) == 0.0


def test_power_saved_ignores_projection_loss():
    """v2.6: the closed-loop magnetorquer delivers only the B-perpendicular
    part of the command, so measured |tau_ctrl| sits well below gate*tau_max.
    That loss belongs to the field geometry, not the gate — eq. (4) must pay
    e_full - e_gate (0.7 here), never e_full - e_act (0.95)."""
    gates = np.full(3, 0.3)
    tau_want = 2.0 * TAU_MAX                  # saturating demand, gate binds
    tau_ctrl = 0.05 * TAU_MAX                 # projection left 1/6 of the gated cmd
    P = power_saved_fraction(tau_ctrl, tau_want, TAU_MAX, gates)
    assert P == pytest.approx(0.7, abs=1e-12)


def test_power_saved_partial_binding():
    """Demand between gate*tau_max and tau_max: saving is exactly the clipped
    part, per axis, normalised by the full-authority duty."""
    gates = np.array([0.3, 1.0, 0.3])
    tau_want = np.array([0.5, 0.5, 0.2]) * TAU_MAX
    P = power_saved_fraction(np.zeros(3), tau_want, TAU_MAX, gates)
    # axis0: 0.5-0.3=0.2 ; axis1: gate open -> 0 ; axis2: 0.2<0.3 -> 0 ; den=1.2
    assert P == pytest.approx(0.2 / 1.2, abs=1e-12)


# --- eq. (2): the grok B3 alignment ----------------------------------------

def test_alignment_uses_env_scale_not_pd_ratio():
    """A 30 nN*m aero torque against a uN*m PD demand must still register
    (the old |env|/|want| ratio buried it at ~1e-3)."""
    tau_env = np.array([3e-8, 0.0, 0.0])      # 30 nN*m, twice tau_env_ref
    tau_want = np.array([2e-6, 0.0, 0.0])     # uN-class PD slew demand
    a = delegation_alignment(tau_env, tau_want, tau_env_ref=1.5e-8)
    assert a[0] == pytest.approx(1.0)         # saturated: big enough to matter


def test_alignment_signed_against_hostile_environment():
    a = delegation_alignment(np.array([-3e-8, 0, 0]), np.array([2e-6, 0, 0]),
                             tau_env_ref=1.5e-8)
    assert a[0] == pytest.approx(-1.0)


def test_alignment_zero_when_controller_wants_nothing():
    a = delegation_alignment(np.array([3e-8, 0, 0]), np.zeros(3))
    assert np.all(a == 0.0)


def test_benefit_weights_by_split():
    align = np.array([1.0, -1.0, 0.0])
    assert delegation_benefit([1.0, 0.0, 0.0], align) == pytest.approx(1.0)
    assert delegation_benefit([0.0, 1.0, 0.0], align) == pytest.approx(-1.0)
    assert delegation_benefit([0.0, 0.0, 0.0], align) == 0.0


# --- stability terms --------------------------------------------------------

def test_decay_stability_one_sided_and_capped():
    # recovery (dE rising) is free
    assert decay_stability(-400.0, -500.0, weight=0.6, scale=100.0) == 0.0
    # degradation is charged
    assert decay_stability(-600.0, -500.0, weight=0.6, scale=100.0) == pytest.approx(-0.6)
    # one storm step cannot exceed the cap
    assert decay_stability(-5000.0, -500.0, weight=0.6, scale=100.0, cap=3.0) == pytest.approx(-1.8)


def test_altitude_margin_quadratic_barrier():
    assert altitude_margin(400.0, 4.0, 320.0, 250.0) == 0.0
    assert altitude_margin(250.0, 4.0, 320.0, 250.0) == pytest.approx(-4.0)
    mid = altitude_margin(285.0, 4.0, 320.0, 250.0)
    assert -4.0 < mid < 0.0


# --- composite wiring -------------------------------------------------------

def _base_info():
    return {
        "dE_actual": -400.0, "dE_baseline": -500.0, "altitude_km": 400.0,
        "battery_soc": 0.5, "battery_depleted": False, "power_gen_norm": 0.5,
        "Cd": 1.0, "gs_point_cos": 0.0, "gs_visible": 0.0,
        "gs_err_axes_deg": np.zeros(3), "eclipse": 1.0, "shade_draw": 0.0,
        "q_new": np.array([1.0, 0, 0, 0]), "q_prev": np.array([1.0, 0, 0, 0]),
        "omega": np.zeros(3), "cmd_angle_deg": 0.0, "action_invalid": False,
        "brownout_recovered": False, "brownout_active": False,
        "torque_effort": 0.1, "recover_steps": 0, "dE_prev": -400.0,
        "tau_want": 2.0 * TAU_MAX, "tau_env_est": np.array([3e-8, 0, 0]),
        "deleg_split": np.ones(3), "gates": np.full(3, 0.3),
        "tau_ctrl_mean": 0.3 * TAU_MAX, "tau_max": TAU_MAX,
    }


def test_v8b_deleg_terms_fire_with_real_inputs():
    cfg = load_reward("v8b")
    total, parts = compose_sc_v8(_base_info(), cfg)
    assert parts["deleg_power"] > 0.0
    assert parts["deleg_accuracy"] != 0.0
    assert np.isfinite(total)


def test_v8a_deleg_terms_are_zero():
    cfg = load_reward("v8a")
    _, parts = compose_sc_v8(_base_info(), cfg)
    assert parts["deleg_power"] == 0.0
    assert parts["deleg_accuracy"] == 0.0


def test_boost_only_amplifies_positive_terms():
    cfg = load_reward("v8b")
    info = _base_info()
    info["dE_actual"] = -600.0                # dE term goes negative
    _, parts = compose_sc_v8(info, cfg)
    cfg_off = load_reward("v8a")
    info2 = _base_info()
    info2["dE_actual"] = -600.0
    _, parts_off = compose_sc_v8(info2, cfg_off)
    # the negative dE term must be identical across arms (no boost on penalties)
    assert parts["dE_vs_baseline"] == pytest.approx(parts_off["dE_vs_baseline"])


def test_flat_override_reaches_v7_weights():
    """grok 1.1: campaign-style {'dE_weight': x} must not be silently ignored."""
    hi = load_reward("v8a", {"dE_weight": 10.0})
    lo = load_reward("v8a", {"dE_weight": 0.1})
    assert hi["v7"]["dE_weight"] == 10.0 and lo["v7"]["dE_weight"] == 0.1
