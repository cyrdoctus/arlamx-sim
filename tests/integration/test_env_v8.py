"""SC_v8/v9 environment integration — pins for the grok blockers B1-B6."""
import numpy as np
import pytest

from arlamx_v2.env import ArlamxV2Env

OPT = dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001, soc=0.5,
           omega_dps=[0.0, 0.0, 0.0])


def _mk(variant, **kw):
    return ArlamxV2Env(seed=3, variant=variant, **kw)


def test_obs_dims_35_47_59():
    for var, dim in (("v8a", 35), ("v8b", 35), ("v9a", 47), ("v9b", 59)):
        e = _mk(var)
        o, _ = e.reset(seed=3, options=dict(OPT))
        assert o.shape == (dim,), f"{var}: {o.shape}"
        assert e.action_space.shape == (7,)
        e.close()


def test_gate_floor_comes_from_yaml():
    """grok B4: gains_mrp.yaml gate_floor 0.3 must actually apply."""
    e = _mk("v8b")
    assert e._gate_floor == pytest.approx(0.3)
    e.close()
    e = _mk("v8b", gate_floor=0.2)     # explicit argument still wins
    assert e._gate_floor == pytest.approx(0.2)
    e.close()


def test_mass_pinned_and_custom_coils():
    """grok fix 7: one vehicle card — mass 0.625, custom-coil torques."""
    e = _mk("v8b")
    e.reset(seed=3, options=dict(OPT))
    assert e._mass == pytest.approx(0.625)
    assert np.allclose(e._tau_max * 1e6, [38.244, 38.244, 11.473], rtol=1e-3)
    e.close()


def test_v8a_gates_pinned_open():
    """grok B6: v8a is a real control arm — action[4:7] must be inert."""
    e = _mk("v8a")
    e.reset(seed=3, options=dict(OPT))
    for split_cmd in (-1.0, 0.0, 1.0):
        _, _, _, _, info = e.step(np.array(
            [1, 0, 0, 0, split_cmd, split_cmd, split_cmd], dtype=np.float32))
        assert np.allclose(info["gates"], 1.0)
    e.close()


def test_v8b_gates_follow_split():
    e = _mk("v8b")
    e.reset(seed=3, options=dict(OPT))
    _, _, _, _, info = e.step(np.array([1, 0, 0, 0, 1, 1, 1], dtype=np.float32))
    assert np.allclose(info["gates"], e._gate_floor)      # full delegation
    _, _, _, _, info = e.step(np.array([1, 0, 0, 0, -1, -1, -1], dtype=np.float32))
    assert np.allclose(info["gates"], 1.0)                # none
    e.close()


def _ideal_torque_physics(tmp_path):
    """Standard preset with the actuator declared ideal (the v2.1 plant the
    v8 gains were tuned on), as a YAML path the env accepts."""
    import yaml
    from arlamx_v2.physics import load as load_phys
    cfg = load_phys("standard", overrides={"attitude": {"ideal_torque": True}})
    p = tmp_path / "physics_ideal.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def _slew_deleg(e, n=6):
    from arlamx_v2.reward_v8 import power_saved_fraction
    q = np.array([np.cos(np.radians(20)), np.sin(np.radians(20)), 0, 0])
    Ps, pins = [], []
    for _ in range(n):
        _, _, _, _, info = e.step(np.concatenate([q, [1, 1, 1]]).astype(np.float32))
        Ps.append(float(info["deleg_P"]))
        # The env must feed eq. (4) the FULL-authority demand, so P computed
        # from the logged inputs equals the P the reward saw.
        pins.append(power_saved_fraction(info["tau_ctrl_mean"], info["tau_want_abs"],
                                         e._tau_max, info["gates"]))
    return Ps, pins


def test_deleg_P_fires_on_saturating_slew():
    """grok B1/B2 end-to-end on the DEFAULT closed-loop plant: a 20 deg slew
    with full delegation (gates at the 0.3 floor) must register a saving on
    the slew step. v2.5 fed eq. (4) the gated demand (tau_cmd_mean <=
    gate*tau_max by construction), so the gate could never 'bind' and P was
    identically 0 — the v8b hypothesis was not in the reward at all.

    The threshold is the yaml significance floor p_min (0.05), not a hand
    number: with the v8 gains only the weak Z coil binds on this slew
    (X/Y demand stays under 0.3*tau_max), which gives P ~ 0.25."""
    from arlamx_v2.config import load_reward
    e = _mk("v8b")
    e.reset(seed=4, options=dict(OPT))
    Ps, pins = _slew_deleg(e)
    p_min = float(load_reward("v8b")["delegation"]["p_min"])
    assert Ps[0] > p_min, "gate binding on the slew must register as saving"
    assert np.allclose(Ps, pins, atol=1e-12), "reward P must be eq. (4) of the logged inputs"
    e.close()


def test_deleg_P_zero_once_tracked_ideal_torque(tmp_path):
    """Once the attitude is tracked the demand is below the gate and P must be
    exactly 0 (no false 1-gate payout). Tracking in 30 min needs the
    ideal-couple actuator: the closed-loop magnetorquer has no authority
    about B and does not converge with the v8 gains
    (docs/modules/15_control_magnetorquer.md).

    A 1 deg/s initial rate makes the PD demand exceed the 0.3 gate on the
    first step (P ~ 0.47); after damping the demand is ~1 % of the gated cap.
    A plain 20 deg slew pays ZERO here on purpose: the 0.125 deg/s rate cap
    throttles it long before the gate does, and eq. (4) credits the gate only
    for what the gate itself clipped (grok B2)."""
    from arlamx_v2.config import load_reward
    p_min = float(load_reward("v8b")["delegation"]["p_min"])
    phys = _ideal_torque_physics(tmp_path)

    e = _mk("v8b", physics=phys)
    assert e.sim.params().ideal_torque is True
    e.reset(seed=4, options=dict(OPT, omega_dps=[1.0, 1.0, 0.5]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    Ps, trk = [], []
    for _ in range(6):
        _, _, _, _, info = e.step(np.concatenate([q, [1, 1, 1]]).astype(np.float32))
        Ps.append(float(info["deleg_P"]))
        trk.append(np.degrees(float(info["tracking_err_rad"])))
    assert Ps[0] > p_min, "gate binding while damping 1 deg/s must register"
    assert trk[-1] < 2.0, "ideal couple must have the hold tracked by 30 min"
    assert Ps[-1] == pytest.approx(0.0, abs=1e-6), "tracked: no false 1-gate payout"
    e.close()

    e = _mk("v8b", physics=phys)
    e.reset(seed=4, options=dict(OPT))
    Ps, _ = _slew_deleg(e)
    assert max(Ps) == pytest.approx(0.0, abs=1e-6), "rate-capped slew: not the gate's saving"
    e.close()


def test_brownout_reachable_for_v8():
    """grok B5: the brownout machinery (mode switch, recovery loop) must be
    wired for v8 — before the fix, v8/v9 could never enter it. force_brownout
    is the deterministic entry the recovery scenario uses."""
    e = _mk("v8b")
    e.reset(seed=3, options=dict(OPT, soc=0.02, force_brownout=True,
                                 omega_dps=[4.0, 4.0, 2.0]))
    _, _, _, _, info = e.step(np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert bool(info.get("brownout_active"))
    # and the natural-depletion flag must be reachable too: empty battery in
    # eclipse-heavy conditions reports depletion through _update_power
    e2 = _mk("v8b")
    e2.reset(seed=3, options=dict(OPT, soc=0.0))
    _, _, _, _, i2 = e2.step(np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert "brownout_active" in i2
    e.close(); e2.close()


def test_v9_history_150s_slot_populated():
    """grok 1.3: the -150 s slot was silently always zero (banker's rounding).
    After 3 steps every history slot must be non-zero."""
    e = _mk("v9b")
    o, _ = e.reset(seed=3, options=dict(OPT))
    for _ in range(3):
        o, *_ = e.step(np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    hist = o[47:59].reshape(3, 4)
    assert np.all(np.any(hist != 0.0, axis=1)), f"empty history rows: {hist}"
    e.close()


def test_v9a_trend_nonzero_without_memory():
    """grok 1.3: v9a's charge trend was identically zero; with the SoC
    projection it must be able to move."""
    e = _mk("v9a")
    e.reset(seed=3, options=dict(OPT))
    seen = 0.0
    for _ in range(6):
        _, _, _, _, info = e.step(np.array([1, 0, 0, 0, 0, 0, 0],
                                           dtype=np.float32))
        seen = max(seen, abs(info["trend"]["d_soc"]))
    assert seen > 1e-4
    e.close()


def test_diagnostics_present_in_info():
    """grok B8: the artifacts must carry the quantities that judge delegation."""
    e = _mk("v8b")
    e.reset(seed=3, options=dict(OPT))
    _, _, _, _, info = e.step(np.array([1, 0, 0, 0, 0.5, 0.5, 0.5],
                                       dtype=np.float32))
    for key in ("deleg_B", "deleg_P", "deleg_S", "deleg_split", "tau_want",
                "tau_env_filt", "tau_env_raw", "mtq_power_W",
                "tau_shortfall_frac", "gate_floor", "mass_kg"):
        assert key in info, key
    e.close()


def test_v7_untouched_by_v8_wiring():
    """v3-v7 must keep the catalogue coils, the 0.15 default floor, and the
    randomized mass path."""
    e = ArlamxV2Env(seed=7, variant="v7")
    assert np.allclose(e._tau_max * 1e6, [18.0, 18.0, 5.7])
    assert e._gate_floor == pytest.approx(0.15)
    e.close()
