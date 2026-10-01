"""Config / CLI wiring that would silently corrupt standard-vs-high comparisons."""
import inspect

import numpy as np
import pytest

from arlamx_v2 import cpp
from arlamx_v2 import physics as physics_mod
from arlamx_v2 import session as S
from arlamx_v2.sweep import expand_runs, run_sweep
from arlamx_v2.train_v12 import _eval_wrapped, _wrap_kw
from conftest import ggm_path


def test_high_preset_keeps_walker_branch():
    cfg = physics_mod.load("high")
    assert cfg["aero"]["gsi"] == "cll"
    assert cfg["aero"]["alpha_n"] < 1.0
    assert cfg["aero"]["alpha_n"] == pytest.approx(0.93)


def test_gsi_case_is_canonical():
    cfg = physics_mod.load("standard", overrides={"aero": {"gsi": "CLL"}})
    assert cfg["aero"]["gsi"] == "cll"
    assert physics_mod.gsi_name(cfg) == "cll"
    p = cpp.SimParams()
    physics_mod.apply_to_params(p, cfg, ggm_path="/tmp/dummy.gfc")
    assert p.gsi == "cll"


def test_ideal_torque_is_declared_physics():
    """attitude.ideal_torque is a snapshotted plant choice, default false."""
    cfg = physics_mod.load("standard")
    assert cfg["attitude"]["ideal_torque"] is False
    p = cpp.SimParams()
    physics_mod.apply_to_params(p, cfg, ggm_path="/tmp/dummy.gfc")
    assert p.ideal_torque is False
    cfg = physics_mod.load("standard", overrides={"attitude": {"ideal_torque": True}})
    physics_mod.apply_to_params(p, cfg, ggm_path="/tmp/dummy.gfc")
    assert p.ideal_torque is True
    with pytest.raises(ValueError, match="ideal_torque"):
        physics_mod.load("standard", overrides={"attitude": {"ideal_torque": "yes"}})


def test_unknown_gsi_rejected():
    with pytest.raises(ValueError, match="gsi"):
        physics_mod.load("standard", overrides={"aero": {"gsi": "maxwell"}})


def test_twobody_keeps_ggm_and_degree_zero():
    cfg = physics_mod.load("standard", overrides={"gravity": {"model": "TWOBODY"}})
    assert cfg["gravity"]["model"] == "twobody"
    path = ggm_path() or "/tmp/dummy.gfc"
    p = cpp.SimParams()
    physics_mod.apply_to_params(p, cfg, ggm_path=path)
    assert p.sh_degree == 0
    assert p.ggm_path == path


def test_validate_does_not_mutate_input():
    raw = {"aero": {"gsi": "CLL"}, "gravity": {"model": "GGM03S", "degree": 4}}
    # Incomplete on purpose: validate should copy, then the missing srp
    # key raises. The input dict must stay untouched either way.
    before = {"aero": dict(raw["aero"]), "gravity": dict(raw["gravity"])}
    with pytest.raises(ValueError, match="srp.enabled"):
        physics_mod.validate(raw)
    assert raw == before
    cfg = physics_mod.load("standard", overrides={"aero": {"gsi": "CLL"}})
    assert cfg["aero"]["gsi"] == "cll"
    std = physics_mod.load("standard")
    # load("standard") must not have been mutated by the CLL override load.
    assert std["aero"]["gsi"] == "sentman"


def test_twobody_eval_is_point_mass_not_j2():
    path = ggm_path()
    if not path:
        pytest.skip("GGM03S.txt not found")
    g = cpp.GravityHarmonics()
    assert g.load_ggm(path, 2)
    r = np.array([5.0e6, 0.0, 4.5e6])
    a0 = g.accel_ecef(r, 0)
    a_tb = cpp.accel_twobody(r, g.mu())
    a_j2 = a_tb + cpp.accel_j2(r, g.mu(), g.Re(), g.J2())
    assert np.linalg.norm(a0 - a_tb) / np.linalg.norm(a_tb) < 1e-14
    assert np.linalg.norm(a0 - a_j2) / np.linalg.norm(a_j2) > 1e-4


def _gravity_only_v(sh_degree, ggm, dt=1.0):
    p = cpp.SimParams()
    p.dt_s = dt
    p.advisor_step_s = dt
    p.rk4_step_s = dt
    p.sh_degree = sh_degree
    p.ggm_path = ggm
    p.wmm_path = ""
    p.use_panel_srp = False
    p.corotating = False
    p.lunisolar = False
    sim = cpp.Simulator(p)
    sim.set_mode("prescribed")
    sim.set_atmosphere(0.0, 900.0, 2.656e-26)
    r = np.array([cpp.RE_WGS + 400e3, 0.0, 0.0])
    v = np.array([0.0, 7670.0, 0.0])
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    return np.asarray(sim.step(np.array([1.0, 0.0, 0.0, 0.0]))["v"], float), sim


def test_twobody_simulator_is_point_mass_not_j2_fallback():
    path = ggm_path()
    if not path:
        pytest.skip("GGM03S.txt not found")
    cfg = physics_mod.load(
        "standard", overrides={"gravity": {"model": "twobody", "lunisolar": False}})
    p = cpp.SimParams()
    p.use_panel_srp = False
    p.corotating = False
    physics_mod.apply_to_params(p, cfg, ggm_path=path, wmm_path="")
    p.use_panel_srp = False
    p.lunisolar = False
    p.dt_s = p.advisor_step_s = p.rk4_step_s = 1.0
    sim = cpp.Simulator(p)
    assert sim.gravity_loaded()
    assert sim.params().sh_degree == 0
    v_cfg, _ = _gravity_only_v(0, path)
    v_nofile, sim0 = _gravity_only_v(0, "")
    v_j2, _ = _gravity_only_v(4, "")
    assert not sim0.gravity_loaded()
    # Loaded GGM degree 0 and no-file degree 0 are both point-mass (μ may
    # differ at file vs WGS). Both must stay off the J2+J3 fallback.
    assert np.linalg.norm(v_cfg - v_j2) / np.linalg.norm(v_cfg) > 1e-6
    assert np.linalg.norm(v_nofile - v_j2) / np.linalg.norm(v_nofile) > 1e-6


def test_bad_ggm_path_throws_at_simulator():
    p = cpp.SimParams()
    p.ggm_path = "/no/such/arlamx_ggm_file.gfc"
    p.wmm_path = ""
    with pytest.raises(RuntimeError, match="GGM failed to load"):
        cpp.Simulator(p)


def test_train_v12_accepts_physics_kwargs():
    from arlamx_v2.train_v12 import make_env, train_run
    assert "physics" in inspect.signature(train_run).parameters
    assert "gsi" in inspect.signature(train_run).parameters
    assert "controller" in inspect.signature(train_run).parameters
    assert "physics" in inspect.signature(make_env).parameters
    assert "physics" in inspect.signature(_eval_wrapped).parameters
    ekw = _wrap_kw(0.5, False, 0.0, False, False, 0.0,
                   physics="high", controller="quaternion", gsi="cll")
    assert ekw["physics"] == "high"
    assert ekw["controller"] == "quaternion"
    assert ekw["gsi"] == "cll"


def test_sweep_omitted_controller_does_not_force_mrp():
    runs = expand_runs({"name": "t", "base": {"train.backend": "v12", "train.physics": "high"}})
    assert len(runs) == 1
    settings = S.load_settings()
    for k, v in runs[0][1].items():
        S.set_path(settings, k, v)
    assert settings["train"]["controller"] is None
    assert settings["train"]["physics"] == "high"


V12_ARM = {
    "train.backend": "v12", "train.physics": "high", "train.controller": "quaternion",
    "train.gsi": "cll", "train.v12.w_slew": 0.1,
    "network.timesteps": 1, "network.n_envs": 1, "network.seed": 0,
}


def _fake_train_run(captured):
    def fake(*args, **kwargs):
        captured.setdefault("calls", []).append((args, kwargs))
        return {"wall_s": 0.0}
    return fake


def test_sweep_v12_forwards_physics(monkeypatch, tmp_path):
    captured = {}
    import arlamx_v2.train_v12 as tv
    monkeypatch.setattr(tv, "train_run", _fake_train_run(captured))
    rows = run_sweep({"name": "wiring_v12", "base": V12_ARM}, out_root=tmp_path)
    assert rows[0]["status"] == "done"
    args, kw = captured["calls"][0]
    assert args[1] == 0.1
    assert kw["physics"]["name"] == "high"
    assert kw["physics"]["aero"]["gsi"] == "cll"
    assert kw["controller"] == "quaternion"
    assert kw["run_dir"] == tmp_path / "models" / rows[0]["id"]
    assert (tmp_path / "snapshots" / f"{rows[0]['id']}.yaml").is_file()


def test_sweep_skips_done_arms(monkeypatch, tmp_path):
    captured = {}
    import arlamx_v2.train_v12 as tv
    monkeypatch.setattr(tv, "train_run", _fake_train_run(captured))
    cfg = {"name": "wiring_skip", "base": V12_ARM}
    run_sweep(cfg, out_root=tmp_path)
    run_sweep(cfg, out_root=tmp_path)
    assert len(captured["calls"]) == 1
    run_sweep(cfg, out_root=tmp_path, force=True)
    assert len(captured["calls"]) == 2
    assert (tmp_path / "results" / "sweeps" / "wiring_skip.csv").is_file()
