"""Training session: snapshot naming, content, pinning and exact replay."""
import re

import yaml

from arlamx_v2 import config as cfg_mod
from arlamx_v2 import session as S


def _fake_train_one(captured):
    def fake(variant, algo, timesteps, n_envs, seed, name, out_root, **kw):
        captured.append(dict(variant=variant, algo=algo, timesteps=timesteps, seed=seed,
                             name=name, pinned=dict(cfg_mod._PINNED), **kw))
        return {"wall_s": 0.0, "steps_per_s": 1.0}
    return fake


def test_session_snapshot_and_replay(monkeypatch, tmp_path):
    calls = []
    import arlamx_v2.train as T
    monkeypatch.setattr(T, "train_one", _fake_train_one(calls))
    settings = S.load_settings()
    S.set_path(settings, "train.variant", "v8a")
    S.set_path(settings, "network.timesteps", 7)
    sid, snap_path, _ = S.run(settings, out_root=tmp_path)
    assert re.fullmatch(r"\d\d-\d\d-\d\d-\d\d_PPO_4x16", sid)
    snap = yaml.safe_load(snap_path.read_text())
    assert snap["session"]["status"] == "done"
    assert set(snap) >= {"session", "network", "orbit", "train", "plant", "result"}
    assert snap["network"]["timesteps"] == 7
    assert snap["plant"]["reward"] == cfg_mod.load_reward("v8a")
    assert snap["plant"]["gains_mrp"] == cfg_mod.load("gains_mrp")
    c = calls[0]
    assert c["run_dir"] == tmp_path / "models" / sid
    assert c["env_kw"]["orbit"] == snap["orbit"]["training"]
    assert c["algo_kw"] == snap["network"]["ppo"]
    assert c["pinned"]["gains_mrp"] == snap["plant"]["gains_mrp"]
    assert cfg_mod._PINNED == {}

    replay = S.load_settings(snapshot=snap_path)
    replay["frozen"]["gains_mrp"]["kd"] = 123.0     # proves the snapshot, not the YAML, is used
    sid2, snap2_path, _ = S.run(replay, out_root=tmp_path)
    assert sid2 != sid and sid2.startswith(sid[:11])
    c2 = calls[1]
    assert c2["env_kw"]["frozen"]["gains_mrp"]["kd"] == 123.0
    assert c2["env_kw"]["physics"] == snap["plant"]["physics"]
    assert yaml.safe_load(snap2_path.read_text())["session"]["replay_of"] == str(snap_path)


def test_session_failure_is_recorded(monkeypatch, tmp_path):
    import arlamx_v2.train as T

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(T, "train_one", boom)
    settings = S.load_settings()
    try:
        S.run(settings, out_root=tmp_path)
    except RuntimeError:
        pass
    snap = yaml.safe_load(next((tmp_path / "snapshots").glob("*.yaml")).read_text())
    assert snap["session"]["status"] == "failed"
    assert "boom" in snap["session"]["error"]
    assert cfg_mod._PINNED == {}
