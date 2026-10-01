"""YAML config loading (config/plant), merge, reward selection per variant, snapshot pinning."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

SETTINGS_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG_DIR = SETTINGS_DIR / "plant"

_CACHE = {}
_PINNED = {}


def pin(cfgs):
    _PINNED.clear()
    for k, v in (cfgs or {}).items():
        if v is not None:
            _PINNED[k] = json.loads(json.dumps(v))


def config_path(name):
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    cand = CONFIG_DIR / f"{name}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"no config '{name}' in {CONFIG_DIR}")


def load(name, use_cache=True):
    if str(name) in _PINNED:
        return json.loads(json.dumps(_PINNED[str(name)]))
    path = config_path(name)
    key = str(path)
    if use_cache and key in _CACHE:
        return json.loads(json.dumps(_CACHE[key]))
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    _CACHE[key] = data
    return json.loads(json.dumps(data))


def merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def load_reward(variant, override=None):
    SIX = [900.0, 750.0, 600.0, 450.0, 300.0, 150.0]
    if "reward" in _PINNED:
        cfg = json.loads(json.dumps(_PINNED["reward"]))
    elif variant == "v11":
        cfg = load("reward_v11")
    elif variant in ("v10", "v10r6"):
        cfg = load("reward_v10")
    elif variant in ("v8a", "v8b"):
        cfg = load(f"reward_{variant}")
    elif variant in ("v9a", "v9b", "v9c", "v9d"):
        cfg = load("reward_v9")
        obs = {"history": variant != "v9a"}
        if variant in ("v9c", "v9d"):
            obs["history_offsets_s"] = SIX
        if variant == "v9d":
            obs["future_offsets_s"] = sorted(SIX)
        cfg = merge(cfg, {"variant": variant, "name": f"SC_{variant}",
                          "observation": obs})
    else:
        raise ValueError(f"no reward config for variant {variant!r}")
    sections = ("v7", "stability", "delegation", "trend", "observation",
                "duo_secondary", "kf", "adapt", "damage", "name", "variant")
    routed = {}
    for k, v in (override or {}).items():
        if k in sections:
            routed[k] = v
        else:
            routed.setdefault("v7", {})
            routed["v7"][k] = v
    return merge(cfg, routed)


def snapshot(cfgs, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfgs, indent=2, sort_keys=True, default=str))
    return out_path
