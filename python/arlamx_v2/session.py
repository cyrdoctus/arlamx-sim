"""Training session: settings -> snapshot (outputs/snapshots) -> backend -> result."""
from __future__ import annotations

import copy
import json
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from arlamx_v2 import __version__
from arlamx_v2 import config as cfg_mod
from arlamx_v2 import physics as physics_mod
from arlamx_v2.config import SETTINGS_DIR
from arlamx_v2.paths import OUTPUTS, ROOT_V2

SECTIONS = ("network", "orbit", "train")
PLANT_NAMES = ("gains_mrp", "power_mtq", "estimator_kf", "sensors_solarcat", "vehicle")
V8PLUS = ("v8a", "v8b", "v9a", "v9b", "v9c", "v9d", "v10", "v11", "v10r6")
BACKENDS = ("train", "v12")


_SCI = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def coerce(obj):
    if isinstance(obj, dict):
        return {k: coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce(v) for v in obj]
    if isinstance(obj, str) and _SCI.match(obj.strip()):
        return float(obj)
    return obj


def read_yaml(path):
    return coerce(yaml.safe_load(Path(path).read_text()) or {})


def _plain(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def write_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_plain(obj), sort_keys=False, default_flow_style=None, width=100))
    return path


def load_settings(network=None, orbit=None, train=None, snapshot=None):
    if snapshot:
        snap = read_yaml(snapshot)
        s = {k: copy.deepcopy(snap.get(k) or {}) for k in SECTIONS}
        s["frozen"] = copy.deepcopy(snap.get("plant") or {})
        s["replay_of"] = str(snapshot)
        return s
    return {
        "network": read_yaml(network or SETTINGS_DIR / "network.yaml"),
        "orbit": read_yaml(orbit or SETTINGS_DIR / "orbit.yaml"),
        "train": read_yaml(train or SETTINGS_DIR / "train.yaml"),
    }


def set_path(settings, dotted, value):
    keys = dotted.split(".")
    if keys[0] not in SECTIONS:
        raise KeyError(f"{dotted!r}: first key must be one of {SECTIONS}")
    d = settings
    for k in keys[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def parse_set(item):
    if "=" not in item:
        raise ValueError(f"--set needs key=value, got {item!r}")
    k, v = item.split("=", 1)
    return k.strip(), coerce(yaml.safe_load(v))


def arch_of(net):
    return [int(net["width"])] * int(net["layers"])


def session_id(net, name=None, when=None, out_root=OUTPUTS):
    when = when or datetime.now()
    base = f"{when:%m-%d-%H-%M}_{str(net['algo']).upper()}_{int(net['layers'])}x{int(net['width'])}"
    if name:
        base += f"_{name}"
    sid, n = base, 1
    while (Path(out_root) / "snapshots" / f"{sid}.yaml").exists() or (Path(out_root) / "models" / sid).exists():
        n += 1
        sid = f"{base}_{n}"
    return sid


def variant_of(tr):
    return "v10" if tr.get("backend", "train") == "v12" else str(tr.get("variant", "v10"))


def resolve_plant(tr, frozen=None):
    if frozen:
        return copy.deepcopy(frozen)
    gsi = tr.get("gsi")
    plant = {"physics": physics_mod.load(tr.get("physics") or "standard",
                                         overrides={"aero": {"gsi": str(gsi).lower()}} if gsi else None)}
    variant = variant_of(tr)
    if variant in V8PLUS:
        plant["reward"] = cfg_mod.load_reward(variant, tr.get("reward_w") or None)
        for n in PLANT_NAMES:
            plant[n] = cfg_mod.load(n)
    else:
        plant["reward"] = f"hard-coded compose_sc_{variant[:2]} (python/arlamx_v2/reward.py)"
    return plant


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "torch", "stable_baselines3", "gymnasium"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def validate(settings):
    net, tr = settings["network"], settings["train"]
    algo = str(net.get("algo", "ppo")).lower()
    if algo not in ("ppo", "sac", "td3"):
        raise ValueError(f"network.algo must be ppo|sac|td3, got {algo!r}")
    backend = str(tr.get("backend", "train"))
    if backend not in BACKENDS:
        raise ValueError(f"train.backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "v12" and algo != "ppo":
        raise ValueError("backend v12 is PPO only")
    for k in ("layers", "width", "timesteps", "n_envs", "seed"):
        if k not in net:
            raise ValueError(f"network.{k} missing")


def run(settings, out_root=OUTPUTS, command=None):
    validate(settings)
    net = copy.deepcopy(settings["network"])
    orb = copy.deepcopy(settings["orbit"])
    tr = copy.deepcopy(settings["train"])
    net["algo"] = algo = str(net["algo"]).lower()
    backend = str(tr.get("backend", "train"))
    arch = arch_of(net)
    out_root = Path(out_root)
    sid = session_id(net, tr.get("name"), out_root=out_root)
    snap_path = out_root / "snapshots" / f"{sid}.yaml"
    model_dir = out_root / "models" / sid

    plant = resolve_plant(tr, settings.get("frozen"))
    frozen = {k: v for k, v in plant.items() if k != "physics" and isinstance(v, dict)}
    session = {
        "id": sid,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "arlamx_version": __version__,
        "backend": backend,
        "variant": variant_of(tr),
        "arch": arch,
        "model_dir": str(model_dir.relative_to(out_root.parent) if model_dir.is_relative_to(out_root.parent) else model_dir),
        "command": command or " ".join(sys.argv),
        "replay": f"python main.py train --from-snapshot {snap_path.relative_to(ROOT_V2) if snap_path.is_relative_to(ROOT_V2) else snap_path}",
        "replay_of": settings.get("replay_of"),
        "host": platform.node(),
        "versions": _versions(),
    }
    snap = {"session": session, "network": net, "orbit": orb, "train": tr, "plant": plant}
    write_yaml(snap, snap_path)
    print(f"[session] {sid}\n[session] snapshot {snap_path}\n[session] model    {model_dir}", flush=True)

    if int(net.get("torch_threads") or 0) > 0:
        os.environ["ARLAMX_TORCH_THREADS"] = str(int(net["torch_threads"]))
    cfg_mod.pin(frozen)
    t0 = time.time()
    metrics = None
    try:
        if backend == "train":
            from arlamx_v2.train import train_one
            env_kw = {"physics": plant["physics"], "orbit": orb.get("training"), "frozen": frozen}
            if tr.get("controller"):
                env_kw["controller"] = tr["controller"]
            if tr.get("reward_w"):
                env_kw["reward_w"] = tr["reward_w"]
            metrics = train_one(variant_of(tr), algo, int(net["timesteps"]), int(net["n_envs"]),
                                int(net["seed"]), sid, out_root / "models", env_kw=env_kw,
                                arch=arch, run_dir=model_dir, algo_kw=net.get(algo) or None,
                                vecnorm=net.get("vecnormalize") or None)
        else:
            from arlamx_v2 import train_v12 as tv
            kw = dict(tr.get("v12") or {})
            w_slew = kw.pop("w_slew", 0.5)
            metrics = tv.train_run(sid, w_slew, int(net["timesteps"]), int(net["n_envs"]),
                                   int(net["seed"]), reward_w=tr.get("reward_w") or None,
                                   physics=plant["physics"], controller=tr.get("controller"),
                                   run_dir=model_dir, arch=arch, algo_kw=net.get("ppo") or None,
                                   vecnorm=net.get("vecnormalize") or None,
                                   orbit=orb.get("training"), frozen=frozen,
                                   ppo_kw=kw.pop("ppo_kw", None) or None, **kw)
        session["status"] = "done"
    except KeyboardInterrupt:
        session["status"] = "interrupted"
        raise
    except Exception as exc:
        session["status"] = "failed"
        session["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cfg_mod.pin({})
        session["finished"] = datetime.now().isoformat(timespec="seconds")
        session["wall_s"] = round(time.time() - t0, 1)
        if metrics is not None:
            snap["result"] = {k: v for k, v in metrics.items() if k not in ("env_kw",)}
        write_yaml(snap, snap_path)
    return sid, snap_path, metrics


def list_snapshots(out_root=OUTPUTS):
    rows = []
    for p in sorted((Path(out_root) / "snapshots").glob("*.yaml")):
        try:
            s = read_yaml(p)
        except Exception:
            continue
        ses, net = s.get("session") or {}, s.get("network") or {}
        rows.append({"id": ses.get("id", p.stem), "status": ses.get("status"),
                     "backend": ses.get("backend"), "variant": ses.get("variant"),
                     "timesteps": net.get("timesteps"), "wall_s": ses.get("wall_s"),
                     "path": str(p)})
    return rows
