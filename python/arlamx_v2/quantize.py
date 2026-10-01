"""Post-training dynamic INT8 quantization of actor Linear layers (PPO/SAC/TD3)."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

_ALGOS = ("ppo", "sac", "td3")
_SRC_ATTR = "_arlamx_quantize_src"
_ALGO_TOKEN = re.compile(r"(?i)(?:^|[^a-z0-9])(ppo|sac|td3)(?:[^a-z0-9]|$)")


def _algo_classes():
    from stable_baselines3 import PPO, SAC, TD3

    return {"ppo": PPO, "sac": SAC, "td3": TD3}


def _algo_of_model(model):
    name = type(model).__name__.lower()
    return name if name in _ALGOS else None


def _algo_of_path(path):
    parts = Path(str(path).replace("\\", "/")).parts
    for part in reversed(parts):
        found = _ALGO_TOKEN.findall(part)
        if found:
            return found[-1].lower()
    return None


def _as_zip(path):
    p = Path(path)
    if p.is_dir():
        z = p / "model.zip"
        if z.is_file():
            return z
        zips = sorted(p.glob("*.zip"))
        if len(zips) == 1:
            return zips[0]
        raise FileNotFoundError(f"no model zip under {p}")
    if p.is_file():
        return p
    z = p if p.suffix == ".zip" else Path(str(p) + ".zip")
    if z.is_file():
        return z
    raise FileNotFoundError(p)


def _dest_zip(save_path):
    p = Path(save_path)
    return p if p.suffix == ".zip" else Path(str(p) + ".zip")


def _same_zip(a, b):
    pa, pb = Path(a).resolve(), Path(b).resolve()
    if pa == pb:
        return True
    try:
        return pa.is_file() and pb.is_file() and os.path.samefile(pa, pb)
    except OSError:
        return False


def _origin_zip(model, src_zip=None):
    if src_zip is not None:
        return Path(src_zip).resolve()
    tagged = getattr(model, _SRC_ATTR, None)
    if tagged:
        return Path(tagged).resolve()
    return None


def _load_model(path, algo=None):
    classes = _algo_classes()
    load_path = str(_as_zip(path))
    preferred = (algo or _algo_of_path(path) or "").lower() or None
    if preferred and preferred not in classes:
        raise ValueError(f"algo must be one of {_ALGOS}, got {preferred!r}")
    order = []
    if preferred:
        order.append(preferred)
    for name in _ALGOS:
        if name not in order:
            order.append(name)
    last = None
    for name in order:
        try:
            model = classes[name].load(load_path)
        except Exception as exc:
            last = exc
            continue
        setattr(model, _SRC_ATTR, str(Path(load_path).resolve()))
        return model, name
    raise ValueError(f"could not load PPO/SAC/TD3 from {load_path}") from last


# Dynamic INT8 quantization of Linear layers (Jacob et al. 2018, CVPR; torch.ao.quantization).
def _quantize_dynamic():
    import torch as th
    import torch.nn as nn

    ao = getattr(th, "ao", None)
    qmod = getattr(ao, "quantization", None) if ao is not None else None
    if qmod is None or not hasattr(qmod, "quantize_dynamic"):
        qmod = getattr(th, "quantization", None)
    if qmod is None or not hasattr(qmod, "quantize_dynamic"):
        raise RuntimeError("torch.ao.quantization.quantize_dynamic is unavailable")
    return qmod.quantize_dynamic, nn, th


def _q_linear(mod):
    quantize_dynamic, nn, th = _quantize_dynamic()
    mod = mod.cpu()
    if isinstance(mod, nn.Linear):
        mod = nn.Sequential(mod)
    return quantize_dynamic(mod, {nn.Linear}, dtype=th.qint8)


def _has_linear(mod, nn):
    return any(isinstance(m, nn.Linear) for m in mod.modules())


def _features(policy, actor):
    critic = getattr(policy, "critic", None)
    a_fe = getattr(actor, "features_extractor", None)
    if a_fe is None:
        return None
    if getattr(policy, "share_features_extractor", False):
        return a_fe
    c_fe = getattr(critic, "features_extractor", None) if critic is not None else None
    if c_fe is not None and c_fe is a_fe:
        return a_fe
    return None


def _quantize_actor(model, algo):
    _, nn, _ = _quantize_dynamic()
    if hasattr(model, "to"):
        model.to("cpu")
    else:
        model.policy.cpu()

    if algo in ("sac", "td3"):
        actor = getattr(model.policy, "actor", None)
        if actor is None:
            raise TypeError(f"{algo} policy has no actor")
        shared_fe = _features(model.policy, actor)
        quantized = False
        for name, child in list(actor.named_children()):
            if shared_fe is not None and (child is shared_fe or name == "features_extractor"):
                continue
            if not _has_linear(child, nn):
                continue
            setattr(actor, name, _q_linear(child))
            quantized = True
        if not quantized:
            raise TypeError(f"{algo} actor has no Linear modules to quantize")
        return model

    pol = model.policy
    mlp = getattr(pol, "mlp_extractor", None)
    policy_net = getattr(mlp, "policy_net", None) if mlp is not None else None
    action_net = getattr(pol, "action_net", None)
    if policy_net is None and action_net is None:
        raise TypeError("PPO policy has no actor Linear modules")
    if policy_net is not None:
        mlp.policy_net = _q_linear(policy_net)
    if action_net is not None:
        pol.action_net = _q_linear(action_net)
    return model


def quantize_int8(path_or_model, algo=None, save_path=None):
    src_zip = None
    if algo is not None:
        algo = str(algo).lower()
        if algo not in _ALGOS:
            raise ValueError(f"algo must be one of {_ALGOS}, got {algo!r}")
    if hasattr(path_or_model, "policy") and hasattr(path_or_model, "predict"):
        model = path_or_model
        algo = algo or _algo_of_model(model)
        if algo is None:
            raise ValueError("could not infer algo from loaded model; pass algo=")
    else:
        src_zip = _as_zip(path_or_model)
        if save_path is not None and _same_zip(_dest_zip(save_path), src_zip):
            raise ValueError(
                "refusing to overwrite source zip; pass a distinct save_path")
        model, inferred = _load_model(src_zip, algo)
        algo = algo or inferred

    _quantize_actor(model, algo)

    if save_path is not None:
        dest = _dest_zip(save_path)
        origin = _origin_zip(model, src_zip)
        if origin is not None and _same_zip(dest, origin):
            raise ValueError(
                "refusing to overwrite source zip; pass a distinct save_path")
        dest = dest.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(dest))
    return model


def main():
    ap = argparse.ArgumentParser(
        description="Dynamic INT8 quantize of actor Linear layers (eval-only).")
    ap.add_argument("--algo", choices=_ALGOS, default=None)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if _same_zip(_dest_zip(args.out), _as_zip(args.inp)):
        raise SystemExit("refusing to overwrite source zip; pass a distinct --out")
    quantize_int8(args.inp, algo=args.algo, save_path=args.out)


if __name__ == "__main__":
    main()
