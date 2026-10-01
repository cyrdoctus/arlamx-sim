"""SB3 trainer (PPO/SAC/TD3): build model, train one run folder."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "python"))


def make_env(rank=0, seed=42, variant="v3", env_kw=None):
    from arlamx_v2.env import ArlamxV2Env
    from stable_baselines3.common.monitor import Monitor

    def _init():
        env = ArlamxV2Env(seed=seed + rank, variant=variant, **(env_kw or {}))
        return Monitor(env)

    return _init


def torch_threads(default=0):
    import torch

    n = int(os.environ.get("ARLAMX_TORCH_THREADS", str(default)))
    if n > 0:
        torch.set_num_threads(n)
    return torch.get_num_threads()


PPO_DEFAULTS = dict(learning_rate=3e-4, n_steps=128, batch_size=64, n_epochs=10,
                    gamma=0.99, ent_coef=0.003, clip_range=0.2, gae_lambda=0.95)
SAC_DEFAULTS = dict(learning_rate=3e-4, buffer_size=100_000, batch_size=256,
                    gamma=0.99, tau=0.005, verbose=1)
TD3_DEFAULTS = dict(learning_rate=3e-4, buffer_size=100_000, batch_size=256,
                    gamma=0.99, tau=0.005, verbose=0)
VECNORM_DEFAULTS = dict(norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)


def build_model(algo, env, arch, seed, log_dir, ppo_kw=None, algo_kw=None):
    from stable_baselines3 import PPO, SAC, TD3

    torch_threads()
    if algo == "ppo":
        kw = dict(PPO_DEFAULTS)
        kw.update(algo_kw or {})
        if ppo_kw:
            kw.update(ppo_kw)
        return PPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": {"pi": arch, "vf": arch}},
            seed=seed,
            verbose=0,
            tensorboard_log=str(log_dir),
            **kw,
        )
    if algo in ("sac", "td3"):
        kw = dict(SAC_DEFAULTS if algo == "sac" else TD3_DEFAULTS)
        kw.update(algo_kw or {})
        cls = SAC if algo == "sac" else TD3
        return cls(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": arch},
            seed=seed,
            tensorboard_log=str(log_dir),
            **kw,
        )
    raise ValueError(algo)


def train_one(variant, algo, timesteps, n_envs, seed, name, out_root, ppo_kw=None,
              env_kw=None, arch=None, run_dir=None, algo_kw=None, vecnorm=None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    out = Path(run_dir) if run_dir else Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    arch = list(arch) if arch else [16, 16, 16, 16]

    if variant.startswith(("v8", "v9")):
        from arlamx_v2.config import load as _load_cfg, load_reward as _load_rw, snapshot
        from arlamx_v2.physics import load as _load_phys
        phys = (env_kw or {}).get("physics", "standard")
        snapshot({
            "reward": _load_rw(variant, (env_kw or {}).get("reward_w")),
            "gains": _load_cfg("gains_mrp"),
            "power": _load_cfg("power_mtq"),
            "estimator": _load_cfg("estimator_kf"),
            "physics": _load_phys(phys, overrides=(
                {"aero": {"gsi": (env_kw or {}).get("gsi")}}
                if (env_kw or {}).get("gsi") else None)),
            "arch": arch, "algo": algo, "variant": variant,
            "timesteps": timesteps, "seed": seed, "n_envs": n_envs,
        }, out / "snapshot.json")
    ctor = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = ctor([make_env(i, seed, variant, env_kw) for i in range(n_envs)])
    vn = dict(VECNORM_DEFAULTS)
    vn.update(vecnorm or {})
    env = VecNormalize(env, **vn)
    model = build_model(algo, env, arch, seed, out / "logs", ppo_kw=ppo_kw, algo_kw=algo_kw)
    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    wall = time.time() - t0
    model.save(str(out / "models" / f"{algo}_{name}"))
    env.save(str(out / "models" / "vecnormalize.pkl"))
    metrics = {
        "name": name,
        "variant": variant,
        "algo": algo,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "arch": arch,
        "wall_s": wall,
        "steps_per_s": timesteps / max(wall, 1e-9),
        "ppo_kw": ppo_kw or {},
        "env_kw": {k: (list(v) if isinstance(v, (tuple,)) else v)
                   for k, v in (env_kw or {}).items()
                   if k not in ("reward_w", "frozen")},
        "reward_w": (env_kw or {}).get("reward_w") or {},
    }
    if algo_kw:
        metrics["algo_kw"] = algo_kw
    if vecnorm:
        metrics["vecnorm"] = vn
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    env.close()
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    import runpy
    sys.argv = [str(ROOT / "main.py"), "train", *sys.argv[1:]]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
