"""SC_v1/v2_6mq: SolarCat_v3 six-rod wrappers (v13 terms rebalanced, panel sun sensing, eclipse mode) and trainer."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arlamx_v2 import cpp
from arlamx_v2.env import ArlamxV2Env
from arlamx_v2.session import read_yaml, write_yaml, _versions
from arlamx_v2.config import SETTINGS_DIR, load
from arlamx_v2.paths import OUTPUTS

CFG = SETTINGS_DIR / "sc_v1_6mq.yaml"
CFG2 = SETTINGS_DIR / "sc_v2_6mq.yaml"


def flow_dcm(r, v, sun=None):
    """C_FN of the plant's flow frame: e1 = v_rel, e3 = orbit normal (flowS: Sun normal to the flow)."""
    we = np.array([0.0, 0.0, cpp.OMEGA_EARTH])
    e1 = v - np.cross(we, r); e1 = e1 / np.linalg.norm(e1)
    h = np.cross(r, v); ref = h
    if sun is not None:
        sp = np.asarray(sun, float) - (np.asarray(sun, float) @ e1) * e1
        if np.linalg.norm(sp) > 0.2:
            ref = sp if sp @ h >= 0 else -sp
    e3 = ref - (ref @ e1) * e1; e3 /= np.linalg.norm(e3)
    return np.vstack([e1, np.cross(e3, e1), e3])


def q_bf_from(ram_b, normal_b=(0.0, 0.0, 1.0)):
    b1 = np.asarray(ram_b, float); b1 /= np.linalg.norm(b1); b3 = np.asarray(normal_b, float)
    return np.asarray(cpp.dcm_to_quat(np.column_stack([b1, np.cross(b3, b1), b3])))


class SC6mq(gym.Wrapper):
    def __init__(self, env, cfg, seed=0):
        super().__init__(env)
        self.cfg = cfg
        self.w = cfg["weights"]
        self.rng = np.random.default_rng(seed + 7)
        lo = np.r_[env.observation_space.low, -np.ones(4)].astype(np.float32)
        hi = np.r_[env.observation_space.high, np.ones(4)].astype(np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

    def _sun(self):
        st = self.env.unwrapped.sim.get_state()
        lit = float(st["eclipse"]) > 0.5
        if not lit:
            return np.zeros(3), 1.0
        s = np.asarray(st["C_BN"]) @ np.asarray(st["sun_N"])
        s = s + self.rng.normal(0.0, float(self.cfg["sun_sigma"]), 3)
        return s / max(np.linalg.norm(s), 1e-9), 0.0

    def _aug(self, o):
        s, ecl = self._sun()
        return np.r_[o, s, ecl].astype(np.float32), ecl

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        return self._aug(o)[0], info

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        w, c = self.w, self.cfg
        e = self.env.unwrapped
        o, ecl = self._aug(o)
        track = float(np.degrees(info.get("tracking_err_rad", 0.0)))
        effort = float(info.get("torque_effort", 0.0))
        soc = float(info.get("battery_soc", 0.5))
        parts = {"base": w["base"] * float(r)}
        parts["slew"] = -w["slew"] * float(float(info.get("cmd_angle_deg", 0.0)) > c["slew_deg"])
        tw = np.asarray(info.get("tau_want", np.zeros(3)), float)
        te = np.asarray(info.get("tau_env_filt", np.zeros(3)), float)
        nt = float(np.linalg.norm(tw))
        parts["env"] = w["env"] * (float(np.clip(te @ (tw / nt) / 1.5e-8, -1, 1)) if nt > 1e-12 else 0.0)
        off = 1.0 - float(np.mean(e._rod_on))
        parts["rest"] = w["rest"] * off * float(track < 15.0)
        if ecl > 0.5:
            dEa, dEb = float(info.get("dE_actual", 0.0)), float(info.get("dE_baseline", 0.0))
            gap = (dEb - dEa) / abs(dEb) if abs(dEb) > 1e-12 else 0.0
            parts["ecl_drag"] = -w["ecl_drag"] * float(np.clip(gap, 0.0, 1.0))
            parts["ecl_mtq"] = -w["ecl_mtq"] * float(np.clip(effort * 10.0, 0.0, 1.0))
        vis, cos_p = float(info.get("gs_visible", 0.0)), float(info.get("gs_point_cos", 0.0))
        parts["downlink"] = w["downlink"] * vis * float(np.clip((cos_p - 0.7) / 0.3, 0.0, 1.0))
        parts["band"] = w["band"] * float(0.4 <= soc <= 0.6)
        thr = float(c["low_thr"])
        parts["low"] = -w["low"] * max(0.0, thr - soc) / thr
        parts["brownout"] = -w["brownout"] * float(bool(info.get("brownout_active", False)))
        info["sc6mq_parts"] = parts
        info["eclipse_est"] = ecl
        info["rods_off_frac"] = off
        return o, float(sum(parts.values())), term, trunc, info


class SC6mqV2(SC6mq):
    """v2: action quaternion = rotation from min drag (flow frame); eclipse -> min drag, all rods."""

    def __init__(self, env, cfg, seed=0, abs_quat=False):
        super().__init__(env, cfg, seed)
        self.abs_quat = abs_quat
        self.q_md = q_bf_from(cfg["ram_axis_body"])
        lo = np.r_[self.observation_space.low, -np.ones(4)].astype(np.float32)
        hi = np.r_[self.observation_space.high, np.ones(4)].astype(np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

    def _q_bf_now(self):
        st = self.env.unwrapped.sim.get_state()
        sun = np.asarray(st["sun_N"]) if self.cfg.get("cmd_frame") == "flowS" else None
        C = np.asarray(st["C_BN"]) @ flow_dcm(np.asarray(st["r"]), np.asarray(st["v"]), sun).T
        return np.asarray(cpp.dcm_to_quat(C))

    def _aug(self, o):
        o, ecl = super()._aug(o)
        return np.r_[o, self._q_bf_now()].astype(np.float32), ecl

    def step(self, action):
        a = np.asarray(action, np.float32).reshape(-1).copy()
        q = a[:4].astype(float)
        if not self.abs_quat:
            q = np.r_[1.0 + np.clip(q[0], -1, 1), np.clip(q[1:], -1, 1)]     # zero action = min drag
            a[4:] = a[4:] + 0.25                                              # zero action = rods on
        n = np.linalg.norm(q)
        q = q / n if np.isfinite(n) and n > 1e-6 else np.array([1.0, 0, 0, 0])
        if not self.abs_quat:
            q = np.asarray(cpp.quat_mul(q, self.q_md))
        sim = self.env.unwrapped.sim
        st = sim.get_state()
        forced = self.cfg.get("eclipse_mode") == "min_drag" and float(st["eclipse"]) < 0.5
        if forced:
            q = self.q_md
            a[4:] = 1.0
        if self.cfg.get("eclipse_frame"):
            sim.set_cmd_frame(self.cfg["eclipse_frame"] if forced else self.cfg["cmd_frame"])
        a[:4] = q
        o, r, term, trunc, info = super().step(a)
        err = float(np.degrees(info.get("tracking_err_rad", 0.0)))
        info["sc6mq_parts"]["att"] = -self.w["att"] * min(1.0, err / 30.0)
        if not forced:
            info["sc6mq_parts"]["sun"] = self.w.get("sun", 0.0) * float(self.env.unwrapped._illum_geo)
        info["forced_min_drag"] = forced
        info["q_bf_cmd"] = q
        return o, float(sum(info["sc6mq_parts"].values())), term, trunc, info


def env_kw(cfg):
    g = cfg["gains"]
    if "wn" in g:
        from arlamx_v2.vehicle import build
        I = float(np.mean(np.diag(build(load(cfg["vehicle"])).I)[:2]))
        kp, kd = 4 * I * g["wn"] ** 2, 2 * g["zeta"] * I * g["wn"]
    else:
        kp, kd = float(g["kp"]), float(g["kd"])
    return dict(variant=cfg["variant"], vehicle=cfg["vehicle"], sensors_cfg=cfg["sensors"],
                mtq_duty=float(cfg["mtq_duty"]), kp=kp, kd=kd, orbit=cfg["orbit"],
                cmd_frame=cfg.get("cmd_frame", "inertial"), advisor_s=cfg.get("advisor_s"), inner_dt=cfg.get("inner_dt"),
                physics=cfg.get("physics", "standard"))


def make_env(cfg, rank=0, seed=0, abs_quat=False):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = ArlamxV2Env(seed=seed + rank, **env_kw(cfg))
        w = SC6mqV2(env, cfg, seed + rank, abs_quat) if int(cfg.get("version", 1)) >= 2 else SC6mq(env, cfg, seed + rank)
        return Monitor(w)
    return _init


def train(minutes, n_envs, out_root, seed=42, cfg_path=CFG, steps=None, tag="SC_v1_6mq", cfg=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(4)
    cfg = cfg or read_yaml(cfg_path)
    sid = f"{datetime.now():%m-%d-%H-%M-%S}_PPO_4x18_{tag}"
    run = Path(out_root) / "models" / sid
    run.mkdir(parents=True, exist_ok=True)
    ppo = dict(learning_rate=3e-4, n_steps=256, batch_size=2048, n_epochs=5, gamma=0.99,
               gae_lambda=0.95, clip_range=0.2, ent_coef=0.003)
    ppo.update(cfg.get("ppo") or {})
    snap = {"session": {"id": sid, "created": datetime.now().isoformat(timespec="seconds"), "status": "running",
                        "backend": "sc6mq", "minutes": minutes, "n_envs": n_envs, "versions": _versions()},
            "network": {"algo": "ppo", "layers": 4, "width": 18, "seed": seed, "ppo": ppo,
                        "vecnormalize": {"norm_obs": False, "norm_reward": True, "clip_reward": 10.0, "gamma": 0.99}},
            "settings": cfg, "env_kw": {k: v for k, v in env_kw(cfg).items() if k != "orbit"},
            "plant": {k: load(k) for k in ("physics", "reward_v10", cfg["vehicle"], cfg["sensors"],
                                           "gains_mrp", "power_mtq", "estimator_kf")}}
    snap_path = Path(out_root) / "snapshots" / f"{sid}.yaml"
    write_yaml(snap, snap_path)
    venv = VecNormalize(SubprocVecEnv([make_env(cfg, i, seed) for i in range(n_envs)]),
                        norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    model = PPO("MlpPolicy", venv, policy_kwargs={"net_arch": {"pi": [18] * 4, "vf": [18] * 4}},
                seed=seed, verbose=0, tensorboard_log=str(run / "logs"), **ppo)
    t_end = time.time() + 60.0 * minutes
    log = open(run / "progress.csv", "w")
    log.write("steps,wall_s,ep_rew_mean,ep_len_mean\n")

    class Stop(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0, self.last = time.time(), 0.0

        def _on_step(self):
            now = time.time()
            if now - self.last > 60:
                self.last = now
                buf = self.model.ep_info_buffer
                rm = np.mean([x["r"] for x in buf]) if buf else float("nan")
                lm = np.mean([x["l"] for x in buf]) if buf else float("nan")
                log.write(f"{self.num_timesteps},{now - self.t0:.0f},{rm:.4f},{lm:.1f}\n"); log.flush()
                print(f"[6mq] {self.num_timesteps} steps  {now - self.t0:.0f}s  ep_rew {rm:.3f}", flush=True)
            if int(now - self.t0) // 900 > getattr(self, "_ck", 0):
                self._ck = int(now - self.t0) // 900
                self.model.save(str(run / f"ckpt_{self.num_timesteps}"))
                venv.save(str(run / f"ckpt_{self.num_timesteps}_vecnormalize.pkl"))
            return now < t_end and (steps is None or self.num_timesteps < steps)

    t0 = time.time()
    model.learn(total_timesteps=10**9, callback=Stop())
    wall = time.time() - t0
    model.save(str(run / "model"))
    venv.save(str(run / "vecnormalize.pkl"))
    snap["session"].update(status="done", wall_s=round(wall, 1), steps=int(model.num_timesteps),
                           steps_per_s=model.num_timesteps / wall, model=str(run / "model.zip"))
    write_yaml(snap, snap_path)
    venv.close()
    print(json.dumps(snap["session"], default=str), flush=True)
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--out", type=Path, default=OUTPUTS / "SC_v1_6mq")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tag", default="SC_v1_6mq")
    a = ap.parse_args()
    train(a.minutes, a.n_envs, a.out, a.seed, a.config, a.steps, a.tag)
