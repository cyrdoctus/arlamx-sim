"""SC_v3_6mq / SC_v3_6mq_nGPS: 10 d or 300 km episodes, data buffer, altitude/data scheduled reward, optional no-GNSS nav; trainer 4xW."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium import spaces

from arlamx_v2 import cpp
from arlamx_v2.config import SETTINGS_DIR, load
from arlamx_v2.env import EPOCH_JD, ArlamxV2Env
from arlamx_v2.nav import Nav, true_crossing
from arlamx_v2.paths import OUTPUTS
from arlamx_v2.sc6mq import SC6mqV2, env_kw, flow_dcm
from arlamx_v2.session import _versions, read_yaml, write_yaml

CFG3 = SETTINGS_DIR / "sc_v3_6mq.yaml"
CFG3N = SETTINGS_DIR / "sc_v3_6mq_ngps.yaml"
WE = np.array([0.0, 0.0, cpp.OMEGA_EARTH])


def load3(path):
    c = read_yaml(Path(path))
    if "extends" in c:
        base = load3(Path(path).parent / c.pop("extends"))
        base.update(c)
        c = base
    return c


def sched(w, s, h_km, b):
    """Scheduled weights: u = lifetime urgency from altitude, b = buffer fill."""
    u = float(np.clip((450.0 - h_km) / 150.0, 0.0, 1.0))
    w = dict(w)
    for k in ("base", "ecl_drag"):
        w[k] = w[k] * (1.0 + s["alt"] * u)
    w["downlink"] = w["downlink"] * max(0.0, 1.0 + s["buf"] * (2.0 * b - 1.0)) * max(0.0, 1.0 - s["dl_alt"] * u)
    return w


class SC3(SC6mqV2):
    def __init__(self, env, cfg, seed=0, abs_quat=False):
        super().__init__(env, cfg, seed, abs_quat)
        self.w0 = dict(cfg["weights"])
        self.est = cfg.get("nav") == "est"
        self.nfore = 6 if cfg.get("foresight") else 0
        lo = np.r_[self.observation_space.low, -np.ones(3 + self.nfore)].astype(np.float32)
        hi = np.r_[self.observation_space.high, np.ones(3 + self.nfore)].astype(np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)
        e = env.unwrapped
        e._term_km = float(cfg["term_km"])
        if self.est:
            e._gps_W = 0.0
            e._tx_fn = self._tx
        self.nav, self._st0 = None, None
        self.nrng = np.random.default_rng(seed + 11)

    # onboard position / velocity
    def rv(self):
        if self.est:
            return self.nav.r.astype(float), self.nav.v.astype(float)
        st = self.env.unwrapped.sim.get_state()
        return np.asarray(st["r"], float), np.asarray(st["v"], float)

    def h_km(self):
        r, _ = self.rv()
        return (float(np.linalg.norm(r)) - cpp.RE_WGS) / 1e3

    def _q_bf_now(self):
        if not self.est:
            return super()._q_bf_now()
        st = self.env.unwrapped.sim.get_state()
        sun = np.asarray(st["sun_N"]) if self.cfg.get("cmd_frame") == "flowS" else None
        r, v = self.rv()
        return np.asarray(cpp.dcm_to_quat(np.asarray(st["C_BN"]) @ flow_dcm(r, v, sun).T))

    def _patch(self, o):
        """Replace the GNSS-derived observation slots by the onboard estimate (r, v, v_B, anomaly, station)."""
        e = self.env.unwrapped
        st = e.sim.get_state()
        r, v = self.rv()
        C = np.asarray(st["C_BN"])
        o[0:3], o[3:6] = r / 1e3 / 7000.0, v / 1e3 / 8.0
        vb = C @ v
        o[13:16] = vb / max(np.linalg.norm(vb), 1e-9)
        h = np.cross(r, v)
        ev = np.cross(v, h) / cpp.MU_WGS - r / np.linalg.norm(r)
        nu = 0.0
        if np.linalg.norm(ev) > 1e-12:
            nu = np.arccos(np.clip(ev @ r / (np.linalg.norm(ev) * np.linalg.norm(r)), -1, 1))
            nu = 2 * np.pi - nu if r @ v < 0 else nu
        o[16:18] = np.sin(nu), np.cos(nu)
        from arlamx_v2.advisors.stations import nearest_visible
        d, vis, _ = nearest_visible(r, cpp.gmst_rad(EPOCH_JD + float(st["t"]) / 86400.0))
        o[22:26] = np.r_[C @ d if vis > 0.5 else np.zeros(3), vis]
        return o

    def _aug(self, o):
        o, ecl = super()._aug(o)
        if self.est:
            o = self._patch(o)
        age = 0.0 if not self.est else min(1.0, (self.nav.t - self.nav.t_fix) / 21600.0)
        u = float(np.clip((450.0 - self.h_km()) / 150.0, 0.0, 1.0))
        o = np.r_[o, 2.0 * self.buf / self.cap - 1.0, u, age, self.fore() if self.nfore else []]
        return np.nan_to_num(o, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32), ecl

    def fore(self, n=8, dt=300.0):
        """Onboard look-ahead (same knowledge as the MPC horizon): next lit station pass time + flow-frame direction,
        next shadow transition time, sunlit share of the next n steps (two-body RK4, 300 s steps)."""
        from arlamx_v2.advisors.stations import nearest_visible
        from arlamx_v2.nav import lit, rk4
        st = self.env.unwrapped.sim.get_state()
        r, v = self.rv()
        sun = np.asarray(st["sun_N"], float)
        t = float(st["t"])
        lit0 = lit(r, sun) > 0
        tp, gF, te, nl = 1.0, np.zeros(3), 1.0, 0
        for k in range(1, n + 1):
            r, v = rk4(r, v, dt, 0.0)
            lk = lit(r, sun) > 0
            nl += lk
            if te == 1.0 and lk != lit0:
                te = k / n
            if tp == 1.0 and lk:
                d, vis, _ = nearest_visible(r, cpp.gmst_rad(EPOCH_JD + (t + k * dt) / 86400.0))
                if vis > 0.5:
                    tp, gF = k / n, flow_dcm(r, v, sun if self.cfg.get("cmd_frame") == "flowS" else None) @ d
        return np.r_[2 * tp - 1, gF, 2 * te - 1, 2 * nl / n - 1]

    def reset(self, **kw):
        o, info = self.env.reset(**kw)
        e = self.env.unwrapped
        e._max_steps = int(float(self.cfg["episode_days"]) * 86400.0 / e._advisor_s)
        d = self.cfg["data"]
        self.cap, self.buf = float(d["cap_min"]), float(d["b0"]) * float(d["cap_min"])
        self.lost = self.sent = 0.0
        if self.est:
            st = e.sim.get_state()
            self.nav = Nav(st["r"], st["v"], float(st["t"]), self.cfg["nav_est"], self.nrng)
            e.sim.set_nav(self.nav.r.astype(float), self.nav.v.astype(float))
        return self._aug(o)[0], info

    def _tx(self, out, c_bn, gmst):
        """Onboard pass decision (env hook, end of step): eclipse-timing fix, then station geometry from the estimate."""
        st0, n = self._st0, self.nav
        n.to(float(out["t"]))
        e0, e1 = float(st0["eclipse"]) > 0.5, float(out["eclipse"]) > 0.5
        self.fix_dt = None
        if e0 != e1:
            c = self.cfg["nav_est"]
            ill = abs(float(np.asarray(out["sun_B"])[2])) if e1 else abs(float(np.asarray(st0["C_BN"]) @ st0["sun_N"] @ [0, 0, 1]))
            tc = true_crossing(st0["r"], st0["v"], np.asarray(st0["sun_N"], float), float(out["t"]) - float(st0["t"]), entering=e0)
            if tc is not None and ill > c["lit_min"]:
                self.fix_dt = n.fix(float(st0["t"]) + tc + self.nrng.normal(0.0, c["t_sigma_s"]), e0, st0["sun_N"])
        return n.gs(c_bn, gmst)

    def step(self, action):
        e = self.env.unwrapped
        st = e.sim.get_state()
        self._st0 = {k: np.asarray(st[k], float) if np.ndim(st[k]) else float(st[k]) for k in ("r", "v", "t", "eclipse", "C_BN", "sun_N")}
        if self.est:
            e.sim.set_nav(self.nav.r.astype(float), self.nav.v.astype(float))
        self.w = sched(self.w0, self.cfg["schedule"], self.h_km(), self.buf / self.cap)
        o, r, term, trunc, info = super().step(action)
        dt_min = e._advisor_s / 60.0
        self.buf += float(self.cfg["data"]["gen_min_d"]) * dt_min / 1440.0
        sent = min(self.buf, dt_min) if info.get("dl_on") else 0.0
        self.buf -= sent
        self.sent += sent
        over = max(0.0, self.buf - self.cap)
        self.buf -= over
        self.lost += over
        p = info["sc6mq_parts"]
        cos_b = float(info.get("gs_point_cos", 0.0))
        if self.est:
            vis_b, cos_b = self.nav.gs(np.asarray(e.sim.get_state()["C_BN"]), cpp.gmst_rad(EPOCH_JD + float(e.sim.get_state()["t"]) / 86400.0))
        else:
            vis_b = float(info.get("gs_visible", 0.0))
        p["downlink"] = self.w["downlink"] * sent / dt_min + self.w["dl_shape"] * vis_b * float(np.clip((cos_b - 0.7) / 0.3, 0.0, 1.0))
        p["term"] = -self.w["term"] * float(term)
        p["alive"] = self.w["alive"]                # per-step value of staying above term_km (other terms are net negative)
        info.update(sent_min=sent, buf_frac=self.buf / self.cap, lost_min=over, h_onboard_km=self.h_km())
        if self.est:
            st1 = e.sim.get_state()
            dr = self.nav.r.astype(float) - np.asarray(st1["r"], float)
            vt = np.asarray(st1["v"], float) / np.linalg.norm(st1["v"])
            info.update(nav_err_m=float(np.linalg.norm(dr)), nav_along_m=float(dr @ vt), nav_fix_dt=self.fix_dt, nav_k=float(self.nav.k))
        return o, float(sum(p.values())), term, trunc, info


def make_env3(cfg, rank=0, seed=0, abs_quat=False):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(SC3(ArlamxV2Env(seed=seed + rank, **env_kw(cfg)), cfg, seed + rank, abs_quat))
    return _init


def train3(out_root, cfg, steps, tag, width=18, n_envs=44, seed=42, threads=4):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(threads)
    sid = f"{datetime.now():%m-%d-%H-%M-%S}_PPO_4x{width}_{tag}"
    run = Path(out_root) / "models" / sid
    run.mkdir(parents=True, exist_ok=True)
    ppo = dict(learning_rate=3e-4, n_steps=256, batch_size=2048, n_epochs=5, gamma=0.99,
               gae_lambda=0.95, clip_range=0.2, ent_coef=0.003)
    ppo.update(cfg.get("ppo") or {})
    snap = {"session": {"id": sid, "created": datetime.now().isoformat(timespec="seconds"), "status": "running",
                        "backend": "sc3", "n_envs": n_envs, "versions": _versions()},
            "network": {"algo": "ppo", "layers": 4, "width": width, "seed": seed, "ppo": ppo,
                        "vecnormalize": {"norm_obs": False, "norm_reward": True, "clip_reward": 10.0, "gamma": 0.99}},
            "settings": cfg, "env_kw": {k: v for k, v in env_kw(cfg).items() if k != "orbit"},
            "plant": {k: load(k) for k in ("physics", "reward_v10", cfg["vehicle"], cfg["sensors"],
                                           "gains_mrp", "power_mtq", "estimator_kf")}}
    snap_path = Path(out_root) / "snapshots" / f"{sid}.yaml"
    write_yaml(snap, snap_path)
    venv = VecNormalize(SubprocVecEnv([make_env3(cfg, i, seed) for i in range(n_envs)], start_method="forkserver"),
                        norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    model = PPO("MlpPolicy", venv, policy_kwargs={"net_arch": {"pi": [width] * 4, "vf": [width] * 4}},
                seed=seed, verbose=0, **ppo)
    log = open(run / "progress.csv", "w")
    log.write("steps,wall_s,ep_rew_mean,ep_len_mean\n")

    class Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0, self.last, self.ck = time.time(), 0.0, 0

        def _on_step(self):
            now = time.time()
            if now - self.last > 60:
                self.last = now
                buf = self.model.ep_info_buffer
                rm = np.mean([x["r"] for x in buf]) if buf else float("nan")
                lm = np.mean([x["l"] for x in buf]) if buf else float("nan")
                log.write(f"{self.num_timesteps},{now - self.t0:.0f},{rm:.4f},{lm:.1f}\n"); log.flush()
            if self.num_timesteps // 1_000_000 > self.ck:
                self.ck = self.num_timesteps // 1_000_000
                self.model.save(str(run / f"ckpt_{self.num_timesteps}"))
            return self.num_timesteps < steps

    t0 = time.time()
    model.learn(total_timesteps=10**10, callback=Cb())
    wall = time.time() - t0
    model.save(str(run / "model"))
    venv.save(str(run / "vecnormalize.pkl"))
    snap["session"].update(status="done", wall_s=round(wall, 1), steps=int(model.num_timesteps),
                           steps_per_s=model.num_timesteps / wall, model=str(run / "model.zip"))
    write_yaml(snap, snap_path)
    venv.close()
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CFG3)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--width", type=int, default=18)
    ap.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--out", type=Path, default=OUTPUTS / "SC_v3_6mq")
    ap.add_argument("--tag", default="SC_v3_6mq")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(train3(a.out, load3(a.config), a.steps, a.tag, a.width, a.n_envs, a.seed))
