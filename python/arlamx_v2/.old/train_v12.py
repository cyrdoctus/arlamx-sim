"""SC_v12 PPO trainer. Does not edit the plant or env.py.

Uses env variant **v10** (existing observation, 7-D action, v10 weather).
NEW r_slew is a gymnasium Wrapper on cmd_angle_deg. Checkpoints every 50 k
with a 10-orbit eval. Writes only under outputs/v12/.

    PYTHONPATH=python python -m arlamx_v2.train_v12 --round r0
    PYTHONPATH=python python -m arlamx_v2.train_v12 --round long --w-slew 0.5 --steps 300000
    PYTHONPATH=python python -m arlamx_v2.train_v12 --round s3 --steps 800000

PPO only, 4x18, n_envs = cores/2. Level: advanced.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
from arlamx_v2.config import snapshot
from arlamx_v2.env import ArlamxV2Env, INERTIA_DIAG
from arlamx_v2.paths import OUTPUTS
from arlamx_v2.train import build_model

OUT = OUTPUTS / "v12"
SEEDS = (42, 43, 44)
ARCH = [18, 18, 18, 18]
VARIANT = "v10"
BURN = 40_000
CKPT_EVERY = 50_000
N_ORBITS_EVAL = 10.0

# Dual-column gates (§1). min-score vs each; brownout disqualifies.
REF_A = dict(decay_nominal=9.28, band_pct=86.6, downlink_min_d=23.2,
             slews_per_day=148.9, mtq_energy_J=2.97)   # published v1, truth
REF_D = dict(decay_nominal=9.19, band_pct=84.1, downlink_min_d=18.6,
             slews_per_day=148.9, mtq_energy_J=3.73)   # v2/v3 freeze, sensors
EVAL_OPT = dict(altitude_km=400.0, inc_deg=23.0, ecc=0.001,
                f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                nu_deg=0.0, mass_kg=0.625, soc=0.5,
                omega_dps=[1.0, 1.0, 0.5])
STORM_OPT = dict(altitude_km=350.0, inc_deg=23.0, ecc=0.001,
                 f107=150.0, ap=4.0, raan_deg=0.0, argp_deg=0.0,
                 nu_deg=0.0, mass_kg=0.625, soc=0.5,
                 omega_dps=[1.0, 1.0, 0.5])
STORM_JUMPS = ((0.30, 130.0, 76.0), (0.70, -130.0, -76.0))
STORM_DAYS = 1.0


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# 1 cm cp-cm offset from power_mtq.yaml. Used to turn tau_env into a force.
R_CP_M = np.array([0.01, 0.0, 0.0])
# KF noise floor: do not treat sub-nN*m residuals as a real fight.
TAU_ENV_LIVE = 1.0e-9


def _unit_q(q):
    q = np.asarray(q, float).reshape(-1)[:4]
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return -q if q[0] < 0 else q


def _tau_want_q(env, q):
    """Same MRP-PD demand the plant evaluates, on the pre-step state."""
    from arlamx_v2 import cpp
    st = env.sim.get_state()
    sig0 = np.asarray(st["sigma"], float)
    om0 = np.asarray(st["omega"], float)
    qn = _unit_q(q)
    if getattr(env, "_controller", "mrp") == "quaternion":
        qctrl = cpp.QuaternionFeedback()
        qctrl.kp = float(env._kp)
        qctrl.kd = float(env._kd)
        qctrl.set_inertia_diag(INERTIA_DIAG)
        q_body = np.asarray(cpp.mrp_to_quat(sig0), float)
        return np.asarray(qctrl.compute(q_body, om0, qn, np.zeros(3), 2.0), float)
    sigma_tgt = np.asarray(cpp.quat_to_mrp(qn), float)
    sig_err = cpp.mrp_error(sig0, sigma_tgt, True)
    return (-float(env._kp) * np.asarray(sig_err, float)
            - float(env._kd) * om0
            + np.cross(om0, INERTIA_DIAG * om0))


def _force_from_tau(tau, r_cp=R_CP_M):
    """Perpendicular force F with tau = r_cp × F. |F| = |tau| / |r_cp| if F ⊥ r."""
    tau = np.asarray(tau, float).reshape(3)
    r = np.asarray(r_cp, float).reshape(3)
    r2 = float(np.dot(r, r))
    if r2 < 1e-16:
        return np.zeros(3)
    return np.cross(tau, r) / r2


def tau_env_from_power(tau_ctrl, B, pcfg, tau_kf):
    """Power/geometry-aware environmental torque.

    The KF subtracts the plant's per-axis tau_ctrl. Coils only produce m×B, so
    the along-B part of tau_ctrl was never actuator work — fold it back into
    tau_env. Quadratic I²R on the achievable dipole is the measured coil cost.
    """
    from arlamx_v2.power import achievable_torque, mtq_power_w
    tau_ctrl = np.asarray(tau_ctrl, float).reshape(3)
    B = np.asarray(B, float).reshape(3)
    tau_kf = np.asarray(tau_kf, float).reshape(3)
    p_w = 0.0
    if float(np.linalg.norm(B)) < 1e-12 or pcfg is None:
        tau_pwr = tau_kf.copy()
    else:
        tau_ach, _ = achievable_torque(tau_ctrl, B, pcfg)
        tau_pwr = tau_kf + (tau_ctrl - tau_ach)
        p_w, _ = mtq_power_w(tau_ctrl, B, pcfg)
    return tau_pwr, _force_from_tau(tau_pwr), float(p_w)


def _corr_flat(a, b):
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if a.size < 8 or float(np.std(a)) < 1e-18 or float(np.std(b)) < 1e-18:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class V12Wrapper(gym.Wrapper):
    """Overlay only: r_slew, optional r_env, comparator, wrong-way gate clip.

    Does not change the C++ plant or env.py. Comparator: score σ_prev vs σ_new
    with SamplingMpcPolicy._score over 2×300 s; keep prev if new is not better
    by `margin` (plan §2.3).

    Wrong-way clip: if τ_env_i * τ_want_i < 0 on a delegated axis, force
    split s_i = 0 (action[4+i] = -1, full magnetorquer authority). The
    environment is fighting; gating into it is a mistake.

    SC_v14: `adapt_mrp=True` turns on the onboard RegimeEstimator. It schedules
    kd, gate_floor, soft wrong-way κ, and comparator margin from τ_env. w_plan
    / w_hold stay dark unless a later phase turns them on.
    """

    def __init__(self, env, w_slew=0.5, slew_deg=2.0, w_env=0.0,
                 use_cmp=False, margin=0.05, clip_wrong_way=False,
                 use_pwr_est=False, w_clip=0.0, w_plan=0.0, w_hold=0.0,
                 plan_horizon=6, plan_scale=8.0, adapt_mrp=False,
                 climate_mix=False, w_band_storm=0.0, w_low=0.0,
                 climate_wide=False, low_soc_frac=0.0, w_low_thr=0.25,
                 w_gen=0.0, w_gen_thr=0.20):
        super().__init__(env)
        self.w_slew = float(w_slew)
        self.slew_deg = float(slew_deg)
        self.w_env = float(w_env)
        self.use_cmp = bool(use_cmp)
        self.margin = float(margin)
        self.clip_wrong_way = bool(clip_wrong_way)
        self.use_pwr_est = bool(use_pwr_est)
        self.w_clip = float(w_clip)
        self.w_plan = float(w_plan)
        self.w_hold = float(w_hold)
        self.plan_horizon = int(plan_horizon)
        self.plan_scale = float(plan_scale)
        self.adapt_mrp = bool(adapt_mrp)
        self.climate_mix = bool(climate_mix)
        self.w_band_storm = float(w_band_storm)
        self.w_low = float(w_low)
        self.climate_wide = bool(climate_wide)
        self.low_soc_frac = float(low_soc_frac)
        self.w_low_thr = float(w_low_thr)
        self.w_gen = float(w_gen)
        self.w_gen_thr = float(w_gen_thr)
        self._climate_mode = "off"
        self._drift_f = 0.0
        self._drift_a = 0.0
        self._f_lo, self._f_hi = 5.0, 250.0
        self._a_lo, self._a_hi = 2.0, 200.0
        self._prev_a = None
        self._mpc = None
        self._planner = None
        self._tau_env_est = np.zeros(3)
        self._F_est = np.zeros(3)
        self.nstep = 0
        self.kept = 0
        self.n_clip_tot = 0
        self.n_wrong_tot = 0
        self._kp0 = float(getattr(env, "_kp", 6.0e-5))
        self._kd0 = float(getattr(env, "_kd", 1.6e-3))
        self._gf0 = float(getattr(env, "_gate_floor", 0.3))
        self._margin_eff = float(self.margin)
        self._clip_kappa = 1.0
        self._regime = None
        if self.adapt_mrp:
            from arlamx_v2.regime import RegimeEstimator
            self._regime = RegimeEstimator()

    def reset(self, **kwargs):
        self._prev_a = None
        self._tau_env_est = np.zeros(3)
        self._F_est = np.zeros(3)
        self.nstep = 0
        self.kept = 0
        self.n_clip_tot = 0
        self.n_wrong_tot = 0
        self._margin_eff = float(self.margin)
        self._clip_kappa = 1.0 if self._regime is None else 0.70
        if self._regime is not None:
            self._regime.reset()
        out = self.env.reset(**kwargs)
        self.env._kp = self._kp0
        self.env._kd = self._kd0 if self._regime is None else 1.0e-3
        self.env._gate_floor = self._gf0 if self._regime is None else 0.30
        opt = kwargs.get("options") or {}
        if self.climate_mix and "f107" not in opt:
            self._draw_climate()
            rng = getattr(self.env, "_rng", None)
            p_low = float(self.low_soc_frac)
            if self._climate_mode == "storm":
                p_low = min(0.55, p_low + 0.15)
            if p_low > 0 and rng is not None and float(rng.random()) < p_low:
                cap = float(getattr(self.env, "_batt_cap_J", 0.0) or 0.0)
                if cap > 0:
                    # Under-band start (supercap-critical side). Not a fixed 32 %.
                    self.env._batt_E = float(rng.uniform(0.16, 0.34)) * cap
        else:
            self._climate_mode = "off"
            self._drift_f = 0.0
            self._drift_a = 0.0
        return out

    def _draw_climate(self):
        """Per-episode weather mix. Does not edit env.py.

        24 parallel envs are not 24 storms per orbit: each reset independently
        draws quiet (~40 %), gradual+few spikes (~35 %), or storm (~25 %).
        Storm swings are *wider* than v10; most envs never see that.
        """
        rng = getattr(self.env, "_rng", np.random.default_rng())
        u = float(rng.random())
        T = float(getattr(self.env, "_max_steps", 2000.0))
        wide = bool(self.climate_wide)
        q_cut, g_cut = (0.30, 0.60) if wide else (0.40, 0.75)
        if u < q_cut:
            mode = "quiet"
            self.env._f107 = float(rng.uniform(60.0, 160.0 if wide else 140.0))
            self.env._ap = float(rng.uniform(2.0, 20.0 if wide else 12.0))
            self._drift_f = float(rng.normal(0.0, 0.35 if wide else 0.20))
            self._drift_a = float(rng.normal(0.0, 0.06 if wide else 0.03))
            n_j = int(rng.integers(0, 3 if wide else 2))
            jumps = []
            for _ in range(n_j):
                jumps.append((
                    float(rng.uniform(0.15, 0.90) * T),
                    float(rng.uniform(-30.0 if wide else -20.0, 30.0 if wide else 20.0)),
                    float(rng.uniform(-6.0 if wide else -3.0, 10.0 if wide else 6.0)),
                ))
            self.env._weather_jumps = jumps
            self.env._srp_flashes = []
            self._f_lo, self._f_hi = 55.0, 180.0 if wide else 160.0
            self._a_lo, self._a_hi = 2.0, 25.0 if wide else 18.0
        elif u < g_cut:
            mode = "gradual"
            self.env._f107 = float(rng.uniform(70.0, 210.0 if wide else 180.0))
            self.env._ap = float(rng.uniform(4.0, 55.0 if wide else 30.0))
            self._drift_f = float(rng.normal(0.0, 0.70 if wide else 0.45))
            self._drift_a = float(rng.normal(0.0, 0.14 if wide else 0.08))
            n_j = int(rng.integers(1, 4 if wide else 3))
            jumps = []
            for _ in range(n_j):
                jumps.append((
                    float(rng.uniform(0.10, 0.90) * T),
                    float(rng.uniform(-80.0 if wide else -60.0, 100.0 if wide else 80.0)),
                    float(rng.uniform(-20.0 if wide else -15.0, 55.0 if wide else 40.0)),
                ))
            self.env._weather_jumps = jumps
            flashes = list(getattr(self.env, "_srp_flashes", []) or [])
            self.env._srp_flashes = flashes[:1]
            self._f_lo, self._f_hi = 65.0, 230.0
            self._a_lo, self._a_hi = 2.0, 80.0 if wide else 60.0
        else:
            mode = "storm"
            self.env._f107 = float(rng.uniform(80.0 if wide else 100.0, 250.0))
            self.env._ap = float(rng.uniform(12.0 if wide else 20.0, 200.0 if wide else 180.0))
            self._drift_f = float(rng.normal(0.0, 1.1 if wide else 0.80))
            self._drift_a = float(rng.normal(0.0, 0.30 if wide else 0.20))
            n_j = int(rng.integers(4 if wide else 3, 11 if wide else 9))
            jumps = []
            for _ in range(n_j):
                jumps.append((
                    float(rng.uniform(0.05, 0.95) * T),
                    float(rng.uniform(-200.0 if wide else -180.0, 200.0 if wide else 180.0)),
                    float(rng.uniform(-60.0 if wide else -50.0, 180.0 if wide else 160.0)),
                ))
            self.env._weather_jumps = jumps
            self._f_lo, self._f_hi = 5.0, 250.0
            self._a_lo, self._a_hi = 2.0, 200.0
        self._climate_mode = mode

    def _tick_climate(self):
        if self._climate_mode in ("off", None):
            return
        e = self.env
        e._f107 = float(np.clip(float(e._f107) + self._drift_f, self._f_lo, self._f_hi))
        e._ap = float(np.clip(float(e._ap) + self._drift_a, self._a_lo, self._a_hi))

    def _ensure_mpc(self):
        if self._mpc is not None:
            return
        from arlamx_v2.advisors.mpc import SamplingMpcPolicy
        e = self.env
        self._mpc = SamplingMpcPolicy(
            horizon_steps=2, n_random=0, seed=0,
            panels=(e._n, e._A, e._c))
        self._mpc.advisor_step_s = e._advisor_s

    def _ensure_planner(self):
        if self._planner is not None:
            return
        from arlamx_v2.advisors.mpc import SamplingMpcPolicy
        e = self.env
        self._planner = SamplingMpcPolicy(
            horizon_steps=max(2, self.plan_horizon), n_random=0, seed=0,
            panels=(e._n, e._A, e._c))
        self._planner.advisor_step_s = e._advisor_s

    def _score_with(self, mpc, q):
        from arlamx_v2.bench_v8 import _advisor_state
        st = _advisor_state(self.env, noisy=False)
        sun = st.get("sun_N")
        if sun is not None:
            sun = np.asarray(sun, float)
            n = float(np.linalg.norm(sun))
            sun = sun / n if n > 1e-12 else None
        s, _ = mpc._score(
            np.asarray(q, float),
            np.asarray(st["r"], float), np.asarray(st["v"], float),
            sun, float(st["battery_soc"]), st.get("rho"),
            float(st.get("T", 900.0)), float(st.get("m_bar", 2.656e-26)),
            st.get("gs_dir_N"), float(st.get("gs_visible", 0.0)))
        return float(s)

    def _score_q(self, q):
        return self._score_with(self._mpc, q)

    def _score_plan(self, q):
        self._ensure_planner()
        return self._score_with(self._planner, q)

    def _apply_regime(self):
        """Set this tick's kd / gate_floor / clip / comparator margin."""
        if self._regime is None:
            self._margin_eff = float(self.margin)
            self._clip_kappa = 1.0
            return
        sch = self._regime.peek()
        self.env._kp = float(sch["kp"])
        self.env._kd = float(sch["kd"])
        self.env._gate_floor = float(sch["gate_floor"])
        self._margin_eff = float(sch["margin"])
        self._clip_kappa = float(sch["clip_kappa"])

    def _adapt_mrp_gains(self):
        """Back-compat name: regime apply. Kept so older call sites still work."""
        self._apply_regime()

    def _clip_wrong_way(self, a):
        te = np.asarray(self._tau_env_est, float).reshape(3)
        if float(np.linalg.norm(te)) < 1e-12:
            te = np.asarray(getattr(self.env, "_tau_env_filt", np.zeros(3)), float)
        tw = _tau_want_q(self.env, a[:4])
        n_wrong = 0
        n_clip = 0
        out = a.copy()
        kappa = float(self._clip_kappa)
        for i in range(3):
            split_i = 0.5 * (float(out[4 + i]) + 1.0)
            oppose = (te[i] * tw[i] < 0.0
                      and abs(float(te[i])) > TAU_ENV_LIVE
                      and abs(float(tw[i])) > 1e-11)
            if oppose and split_i > 0.05:
                n_wrong += 1
                s_new = split_i * (1.0 - kappa)
                out[4 + i] = np.float32(2.0 * s_new - 1.0)
                n_clip += 1
        return n_wrong, n_clip, out

    def _update_tau_env(self, info):
        tkf = np.asarray(info.get("tau_env_filt", np.zeros(3)), float)
        if not self.use_pwr_est:
            self._tau_env_est = tkf
            self._F_est = _force_from_tau(tkf)
            info["tau_env_pwr"] = tkf.copy()
            info["F_est"] = self._F_est.copy()
            info["mtq_power_quad_W"] = float(info.get("mtq_power_W", 0.0))
            return
        tc = np.asarray(info.get("tau_ctrl_mean", np.zeros(3)), float)
        bb = np.asarray(info.get("B_B", getattr(self.env, "_B_B", np.zeros(3))), float)
        pcfg = getattr(self.env, "_pcfg", None)
        tau_pwr, F, p_w = tau_env_from_power(tc, bb, pcfg, tkf)
        self._tau_env_est = 0.7 * tkf + 0.3 * tau_pwr
        self._F_est = F
        info["tau_env_pwr"] = tau_pwr
        info["F_est"] = F
        info["mtq_power_quad_W"] = p_w

    def step(self, action):
        a = np.asarray(action, np.float32).reshape(-1).copy()
        self._tick_climate()
        self._apply_regime()
        kept = 0
        t_cmp0 = time.perf_counter()
        if self.use_cmp and self._prev_a is not None:
            self._ensure_mpc()
            try:
                sn = self._score_q(_unit_q(a))
                sp = self._score_q(_unit_q(self._prev_a))
                if sn <= sp + float(self._margin_eff):
                    a = self._prev_a.copy()
                    kept = 1
            except Exception:
                kept = 0
        cmp_ms = (time.perf_counter() - t_cmp0) * 1e3

        n_wrong, n_clip = 0, 0
        t_clip0 = time.perf_counter()
        if self.clip_wrong_way:
            n_wrong, n_clip, a = self._clip_wrong_way(a)
        clip_ms = (time.perf_counter() - t_clip0) * 1e3

        r_plan = 0.0
        if self.w_plan:
            try:
                raw = self._score_plan(_unit_q(a))
                r_plan = self.w_plan * float(np.clip(raw / max(self.plan_scale, 1e-6), -1.0, 1.0))
            except Exception:
                r_plan = 0.0

        soc_pre = 0.5
        cap = float(getattr(self.env, "_batt_cap_J", 0.0) or 0.0)
        if cap > 0:
            soc_pre = float(np.clip(getattr(self.env, "_batt_E", 0.5 * cap) / cap, 0.0, 1.0))

        self.nstep += 1
        self.kept += kept
        self.n_clip_tot += n_clip
        self.n_wrong_tot += n_wrong
        obs, r, term, trunc, info = self.env.step(a)
        self._update_tau_env(info)
        if self._regime is not None:
            td = np.zeros(3)
            kf = getattr(self.env, "kf", None)
            if kf is not None:
                td = np.asarray(kf.tau_dot, float)
            bb = np.asarray(info.get("B_B", getattr(self.env, "_B_B", np.zeros(3))),
                            float)
            self._regime.update(self._tau_env_est, td, bb)
            info.update(self._regime.as_info())
        slew = 1.0 if float(info.get("cmd_angle_deg", 0.0)) > self.slew_deg else 0.0
        extra = -self.w_slew * slew
        extra -= self.w_clip * float(n_clip)
        r_hold = 0.0
        if self.w_hold and slew > 0.5 and 0.4 <= soc_pre <= 0.6:
            r_hold = -self.w_hold
            extra += r_hold
        extra += r_plan
        r_env = 0.0
        if self.w_env:
            tw = np.asarray(info.get("tau_want", np.zeros(3)), float)
            te = np.asarray(self._tau_env_est, float)
            nt = float(np.linalg.norm(tw))
            if nt > 1e-12:
                r_env = self.w_env * float(np.clip(
                    np.dot(te, tw / nt) / 1.5e-8, -1.0, 1.0))
        extra += r_env
        soc = float(info.get("battery_soc", soc_pre))
        eta_eff = float(info.get("eta_eff", 0.0))
        r_band = 0.0
        if self.w_band_storm and eta_eff > 0.05:
            if 0.4 <= soc <= 0.6:
                r_band = self.w_band_storm * eta_eff
            elif soc < 0.4:
                r_band = -self.w_band_storm * eta_eff * min(1.0, (0.4 - soc) / 0.4)
        extra += r_band
        r_low = 0.0
        thr = float(getattr(self, "w_low_thr", 0.25) or 0.25)
        if self.w_low and soc < thr:
            r_low = -self.w_low * (thr - soc) / max(thr, 1e-6)
            extra += r_low
        r_gen = 0.0
        gthr = float(getattr(self, "w_gen_thr", 0.20) or 0.20)
        if self.w_gen and soc < gthr:
            # Charge when the tank is low — opposite of w_low (do not idle the
            # coils; point for generation). power_gen_norm is 0 in eclipse.
            gen = float(info.get("power_gen_norm", 0.0))
            r_gen = self.w_gen * max(0.0, gen) * (gthr - soc) / max(gthr, 1e-6)
            extra += r_gen
        info["r_slew"] = -self.w_slew * slew
        info["r_env"] = r_env
        info["r_plan"] = r_plan
        info["r_hold"] = r_hold
        info["r_band"] = r_band
        info["r_low"] = r_low
        info["r_gen"] = r_gen
        info["slew_flag"] = slew
        info["kept_prev"] = kept
        info["kept_prev_frac"] = self.kept / max(self.nstep, 1)
        info["n_clip"] = n_clip
        info["wrong_way"] = n_wrong
        info["clip_frac"] = self.n_clip_tot / max(self.nstep, 1)
        info["wrong_way_frac"] = self.n_wrong_tot / max(self.nstep, 1)
        info["cmp_ms"] = float(cmp_ms)
        info["clip_ms"] = float(clip_ms)
        info["tau_env_est"] = np.asarray(self._tau_env_est, float).copy()
        info["climate_mode"] = self._climate_mode
        parts = dict(info.get("reward_parts") or {})
        parts["r_slew"] = -self.w_slew * slew
        parts["r_env"] = r_env
        parts["r_clip"] = -self.w_clip * float(n_clip)
        parts["r_plan"] = r_plan
        parts["r_hold"] = r_hold
        parts["r_band"] = r_band
        parts["r_low"] = r_low
        parts["r_gen"] = r_gen
        info["reward_parts"] = parts
        self._prev_a = a.copy()
        return obs, float(r) + extra, term, trunc, info


def min_score(agg, ref):
    if float(agg.get("brownouts", 0)) > 0.5:
        return None
    dec = float(agg.get("decay_km_d", agg.get("decay_nominal", 9.0)))
    ratios = [
        (1.1 * ref["decay_nominal"]) / max(dec, 0.05),
        float(agg["band_pct"]) / max(0.9 * ref["band_pct"], 1.0),
        float(agg["downlink_min_d"]) / max(ref["downlink_min_d"], 0.1),
        (0.5 * ref["slews_per_day"]) / max(float(agg.get("slews_per_day", 1.0)), 0.05),
        ref["mtq_energy_J"] / max(float(agg.get("mtq_energy_J", 1.0)), 0.01),
    ]
    return float(min(ratios))


def _eval_wrapped(predict, w_slew, seed=1001, use_cmp=False, w_env=0.0,
                  clip_wrong_way=False, use_pwr_est=False, w_clip=0.0,
                  w_plan=0.0, w_hold=0.0, adapt_mrp=False, storm=False,
                  w_band_storm=0.0, w_low=0.0,
                  physics="standard", controller=None, gsi=None):
    from arlamx_v2.power import mtq_power_w
    base = ArlamxV2Env(seed=0, variant=VARIANT,
                       physics=physics, controller=controller, gsi=gsi)
    env = V12Wrapper(base, w_slew=w_slew, w_env=w_env, use_cmp=use_cmp,
                     clip_wrong_way=clip_wrong_way, use_pwr_est=use_pwr_est,
                     w_clip=w_clip, w_plan=w_plan, w_hold=w_hold,
                     adapt_mrp=adapt_mrp, w_band_storm=w_band_storm, w_low=w_low)
    if storm:
        opt = dict(STORM_OPT)
        env.env._max_steps = int(STORM_DAYS * 86400.0 / env.env._advisor_s)
        obs, _ = env.reset(seed=seed, options=opt)
        env.env._weather_jumps = [
            (f * env.env._max_steps, df, da) for f, df, da in STORM_JUMPS]
        env.env._srp_flashes = []
    else:
        alt = float(EVAL_OPT["altitude_km"])
        period = 2 * np.pi * np.sqrt((6371e3 + alt * 1e3) ** 3 / 3.986004418e14)
        env.env._max_steps = int(N_ORBITS_EVAL * period / env.env._advisor_s)
        obs, _ = env.reset(seed=seed, options=dict(EVAL_OPT))
        env.env._weather_jumps = []
        env.env._srp_flashes = []
    H = {k: [] for k in ("sma", "soc", "dep", "dl", "mtqW", "slew", "p_pd",
                         "deleg_P", "deleg_B", "kept", "idle", "clip", "wrong",
                         "cmp_ms", "p_quad", "F", "aero", "kf", "pwr",
                         "eta", "eta_slow", "shock", "kd_eff", "gate_floor",
                         "margin_eff")}
    infer_s = 0.0
    n_inf = 0
    done = False
    pcfg = getattr(env.env, "_pcfg", None)
    while not done:
        t0 = time.perf_counter()
        a = predict(obs, env.env)
        infer_s += time.perf_counter() - t0
        n_inf += 1
        obs, _r, term, trunc, info = env.step(np.asarray(a, np.float32))
        done = term or trunc
        H["sma"].append(float(info["sma_m"]))
        H["soc"].append(float(info["battery_soc"]))
        H["dep"].append(1.0 if info.get("battery_depleted") else 0.0)
        vis = float(info.get("gs_visible", 0)) > 0.5
        lit = float(info.get("eclipse", 1)) > 0.5
        H["dl"].append(1.0 if (vis and lit and float(info.get("gs_point_cos", 0)) > 0.7) else 0.0)
        H["mtqW"].append(float(info.get("mtq_power_W", 0.0)))
        H["slew"].append(1.0 if float(info.get("cmd_angle_deg", 0.0)) > 2.0 else 0.0)
        H["deleg_P"].append(float(info.get("deleg_P", 0.0)))
        H["deleg_B"].append(float(info.get("deleg_B", 0.0)))
        H["kept"].append(float(info.get("kept_prev", 0.0)))
        H["idle"].append(1.0 if float(info.get("mtq_power_W", 1.0)) < 1e-6 else 0.0)
        H["clip"].append(float(info.get("n_clip", 0.0)))
        H["wrong"].append(float(info.get("wrong_way", 0.0)))
        H["cmp_ms"].append(float(info.get("cmp_ms", 0.0)))
        H["p_quad"].append(float(info.get("mtq_power_quad_W", 0.0)))
        H["F"].append(float(np.linalg.norm(info.get("F_est", np.zeros(3)))))
        H["aero"].append(np.asarray(info.get("tau_aero_mean", np.zeros(3)), float))
        H["kf"].append(np.asarray(info.get("tau_env_filt", np.zeros(3)), float))
        H["pwr"].append(np.asarray(info.get("tau_env_pwr", np.zeros(3)), float))
        H["eta"].append(float(info.get("eta", 0.0)))
        H["eta_slow"].append(float(info.get("eta_slow", 0.0)))
        H["shock"].append(float(info.get("shock", 0.0)))
        H["kd_eff"].append(float(info.get("kd_eff", getattr(env.env, "_kd", 0.0))))
        H["gate_floor"].append(float(info.get("gate_floor_eff",
                                             getattr(env.env, "_gate_floor", 0.3))))
        H["margin_eff"].append(float(info.get("margin_eff", 0.05)))
        tw = np.asarray(info.get("tau_want", np.zeros(3)), float)
        bb = np.asarray(info.get("B_B", getattr(env.env, "_B_B", np.zeros(3))), float)
        if pcfg is not None and float(np.linalg.norm(bb)) > 1e-12:
            p_pd, _ = mtq_power_w(tw, bb, pcfg)
        else:
            p_pd = float(info.get("mtq_power_W", 0.0))
        H["p_pd"].append(float(p_pd))
    n = max(1, len(H["soc"]))
    days = n * env.env._advisor_s / 86400.0
    sma = np.asarray(H["sma"])
    dep = np.asarray(H["dep"]) > 0.5
    soc = np.asarray(H["soc"])
    e_act = float(np.sum(H["mtqW"]) * env.env._advisor_s)
    e_pd = float(np.sum(H["p_pd"]) * env.env._advisor_s)
    e_quad = float(np.sum(H["p_quad"]) * env.env._advisor_s)
    decay = float((sma[0] - sma[-1]) / 1e3 / max(days, 1e-9))
    infer_pol = float(infer_s / max(n_inf, 1) * 1e3)
    infer_cmp = float(np.mean(H["cmp_ms"])) if H["cmp_ms"] else 0.0
    env.close()
    return {
        "decay_km_d": decay,
        "band_pct": float(np.mean((soc >= 0.4) & (soc <= 0.6)) * 100.0),
        "brownouts": int(np.sum(np.diff(dep.astype(int)) > 0) + (1 if dep[:1].any() else 0)),
        "downlink_min_d": float(np.sum(H["dl"]) * env.env._advisor_s / 60.0 / max(days, 1e-9)),
        "mtq_energy_J": e_act,
        "mtq_energy_quad_J": e_quad,
        "slews_per_day": float(np.sum(H["slew"]) / max(days, 1e-9)),
        "deleg_P": float(np.mean(H["deleg_P"])),
        "deleg_B": float(np.mean(H["deleg_B"])),
        "coil_idle_frac": float(np.mean(H["idle"])),
        "kept_prev_frac": float(np.mean(H["kept"])),
        "clip_frac": float(np.mean(np.asarray(H["clip"]) > 0.5)),
        "wrong_way_frac": float(np.mean(np.asarray(H["wrong"]) > 0.5)),
        "pei_act_over_pd": float(e_act / max(e_pd, 1e-18)),
        "pei_saved": float(1.0 - e_act / max(e_pd, 1e-18)),
        "kf_aero_r": _corr_flat(H["kf"], H["aero"]),
        "pwr_aero_r": _corr_flat(H["pwr"], H["aero"]),
        "F_est_uN": float(np.mean(H["F"]) * 1e6),
        "infer_ws_ms": infer_pol,
        "infer_cmp_ms": infer_cmp,
        "infer_total_ms": infer_pol + infer_cmp,
        "lifetime_d_500_300": float(200.0 / max(decay, 0.05)),
        "soc_lo_pct": float(np.mean(soc < 0.4) * 100.0),
        "soc_hi_pct": float(np.mean(soc > 0.6) * 100.0),
        "days": float(days), "steps": n,
        "eta_mean": float(np.mean(H["eta"])) if H["eta"] else 0.0,
        "eta_slow_mean": float(np.mean(H["eta_slow"])) if H["eta_slow"] else 0.0,
        "shock_frac": float(np.mean(np.asarray(H["shock"]) > 0.5)) if H["shock"] else 0.0,
        "kd_eff_mean": float(np.mean(H["kd_eff"])) if H["kd_eff"] else 0.0,
        "gate_floor_mean": float(np.mean(H["gate_floor"])) if H["gate_floor"] else 0.0,
        "margin_eff_mean": float(np.mean(H["margin_eff"])) if H["margin_eff"] else 0.0,
    }


def make_env(rank, seed, w_slew, reward_w=None, use_cmp=False, w_env=0.0,
             clip_wrong_way=False, use_pwr_est=False, w_clip=0.0,
             w_plan=0.0, w_hold=0.0, adapt_mrp=False, climate_mix=False,
             w_band_storm=0.0, w_low=0.0, climate_wide=False,
             low_soc_frac=0.0, w_low_thr=0.25, w_gen=0.0, w_gen_thr=0.20,
             physics="standard", controller=None, gsi=None):
    from stable_baselines3.common.monitor import Monitor

    def _init():
        env = ArlamxV2Env(seed=seed + rank, variant=VARIANT, reward_w=reward_w,
                          physics=physics, controller=controller, gsi=gsi)
        env = V12Wrapper(env, w_slew=w_slew, w_env=w_env, use_cmp=use_cmp,
                         clip_wrong_way=clip_wrong_way, use_pwr_est=use_pwr_est,
                         w_clip=w_clip, w_plan=w_plan, w_hold=w_hold,
                         adapt_mrp=adapt_mrp, climate_mix=climate_mix,
                         w_band_storm=w_band_storm, w_low=w_low,
                         climate_wide=climate_wide, low_soc_frac=low_soc_frac,
                         w_low_thr=w_low_thr, w_gen=w_gen, w_gen_thr=w_gen_thr)
        return Monitor(env)
    return _init


def infer_costs():
    """U575 analytic + one workstation sample. MPC vs v12 4x18."""
    from arlamx_v2.inference_budget import (mpc_u575_ms, policy_entry, mpc_cost)
    env = ArlamxV2Env(seed=0, variant=VARIANT)
    obs_dim = int(env.observation_space.shape[0])
    env.close()
    pol = policy_entry("v12 PPO 4x18 v10-obs", obs_dim, ARCH, 7)
    cmp = mpc_cost(horizon_steps=2, n_random=0, n_named=1)
    cmp2 = dict(cmp)
    cmp2["flops"] = 2 * cmp["flops"]
    cmp2["transcend"] = 2 * cmp["transcend"]
    from arlamx_v2.inference_budget import to_ms
    pol_plus = policy_entry(
        "v12 4x18 + comparator 2×600s", obs_dim, ARCH, 7,
        extra=[("comparator 2 cand × 2 H", (cmp2["flops"], cmp2["transcend"]))])
    out = {
        "obs_dim": obs_dim, "arch": ARCH, "act_dim": 7,
        "mpc_u575_ms_published": mpc_u575_ms(6, 8, scale="published"),
        "mpc_u575_ms_actual": mpc_u575_ms(6, 8, scale="actual"),
        "v12_policy_u575_ms": pol["u575_ms"],
        "v12_policy_plus_cmp_u575_ms": pol_plus["u575_ms"],
        "comparator_u575_ms": to_ms(cmp2["flops"], cmp2["transcend"]),
        "note": "U575 is analytic FLOP/0.35; workstation infer_ws_ms is in eval_curve.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "infer.json").write_text(json.dumps(out, indent=2))
    print(f"[v12] infer U575  MPC={out['mpc_u575_ms_published']:.2f}ms  "
          f"v12_pol={out['v12_policy_u575_ms']:.3f}ms  "
          f"v12+cmp={out['v12_policy_plus_cmp_u575_ms']:.2f}ms", flush=True)
    return out


def _write_best(run_dir, rec):
    (run_dir / "BEST.md").write_text(
        f"# best checkpoint\n\n"
        f"- step: {rec['step']}\n"
        f"- min_score_A: {rec.get('score_A')}\n"
        f"- min_score_D: {rec.get('score_D')}\n"
        f"- decay: {rec['agg']['decay_km_d']:.3f}  band: {rec['agg']['band_pct']:.1f}  "
        f"dl: {rec['agg']['downlink_min_d']:.2f}  slews/d: {rec['agg']['slews_per_day']:.1f}\n"
        f"- path: `{rec['path']}`\n"
        f"- ref: dual A (v1 truth) and D (v2 sensors)\n"
        f"- why: highest min_score_A among checkpoints with 0 brownouts "
        f"(D as secondary)\n"
        f"- when: {_now()}\n"
    )


def _campaign_row(name, rec):
    p = OUT / "BEST_SO_FAR.md"
    line = (f"| {name} | {rec.get('step','')} | {rec.get('score_A')} | "
            f"{rec.get('score_D')} | {rec['agg']['decay_km_d']:.2f} | "
            f"{rec['agg']['band_pct']:.1f} | {rec['agg']['downlink_min_d']:.1f} | "
            f"{rec['agg']['slews_per_day']:.0f} | {rec['agg']['brownouts']} |\n")
    if not p.exists():
        p.write_text(
            "# v12 best-so-far (cancel-safe)\n\n"
            "| run | step | score_A | score_D | decay | band | dl | slews/d | brn |\n"
            "|---|---|---|---|---|---|---|---|---|\n" + line
        )
        return
    text = p.read_text()
    lines = text.splitlines(True)
    kept = [ln for ln in lines if not ln.startswith(f"| {name} |")]
    # insert before end
    if kept and kept[-1].startswith("|"):
        kept.append(line)
    else:
        kept.append(line)
    p.write_text("".join(kept))


CURVE_FIELDS = [
    "step", "score_A", "score_D", "decay_km_d", "band_pct",
    "downlink_min_d", "slews_per_day", "mtq_energy_J", "brownouts",
    "deleg_P", "deleg_B", "coil_idle_frac", "kept_prev_frac",
    "pei_act_over_pd", "pei_saved", "infer_ws_ms", "infer_cmp_ms",
    "infer_total_ms", "clip_frac", "wrong_way_frac", "kf_aero_r",
    "pwr_aero_r", "F_est_uN", "lifetime_d_500_300",
    "storm_band_pct", "storm_downlink_min_d", "storm_deleg_B",
    "storm_decay_km_d", "storm_brownouts", "dual_ok",
    "eta_mean", "eta_slow_mean", "shock_frac", "kd_eff_mean",
    "gate_floor_mean", "margin_eff_mean",
]


def _row_from_agg(step, sA, sD, agg):
    def r(k, nd=4):
        v = agg.get(k)
        if v is None:
            return ""
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(fv):
            return ""
        return round(fv, nd)
    return {
        "step": step,
        "score_A": "" if sA is None else round(sA, 4),
        "score_D": "" if sD is None else round(sD, 4),
        "decay_km_d": r("decay_km_d"),
        "band_pct": r("band_pct", 3),
        "downlink_min_d": r("downlink_min_d", 3),
        "slews_per_day": r("slews_per_day", 3),
        "mtq_energy_J": r("mtq_energy_J"),
        "brownouts": agg.get("brownouts", ""),
        "deleg_P": r("deleg_P"),
        "deleg_B": r("deleg_B"),
        "coil_idle_frac": r("coil_idle_frac"),
        "kept_prev_frac": r("kept_prev_frac"),
        "pei_act_over_pd": r("pei_act_over_pd"),
        "pei_saved": r("pei_saved"),
        "infer_ws_ms": r("infer_ws_ms", 3),
        "infer_cmp_ms": r("infer_cmp_ms", 3),
        "infer_total_ms": r("infer_total_ms", 3),
        "clip_frac": r("clip_frac"),
        "wrong_way_frac": r("wrong_way_frac"),
        "kf_aero_r": r("kf_aero_r"),
        "pwr_aero_r": r("pwr_aero_r"),
        "F_est_uN": r("F_est_uN", 3),
        "lifetime_d_500_300": r("lifetime_d_500_300", 2),
        "eta_mean": r("eta_mean"),
        "eta_slow_mean": r("eta_slow_mean"),
        "shock_frac": r("shock_frac"),
        "kd_eff_mean": r("kd_eff_mean", 6),
        "gate_floor_mean": r("gate_floor_mean"),
        "margin_eff_mean": r("margin_eff_mean"),
    }


def _wrap_kw(w_slew, use_cmp, w_env, clip_wrong_way, use_pwr_est, w_clip,
             w_plan=0.0, w_hold=0.0, adapt_mrp=False,
             w_band_storm=0.0, w_low=0.0,
             physics="standard", controller=None, gsi=None):
    return dict(w_slew=w_slew, use_cmp=use_cmp, w_env=w_env,
                clip_wrong_way=clip_wrong_way, use_pwr_est=use_pwr_est,
                w_clip=w_clip, w_plan=w_plan, w_hold=w_hold,
                adapt_mrp=adapt_mrp, w_band_storm=w_band_storm, w_low=w_low,
                physics=physics, controller=controller, gsi=gsi)


def _log_line(tag, name, n, sA, sD, agg):
    print(
        f"[v12] {name} {tag} {n} score_A={sA} score_D={sD} "
        f"dec={agg['decay_km_d']:.2f} band={agg['band_pct']:.1f} "
        f"dl={agg['downlink_min_d']:.1f} slew/d={agg['slews_per_day']:.0f} "
        f"kept={agg.get('kept_prev_frac', 0):.2f} "
        f"idle={agg.get('coil_idle_frac', 0):.2f} "
        f"P={agg.get('deleg_P', 0):.2f} B={agg.get('deleg_B', 0):.2f} "
        f"clip={agg.get('clip_frac', 0):.2f} "
        f"infer={agg.get('infer_ws_ms', 0):.2f}ms "
        f"tot={agg.get('infer_total_ms', 0):.2f}ms "
        f"kf_r={agg.get('kf_aero_r', float('nan')):.2f} "
        f"pwr_r={agg.get('pwr_aero_r', float('nan')):.2f} "
        f"eta={agg.get('eta_mean', 0):.2f} "
        f"kd={agg.get('kd_eff_mean', 0):.2e} "
        f"shock={agg.get('shock_frac', 0):.2f}",
        flush=True,
    )


def _dual_ok(quiet, storm, gate):
    if storm is None or quiet is None:
        return False
    if gate == "v14c":
        return keep_ok_v14c(quiet, storm)
    if gate == "v14":
        return keep_ok_v14(quiet, storm)
    return keep_ok_v135(quiet, storm)


def _apply_ppo_kw(model, ppo_kw, n_envs):
    """Retune PPO after load without reinitialising weights.

    Larger batch / fewer epochs / lower LR = more seed-stable updates on a
    resume. n_steps * n_envs must stay divisible by batch_size.
    """
    if not ppo_kw:
        return model
    from stable_baselines3.common.buffers import RolloutBuffer
    if "learning_rate" in ppo_kw:
        lr = float(ppo_kw["learning_rate"])
        lr_end = float(ppo_kw.get("learning_rate_end", lr))
        model.learning_rate = lr
        model.lr_schedule = lambda progress, a=lr, b=lr_end: b + (a - b) * progress
    if "n_epochs" in ppo_kw:
        model.n_epochs = int(ppo_kw["n_epochs"])
    if "ent_coef" in ppo_kw:
        model.ent_coef = float(ppo_kw["ent_coef"])
    if "clip_range" in ppo_kw:
        cr = float(ppo_kw["clip_range"])
        model.clip_range = lambda _: cr
    n_steps = int(ppo_kw.get("n_steps", model.n_steps))
    batch = int(ppo_kw.get("batch_size", model.batch_size))
    model.n_steps = n_steps
    model.batch_size = batch
    model.rollout_buffer = RolloutBuffer(
        n_steps, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma,
        gae_lambda=model.gae_lambda, n_envs=n_envs)
    print(f"[v12] ppo_kw n_steps={n_steps} batch={batch} epochs={model.n_epochs} "
          f"ent={model.ent_coef} lr={ppo_kw.get('learning_rate')}", flush=True)
    return model


def train_run(name, w_slew, steps, n_envs, seed, reward_w=None,
              use_cmp=False, w_env=0.0, clip_wrong_way=False,
              use_pwr_est=False, w_clip=0.0, resume=None,
              w_plan=0.0, w_hold=0.0, adapt_mrp=False, dual_eval=False,
              dual_gate="v135", climate_mix=False, ppo_kw=None,
              w_band_storm=0.0, w_low=0.0, climate_wide=False,
              low_soc_frac=0.0, w_low_thr=0.25, w_gen=0.0, w_gen_thr=0.20,
              physics="standard", controller=None, gsi=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    run_dir = OUT / "runs" / name
    if (run_dir / "metrics.json").exists() and (run_dir / "eval_curve.csv").exists():
        print(f"[v12] skip {name} (metrics exist)", flush=True)
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "best").mkdir(exist_ok=True)
    snapshot({
        "name": name, "variant": VARIANT, "w_slew": w_slew, "steps": steps,
        "n_envs": n_envs, "seed": seed, "arch": ARCH, "reward_w": reward_w or {},
        "use_cmp": use_cmp, "w_env": w_env, "clip_wrong_way": clip_wrong_way,
        "use_pwr_est": use_pwr_est, "w_clip": w_clip,
        "w_plan": w_plan, "w_hold": w_hold, "adapt_mrp": adapt_mrp,
        "dual_eval": dual_eval, "dual_gate": dual_gate,
        "climate_mix": climate_mix, "ppo_kw": ppo_kw or {},
        "w_band_storm": w_band_storm, "w_low": w_low,
        "climate_wide": climate_wide, "low_soc_frac": low_soc_frac,
        "w_low_thr": w_low_thr, "w_gen": w_gen, "w_gen_thr": w_gen_thr,
        "resume": None if resume is None else str(resume),
        "ref_A": REF_A, "ref_D": REF_D,
        "physics": physics, "controller": controller, "gsi": gsi,
    }, run_dir / "snapshot.json")

    ctor = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = ctor([make_env(i, seed, w_slew, reward_w, use_cmp, w_env,
                          clip_wrong_way, use_pwr_est, w_clip,
                          w_plan, w_hold, adapt_mrp, climate_mix,
                          w_band_storm, w_low, climate_wide, low_soc_frac,
                          w_low_thr, w_gen, w_gen_thr,
                          physics=physics, controller=controller, gsi=gsi)
                 for i in range(n_envs)])
    ekw = _wrap_kw(w_slew, use_cmp, w_env, clip_wrong_way, use_pwr_est, w_clip,
                   w_plan, w_hold, adapt_mrp, w_band_storm, w_low,
                   physics=physics, controller=controller, gsi=gsi)
    if resume is not None:
        rdir = Path(resume)
        venv = VecNormalize.load(str(rdir / "vecnormalize.pkl"), venv)
        venv.training = True
        venv.norm_reward = True
        model = PPO.load(str(rdir / "model.zip"), env=venv,
                         tensorboard_log=str(run_dir / "logs"))
        print(f"[v12] resume {name} from {rdir}", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=0.99)
        model = build_model("ppo", venv, ARCH, seed, run_dir / "logs",
                            ppo_kw=ppo_kw)
    if ppo_kw:
        model = _apply_ppo_kw(model, ppo_kw, n_envs)

    fields = CURVE_FIELDS
    curve = run_dir / "eval_curve.csv"
    if not curve.exists():
        with open(curve, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    best = {"score_A": -1e9, "dual": -1e9}

    def _run_eval(predict, n_step):
        agg = _eval_wrapped(predict, **ekw)
        sA, sD = min_score(agg, REF_A), min_score(agg, REF_D)
        row = _row_from_agg(n_step, sA, sD, agg)
        storm = None
        if dual_eval:
            storm = _eval_wrapped(predict, **ekw, storm=True)
            row["storm_band_pct"] = round(float(storm["band_pct"]), 3)
            row["storm_downlink_min_d"] = round(float(storm["downlink_min_d"]), 3)
            row["storm_deleg_B"] = round(float(storm.get("deleg_B", 0)), 4)
            row["storm_decay_km_d"] = round(float(storm["decay_km_d"]), 4)
            row["storm_brownouts"] = storm.get("brownouts", "")
            row["dual_ok"] = int(_dual_ok(agg, storm, dual_gate))
            _log_line("storm", name, n_step, sA, sD, storm)
            print(f"[v12] {name} dual_ok={row['dual_ok']} gate={dual_gate} "
                  f"quiet_dl={agg['downlink_min_d']:.1f} "
                  f"storm_band={storm['band_pct']:.1f} "
                  f"B={agg.get('deleg_B', 0):+.2f}", flush=True)
        return agg, storm, sA, sD, row

    class Ckpt(BaseCallback):
        def __init__(self):
            super().__init__()
            self._last = 0

        def _on_step(self):
            n = int(self.num_timesteps)
            if n < CKPT_EVERY or n - self._last < CKPT_EVERY:
                return True
            self._last = n
            tag = f"step_{n}"
            ck = run_dir / "ckpts" / tag
            ck.mkdir(parents=True, exist_ok=True)
            self.model.save(str(ck / "model"))
            venv.save(str(ck / "vecnormalize.pkl"))

            def predict(obs, env):
                a, _ = self.model.predict(obs, deterministic=True)
                return a
            agg, storm, sA, sD, row = _run_eval(predict, n)
            with open(curve, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            rec = {"step": n, "score_A": sA, "score_D": sD, "agg": agg,
                   "storm": storm, "path": str(ck),
                   "dual_ok": bool(_dual_ok(agg, storm, dual_gate))}
            key = -1e9 if sA is None else sA
            if dual_eval:
                if rec["dual_ok"]:
                    dkey = min(key, float(storm["band_pct"]) / 80.0)
                    if dkey >= best["dual"]:
                        best.update(dual=dkey, rec=rec, score_A=key)
                        self.model.save(str(run_dir / "best" / "model"))
                        venv.save(str(run_dir / "best" / "vecnormalize.pkl"))
                        _write_best(run_dir, rec)
                        _campaign_row(name, rec)
            elif key >= best["score_A"]:
                best.update(score_A=key, rec=rec)
                self.model.save(str(run_dir / "best" / "model"))
                venv.save(str(run_dir / "best" / "vecnormalize.pkl"))
                _write_best(run_dir, rec)
                _campaign_row(name, rec)
            _log_line("ckpt", name, n, sA, sD, agg)
            return True

    t0 = time.time()
    model.learn(total_timesteps=int(steps), callback=Ckpt(),
                reset_num_timesteps=True)
    wall = time.time() - t0
    model.save(str(run_dir / "models" / f"ppo_{name}"))
    venv.save(str(run_dir / "models" / "vecnormalize.pkl"))

    def predict_final(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return a
    agg, storm, sA, sD, row = _run_eval(predict_final, int(steps))
    with open(curve, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)
    rec = {"step": int(steps), "score_A": sA, "score_D": sD, "agg": agg,
           "storm": storm, "path": str(run_dir / "models"),
           "dual_ok": bool(_dual_ok(agg, storm, dual_gate))}
    key = -1e9 if sA is None else sA
    if dual_eval:
        if rec["dual_ok"]:
            dkey = min(key, float(storm["band_pct"]) / 80.0)
            if dkey >= best.get("dual", -1e9):
                best.update(dual=dkey, rec=rec, score_A=key)
                model.save(str(run_dir / "best" / "model"))
                venv.save(str(run_dir / "best" / "vecnormalize.pkl"))
                _write_best(run_dir, rec)
                _campaign_row(name, rec)
            elif "rec" in best:
                _write_best(run_dir, best["rec"])
                _campaign_row(name, best["rec"])
        elif "rec" in best:
            _write_best(run_dir, best["rec"])
            _campaign_row(name, best["rec"])
    elif key >= best["score_A"]:
        best.update(score_A=key, rec=rec)
        model.save(str(run_dir / "best" / "model"))
        venv.save(str(run_dir / "best" / "vecnormalize.pkl"))
        _write_best(run_dir, rec)
        _campaign_row(name, rec)
    elif "rec" in best:
        _write_best(run_dir, best["rec"])
        _campaign_row(name, best["rec"])
    _log_line("final", name, int(steps), sA, sD, agg)
    metrics = dict(name=name, variant=VARIANT, w_slew=w_slew, steps=steps,
                   n_envs=n_envs, seed=seed, wall_s=round(wall, 1),
                   steps_per_s=steps / max(wall, 1e-9), arch=ARCH,
                   use_cmp=use_cmp, w_env=w_env, clip_wrong_way=clip_wrong_way,
                   use_pwr_est=use_pwr_est, w_clip=w_clip,
                   w_plan=w_plan, w_hold=w_hold, adapt_mrp=adapt_mrp,
                   dual_eval=dual_eval, dual_gate=dual_gate,
                   climate_mix=climate_mix, ppo_kw=ppo_kw or {},
                   eval=agg, storm=storm,
                   score_A=sA, score_D=sD,
                   best_score_A=None if "rec" not in best else best["rec"].get("score_A"),
                   dual_ok=rec.get("dual_ok"))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    venv.close()
    print(f"[v12] done {name} {wall:.0f}s  {metrics['steps_per_s']:.0f} steps/s",
          flush=True)
    return metrics


def round_r0(n_envs):
    """3 seeds × 40 k, w_slew in {0.25, 0.5, 1.0}. Select by median score_A."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in (0.25, 0.5, 1.0):
        for seed in SEEDS:
            name = f"r0_slew{w:g}_4x18_40k_s{seed}"
            print(f"[v12] r0 {name}", flush=True)
            train_run(name, w, BURN, n_envs, seed)
            curve = OUT / "runs" / name / "eval_curve.csv"
            last = {}
            if curve.exists():
                last = list(csv.DictReader(open(curve)))[-1]
            rows.append({"w_slew": w, "seed": seed, **last, "name": name})
    # median score_A per w
    print("\n[v12] r0 medians (score_A vs v1-truth gate)", flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["w_slew"], []).append(r)
    winner, best = None, -1e9
    for w, rs in by.items():
        def med(k):
            vals = []
            for x in rs:
                try:
                    vals.append(float(x.get(k) or "nan"))
                except ValueError:
                    pass
            return float(np.nanmedian(vals)) if vals else float("nan")
        sA = med("score_A")
        print(f"  w_slew={w:g}  score_A={sA:.3f}  dec={med('decay_km_d'):.2f}  "
              f"band={med('band_pct'):.1f}  dl={med('downlink_min_d'):.1f}  "
              f"slew/d={med('slews_per_day'):.0f}  brn={med('brownouts'):.0f}",
              flush=True)
        if np.isfinite(sA) and sA > best:
            best, winner = sA, w
    (OUT / "r0.json").write_text(json.dumps(
        {"rows": rows, "winner_w_slew": winner, "when": _now()}, indent=2, default=str))
    print(f"[v12] r0 winner w_slew={winner}", flush=True)
    return winner or 0.5


def round_long(n_envs, w_slew, steps):
    for seed in SEEDS:
        name = f"long_slew{w_slew:g}_4x18_{steps // 1000}k_s{seed}"
        print(f"[v12] long {name}", flush=True)
        train_run(name, w_slew, steps, n_envs, seed)


def round_s2(n_envs, w_slew, steps):
    """Comparator in the loop + r_env. New run names (does not skip stage-1)."""
    infer_costs()
    for seed in SEEDS:
        name = f"s2_cmp_slew{w_slew:g}_4x18_{steps // 1000}k_s{seed}"
        print(f"[v12] s2 {name}", flush=True)
        train_run(name, w_slew, steps, n_envs, seed, use_cmp=True, w_env=0.1)


def _s3_resume(seed):
    """Warm-start s3 from the best s2 checkpoint per seed (s44 from s43)."""
    s43 = OUT / "runs" / "s2_cmp_slew0.5_4x18_300k_s43" / "ckpts" / "step_300096"
    s42 = OUT / "runs" / "s2_cmp_slew0.5_4x18_300k_s42" / "ckpts" / "step_250080"
    pick = {42: s42, 43: s43, 44: s43}.get(int(seed), s43)
    if (pick / "model.zip").exists() and (pick / "vecnormalize.pkl").exists():
        return pick
    return None


def round_s3(n_envs, w_slew, steps):
    """Wrong-way gate clip + power-based tau_env. Warm-start s2 best ckpts.

    Not named SC_v13 until actuation (deleg_B, clip_frac, PEI) improves
    without a decay regression.
    """
    infer_costs()
    for seed in SEEDS:
        name = f"s3_clip_slew{w_slew:g}_4x18_{steps // 1000}k_s{seed}"
        resume = _s3_resume(seed)
        print(f"[v12] s3 {name} resume={resume}", flush=True)
        train_run(name, w_slew, steps, n_envs, seed, use_cmp=True, w_env=0.15,
                  clip_wrong_way=True, use_pwr_est=True, w_clip=0.08,
                  resume=resume)


V13_CKPT = (OUT / "runs" / "s3_clip_slew0.5_4x18_2000k_s42" / "ckpts"
            / "step_600192")


def keep_ok(agg):
    """Promotion bar: do not keep a ckpt that fails any of these."""
    if float(agg.get("brownouts", 0)) > 0.5:
        return False
    if float(agg.get("decay_km_d", 99)) > 16.5:
        return False
    if float(agg.get("band_pct", 0)) < 75.0:
        return False
    if float(agg.get("downlink_min_d", 0)) < 20.0:
        return False
    if float(agg.get("deleg_B", -1)) < -0.05:
        return False
    return True


def keep_ok_storm(agg):
    if float(agg.get("brownouts", 0)) > 0.5:
        return False
    if float(agg.get("band_pct", 0)) < 55.0:
        return False
    if float(agg.get("downlink_min_d", 0)) < 10.0:
        return False
    if float(agg.get("deleg_B", -1)) < -0.05:
        return False
    return True


def keep_ok_v135(quiet, storm):
    return keep_ok(quiet) and keep_ok_storm(storm)


def keep_ok_quiet_v14(agg):
    """Quiet regression for v14: MPC-like downlink, not v13's 39 min/d."""
    if float(agg.get("brownouts", 0)) > 0.5:
        return False
    if float(agg.get("decay_km_d", 99)) > 16.5:
        return False
    if float(agg.get("band_pct", 0)) < 75.0:
        return False
    if float(agg.get("downlink_min_d", 0)) < 18.0:
        return False
    if float(agg.get("deleg_B", -1)) < 0.0:
        return False
    return True


def keep_ok_v14(quiet, storm):
    return keep_ok_quiet_v14(quiet) and keep_ok_storm(storm)


def keep_ok_quiet_v14c(agg):
    """Keep v14b quiet floor: band, dl 23-class, days not worse, B ≥ 0."""
    if float(agg.get("brownouts", 0)) > 0.5:
        return False
    if float(agg.get("decay_km_d", 99)) > 15.0:
        return False
    if float(agg.get("band_pct", 0)) < 75.0:
        return False
    if float(agg.get("downlink_min_d", 0)) < 22.0:
        return False
    if float(agg.get("deleg_B", -1)) < 0.0:
        return False
    return True


def keep_ok_storm_v14c(agg):
    if float(agg.get("brownouts", 0)) > 0.5:
        return False
    if float(agg.get("band_pct", 0)) < 62.0:
        return False
    if float(agg.get("downlink_min_d", 0)) < 12.0:
        return False
    if float(agg.get("deleg_B", -1)) < 0.0:
        return False
    return True


def keep_ok_v14c(quiet, storm):
    return keep_ok_quiet_v14c(quiet) and keep_ok_storm_v14c(storm)


def round_s4(n_envs, w_slew, steps):
    """One knob at a time from the SC_v13 candidate (s3 s42 @ 600 k).

    Late s3 collapsed because the policy kept proposing wrong-way splits
    (clip_frac → 0.7). s4 is short (default 400 k), always resumes 600 k,
    and only *keeps* a ckpt that still clears the promotion bar.

    Order (one variable per run):
      hold     w_env=0.15 w_clip=0.08   control
      clip0.20 w_env=0.15 w_clip=0.20   unlearn the fight
      env0.25  w_env=0.25 w_clip=0.08   more environment in the reward
      env0.35  w_env=0.35 w_clip=0.08   still more environment
    """
    infer_costs()
    resume = V13_CKPT
    if not (resume / "model.zip").exists():
        raise SystemExit(f"s4 needs {resume}")
    jobs = (
        ("s4_hold", 0.15, 0.08),
        ("s4_clip0.20", 0.15, 0.20),
        ("s4_env0.25", 0.25, 0.08),
        ("s4_env0.35", 0.35, 0.08),
    )
    seeds = (42, 43)
    for tag, w_env, w_clip in jobs:
        for seed in seeds:
            name = f"{tag}_slew{w_slew:g}_4x18_{steps // 1000}k_s{seed}"
            print(f"[v12] s4 {name} w_env={w_env} w_clip={w_clip} "
                  f"resume={resume}", flush=True)
            rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                            w_env=w_env, clip_wrong_way=True, use_pwr_est=True,
                            w_clip=w_clip, resume=resume)
            agg = (rec or {}).get("eval") or {}
            flag = "KEEP" if keep_ok(agg) else "drop"
            print(f"[v12] s4 {name} {flag}  A={rec.get('score_A')}  "
                  f"dec={agg.get('decay_km_d')} band={agg.get('band_pct')}  "
                  f"B={agg.get('deleg_B')} clip={agg.get('clip_frac')}",
                  flush=True)


V13_FREEZE = OUTPUTS / "v13" / "ckpt"


def _pick_keep_ckpt(run_dir):
    """Best keep_ok row on eval_curve; else highest score_A that still has band."""
    import shutil
    curve = Path(run_dir) / "eval_curve.csv"
    if not curve.exists():
        return None
    rows = list(csv.DictReader(open(curve)))
    kept = []
    for r in rows:
        try:
            agg = {k: float(r[k]) for k in (
                "decay_km_d", "band_pct", "downlink_min_d", "deleg_B", "brownouts")
                if r.get(k) not in (None, "")}
        except ValueError:
            continue
        agg.setdefault("brownouts", 0)
        agg.setdefault("deleg_B", -1)
        if keep_ok(agg):
            try:
                kept.append((float(r.get("score_A") or -1e9), r))
            except ValueError:
                pass
    pick = max(kept, key=lambda x: x[0])[1] if kept else None
    if pick is None and rows:
        # Fallback: highest score_A with 0 brownouts (report that the bar failed).
        scored = []
        for r in rows:
            try:
                if float(r.get("brownouts") or 0) > 0.5:
                    continue
                scored.append((float(r.get("score_A") or -1e9), r))
            except ValueError:
                pass
        pick = max(scored, key=lambda x: x[0])[1] if scored else rows[-1]
        print("[v13b] no ckpt cleared the promotion bar; using fallback",
              flush=True)
    if pick is None:
        return None
    step = pick.get("step")
    src = Path(run_dir) / "ckpts" / f"step_{step}"
    if not src.exists():
        src = Path(run_dir) / "ckpts" / f"step_{int(float(step))}"
    dest = OUTPUTS / "v13" / "ckpt_v13b"
    dest.mkdir(parents=True, exist_ok=True)
    if (src / "model.zip").exists():
        shutil.copy2(src / "model.zip", dest / "model.zip")
        shutil.copy2(src / "vecnormalize.pkl", dest / "vecnormalize.pkl")
    (OUTPUTS / "v13" / "SC_v13b.md").write_text(
        f"# SC_v13b — faster advisor (no comparator)\n\n"
        f"Resume SC_v13 freeze, **use_cmp=False**, wrong-way clip on, "
        f"w_env=0.35, w_clip=0.08, same 4×18.\n\n"
        f"Checkpoint `{src}` copied to `outputs/v13/ckpt_v13b/`.\n\n"
        f"| | value |\n|---|---|\n"
        f"| step | {step} |\n"
        f"| score_A | {pick.get('score_A')} |\n"
        f"| decay km/d | {pick.get('decay_km_d')} |\n"
        f"| band % | {pick.get('band_pct')} |\n"
        f"| downlink min/d | {pick.get('downlink_min_d')} |\n"
        f"| B | {pick.get('deleg_B')} |\n"
        f"| clip_frac | {pick.get('clip_frac')} |\n"
        f"| U575 | **0.172 ms policy** (no 0.83 ms comparator) |\n"
        f"| vs MPC 6.81 ms | **~40×** |\n"
        f"| vs SC_v13 1.00 ms | **~6×** |\n"
    )
    print(f"[v13b] freeze ckpt step={step} -> {dest}", flush=True)
    return dest


def round_v13b(n_envs, w_slew, steps):
    """SC_v13b: same 4×18 weights, train the comparator *out*.

    The 0.83 ms comparator is 83 % of the 1.00 ms advisor. Dropping it
    (slew clip + wrong-way clip stay) is the only way to move from ~7× to
    ~40× vs MPC without a new observation. Resume the freeze so we do not
    throw away B-positive actuation. Short run; pick by eval_curve + bar.
    """
    infer_costs()
    resume = V13_FREEZE
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v13b needs {resume}")
    if steps <= 0 or steps == 300_000:
        steps = 200_000
    seed = 43
    name = f"v13b_nocmp_slew{w_slew:g}_4x18_{steps // 1000}k_s{seed}"
    print(f"[v13b] {name} resume={resume} use_cmp=False steps={steps}",
          flush=True)
    rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=False,
                    w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                    w_clip=0.08, resume=resume)
    dest = _pick_keep_ckpt(OUT / "runs" / name)
    agg = (rec or {}).get("eval") or {}
    print(f"[v13b] final A={rec.get('score_A')} dec={agg.get('decay_km_d')} "
          f"band={agg.get('band_pct')} B={agg.get('deleg_B')} dest={dest}",
          flush=True)
    return rec


def _copy_dual_ckpt(run_dir, dest_name):
    import shutil
    curve = Path(run_dir) / "eval_curve.csv"
    if not curve.exists():
        return None
    rows = list(csv.DictReader(open(curve)))
    dual = [r for r in rows if str(r.get("dual_ok", "")).strip() in ("1", "True", "true")]
    if not dual:
        print(f"[v135] no dual-gated ckpt in {run_dir}", flush=True)
        return None
    def key(r):
        try:
            return (float(r.get("storm_band_pct") or 0),
                    float(r.get("score_D") or r.get("score_A") or 0))
        except ValueError:
            return (0.0, 0.0)
    pick = max(dual, key=key)
    step = pick.get("step")
    src = Path(run_dir) / "ckpts" / f"step_{step}"
    if not (src / "model.zip").exists():
        src = Path(run_dir) / "ckpts" / f"step_{int(float(step))}"
    dest = OUTPUTS / "v13" / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "model.zip", dest / "model.zip")
    shutil.copy2(src / "vecnormalize.pkl", dest / "vecnormalize.pkl")
    (dest / "PICK.json").write_text(json.dumps(pick, indent=2))
    print(f"[v135] copied {src} -> {dest}  storm_band={pick.get('storm_band_pct')} "
          f"quiet_band={pick.get('band_pct')} B={pick.get('deleg_B')}", flush=True)
    return dest


def round_v135(n_envs, w_slew, steps):
    """SC_v13.5: H=6 train-only plan score + hold-in-band + dual picker."""
    infer_costs()
    resume = V13_FREEZE
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v13.5 needs {resume}")
    if steps <= 0:
        steps = 300_000
    seed = 43
    name = f"v135_plan0.20_hold0.25_4x18_{steps // 1000}k_s{seed}"
    print(f"[v135] {name} resume={resume} dual_eval w_plan=0.20 w_hold=0.25",
          flush=True)
    rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                    w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                    w_clip=0.08, w_plan=0.20, w_hold=0.25,
                    dual_eval=True, resume=resume)
    dest = _copy_dual_ckpt(OUT / "runs" / name, "ckpt_v135")
    if dest is None and seed == 43:
        seed2 = 42
        name2 = f"v135_plan0.20_hold0.25_4x18_{steps // 1000}k_s{seed2}"
        print(f"[v135] seed 43 had no dual-gated ckpt; trying {name2}", flush=True)
        rec = train_run(name2, w_slew, steps, n_envs, seed2, use_cmp=True,
                        w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                        w_clip=0.08, w_plan=0.20, w_hold=0.25,
                        dual_eval=True, resume=resume)
        dest = _copy_dual_ckpt(OUT / "runs" / name2, "ckpt_v135")
    print(f"[v135] dest={dest}", flush=True)
    return rec


def round_v14(n_envs, w_slew, steps):
    """SC_v14 Phase 1: resume SC_v13 freeze with onboard regime scheduler.

    No r_plan / r_hold. Dual picker uses quiet dl >= 18 (MPC column D), not
    v13's 39 min/d. Train *with* scheduled kd so train/flight match.
    """
    infer_costs()
    resume = V13_FREEZE
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v14 needs SC_v13 freeze {resume}")
    if steps <= 0:
        steps = 300_000
    seed = 43
    name = f"v14_reg_kd1e3_softclip_4x18_{steps // 1000}k_s{seed}"
    print(f"[v14] {name} resume={resume} adapt_mrp dual_eval gate=v14 "
          f"w_plan=0 w_hold=0", flush=True)
    rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                    w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                    w_clip=0.08, w_plan=0.0, w_hold=0.0,
                    adapt_mrp=True, dual_eval=True, dual_gate="v14",
                    resume=resume)
    dest = _copy_dual_ckpt(OUT / "runs" / name, "ckpt_v14")
    if dest is None:
        seed2 = 42
        name2 = f"v14_reg_kd1e3_softclip_4x18_{steps // 1000}k_s{seed2}"
        print(f"[v14] seed 43 had no dual-gated ckpt; trying {name2}",
              flush=True)
        rec = train_run(name2, w_slew, steps, n_envs, seed2, use_cmp=True,
                        w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                        w_clip=0.08, w_plan=0.0, w_hold=0.0,
                        adapt_mrp=True, dual_eval=True, dual_gate="v14",
                        resume=resume)
        dest = _copy_dual_ckpt(OUT / "runs" / name2, "ckpt_v14")
    print(f"[v14] dest={dest}", flush=True)
    return rec


V14_KEEP = OUTPUTS / "v13" / "ckpt_v14_keep"

# Resume recipe: lower LR, larger batch, fewer epochs. 24 envs × 256 steps
# = 6144 samples / update, 24 minibatches of 256. ~3× less noisy than
# 24 × 128 / 64.
PPO_STABLE = dict(
    learning_rate=1.0e-4,
    learning_rate_end=3.0e-5,
    n_steps=256,
    batch_size=256,
    n_epochs=4,
    ent_coef=0.002,
    clip_range=0.15,
)


def _copy_dual_ckpt_many(run_dirs, dest_name):
    """Best dual-gated row across several runs (seed resilience)."""
    import shutil
    dual = []
    for run_dir in run_dirs:
        curve = Path(run_dir) / "eval_curve.csv"
        if not curve.exists():
            continue
        for r in csv.DictReader(open(curve)):
            if str(r.get("dual_ok", "")).strip() not in ("1", "True", "true"):
                continue
            dual.append((Path(run_dir), r))
    if not dual:
        print(f"[v14b] no dual-gated ckpt in {list(run_dirs)}", flush=True)
        return None

    def key(item):
        r = item[1]
        try:
            return (float(r.get("storm_band_pct") or 0),
                    float(r.get("deleg_B") or 0),
                    float(r.get("downlink_min_d") or 0),
                    float(r.get("score_D") or r.get("score_A") or 0))
        except ValueError:
            return (0.0, 0.0, 0.0, 0.0)
    run_dir, pick = max(dual, key=key)
    step = pick.get("step")
    src = run_dir / "ckpts" / f"step_{step}"
    if not (src / "model.zip").exists():
        src = run_dir / "ckpts" / f"step_{int(float(step))}"
    dest = OUTPUTS / "v13" / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "model.zip", dest / "model.zip")
    shutil.copy2(src / "vecnormalize.pkl", dest / "vecnormalize.pkl")
    (dest / "PICK.json").write_text(json.dumps(pick, indent=2))
    print(f"[v14b] copied {src} -> {dest}  storm_band={pick.get('storm_band_pct')} "
          f"quiet_band={pick.get('band_pct')} dl={pick.get('downlink_min_d')} "
          f"B={pick.get('deleg_B')}", flush=True)
    return dest


def round_v14b(n_envs, w_slew, steps):
    """SC_v14b: keep dl-47 zip, stable PPO, mixed climates, both seeds.

    Resume ckpt_v14_keep (s42 @ 200 k: quiet band 75.5, dl 47). Train-time
    climate mix is wrapper-only (env.py untouched). Eval still uses the
    pinned quiet 10-orbit and storm1d protocols.
    """
    infer_costs()
    resume = V14_KEEP
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v14b needs keep zip {resume}")
    if steps <= 0:
        steps = 300_000
    print("[v14b] climate mix 40% quiet-gradual / 35% gradual+spikes / "
          "25% storm (wider than v10). 24 parallel envs, not 24 storms/orbit.",
          flush=True)
    print(f"[v14b] ppo {PPO_STABLE} resume={resume}", flush=True)
    run_dirs = []
    rec = None
    for seed in (42, 43):
        name = f"v14b_mix_ppo_4x18_{steps // 1000}k_s{seed}"
        print(f"[v14b] {name}", flush=True)
        rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                        w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                        w_clip=0.08, w_plan=0.0, w_hold=0.0,
                        adapt_mrp=True, dual_eval=True, dual_gate="v14",
                        climate_mix=True, ppo_kw=PPO_STABLE, resume=resume)
        run_dirs.append(OUT / "runs" / name)
    dest = _copy_dual_ckpt_many(run_dirs, "ckpt_v14b")
    print(f"[v14b] dest={dest}", flush=True)
    return rec


V14B_CKPT = OUTPUTS / "v13" / "ckpt_v14b"
PPO_STABLE_C = dict(
    learning_rate=7.0e-5,
    learning_rate_end=2.0e-5,
    n_steps=256,
    batch_size=256,
    n_epochs=4,
    ent_coef=0.002,
    clip_range=0.12,
)


def eval_prec(ckpt=None):
    """FP32 vs FP16-weight round-trip vs dynamic INT8 on the v14b zip.

    STM32U575 is fp32 FPU; this asks whether storing/running at lower
    precision loses the quiet/storm numbers. Not a retrained fp16 policy.
    """
    from stable_baselines3 import PPO

    ckpt = Path(ckpt or V14B_CKPT)
    if not (ckpt / "model.zip").exists():
        raise SystemExit(f"prec needs {ckpt}")
    wrap = dict(w_slew=0.5, w_env=0.35, use_cmp=True, clip_wrong_way=True,
                use_pwr_est=True, w_clip=0.08, adapt_mrp=True)
    out = {"ckpt": str(ckpt), "when": _now()}

    def load_predict(cast=None):
        model = PPO.load(str(ckpt / "model.zip"))
        if cast == "fp16":
            sd = {k: (v.half().float() if v.is_floating_point() else v)
                  for k, v in model.policy.state_dict().items()}
            model.policy.load_state_dict(sd)
        elif cast == "int8":
            from arlamx_v2.quantize import quantize_int8
            model = quantize_int8(model, algo="ppo")

        def predict(obs, env, _m=model):
            a, _ = _m.predict(obs, deterministic=True)
            return a
        return predict

    for tag, cast in (("fp32", None), ("fp16", "fp16"), ("int8", "int8")):
        try:
            pred = load_predict(cast)
            quiet = _eval_wrapped(pred, **wrap)
            storm = _eval_wrapped(pred, **wrap, storm=True)
            rec = {
                "quiet": {k: quiet[k] for k in (
                    "decay_km_d", "band_pct", "downlink_min_d", "deleg_B",
                    "brownouts", "mtq_energy_J", "clip_frac")},
                "storm": {k: storm[k] for k in (
                    "decay_km_d", "band_pct", "downlink_min_d", "deleg_B",
                    "brownouts", "mtq_energy_J", "clip_frac")},
            }
            print(f"[prec] {tag} quiet band={quiet['band_pct']:.1f} dl="
                  f"{quiet['downlink_min_d']:.1f} B={quiet['deleg_B']:+.2f} "
                  f"brn={quiet['brownouts']}  storm band={storm['band_pct']:.1f} "
                  f"dl={storm['downlink_min_d']:.1f} B={storm['deleg_B']:+.2f} "
                  f"brn={storm['brownouts']}", flush=True)
            out[tag] = rec
        except Exception as e:
            print(f"[prec] {tag} FAILED: {e}", flush=True)
            out[tag] = {"error": str(e)}
    dest = OUTPUTS / "v13" / "PREC_v14b.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    lines = ["# Precision check — SC_v14b weights, not a retrain\n",
             "Quiet 10-orbit and storm1d. FP16 = store weights in half, run fp32.\n",
             "INT8 = PyTorch dynamic quantize of actor Linear layers.\n\n",
             "| prec | quiet band | quiet dl | quiet B | storm band | storm dl | storm B | brn Q/S |\n",
             "|---|---|---|---|---|---|---|---|\n"]
    for tag in ("fp32", "fp16", "int8"):
        rec = out.get(tag) or {}
        if "error" in rec or "quiet" not in rec:
            lines.append(f"| {tag} | FAIL | | | | | | |\n")
            continue
        q, s = rec["quiet"], rec["storm"]
        lines.append(
            f"| {tag} | {q['band_pct']:.1f} | {q['downlink_min_d']:.1f} | "
            f"{q['deleg_B']:+.2f} | {s['band_pct']:.1f} | {s['downlink_min_d']:.1f} | "
            f"{s['deleg_B']:+.2f} | {q['brownouts']}/{s['brownouts']} |\n")
    (OUTPUTS / "v13" / "PREC_v14b.md").write_text("".join(lines))
    print(f"[prec] wrote {dest}", flush=True)
    return out


def round_v14c(n_envs, w_slew, steps):
    """SC_v14c: keep v14b days/J, push MC band, B, downlink, 0 brownouts.

    Resume ckpt_v14b. Storm-weighted band bonus + low-SoC penalty (no extra
    decay weight). Same climate mix and a slightly more conservative PPO.
    """
    infer_costs()
    resume = V14B_CKPT
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v14c needs {resume}")
    print("[v14c] precision check on v14b (fp32 / fp16 / int8)", flush=True)
    try:
        eval_prec(resume)
    except Exception as e:
        print(f"[v14c] prec skipped: {e}", flush=True)
    if steps <= 0:
        steps = 300_000
    print("[v14c] w_band_storm=0.20 w_low=0.15  keep days & J<6, "
          "push band / B / dl / 0 brownouts", flush=True)
    print(f"[v14c] ppo {PPO_STABLE_C} resume={resume}", flush=True)
    run_dirs = []
    rec = None
    for seed in (42, 43):
        name = f"v14c_band_4x18_{steps // 1000}k_s{seed}"
        print(f"[v14c] {name}", flush=True)
        rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                        w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                        w_clip=0.08, w_plan=0.0, w_hold=0.0,
                        adapt_mrp=True, dual_eval=True, dual_gate="v14c",
                        climate_mix=True, ppo_kw=PPO_STABLE_C, resume=resume,
                        w_band_storm=0.20, w_low=0.15)
        run_dirs.append(OUT / "runs" / name)
    dest = _copy_dual_ckpt_many(run_dirs, "ckpt_v14c")
    print(f"[v14c] dest={dest}", flush=True)
    return rec


def round_v14d(n_envs, w_slew, steps):
    """SC_v14d: one-knob arms from v14b. Do not stack like v14c.

    D1 w_low=0.10 (brownouts). D2 w_env=0.40 (B). D3 w_band_storm=0.08
    (lighter than c). Picker = v14 gate (not the empty v14c storm-dl≥12).
    """
    infer_costs()
    resume = V14B_CKPT
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v14d needs {resume}")
    if steps <= 0 or steps == 300_000:
        steps = 200_000
    jobs = (
        ("low0.10", dict(w_low=0.10, w_band_storm=0.0, w_env=0.35)),
        ("env0.40", dict(w_low=0.0, w_band_storm=0.0, w_env=0.40)),
        ("band0.08", dict(w_low=0.0, w_band_storm=0.08, w_env=0.35)),
    )
    print("[v14d] one knob per arm, resume v14b, dual_gate=v14, "
          f"steps={steps} seed=42", flush=True)
    run_dirs = []
    rec = None
    for tag, extra in jobs:
        seed = 42
        name = f"v14d_{tag}_4x18_{steps // 1000}k_s{seed}"
        print(f"[v14d] {name} {extra}", flush=True)
        rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                        clip_wrong_way=True, use_pwr_est=True, w_clip=0.08,
                        w_plan=0.0, w_hold=0.0, adapt_mrp=True,
                        dual_eval=True, dual_gate="v14", climate_mix=True,
                        ppo_kw=PPO_STABLE_C, resume=resume, **extra)
        run_dirs.append(OUT / "runs" / name)
    dest = _copy_dual_ckpt_many(run_dirs, "ckpt_v14d")
    print(f"[v14d] dest={dest}", flush=True)
    return rec


V14D_CKPT = OUTPUTS / "v13" / "ckpt_v14d"


def round_v14e(n_envs, w_slew, steps):
    """SC_v14e: resume v14d (13.3 d, J 5.72). Wider weather, low-SoC starts.

    Keep days/J/decay. Do not raise w_low (MC brownouts got worse at 0.10
    mixed with long MC). Tighten the floor to SoC<0.15, nudge B with w_env=0.38,
    more storm episodes. Seed 42 first; seed 43 only if 42 misses the dual gate.
    """
    infer_costs()
    resume = V14D_CKPT
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v14e needs {resume}")
    if steps <= 0 or steps == 300_000:
        steps = 250_000
    print("[v14e] climate_wide 30/30/40 + low_soc 20% (storm-boosted, "
          "start SoC U[0.16,0.34])  w_env=0.38  w_low=0.10 thr=0.15  "
          "no w_band_storm", flush=True)
    run_dirs = []
    rec = None
    dest = None
    for seed in (42, 43):
        name = f"v14e_wide_4x18_{steps // 1000}k_s{seed}"
        print(f"[v14e] {name}", flush=True)
        rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                        w_env=0.38, clip_wrong_way=True, use_pwr_est=True,
                        w_clip=0.08, w_plan=0.0, w_hold=0.0, adapt_mrp=True,
                        dual_eval=True, dual_gate="v14", climate_mix=True,
                        ppo_kw=PPO_STABLE_C, resume=resume, w_band_storm=0.0,
                        w_low=0.10, climate_wide=True, low_soc_frac=0.20,
                        w_low_thr=0.15)
        run_dirs.append(OUT / "runs" / name)
        dest = _copy_dual_ckpt_many(run_dirs, "ckpt_v14e")
        if dest is not None and seed == 42:
            pick = json.loads((dest / "PICK.json").read_text())
            storm_band = float(pick.get("storm_band_pct") or 0)
            if storm_band >= 62.0:
                print("[v14e] seed 42 dual-gated storm_band="
                      f"{storm_band:.1f}; skipping seed 43", flush=True)
                break
            print("[v14e] seed 42 dual-gated but storm_band="
                  f"{storm_band:.1f} < 62; training seed 43", flush=True)
    print(f"[v14e] dest={dest}", flush=True)
    return rec


V14E_CKPT = OUTPUTS / "v13" / "ckpt_v14e"


def round_v14f(n_envs, w_slew, steps):
    """SC_v14f: resume v14d. One new knob — charge when SoC is low.

    Keep days ~13.3 and J. Do not raise w_low (that raised MC brownouts).
    Do not reopen climate_wide (v14e). w_gen rewards power_gen_norm below
    SoC 0.20 so the policy points to charge instead of idling the coils.
    """
    infer_costs()
    resume = V14D_CKPT
    if not (resume / "model.zip").exists():
        raise SystemExit(f"v14f needs {resume}")
    if steps <= 0 or steps == 300_000:
        steps = 200_000
    print("[v14f] resume v14d  w_gen=0.12 thr=0.20  keep w_low=0.10  "
          "no climate_wide", flush=True)
    run_dirs = []
    rec = None
    dest = None
    for seed in (42, 43):
        name = f"v14f_gen_4x18_{steps // 1000}k_s{seed}"
        print(f"[v14f] {name}", flush=True)
        rec = train_run(name, w_slew, steps, n_envs, seed, use_cmp=True,
                        w_env=0.35, clip_wrong_way=True, use_pwr_est=True,
                        w_clip=0.08, w_plan=0.0, w_hold=0.0, adapt_mrp=True,
                        dual_eval=True, dual_gate="v14", climate_mix=True,
                        ppo_kw=PPO_STABLE_C, resume=resume, w_band_storm=0.0,
                        w_low=0.10, climate_wide=False, low_soc_frac=0.0,
                        w_low_thr=0.25, w_gen=0.12, w_gen_thr=0.20)
        run_dirs.append(OUT / "runs" / name)
        dest = _copy_dual_ckpt_many(run_dirs, "ckpt_v14f")
        if dest is not None and seed == 42:
            pick = json.loads((dest / "PICK.json").read_text())
            storm_band = float(pick.get("storm_band_pct") or 0)
            if storm_band >= 62.0:
                print("[v14f] seed 42 dual-gated storm_band="
                      f"{storm_band:.1f}; skipping seed 43", flush=True)
                break
            print("[v14f] seed 42 dual-gated but storm_band="
                  f"{storm_band:.1f} < 62; training seed 43", flush=True)
    print(f"[v14f] dest={dest}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="r0",
                    choices=("r0", "long", "s2", "s3", "s4", "v13b",
                             "v135", "v14", "v14b", "v14c", "v14d", "v14e",
                             "v14f", "prec", "all", "infer"))
    ap.add_argument("--w-slew", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int,
                    default=max(8, (os.cpu_count() or 16) // 2))
    a = ap.parse_args()
    print(f"[v12] round={a.round} n_envs={a.n_envs} cpu={os.cpu_count()} {_now()}",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "GATE_MPC.md").write_text(
        "# MPC reference for v12 (from outputs/mpc/checks)\n\n"
        "Check 1 PASS: plant reproduces v1 probe20 (9.282/86.61/23.24).\n"
        "Check 2: v2 downlink/band/mtq hit is **sensors**, not slew 0.5.\n"
        "Frozen v3 = v2 knobs. Dual columns A (v1 truth) and D (v2 sensors).\n"
        "mpc_v2.yaml was not edited.\n"
    )
    (OUT / "PLAN_STATUS.md").write_text(
        "# SC_v12 plan vs what actually ran\n\n"
        "| item | status |\n|---|---|\n"
        "| PPO 4x18, v10 env, r_slew wrapper | **done** (r0 + 300k) |\n"
        "| Dual min-score vs v1 A / v2 D | **done** |\n"
        "| Checkpoint 50k + BEST.md | **done** |\n"
        "| Comparator (MPC surrogate, keep-if-not-better) | **done** (s2) |\n"
        "| r_env (tau_env · tau_want) | **done** (s2) |\n"
        "| PEI, deleg_P/B, coil_idle, kept_prev, infer ms | **done** |\n"
        "| U575 analytic MPC vs v12 | **done** infer.json |\n"
        "| Wrong-way gate clip (s_i=0 if tau_env fights) | **s3** |\n"
        "| Power/geometry tau_env + force from 1 cm offset | **s3** |\n"
        "| MRP action / shadow_flag / hold output | **not** (needs env.py) |\n"
        "| T0 MRP vs quat | **not** |\n"
        "| T1 period sweep | **not** |\n"
        "| T2 seed-count / T3 size | **not** |\n"
        "| Power-limited slew / disturbance reserve | **not** |\n"
        "| v12b re-inference | **not** |\n"
        "| History 50–600 s | **not** (still v9d 150–900 stacked, not LSTM) |\n"
        "| Damage/v11 | **out** (plan) |\n"
        "| MC 512 vs MPC | **not yet** (after a competitive checkpoint) |\n"
        "| Rename SC_v13 | **candidate** s3 s42 @ 600 k (B +0.38, bar cleared) |\n"
        "| s4 one-knob from 600 k (w_clip, w_env) | **s4** |\n"
    )
    if a.round == "r0":
        round_r0(a.n_envs)
    elif a.round == "long":
        round_long(a.n_envs, a.w_slew, a.steps)
    elif a.round == "s2":
        round_s2(a.n_envs, a.w_slew, a.steps)
    elif a.round == "s3":
        round_s3(a.n_envs, a.w_slew, a.steps)
    elif a.round == "s4":
        round_s4(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v13b":
        round_v13b(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v135":
        round_v135(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v14":
        round_v14(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v14b":
        round_v14b(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v14c":
        round_v14c(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v14d":
        round_v14d(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v14e":
        round_v14e(a.n_envs, a.w_slew, a.steps)
    elif a.round == "v14f":
        round_v14f(a.n_envs, a.w_slew, a.steps)
    elif a.round == "prec":
        eval_prec()
    elif a.round == "infer":
        infer_costs()
    else:
        w = round_r0(a.n_envs)
        round_long(a.n_envs, w, a.steps)
        round_s2(a.n_envs, w, a.steps)
        round_s3(a.n_envs, w, a.steps)
        round_s3(a.n_envs, w, a.steps)


if __name__ == "__main__":
    main()
