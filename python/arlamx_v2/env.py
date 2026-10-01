"""Gymnasium env: C++ plant + MSIS + sensors, observations, rewards and actuator gating per variant (v3-v11, v10r6)."""
from __future__ import annotations

import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import dcm_bn_axis_along, quat_body_z_along
from arlamx_v2.advisors.stations import nearest_visible
from arlamx_v2.atmosphere import jd_to_datetime, query_msis
from arlamx_v2 import physics as physics_mod
from arlamx_v2.decay_run import coe_to_rv
from arlamx_v2.geometry import load_geom
from arlamx_v2.paths import HEX_GEOM, resolve_ggm
from arlamx_v2.advisors.mpc import _two_body_rk4
from arlamx_v2.reward import (compose_sc_v3, compose_sc_v4, compose_sc_v5,
                              compose_sc_v6, compose_sc_v7)
from arlamx_v2 import power as mtq_power
from arlamx_v2 import propagator as fastprop
from arlamx_v2 import config as cfg_mod
from arlamx_v2.config import load as load_cfg, load_reward
from arlamx_v2.estimators import TorqueKF, TorqueKFConfig, env_torque
from arlamx_v2.reward_v8 import compose_sc_v8, split_to_gates
from arlamx_v2.sensors import DEG as SENS_DEG, SensorConfig, SensorSuite
from arlamx_v2.damage import apply_damage as dmg_apply, plan_damage as dmg_plan
from arlamx_v2 import vehicle as veh_mod

V8_VARIANTS = ("v8a", "v8b")
V9_VARIANTS = ("v9a", "v9b", "v9c", "v9d", "v10", "v11", "v10r6")

EPOCH_JD = 2461060.5

ORBIT_KEYS = ("altitude_km", "inc_deg", "ecc", "f107", "ap", "raan_deg",
              "argp_deg", "nu_deg", "omega_dps", "mass_kg", "soc",
              "episode_orbits")

MTQ_TAU_NM = np.array([1.8e-5, 1.8e-5, 5.7e-6])
MTQ_M_MAX = np.array([0.6, 0.6, 0.19])
MTQ_KP = 4.0e-4
MTQ_KD = 8.0e-3
MTQ_RATE_DPS = 0.125
INERTIA_DIAG = np.array([0.0125, 0.0125, 0.025])


def _cmd_angle_deg(q_a, q_b):
    d = abs(float(np.dot(q_a, q_b)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def _axis_errors_deg(c_bn, dir_n):
    r_ideal = dcm_bn_axis_along(dir_n, "z")
    r_err = r_ideal @ np.asarray(c_bn, float).T
    tr = float(np.trace(r_err))
    ang = float(np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0)))
    skew = np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ]
    )
    n = float(np.linalg.norm(skew))
    if n < 1e-12 or ang < 1e-12:
        return np.zeros(3)
    axis = skew / n
    return np.degrees(ang * axis)


class ArlamxV2Env(gym.Env):
    metadata = {"render_modes": []}
    _msis_warned = False
    _prop_warned = False

    def __init__(self, geom_path=None, ggm_path=None, seed=None, variant="v3",
                 gate_floor=None, kp=None, kd=None, obs_extra="observer",
                 reward_w=None, controller=None, physics="standard", gsi=None,
                 orbit=None, frozen=None, vehicle="vehicle", sensors_cfg="sensors_solarcat",
                 mtq_duty=1.0, cmd_frame="inertial", advisor_s=None, inner_dt=None):
        super().__init__()
        if frozen:
            cfg_mod.pin(frozen)
        self._orbit = {k: v for k, v in dict(orbit or {}).items() if v is not None}
        bad = sorted(set(self._orbit) - set(ORBIT_KEYS))
        if bad:
            raise ValueError(f"unknown orbit keys {bad}; allowed {ORBIT_KEYS}")
        w = self._orbit.get("omega_dps")
        if w is not None and not (isinstance(w, (list, tuple)) and len(w) == 2):
            raise ValueError("orbit.omega_dps must be [lo, hi] in deg/s")
        allowed = ("v3", "v4a", "v4b", "v5a", "v5b", "v6", "v7") + V8_VARIANTS + V9_VARIANTS
        if variant not in allowed:
            raise ValueError(f"variant must be one of {allowed}")
        self.variant = variant
        self._r6 = variant == "v10r6"
        self._veh = veh_mod.build(load_cfg(vehicle)) if self._r6 else None
        if geom_path is None and self._r6:
            from arlamx_v2.paths import DATA
            geom_path = DATA / load_cfg(vehicle)["geom"]
        self.geom_path = str(geom_path or HEX_GEOM)
        self.ggm_path = ggm_path if ggm_path is not None else resolve_ggm()
        n, a, c = load_geom(self.geom_path)
        self._n, self._A, self._c = n, a, c
        self._v8 = variant in V8_VARIANTS
        self._v9 = variant in V9_VARIANTS
        self._v10 = variant in ("v10", "v11", "v10r6")
        self._I = self._veh.I_diag if self._r6 else INERTIA_DIAG.copy()
        self._v11 = variant == "v11"
        self._v8plus = self._v8 or self._v9
        self._v7 = variant == "v7" or self._v8plus
        self._v4plus = (variant.startswith("v4") or variant.startswith("v5")
                        or variant in ("v6", "v7") or self._v8plus)
        self._v5plus = variant.startswith("v5") or variant == "v6" or self._v8plus
        self._advisor_s = float(advisor_s) if advisor_s else (150.0 if variant == "v6" else 300.0)
        if obs_extra not in ("none", "ctrl", "observer", "observer_fore"):
            raise ValueError("obs_extra must be none|ctrl|observer|observer_fore")
        self._obs_extra = obs_extra
        self._gate_floor = (float(np.clip(gate_floor, 0.0, 1.0))
                            if gate_floor is not None else 0.15)
        self._tau_max = MTQ_TAU_NM.copy()
        self._m_max = MTQ_M_MAX.copy()
        self._kp = float(kp) if kp is not None else MTQ_KP
        self._kd = float(kd) if kd is not None else MTQ_KD
        self._reward_w = dict(reward_w) if reward_w else None
        self._physics = physics_mod.load(physics)
        if gsi is not None:
            from arlamx_v2.config import merge as _merge
            self._physics = physics_mod.validate(
                _merge(self._physics, {"aero": {"gsi": str(gsi).lower()}}))
        law = str(controller if controller is not None
                  else (self._physics.get("attitude") or {}).get("law")
                  or "mrp").lower()
        if law not in ("mrp", "quaternion"):
            raise ValueError("controller must be 'mrp' or 'quaternion'")
        self._controller = law
        self._msis_version = physics_mod.msis_version(self._physics)
        self._atmo_model = str((self._physics.get("atmosphere") or {}).get("model",
                                                                          "msis21")).lower()
        self._atmo_interp = bool((self._physics.get("atmosphere") or {}).get("interp", False))

        self.rcfg = None
        self.sensors = None
        self.kf = None
        self._pcfg = None
        if self._v8plus:
            self.rcfg = load_reward(variant, reward_w)
            gains = load_cfg("gains_mrp")
            self._pcfg = load_cfg("power_mtq")
            if kp is None:
                self._kp = float(gains["kp"])
            if kd is None:
                self._kd = float(gains["kd"])
            if gate_floor is None:
                self._gate_floor = float(gains["gate_floor"])
            self.sensors = SensorSuite(
                SensorConfig.from_dict(load_cfg(sensors_cfg)),
                seed=seed if seed is not None else 0)
            kcfg = load_cfg("estimator_kf")
            if self._r6:
                kcfg["inertia_diag"] = [float(x) for x in self._I]
            kcfg.update(self.rcfg.get("kf", {}))
            self.kf = TorqueKF(TorqueKFConfig(
                enabled=bool(kcfg["enabled"]),
                sigma_drive=float(kcfg["sigma_drive"]),
                sigma_tau_cmd=float(kcfg["sigma_tau_cmd"]),
                p0_tau=float(kcfg["p0_tau"]),
                p0_taudot=float(kcfg["p0_taudot"]),
                innovation_gate_sigma=float(kcfg["innovation_gate_sigma"]),
                inertia_diag=tuple(kcfg["inertia_diag"])))
            self._omega_prev_meas = np.zeros(3)
            self._dE_prev = None
            self._soc_prev = None
            self._hist = []
            self._tau_env_filt = np.zeros(3)
            self._tau_env_raw = np.zeros(3)
            self._deleg_split = np.zeros(3)
            self._trend = None
            self._pdiag = {}
            self._mtq_peak_W = float(np.sum(
                np.asarray(self._pcfg["magnetorquers"]["power_peak_W"], float)))
            self._deleg_live = bool(self.rcfg.get("delegation", {}).get("enabled", False))
            self._duty_lin = 0.0
            self._hist_now = np.zeros(4)
            self._fut_alt_600 = 0.0
            self._fut_soc_600 = 0.5
            self._tau_max = np.asarray(
                self._pcfg["magnetorquers"]["torque_max_Nm"], float)
            self._m_max = np.asarray(
                self._pcfg["magnetorquers"]["dipole_max_Am2"], float)
            if self._r6:
                self._tau_max = self._veh.tau_max.copy()
                self._mtq_peak_W = float(self._veh.rod_pmax.sum())

        p = cpp.SimParams()
        p.mass = 0.625
        p.mtq_duty = float(mtq_duty)
        p.cmd_frame = str(cmd_frame)
        p.dt_s = 2.0
        p.advisor_step_s = self._advisor_s
        p.epoch_jd = EPOCH_JD
        p.max_slew_rad = np.radians(40.0 if self._v4plus else 45.0)
        physics_mod.apply_to_params(
            p, self._physics, ggm_path=self.ggm_path, controller=self._controller)
        if inner_dt:
            p.dt_s = float(inner_dt)
            p.rk4_step_s = min(p.rk4_step_s, float(inner_dt))
        self._want_wmm = str((self._physics.get("magnetics") or {}).get("model",
                                                                       "wmm")).lower() == "wmm"
        self.sim = cpp.Simulator(p)
        self._confirm_gravity(p.ggm_path)
        self._confirm_wmm(p.wmm_path)
        atmo = self._physics.get("atmosphere") or {}
        self._f107 = float(atmo.get("f107", 150.0))
        self._ap = float(atmo.get("ap", 4.0))
        if self._v8plus:
            cp_off = (self._veh.cp_offset if self._r6 else np.asarray(
                self._pcfg["vehicle"].get("cp_offset_m", [0.0, 0.0, 0.0]), float))
            c = c + cp_off[None, :]
            self._c = c
        self.sim.set_panels(n, a, c)
        self.sim.set_inertia_diag(self._I)
        self._n_rod = len(self._veh.rod_axis) if self._r6 else 6
        self._rod_on = np.ones(self._n_rod, bool)
        if self._r6:
            self.sim.set_rods(self._veh.rod_axis, self._veh.rod_dmax)
            off = self._veh.I - np.diag(np.diag(self._veh.I))
            if np.abs(off).max() > 1e-9 * np.abs(self._veh.I).max():
                self.sim.set_inertia(self._veh.I)   # plant: full tensor; flight software keeps the diagonal
        self._mtq_scale = 1.0
        self._gates = np.ones(3)
        self._configure_bdot()
        self._set_plant_dipoles(np.ones(3))
        if self._v5plus:
            self._install_mtq(1.0)
        if self._v7:
            self._install_mtq_axes(np.ones(3))

        if variant == "v7" or self._v8plus:
            act_n = 10 if self._r6 else 7
        elif variant == "v6":
            act_n = 5
        else:
            act_n = 4
        obs_n = 35 if self._v7 else 26
        obs_n += 6 if self._r6 else 0
        if self._v9:
            ocfg = self.rcfg.get("observation", {})
            self._fut_offsets = [float(x) for x in ocfg.get("future_offsets_s", [])]
            self._hist_offsets = [float(x) for x in ocfg.get("history_offsets_s", [])]
            self._hist_on = bool(ocfg.get("history", False))
            self._nfeat = int(ocfg.get("history_features", 4))
            obs_n += self._nfeat * len(self._fut_offsets)
            if self._hist_on:
                obs_n += self._nfeat * len(self._hist_offsets)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(act_n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_n,), dtype=np.float32)

        self._batt_cap_J = 0.53 * 3600.0
        self._batt_E = 0.5 * self._batt_cap_J
        self._panel_peak = 0.78 * 0.85
        self._q_prev = np.array([1.0, 0.0, 0.0, 0.0])
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._mass = 0.625
        period = 2 * np.pi * np.sqrt((6371e3 + 400e3) ** 3 / 3.986004418e14)
        n_orb = 160 if self._v11 else (120 if self._v10 else 40)
        n_orb = float(self._orbit.get("episode_orbits", n_orb))
        self._max_steps = int(n_orb * period / self._advisor_s)
        self._weather_jumps = []
        self._srp_flashes = []
        self._brownout = False
        self._brownout_steps = 0
        self._gps_good = None
        self._gps_t = 0.0
        self._hold_left = 0
        self._v4 = self._v4plus
        self._srp_now = 1.0
        self._sun_search = False
        self._gps_W = 0.205       # GNSS receiver, powered while sunlit
        self._tx_fn = None        # onboard (vis, cos) for the transmitter gate; None = truth geometry
        self._term_km = 250.0

    def _confirm_gravity(self, requested_path):
        loaded = bool(self.sim.gravity_loaded())
        requested = str(requested_path or "")
        planted = str(self.sim.params().ggm_path or "")
        degree = int(self.sim.params().sh_degree)
        if requested:
            if not loaded:
                raise RuntimeError(
                    f"GGM failed to load from {requested!r} "
                    f"(plant ggm_path={planted!r}, gravity_loaded=False); "
                    "refusing silent two-body+J2+J3 fallback."
                )
            return
        if loaded:
            raise RuntimeError(
                "gravity_loaded() is True with an empty resolve_ggm() path"
            )
        if degree < 2:
            return
        warnings.warn(
            "GGM file not found (resolve_ggm empty); plant uses the "
            "two-body+J2+J3 fallback. Training is not on GGM.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _confirm_wmm(self, requested_path):
        loaded = bool(self.sim.wmm_loaded())
        requested = str(requested_path or "")
        planted = str(self.sim.params().wmm_path or "")
        if requested:
            if not loaded:
                raise RuntimeError(
                    f"WMM failed to load from {requested!r} "
                    f"(plant wmm_path={planted!r}, wmm_loaded=False); "
                    "refusing silent dipole-field fallback."
                )
            return
        if loaded:
            raise RuntimeError(
                "wmm_loaded() is True with an empty resolve_wmm() path"
            )
        if not getattr(self, "_want_wmm", True):
            return
        warnings.warn(
            "WMM COF not found (resolve_wmm empty); plant uses the "
            "tilted-dipole fallback. Training is not on WMM.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _configure_bdot(self):
        b = cpp.BDot()
        self.sim.set_bdot(b)

    def _set_plant_dipoles(self, scale):
        s = np.clip(np.asarray(scale, float).reshape(-1), 0.0, 1.0)
        if s.size == 1:
            s = np.full(3, float(s[0]))
        self.sim.set_dipole_max(self._m_max * s)

    def _enter_detumble(self):
        self._set_plant_dipoles(np.ones(3))
        self.sim.set_mode("detumble")

    def _install_mtq(self, scale):
        scale = float(np.clip(scale, 0.15, 1.0))
        self._mtq_scale = scale
        ctrl = cpp.MRPFeedback()
        ctrl.kp = self._kp if self._v7 else MTQ_KP
        ctrl.kd = self._kd if self._v7 else MTQ_KD
        ctrl.max_body_rate = np.radians(MTQ_RATE_DPS * scale)
        ctrl.set_inertia_diag(self._I)
        ctrl.set_max_torque(self._tau_max * scale)
        self.sim.set_controller(ctrl)
        qctrl = cpp.QuaternionFeedback()
        qctrl.kp = ctrl.kp
        qctrl.kd = ctrl.kd
        qctrl.max_body_rate = ctrl.max_body_rate
        qctrl.set_inertia_diag(self._I)
        qctrl.set_max_torque(self._tau_max * scale)
        self.sim.set_quat_controller(qctrl)
        self._set_plant_dipoles(scale)

    def _install_mtq_axes(self, gates):
        g = np.clip(np.asarray(gates, float).reshape(3), self._gate_floor, 1.0)
        self._gates = g
        ctrl = cpp.MRPFeedback()
        ctrl.kp = self._kp
        ctrl.kd = self._kd
        ctrl.max_body_rate = np.radians(MTQ_RATE_DPS)
        ctrl.set_inertia_diag(self._I)
        ctrl.set_max_torque(self._tau_max * g)
        self.sim.set_controller(ctrl)
        qctrl = cpp.QuaternionFeedback()
        qctrl.kp = ctrl.kp
        qctrl.kd = ctrl.kd
        qctrl.max_body_rate = ctrl.max_body_rate
        qctrl.set_inertia_diag(self._I)
        qctrl.set_max_torque(self._tau_max * g)
        self.sim.set_quat_controller(qctrl)
        self._set_plant_dipoles(g)

    def _obs(self, st, extra):
        r = np.asarray(st["r"], float)
        v = np.asarray(st["v"], float)
        w = np.asarray(st["omega"], float)
        bb = np.asarray(st.get("B_B", extra.get("B_B", np.zeros(3))), float)
        bn = np.linalg.norm(bb)
        bhat = bb / bn if bn > 1e-18 else np.zeros(3)
        c_bn = cpp.mrp_to_dcm(np.asarray(st["sigma"], float))
        vb = c_bn @ v
        vn = np.linalg.norm(vb)
        vhat = vb / vn if vn > 1e-9 else np.zeros(3)
        mu = cpp.MU_WGS
        h = np.cross(r, v)
        evec = np.cross(v, h) / mu - r / max(np.linalg.norm(r), 1.0)
        ev = np.linalg.norm(evec)
        nu = 0.0
        if ev > 1e-12:
            nu = np.arccos(np.clip(np.dot(evec, r) / (ev * np.linalg.norm(r)), -1, 1))
            if np.dot(r, v) < 0:
                nu = 2 * np.pi - nu
        g3 = np.asarray(extra.get("gs_dir_B", np.zeros(3)), float).reshape(-1)
        vis = float(extra.get("gs_visible", 0.0))
        gs = np.array([g3[0], g3[1], g3[2], vis]) if g3.size >= 3 else np.zeros(4)
        o = np.concatenate(
            [
                r / 1e3 / 7000.0,
                v / 1e3 / 8.0,
                w,
                bhat,
                [bn * 1e6 / 50.0],
                vhat,
                [np.sin(nu), np.cos(nu)],
                [float(self._mass)],
                [np.log10(max(extra.get("rho", 1e-16), 1e-16))],
                [extra.get("battery_soc", 1.0)],
                [extra.get("power_gen_norm", 0.0)],
                gs,
            ]
        )
        if self._v7:
            tau_c = np.asarray(extra.get("tau_ctrl_mean", np.zeros(3)), float)
            tau_d = np.asarray(extra.get("tau_dist_est", np.zeros(3)), float)
            ctrl_n = np.clip(tau_c / self._tau_max, -1.0, 1.0)
            dist_n = np.clip(tau_d / self._tau_max, -5.0, 5.0)
            gate_mean = float(extra.get("gate_mean", 1.0))
            fore = np.asarray(extra.get("foresight", np.zeros(2)), float).reshape(2)
            mode = self._obs_extra
            if mode == "none":
                block = np.zeros(9)
            elif mode == "ctrl":
                block = np.concatenate([ctrl_n, np.zeros(3), [gate_mean], np.zeros(2)])
            elif mode == "observer":
                block = np.concatenate([ctrl_n, dist_n, [gate_mean], np.zeros(2)])
            else:
                block = np.concatenate([ctrl_n, dist_n, [gate_mean], fore])
            o = np.concatenate([o, block])
        if self._v9:
            o = np.concatenate([o,
                                np.asarray(extra.get("future_block", np.zeros(
                                    self._nfeat * len(self._fut_offsets))), float)])
            if self._hist_on:
                o = np.concatenate([o,
                                    np.asarray(extra.get("history_block", np.zeros(
                                        self._nfeat * len(self._hist_offsets))), float)])
        if self._r6:
            o = np.concatenate([o, self._rod_on[:6].astype(float)])
        return o.astype(np.float32)

    def _snapshot_features(self, alt_km, soc, rho, omega_norm):
        return np.array([alt_km / 7000.0, soc,
                         np.log10(max(float(rho), 1e-16)), float(omega_norm)])

    def _future_block(self, r, v, sun_n, rho, alt_km, soc, gen_n, drag_N, vrel):
        bc = fastprop.bc_from_drag(drag_N, rho, vrel, self._mass)
        n_slots = self._nfeat * len(self._fut_offsets)
        self._fut_alt_600 = alt_km
        self._fut_soc_600 = soc
        try:
            samples = fastprop.propagate_samples(
                r, v, self._fut_offsets, dt_s=30.0, bc_inv=bc, rho0=rho,
                alt0_m=self._alt_q_km * 1e3, a_srp=self._a_srp, sun=sun_n)
        except Exception as exc:
            if not ArlamxV2Env._prop_warned:
                ArlamxV2Env._prop_warned = True
                warnings.warn(f"onboard propagation failed ({exc!r}); v9 future "
                              f"block zeroed for the rest of this process.",
                              RuntimeWarning, stacklevel=2)
            return np.zeros(n_slots)
        sma0 = fastprop.sma_m(r, v)
        p_gen_geo = self._illum_geo * self._panel_peak
        out, t_prev, soc_f = [], 0.0, float(soc)
        for (r1, v1), off in zip(samples, self._fut_offsets):
            alt1 = float(cpp.ecef_to_geodetic(np.asarray(r1, float))[2]) / 1e3
            sma1 = fastprop.sma_m(r1, v1)
            d_sma = ((sma0 - sma1) / 1e3) if (np.isfinite(sma0) and np.isfinite(sma1)) else 0.0
            sunlit = (sun_n is None or
                      cpp.eclipse_cylindrical(np.asarray(r1, float),
                                              np.asarray(sun_n, float),
                                              cpp.RE_WGS) > 0.5)
            gen = p_gen_geo if sunlit else 0.0
            load = 0.007 + (0.205 if sunlit else 0.0)
            soc_f = float(np.clip(
                soc_f + (gen - load) * (off - t_prev) / self._batt_cap_J, 0.0, 1.0))
            t_prev = off
            out.extend([(alt1 - alt_km) / 10.0, d_sma / 10.0, soc_f, off / 600.0])
            self._fut_alt_600, self._fut_soc_600 = alt1, soc_f
        return np.asarray(out, float)

    def _history_block(self):
        n = len(self._hist_offsets)
        blk = np.zeros(self._nfeat * n)
        step_s = self._advisor_s
        for i, off in enumerate(self._hist_offsets):
            frac = off / step_s
            lo = int(np.floor(frac))
            hi = int(np.ceil(frac))
            w = frac - lo
            newer = self._hist[-lo] if 0 < lo <= len(self._hist) else None
            older = self._hist[-hi] if 0 < hi <= len(self._hist) else None
            if lo == 0:
                newer = self._hist_now
            if older is None:
                continue
            if newer is None:
                newer = older
            blk[i * self._nfeat:(i + 1) * self._nfeat] = (
                (1.0 - w) * np.asarray(newer, float) + w * np.asarray(older, float))
        return blk

    def _update_power(self, sun_b, eclipse, gs_cos, gs_vis, torque_effort, brownout, tumble=False):
        sunlit = float(eclipse) > 0.5
        # Cells on both membrane faces (user, 2026-09-25): illumination |s_B,z|.
        self._illum_geo = 0.25 if tumble else abs(float(sun_b[2]))
        illum = self._illum_geo if sunlit else 0.0
        p_gen = self._panel_peak * illum
        if brownout:
            p_load = 0.007 + (self._mtq_peak_W if self._v8plus else 0.270) * torque_effort
        else:
            p_load = 0.007
            if sunlit:
                p_load += self._gps_W
            if gs_vis > 0.5 and gs_cos > 0.7 and sunlit:
                p_load += 0.400
            p_load += (self._mtq_peak_W if self._v8plus else 0.270) * torque_effort
        dt = self._advisor_s
        self._batt_E = min(self._batt_cap_J, max(0.0, self._batt_E + (p_gen - p_load) * dt))
        soc = self._batt_E / self._batt_cap_J if self._batt_cap_J > 0 else 1.0
        return soc, p_gen / max(self._panel_peak, 1e-9), soc <= 1e-6

    def _plan_jumps(self):
        self._weather_jumps = []
        self._srp_flashes = []
        if not self._v4:
            return
        if self._v10:
            n_j = int(self._rng.integers(2, 9))
            times = np.sort(self._rng.uniform(0, self._max_steps, size=n_j))
            for t in times:
                self._weather_jumps.append(
                    (float(t), float(self._rng.uniform(-150, 150)),
                     float(self._rng.uniform(-60, 150))))
        else:
            n_j = int(self._rng.integers(1, 5))
            times = np.sort(self._rng.uniform(0, self._max_steps, size=n_j))
            for t in times:
                self._weather_jumps.append(
                    (
                        float(t),
                        float(self._rng.uniform(-80, 80)),
                        float(self._rng.uniform(-20, 35)),
                    )
                )
        if self._v5plus:
            n_f = int(self._rng.integers(1, 4))
            ft = np.sort(self._rng.uniform(0, self._max_steps, size=n_f))
            for t in ft:
                self._srp_flashes.append((float(t), float(t) + 3.0, float(self._rng.uniform(2.0, 6.0))))

    def _apply_jumps(self):
        keep = []
        for t, df, da in self._weather_jumps:
            if self._step >= t:
                lo_f, hi_a = (5.0, 200.0) if self._v10 else (65.0, 80.0)
                self._f107 = float(np.clip(self._f107 + df, lo_f, 250.0))
                self._ap = float(np.clip(self._ap + da, 2.0, hi_a))
            else:
                keep.append((t, df, da))
        self._weather_jumps = keep

    def _pick(self, opt, key, lo, hi):
        v = self._orbit.get(key)
        if isinstance(v, (list, tuple)):
            lo, hi = float(v[0]), float(v[1])
        d = float(self._rng.uniform(lo, hi))
        if key in opt:
            return float(opt[key])
        return d if v is None or isinstance(v, (list, tuple)) else float(v)

    def _pick_fixed(self, opt, key, default):
        if key in opt:
            return float(opt[key])
        v = self._orbit.get(key)
        if isinstance(v, (list, tuple)):
            return float(self._rng.uniform(float(v[0]), float(v[1])))
        return float(default if v is None else v)

    def _pick_angle(self, opt, key, random=True):
        if key in opt:
            return float(np.radians(opt[key]))
        v = self._orbit.get(key)
        if isinstance(v, (list, tuple)):
            return float(np.radians(self._rng.uniform(float(v[0]), float(v[1]))))
        if v is not None:
            return float(np.radians(v))
        return float(self._rng.uniform(0, 2 * np.pi)) if random else 0.0

    # MSIS at geodetic height (Bowring) and Earth-fixed longitude (GMST rotation).
    def _cmd_q(self, q_bn, st):
        """Inertial target -> the plant's command frame (flow frames: q_BF = q_BN C_FN^T)."""
        frame = self.sim.params().cmd_frame
        if frame == "inertial":
            return q_bn
        r, v = np.asarray(st["r"], float), np.asarray(st["v"], float)
        we = np.array([0.0, 0.0, cpp.OMEGA_EARTH])
        e1 = v - np.cross(we, r); e1 /= np.linalg.norm(e1)
        h = np.cross(r, v); ref = h
        if frame == "flowS":
            sN = np.asarray(st["sun_N"], float); sp = sN - (sN @ e1) * e1
            if np.linalg.norm(sp) > 0.2:
                ref = sp if sp @ h >= 0 else -sp
        e3 = ref - (ref @ e1) * e1; e3 /= np.linalg.norm(e3)
        C_FN = np.vstack([e1, np.cross(e3, e1), e3])
        C_BN = np.asarray(cpp.mrp_to_dcm(np.asarray(cpp.quat_to_mrp(q_bn))))
        return np.asarray(cpp.dcm_to_quat(C_BN @ C_FN.T))

    def _atmo_at(self, r, jd):
        lat, lon, alt_m = cpp.ecef_to_geodetic(r)
        gmst = cpp.gmst_rad(jd)
        lon_ecef = (lon - gmst + np.pi) % (2 * np.pi) - np.pi
        chi = None
        if self._atmo_model == "constant":
            rho, temp, mb = physics_mod.constant_atmo(self._physics)
        else:
            try:
                want_chi = physics_mod.gsi_name(self._physics) == "cll"
                out_msis = query_msis(
                    alt_m / 1e3,
                    np.degrees(lat),
                    np.degrees(lon_ecef),
                    jd_to_datetime(jd),
                    self._f107,
                    self._ap,
                    version=self._msis_version,
                    return_species=want_chi,
                )
                if want_chi:
                    rho, temp, mb, chi = out_msis
                else:
                    rho, temp, mb = out_msis
            except Exception as exc:
                rho, temp, mb = 3e-12, 900.0, 2.656e-26
                chi = None
                if not ArlamxV2Env._msis_warned:
                    ArlamxV2Env._msis_warned = True
                    warnings.warn(f"MSIS query failed ({exc!r}); falling back to a "
                                  f"constant atmosphere rho=3e-12, T=900 K for the "
                                  f"rest of this process.", RuntimeWarning, stacklevel=2)
        if not (np.isfinite(rho) and np.isfinite(temp) and np.isfinite(mb)
                and 0.0 <= rho < 1e-8 and 150.0 < temp < 4000.0
                and 8e-28 < mb < 1e-25):
            rho, temp, mb = 1e-13, 600.0, 2.656e-26
            chi = None
        return rho, temp, mb, chi, alt_m / 1e3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opt = options or {}
        if "altitude_km" in opt and not (200.0 < float(opt["altitude_km"]) < 2000.0):
            raise ValueError("altitude_km is not a feasible LEO start")
        if self._v10:
            alt = self._pick(opt, "altitude_km", 300, 500)
            inc = self._pick(opt, "inc_deg", 20, 40)
            ecc = self._pick(opt, "ecc", 0.0, 0.01)
            self._f107 = self._pick(opt, "f107", 5, 250)
            self._ap = self._pick(opt, "ap", 2, 200)
            raan = self._pick_angle(opt, "raan_deg")
            argp = self._pick_angle(opt, "argp_deg")
            nu = self._pick_angle(opt, "nu_deg")
        elif self._v4:
            alt = self._pick(opt, "altitude_km", 300, 500)
            inc = self._pick(opt, "inc_deg", 20, 30)
            ecc = self._pick(opt, "ecc", 0.001, 0.01)
            self._f107 = self._pick(opt, "f107", 65, 250)
            self._ap = self._pick(opt, "ap", 2, 40)
            raan = self._pick_angle(opt, "raan_deg")
            argp = self._pick_angle(opt, "argp_deg")
            nu = self._pick_angle(opt, "nu_deg")
        else:
            alt = self._pick(opt, "altitude_km", 300, 500)
            inc = self._pick(opt, "inc_deg", 20, 30)
            ecc = self._pick_fixed(opt, "ecc", 0.0)
            self._f107 = self._pick(opt, "f107", 80, 200)
            self._ap = self._pick(opt, "ap", 2, 15)
            raan = self._pick_angle({}, "raan_deg", random=False)
            argp = self._pick_angle({}, "argp_deg", random=False)
            nu = self._pick_angle(opt, "nu_deg", random=False)
        if self._v8plus:
            mass = self._pick_fixed(opt, "mass_kg", self._veh.mass if self._r6
                                    else self._pcfg["vehicle"]["mass_kg"])
        else:
            mass = self._pick(opt, "mass_kg", 0.5, 0.75)
        self._mass = mass
        self.sim.set_mass(mass)
        r, v = coe_to_rv(cpp.RE_WGS + alt * 1e3, ecc, np.radians(inc), raan, argp, nu)
        sig = self._rng.normal(0, 0.1, size=3)
        sig = sig / max(np.linalg.norm(sig), 1e-9) * float(self._rng.uniform(0, 0.3))
        if "omega_dps" in opt:
            om = np.radians(np.asarray(opt["omega_dps"], float).reshape(3))
        else:
            w_lo, w_hi = self._orbit.get("omega_dps", (-5, 5))
            om = np.radians(self._rng.uniform(w_lo, w_hi, size=3))
        self._brownout = bool(opt.get("force_brownout", False))
        if self._brownout and "omega_dps" not in opt:
            om = np.radians(self._rng.uniform(3.0, 5.0, size=3))
        st = self.sim.reset(r, v, sig, om)
        self.sim.set_mode("point")
        soc0 = self._pick_fixed(opt, "soc", 0.5)
        self._batt_E = float(np.clip(soc0, 0.0, 1.0)) * self._batt_cap_J
        self._q_prev = np.array([1.0, 0.0, 0.0, 0.0])
        self._step = 0
        self._brownout_steps = 0
        self._gps_good = (np.asarray(st["r"], float).copy(), np.asarray(st["v"], float).copy())
        self._gps_t = float(st["t"])
        self._hold_left = 0
        self._mtq_scale = 1.0
        self._srp_now = 1.0
        self._sun_search = False
        self._illum_geo = 0.0
        self._a_srp = 0.0
        self._rod_on = np.ones(self._n_rod, bool)
        if self._r6:
            self.sim.set_rod_gates([True] * self._n_rod)
        self._alt_q_km = alt
        self.sim.set_srp_scale(1.0)
        if self._v11:
            dcfg = self.rcfg.get("damage", {})
            self._damage_plan = None
            self._damage_step = -1
            self._damage_summary = None
            self._damage_applied = False
            p_ep = float(opt.get("damage_prob", dcfg.get("prob", 0.6)))
            forced_kind = opt.get("damage_kind")
            if forced_kind or self._rng.random() < p_ep:
                self._damage_plan = dmg_plan(self._rng, kind=forced_kind)
                frac = opt.get("damage_step_frac")
                if frac is None:
                    frac = float(self._rng.uniform(
                        dcfg.get("step_frac_lo", 0.15),
                        dcfg.get("step_frac_hi", 0.70)))
                self._damage_step = int(float(frac) * self._max_steps)
            self._track_hist = []
        if self._v8plus:
            self.sensors.reset()
            self._B_B = np.zeros(3)
            self.kf.reset()
            self._soc_prev = None
            self._omega_prev_meas = self.sensors.read_gyro(
                np.asarray(st["omega"], float), self._advisor_s)
            self._dE_prev = None
            self._hist = []
            self._trend = None
            self._deleg_split = np.zeros(3)
            self._tau_env_filt = np.zeros(3)
            self._tau_env_raw = np.zeros(3)
            n_f = self._nfeat * len(self._fut_offsets) if self._v9 else 0
            self._fut_block = np.zeros(n_f)
            self._hist_block = (np.zeros(self._nfeat * len(self._hist_offsets))
                                if (self._v9 and self._hist_on) else None)
        if self._v5plus:
            self._install_mtq(1.0)
        if self._v7:
            self._install_mtq_axes(np.ones(3))
        if self._brownout:
            self._enter_detumble()
        self._plan_jumps()
        extra = {
            "rho": 1e-12,
            "battery_soc": 0.5,
            "power_gen_norm": 0.0,
            "gs_dir_B": np.zeros(3),
            "gs_visible": 0.0,
            "B_B": np.zeros(3),
        }
        return self._obs(st, extra), {}

    def step(self, action):
        raw = np.asarray(action, float).reshape(-1)
        invalid = raw.size < 4 or (not np.all(np.isfinite(raw[:4]))) or float(np.linalg.norm(raw[:4])) < 1e-6
        if invalid:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q = raw[:4] / np.linalg.norm(raw[:4])
            if q[0] < 0:
                q = -q
        tscale = 1.0
        if self.variant == "v6" and raw.size >= 5 and np.isfinite(raw[4]):
            tscale = 0.15 + 0.85 * 0.5 * (float(np.clip(raw[4], -1.0, 1.0)) + 1.0)
            self._install_mtq(tscale)
        self._deleg_split = np.zeros(3)
        if self._r6:
            a_r = raw[4:10] if raw.size >= 10 else np.ones(6)
            on = (~np.isfinite(a_r)) | (a_r > 0.0) | self._brownout
            self._rod_on = np.r_[np.asarray(on, bool), np.ones(self._n_rod - 6, bool)]
            self.sim.set_rod_gates([bool(x) for x in self._rod_on])
        if self._v7 and not self._brownout:
            a_g = raw[4:7] if raw.size >= 7 else np.ones(3)
            a_g = np.where(np.isfinite(a_g), np.clip(a_g, -1.0, 1.0), 1.0)
            if self._r6:
                self._deleg_split = 1.0 - veh_mod.axis_authority(self._rod_on, self._veh)
                gates = np.ones(3)
            elif self._v8plus and not self._deleg_live:
                gates = np.ones(3)
            # Delegation split s in [0, 1]; gate from eq. (1), docs/modules/14_reward_v8.md §2.1.
            elif self._v8plus:
                split = 0.5 * (a_g + 1.0)
                self._deleg_split = split
                gates = split_to_gates(split, self._gate_floor)
            else:
                gates = self._gate_floor + (1.0 - self._gate_floor) * 0.5 * (a_g + 1.0)
            self._install_mtq_axes(gates)
        cmd_ang = _cmd_angle_deg(q, self._q_prev)

        self._apply_jumps()
        if self._v11 and not self._damage_applied and self._step >= self._damage_step >= 0:
            A_dam, kept, summ = dmg_apply(self._n, self._A, self._c,
                                          self._damage_plan)
            self.sim.set_panels(self._n, A_dam, self._c)
            self._damage_applied = True
            self._damage_summary = summ
            self._damage_kept = kept
        st0 = self.sim.get_state()
        r0 = np.asarray(st0["r"], float)
        jd = EPOCH_JD + float(st0["t"]) / 86400.0
        gmst = cpp.gmst_rad(jd)
        rho, temp, mb, chi, self._alt_q_km = self._atmo_at(r0, jd)
        self.sim.set_atmosphere(float(rho), float(temp), float(mb), chi)
        if self._atmo_interp:
            r1, _ = fastprop.propagate(r0, np.asarray(st0["v"], float), self._advisor_s)
            e = self._atmo_at(np.asarray(r1, float), jd + self._advisor_s / 86400.0)
            self.sim.set_atmosphere_end(float(e[0]), float(e[1]), float(e[2]), e[3])

        recovered = False
        self._srp_now = 1.0
        alt_now = float(st0.get("altitude_km", (np.linalg.norm(r0) - cpp.RE_WGS) / 1e3))
        if self._v5plus:
            scale = 1.0
            for t0, t1, sc in self._srp_flashes:
                if t0 <= self._step < t1:
                    scale = sc
            if alt_now < 450.0:
                scale = 1.0
            self._srp_now = scale
            self.sim.set_srp_scale(scale)

        held = False
        if self.variant == "v6" and self._hold_left > 0 and not invalid:
            q = self._q_prev.copy()
            held = True
            self._hold_left -= 1

        rec_steps = 0
        self._sun_search = False
        tumble_gen = False
        bo_effort = tscale
        if (self.variant in ("v4b", "v5b", "v6", "v7") or self._v8plus) and self._brownout:
            omg = np.asarray(st0["omega"], float)
            omg_dps = float(np.degrees(np.linalg.norm(omg)))
            soc_now = self._batt_E / self._batt_cap_J
            if soc_now >= 0.15 and omg_dps <= 0.5:
                self._brownout = False
                recovered = True
                rec_steps = self._brownout_steps
                self.sim.set_mode("point")
            else:
                self._brownout_steps += 1
                rec_steps = self._brownout_steps
                if omg_dps > 2.0:
                    self._enter_detumble()
                    q = self._q_prev.copy()
                    tumble_gen = True
                    bo_effort = 0.5
                else:
                    self.sim.set_mode("point")
                    sun_n = np.asarray(st0.get("sun_N", np.array([1.0, 0.0, 0.0])), float)
                    q = self._cmd_q(quat_body_z_along(sun_n), st0)
                    self._sun_search = True
                    bo_effort = 0.35
                    if self._v5plus:
                        self._install_mtq(1.0)
                    if self._v7:
                        self._install_mtq_axes(np.ones(3))
        else:
            rec_steps = 0

        out = self.sim.step(q)
        c_bn = cpp.mrp_to_dcm(np.asarray(out["sigma"], float))
        gs_n, gs_vis, _el = nearest_visible(np.asarray(out["r"], float), gmst)
        gs_b = c_bn @ gs_n if gs_vis > 0.5 else np.zeros(3)
        z_n = c_bn.T @ np.array([0.0, 0.0, 1.0])
        gs_cos = float(np.dot(z_n, gs_n)) if gs_vis > 0.5 else 0.0
        gs_err = _axis_errors_deg(c_bn, gs_n) if gs_vis > 0.5 else np.zeros(3)
        tx_vis, tx_cos = self._tx_fn(out, c_bn, gmst) if self._tx_fn else (gs_vis, gs_cos)
        tx_on = tx_vis > 0.5 and tx_cos > 0.7 and float(out["eclipse"]) > 0.5 and not self._brownout

        tau_ctrl_mean = np.asarray(out.get("tau_ctrl_mean", np.zeros(3)), float)
        tau_cmd_mean = np.asarray(out.get("tau_cmd_mean", np.zeros(3)), float)
        tau_demand_mean = np.asarray(out.get("tau_demand_mean", tau_cmd_mean), float)
        tau_demand_abs = np.asarray(out.get("tau_demand_absmean",
                                            np.abs(tau_demand_mean)), float)
        if out.get("gyro_mean") is not None:
            gyro_mean = np.asarray(out["gyro_mean"], float)
        else:
            _om1 = np.asarray(out["omega"], float)
            gyro_mean = np.cross(_om1, self._I * _om1)
        m_mean = np.asarray(out.get("m_mean", np.zeros(3)), float)
        self._B_B = np.asarray(out.get("B_B", np.zeros(3)), float)
        if self._v8plus:
            if self._r6:
                d2 = np.asarray(out.get("rod_duty2_mean", np.zeros(self._n_rod)), float)
                p_w = float(self._veh.rod_pmax @ d2)
                effort = float(np.clip(p_w / self._mtq_peak_W, 0.0, 1.0))
                self._pdiag = {"rod_duty2": d2, "power_W": p_w, "model": "rods_i2r"}
            else:
                effort, self._pdiag = mtq_power.effort_from_dipole(m_mean, self._pcfg)
            self._duty_lin = float(np.clip(
                np.mean(np.abs(tau_ctrl_mean) / self._tau_max), 0.0, 1.0))
        elif self._v7:
            effort = float(np.clip(np.mean(np.abs(tau_ctrl_mean) / self._tau_max), 0.0, 1.0))
        elif self._brownout:
            effort = float(bo_effort)
        else:
            effort = float(np.clip(cmd_ang / 40.0, 0.0, 1.0))
            if self.variant == "v6":
                effort = float(np.clip(tscale * max(effort, 0.05), 0.0, 1.0))

        soc, gen_n, dep = self._update_power(
            out["sun_B"], out["eclipse"], tx_cos, tx_vis, effort, self._brownout, tumble=tumble_gen
        )
        if int(out.get("n_sunlit", 0)) > 0:
            self._a_srp = float(out.get("a_srp_sun", 0.0))
        if (self.variant in ("v4b", "v5b", "v6", "v7") or self._v8plus) and dep and not self._brownout:
            self._brownout = True
            self._brownout_steps = 0
            self._enter_detumble()

        if self.variant == "v6" and (not held) and (not self._brownout):
            rate = MTQ_RATE_DPS * self._mtq_scale
            slew_s = cmd_ang / max(rate, 1e-4)
            extra = int(np.ceil(slew_s / self._advisor_s)) - 1
            if extra > 0:
                self._hold_left = extra
                held = True
            elif cmd_ang < 8.0:
                self._hold_left = 1
                held = True

        used_prop = False
        if self.variant == "v6":
            t_now = float(out["t"])
            if self._rng.random() < 0.35 or self._gps_good is None:
                used_prop = True
                pr, pv = self._gps_good
                dt_g = max(self._advisor_s, t_now - self._gps_t)
                nint = max(1, int(round(dt_g / 30.0)))
                for _ in range(nint):
                    pr, pv = _two_body_rk4(pr, pv, dt_g / nint)
                obs_st = dict(out)
                obs_st["r"] = pr
                obs_st["v"] = pv
            else:
                self._gps_good = (np.asarray(out["r"], float).copy(), np.asarray(out["v"], float).copy())
                self._gps_t = t_now
                obs_st = out
        else:
            obs_st = out
        dE_srp = float(out["dE_actual"]) - float(out.get("dE_drag", 0.0)) - float(out.get("dE_lift", 0.0))

        tau_dist_est = np.zeros(3)
        foresight = np.zeros(2)
        if self._v7:
            om0 = np.asarray(st0["omega"], float)
            om1 = np.asarray(out["omega"], float)
            if self._v8plus:
                gc = self.sensors.cfg.gyro
                om0_m = self._omega_prev_meas
                om1_m = self.sensors.read_gyro(om1, self._advisor_s)
                sigma_w = (float(gc.arw_dps_rthz) * SENS_DEG
                           / np.sqrt(2.0 * self._advisor_s))
                tau_f, tau_raw = env_torque(
                    self.kf, om0_m, om1_m, self._advisor_s, tau_ctrl_mean,
                    sigma_w, gyro_mean=gyro_mean)
                self._omega_prev_meas = om1_m
                self._tau_env_filt = tau_f
                self._tau_env_raw = tau_raw
                tau_dist_est = tau_f
            else:
                dom = (om1 - om0) / self._advisor_s
                tau_dist_est = self._I * dom + gyro_mean - tau_ctrl_mean
            r_now = np.asarray(out["r"], float)
            v_now = np.asarray(out["v"], float)
            c_now = cpp.mrp_to_dcm(np.asarray(out["sigma"], float))
            s_hat_n = c_now.T @ np.asarray(out["sun_B"], float)
            sn = np.linalg.norm(s_hat_n)
            if sn > 1e-9:
                s_hat_n = s_hat_n / sn
                r_hat = r_now / max(np.linalg.norm(r_now), 1.0)
                v_hat = v_now / max(np.linalg.norm(v_now), 1.0)
                foresight = np.array([float(np.dot(r_hat, s_hat_n)),
                                      float(np.dot(v_hat, s_hat_n))])

        info_r = {
            "dE_actual": out["dE_actual"],
            "dE_baseline": out["dE_baseline"],
            "altitude_km": out["altitude_km"],
            "battery_soc": soc,
            "battery_depleted": dep,
            "power_gen_norm": gen_n,
            "Cd": out["Cd"],
            "gs_point_cos": gs_cos,
            "gs_visible": gs_vis,
            "gs_err_axes_deg": gs_err,
            "eclipse": float(out["eclipse"]),
            "shade_draw": 0.0 if float(out["eclipse"]) > 0.5 else 0.1,
            "q_new": q,
            "q_prev": self._q_prev,
            "omega": np.asarray(out["omega"], float),
            "cmd_angle_deg": cmd_ang,
            "action_invalid": invalid,
            "brownout_recovered": recovered,
            "brownout_active": self._brownout,
            "dE_lift": float(out.get("dE_lift", 0.0)),
            "dE_srp": dE_srp,
            "srp_scale": float(self._srp_now),
            "torque_effort": effort,
            "recover_steps": rec_steps,
            "used_propagator": used_prop,
            "held_inference": held,
        }
        if self._v9:
            alt_now = float(out["altitude_km"])
            r_now = np.asarray(out["r"], float)
            v_now = np.asarray(out["v"], float)
            om_now = float(np.linalg.norm(np.asarray(out["omega"], float)))
            we = np.array([0.0, 0.0, 7.2921150e-5])
            vrel_now = float(np.linalg.norm(v_now - np.cross(we, r_now)))
            c_now9 = cpp.mrp_to_dcm(np.asarray(out["sigma"], float))
            sun_n9 = c_now9.T @ np.asarray(out["sun_B"], float)
            self._hist_now = self._snapshot_features(alt_now, soc, rho, om_now)
            self._fut_block = self._future_block(
                r_now, v_now, sun_n9, rho, alt_now, soc, gen_n,
                float(out.get("drag_N", 0.0)), vrel_now)
            self._hist_block = self._history_block() if self._hist_on else None
            tc = self.rcfg.get("trend", {})
            h_ref = float(tc.get("h_ref_km", 2.0))
            alt_fut, soc_fut = self._fut_alt_600, self._fut_soc_600
            if self._hist_on and self._hist:
                back = min(len(self._hist),
                           max(1, int(np.ceil(self._hist_offsets[0] / self._advisor_s))))
                alt_past = float(self._hist[-back][0]) * 7000.0
                soc_past = float(self._hist[-back][1])
            else:
                alt_past, soc_past = alt_now, soc
            self._trend = {"d_alt_norm": (alt_fut - alt_past) / max(h_ref, 1e-9),
                           "d_soc": soc_fut - soc_past}
            self._hist.append(self._hist_now)
            if len(self._hist) > 64:
                self._hist.pop(0)

        if self._v8plus:
            tau_want = tau_demand_mean
            info_r["tau_want"] = tau_want
            info_r["tau_want_abs"] = tau_demand_abs
            info_r["tau_env_est"] = self._tau_env_filt
            info_r["deleg_split"] = self._deleg_split
            info_r["gates"] = self._gates.copy()
            info_r["tau_max"] = self._tau_max
            info_r["dE_prev"] = self._dE_prev
            info_r["soc_prev"] = self._soc_prev
            info_r["tau_ctrl_mean"] = tau_ctrl_mean
            info_r["torque_effort"] = self._duty_lin
            if self._v9:
                info_r["trend"] = self._trend
            if self._v11:
                terr = float(out.get("tracking_err_rad", 0.0))
                if not self._damage_applied:
                    self._track_hist.append(terr)
                    if len(self._track_hist) > 40:
                        self._track_hist.pop(0)
                info_r["tracking_err_rad"] = terr
                info_r["track_ref_rad"] = (float(np.median(self._track_hist))
                                           if self._track_hist else 0.0)
                info_r["damage_active"] = self._damage_applied
            rew, parts = compose_sc_v8(info_r, self.rcfg)
            self._dE_prev = float(out["dE_actual"])
            self._soc_prev = float(soc)
        elif self.variant == "v3":
            rew, parts = compose_sc_v3(info_r)
        elif self.variant == "v7":
            rew, parts = compose_sc_v7(info_r, w=self._reward_w)
        elif self.variant.startswith("v4"):
            rew, parts = compose_sc_v4(info_r, brownout=self.variant == "v4b")
        elif self.variant.startswith("v5"):
            rew, parts = compose_sc_v5(info_r, brownout=self.variant == "v5b")
        else:
            rew, parts = compose_sc_v6(info_r, brownout=True)
        self._q_prev = q.copy()
        self._step += 1
        extra = {
            "rho": rho,
            "battery_soc": soc,
            "power_gen_norm": gen_n,
            "gs_dir_B": gs_b,
            "gs_visible": gs_vis,
            "B_B": out["B_B"],
            "tau_ctrl_mean": tau_ctrl_mean,
            "tau_dist_est": tau_dist_est,
            "gate_mean": float(np.mean(self._gates)),
            "foresight": foresight,
        }
        if self._v9:
            extra["future_block"] = self._fut_block
            if self._hist_on:
                extra["history_block"] = self._hist_block
        obs = self._obs(obs_st, extra)
        term = bool(out["altitude_km"] < self._term_km)
        trunc = bool(self._step >= self._max_steps)
        omg = np.asarray(out["omega"], float)
        info = {
            "Cd": out["Cd"],
            "altitude_km": out["altitude_km"],
            "reward_parts": parts,
            "sma_m": out["sma_m"],
            "battery_soc": soc,
            "gs_visible": gs_vis,
            "brownout_active": self._brownout,
            "brownout_recovered": recovered,
            "omega_dps": float(np.degrees(np.linalg.norm(omg))),
            "recover_steps": rec_steps,
            "srp_scale": float(self._srp_now),
            "used_propagator": used_prop,
            "held_inference": held,
            "torque_effort": effort,
            "cmd_angle_deg": cmd_ang,
            "eclipse": float(out["eclipse"]),
            "power_gen_norm": gen_n,
            "dE_lift": float(out.get("dE_lift", 0.0)),
            "dE_srp": dE_srp,
            "dE_drag": float(out.get("dE_drag", 0.0)),
            "dE_actual": float(out["dE_actual"]),
            "dE_baseline": float(out["dE_baseline"]),
            "battery_depleted": dep,
            "gs_point_cos": gs_cos,
            "gs_err_axes_deg": gs_err,
            "tracking_err_rad": float(out.get("tracking_err_rad", 0.0)),
            "tx_on": tx_on,
            "dl_on": tx_on and gs_vis > 0.5 and gs_cos > 0.7,
        }
        if self._v7:
            info["gates"] = self._gates.copy()
            info["tau_ctrl_mean"] = tau_ctrl_mean
            info["tau_dist_est"] = tau_dist_est
            info["gyro_mean"] = gyro_mean
            info["tau_demand_mean"] = tau_demand_mean
            info["tau_aero_mean"] = np.asarray(out.get("tau_aero_mean", np.zeros(3)), float)
            info["tau_env_mean"] = np.asarray(out.get("tau_env_mean", np.zeros(3)), float)
        if self._v8plus:
            diag = info_r.get("_diag", {})
            info["deleg_B"] = float(diag.get("deleg_B", 0.0))
            info["deleg_P"] = float(diag.get("deleg_P", 0.0))
            info["deleg_S"] = float(diag.get("deleg_S", 0.0))
            info["deleg_boost"] = float(diag.get("deleg_boost", 1.0))
            info["deleg_align"] = np.asarray(diag.get("deleg_align", np.zeros(3)), float)
            info["deleg_split"] = self._deleg_split.copy()
            info["tau_want"] = np.asarray(info_r.get("tau_want", np.zeros(3)), float)
            info["tau_want_abs"] = np.asarray(info_r.get("tau_want_abs", np.zeros(3)), float)
            info["tau_env_filt"] = self._tau_env_filt.copy()
            info["tau_env_raw"] = self._tau_env_raw.copy()
            info["mtq_power_W"] = float(effort) * self._mtq_peak_W
            info["gate_floor"] = float(self._gate_floor)
            info["mass_kg"] = float(self._mass)
            bb = np.asarray(out.get("B_B", np.zeros(3)), float)
            short = float(out.get("tau_shortfall_frac", 0.0))
            info["tau_shortfall_frac"] = short
            info["B_B"] = bb
            if self._v9:
                info["trend"] = dict(self._trend) if self._trend else {}
            if self._v11:
                info["damage_active"] = self._damage_applied
                info["damage_step"] = self._damage_step
                info["damage_summary"] = self._damage_summary or {}
                info["tracking_err_rad"] = float(out.get("tracking_err_rad", 0.0))
        return obs, float(rew), term, trunc, info
