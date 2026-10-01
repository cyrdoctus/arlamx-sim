"""Sampling MPC advisor over the C++ plant."""

from __future__ import annotations

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import MinDragPolicy, quat_body_z_along

MU = 3.986004418e14
RE = 6378137.0


def _unit_q(q):
    q = np.asarray(q, float).reshape(-1)
    n = float(np.linalg.norm(q))
    if q.size != 4 or n < 1e-12 or not np.all(np.isfinite(q)):
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / n
    return -q if q[0] < 0.0 else q


def _qmul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def _q_angle_deg(a, b):
    d = abs(float(np.dot(_unit_q(a), _unit_q(b))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def _slerp_toward(q_from, q_to, max_angle_deg):
    qa = _unit_q(q_from)
    qb = _unit_q(q_to)
    ch = float(np.dot(qa, qb))
    if ch < 0.0:
        qb = -qb
        ch = -ch
    ch = min(1.0, ch)
    half = float(np.arccos(ch))
    full_deg = float(np.degrees(2.0 * half))
    if full_deg <= max_angle_deg + 1e-9:
        return qb
    frac = max_angle_deg / full_deg
    sh = float(np.sin(half))
    if sh < 1e-9:
        return qa.copy()
    a = np.sin((1.0 - frac) * half) / sh
    b = np.sin(frac * half) / sh
    return _unit_q(a * qa + b * qb)


# Two-body a = -mu r/|r|^3 (Vallado 2013, ch. 1), classical RK4.
def _two_body_rk4(r, v, dt):
    def acc(rr):
        n = float(np.linalg.norm(rr))
        return -MU * rr / (n**3)

    k1v, k1r = acc(r), v
    k2v, k2r = acc(r + 0.5 * dt * k1r), v + 0.5 * dt * k1v
    k3v, k3r = acc(r + 0.5 * dt * k2r), v + 0.5 * dt * k2v
    k4v, k4r = acc(r + dt * k3r), v + dt * k3v
    rn = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
    vn = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return rn, vn


# Cylindrical shadow (Vallado 2013, ch. 5, eclipse geometry).
def _sunlit(r, sun):
    along = float(np.dot(r, sun))
    if along >= 0.0:
        return True
    perp = r - along * sun
    return float(np.linalg.norm(perp)) >= RE


def _gs_tier(err_deg, w, wpen):
    if err_deg <= 2.0:
        return w
    if err_deg <= 5.0:
        return 0.5 * w
    if err_deg <= 10.0:
        return 0.25 * w
    return -wpen


def _quat_to_dcm(q):
    q0, q1, q2, q3 = _unit_q(q)
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 + q0 * q1)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ]
    )


class SamplingMpcPolicy:
    name = "mpc"
    DEFAULT_W = {
        "power_band": 1.0,
        "power_low_pen": 4.0,
        "gs": 2.0,
        "gs_pen": 0.5,
        "gs_align": 1.0,
        "cd": 0.5,
        "slew": 0.2,
    }
    SOC_LO, SOC_HI, SOC_K = 0.4, 0.6, 6.0
    DIPOLE_QUANT_BITS = None

    def __init__(
        self,
        horizon_steps=6,
        n_random=8,
        max_cmd_deg=40.0,
        weights=None,
        seed=0,
        panels=None,
        use_sun_ephemeris=True,
        soc_k=None,
        soc_lo=None,
        soc_hi=None,
    ):
        self.horizon_steps = int(horizon_steps)
        self.n_random = int(n_random)
        self.max_cmd_deg = float(max_cmd_deg)
        self.w = dict(self.DEFAULT_W)
        if weights:
            self.w.update(weights)
        self._rng = np.random.default_rng(int(seed))
        self._min_drag = MinDragPolicy()
        self._q = np.array([1.0, 0.0, 0.0, 0.0])
        self.panels = panels
        self.use_sun_ephemeris = bool(use_sun_ephemeris)
        self.SOC_K = float(self.SOC_K if soc_k is None else soc_k)
        self.SOC_LO = float(self.SOC_LO if soc_lo is None else soc_lo)
        self.SOC_HI = float(self.SOC_HI if soc_hi is None else soc_hi)
        self.T_w = 300.0
        self.alpha_E = 0.93
        self.panel_peak_W = 0.78
        self.panel_eff = 0.85
        self.load_base_W = 0.007
        self.load_gps_W = 0.205
        self.load_tx_W = 0.400
        self.batt_cap_J = 0.53 * 3600.0
        self.tx_align_min = 0.7
        self.advisor_step_s = 300.0

    @classmethod
    def from_config(cls, cfg, seed=0, panels=None):
        cfg = dict(cfg or {})
        w = dict(cls.DEFAULT_W)
        for k, v in (cfg.get("weights") or {}).items():
            if k in w:
                w[k] = float(v)
        return cls(
            horizon_steps=int(cfg.get("horizon_steps", 6)),
            n_random=int(cfg.get("n_random", 8)),
            max_cmd_deg=float(cfg.get("max_cmd_deg", 40.0)),
            weights=w,
            seed=int(seed),
            panels=panels,
            use_sun_ephemeris=bool(cfg.get("use_sun_ephemeris", True)),
            soc_k=float(cfg.get("soc_k", cls.SOC_K)),
            soc_lo=float(cfg.get("soc_lo", cls.SOC_LO)),
            soc_hi=float(cfg.get("soc_hi", cls.SOC_HI)),
        )

    def to_config(self):
        return {
            "horizon_steps": self.horizon_steps,
            "n_random": self.n_random,
            "max_cmd_deg": self.max_cmd_deg,
            "use_sun_ephemeris": self.use_sun_ephemeris,
            "soc_lo": self.SOC_LO,
            "soc_hi": self.SOC_HI,
            "soc_k": self.SOC_K,
            "advisor_step_s": self.advisor_step_s,
            "weights": dict(self.w),
        }

    def reset(self):
        self._q = np.array([1.0, 0.0, 0.0, 0.0])

    def _candidates(self, state, sun_n):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        h = np.cross(r, v)
        out = [("hold", self._q.copy())]
        if sun_n is not None and self.use_sun_ephemeris:
            out.append(("sun", quat_body_z_along(sun_n, helper_n=h)))
        gs = state.get("gs_dir_N")
        if gs is None:
            C = state.get("C_BN")
            gsb = state.get("gs_dir_B")
            if C is not None and gsb is not None and float(state.get("gs_visible", 0.0)) > 0.5:
                gs = np.asarray(C, float).T @ np.asarray(gsb, float)
        if gs is not None and float(state.get("gs_visible", 0.0)) > 0.5:
            out.append(("gs", quat_body_z_along(gs, helper_n=h)))
        qmd, _ = self._min_drag.predict(None, state)
        out.append(("min_drag", qmd))
        out.append(("nadir", quat_body_z_along(-r, helper_n=h)))
        for i in range(self.n_random):
            ax = self._rng.normal(size=3)
            an = float(np.linalg.norm(ax))
            ax = ax / an if an > 1e-12 else np.array([1.0, 0.0, 0.0])
            ang = np.radians(self._rng.uniform(2.0, 0.85 * self.max_cmd_deg))
            half = 0.5 * ang
            qd = np.concatenate(([np.cos(half)], np.sin(half) * ax))
            out.append((f"rand{i}", _unit_q(_qmul(qd, self._q))))
        return out

    # Sampling MPC: roll each candidate forward, score, keep the best (Rawlings, Mayne & Diehl 2017).
    def _score(self, q_cand, r0, v0, sun_n, soc0, rho, T, m_bar, gs0, gs_vis0):
        dt_adv = self.advisor_step_s
        n_int = max(1, int(round(dt_adv / 60.0)))
        dt_int = dt_adv / n_int
        C = _quat_to_dcm(q_cand)
        z_n = C.T @ np.array([0.0, 0.0, 1.0])
        score = 0.0
        r, v = r0.copy(), v0.copy()
        soc = soc0
        name_used = None
        for _ in range(self.horizon_steps):
            for _ in range(n_int):
                r, v = _two_body_rk4(r, v, dt_int)
            lit = sun_n is not None and _sunlit(r, sun_n)
            illum = 0.0
            if lit:
                illum = abs(float(np.dot(z_n, sun_n)))
            p_gen = self.panel_peak_W * self.panel_eff * illum
            gs_err = None
            cos_align = 0.0
            if gs_vis0 > 0.5 and gs0 is not None:
                gn = float(np.linalg.norm(gs0))
                if gn > 1e-12:
                    dhat = gs0 / gn
                    cos_align = float(np.clip(np.dot(z_n, dhat), -1.0, 1.0))
                    gs_err = float(np.degrees(np.arccos(cos_align)))
            downlink = gs_err is not None and cos_align > self.tx_align_min and lit
            p_load = self.load_base_W + (self.load_gps_W if lit else 0.0)
            if downlink:
                p_load += self.load_tx_W
            soc = min(1.0, max(0.0, soc + (p_gen - p_load) * dt_adv / max(self.batt_cap_J, 1.0)))
            if soc < self.SOC_LO:
                score -= self.w["power_low_pen"] * (self.SOC_LO - soc) / self.SOC_LO
            elif soc <= self.SOC_HI:
                score += self.w["power_band"]
            else:
                score += self.w["power_band"] * float(np.exp(-self.SOC_K * (soc - self.SOC_HI)))
            if gs_err is not None and lit:
                score += _gs_tier(gs_err, self.w["gs"], self.w["gs_pen"])
                score += self.w["gs_align"] * max(0.0, cos_align)
            if self.panels is not None and rho is not None:
                v_gas = -(C @ v)
                cf = cpp.coefficients_only(
                    self.panels[0],
                    self.panels[1],
                    self.panels[2],
                    v_gas,
                    rho,
                    T,
                    m_bar,
                    self.T_w,
                    self.alpha_E,
                    True,
                )
                score -= self.w["cd"] * float(cf["Cd"])
        score -= self.w["slew"] * _q_angle_deg(q_cand, self._q) / self.max_cmd_deg
        return score, name_used

    def predict(self, obs, state):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        sun = state.get("sun_N")
        if sun is not None:
            sun = np.asarray(sun, float)
            sn = float(np.linalg.norm(sun))
            sun = sun / sn if sn > 1e-12 else None
        soc = float(state.get("battery_soc", 0.5))
        rho = state.get("rho")
        T = float(state.get("T", 900.0))
        mb = float(state.get("m_bar", 2.656e-26))
        gs = state.get("gs_dir_N")
        gs_vis = float(state.get("gs_visible", 0.0))
        best_s, best_q, best_n = -1e300, self._q, "hold"
        for name, q in self._candidates(state, sun):
            q = _slerp_toward(self._q, q, self.max_cmd_deg)
            s, _ = self._score(q, r, v, sun, soc, rho, T, mb, gs, gs_vis)
            if s > best_s:
                best_s, best_q, best_n = s, q, name
        self._q = _unit_q(best_q)
        return self._q, {"score": best_s, "name": self.name, "pick": best_n}
