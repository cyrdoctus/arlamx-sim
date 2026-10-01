"""Flow-frame sampling MPC for SC_v3_6mq: actuator-limited attitude rollout, Cd look-up, both-face power, data buffer, rod gates."""
from __future__ import annotations

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.mpc import _q_angle_deg, _quat_to_dcm, _slerp_toward, _unit_q
from arlamx_v2.advisors.stations import nearest_visible
from arlamx_v2.env import EPOCH_JD
from arlamx_v2.nav import fly
from arlamx_v2.sc6mq import flow_dcm, q_bf_from

P0 = dict(H=6, n_rand=8, rand_deg=35.0, slew_dps=0.06, gamma=0.95, w_drag=1.0, w_dl=3.0, w_band=0.1, w_low=1.5,
          w_att=0.6, w_slew=0.3, w_rod=0.2, rod_W=0.06, gate_deg=6.0, soc_tx=0.3, low_thr=0.25,
          s_alt=1.0, s_buf=0.5, s_dl_alt=0.5, plans=(1, 6))


def fib_dirs(n):
    """Fibonacci sphere (Gonzalez 2010, Math. Geosci. 42)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    t = np.pi * (1.0 + 5 ** 0.5) * i
    s = np.sqrt(1.0 - z * z)
    return np.c_[s * np.cos(t), s * np.sin(t), z]


def triad(a_b, s_b, a_f, s_f):
    """C_BF with body a_b on frame a_f and s_b as close as possible to s_f (Shuster & Oh 1981, TRIAD)."""
    def basis(a, s):
        a = a / np.linalg.norm(a)
        t = s - (s @ a) * a
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(a, [0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0.0, 0.0])
        t /= np.linalg.norm(t)
        return np.c_[a, t, np.cross(a, t)]
    return basis(np.asarray(a_b, float), np.asarray(s_b, float)) @ basis(np.asarray(a_f, float), np.asarray(s_f, float)).T


def rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


class FlowMpc:
    """Receding-horizon plan search: plan = candidate q_BF for m steps, then min drag; eclipse steps forced to min drag."""

    def __init__(self, panels, ram_b, cfg, params=None, seed=0, advisor_s=300.0, n_lut=2048):
        self.p = dict(P0, **(params or {}))
        self.cfg = cfg
        self.dt = float(advisor_s)
        self.rng = np.random.default_rng(seed)
        self.C_md = _quat_to_dcm(q_bf_from(ram_b))
        self.q_md = _unit_q(q_bf_from(ram_b))
        self.q_cmd = self.q_md.copy()
        self.lut_u = fib_dirs(n_lut)
        n, A, c = panels
        self.lut_cd = np.array([cpp.coefficients_only(n, A, c, u * 7600.0, 3e-12, 900.0, 2.656e-26, 300.0, 0.93, True)["Cd"]
                                for u in self.lut_u])
        self.cd_md = self.cd(self.C_md[None])[0]
        self.last = {}

    def cd(self, C):
        """Cd from the look-up table at the gas direction in body axes, g_B = C_BF (-e1)."""
        g = -np.asarray(C)[:, :, 0]
        return self.lut_cd[np.argmax(g @ self.lut_u.T, axis=1)]

    def weights(self, h_km, b):
        p, u = self.p, float(np.clip((450.0 - h_km) / 150.0, 0.0, 1.0))
        return (p["w_drag"] * (1.0 + p["s_alt"] * u),
                p["w_dl"] * max(0.0, 1.0 + p["s_buf"] * (2.0 * b - 1.0)) * max(0.0, 1.0 - p["s_dl_alt"] * u))

    def horizon(self, s):
        """Orbit, Sun and station geometry for k = 1..H in the flow frame (two-body, onboard r, v)."""
        r, v = np.asarray(s["r"], float), np.asarray(s["v"], float)
        sun = np.asarray(s["sun_N"], float)
        sun = sun / np.linalg.norm(sun)
        out = []
        for k in range(1, int(self.p["H"]) + 1):
            r, v = fly(r, v, self.dt, 0.0, 60.0)
            t = float(s["t"]) + k * self.dt
            C_FN = flow_dcm(r, v, sun if self.cfg.get("cmd_frame") == "flowS" else None)
            along = r @ sun
            lit = along >= 0.0 or np.linalg.norm(r - along * sun) >= cpp.RE_WGS
            d, vis, _ = nearest_visible(r, cpp.gmst_rad(EPOCH_JD + t / 86400.0))
            out.append(dict(lit=bool(lit), sF=C_FN @ sun, gF=C_FN @ d if vis > 0.5 else None))
        return out

    def candidates(self, geo, q_now):
        C_md, ram, z = self.C_md, self.C_md[:, 0], np.array([0.0, 0.0, 1.0])
        c = [("min_drag", self.q_md), ("hold", self.q_cmd)]
        gk = next((g["gF"] for g in geo if g["gF"] is not None and g["lit"]), None)
        if gk is not None:
            qg = _unit_q(cpp.dcm_to_quat(triad(z, ram, gk, [1.0, 0.0, 0.0])))
            c += [("gs", qg), ("gs_half", _slerp_toward(self.q_md, qg, 0.5 * _q_angle_deg(self.q_md, qg)))]
        sl = next((g["sF"] for g in geo if g["lit"]), None)
        if sl is not None:
            c.append(("sun", _unit_q(cpp.dcm_to_quat(triad(z, ram, sl, [1.0, 0.0, 0.0])))))
        for ax in np.eye(3):
            for a in (-45.0, -20.0, 20.0, 45.0):
                c.append(("grid", _unit_q(cpp.dcm_to_quat(C_md @ rot(ax, np.radians(a)).T))))
        base = [q for n, q in c if n in ("hold", "gs")] or [self.q_md]
        for i in range(int(self.p["n_rand"])):
            ax = self.rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            dq = np.r_[np.cos(0.5 * np.radians(self.rng.uniform(3.0, self.p["rand_deg"]))), 0, 0, 0]
            dq[1:] = ax * np.sqrt(1.0 - dq[0] ** 2)
            c.append(("rand", _unit_q(cpp.quat_mul(dq, base[i % len(base)]))))
        return c

    def rollout(self, q_c, m, geo, q_now, s, w_drag, w_dl):
        p, dt = self.p, self.dt
        dmin = dt / 60.0
        soc, buf = float(s["soc"]), float(s["buf"]) * float(s["cap"])
        q, J, disc = q_now, 0.0, 1.0
        J -= p["w_slew"] * float(_q_angle_deg(q_c, self.q_cmd) > 2.0)
        Cs, errs, moves, gens, txs = [], [], [], [], []
        for k, g in enumerate(geo):
            tgt = q_c if (k < m and g["lit"]) else self.q_md
            q1 = _slerp_toward(q, tgt, p["slew_dps"] * dt)
            moves.append(_q_angle_deg(q, q1))
            q = q1
            C = _quat_to_dcm(q)
            Cs.append(C)
            errs.append(_q_angle_deg(q, tgt))
            gens.append(0.663 * abs(float(C[2] @ g["sF"])) if g["lit"] else 0.0)
            txs.append(g["gF"] is not None and g["lit"] and float(C[2] @ g["gF"]) > 0.7)
        cd = self.cd(np.array(Cs))
        for k, g in enumerate(geo):
            prod = p["rod_W"] * min(1.0, moves[k] / max(p["slew_dps"] * dt, 1e-9))
            tx = txs[k] and soc > p["soc_tx"]
            load = 0.007 + (float(s["gps_W"]) if g["lit"] else 0.0) + (0.4 if tx else 0.0) + prod
            soc = min(1.0, max(0.0, soc + (gens[k] - load) * dt / (0.53 * 3600.0)))
            buf += float(s["gen_min_d"]) * dmin / 1440.0
            sent = min(buf, dmin) if tx else 0.0
            buf -= sent
            r = -w_drag * (cd[k] / self.cd_md - 1.0) + w_dl * sent / dmin
            r += p["w_band"] * float(0.4 <= soc <= 0.6) - p["w_low"] * max(0.0, p["low_thr"] - soc) / p["low_thr"]
            r -= p["w_att"] * min(1.0, errs[k] / 30.0) + p["w_rod"] * prod / 0.05
            J += disc * r
            disc *= p["gamma"]
        return J

    def act(self, s):
        """s: onboard r, v, t, C_BN, sun_N, soc, buf (fraction), cap, gen_min_d, gps_W, eclipse, h_km -> action [q_BF, gates]."""
        r, v = np.asarray(s["r"], float), np.asarray(s["v"], float)
        sun = np.asarray(s["sun_N"], float)
        C_FN = flow_dcm(r, v, sun if self.cfg.get("cmd_frame") == "flowS" else None)
        q_now = _unit_q(cpp.dcm_to_quat(np.asarray(s["C_BN"]) @ C_FN.T))
        geo = self.horizon(s)
        w_drag, w_dl = self.weights(float(s["h_km"]), float(s["buf"]))
        best = (-1e300, self.q_md, "min_drag", 1)
        for name, q in self.candidates(geo, q_now):
            for m in self.p["plans"]:
                J = self.rollout(q, int(m), geo, q_now, s, w_drag, w_dl)
                if J > best[0]:
                    best = (J, q, name, m)
        self.q_cmd = best[1]
        err = _q_angle_deg(q_now, self.q_cmd)
        on = bool(s["eclipse"]) or err > self.p["gate_deg"]
        self.last = dict(pick=best[2], m=best[3], J=best[0], err=err, rods=on)
        return np.r_[self.q_cmd, np.full(6, 1.0 if on else -1.0)].astype(np.float32)
