"""Power-aware heuristic advisor bank."""

from __future__ import annotations

import numpy as np

from arlamx_v2.advisors.attitudes import MinDragPolicy, quat_body_z_along


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


def _rotate_body(q_bn, axis, ang):
    ax = np.asarray(axis, float)
    an = float(np.linalg.norm(ax))
    ax = ax / an if an > 1e-12 else np.array([1.0, 0.0, 0.0])
    h = 0.5 * float(ang)
    qd = np.concatenate(([np.cos(h)], np.sin(h) * ax))
    return _unit_q(_qmul(qd, q_bn))


class HeuristicPowerPolicy:
    name = "heuristic_power"
    SWEEP, CLIMB, HOLD, ECLIPSE = "sweep", "climb", "hold", "eclipse"

    def __init__(
        self,
        soc_thresholds=(0.60, 0.50, 0.40, 0.30, 0.20),
        sweep_step_deg=15.0,
        climb_step_deg=6.0,
        sense_floor=0.05,
        gen_target=0.70,
        eclipse_hold_steps=6,
        label=None,
        rich_mode="min_drag",
        allow_gs=True,
    ):
        self.soc_thresholds = [float(t) for t in soc_thresholds]
        if sorted(self.soc_thresholds, reverse=True) != self.soc_thresholds:
            raise ValueError("soc_thresholds must be strictly descending")
        self.sweep_step_deg = float(sweep_step_deg)
        self.climb_step_deg = float(climb_step_deg)
        self.sense_floor = float(sense_floor)
        self.gen_target = float(gen_target)
        self.eclipse_hold_steps = int(eclipse_hold_steps)
        if rich_mode not in ("min_drag", "nadir", "hold_sun"):
            raise ValueError("rich_mode must be min_drag, nadir, or hold_sun")
        self.rich_mode = rich_mode
        self.allow_gs = bool(allow_gs)
        self.name = label or "heuristic_power"
        self._min_drag = MinDragPolicy()
        self.reset()

    def reset(self):
        self._q = np.array([1.0, 0.0, 0.0, 0.0])
        self._prev_gen = 0.0
        self._phase = self.SWEEP
        self._sweep_axis = 1
        self._sweep_deg = 0.0
        self._sweep_axes = 0
        self._climb_idx = 0
        self._climb_ddeg = self.climb_step_deg
        self._climb_pending = False
        self._climb_fails = 0
        self._ecl_wait = 0
        self._best_gen = 0.0
        self._q_sun = None

    def _search_gains(self, soc):
        t = self.soc_thresholds
        if soc >= t[2]:
            return self.sweep_step_deg, self.climb_step_deg, self.gen_target
        if soc >= t[3]:
            return 1.5 * self.sweep_step_deg, 1.5 * self.climb_step_deg, self.gen_target
        if soc >= t[4]:
            return 2.0 * self.sweep_step_deg, 2.0 * self.climb_step_deg, self.gen_target
        return 2.0 * self.sweep_step_deg, 2.0 * self.climb_step_deg, self.sense_floor

    def _sun_search(self, gen, sweep_deg, climb_deg, gen_tgt):
        if gen >= self.sense_floor:
            self._best_gen = max(self._best_gen, gen)
            if gen >= 0.5:
                self._q_sun = self._q.copy()
        if self._phase == self.ECLIPSE:
            self._ecl_wait -= 1
            if gen >= self.sense_floor:
                self._phase = self.CLIMB
                self._climb_pending = False
            elif self._ecl_wait <= 0:
                self._phase = self.SWEEP
                self._sweep_deg = 0.0
                self._sweep_axes = 0
            return
        if self._phase == self.SWEEP:
            if gen >= self.sense_floor:
                self._phase = self.CLIMB
                self._climb_pending = False
                self._climb_fails = 0
                self._climb_ddeg = climb_deg
                return
            self._q = _rotate_body(self._q, np.eye(3)[self._sweep_axis], np.radians(sweep_deg))
            self._sweep_deg += sweep_deg
            if self._sweep_deg >= 360.0:
                self._sweep_deg = 0.0
                self._sweep_axes += 1
                self._sweep_axis = 0 if self._sweep_axis == 1 else 1
                if self._sweep_axes >= 2:
                    self._phase = self.ECLIPSE
                    self._ecl_wait = self.eclipse_hold_steps
                    self._sweep_axes = 0
            return
        if self._phase == self.CLIMB:
            if gen < 1e-6:
                self._phase = self.ECLIPSE
                self._ecl_wait = self.eclipse_hold_steps
                return
            cycle = ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0))
            if self._climb_pending:
                if gen > self._prev_gen + 1e-3:
                    self._climb_fails = 0
                else:
                    ax, sg = cycle[self._climb_idx]
                    self._q = _rotate_body(
                        self._q, np.eye(3)[ax], -sg * np.radians(self._climb_ddeg)
                    )
                    self._climb_idx = (self._climb_idx + 1) % 4
                    self._climb_fails += 1
                    if self._climb_fails >= 4:
                        self._climb_ddeg *= 0.5
                        self._climb_fails = 0
                        if self._climb_ddeg < 1.0:
                            self._phase = self.HOLD
                            self._climb_pending = False
                            return
                self._climb_pending = False
            if gen >= gen_tgt:
                self._phase = self.HOLD
                return
            ax, sg = cycle[self._climb_idx]
            self._q = _rotate_body(self._q, np.eye(3)[ax], sg * np.radians(self._climb_ddeg))
            self._climb_pending = True
            return
        if self._phase == self.HOLD:
            if gen < 1e-6:
                self._phase = self.ECLIPSE
                self._ecl_wait = self.eclipse_hold_steps
            elif gen < max(self.sense_floor, 0.5 * self._best_gen):
                self._phase = self.CLIMB
                self._climb_pending = False
                self._climb_fails = 0
                self._climb_ddeg = climb_deg

    def _gs_quat(self, state):
        if not self.allow_gs:
            return None
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        gs = state.get("gs_dir_N")
        if gs is not None and float(state.get("gs_visible", 0.0)) >= 0.5:
            return quat_body_z_along(np.asarray(gs, float), helper_n=np.cross(r, v))
        gsb = state.get("gs_dir_B")
        C = state.get("C_BN")
        if gsb is None or C is None or float(state.get("gs_visible", 0.0)) < 0.5:
            return None
        gs_n = np.asarray(C, float).T @ np.asarray(gsb, float)
        return quat_body_z_along(gs_n, helper_n=np.cross(r, v))

    def _rich_quat(self, obs, state):
        if self.rich_mode == "nadir":
            r = np.asarray(state["r"], float)
            v = np.asarray(state["v"], float)
            return quat_body_z_along(-r, helper_n=np.cross(r, v))
        if self.rich_mode == "hold_sun" and self._q_sun is not None:
            return self._q_sun.copy()
        q, _ = self._min_drag.predict(obs, state)
        return q

    def predict(self, obs, state):
        soc = float(state.get("battery_soc", 0.5))
        gen = float(state.get("power_gen_norm", 0.0))
        t = self.soc_thresholds
        sw, cl, tgt = self._search_gains(soc)
        if soc >= t[0]:
            qg = self._gs_quat(state)
            self._q = qg if qg is not None else self._rich_quat(obs, state)
        elif soc >= t[1]:
            qg = self._gs_quat(state)
            if qg is not None:
                self._q = qg
            elif self._q_sun is not None:
                self._q = self._q_sun.copy()
            else:
                self._sun_search(gen, sw, cl, tgt)
        else:
            self._sun_search(gen, sw, cl, tgt)
        self._prev_gen = gen
        return _unit_q(self._q), {"phase": self._phase, "name": self.name}


def _hp(**kw):
    return HeuristicPowerPolicy(**kw)


HEURISTIC_BANK = {
    "mission_v17": _hp(
        soc_thresholds=(0.60, 0.50, 0.40, 0.30, 0.20),
        sweep_step_deg=15.0,
        climb_step_deg=6.0,
        sense_floor=0.05,
        gen_target=0.70,
        eclipse_hold_steps=6,
        label="heuristic_mission",
    ),
    "conserve": _hp(
        soc_thresholds=(0.75, 0.65, 0.55, 0.45, 0.35),
        sweep_step_deg=12.0,
        climb_step_deg=5.0,
        sense_floor=0.08,
        gen_target=0.80,
        eclipse_hold_steps=10,
        label="heuristic_conserve",
    ),
    "eclipse_hold": _hp(
        soc_thresholds=(0.70, 0.60, 0.50, 0.40, 0.30),
        sweep_step_deg=10.0,
        climb_step_deg=4.0,
        sense_floor=0.10,
        gen_target=0.75,
        eclipse_hold_steps=16,
        label="heuristic_eclipse",
    ),
    "eclipse_deep": _hp(
        soc_thresholds=(0.80, 0.70, 0.55, 0.40, 0.25),
        sweep_step_deg=10.0,
        climb_step_deg=4.0,
        sense_floor=0.10,
        gen_target=0.80,
        eclipse_hold_steps=24,
        label="heuristic_eclipse_deep",
    ),
    "power_first": _hp(
        soc_thresholds=(0.85, 0.70, 0.55, 0.40, 0.25),
        sweep_step_deg=18.0,
        climb_step_deg=8.0,
        sense_floor=0.04,
        gen_target=0.85,
        eclipse_hold_steps=8,
        label="heuristic_power_first",
    ),
    "ground_first": _hp(
        soc_thresholds=(0.45, 0.38, 0.30, 0.22, 0.15),
        sweep_step_deg=15.0,
        climb_step_deg=6.0,
        sense_floor=0.05,
        gen_target=0.55,
        eclipse_hold_steps=4,
        label="heuristic_ground",
        rich_mode="nadir",
    ),
    "ground_rich_nadir": _hp(
        soc_thresholds=(0.65, 0.55, 0.45, 0.30, 0.20),
        sweep_step_deg=15.0,
        climb_step_deg=6.0,
        sense_floor=0.05,
        gen_target=0.70,
        eclipse_hold_steps=8,
        label="heuristic_ground_nadir",
        rich_mode="nadir",
    ),
    "low_threshold_search": _hp(
        soc_thresholds=(0.55, 0.45, 0.35, 0.25, 0.15),
        sweep_step_deg=20.0,
        climb_step_deg=8.0,
        sense_floor=0.03,
        gen_target=0.50,
        eclipse_hold_steps=6,
        label="heuristic_low_sense",
    ),
    "fine_search": _hp(
        soc_thresholds=(0.60, 0.50, 0.40, 0.30, 0.20),
        sweep_step_deg=8.0,
        climb_step_deg=3.0,
        sense_floor=0.05,
        gen_target=0.75,
        eclipse_hold_steps=8,
        label="heuristic_fine_search",
    ),
    "coarse_search": _hp(
        soc_thresholds=(0.60, 0.50, 0.40, 0.30, 0.20),
        sweep_step_deg=25.0,
        climb_step_deg=10.0,
        sense_floor=0.04,
        gen_target=0.60,
        eclipse_hold_steps=6,
        label="heuristic_coarse_search",
    ),
    "high_floor": _hp(
        soc_thresholds=(0.70, 0.60, 0.50, 0.35, 0.25),
        sweep_step_deg=15.0,
        climb_step_deg=6.0,
        sense_floor=0.20,
        gen_target=0.80,
        eclipse_hold_steps=10,
        label="heuristic_high_floor",
    ),
    "any_power_hold": _hp(
        soc_thresholds=(0.60, 0.50, 0.40, 0.30, 0.20),
        sweep_step_deg=15.0,
        climb_step_deg=6.0,
        sense_floor=0.02,
        gen_target=0.15,
        eclipse_hold_steps=8,
        label="heuristic_any_power",
    ),
    "search_only": _hp(
        soc_thresholds=(0.99, 0.98, 0.50, 0.30, 0.15),
        sweep_step_deg=15.0,
        climb_step_deg=6.0,
        sense_floor=0.05,
        gen_target=0.70,
        eclipse_hold_steps=6,
        label="heuristic_search_only",
        allow_gs=False,
        rich_mode="min_drag",
    ),
}
