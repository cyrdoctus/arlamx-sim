"""Attitude helpers and fixed-attitude policies (min drag, fixed AoA, axis pointing)."""

from __future__ import annotations

import numpy as np


def _unit(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array(fallback, float)
    return v / n


def drag_extreme_axes(normals, areas):
    nrm = np.asarray(normals, float)
    area = np.asarray(areas, float)
    names = ("x", "y", "z")
    basis = np.eye(3)
    projected = {}
    for name, axis in zip(names, basis):
        projected[name] = float(np.sum(area * np.maximum(0.0, nrm @ axis)))
    min_axis = min(projected, key=projected.get)
    max_axis = max(projected, key=projected.get)
    return min_axis, max_axis, projected


def ram_area(normals, areas, dir_b):
    d = _unit(dir_b)
    return float(np.sum(np.asarray(areas, float) * np.maximum(0.0, np.asarray(normals, float) @ d)))


# Triad construction of C_BN (Schaub & Junkins 2018, ch. 3).
def dcm_bn_axis_along(target_n, body_axis, helper_n=None):
    target = _unit(target_n)
    helper = _unit(helper_n if helper_n is not None else (0.0, 0.0, 1.0), fallback=(1.0, 0.0, 0.0))
    if abs(float(np.dot(target, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    if body_axis == "z":
        b3 = target
        b1 = _unit(np.cross(helper, b3))
        b2 = np.cross(b3, b1)
    elif body_axis == "x":
        b1 = target
        b2 = _unit(np.cross(helper, b1))
        b3 = np.cross(b1, b2)
    elif body_axis == "y":
        b2 = target
        b3 = _unit(np.cross(helper, b2))
        b1 = np.cross(b2, b3)
    else:
        raise ValueError("body_axis must be x, y, or z")
    r_nb = np.column_stack([b1, b2, b3])
    return r_nb.T


# Sheppard's method, scalar first (Schaub & Junkins 2018, ch. 3; Diebel 2006).
def quat_from_dcm(c_bn):
    r = np.asarray(c_bn, float)
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q0 = 0.25 / s
        q1 = (r[1, 2] - r[2, 1]) * s
        q2 = (r[2, 0] - r[0, 2]) * s
        q3 = (r[0, 1] - r[1, 0]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q0 = (r[1, 2] - r[2, 1]) / s
        q1 = 0.25 * s
        q2 = (r[0, 1] + r[1, 0]) / s
        q3 = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q0 = (r[2, 0] - r[0, 2]) / s
        q1 = (r[0, 1] + r[1, 0]) / s
        q2 = 0.25 * s
        q3 = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q0 = (r[0, 1] - r[1, 0]) / s
        q1 = (r[0, 2] + r[2, 0]) / s
        q2 = (r[1, 2] + r[2, 1]) / s
        q3 = 0.25 * s
    q = np.array([q0, q1, q2, q3], float)
    n = float(np.linalg.norm(q))
    q = q / n
    return -q if q[0] < 0.0 else q


def quat_body_axis_along(dir_n, body_axis="z", helper_n=None):
    return quat_from_dcm(dcm_bn_axis_along(dir_n, body_axis, helper_n))


def quat_body_z_along(dir_n, helper_n=None):
    return quat_body_axis_along(dir_n, "z", helper_n)


class MinDragPolicy:
    name = "min_drag"

    def __init__(self, body_axis="y"):
        self.body_axis = body_axis

    def predict(self, obs, state):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        axis = state.get("min_drag_axis", self.body_axis)
        return quat_body_axis_along(v, axis, helper_n=np.cross(r, v)), None


class MaxDragPolicy:
    name = "max_drag"

    def __init__(self, body_axis="z"):
        self.body_axis = body_axis

    def predict(self, obs, state):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        axis = state.get("max_drag_axis", self.body_axis)
        return quat_body_axis_along(v, axis, helper_n=np.cross(r, v)), None


class NadirPolicy:
    name = "nadir"

    def predict(self, obs, state):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        return quat_body_z_along(-r, helper_n=np.cross(r, v)), None


class OffNadirPolicy:
    def __init__(self, off_deg=0.0, toward="velocity"):
        self.off_deg = float(off_deg)
        self.toward = toward
        self.name = f"off_nadir_{self.off_deg:.0f}"

    def predict(self, obs, state):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        nadir = _unit(-r)
        if self.toward == "velocity":
            tilt = _unit(v)
        else:
            tilt = _unit(np.cross(r, v))
        ang = np.radians(self.off_deg)
        d = _unit(np.cos(ang) * nadir + np.sin(ang) * tilt)
        return quat_body_z_along(d, helper_n=np.cross(r, v)), None


class GroundStationPolicy:
    name = "ground_station"

    def predict(self, obs, state):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        gs = state.get("gs_dir_N")
        if gs is None or float(state.get("gs_visible", 0.0)) < 0.5:
            return quat_body_z_along(-r, helper_n=np.cross(r, v)), {"mode": "nadir"}
        return quat_body_z_along(gs, helper_n=np.cross(r, v)), {"mode": "gs"}


OMEGA_EARTH = 7.2921150e-5


class FixedAoAPolicy:

    def __init__(self, alpha_deg, min_axis="y", max_axis="z", lift_toward="normal", corotating=True):
        if min_axis == max_axis:
            raise ValueError("min_axis and max_axis must differ")
        if lift_toward not in ("normal", "anti_normal", "zenith", "nadir"):
            raise ValueError("lift_toward must be normal, anti_normal, zenith or nadir")
        self.alpha = float(np.radians(alpha_deg))
        self.min_axis = min_axis
        self.max_axis = max_axis
        self.lift_toward = lift_toward
        self.corotating = bool(corotating)
        self.name = f"aoa_{float(alpha_deg):.0f}_{lift_toward}"

    def wind_dir(self, r, v):
        r = np.asarray(r, float)
        v = np.asarray(v, float)
        if self.corotating:
            v = v - np.cross([0.0, 0.0, OMEGA_EARTH], r)
        return _unit(v)

    def dcm_bn(self, r, v):
        r = np.asarray(r, float)
        v = np.asarray(v, float)
        w = self.wind_dir(r, v)
        h = _unit(np.cross(r, v))
        if self.lift_toward == "normal":
            lift = h
        elif self.lift_toward == "anti_normal":
            lift = -h
        elif self.lift_toward == "zenith":
            lift = _unit(r)
        else:
            lift = -_unit(r)
        t = -lift
        t = _unit(t - float(np.dot(t, w)) * w)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        e_max_n = sa * w + ca * t
        e_min_n = ca * w - sa * t
        axes = {self.min_axis: e_min_n, self.max_axis: e_max_n}
        third = ({"x", "y", "z"} - {self.min_axis, self.max_axis}).pop()
        order = ("x", "y", "z")
        idx = order.index(third)
        nxt, prv = order[(idx + 1) % 3], order[(idx + 2) % 3]
        axes[third] = np.cross(axes[nxt], axes[prv])
        return np.vstack([axes["x"], axes["y"], axes["z"]])

    def predict(self, obs, state):
        r = np.asarray(state["r"], float)
        v = np.asarray(state["v"], float)
        return quat_from_dcm(self.dcm_bn(r, v)), None
