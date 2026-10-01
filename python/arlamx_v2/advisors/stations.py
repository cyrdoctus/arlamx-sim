"""Ground stations and visibility."""

from __future__ import annotations

import numpy as np

from arlamx_v2 import cpp

RE = cpp.RE_WGS

STATIONS = (
    ("honolulu", 21.3069, -157.8583),
    ("mexico_city", 19.4326, -99.1332),
    ("mumbai", 19.0760, 72.8777),
    ("singapore", 1.3521, 103.8198),
    ("quito", -0.1807, -78.4678),
    ("darwin", -12.4634, 130.8456),
)


# Geodetic -> ECEF on WGS-84 (Vallado 2013, ch. 3, site coordinates).
def station_ecef(lat_deg, lon_deg, alt_m=0.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    n = RE / np.sqrt(1.0 - e2 * sl * sl)
    return np.array(
        [
            (n + alt_m) * cl * co,
            (n + alt_m) * cl * so,
            (n * (1.0 - e2) + alt_m) * sl,
        ]
    )


STATION_ECEF = np.stack([station_ecef(la, lo) for _, la, lo in STATIONS])


def ecef_to_eci(r_ecef, gmst):
    c, s = np.cos(gmst), np.sin(gmst)
    x, y, z = np.asarray(r_ecef, float)
    return np.array([c * x - s * y, s * x + c * y, z])


# Elevation from the local vertical (Vallado 2013, ch. 4, topocentric frame).
def nearest_visible(r_n, gmst, min_el_deg=10.0):
    r = np.asarray(r_n, float)
    best = None
    best_el = np.radians(min_el_deg)
    for i, rec in enumerate(STATION_ECEF):
        gs_n = ecef_to_eci(rec, gmst)
        look = r - gs_n
        ln = float(np.linalg.norm(look))
        if ln < 1.0:
            continue
        up = gs_n / max(float(np.linalg.norm(gs_n)), 1.0)
        el = float(np.arcsin(np.clip(np.dot(look / ln, up), -1.0, 1.0)))
        if el > best_el:
            best_el = el
            d = gs_n - r
            dn = float(np.linalg.norm(d))
            best = d / dn if dn > 1.0 else -r / max(float(np.linalg.norm(r)), 1.0)
    vis = 1.0 if best is not None else 0.0
    if best is None:
        best = np.zeros(3)
    return best, vis, best_el if vis > 0.5 else 0.0
