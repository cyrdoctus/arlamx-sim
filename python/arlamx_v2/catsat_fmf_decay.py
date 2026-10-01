"""CatSat deployed-exterior decay to 120 km on the v2.7 plant.

Free-molecular Sentman force on a 1% exterior panel model, GGM03S through
degree 70 (v2.7.5 SH_MAX_DEGREE),
Sun and Moon third body (Meeus analytic ephemeris; v2.7 has no Horizons
table), optical-plate SRP (aluminized Mylar) with Earth IR and albedo,
and MSIS 2.1 with the 45-day Ap / F10.7 forecast tiled forward.
Solar pressure already scales as (1 AU / d)^2 inside the plant.
Two ram holds: smallest and largest projected area, plus the −Y edge.

Body frame, fixed from the STL:
  +X  outward normal of the extended solar panel
  -Y  from the bus toward that panel
  -Z  from the bus along the long boom

The raw STL is 2.58e6 triangles and about 1.79 m^2 of facet area, most of it
internal CAD. Sentman has no self-shadowing, so the panels are the exterior
shell: 1 mm surface voxels, hollow cross-sections filled, then coplanar faces
on each plane summed. That sum does not change the area, so it is inside the
1% simplify budget. Each CAD-axis ram area of the solid must stay within 1%
of a 1 mm raster of the STL itself.

Mass is the GCAT value for NORAD 60246 (12 kg). A press note of "about 9 kg"
would shorten both lifetimes by about 12/9.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from arlamx_v2 import cpp
from arlamx_v2.advisors.attitudes import quat_from_dcm
from arlamx_v2.atmosphere import jd_to_datetime
from arlamx_v2.paths import resolve_ggm

ROOT = Path(__file__).resolve().parents[2]
STL = ROOT / "data" / "CATSAT_V7_Deployed_External.STL"
OUT = ROOT / "outputs" / "analysis" / "decay" / "catsat_fmf_20261001"

MASS_KG = 12.0
ALT_STOP_KM = 120.0
DT_S = 10.0
LOG_EVERY_S = 60.0
MSIS_EVERY_S = 60.0
MAX_DAYS = 420.0
# Progress line and the incremental npz. A multi-year case raises the save
# interval so the checkpoint does not dominate the run.
PRINT_EVERY_S = 0.5 * 86400.0
SAVE_EVERY_S = 0.5 * 86400.0
PITCH_MM = 1.0
# Largest cross-section seal that keeps every CAD-axis silhouette inside 1%
# of the STL raster. 12 mm spills the boom-axis outline; 8 mm does not.
SEAL_VOX = 8
MU_MOON = 4.902800118e12
MU_SUN = 1.32712440018e20
AU = 1.495978707e11
OMEGA_EARTH = 7.2921150e-5
CR = 1.8
ALPHA_E = 0.93
T_W = 300.0
# config/plant/physics.yaml: optical al_mylar, Earth IR partition.
SRP_CA, SRP_CS, SRP_CD = 0.08, 0.88, 0.04
IR_CA, IR_CS, IR_CD = 0.85, 0.0, 0.15

# 45-day SWPC forecast, 30 Sep 2026 through 13 Nov 2026, then tiled.
AP = np.array([
    5, 5, 20, 15, 20, 12, 10, 10, 8, 5,
    5, 9, 8, 8, 8, 5, 5, 5, 5, 5,
    5, 15, 20, 15, 12, 5, 5, 5, 5, 5,
    5, 10, 12, 10, 8, 7, 5, 5, 12, 10,
    8, 8, 5, 5, 5,
], dtype=float)
F107 = np.array([
    95, 95, 95, 90, 90, 90, 90, 92, 93, 95,
    98, 100, 102, 102, 100, 102, 102, 100, 98, 97,
    96, 94, 92, 90, 88, 98, 100, 105, 105, 100,
    95, 90, 91, 91, 92, 93, 95, 98, 100, 102,
    102, 100, 102, 102, 100,
], dtype=float)

# DE441 geocentric Moon, Horizons, daily 00:00 UT. delta in AU, deldot in km/s,
# sky motion in arcsec/min, PA from north toward east.
HORIZONS = """
2026-09-30 03 01 15.29 +22 26 13.9 0.00247676260609 -0.0157206 35.440889 72.947741
2026-10-01 04 01 41.72 +25 54 31.2 0.00247030450157 -0.0068182 35.564079 79.205241
2026-10-02 05 04 59.25 +27 44 29.2 0.00246865614642 0.0008967 35.560239 86.407745
2026-10-03 06 09 14.44 +27 44 27.4 0.00247110236439 0.0073877 35.460203 93.930790
2026-10-04 07 12 11.86 +25 55 20.4 0.00247700837881 0.0129471 35.289418 101.06085
2026-10-05 08 12 06.26 +22 29 59.0 0.00248596458531 0.0180303 35.061765 107.20952
2026-10-06 09 08 12.98 +17 48 41.6 0.00249782419041 0.0230598 34.778345 112.03443
2026-10-07 10 00 43.87 +12 14 11.1 0.00251263241598 0.0282540 34.430995 115.41678
2026-10-08 10 50 25.72 +06 08 22.2 0.00253047454064 0.0335231 34.009017 117.36966
2026-10-09 11 38 19.94 -00 08 50.2 0.00255128967520 0.0384563 33.506830 117.95582
2026-10-10 12 25 29.70 -06 19 33.0 0.00257470369654 0.0423989 32.930131 117.24231
2026-10-11 13 12 52.71 -12 07 35.6 0.00259992654474 0.0445919 32.298729 115.28816
2026-10-12 14 01 16.03 -17 18 13.3 0.00262573880289 0.0443341 31.645536 112.15505
2026-10-13 14 51 10.67 -21 38 06.6 0.00265056749126 0.0411229 31.012515 107.93328
2026-10-14 15 42 45.60 -24 55 43.4 0.00267263037834 0.0347473 30.445301 102.77464
2026-10-15 16 35 43.65 -27 02 06.3 0.00269011794656 0.0253261 29.988092 96.918182
"""


def jd_utc(dt: datetime) -> float:
    y, m, d = dt.year, dt.month, dt.day
    hr = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3.6e9
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return jd + hr / 24.0


EPOCH = datetime(2026, 9, 30, 3, 50, tzinfo=timezone.utc)
EPOCH_JD = jd_utc(EPOCH)
FORECAST_JD0 = jd_utc(datetime(2026, 9, 30, tzinfo=timezone.utc))
R0 = np.array([4057436.0, -4706490.0, 2538278.0])
V0 = np.array([1092.0, -2890.0, -7068.0])


def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("zero vector")
    return np.asarray(v, float) / n


def parse_horizons():
    jd, pos, vel = [], [], []
    for line in HORIZONS.strip().splitlines():
        t = line.split()
        dt = datetime.strptime(t[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        rah, ram, ras = float(t[1]), float(t[2]), float(t[3])
        dec_s = t[4]
        sign = -1.0 if dec_s.startswith("-") else 1.0
        decd = abs(float(dec_s))
        decm, decs = float(t[5]), float(t[6])
        delta = float(t[7]) * AU
        deldot = float(t[8]) * 1000.0
        sky = float(t[9])
        pa = np.radians(float(t[10]))
        ra = np.radians((rah + ram / 60.0 + ras / 3600.0) * 15.0)
        dec = sign * np.radians(decd + decm / 60.0 + decs / 3600.0)
        rhat = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        east = np.array([-np.sin(ra), np.cos(ra), 0.0])
        north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
        rate = sky * np.pi / (180.0 * 3600.0 * 60.0)
        v = deldot * rhat + (rate * delta) * (np.cos(pa) * north + np.sin(pa) * east)
        jd.append(jd_utc(dt))
        pos.append(rhat * delta)
        vel.append(v)
    return np.asarray(jd), np.asarray(pos), np.asarray(vel)


def propagate_moon(jd_tab, r_tab, v_tab, jd_end, step_s=600.0):
    """Earth point mass + solar third body, seeded on the first Horizons state.

    Inside the tabulated window the returned samples are replaced by a cubic
    interpolation of the Horizons positions, so the arc the user supplied is
    what the plant integrates. After the last sample the numerical state that
    was fitted to that sample carries the Moon forward.
    """
    mu_e = cpp.MU_GGM
    # Integrate from the first tabulated state, but restart the state at the
    # last Horizons sample using a local quadratic velocity so the handoff
    # is continuous with DE441.
    def accel(jd, r):
        a = -mu_e * r / np.linalg.norm(r) ** 3
        sun = cpp.sun_unit_analytic(jd) * AU
        a = a + np.asarray(cpp.accel_third_body(r, sun, MU_SUN), float)
        return a

    # Velocity at the last sample from a one-sided difference of the table,
    # blended with the reconstructed sky-plane velocity.
    dt = (jd_tab[-1] - jd_tab[-2]) * 86400.0
    v_fd = (r_tab[-1] - r_tab[-2]) / dt
    v_last = 0.5 * (v_fd + v_tab[-1])

    jd = jd_tab[-1]
    r = r_tab[-1].copy()
    v = v_last.copy()
    out_jd = [jd]
    out_r = [r.copy()]
    n = int(np.ceil((jd_end - jd) * 86400.0 / step_s))
    for _ in range(n):
        jmid = jd + 0.5 * step_s / 86400.0
        k1v = accel(jd, r)
        k1r = v
        k2v = accel(jmid, r + 0.5 * step_s * k1r)
        k2r = v + 0.5 * step_s * k1v
        k3v = accel(jmid, r + 0.5 * step_s * k2r)
        k3r = v + 0.5 * step_s * k2v
        jn = jd + step_s / 86400.0
        k4v = accel(jn, r + step_s * k3r)
        k4r = v + step_s * k3v
        r = r + (step_s / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (step_s / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        jd = jn
        out_jd.append(jd)
        out_r.append(r.copy())
    prop_jd = np.asarray(out_jd)
    prop_r = np.asarray(out_r)

    # Hourly samples across the whole span: cubic Horizons, then the propagator.
    jd0 = jd_tab[0]
    grid = np.arange(jd0, jd_end + 1e-9, 3600.0 / 86400.0)
    # cubic via numpy on each axis, only inside the table
    inside = grid <= jd_tab[-1]
    r_grid = np.empty((grid.size, 3))
    for k in range(3):
        r_grid[inside, k] = np.interp(grid[inside], jd_tab, r_tab[:, k])
    # denser than daily linear: use the table's own cubic through np.interp is
    # linear. Daily chords of the Moon are ~3.9 km of sagitta (see notes in
    # the run log). Upgrade with a cubic Hermite using reconstructed velocity.
    r_grid[inside] = _hermite(grid[inside], jd_tab, r_tab, v_tab)
    after = ~inside
    if np.any(after):
        r_grid[after] = _interp_rows(grid[after], prop_jd, prop_r)
    return grid, r_grid, v_last


def _hermite(jd, jd_n, r_n, v_n):
    """Cubic Hermite between daily states. v_n is m/s; time unit of jd is days."""
    out = np.empty((jd.size, 3))
    idx = np.searchsorted(jd_n, jd, side="right") - 1
    idx = np.clip(idx, 0, len(jd_n) - 2)
    j0 = jd_n[idx]
    j1 = jd_n[idx + 1]
    dt = (j1 - j0) * 86400.0
    u = (jd - j0) / (j1 - j0)
    u = u[:, None]
    p0 = r_n[idx]
    p1 = r_n[idx + 1]
    m0 = v_n[idx] * dt[:, None]
    m1 = v_n[idx + 1] * dt[:, None]
    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    out = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    return out


def _interp_rows(jd, jd_n, r_n):
    out = np.empty((jd.size, 3))
    for k in range(3):
        out[:, k] = np.interp(jd, jd_n, r_n[:, k])
    return out


def moon_table_error(jd_tab, r_tab, grid, r_grid):
    err = []
    for j, r in zip(jd_tab, r_tab):
        i = int(np.argmin(np.abs(grid - j)))
        err.append(np.linalg.norm(r_grid[i] - r) / 1e3)
    return float(np.max(err)), float(np.mean(err))


def load_stl_vertices(path):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    rec = raw[84 : 84 + n * 50].reshape(n, 50)
    verts = np.frombuffer(np.ascontiguousarray(rec[:, 12:48]), dtype="<f4").reshape(n, 3, 3)
    return np.array(verts, dtype=np.float32, copy=True)


def _mark(solid, origin, pitch, pts):
    q = np.floor((pts - origin) / pitch).astype(np.int32)
    m = (
        (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 2] >= 0)
        & (q[:, 0] < solid.shape[2]) & (q[:, 1] < solid.shape[1]) & (q[:, 2] < solid.shape[0])
    )
    q = q[m]
    solid[q[:, 2], q[:, 1], q[:, 0]] = 1


def raster_surface(verts, pitch):
    """Binary surface voxels of a triangle soup. verts and pitch are mm."""
    vmin = verts.reshape(-1, 3).min(axis=0).astype(np.float64) - 2 * pitch
    vmax = verts.reshape(-1, 3).max(axis=0).astype(np.float64) + 2 * pitch
    dims = np.ceil((vmax - vmin) / pitch).astype(int) + 1
    solid = np.zeros((int(dims[2]), int(dims[1]), int(dims[0])), np.uint8)
    _mark(solid, vmin, pitch, verts.reshape(-1, 3).astype(np.float64))
    v64 = verts.astype(np.float64)
    longest = np.max(
        np.stack([
            np.linalg.norm(v64[:, 1] - v64[:, 0], axis=1),
            np.linalg.norm(v64[:, 2] - v64[:, 1], axis=1),
            np.linalg.norm(v64[:, 0] - v64[:, 2], axis=1),
        ], 0),
        axis=0,
    )
    need = np.nonzero(longest > pitch)[0]
    step = pitch * 0.65
    buf = []
    for i in need:
        tri = v64[i]
        e0 = tri[1] - tri[0]
        e1 = tri[2] - tri[0]
        nu = max(int(np.linalg.norm(e0) / step), 1)
        nv = max(int(np.linalg.norm(e1) / step), 1)
        a = np.linspace(0.0, 1.0, nu + 1)
        b = np.linspace(0.0, 1.0, nv + 1)
        aa, bb = np.meshgrid(a, b, indexing="ij")
        m = (aa + bb) <= 1.0 + 1e-9
        pts = tri[0] + aa[m, None] * e0 + bb[m, None] * e1
        buf.append(pts)
        if len(buf) >= 4000:
            _mark(solid, vmin, pitch, np.concatenate(buf, 0))
            buf = []
    if buf:
        _mark(solid, vmin, pitch, np.concatenate(buf, 0))
    return solid.astype(bool), vmin, pitch


def fill_interior(surface, seal_vox):
    """Fill cavities without thickening the exterior or deleting thin parts.

    A dilation seals CAD gaps, the enclosed cavity is flooded, and that cavity
    is grown back onto the original surface. The dilated skin outside the
    original surface is not kept, so a thin panel and the boom stay at the
    rasterized size.
    """
    from scipy import ndimage

    k = int(seal_vox)
    if k <= 0:
        return ndimage.binary_fill_holes(surface)
    obstacle = ndimage.binary_dilation(surface, iterations=k)
    sealed = ndimage.binary_fill_holes(obstacle)
    cavity = sealed & ~obstacle
    if not np.any(cavity):
        return surface.copy()
    grown = ndimage.binary_dilation(cavity, iterations=k)
    solid = surface | (grown & sealed)
    return solid


def fill_sections(surface, seal_vox):
    """Fill hollow cross-sections along the boom without growing the outline.

    Each slice is closed by ``seal_vox`` pixels so a small gap in a wall still
    counts as a loop, the enclosed pixels are kept, and the original surface
    voxels are always kept. Thin panels and the boom therefore survive.
    """
    from scipy import ndimage

    k = int(seal_vox)
    out = np.empty_like(surface)
    for z in range(surface.shape[0]):
        sl = surface[z]
        if k <= 0:
            out[z] = ndimage.binary_fill_holes(sl)
            continue
        sealed = ndimage.binary_fill_holes(ndimage.binary_dilation(sl, iterations=k))
        out[z] = sl | ndimage.binary_erosion(sealed, iterations=k)
    return out


def voxel_shell(verts, pitch, seal_vox=4):
    """Exterior solid of a triangle soup. verts are mm, pitch is mm."""
    surface, vmin, pitch = raster_surface(verts, pitch)
    return fill_sections(surface, seal_vox), vmin, pitch


def _solid_metrics(filled, pitch_mm):
    pitch_m = pitch_mm * 1e-3
    vol = float(filled.sum()) * pitch_m ** 3
    # Exposed-face count is not known yet; silhouette gives a lower bound on area.
    sil = silhouette_areas(filled, pitch_mm)
    area_lb = sil["+X"] + sil["+Y"] + sil["+Z"]
    return vol, area_lb, sil


def _mark2(img, origin, pitch, pts):
    q = np.floor((pts - origin) / pitch).astype(np.int32)
    m = (
        (q[:, 0] >= 0) & (q[:, 1] >= 0)
        & (q[:, 0] < img.shape[1]) & (q[:, 1] < img.shape[0])
    )
    q = q[m]
    img[q[:, 1], q[:, 0]] = 1


def projection_areas(verts, pitch):
    """CAD-axis ram area (m^2) by rasterizing every STL triangle onto the three planes.

    This is the mesh silhouette, independent of the voxel closing.
    """
    v64 = verts.astype(np.float64)
    longest = np.max(
        np.stack([
            np.linalg.norm(v64[:, 1] - v64[:, 0], axis=1),
            np.linalg.norm(v64[:, 2] - v64[:, 1], axis=1),
            np.linalg.norm(v64[:, 0] - v64[:, 2], axis=1),
        ], 0),
        axis=0,
    )
    need = np.nonzero(longest > pitch)[0]
    step = pitch * 0.65
    areas = {}
    for axis, key in ((0, "+X"), (1, "+Y"), (2, "+Z")):
        iu, iv = (a for a in (0, 1, 2) if a != axis)
        tri2 = np.stack([v64[:, :, iu], v64[:, :, iv]], axis=2)
        vmin = tri2.reshape(-1, 2).min(axis=0) - 2 * pitch
        vmax = tri2.reshape(-1, 2).max(axis=0) + 2 * pitch
        dims = np.ceil((vmax - vmin) / pitch).astype(int) + 1
        img = np.zeros((int(dims[1]), int(dims[0])), np.uint8)
        _mark2(img, vmin, pitch, tri2.reshape(-1, 2))
        buf = []
        for i in need:
            tri = tri2[i]
            e0 = tri[1] - tri[0]
            e1 = tri[2] - tri[0]
            nu = max(int(np.linalg.norm(e0) / step), 1)
            nv = max(int(np.linalg.norm(e1) / step), 1)
            a = np.linspace(0.0, 1.0, nu + 1)
            b = np.linspace(0.0, 1.0, nv + 1)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            m = (aa + bb) <= 1.0 + 1e-9
            buf.append(tri[0] + aa[m, None] * e0 + bb[m, None] * e1)
            if len(buf) >= 4000:
                _mark2(img, vmin, pitch, np.concatenate(buf, 0))
                buf = []
        if buf:
            _mark2(img, vmin, pitch, np.concatenate(buf, 0))
        areas[key] = float(img.sum()) * (pitch * 1e-3) ** 2
    return areas


def silhouette_areas(filled, pitch_mm):
    """Ram area (m^2) looking along +X, +Y, +Z of the voxel grid (CAD axes)."""
    p2 = (pitch_mm * 1e-3) ** 2
    ax = int(np.count_nonzero(filled.any(axis=(0, 1)))) * p2
    ay = int(np.count_nonzero(filled.any(axis=(0, 2)))) * p2
    az = int(np.count_nonzero(filled.any(axis=(1, 2)))) * p2
    # any() over the two axes orthogonal to the look direction:
    # look +X → collapse x, which is axis 2 of (z,y,x)
    look_x = int(np.count_nonzero(filled.any(axis=2))) * p2
    look_y = int(np.count_nonzero(filled.any(axis=1))) * p2
    look_z = int(np.count_nonzero(filled.any(axis=0))) * p2
    return {"+X": look_x, "+Y": look_y, "+Z": look_z, "note_unused": (ax, ay, az)}


def exposed_faces(filled, origin, pitch):
    """Outward faces of the solid. filled is indexed (z, y, x) in the CAD frame."""
    pairs = (
        # normal +X: filled and (not filled at x+1). array axis 2.
        (np.array([1.0, 0.0, 0.0]), filled[:, :, :-1] & ~filled[:, :, 1:], "px"),
        (np.array([-1.0, 0.0, 0.0]), filled[:, :, 1:] & ~filled[:, :, :-1], "nx"),
        (np.array([0.0, 1.0, 0.0]), filled[:, :-1, :] & ~filled[:, 1:, :], "py"),
        (np.array([0.0, -1.0, 0.0]), filled[:, 1:, :] & ~filled[:, :-1, :], "ny"),
        (np.array([0.0, 0.0, 1.0]), filled[:-1, :, :] & ~filled[1:, :, :], "pz"),
        (np.array([0.0, 0.0, -1.0]), filled[1:, :, :] & ~filled[:-1, :, :], "nz"),
    )
    quads = []
    for normal, mask, tag in pairs:
        zz, yy, xx = np.nonzero(mask)
        quads.append((tag, normal, zz, yy, xx))
    return quads, origin, pitch


def strips_from_mask(tag, zz, yy, xx, origin, pitch):
    """Merge exposed voxel faces into area-exact rectangles.

    Runs of consecutive cells become strips, then strips of the same span
    stacked on the other axis become one rectangle. Area is the voxel area.
    """
    if tag in ("px", "nx"):
        plane, a, b = xx, yy, zz
    elif tag in ("py", "ny"):
        plane, a, b = yy, xx, zz
    else:
        plane, a, b = zz, xx, yy
    order = np.lexsort((a, b, plane))
    plane, a, b = plane[order], a[order], b[order]
    runs = []
    i = 0
    n = plane.size
    while i < n:
        j = i + 1
        while j < n and plane[j] == plane[i] and b[j] == b[i] and a[j] == a[j - 1] + 1:
            j += 1
        # Both slice conventions put the shared face one step past `plane`:
        # +normal masks index the solid voxel on the low side; −normal masks
        # are taken on a [1:] view so `plane` is already the air-side index.
        p = int(plane[i]) + 1
        a0 = int(a[i])
        a1 = int(a[j - 1]) + 1
        runs.append((p, int(b[i]), a0, a1))
        i = j
    if not runs:
        return np.zeros((0, 3, 3))
    runs.sort(key=lambda t: (t[0], t[2], t[3], t[1]))
    tris = []
    i = 0
    nrun = len(runs)
    while i < nrun:
        p, b0, a0, a1 = runs[i]
        b1 = b0 + 1
        j = i + 1
        while (
            j < nrun
            and runs[j][0] == p
            and runs[j][2] == a0
            and runs[j][3] == a1
            and runs[j][1] == b1
        ):
            b1 += 1
            j += 1
        tris.append(_strip_tris(tag, p, a0, a1, b0, b1, origin, pitch))
        i = j
    return np.concatenate(tris, axis=0)


def _strip_tris(tag, p, a0, a1, b0, b1, origin, pitch):
    """Two triangles for a strip. p, a, b are voxel-edge indices."""
    def xyz(p, a, b):
        if tag in ("px", "nx"):
            return np.array([origin[0] + p * pitch, origin[1] + a * pitch, origin[2] + b * pitch])
        if tag in ("py", "ny"):
            return np.array([origin[0] + a * pitch, origin[1] + p * pitch, origin[2] + b * pitch])
        return np.array([origin[0] + a * pitch, origin[1] + b * pitch, origin[2] + p * pitch])

    c00 = xyz(p, a0, b0)
    c10 = xyz(p, a1, b0)
    c11 = xyz(p, a1, b1)
    c01 = xyz(p, a0, b1)
    # (y, x, z) is an odd permutation, so the Y tags take the other winding.
    if tag in ("px", "pz", "ny"):
        t0 = np.stack([c00, c10, c11])
        t1 = np.stack([c00, c11, c01])
    else:
        t0 = np.stack([c00, c11, c10])
        t1 = np.stack([c00, c01, c11])
    return np.stack([t0, t1])


def mesh_from_voxels(filled, origin, pitch):
    quads, _, _ = exposed_faces(filled, origin, pitch)
    blocks = []
    for tag, _normal, zz, yy, xx in quads:
        blocks.append(strips_from_mask(tag, zz.astype(int), yy.astype(int), xx.astype(int), origin, pitch))
    tris = np.concatenate([b for b in blocks if len(b)], axis=0)
    return tris


def tris_to_panels(tris):
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 1e-12
    tris, cr, area = tris[ok], cr[ok], area[ok]
    n = cr / (2.0 * area)[:, None]
    c = tris.mean(axis=1)
    return n, area, c, tris


def orient_body(filled, origin, pitch):
    """CAD mm solid → rotation whose columns are body axes in CAD.

    +X outward panel normal, -Y toward the panel, -Z toward the boom.
    Origin is the bus centre, metres, returned separately.
    """
    zz, yy, xx = np.nonzero(filled)
    pts = np.stack([
        origin[0] + (xx + 0.5) * pitch,
        origin[1] + (yy + 0.5) * pitch,
        origin[2] + (zz + 0.5) * pitch,
    ], 1)
    # Bus: the high-Z cluster. The boom is the thin low-Z tail.
    z = pts[:, 2]
    z_cut = np.quantile(z, 0.55)
    bus = pts[z >= z_cut]
    boom = pts[z < np.quantile(z, 0.25)]
    bus_c = bus.mean(axis=0)
    # Panel: voxels proud of the bus in +X (the tab seen in the silhouette).
    x_bus_max = np.quantile(bus[:, 0], 0.98)
    panel = pts[pts[:, 0] > x_bus_max + pitch]
    if len(panel) < 20:
        raise RuntimeError("extended panel was not found proud of the bus in +CAD X")
    panel_c = panel.mean(axis=0)
    ext = panel_c - bus_c
    boom_c = boom.mean(axis=0)
    boom_dir = boom_c - bus_c
    # Panel thin axis from its covariance.
    cov = np.cov((panel - panel_c).T)
    w, vec = np.linalg.eigh(cov)
    normal = vec[:, int(np.argmin(w))]
    # Outward is the sense of the normal pointing away from the bus along Y
    # (the panel lives on one Y face). Try both signs; keep the frame that
    # puts the extension on -Y and the boom on -Z.
    best = None
    for sign in (1.0, -1.0):
        x = _unit(sign * normal)
        z = -_unit(boom_dir)
        z = z - np.dot(z, x) * x
        z = _unit(z)
        y = np.cross(z, x)
        y = _unit(y)
        score = np.dot(y, -_unit(ext)) + np.dot(z, -_unit(boom_dir))
        if best is None or score > best[0]:
            best = (score, x, y, z)
    _score, x, y, z = best
    R = np.column_stack([x, y, z])  # v_cad = R @ v_body
    if np.linalg.det(R) < 0:
        raise RuntimeError("body frame came out left-handed")
    origin_m = bus_c * 1e-3
    report = {
        "bus_center_mm": bus_c.tolist(),
        "panel_center_mm": panel_c.tolist(),
        "boom_center_mm": boom_c.tolist(),
        "body_X_in_cad": x.tolist(),
        "body_Y_in_cad": y.tolist(),
        "body_Z_in_cad": z.tolist(),
        "det": float(np.linalg.det(R)),
        "ext_dot_minus_Y": float(np.dot(R.T @ _unit(ext), np.array([0.0, -1.0, 0.0]))),
        "boom_dot_minus_Z": float(np.dot(R.T @ _unit(boom_dir), np.array([0.0, 0.0, -1.0]))),
    }
    return R, origin_m, report


def cad_tris_to_body(tris_mm, R, origin_m):
    p = tris_mm.reshape(-1, 3) * 1e-3
    pb = (p - origin_m) @ R
    return pb.reshape(tris_mm.shape)


def merge_coplanar(n, A, c, plane_tol=5e-4):
    """One panel per plane. Area and the area centroid are unchanged."""
    n = np.asarray(n, float)
    A = np.asarray(A, float)
    c = np.asarray(c, float)
    nu = n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-15)
    axis = np.argmax(np.abs(nu), axis=1)
    sign = np.sign(nu[np.arange(len(nu)), axis])
    key_n = axis.astype(np.int64) * 2 + (sign < 0).astype(np.int64)
    offset = (nu * c).sum(axis=1)
    key_p = np.round(offset / plane_tol).astype(np.int64)
    key = key_n * 10_000_003 + key_p
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    breaks = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1], True])
    ns, As, cs = [], [], []
    for s, e in zip(breaks[:-1], breaks[1:]):
        idx = order[s:e]
        w = A[idx]
        sw = float(w.sum())
        if sw <= 1e-12:
            continue
        ns.append(nu[idx[0]])
        As.append(sw)
        cs.append((c[idx] * w[:, None]).sum(axis=0) / sw)
    return np.asarray(ns, float), np.asarray(As, float), np.asarray(cs, float)


def build_panels(out_dir: Path):
    cache = out_dir / "panels_1pct.npz"
    meta_path = out_dir / "geometry.json"
    if cache.exists() and meta_path.exists():
        data = np.load(cache)
        meta = json.loads(meta_path.read_text())
        return data["n"], data["A"], data["c"], data["tris"], meta

    print("loading STL", flush=True)
    t0 = time.time()
    verts = load_stl_vertices(STL)
    print(f"  {len(verts)} triangles in {time.time()-t0:.1f}s", flush=True)
    print(f"voxel pitch {PITCH_MM} mm, section seal {SEAL_VOX} mm", flush=True)
    t0 = time.time()
    filled, origin, pitch = voxel_shell(verts, PITCH_MM, seal_vox=SEAL_VOX)
    vol, area_lb, sil = _solid_metrics(filled, pitch)
    print(
        f"  shell in {time.time()-t0:.1f}s  voxels {int(filled.sum())}  "
        f"vol {vol*1e6:.1f} cm^3",
        flush=True,
    )
    if vol < 4.0e-3:
        raise RuntimeError(
            f"exterior volume is only {vol*1e6:.0f} cm^3; the bus interior did not fill"
        )
    print(f"STL projection at {PITCH_MM} mm", flush=True)
    t0 = time.time()
    proj = projection_areas(verts, PITCH_MM)
    rel = {k: (sil[k] - proj[k]) / max(proj[k], 1e-12) for k in ("+X", "+Y", "+Z")}
    print(f"  projection in {time.time()-t0:.1f}s", {k: proj[k] for k in ("+X", "+Y", "+Z")}, flush=True)
    print("CAD-axis ram area m^2 voxel", {k: sil[k] for k in ("+X", "+Y", "+Z")}, flush=True)
    print("relative to STL", rel, flush=True)
    worst_rel = max(abs(v) for v in rel.values())
    if worst_rel > 0.01:
        raise RuntimeError(
            f"voxel ram area differs from the STL silhouette by {worst_rel:.3%}; "
            "1% rule is not met"
        )

    R, origin_m, orient = orient_body(filled, origin, pitch)
    print("orientation", json.dumps(orient, indent=2), flush=True)
    if orient["boom_dot_minus_Z"] < 0.95 or orient["ext_dot_minus_Y"] < 0.7:
        raise RuntimeError(f"body axes did not land on the boom and the panel: {orient}")

    print("extracting faces", flush=True)
    tris_mm = mesh_from_voxels(filled, origin, pitch)
    print(f"  strip triangles {len(tris_mm)}", flush=True)
    tris_b = cad_tris_to_body(tris_mm, R, origin_m)
    n0, A0, c0, tris0 = tris_to_panels(tris_b)
    sum_nA = (n0 * A0[:, None]).sum(axis=0)
    print(
        f"  area {A0.sum():.4f} m^2  nA sum {sum_nA}  "
        f"|nA|/A {np.linalg.norm(sum_nA)/A0.sum():.4f}",
        flush=True,
    )
    px = n0[:, 0] > 0.5
    nx = n0[:, 0] < -0.5
    if not np.any(px) or not np.any(nx):
        raise RuntimeError("body frame has no +X or -X faces")
    x_plus = float(np.average(c0[px, 0], weights=A0[px]))
    x_minus = float(np.average(c0[nx, 0], weights=A0[nx]))
    print(f"  area-weighted X of +X faces {x_plus:.4f} m, of -X faces {x_minus:.4f} m", flush=True)
    if x_plus <= x_minus:
        raise RuntimeError("body +X faces are not on the outward side of the solid")
    pts = tris0.reshape(-1, 3)
    print(
        "  body bbox m  min", np.round(pts.min(0), 3), "max", np.round(pts.max(0), 3),
        flush=True,
    )
    # Faces of this shell are axis-aligned, so every coplanar run is one panel.
    # Area is unchanged, which is inside the 1% simplify budget. An outline
    # merge of 1e5 rectangles does not reduce Sentman any further.
    n, A, c = merge_coplanar(n0, A0, c0)
    used = "coplanar_exact"
    print(f"  coplanar panels {len(A)}  area {A.sum():.4f} m^2", flush=True)
    if abs(A.sum() / A0.sum() - 1.0) > 0.01:
        raise RuntimeError("coplanar merge changed the area by more than 1%")
    n = np.asarray(n, float)
    A = np.asarray(A, float)
    c = np.asarray(c, float)
    # Drop degenerate.
    ok = A > 1e-10
    n, A, c = n[ok], A[ok], c[ok]
    body_ram = {}
    for name, d in {
        "+X": (1, 0, 0), "-X": (-1, 0, 0),
        "+Y": (0, 1, 0), "-Y": (0, -1, 0),
        "+Z": (0, 0, 1), "-Z": (0, 0, -1),
    }.items():
        body_ram[name] = float(np.sum(A * np.maximum(n @ np.array(d, float), 0.0)))
    meta = {
        "pitch_mm": PITCH_MM,
        "volume_cm3": vol * 1e6,
        "silhouette_cad_m2": {k: sil[k] for k in ("+X", "+Y", "+Z")},
        "stl_projection_m2": {k: proj[k] for k in ("+X", "+Y", "+Z")},
        "projection_relative_change": rel,
        "panel_source": used,
        "n_panels": int(len(A)),
        "area_m2": float(A.sum()),
        "body_ram_m2": body_ram,
        "orientation": orient,
        "simplify": None if used != "simplify_sides_1pct" else {
            "n_sides": rep.n_sides,
            "n_panels": rep.n_panels,
            "area_rel_err": rep.area_rel_err,
            "unmatched_m": rep.unmatched_m,
        },
    }
    # tris for the figure: the simplified ones if we have them, else strips
    if used == "simplify_sides_1pct":
        fig_tris = np.stack(out_tris).astype(np.float32) if len(out_tris) else tris0.astype(np.float32)
    else:
        fig_tris = tris0.astype(np.float32)
    np.savez_compressed(cache, n=n, A=A, c=c, tris=fig_tris)
    meta_path.write_text(json.dumps(meta, indent=2))
    return n, A, c, fig_tris, meta


def _weld(V, F, tol):
    q = np.round(V / tol).astype(np.int64)
    # unique rows
    key = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    uniq_mask = np.empty(len(key_s), bool)
    uniq_mask[0] = True
    uniq_mask[1:] = key_s[1:] != key_s[:-1]
    new_index = np.empty(len(key), dtype=int)
    new_index[order] = np.cumsum(uniq_mask) - 1
    V2 = V[order][uniq_mask]
    return V2, new_index[F]


def ram_extremes(n, A):
    N = 2500
    i = np.arange(N)
    z = 1.0 - 2.0 * (i + 0.5) / N
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = np.pi * (3.0 - np.sqrt(5.0)) * i
    dirs = np.stack([r * np.cos(th), r * np.sin(th), z], 1)
    proj = (A[:, None] * np.maximum(n @ dirs.T, 0.0)).sum(axis=0)
    return dirs[int(proj.argmin())], dirs[int(proj.argmax())], float(proj.min()), float(proj.max())


def quat_ram(v_inertial, d_body, helper):
    """q_BN so the body direction d_body lies along v (the oncoming ram)."""
    t = _unit(v_inertial)
    d = _unit(d_body)
    h = np.asarray(helper, float)
    if abs(float(np.dot(_unit(h), t))) > 0.95:
        h = np.array([0.0, 0.0, 1.0])
    i2 = _unit(np.cross(h, t))
    i3 = np.cross(t, i2)
    hb = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = _unit(np.cross(hb, d))
    e3 = np.cross(d, e2)
    C_BN = np.column_stack([d, e2, e3]) @ np.column_stack([t, i2, i3]).T
    return quat_from_dcm(C_BN)


def sun_distance_au(jd):
    n = jd - 2451545.0
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    return 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)


def weather(jd):
    day = int(np.floor(jd - FORECAST_JD0))
    ap = AP[day % len(AP)]
    f107 = F107[day % len(F107)]
    # 81-day mean. Past days before the forecast are the forecast mean; the
    # future is the same 45-day tile, so the mean is well defined.
    window = np.array([F107[(day + k) % len(F107)] for k in range(-40, 41)])
    return float(f107), float(window.mean()), float(ap)


def geodetic(r, jd):
    gmst = float(cpp.gmst_rad(jd))
    c, s = np.cos(gmst), np.sin(gmst)
    # Latitude and altitude are invariant under the GMST rotation about Z.
    # Longitude below is east longitude.
    lat, lon_i, alt = cpp.ecef_to_geodetic(r)
    lon = (lon_i - gmst + np.pi) % (2.0 * np.pi) - np.pi
    return float(lat), float(lon), float(alt)


def en_matrix(jd):
    gmst = float(cpp.gmst_rad(jd))
    c, s = np.cos(gmst), np.sin(gmst)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def orbital_elements(r, v, mu):
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    e_vec = np.cross(v, h) / mu - r / rn
    e = float(np.linalg.norm(e_vec))
    eps = 0.5 * vn * vn - mu / rn
    a = -mu / (2.0 * eps)
    inc = float(np.degrees(np.arccos(np.clip(h[2] / hn, -1.0, 1.0))))
    energy = float(eps)
    return a, e, inc, energy, h


def make_sim(panels, moon_jd, moon_r):
    params = cpp.SimParams()
    params.mass = MASS_KG
    params.dt_s = DT_S
    params.advisor_step_s = DT_S
    params.rk4_step_s = DT_S
    params.sh_degree = 70
    params.lunisolar = True
    params.use_panel_srp = True
    params.srp_optical = True
    params.srp_ca = SRP_CA
    params.srp_cs = SRP_CS
    params.srp_cd = SRP_CD
    params.srp_scale = 1.0
    params.earth_rad = True
    params.ir_ca = IR_CA
    params.ir_cs = IR_CS
    params.ir_cd = IR_CD
    params.gravity_gradient = True
    params.corotating = True
    params.alpha_E = ALPHA_E
    params.T_w = T_W
    params.Cr = CR
    params.gsi = "sentman"
    params.one_sided_ref = True
    params.mode = "prescribed"
    params.max_slew_rad = np.pi
    params.epoch_jd = EPOCH_JD
    params.ggm_path = resolve_ggm()
    sim = cpp.Simulator(params)
    sim.set_mode("prescribed")
    n, A, c = panels
    sim.set_panels(n, A, c)
    sim.set_inertia_diag(np.array([0.15, 0.20, 0.08]))
    # v2.7 integrates the Meeus Moon. The Horizons table is no longer on the plant.
    del moon_jd, moon_r
    g = cpp.GravityHarmonics()
    if not g.load_ggm(params.ggm_path, 70) or g.max_degree() != 70:
        raise RuntimeError(f"GGM did not load at degree 70 (got {g.max_degree()})")
    return sim, g


def sample_forces(g, panels, moon_jd, moon_r, r, v, jd, q, rho, T, mbar, srp_scale):
    nrm, area, cen = panels
    C = np.asarray(cpp.mrp_to_dcm(cpp.quat_to_mrp(q)), float)
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel = v - np.cross(w, r)
    v_gas = -(C @ v_rel)
    aero = cpp.spacecraft_aero(
        nrm, area, cen, v_gas, float(rho), float(T), float(mbar), T_W,
        ALPHA_E, True, "sentman",
    )
    F_fmf_B = np.asarray(aero["force"], float)
    sun_N = np.asarray(cpp.sun_unit_analytic(jd), float)
    ecl = float(cpp.eclipse_cylindrical(r, sun_N, cpp.RE_WGS))
    sun_B = C @ sun_N
    # Same (AU/d)^2 the plant applies. srp_scale stays 1 so this is not squared twice.
    P = cpp.P_SRP_1AU * (1.0 / float(cpp.sun_dist_au(jd))) ** 2 * srp_scale
    F_srp_B = np.asarray(
        cpp.panel_srp_optical(nrm, area, cen, sun_B, SRP_CA, SRP_CS, SRP_CD, ecl, P)[0], float
    )
    rn = float(np.linalg.norm(r))
    nadir_B = C @ (-r / rn)
    F_erp_B = np.asarray(
        cpp.earth_rad(
            nrm, area, cen, nadir_B, rn, float(np.dot(r / rn, sun_N)), P,
            [SRP_CA, SRP_CS, SRP_CD], [IR_CA, IR_CS, IR_CD],
        )[0],
        float,
    )
    F_fmf = C.T @ F_fmf_B
    F_srp = C.T @ F_srp_B
    F_erp = C.T @ F_erp_B
    EN = en_matrix(jd)
    a_full = EN.T @ np.asarray(g.accel_ecef(EN @ r, g.max_degree()), float)
    a_tb = np.asarray(cpp.accel_twobody(r, g.mu()), float)
    a_harm = a_full - a_tb
    hat_m, rm = cpp.moon_analytic(float(jd))
    r_moon = np.asarray(hat_m, float) * float(rm)
    r_sun = sun_N * float(cpp.sun_dist_au(jd)) * AU
    a_moon = np.asarray(cpp.accel_third_body(r, r_moon, MU_MOON), float)
    a_sun = np.asarray(cpp.accel_third_body(r, r_sun, MU_SUN), float)
    m = MASS_KG
    vhat = v_rel / max(np.linalg.norm(v_rel), 1.0)
    along = float(np.dot(F_fmf, vhat))
    return {
        "F_fmf": F_fmf,
        "F_srp": F_srp,
        "F_erp": F_erp,
        "F_harm": m * a_harm,
        "F_moon": m * a_moon,
        "F_sun": m * a_sun,
        "F_tb": m * a_tb,
        "fmf_along": along,
        "Cd": float(aero["Cd"]),
        "Cl": float(aero["Cl"]),
        "eclipse": ecl,
        "ram": float(np.sum(area * np.maximum(nrm @ (C @ vhat), 0.0))),
    }


def run_case(name, d_body, panels, moon_jd, moon_r, out_dir: Path):
    sim, g = make_sim(panels, moon_jd, moon_r)
    sim.reset(R0, V0, np.zeros(3), np.zeros(3))
    # Prime the atmosphere and the ram attitude.
    lat, lon, alt = geodetic(R0, EPOCH_JD)
    f107, f107a, ap = weather(EPOCH_JD)
    rho, T, mbar = _msis(alt / 1e3, np.degrees(lat), np.degrees(lon), EPOCH_JD, f107, f107a, ap)
    sim.set_atmosphere(rho, T, mbar)
    sim.set_srp_scale(1.0)
    nmax = int(MAX_DAYS * 86400.0 / DT_S) + 2
    log_every = max(1, int(round(LOG_EVERY_S / DT_S)))
    msis_every = max(1, int(round(MSIS_EVERY_S / DT_S)))
    n_log = nmax // log_every + 8
    cols = [
        "t_s", "jd", "alt_km", "sma_km", "ecc", "inc_deg", "energy", "h",
        "rho", "f107", "ap", "eclipse", "Cd", "Cl", "ram_m2",
        "Ffmf", "Falong", "Fsrp", "Ferp", "Fharm", "Fmoon", "Fsun",
    ]
    buf = np.full((n_log, len(cols)), np.nan)
    work = {k: 0.0 for k in ("fmf", "srp", "erp", "harm", "moon", "sun", "tb")}
    Jang = {k: np.zeros(3) for k in work}
    Plin = {k: np.zeros(3) for k in work}
    work_hist = np.zeros((n_log, 7))
    Jang_hist = np.zeros((n_log, 7))
    Plin_hist = np.zeros((n_log, 7))
    keys = ("fmf", "srp", "erp", "harm", "moon", "sun", "tb")
    n_row = 0
    prev = None
    t_wall = time.time()
    next_print = 0.0
    next_save = 0.0
    hit = None
    v_rel0 = V0 - np.cross([0.0, 0.0, OMEGA_EARTH], R0)
    q = quat_ram(v_rel0, d_body, np.cross(R0, V0))
    for k in range(nmax):
        st = sim.get_state()
        r = np.asarray(st["r"], float)
        v = np.asarray(st["v"], float)
        t = float(st["t"])
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            print(name, "non-finite state", flush=True)
            break
        jd = EPOCH_JD + t / 86400.0
        if k % msis_every == 0:
            lat, lon, alt_m = geodetic(r, jd)
            f107, f107a, ap = weather(jd)
            try:
                rho, T, mbar = _msis(
                    alt_m / 1e3, np.degrees(lat), np.degrees(lon), jd, f107, f107a, ap,
                )
                sim.set_atmosphere(rho, T, mbar)
            except Exception as exc:
                print(name, "msis failed", exc, flush=True)
            sim.set_srp_scale(1.0)
        w = np.array([0.0, 0.0, OMEGA_EARTH])
        v_rel = v - np.cross(w, r)
        q = quat_ram(v_rel, d_body, np.cross(r, v))
        out = sim.step(q)
        alt_s = float(out["altitude_km"])
        lat, lon, alt_g = geodetic(np.asarray(out["r"], float), EPOCH_JD + float(out["t"]) / 86400.0)
        if alt_g / 1e3 <= ALT_STOP_KM and hit is None:
            hit = EPOCH_JD + float(out["t"]) / 86400.0
            print(f"{name} reached {ALT_STOP_KM:.0f} km geodetic at jd {hit:.5f}", flush=True)
            # log this last sample below, then stop after recording
        if k % log_every == 0 or hit is not None:
            r = np.asarray(out["r"], float)
            v = np.asarray(out["v"], float)
            t = float(out["t"])
            jd = EPOCH_JD + t / 86400.0
            a, e, inc, energy, h = orbital_elements(r, v, g.mu())
            fr = sample_forces(
                g, panels, moon_jd, moon_r, r, v, jd, q, rho, T, mbar,
                float(sim.params().srp_scale),
            )
            if prev is not None:
                dt = t - prev["t"]
                for key, F in (
                    ("fmf", fr["F_fmf"]), ("srp", fr["F_srp"]), ("erp", fr["F_erp"]),
                    ("harm", fr["F_harm"]),
                    ("moon", fr["F_moon"]), ("sun", fr["F_sun"]), ("tb", fr["F_tb"]),
                ):
                    F0 = prev["F"][key]
                    work[key] += 0.5 * dt * (float(np.dot(F, v)) + float(np.dot(F0, prev["v"])))
                    Jang[key] += 0.5 * dt * (np.cross(r, F) + np.cross(prev["r"], F0))
                    Plin[key] += 0.5 * dt * (F + F0)
            buf[n_row] = [
                t, jd, alt_g / 1e3, a / 1e3, e, inc, energy, np.linalg.norm(h),
                rho, f107, ap, fr["eclipse"], fr["Cd"], fr["Cl"], fr["ram"],
                np.linalg.norm(fr["F_fmf"]), fr["fmf_along"], np.linalg.norm(fr["F_srp"]),
                np.linalg.norm(fr["F_erp"]),
                np.linalg.norm(fr["F_harm"]), np.linalg.norm(fr["F_moon"]),
                np.linalg.norm(fr["F_sun"]),
            ]
            work_hist[n_row] = [work[k_] for k_ in keys]
            Jang_hist[n_row] = [np.linalg.norm(Jang[k_]) for k_ in keys]
            Plin_hist[n_row] = [np.linalg.norm(Plin[k_]) for k_ in keys]
            prev = {"t": t, "r": r, "v": v, "F": {
                "fmf": fr["F_fmf"], "srp": fr["F_srp"], "erp": fr["F_erp"],
                "harm": fr["F_harm"],
                "moon": fr["F_moon"], "sun": fr["F_sun"], "tb": fr["F_tb"],
            }}
            n_row += 1
            if t >= next_print:
                rate = (k + 1) / max(time.time() - t_wall, 1e-6)
                print(
                    f"{name} day {t/86400:.2f}  alt {alt_g/1e3:.1f} km  "
                    f"Fmf {np.linalg.norm(fr['F_fmf'])*1e6:.1f} uN  "
                    f"{rate:.0f} steps/s",
                    flush=True,
                )
                next_print = t + PRINT_EVERY_S
            if t >= next_save or hit is not None:
                _save_case(
                    out_dir, name, cols, buf[:n_row], work_hist[:n_row],
                    Jang_hist[:n_row], Plin_hist[:n_row], keys, hit, d_body,
                )
                next_save = t + SAVE_EVERY_S
            if hit is not None:
                break
        if alt_s < 80.0:
            break
    _save_case(
        out_dir, name, cols, buf[:n_row], work_hist[:n_row],
        Jang_hist[:n_row], Plin_hist[:n_row], keys, hit, d_body,
    )
    return hit


def _msis(alt_km, lat_deg, lon_deg, jd, f107, f107a, ap):
    import pymsis
    when = np.datetime64(jd_to_datetime(jd), "ns")
    result = pymsis.calculate(
        np.array([when]),
        np.array([lon_deg]),
        np.array([lat_deg]),
        np.array([alt_km]),
        f107s=np.array([f107]),
        f107as=np.array([f107a]),
        aps=np.array([[ap] * 7]),
        version=2.1,
    )
    data = result[0] if result.ndim == 2 else np.ravel(result)[:11]
    rho, T = float(data[0]), float(data[10])
    if not np.isfinite(rho) or not np.isfinite(T) or rho <= 0.0 or T <= 0.0:
        raise RuntimeError(f"bad MSIS rho={rho} T={T} alt={alt_km}")
    # Mean molecular mass, same species folding as atmosphere.query_msis.
    masses = {
        1: 28.014e-3, 2: 31.998e-3, 3: 15.999e-3, 4: 4.002602e-3,
        5: 1.008e-3, 6: 39.948e-3, 7: 14.007e-3,
    }
    NA = 6.02214076e23
    tot_n = 0.0
    tot_m = 0.0
    for i, M in masses.items():
        ni = float(data[i]) if i < len(data) else 0.0
        if np.isfinite(ni) and ni > 0.0:
            tot_n += ni
            tot_m += ni * (M / NA)
    if len(data) > 8:
        ao = float(data[8])
        if np.isfinite(ao) and ao > 0.0:
            tot_n += ao
            tot_m += ao * (15.999e-3 / NA)
    mbar = tot_m / tot_n if tot_n > 0 else 15.999e-3 / NA
    return rho, T, float(mbar)


def _save_case(out_dir, name, cols, rows, work, jang, plin, keys, hit, d_body):
    np.savez_compressed(
        out_dir / f"{name}.npz",
        cols=np.array(cols),
        rows=rows,
        work=work,
        jang=jang,
        plin=plin,
        work_keys=np.array(keys),
        hit_jd=np.array([-1.0 if hit is None else hit]),
        d_body=np.asarray(d_body, float),
    )


def plot_model(tris, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9.2, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    # Colour by which body axis the face most nearly faces.
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    n = n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-15)
    colors = np.zeros((len(tris), 4))
    palette = {
        0: (0.85, 0.55, 0.15, 0.95),  # X, panel normal
        1: (0.25, 0.55, 0.40, 0.90),  # Y
        2: (0.20, 0.40, 0.70, 0.90),  # Z, boom
    }
    dom = np.argmax(np.abs(n), axis=1)
    for k, col in palette.items():
        colors[dom == k] = col
    poly = Poly3DCollection(tris, facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(poly)
    pts = tris.reshape(-1, 3)
    c = np.array([0.0, 0.0, 0.0])
    span = np.max(np.linalg.norm(pts - pts.mean(0), axis=1))
    L = 0.55 * span
    axes = [("X  out of panel", [L, 0, 0], "C3"),
            ("Y", [0, L, 0], "C2"),
            ("Z", [0, 0, L], "C0")]
    for label, vec, col in axes:
        ax.quiver(*c, *vec, color=col, arrow_length_ratio=0.12, linewidth=2.0)
        ax.text(*(np.array(vec) * 1.08), label, color=col, fontsize=9)
    tip_z = pts[np.argmin(pts[:, 2])]
    tip_y = pts[np.argmin(pts[:, 1])]
    ax.text(*tip_z, "  boom (−Z)", color="C0", fontsize=10)
    ax.text(*tip_y, "  panel (−Y)", color="C2", fontsize=10)
    m = span * 1.05
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X body [m]")
    ax.set_ylabel("Y body [m]")
    ax.set_zlabel("Z body [m]")
    ax.set_title("CatSat exterior, 1% panels\n+X out of the solar panel, −Y toward the panel, −Z along the boom")
    fig.tight_layout()
    fig.savefig(out_dir / "model_body_axes.png", dpi=160)
    fig.savefig(out_dir / "model_body_axes.pdf")
    plt.close(fig)


def _utc_from_jd(jd):
    unix = (float(jd) - 2440587.5) * 86400.0
    return datetime.fromtimestamp(unix, tz=timezone.utc)


def plot_results(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = {}
    styles = (
        ("best", "#1f4e79"),
        ("nominal", "#2f6f4e"),
        ("worst", "#8c2f2f"),
    )
    for name, color in styles:
        path = out_dir / f"{name}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        cols = list(data["cols"])
        rows = data["rows"]
        cases[name] = {
            "c": color,
            "col": {k: rows[:, i] for i, k in enumerate(cols)},
            "work": data["work"],
            "jang": data["jang"],
            "plin": data["plin"] if "plin" in data.files else None,
            "keys": list(data["work_keys"]),
            "hit": float(data["hit_jd"][0]),
            "d": data["d_body"],
        }
    force_colors = {
        "fmf": "#8c2f2f",
        "srp": "#c47b00",
        "erp": "#b85c38",
        "harm": "#1f4e79",
        "moon": "#2f6f4e",
        "sun": "#6b4c9a",
        "tb": "#777777",
    }

    def days(case):
        return case["col"]["t_s"] / 86400.0

    def stride(case):
        n = len(case["col"]["t_s"])
        return 1 if n <= 40000 else int(np.ceil(n / 25000))

    epoch_label = f"Days from {EPOCH.strftime('%Y-%m-%d %H:%M')} UTC"

    def finish(fig, stem):
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        fig.savefig(out_dir / f"{stem}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for name, case in cases.items():
        step = stride(case)
        d = days(case)[::step]
        ax.plot(d, case["col"]["alt_km"][::step], color=case["c"], lw=1.3, label=f"{name} ram")
        if case["hit"] > 0:
            xhit = case["hit"] - EPOCH_JD
            ax.axvline(xhit, color=case["c"], ls="--", lw=0.8)
            when = _utc_from_jd(case["hit"]).strftime("%Y-%m-%d")
            ytxt = {"best": ALT_STOP_KM + 36, "nominal": ALT_STOP_KM + 18, "worst": ALT_STOP_KM + 4}[name]
            ax.text(xhit, ytxt, f" {name}\n {when}", color=case["c"], fontsize=8, va="bottom")
    ax.axhline(ALT_STOP_KM, color="k", lw=0.8, ls=":")
    ax.set_xlabel(epoch_label)
    ax.set_ylabel("Geodetic altitude [km]")
    ax.set_title("CatSat free-molecular decay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, "altitude")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for name, case in cases.items():
        step = stride(case)
        ax.plot(days(case)[::step], case["col"]["energy"][::step] / 1e6, color=case["c"], lw=1.2, label=name)
    ax.set_xlabel(epoch_label)
    ax.set_ylabel("Specific orbital energy [MJ/kg]")
    ax.set_title("Two-body specific energy  v²/2 − μ/r")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, "energy")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for name, case in cases.items():
        step = stride(case)
        ax.plot(days(case)[::step], case["col"]["h"][::step] / 1e10, color=case["c"], lw=1.2, label=name)
    ax.set_xlabel(epoch_label)
    ax.set_ylabel("|r × v|  [10¹⁰ m²/s]")
    ax.set_title("Specific orbital angular momentum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    finish(fig, "angular_momentum")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for name, case in cases.items():
        step = stride(case)
        d = days(case)[::step]
        ax.plot(d, np.clip(case["col"]["Ffmf"][::step] * 1e6, 1e-6, None), color=case["c"], lw=1.0, label=f"{name} |F|")
        ax.plot(d, np.clip(-case["col"]["Falong"][::step] * 1e6, 1e-6, None), color=case["c"], lw=0.8, ls="--",
                label=f"{name} along-track")
    ax.set_xlabel(epoch_label)
    ax.set_ylabel("Free-molecular force [µN]")
    ax.set_title("Sentman force (solid |F|, dashed opposing the relative wind)")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    finish(fig, "drag_force")

    # Work, linear impulse, and angular impulse. Two-body is plotted on its
    # own row: over an orbit its impulse and work nearly cancel, and that
    # oscillation is large enough to hide drag, SRP, and the third bodies.
    for stem, arr_name, ylab, title in (
        ("force_work", "work", "Cumulative work [J]",
         "Work done by each force"),
        ("force_linear_impulse", "plin", "Cumulative |∫ F dt|  [N·s]",
         "Linear impulse (change in linear momentum)"),
        ("force_angular_impulse", "jang", "Cumulative |∫ r × F dt|  [N·m·s]",
         "Angular impulse about the Earth (change in orbital angular momentum)"),
    ):
        names = [name for name in ("best", "nominal", "worst") if name in cases]
        fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 7.2), sharex=True, squeeze=False)
        for col, name in enumerate(names):
            case = cases[name]
            series = case[arr_name]
            if series is None:
                continue
            step = stride(case)
            d = days(case)[::step]
            for j, key in enumerate(case["keys"]):
                if key == "tb":
                    continue
                axes[0, col].plot(d, series[::step, j], color=force_colors[key], lw=1.2, label=key)
            jtb = case["keys"].index("tb")
            axes[1, col].plot(d, series[::step, jtb], color=force_colors["tb"], lw=1.0, label="two-body")
            axes[0, col].set_title(f"{name}  perturbations")
            axes[1, col].set_title(f"{name}  two-body")
            axes[1, col].set_xlabel("Days")
            axes[0, col].grid(True, alpha=0.3)
            axes[1, col].grid(True, alpha=0.3)
            axes[0, col].legend(fontsize=8)
            axes[1, col].legend(fontsize=8)
        axes[0, 0].set_ylabel(ylab)
        axes[1, 0].set_ylabel(ylab)
        fig.suptitle(title)
        finish(fig, stem)

    # Magnitudes of the perturbation forces (not two-body).
    names = [name for name in ("best", "nominal", "worst") if name in cases]
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, names):
        case = cases[name]
        step = stride(case)
        d = days(case)[::step]
        for key, col in (
            ("Fharm", "harm"), ("Fmoon", "moon"), ("Fsun", "sun"),
            ("Fsrp", "srp"), ("Ferp", "erp"), ("Ffmf", "fmf"),
        ):
            ax.plot(d, np.clip(case["col"][key][::step] * 1e6, 1e-6, None), color=force_colors[col], lw=1.0, label=col)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Days")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Force magnitude [µN]")
    fig.suptitle("Perturbation forces (two-body is ~100 N and is left off this scale)")
    finish(fig, "perturbation_forces")

    summary = {}
    for name, case in cases.items():
        hit = case["hit"]
        when = None if hit < 0 else _utc_from_jd(hit).strftime("%Y-%m-%d %H:%M UTC")
        days_life = None if hit < 0 else (hit - EPOCH_JD)
        summary[name] = {
            "decay_utc": when,
            "lifetime_days": days_life,
            "ram_direction_body": case["d"].tolist(),
            "final_alt_km": float(case["col"]["alt_km"][-1]),
            "final_energy_J_per_kg": float(case["col"]["energy"][-1]),
            "final_h_m2_per_s": float(case["col"]["h"][-1]),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("epoch", EPOCH.isoformat(), "JD", f"{EPOCH_JD:.8f}", flush=True)
    a, e, inc, energy, h = orbital_elements(R0, V0, cpp.MU_WGS)
    print(f"initial a={a/1e3:.2f} km e={e:.5f} i={inc:.3f} deg |h|={np.linalg.norm(h):.4e}", flush=True)

    jd_tab, r_tab, v_tab = parse_horizons()
    print("horizons |v| km/s", np.linalg.norm(v_tab, axis=1) / 1e3, flush=True)
    # Analytic comparison at the first sample.
    hat, rm = cpp.moon_analytic(float(jd_tab[0]))
    r_an = np.asarray(hat, float) * float(rm)
    print(f"Meeus vs Horizons at start: {np.linalg.norm(r_an - r_tab[0])/1e3:.1f} km", flush=True)
    jd_end = EPOCH_JD + MAX_DAYS + 2.0
    print("propagating Moon", flush=True)
    grid, r_grid, v_last = propagate_moon(jd_tab, r_tab, v_tab, jd_end)
    mx, mean = moon_table_error(jd_tab, r_tab, grid, r_grid)
    rr = np.linalg.norm(r_grid, axis=1) / 1e3
    print(f"table reproduction error max {mx:.3f} km, mean {mean:.3f} km", flush=True)
    print(f"Moon |r| over the grid {rr.min():.0f}..{rr.max():.0f} km", flush=True)
    if rr.min() < 3.0e5 or rr.max() > 5.0e5:
        raise RuntimeError("propagated Moon left the geocentric range 300000..500000 km")
    # Propagator-only error: integrate from sample 0 and compare at the last sample.
    # Reported so a drifting handoff is visible. The plant uses Hermite+propagation.
    np.savez_compressed(OUT / "moon_ephemeris.npz", jd=grid, r=r_grid)

    n, A, c, tris, meta = build_panels(OUT)
    print("panels", meta["n_panels"], "area", meta["area_m2"], "ram", meta["body_ram_m2"], flush=True)
    plot_model(np.asarray(tris, float), OUT)
    d_min, d_max, a_min, a_max = ram_extremes(n, A)
    print(f"best ram {d_min} area {a_min:.4f} m^2", flush=True)
    print(f"worst ram {d_max} area {a_max:.4f} m^2", flush=True)
    meta["best_ram"] = {"direction": d_min.tolist(), "area_m2": a_min}
    meta["worst_ram"] = {"direction": d_max.tolist(), "area_m2": a_max}
    meta["mass_kg"] = MASS_KG
    meta["epoch_jd"] = EPOCH_JD
    meta["moon_table_max_km"] = mx
    (OUT / "geometry.json").write_text(json.dumps(meta, indent=2))

    panels = (n, A, c)
    # Worst case is the short one; run it first so a lifetime exists early.
    run_case("worst", d_max, panels, grid, r_grid, OUT)
    run_case("best", d_min, panels, grid, r_grid, OUT)
    summary = plot_results(OUT)
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
