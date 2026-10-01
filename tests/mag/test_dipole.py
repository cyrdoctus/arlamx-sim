"""09 — Geomagnetic field. Citations: Wertz 1978 App. H; Alken et al. 2021 (IGRF-13);
Chulliat et al. 2020 (WMM2020 Technical Report, eq. 5-6 recursion)."""
import numpy as np
import pytest
from arlamx_v2 import cpp

RE = cpp.RE_WGS
A_REF = 6371200.0  # IGRF/WMM reference radius the dipole is scaled on
# IGRF-13 / WMM2020 epoch-2020 degree-1 Gauss coefficients [nT].
G10, G11, H11 = -29404.8, -1450.9, 4652.5


def _mag_equator_point(alt=0.0):
    """A point on the geomagnetic equator: perpendicular to the dipole axis."""
    d = np.array([G11, H11, G10])
    dhat = d / np.linalg.norm(d)
    # Any vector orthogonal to the dipole axis lies on the magnetic equator.
    v = np.cross(dhat, [0.0, 0.0, 1.0])
    if np.linalg.norm(v) < 1e-9:
        v = np.cross(dhat, [1.0, 0.0, 0.0])
    v /= np.linalg.norm(v)
    return (A_REF + alt) * v


def test_dipole_points_north_at_equator():
    """Earth's dipole moment points toward geographic SOUTH (g10 < 0), so the
    field at the magnetic equator points NORTH. A +Z moment would invert every
    field direction the ADCS sees."""
    r = _mag_equator_point()
    B = np.asarray(cpp.dipole_field_ecef(r))
    assert B[2] > 0, "equatorial field must point north (+Z)"
    # On the magnetic equator the field is purely along the dipole axis (-d),
    # so the radial component vanishes.
    assert abs(np.dot(B, r / np.linalg.norm(r))) < 1e-3 * np.linalg.norm(B)


def test_dipole_axis_tilt_matches_igrf():
    """The implemented dipole must actually be tilted (spec 09), ~9.4 deg for
    IGRF-13 epoch 2020 — not the untilted +Z axis."""
    d = np.array([G11, H11, G10])
    tilt = np.degrees(np.arccos(abs(d[2]) / np.linalg.norm(d)))
    assert 9.0 < tilt < 11.5, f"dipole tilt {tilt:.2f} deg outside IGRF range"

    # The field on the geographic equator must therefore have a small but
    # non-zero radial component, unlike an untilted dipole.
    r = np.array([RE, 0.0, 0.0])
    B = np.asarray(cpp.dipole_field_ecef(r))
    br = abs(np.dot(B, r / np.linalg.norm(r)))
    assert br > 1e-3 * np.linalg.norm(B), "field looks untilted"


def test_dipole_magnitude_bands():
    """Magnetic-equator |B| equals |d|; the pole is twice that. Both fall as
    1/r^3."""
    d_nT = np.linalg.norm([G11, H11, G10])
    b_eq = np.linalg.norm(cpp.dipole_field_ecef(_mag_equator_point())) * 1e9
    assert abs(b_eq - d_nT) < 1.0, f"|B|_mag-equator {b_eq:.1f} != |d| {d_nT:.1f}"

    # LEO band: 500 km equatorial is ~24 uT, polar ~47 uT.
    b500 = np.linalg.norm(cpp.dipole_field_ecef(_mag_equator_point(500e3))) * 1e9
    assert 20e3 < b500 < 45e3, f"500 km equatorial |B| = {b500:.0f} nT"
    # 1/r^3 falloff.
    ratio = b_eq / b500
    assert abs(ratio - ((A_REF + 500e3) / A_REF) ** 3) < 1e-9


def test_dipole_inclination_sign_at_poles():
    """At the northern magnetic pole the field points DOWN (into the Earth)."""
    r = np.array([0.0, 0.0, RE + 400e3])
    B = np.asarray(cpp.dipole_field_ecef(r))
    assert np.dot(B, r) < 0, "field must dip inward over the north pole"


# --------------------------------------------------------------------------
# WMM. The old sanity band (10-80 uT) was ~8x wide and could not detect a 33 %
# error in the Legendre normalization, which is exactly the bug it missed.
# These pin the synthesis against an independent implementation instead.
# --------------------------------------------------------------------------

# Verified 2026-08-20 against a from-scratch Schmidt synthesis built on
# scipy.special.lpmv (agreement 0.0000 %). Points at 450 km, decimal year 2025.
WMM_REF_NT = {
    "equator_x": (11326.2280, -1648.7910, 22095.2865),
    "equator_y": (1176.8475, 9801.9205, 32179.8703),
    "near_pole": (-1185.5633, 72.3501, -46818.4824),
    "mid_lat": (-33921.5980, -17323.9504, -12168.7064),
    "south_atlantic": (12104.3812, -16192.5780, -771.7561),
}


def _wmm_points():
    """Spherical points at 450 km. The polar sample sits 0.1 deg off the axis:
    at the exact pole the Bphi/sin(theta) term is 0/0 and any finite-difference
    reference is degenerate, so the axis itself is covered by a separate
    finiteness test."""
    h = RE + 450e3

    def sph(lat_deg, lon_deg):
        la, lo = np.radians(lat_deg), np.radians(lon_deg)
        return h * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

    return {
        "equator_x": sph(0.0, 0.0),
        "equator_y": sph(0.0, 90.0),
        "near_pole": sph(89.9, 0.0),
        "mid_lat": sph(45.0, 30.0),
        "south_atlantic": sph(-40.0, -45.0),
    }


def _wmm():
    from conftest import wmm_path
    p = wmm_path()
    if not p:
        pytest.skip("WMM.COF not available")
    w = cpp.WMM()
    assert w.load(p)
    return w


def test_wmm_reference_points():
    """Pinned values — catches any regression in the Schmidt recursion."""
    w = _wmm()
    for name, r in _wmm_points().items():
        B = np.asarray(w.field_ecef(r, 2025.0)) * 1e9
        ref = np.array(WMM_REF_NT[name])
        err = np.linalg.norm(B - ref)
        assert err < 1.0, f"{name}: {err:.2f} nT from reference {ref} (got {B})"


def test_wmm_matches_independent_synthesis():
    """The real gate: rebuild the field from scipy's associated Legendre
    functions and compare. An independent normalization path, so a repeat of
    the double-normalization bug cannot pass."""
    sp = pytest.importorskip("scipy.special")
    w = _wmm()
    from conftest import wmm_path

    g = np.zeros((13, 13)); h = np.zeros((13, 13))
    gt = np.zeros((13, 13)); ht = np.zeros((13, 13))
    with open(wmm_path()) as f:
        epoch = float(f.readline().split()[0])
        for line in f:
            if "9999" in line:
                break
            p = line.split()
            if len(p) < 6:
                continue
            n, m = int(p[0]), int(p[1])
            if 1 <= n <= 12 and 0 <= m <= n:
                g[n][m], h[n][m], gt[n][m], ht[n][m] = map(float, p[2:6])

    def schmidt(n, m, x):
        # Strip the Condon-Shortley phase, then apply Schmidt semi-normalization.
        p = sp.lpmv(m, n, x) * (-1.0) ** m
        if m == 0:
            return p
        return p * np.sqrt(2.0 * sp.factorial(n - m) / sp.factorial(n + m))

    a, year, eps = 6371200.0, 2025.0, 1e-7
    for r_ecef in _wmm_points().values():
        x, y, z = r_ecef
        r = np.linalg.norm(r_ecef)
        theta = np.arctan2(np.hypot(x, y), z)
        lam = np.arctan2(y, x)
        ct, st = np.cos(theta), max(np.sin(theta), 1e-14)
        Br = Bt = Bp = 0.0
        for n in range(1, 13):
            arp = (a / r) ** (n + 2)
            for m in range(0, n + 1):
                gg = g[n][m] + gt[n][m] * (year - epoch)
                hh = h[n][m] + ht[n][m] * (year - epoch)
                gh = gg * np.cos(m * lam) + hh * np.sin(m * lam)
                P = schmidt(n, m, ct)
                dP = (schmidt(n, m, np.cos(theta + eps))
                      - schmidt(n, m, np.cos(theta - eps))) / (2 * eps)
                Br += (n + 1.0) * arp * gh * P
                Bt += -arp * gh * dP
                Bp += arp * m * (gg * np.sin(m * lam) - hh * np.cos(m * lam)) * P / st
        cl, sl = np.cos(lam), np.sin(lam)
        ref = np.array([Br * st * cl + Bt * ct * cl - Bp * sl,
                        Br * st * sl + Bt * ct * sl + Bp * cl,
                        Br * ct - Bt * st])
        got = np.asarray(w.field_ecef(r_ecef, year)) * 1e9
        assert np.linalg.norm(got - ref) < 1.0, f"{got} vs {ref}"


def test_wmm_exact_pole_is_finite():
    """The Bphi/sin(theta) term is 0/0 on the spin axis; the guard must keep it
    finite and of the right magnitude rather than producing NaN or a blow-up."""
    w = _wmm()
    h = RE + 450e3
    for r in (np.array([0.0, 0.0, h]), np.array([0.0, 0.0, -h])):
        B = np.asarray(w.field_ecef(r, 2025.0))
        assert np.all(np.isfinite(B))
        assert 10e3 < np.linalg.norm(B) * 1e9 < 80e3


def test_wmm_and_dipole_agree_in_bulk():
    """The tilted dipole is the degree-1 truncation of WMM and carries most of
    the field energy, so it must track WMM in the bulk. It is deliberately NOT
    a pointwise test: over the South Atlantic Anomaly the true field is ~40 %
    weaker than the dipole, and near the poles the horizontal component is
    small enough that direction is ill-conditioned."""
    w = _wmm()
    rng = np.random.default_rng(3)
    angs, ratios = [], []
    for _ in range(200):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        r = (RE + 450e3) * u
        bd = np.asarray(cpp.dipole_field_ecef(r))
        bw = np.asarray(w.field_ecef(r, 2025.0))
        cos = np.dot(bd, bw) / (np.linalg.norm(bd) * np.linalg.norm(bw))
        angs.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        ratios.append(np.linalg.norm(bd) / np.linalg.norm(bw))
    angs, ratios = np.array(angs), np.array(ratios)
    assert np.median(angs) < 12.0, f"median direction error {np.median(angs):.1f} deg"
    assert np.percentile(angs, 90) < 30.0
    assert 0.9 < np.median(ratios) < 1.15, f"median |B| ratio {np.median(ratios):.3f}"
