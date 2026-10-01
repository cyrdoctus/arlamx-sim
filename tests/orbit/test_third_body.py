"""04 — Third body + eclipse. Citations: Montenbruck & Gill §3.3; Acton 1996."""
import numpy as np
from arlamx_v2 import cpp


def test_third_body_hand_eval():
    # Fake Sun on +X at 1 AU, sat at origin+offset.
    rs = np.array([6.778e6, 0.0, 0.0])
    rb = np.array([1.495978707e11, 0.0, 0.0])
    mu = 1.32712440018e20
    a = cpp.accel_third_body(rs, rb, mu)
    d = rb - rs
    hand = mu * (d / np.linalg.norm(d) ** 3 - rb / np.linalg.norm(rb) ** 3)
    assert np.linalg.norm(a - hand) / np.linalg.norm(hand) < 1e-11


def test_eclipse_behind_and_sunward():
    s = np.array([1.0, 0.0, 0.0])
    Re = cpp.RE_WGS
    r_dark = np.array([-2.0 * Re, 0.0, 0.0])
    r_lit = np.array([2.0 * Re, 0.0, 0.0])
    assert cpp.eclipse_cylindrical(r_dark, s, Re) == 0.0
    assert cpp.eclipse_cylindrical(r_lit, s, Re) == 1.0


def test_sun_analytic_unit():
    s = cpp.sun_unit_analytic(2461060.5)  # 2026-01-15
    assert abs(np.linalg.norm(s) - 1.0) < 1e-14


def _moon_track(n=400, step=0.25, jd0=2461060.5):
    """Sample the Moon over ~100 days (several synodic months)."""
    dec, dist = [], []
    for i in range(n):
        rh, rm = cpp.moon_analytic(jd0 + i * step)
        rh = np.asarray(rh)
        dec.append(np.degrees(np.arcsin(rh[2])))
        dist.append(rm)
    return np.array(dec), np.array(dist)


def test_moon_is_equatorial_not_ecliptic():
    """Meeus gives ecliptic (lon, lat); every consumer wants equatorial J2000.
    Without the obliquity rotation the lunar declination is capped at the
    +-5.3 deg ecliptic latitude instead of the true +-18..28 deg, which points
    the lunar third-body term in the wrong direction by up to ~20 deg."""
    dec, _ = _moon_track()
    assert dec.max() > 18.0, f"max declination {dec.max():.2f} deg — still ecliptic?"
    assert dec.min() < -18.0, f"min declination {dec.min():.2f} deg — still ecliptic?"
    # The Moon never leaves the +-(23.44 + 5.15) deg band.
    assert dec.max() < 29.0 and dec.min() > -29.0


def test_moon_distance_range():
    """Perigee/apogee must bracket the true 363 300 / 405 500 km."""
    _, dist = _moon_track()
    assert 3.55e8 < dist.min() < 3.70e8, f"perigee {dist.min()/1e3:.0f} km"
    assert 4.00e8 < dist.max() < 4.10e8, f"apogee {dist.max()/1e3:.0f} km"


def test_moon_unit_vector():
    rh, rm = cpp.moon_analytic(2461060.5)
    assert abs(np.linalg.norm(rh) - 1.0) < 1e-14
    assert np.isfinite(rm)
