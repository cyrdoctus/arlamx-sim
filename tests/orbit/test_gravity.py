"""03 — Gravity. Citations: Vallado 2013 §8; Montenbruck & Gill 2000 §3.2."""
import numpy as np
import pytest
from arlamx_v2 import cpp
from conftest import ggm_path


def test_point_mass_400km():
    r = np.array([cpp.RE_WGS + 400e3, 0.0, 0.0])
    a = cpp.accel_twobody(r, cpp.MU_WGS)
    expect = -cpp.MU_WGS / np.dot(r, r)
    assert abs(a[0] / expect - 1.0) < 1e-14
    assert abs(a[1]) < 1e-18 and abs(a[2]) < 1e-18


def test_j2_closed_form_finite():
    r = np.array([4.5e6, 1.0e6, 5.0e6])
    a = cpp.accel_j2(r, cpp.MU_GGM, cpp.RE_GGM, cpp.J2_GGM)
    assert np.all(np.isfinite(a))
    assert np.linalg.norm(a) > 0


def test_ggm_degree2_zonal_matches_j2():
    path = ggm_path()
    if not path:
        pytest.skip("GGM03S.txt not found (Basilisk supportData used as data)")
    g = cpp.GravityHarmonics()
    assert g.load_ggm(path, 2)
    # Point off equator so J2 has a z component.
    r = np.array([5.0e6, 0.0, 4.5e6])
    a_j2 = cpp.accel_twobody(r, g.mu()) + cpp.accel_j2(r, g.mu(), g.Re(), g.J2())
    # SH with only C20: load degree 2 then we still have C21/C22. Compare J2 of file.
    # Isolated zonal: build by subtracting tesseral? Instead check J2 conversion.
    assert abs(g.J2() / cpp.J2_GGM - 1.0) < 1e-6
    a_sh = g.accel_ecef(r, 2)
    # Degree-2 full field vs two-body+J2 should be close (C22 is ~2e-6 vs J2 1e-3).
    rel = np.linalg.norm(a_sh - a_j2) / np.linalg.norm(a_j2)
    assert rel < 5e-3


def test_degree70_loads_and_moves_the_field():
    """70x70 is the library cap. It must load, stay finite, and not clamp to 24."""
    path = ggm_path()
    if not path:
        pytest.skip("GGM03S.txt not found")
    assert cpp.SH_MAX_DEGREE >= 70
    g = cpp.GravityHarmonics()
    assert g.load_ggm(path, 70)
    assert g.max_degree() == 70
    clamped = cpp.GravityHarmonics()
    assert clamped.load_ggm(path, 90)
    assert clamped.max_degree() == cpp.SH_MAX_DEGREE
    pts = [
        np.array([g.Re() + 400e3, 0.0, 0.0]),
        np.array([5.0e6, 1.2e6, 4.0e6]),
        np.array([0.0, 0.0, g.Re() + 400e3]),
        np.array([
            (g.Re() + 500e3) * np.sin(np.radians(0.1)),
            0.0,
            (g.Re() + 500e3) * np.cos(np.radians(0.1)),
        ]),
    ]
    for r in pts:
        a70 = np.array(g.accel_ecef(r, 70), float)
        a24 = np.array(g.accel_ecef(r, 24), float)
        assert np.all(np.isfinite(a70))
        delta = np.linalg.norm(a70 - a24)
        # Degrees 25..70 move LEO acceleration by tens of µm/s², far below two-body.
        assert 1e-8 < delta < 1e-3, f"delta={delta} at r={r}"


def test_polar_finite():
    path = ggm_path()
    if not path:
        pytest.skip("GGM03S.txt not found")
    g = cpp.GravityHarmonics()
    assert g.load_ggm(path, 8)
    r = np.array([0.0, 0.0, cpp.RE_GGM + 400e3])
    a = g.accel_ecef(r, 8)
    assert np.all(np.isfinite(a))
    assert a[2] < 0
