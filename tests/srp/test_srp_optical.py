"""Optical plate SRP. Montenbruck & Gill 2000 §3.4; Vallado 2013 §8.6.4."""

import numpy as np

from arlamx_v2 import cpp

P = cpp.P_SRP_1AU


def _plate():
    return np.array([[0.0, 0.0, 1.0]]), np.array([1.0]), np.zeros((1, 3))


def test_absorber_along_sun():
    n, A, c = _plate()
    sun = np.array([0.0, 0.0, 1.0])
    F = cpp.panel_srp_optical(n, A, c, sun, 1.0, 0.0, 0.0)[0]
    assert abs(F[2] + P) < 1e-15
    assert abs(F[0]) < 1e-15 and abs(F[1]) < 1e-15


def test_perfect_specular_along_normal():
    n, A, c = _plate()
    sun = np.array([0.0, 0.0, 1.0])
    F = cpp.panel_srp_optical(n, A, c, sun, 0.0, 1.0, 0.0)[0]
    assert abs(F[2] + 2.0 * P) < 1e-15
    assert abs(F[0]) < 1e-15 and abs(F[1]) < 1e-15


def test_lambert_has_normal_lobe():
    n, A, c = _plate()
    sun = np.array([0.0, 0.0, 1.0])
    F = cpp.panel_srp_optical(n, A, c, sun, 0.0, 0.0, 1.0)[0]
    assert abs(F[2] + P * (1.0 + 2.0 / 3.0)) < 1e-14


def test_tilt_specular_not_along_sun():
    n, A, c = _plate()
    ang = np.radians(30.0)
    sun = np.array([np.sin(ang), 0.0, np.cos(ang)])
    F = cpp.panel_srp_optical(n, A, c, sun, 0.0, 1.0, 0.0)[0]
    assert abs(F[0]) < 1e-15
    assert F[2] < 0.0
