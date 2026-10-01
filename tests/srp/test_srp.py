"""02 — Panel SRP. Citations: Montenbruck & Gill 2000 §3.4; Vallado 2013 §8.6.4."""
import numpy as np
from arlamx_v2 import cpp

P = cpp.P_SRP_1AU


def test_face_on_absorber():
    n = np.array([[0.0, 0.0, 1.0]])
    A = np.array([1.0])
    c = np.array([[0.0, 0.0, 0.0]])
    sun = np.array([0.0, 0.0, 1.0])
    F = cpp.panel_srp_force(n, A, c, sun, 1.0, P, 1.0)
    assert abs(F[2] + P) < 1e-15
    assert abs(F[0]) < 1e-15 and abs(F[1]) < 1e-15


def test_edge_on_zero():
    n = np.array([[0.0, 0.0, 1.0]])
    A = np.array([1.0])
    c = np.zeros((1, 3))
    sun = np.array([1.0, 0.0, 0.0])
    F = cpp.panel_srp_force(n, A, c, sun, 1.0, P, 1.0)
    assert np.linalg.norm(F) < 1e-18


def test_closed_box_only_sunward():
    n = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    A = np.array([1.0, 1.0])
    c = np.array([[0.5, 0, 0], [-0.5, 0, 0]])
    sun = np.array([1.0, 0, 0])
    F = cpp.panel_srp_force(n, A, c, sun, 1.0, P, 1.0)
    # Only +X face, force along -sun
    assert F[0] < 0
    assert abs(F[0] + P) < 1e-15
