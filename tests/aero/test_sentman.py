"""01 — Sentman classroom tests. Citations: Sentman 1961; Moe & Moe 2005."""
import math
import numpy as np
from arlamx_v2 import cpp


def test_hyperthermal_identity_large_s():
    s = 80.0
    theta = 0.4
    Cp, Ctau = cpp.sentman(theta, s, 1.0, 1.0)
    combo = Cp * math.cos(theta) + Ctau * math.sin(theta)
    # Incident identity is 2 cosθ; diffuse re-emission adds O(1/s).
    assert abs(combo - 2.0 * math.cos(theta)) < 0.03


def test_m01_alpha_one_matches_tw():
    # At alpha_E=1, Tr/Ti = Tw/Ti. Face-on, compare two Tw/Ti values exist and are finite.
    Cp1, _ = cpp.sentman(0.0, 8.0, 1.0, 1.0)
    Cp2, _ = cpp.sentman(0.0, 8.0, 0.3, 1.0)
    assert math.isfinite(Cp1) and math.isfinite(Cp2)
    assert Cp1 != Cp2


def test_m01_alpha_zero_uses_kinetic_energy():
    # alpha_E=0 => Tr/Ti = s^2/2. Distinct from the old 1 + alpha*(Tw/Ti-1)=1.
    s = 8.0
    Cp_new, _ = cpp.sentman(0.0, s, 1.0, 0.0)
    Cp_acc, _ = cpp.sentman(0.0, s, 1.0, 1.0)
    # Unaccommodated re-emission is much hotter (s^2/2 = 32 vs Tw/Ti=1).
    assert Cp_new > Cp_acc


def test_unit_square_face_on_cd():
    # Two-sided unit square, one_sided A_ref = 1. Flow along -n of +z face.
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    A = np.array([1.0, 1.0])
    c = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]])
    v = np.array([0.0, 0.0, -7500.0])  # gas onto +z
    rho, T, mb, Tw = 1e-11, 1000.0, 2.656e-26, 1000.0
    out = cpp.spacecraft_aero(n, A, c, v, rho, T, mb, Tw, 1.0, True)
    assert 1.8 < out["Cd"] < 2.4
    assert out["A_ref"] == 1.0
