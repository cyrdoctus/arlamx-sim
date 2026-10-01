"""06 — MRP / Euler. Citations: Schaub & Junkins 2018 Ch.3–4; Shuster 1993."""
import math
import numpy as np
from arlamx_v2 import cpp


def test_zero_mrp_is_identity():
    C = cpp.mrp_to_dcm(np.zeros(3))
    assert np.allclose(C, np.eye(3), atol=1e-15)


def test_90deg_about_z():
    # σ = tan(Φ/4) ê, Φ=90° → tan(22.5°)
    s = np.array([0.0, 0.0, math.tan(math.radians(22.5))])
    C = cpp.mrp_to_dcm(s)
    x = C @ np.array([1.0, 0.0, 0.0])
    # σ_BN = +90° about z: inertial +X is −Y in the body (right-hand DCM_BN).
    assert abs(x[1] + 1.0) < 1e-12
    assert abs(x[0]) < 1e-12


def test_shadow_maps_outside_unit():
    s = np.array([2.0, 0.0, 0.0])
    sh = cpp.mrp_shadow(s)
    assert np.dot(sh, sh) <= 1.0 + 1e-15
    assert np.allclose(sh, np.array([-0.5, 0.0, 0.0]))


def test_compose_inverse_is_identity():
    s = np.array([0.1, -0.2, 0.05])
    err = cpp.mrp_error(s, s, True)
    assert np.linalg.norm(err) < 1e-14
