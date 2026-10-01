"""13 — GGM03S vs Basilisk referee (plant does not use Basilisk).

The reference model is SphericalHarmonicsGravityModel from Basilisk,
Autonomous Vehicle Systems Laboratory, University of Colorado Boulder
(https://github.com/AVSLab/basilisk, ISC). Cite that model with the
spherical-harmonic segments in cpp/src/orbit/gravity.cpp.

Other citations: GGM03S / Tapley; Vallado 2013 §8; Montenbruck & Gill §3.2.
"""
import numpy as np
import pytest

pytest.importorskip("Basilisk")

from Basilisk.simulation.sphericalHarmonicsGravityModel import SphericalHarmonicsGravityModel
from arlamx_v2 import cpp
from conftest import ggm_path


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_sh_matches_basilisk_computeField(degree):
    path = ggm_path()
    if not path:
        pytest.skip("GGM03S data file missing")
    g = cpp.GravityHarmonics()
    assert g.load_ggm(path, degree)
    sh = SphericalHarmonicsGravityModel()
    sh.loadFromFile(path, degree)
    sh.initializeParameters()

    pts = [
        np.array([g.Re() + 400e3, 0.0, 0.0]),
        np.array([5.0e6, 1.2e6, 4.0e6]),
        # 0.1° off the pole: spherical (λ,φ) is regular; the exact axis
        # needs a Pines evaluator (queued). az already matches at the pole.
        np.array([
            (g.Re() + 500e3) * 0.001745328365898,  # sin(0.1°)
            0.0,
            (g.Re() + 500e3) * 0.999998476913288,
        ]),
    ]
    for r in pts:
        a_ref = np.array(sh.computeField(r.tolist()), dtype=float).reshape(3)
        a_ours = np.array(g.accel_ecef(r, degree), dtype=float)
        rel = np.linalg.norm(a_ours - a_ref) / np.linalg.norm(a_ref)
        assert rel < 1e-8, f"rel={rel} at r={r} deg={degree}\n ours={a_ours}\n bsk={a_ref}"
