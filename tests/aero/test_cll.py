"""Walker–CLL classroom checks (docs/CLL_TEST_SPEC.md T-CLL-1/4/5)."""
import math

import numpy as np
import pytest

from arlamx_v2 import cpp


def test_tcll1_fully_accommodated_identity():
    s, Tw = 8.0, 1.0
    for theta in (0.0, math.pi / 4):
        Cp_c, Ct_c = cpp.cll(theta, s, Tw, 1.0, 1.0)
        Cp_s, Ct_s = cpp.sentman(theta, s, Tw, 1.0)
        assert abs(Cp_c - Cp_s) < 1e-12
        assert abs(Ct_c - Ct_s) < 1e-12


def test_tcll4_mixture_differs_from_pure_o():
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    A = np.array([1.0, 1.0])
    c = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]])
    v = np.array([0.0, 0.0, -7500.0])
    from arlamx_v2.atmosphere import SPECIES_MASS
    rho, T, mb, Tw = 1e-12, 900.0, SPECIES_MASS["O"], 300.0
    chi_o = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    chi_mix = np.array([0.15, 0.80, 0.05, 0.0, 0.0, 0.0, 0.0])
    kw = dict(n=n, A=A, c=c, v_rel_B=v, rho=rho, T=T, m_bar=mb, T_w=Tw,
              alpha_E=0.93, one_sided_ref=True, gsi="cll", alpha_n=0.93,
              alpha_t=1.0)
    o_only = cpp.spacecraft_aero(**kw, chi=chi_o)
    missing = cpp.spacecraft_aero(**kw, chi=None)
    mixed = cpp.spacecraft_aero(**kw, chi=chi_mix)
    assert math.isfinite(o_only["Cd"]) and math.isfinite(mixed["Cd"])
    assert abs(o_only["Cd"] - missing["Cd"]) < 1e-12
    assert abs(mixed["Cd"] - o_only["Cd"]) > 1e-6


def test_tcll5_unknown_gsi_throws():
    p = cpp.SimParams()
    p.gsi = "maxwell"
    with pytest.raises(Exception):
        cpp.Simulator(p)
    n = np.array([[0.0, 0.0, 1.0]])
    A = np.array([1.0])
    c = np.array([[0.0, 0.0, 0.0]])
    v = np.array([0.0, 0.0, -7500.0])
    with pytest.raises(Exception):
        cpp.spacecraft_aero(n, A, c, v, 1e-12, 900.0, 2.656e-26, 300.0,
                            gsi="foo")


def test_walker_branch_differs_from_schaaf():
    Cp_w, _ = cpp.cll(0.0, 8.0, 1.0 / 3.0, 0.93, 1.0)
    Cp_s, _ = cpp.cll(0.0, 8.0, 1.0 / 3.0, 1.0, 1.0)
    assert abs(Cp_w - Cp_s) > 0.01
