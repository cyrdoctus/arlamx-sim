"""Classroom checks for the criticv1 plant fixes (F1-F3, F5-F9, F11, speed)."""
import math

import numpy as np
import pytest

from arlamx_v2 import cpp
from arlamx_v2 import propagator as prop
from arlamx_v2.sail_optics import PRESETS
from conftest import ggm_path

P = cpp.P_SRP_1AU
MU = cpp.MU_WGS
RE = cpp.RE_WGS
MYLAR = PRESETS["al_mylar"]


def plate(n, c=(0.0, 0.0, 0.0), A=1.0):
    return np.array([n], float), np.array([A]), np.array([c], float)


def hex_panels():
    from arlamx_v2.geometry import load_geom
    from arlamx_v2.paths import HEX_GEOM
    return load_geom(str(HEX_GEOM))


# F1 -------------------------------------------------------------------------

def test_f1_srp_torque_is_r_cross_f():
    n, A, c = plate([0, 0, 1], c=(0.01, 0, 0))
    F, tau = cpp.panel_srp_force_torque(n, A, c, np.array([0, 0, 1.0]), 1.0, P, 1.0)
    assert np.linalg.norm(F) == pytest.approx(P, rel=1e-15)
    assert np.linalg.norm(tau) == pytest.approx(0.01 * P, rel=1e-12)
    assert np.allclose(tau, np.cross([0.01, 0, 0], F), atol=0)
    _, tau0 = cpp.panel_srp_force_torque(*plate([0, 0, 1]), np.array([0, 0, 1.0]), 1.0, P, 1.0)
    assert np.all(tau0 == 0.0)
    Fo, to = cpp.panel_srp_optical(n, A, c, np.array([0, 0, 1.0]), 0.2, 0.7, 0.1, 1.0, P)
    assert np.allclose(to, np.cross([0.01, 0, 0], Fo), rtol=1e-14, atol=0)


def _sim(**kw):
    p = cpp.SimParams()
    p.dt_s = 2.0
    p.advisor_step_s = kw.pop("advisor_step_s", 2.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return cpp.Simulator(p)


def _circ(alt=400e3):
    return np.array([RE + alt, 0, 0.0]), np.array([0, math.sqrt(MU / (RE + alt)), 0.0])


def test_f1_plant_applies_srp_moment_and_holds_prescribed():
    sim = _sim(gravity_gradient=False, srp_optical=False, Cr=1.0)
    s = np.asarray(cpp.sun_unit_analytic(sim.params().epoch_jd), float)
    off = 0.01 * np.cross(s, [0, 0, 1.0]) / np.linalg.norm(np.cross(s, [0, 0, 1.0]))
    n, A, c = plate(s, c=off)                     # identity attitude: plate faces the Sun
    sim.set_panels(n, A, c)
    r, v = _circ()
    r = (RE + 400e3) * s                          # sub-solar side, sunlit
    v = np.cross([0, 0, 1.0], s); v = math.sqrt(MU / (RE + 400e3)) * v / np.linalg.norm(v)
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    sim.set_mode("prescribed")
    o = sim.step(np.array([1.0, 0, 0, 0]))
    assert o["n_sunlit"] == 1
    d2 = (1.0 / cpp.sun_dist_au(sim.params().epoch_jd)) ** 2
    F, tau = cpp.panel_srp_force_torque(n, A, c, s, 1.0, P * d2, 1.0)
    assert np.allclose(o["tau_srp_mean"], tau, rtol=1e-6)
    assert np.linalg.norm(o["tau_srp_mean"]) == pytest.approx(0.01 * P * d2, rel=1e-6)
    assert np.all(np.asarray(o["omega"]) == 0.0)
    assert np.allclose(o["tau_env_mean"], np.asarray(o["tau_aero_mean"]) + o["tau_srp_mean"]
                       + o["tau_erp_mean"] + o["tau_gg_mean"], rtol=0, atol=1e-30)


# F2 -------------------------------------------------------------------------

def _optical_hand(n, A, s, part, p):
    ca, cs, cd = part["ca"], part["cs"], part["cd"]
    F = np.zeros(3)
    for ni, ai in zip(n, A):
        ct = float(ni @ s)
        if ct > 1e-12:
            F += -p * ai * ct * ((ca + cd) * s + (2 * cs * ct + 2 * cd / 3) * ni)
    return F


def test_f2_face_on_hex_optical_mylar():
    n, A, c = hex_panels()
    s = np.array([0, 0, 1.0])
    F, _ = cpp.panel_srp_optical(n, A, c, s, MYLAR["ca"], MYLAR["cs"], MYLAR["cd"], 1.0, P)
    assert np.allclose(F, _optical_hand(n, A, s, MYLAR, P), rtol=1e-12)
    assert np.linalg.norm(F) == pytest.approx(5.692e-6, rel=1e-3)


def test_f2_edge_cases():
    n, A, c = plate([0, 0, 1])
    F90, _ = cpp.panel_srp_optical(n, A, c, np.array([1.0, 0, 0]), 0.08, 0.88, 0.04, 1.0, P)
    assert np.all(F90 == 0.0)
    Fcb = cpp.panel_srp_force(n, A, c, np.array([0, 0, 1.0]), 1.0, P, 1.0)
    assert np.linalg.norm(Fcb) == pytest.approx(P, rel=1e-15)


def test_f2_sun_distance_series():
    jd_peri = 2461043.5 + 17.0 / 24.0     # 2026-01-03 17 UT perihelion
    jd_aph = 2461227.5 + 18.0 / 24.0      # 2026-07-06 18 UT aphelion
    assert cpp.sun_dist_au(jd_peri) == pytest.approx(0.98330, abs=2e-4)
    assert cpp.sun_dist_au(jd_aph) == pytest.approx(1.01666, abs=2e-4)


def test_f2_presets_are_optical_mylar():
    from arlamx_v2 import physics
    for name in ("fast", "standard", "high"):
        cfg = physics.load(name)
        assert cfg["srp"]["optical"] is True and cfg["srp"]["optics"] == "al_mylar"
        p = physics.apply_to_params(cpp.SimParams(), cfg, ggm_path="")
        assert (p.srp_ca, p.srp_cs, p.srp_cd) == (MYLAR["ca"], MYLAR["cs"], MYLAR["cd"])
        assert p.earth_rad is True and p.gravity_gradient is True


# F3 -------------------------------------------------------------------------

def test_f3_eclipse_now_does_not_zero_future_sunlit_generation():
    from arlamx_v2.env import ArlamxV2Env
    env = ArlamxV2Env(seed=0, variant="v9d")
    env.reset(seed=0)
    soc, gen_n, _ = env._update_power(np.array([0, 0, 1.0]), 0.0, 0.0, 0.0, 0.0, False)
    assert gen_n == 0.0 and env._illum_geo == 1.0
    R = RE + 400e3
    a = math.radians(21.0)                        # inside the cylinder, sun along +x
    r = np.array([-R * math.sin(a), R * math.cos(a), 0.0])
    vt = math.sqrt(MU / R)
    v = np.array([vt * math.cos(a), vt * math.sin(a), 0.0])   # heading out of shadow
    sun = np.array([1.0, 0, 0])
    assert cpp.eclipse_cylindrical(r, sun, RE) == 0.0
    blk = env._future_block(r, v, sun, 3e-12, 400.0, 0.5, 0.0, 0.0, vt)
    env._illum_geo = 0.0
    blk0 = env._future_block(r, v, sun, 3e-12, 400.0, 0.5, 0.0, 0.0, vt)
    assert blk[-2] > blk0[-2]                     # +last SoC: generation counted when lit


# F5 -------------------------------------------------------------------------

def test_f5_gravity_gradient():
    I = np.array([0.0125, 0.0125, 0.025])
    r = RE + 450e3
    assert np.all(np.asarray(cpp.gg_torque(I, np.array([r, 0, 0]), MU)) == 0.0)
    r45 = r * np.array([math.sqrt(0.5), 0, math.sqrt(0.5)])
    tau = np.asarray(cpp.gg_torque(I, r45, MU))
    assert np.linalg.norm(tau) == pytest.approx(3 * MU / r**3 * 0.5 * abs(I[2] - I[0]), rel=1e-12)


# F6 -------------------------------------------------------------------------

IR = (0.85, 0.0, 0.15)
SOL = (MYLAR["ca"], MYLAR["cs"], MYLAR["cd"])


def test_f6_earth_ir_pushes_nadir_plate_outward_in_eclipse():
    nadir = np.array([0, 0, 1.0])                 # body +z points at the Earth
    r = RE + 400e3
    F, _ = cpp.earth_rad(*plate(nadir), nadir, r, -1.0, P, SOL, IR)
    ca, cs, cd = IR
    mag = P * 0.25 * cpp.EARTH_EMISS * (RE / r) ** 2 * ((ca + cd) + 2 * cs + 2 * cd / 3)
    assert np.allclose(F, -mag * nadir, rtol=1e-12)
    Fspace, _ = cpp.earth_rad(*plate(-nadir), nadir, r, 1.0, P, SOL, IR)
    assert np.all(Fspace == 0.0)


def test_f6_albedo_only_on_day_side():
    nadir = np.array([0, 0, 1.0])
    r = RE + 400e3
    Fn, _ = cpp.earth_rad(*plate(nadir), nadir, r, 0.0, P, SOL, IR)
    Fd, _ = cpp.earth_rad(*plate(nadir), nadir, r, 1.0, P, SOL, IR)
    ca, cs, cd = SOL
    alb = P * cpp.EARTH_ALBEDO * (RE / r) ** 2 * ((ca + cd) + 2 * cs + 2 * cd / 3)
    assert np.allclose(Fd - Fn, -alb * nadir, rtol=1e-12)


def test_f6_switch_leaves_solar_term_alone():
    outs = []
    for flag in (False, True):
        sim = _sim(earth_rad=flag, gravity_gradient=False, srp_optical=True,
                   srp_ca=SOL[0], srp_cs=SOL[1], srp_cd=SOL[2])
        sim.set_panels(*hex_panels())
        r, v = _circ()
        sim.reset(r, v, np.zeros(3), np.zeros(3))
        sim.set_mode("prescribed")
        outs.append(sim.step(np.array([1.0, 0, 0, 0])))
    assert np.all(outs[0]["tau_erp_mean"] == 0.0)
    assert np.array_equal(outs[0]["tau_srp_mean"], outs[1]["tau_srp_mean"])


# F7 -------------------------------------------------------------------------

def _drag_with_wind(wind):
    sim = _sim(gravity_gradient=False, use_panel_srp=False)
    n, A, c = plate([0, 1.0, 0])
    sim.set_panels(n, A, c)
    sim.set_atmosphere(3e-12, 900.0, 2.656e-26)
    r, v = _circ()
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    sim.set_mode("prescribed")
    if wind is not None:
        sim.set_wind(np.asarray(wind, float))
    return sim.step(np.array([1.0, 0, 0, 0]))


def test_f7_wind_hook():
    base = _drag_with_wind(None)
    zero = _drag_with_wind([0, 0, 0])
    assert base["drag_N"] == zero["drag_N"]
    opposing = _drag_with_wind([0, -100.0, 0])    # spacecraft flies +y
    assert opposing["drag_N"] > base["drag_N"]


# F8 -------------------------------------------------------------------------

def test_f8_forecast_srp_energy_and_zero_srp_identity():
    R = RE + 450e3
    r0 = np.array([R, 0, 0], np.float32)
    v0 = np.array([0, math.sqrt(MU / R), 0], np.float32)
    polar_sun = np.array([0, 0, 1.0])             # orbit never in shadow
    kw = dict(dt_s=10.0, bc_inv=0.0, rho0=0.0, alt0_m=450e3)
    T, a = 600.0, -1e-3
    for use_cpp in (False, True):
        r1, v1 = prop.propagate(r0, v0, T, use_cpp=use_cpp, a_srp=a, sun=polar_sun, **kw)
        r2, v2 = prop.propagate(r0, v0, T, use_cpp=use_cpp, **kw)
        e = lambda r, v: 0.5 * float(np.dot(v, v)) - MU / float(np.linalg.norm(r))
        de = e(r1.astype(float), v1.astype(float)) - e(r2.astype(float), v2.astype(float))
        assert de == pytest.approx(a * math.sqrt(MU / R) * T, rel=5e-3)
        r3, v3 = prop.propagate(r0, v0, T, use_cpp=use_cpp, a_srp=0.0, sun=polar_sun, **kw)
        assert np.array_equal(r3, r2) and np.array_equal(v3, v2)


def test_f8_drag_is_along_track():
    o = _drag_with_wind(None)
    assert o["drag_N"] > 0.0


# F9 -------------------------------------------------------------------------

def test_f9_altitude_is_geodetic():
    p = cpp.SimParams()
    p.ggm_path = ggm_path() or ""
    sim = cpp.Simulator(p)
    r = np.array([0, 0, RE + 400e3])
    sim.reset(r, np.array([7600.0, 0, 0]), np.zeros(3), np.zeros(3))
    h = cpp.ecef_to_geodetic(r)[2] / 1e3
    assert sim.get_state()["altitude_km"] == h
    assert h == pytest.approx(421.4, abs=0.1)


# F11 ------------------------------------------------------------------------

def test_f11_ap_history():
    pytest.importorskip("pymsis")
    from datetime import datetime
    from arlamx_v2.atmosphere import query_msis
    t = datetime(2026, 3, 1)
    import pymsis
    a = query_msis(400.0, 10.0, 20.0, t, 150.0, 15.0)
    ref = pymsis.calculate(np.array([np.datetime64(t, "ns")]), [20.0], [10.0], [400.0],
                           [150.0], [150.0], [[15.0] * 7], version=2.1)
    assert a[0] == float(np.ravel(ref)[0])        # scalar = steady-Ap fill, unchanged
    quiet = query_msis(400.0, 10.0, 20.0, t, 150.0, [15.0] * 7)
    storm = query_msis(400.0, 10.0, 20.0, t, 150.0, [15, 180, 150, 120, 80, 40, 15])
    assert storm[0] > quiet[0]
    with pytest.raises(ValueError):
        query_msis(400.0, 10.0, 20.0, t, 150.0, [15.0] * 3)


# speed: gamma > 6 limit -----------------------------------------------------

def _sentman_ref(theta, s, tw, ae):
    g = s * math.cos(theta)
    Z, E = 1 + math.erf(g), math.exp(-g * g)
    cp_i = ((g * g + 0.5) * Z + g * E / math.sqrt(math.pi)) / s**2
    ct = math.sin(theta) * (g * Z + E / math.sqrt(math.pi)) / s
    tr = ae * tw + (1 - ae) * s * s / 2
    cp_r = math.sqrt(tr) * (E + math.sqrt(math.pi) * g * Z) / (2 * s**2)
    return cp_i + cp_r, ct


@pytest.mark.parametrize("gamma", [0.5, 3.0, 5.9, 6.0, 6.1, 7.5, 8.0])
def test_sentman_gamma_limit(gamma):
    s = 8.0
    th = math.acos(gamma / s)
    cp, ct = cpp.sentman(th, s, 1 / 3, 0.93)
    rcp, rct = _sentman_ref(th, s, 1 / 3, 0.93)
    assert abs(cp - rcp) <= 1e-15 * abs(rcp)
    assert abs(ct - rct) <= 1e-15 * max(abs(rct), 1e-300)


def _drag_interp(rho0, rho1=None):
    sim = _sim(gravity_gradient=False, use_panel_srp=False, advisor_step_s=20.0)
    sim.set_panels(*plate([0, 1.0, 0]))
    sim.set_atmosphere(rho0, 900.0, 2.656e-26)
    if rho1 is not None:
        sim.set_atmosphere_end(rho1, 900.0, 2.656e-26)
    r, v = _circ()
    sim.reset(r, v, np.zeros(3), np.zeros(3))
    sim.set_mode("prescribed")
    o1 = sim.step(np.array([1.0, 0, 0, 0]))
    o2 = sim.step(np.array([1.0, 0, 0, 0]))
    return o1["drag_N"], o2["drag_N"]


def test_f11_density_interpolated_across_substeps():
    n = 10                                        # 20 s / 2 s substeps
    lo, hi = 2e-12, 4e-12
    mix, after = _drag_interp(lo, hi)
    f = (n - 1) / (2 * n)                         # mean of s/n, s = 0..n-1
    ref, _ = _drag_interp(lo + f * (hi - lo))
    assert mix == pytest.approx(ref, rel=1e-4)
    held, _ = _drag_interp(hi)
    assert after == pytest.approx(held, rel=1e-3)  # end sample is held afterwards
