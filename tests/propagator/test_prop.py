"""Onboard FP32 propagator. Citations: Vallado 2013 eq. 8-30 (J2); Montenbruck
& Gill 2000 Sec. 3.5 (exponential drag)."""
import numpy as np
import pytest

from arlamx_v2 import cpp
from arlamx_v2 import propagator as prop

MU = 3.986004418e14
RE = 6378137.0


def _circ(alt):
    r = np.array([RE + alt, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, np.sqrt(MU / (RE + alt)), 0.0], dtype=np.float32)
    return r, v


def test_vacuum_energy_conservation():
    r0, v0 = _circ(400e3)
    r1, v1 = prop.propagate(r0, v0, 5400.0, dt_s=30.0, bc_inv=0.0)
    a0, a1 = prop.sma_m(r0, v0), prop.sma_m(r1, v1)
    # FP32 RK4: metres-level drift over one orbit is the honest budget.
    assert abs(a1 - a0) < 50.0


def test_j2_matches_cpp_closed_form():
    """The propagator's J2 must agree with the validated C++ accel_j2."""
    r = np.array([RE + 400e3, 1.2e6, 2.1e6], dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    a_prop = prop._accel(r, v, np.float32(0.0), np.float32(0.0),
                         np.float32(400e3), prop.H_SCALE_DEFAULT)
    a_cpp = (np.asarray(cpp.accel_twobody(r.astype(float), MU))
             + np.asarray(cpp.accel_j2(r.astype(float), MU, RE, 1.0826353865e-3)))
    assert np.allclose(a_prop, a_cpp, rtol=2e-6)   # float32 floor


def test_drag_decays_and_scales_with_bc():
    r0, v0 = _circ(400e3)
    d1 = prop.decay_rate(r0, v0, 21600.0, dt_s=60.0, bc_inv=1.0,
                                    rho0=3e-12, alt0_m=400e3)
    d2 = prop.decay_rate(r0, v0, 21600.0, dt_s=60.0, bc_inv=2.0,
                                    rho0=3e-12, alt0_m=400e3)
    assert d1 > 0 and d2 > 1.7 * d1       # ~linear in Cd*A/m over 6 h


def test_bc_inverse_recovery():
    """F = 0.5 rho Cd A v^2 must invert to Cd*A/m."""
    rho, v, m = 3e-12, 7670.0, 0.625
    cdam = 2.2 * 0.65 / m
    F = 0.5 * rho * cdam * m * v * v
    assert prop.bc_from_drag(F, rho, v, m) == pytest.approx(cdam, rel=1e-9)
    assert prop.bc_from_drag(1.0, 0.0, v, m) == 0.0      # degenerate


def test_samples_monotone_single_sweep():
    r0, v0 = _circ(350e3)
    s = prop.propagate_samples(r0, v0, [150.0, 300.0, 600.0], dt_s=30.0,
                               bc_inv=1.0, rho0=1e-11, alt0_m=350e3)
    alts = [float(np.linalg.norm(r)) - RE for r, _ in s]
    assert alts[0] > alts[1] > alts[2]    # decaying, and offsets in order
    assert all(x[0].dtype == np.float32 for x in s)


# ---------------------------------------------------------------------------
# v2.1: the C++ float32 mirror must reproduce the numpy reference. Both are
# IEEE binary32 evaluations of the same RK4; they differ only in operation
# ordering / fused multiply-add, i.e. at the float32 rounding floor (~0.5 m at
# LEO radius per operation), far below the 10 m GNSS noise the features carry.
# ---------------------------------------------------------------------------

def test_cpp_mirror_matches_numpy_reference():
    if getattr(prop, "_CPP_SAMPLES", None) is None:
        pytest.skip("compiled plant without propagate_f32_samples")
    r0 = np.array([RE + 380e3, 1.2e6, -0.9e6], dtype=np.float32)
    r0 *= np.float32((RE + 380e3) / np.linalg.norm(r0))
    v0 = np.cross([0.0, 0.0, 1.0], r0.astype(float)); v0 /= np.linalg.norm(v0)
    v0 = (v0 * np.sqrt(MU / (RE + 380e3))).astype(np.float32)
    offs = [150.0, 300.0, 450.0, 600.0, 750.0, 900.0]
    kw = dict(dt_s=30.0, bc_inv=2.2 * 0.65 / 0.625, rho0=8e-12, alt0_m=380e3)
    ref = prop.propagate_samples(r0, v0, offs, use_cpp=False, **kw)
    got = prop.propagate_samples(r0, v0, offs, use_cpp=True, **kw)
    assert len(ref) == len(got) == len(offs)
    for (rr, rv), (gr, gv) in zip(ref, got):
        assert gr.dtype == np.float32 and gv.dtype == np.float32
        assert np.linalg.norm(gr.astype(float) - rr.astype(float)) < 5.0      # m
        assert np.linalg.norm(gv.astype(float) - rv.astype(float)) < 5e-3     # m/s
    # the features built from it: altitude drop over 900 s must agree to ~1 m
    d_ref = np.linalg.norm(ref[-1][0].astype(float)) - np.linalg.norm(r0.astype(float))
    d_got = np.linalg.norm(got[-1][0].astype(float)) - np.linalg.norm(r0.astype(float))
    assert d_ref < -100.0                      # it is decaying (dense test atmosphere)
    assert abs(d_ref - d_got) < 2.0


def test_cpp_mirror_single_horizon_and_speed():
    if getattr(prop, "_CPP_SAMPLES", None) is None:
        pytest.skip("compiled plant without propagate_f32_samples")
    r0, v0 = _circ(400e3)
    a = prop.propagate(r0, v0, 21600.0, dt_s=30.0, bc_inv=1.0, rho0=3e-12, alt0_m=400e3)
    b = prop.propagate(r0, v0, 21600.0, dt_s=30.0, bc_inv=1.0, rho0=3e-12, alt0_m=400e3,
                       use_cpp=False)
    assert np.linalg.norm(a[0].astype(float) - b[0].astype(float)) < 30.0    # 6 h, 720 steps
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        prop.propagate_samples(r0, v0, [150.0, 300.0, 450.0, 600.0, 750.0, 900.0],
                               dt_s=30.0, bc_inv=1.0, rho0=3e-12, alt0_m=400e3)
    per_call = (time.perf_counter() - t0) / 20
    assert per_call < 2e-3, f"C++ propagator path took {per_call*1e3:.2f} ms"
