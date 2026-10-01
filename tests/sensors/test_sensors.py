"""11 — Sensor error injection (ICM-42688-P / 2x MMC5983MA / Orion B16-01).

Citations: IEEE Std 952-1997 App. B & C (overlapping Allan variance, ARW and
RRW coefficients); Groves 2013 §9.4 (first-order Gauss-Markov GNSS errors).

The gate that matters here is the noise-bandwidth convention: the env samples
these models at a 300 s step while the hardware runs at 100 Hz, so a stream
generated through the public API must show the injected datasheet density back
in its Allan deviation.
"""
from pathlib import Path

import numpy as np
import pytest

from arlamx_v2.sensors import (
    DEG,
    G0,
    AccelConfig,
    GnssConfig,
    GyroConfig,
    MagConfig,
    SensorConfig,
    SensorSuite,
)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "plant" / "sensors_solarcat.yaml"

W = np.array([0.01, -0.02, 0.005])  # rad/s
A = np.array([0.0, 0.0, -9.0])  # m/s^2
B = np.array([2.0e-5, -1.5e-5, 3.0e-5])  # T
R = np.array([6.8e6, 1.0e5, -2.0e5])  # m
V = np.array([-100.0, 7500.0, 30.0])  # m/s


def _quiet_gyro(**kw):
    """A gyro with every stochastic and per-run term switched off, so the one
    term under test is the only thing left in the output."""
    base = dict(
        arw_dps_rthz=0.0,
        turn_on_bias_dps=0.0,
        bias_tempco_dps_per_C=0.0,
        rrw_dps_per_s_rthz=0.0,
        scale_factor_err=0.0,
        misalign_deg=0.0,
    )
    base.update(kw)
    return GyroConfig(**base)


# --------------------------------------------------------------------------
# Pass-through and reproducibility
# --------------------------------------------------------------------------


def test_disabled_is_bit_for_bit_truth():
    """enabled=False must be a pure wire, not 'small' errors: the RL env
    compares noisy and noise-free rollouts, and a 1-LSB offset in the baseline
    would be charged to the policy."""
    s = SensorSuite(SensorConfig(enabled=False, seed=5))
    for dt in (0.01, 300.0):
        assert np.array_equal(s.read_gyro(W, dt), W)
        assert np.array_equal(s.read_accel(A, dt), A)
        a, b = s.read_mag(B, dt, dipole_cmd=[0.2, 0.0, -0.1])
        assert np.array_equal(a, B) and np.array_equal(b, B)
    fix = s.read_gnss(R, V, 0.0)
    assert fix["valid"] and np.array_equal(fix["r"], R) and np.array_equal(fix["v"], V)
    # The power gate is a spacecraft state, not an error term: it still bites.
    assert not s.read_gnss(R, V, 1.0, powered=False)["valid"]


def test_same_seed_same_stream():
    a = SensorSuite(SensorConfig(seed=42))
    b = SensorSuite(SensorConfig(seed=42))
    c = SensorSuite(SensorConfig(seed=43))
    for _ in range(20):
        ga, gb, gc = (s.read_gyro(W, 0.01) for s in (a, b, c))
        assert np.array_equal(ga, gb)
    assert not np.array_equal(ga, gc)
    # Explicit seed argument must override the config seed.
    d = SensorSuite(SensorConfig(seed=42), seed=43)
    assert np.array_equal(d.read_gyro(W, 0.01), SensorSuite(SensorConfig(seed=43)).read_gyro(W, 0.01))


def test_turn_on_terms_are_constant_within_a_run_and_redrawn_on_reset():
    """Turn-on bias must not wander inside an episode (the estimator would
    never converge) but must differ between episodes (or the policy learns one
    bias realization)."""
    s = SensorSuite(SensorConfig(seed=8, gyro=_quiet_gyro()))
    b0 = s.read_gyro(np.zeros(3), 0.01)
    s2 = SensorSuite(SensorConfig(seed=8, gyro=_quiet_gyro(turn_on_bias_dps=0.5)))
    first = s2.read_gyro(np.zeros(3), 0.01)
    for _ in range(50):
        assert np.array_equal(s2.read_gyro(np.zeros(3), 0.01), first)
        assert np.array_equal(s.read_gyro(np.zeros(3), 0.01), b0)  # zero bias config
    s2.reset()
    assert not np.array_equal(s2.read_gyro(np.zeros(3), 0.01), first)
    s2.reset(8)  # reset(seed) rewinds to run 0 == construction
    assert np.array_equal(s2.read_gyro(np.zeros(3), 0.01), first)


def test_from_yaml_round_trip_and_typo_rejection(tmp_path):
    cfg = SensorConfig.from_yaml(CONFIG)
    assert cfg.gyro.arw_dps_rthz == 0.0028
    assert cfg.gyro.rrw_on_datasheet is False  # the [MEASURE] flag must survive the load
    assert cfg.mag.lsb_mG == 0.0625
    assert cfg.gnss.pos_gm_tau_s == 300.0
    assert np.allclose(cfg.mag.soft_iron_m, np.eye(3))
    assert np.allclose(cfg.mag.dipole_m, 0.0)
    # A silently ignored typo in an error budget is worse than a crash.
    bad = tmp_path / "bad.yaml"
    bad.write_text("gyro:\n  arw_dps_rtHZ: 0.1\n")
    with pytest.raises(ValueError):
        SensorConfig.from_yaml(bad)


# --------------------------------------------------------------------------
# Noise bandwidth
# --------------------------------------------------------------------------


def test_white_noise_scales_with_requested_dt():
    """Convention (W1): sigma = D / sqrt(2*dt). The env steps at 300 s and the
    flight loop at 100 Hz; a model that ignored dt would hand the 300 s env the
    100 Hz noise, ~170x too much."""
    cfg = SensorConfig(seed=2, gyro=_quiet_gyro(arw_dps_rthz=0.0028, n_bits=0))
    s = SensorSuite(cfg)
    D = 0.0028
    for dt in (0.01, 1.0, 300.0):
        x = np.array([s.read_gyro(np.zeros(3), dt) for _ in range(6000)]) / DEG
        sigma = float(x.std())
        assert sigma == pytest.approx(D / np.sqrt(2.0 * dt), rel=0.05), f"dt={dt}"


# --------------------------------------------------------------------------
# Allan deviation
# --------------------------------------------------------------------------


def overlapping_adev(x, dt, taus):
    """Overlapping Allan deviation of a *rate* series, IEEE 952 eq. C-6:

        sigma^2(tau) = 1/(2 tau^2 (N-2m+1)) * sum (theta_{k+2m} - 2 theta_{k+m} + theta_k)^2

    with theta the integrated angle. Implemented here rather than pulled from
    allantools so the test carries no new dependency.
    """
    theta = np.concatenate(([0.0], np.cumsum(x) * dt))
    out = np.full(len(taus), np.nan)
    for i, tau in enumerate(taus):
        m = int(round(tau / dt))
        if m < 1 or len(theta) < 2 * m + 1:
            continue
        d = theta[2 * m :] - 2.0 * theta[m:-m] + theta[: -2 * m]
        out[i] = np.sqrt(np.sum(d * d) / (2.0 * tau * tau * len(d)))
    return out


def _gyro_stream(cfg, n, dt=0.01):
    """Native-rate stream through the public API, in deg/s. Truth is zero so
    scale factor and misalignment drop out and the Allan curve is pure error."""
    s = SensorSuite(cfg)
    z = np.zeros(3)
    return np.array([s.read_gyro(z, dt) for _ in range(n)]) / DEG


def test_allan_deviation_recovers_injected_arw():
    """Recover the datasheet noise density from the tau^-1/2 branch.

    From (W2), ADEV(tau) = D/sqrt(2 tau), so D = ADEV(tau)*sqrt(2 tau). The
    equivalent ARW coefficient is N = D/sqrt(2) = 0.00198 deg/s/sqrt(s), the
    usual one-sided-PSD factor.

    Run at the +/-250 dps ADCS full scale: at +/-2000 dps the 16-bit LSB is
    0.061 dps and quantization noise (LSB/sqrt(12) = 0.018 dps) is as large as
    the ARW itself, which is a real property of this part, not a model bug.
    Even at 250 dps quantization still inflates the recovered density by the
    predicted ~0.6 %, hence the 5 % tolerance.
    """
    dt, n = 0.01, 300_000  # 3000 s at the 100 Hz control-loop rate
    x = _gyro_stream(SensorConfig(seed=7, gyro=GyroConfig(fs_dps=250.0)), n, dt)
    taus = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    adev = np.mean([overlapping_adev(x[:, k], dt, taus) for k in range(3)], axis=0)

    D_hat = adev * np.sqrt(2.0 * taus)
    assert np.all(np.abs(D_hat / 0.0028 - 1.0) < 0.05), f"recovered D {D_hat}"
    # The branch must actually have the tau^-1/2 slope, not merely the right
    # value at one point.
    slope = np.polyfit(np.log(taus), np.log(adev), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.02), f"short-tau slope {slope:.3f}"

    # Sanity on the long-tau end of the same stream against the full model,
    # sqrt(D^2/(2 tau) + K^2 tau/3). Only ~10 independent clusters live at
    # tau = 300 s in a 3000 s record, so this band is deliberately wide; the
    # tight RRW check is the next test.
    long_taus = np.array([50.0, 100.0, 200.0, 300.0])
    long_adev = np.mean([overlapping_adev(x[:, k], dt, long_taus) for k in range(3)], axis=0)
    model = np.sqrt(0.0028**2 / (2.0 * long_taus) + 1e-5**2 * long_taus / 3.0)
    assert np.all(np.abs(long_adev / model - 1.0) < 0.4), f"{long_adev} vs {model}"


def test_allan_floor_matches_injected_rate_random_walk():
    """The bias-instability floor and the rising tau^+1/2 branch must come out
    of the injected RRW, K.

    The datasheet-default K = 1e-5 dps/s/sqrt(Hz) puts the floor at
    tau* = D*sqrt(1.5)/K = 343 s, which needs a multi-hour record to resolve.
    The floor is a property of the (W3) sqrt(dt) integration, not of K's
    magnitude, so K is raised to 2e-3 here to pull tau* down to 1.7 s and keep
    the test inside its time budget. K is recovered from the rising branch as
    K = ADEV(tau)*sqrt(3/tau).
    """
    dt, n, K, D = 0.01, 60_000, 2.0e-3, 0.0028  # 600 s
    cfg = SensorConfig(seed=4, gyro=GyroConfig(fs_dps=250.0, rrw_dps_per_s_rthz=K))
    x = _gyro_stream(cfg, n, dt)
    taus = np.array([0.05, 0.2, 1.0, 1.71, 3.0, 6.0, 12.0, 25.0])
    adev = np.mean([overlapping_adev(x[:, k], dt, taus) for k in range(3)], axis=0)

    tau_star = D * np.sqrt(1.5) / K
    floor = np.sqrt(D**2 / (2.0 * tau_star) + K**2 * tau_star / 3.0)
    assert adev.min() == pytest.approx(floor, rel=0.15), f"floor {adev.min():.5f} vs {floor:.5f}"
    assert taus[int(np.argmin(adev))] == pytest.approx(tau_star, rel=1.5)

    K_hat = float(np.mean(adev[-3:] * np.sqrt(3.0 / taus[-3:])))
    assert 0.7 * K < K_hat < 1.35 * K, f"recovered RRW {K_hat:.2e} vs injected {K:.2e}"
    # The curve must turn up again; a model missing the RRW would keep falling.
    assert adev[-1] > adev.min()


# --------------------------------------------------------------------------
# Magnetometers
# --------------------------------------------------------------------------


def test_two_magnetometers_are_independent():
    """Two parts on two buses. If one RNG draw were shared between them the
    redundant pair would agree perfectly and the fault-detection logic that
    differences them would never trip. Both the noise and the hard iron are
    checked, black-box, from the outputs only."""
    n = 4000
    for seed in (1, 2, 3):
        s = SensorSuite(SensorConfig(seed=seed))
        a, b = np.array([s.read_mag(B, 0.01) for _ in range(n)]).transpose(1, 0, 2)
        # Means carry the per-unit hard iron; residuals carry the noise.
        hi = (a.mean(axis=0) - b.mean(axis=0)) / 1e-7  # mG
        assert np.linalg.norm(hi) > 5.0, f"hard-iron offsets nearly identical: {hi} mG"
        da, db = a - a.mean(axis=0), b - b.mean(axis=0)
        for k in range(3):
            r = float(np.corrcoef(da[:, k], db[:, k])[0, 1])
            assert abs(r) < 0.08, f"axis {k}: units correlated, r = {r:.3f}"
        # ... and the noise must still be the datasheet noise, per unit.
        sigma = da.std(axis=0).mean() / 1e-7
        assert sigma == pytest.approx(0.4 / np.sqrt(100.0) / np.sqrt(2 * 0.01), rel=0.1)


def test_magnetorquer_coupling_hook():
    """Zero by default, but the interface has to exist: on SolarCat the rods
    sit inside the magnetometer baseline and a commanded dipole shows up in
    the field reading."""
    cfg = SensorConfig(seed=1, mag=MagConfig(noise_mG_rms=0.0, hard_iron_mG=0.0))
    s = SensorSuite(cfg)
    assert np.allclose(s.read_mag(B, 0.01, dipole_cmd=[1.0, -2.0, 0.5])[0], s.read_mag(B, 0.01)[0])

    coupled = MagConfig(
        noise_mG_rms=0.0,
        hard_iron_mG=0.0,
        nonlinearity_pct_fs=0.0,
        dipole_coupling_mG_per_Am2=((30.0, 0.0, 0.0), (0.0, 30.0, 0.0), (0.0, 0.0, 30.0)),
    )
    s = SensorSuite(SensorConfig(seed=1, mag=coupled))
    m = np.array([0.4, 0.0, 0.0])
    d = s.read_mag(B, 0.01, dipole_cmd=m)[0] - s.read_mag(B, 0.01)[0]
    assert d[0] == pytest.approx(30.0 * 0.4 * 1e-7, rel=0.02)  # 12 mG on the x channel
    assert np.allclose(d[1:], 0.0, atol=1e-10)


# --------------------------------------------------------------------------
# ADC behaviour
# --------------------------------------------------------------------------


def test_saturation_clips_at_full_scale():
    """Full scale is an exact multiple of the LSB, so a pinned channel reads
    exactly +/-FS -- and, crucially, a finite number rather than the truth."""
    s = SensorSuite(SensorConfig(seed=1))
    g = s.read_gyro(np.full(3, 1e5), 0.01)
    assert np.allclose(g, 2000.0 * DEG, rtol=1e-12)
    assert np.allclose(s.read_gyro(np.full(3, -1e5), 0.01), -2000.0 * DEG, rtol=1e-12)
    assert np.allclose(s.read_accel(np.full(3, 1e5), 0.01), 16.0 * G0, rtol=1e-12)
    a, b = s.read_mag(np.full(3, 1.0), 0.01)  # 1 T = 10 kG, way past the 8 G FS
    assert np.allclose(a, 8e-4, rtol=1e-12) and np.allclose(b, 8e-4, rtol=1e-12)


def test_quantization_lands_on_the_lsb_grid():
    cfg = SensorConfig(seed=1, gyro=_quiet_gyro(n_bits=16, fs_dps=2000.0))
    s = SensorSuite(cfg)
    lsb = 2.0 * 2000.0 / 2**16  # 0.061035 dps
    for truth_dps in (0.0, 0.013, -7.734, 123.456):
        out = s.read_gyro(np.full(3, truth_dps * DEG), 0.01) / DEG
        k = out / lsb
        assert np.allclose(k, np.round(k), atol=1e-9), f"{out} off the LSB grid"
        assert np.all(np.abs(out - truth_dps) <= lsb / 2 + 1e-12)

    # 18-bit mag: the datasheet quotes the LSB directly rather than a width.
    m = SensorSuite(SensorConfig(seed=1, mag=MagConfig(noise_mG_rms=0.0, hard_iron_mG=0.0)))
    q = np.asarray(m.read_mag(B, 0.01)[0]) / 1e-7 / 0.0625
    assert np.allclose(q, np.round(q), atol=1e-9)

    # n_bits <= 0 is the ideal-ADC escape hatch used by the noise tests.
    ideal = SensorSuite(SensorConfig(seed=1, gyro=_quiet_gyro(n_bits=0)))
    assert np.allclose(ideal.read_gyro(W, 0.01), W, rtol=1e-14)


def test_accel_bias_and_tempco_are_in_datasheet_units():
    """20 mg turn-on, 0.15 mg/degC. Getting the mg -> m/s^2 conversion wrong is
    a silent 1e-3 error that no other test would catch."""
    # n_bits=0: the 3 mg accel LSB would otherwise swallow the 3 mg tempco step.
    quiet = dict(noise_density_ug_rthz=0.0, turn_on_bias_mg=0.0, n_bits=0)
    s = SensorSuite(SensorConfig(seed=3, accel=AccelConfig(bias_tempco_mg_per_C=0.15, **quiet)))
    hot = s.read_accel(np.zeros(3), 0.01, temp_C=45.0)  # +20 degC over the 25 degC ref
    assert np.allclose(hot, 20.0 * 0.15e-3 * G0, atol=1e-9)
    assert np.allclose(s.read_accel(np.zeros(3), 0.01, temp_C=25.0), 0.0, atol=1e-12)

    draws = np.array([SensorSuite(
        SensorConfig(seed=k, accel=AccelConfig(noise_density_ug_rthz=0.0, bias_tempco_mg_per_C=0.0)),
    ).read_accel(np.zeros(3), 0.01) for k in range(200)])
    assert draws.std() == pytest.approx(20e-3 * G0, rel=0.2)  # 20 mg 1-sigma

    # 70 ug/rtHz through (W1), in m/s^2, with the ADC out of the way.
    n = SensorSuite(SensorConfig(seed=4, accel=AccelConfig(turn_on_bias_mg=0.0, n_bits=0)))
    x = np.array([n.read_accel(A, 0.01) for _ in range(6000)])
    assert x.std(axis=0).mean() == pytest.approx(70e-6 * G0 / np.sqrt(2 * 0.01), rel=0.06)


def test_gyro_temperature_hook():
    """The tempco is an [ASSUME]; the hook is what TVAC data gets dropped into,
    so both entry points have to work."""
    poly = SensorConfig(seed=1, gyro=_quiet_gyro(bias_tempco_poly=(0.001, 0.02, 0.0), n_bits=0))
    out = SensorSuite(poly).read_gyro(np.zeros(3), 0.01, temp_C=35.0) / DEG
    assert np.allclose(out, 0.001 * 100.0 + 0.02 * 10.0, atol=1e-9)  # dT = 10 degC

    g = _quiet_gyro(n_bits=0)
    g.bias_temp_hook = lambda t: np.array([0.1, 0.0, -0.1]) * (t - 25.0)
    out = SensorSuite(SensorConfig(seed=1, gyro=g)).read_gyro(np.zeros(3), 0.01, temp_C=30.0) / DEG
    assert np.allclose(out, [0.5, 0.0, -0.5], atol=1e-9)


# --------------------------------------------------------------------------
# GNSS
# --------------------------------------------------------------------------


def test_gnss_invalid_during_ttff_and_when_unpowered():
    """Downstream must be able to see the no-fix case: valid=False and NaN
    states, so a consumer that ignores the flag fails loudly."""
    s = SensorSuite(SensorConfig(seed=1))
    for t in (0.0, 30.0, 59.0):
        fix = s.read_gnss(R, V, t)
        assert not fix["valid"]
        assert np.all(np.isnan(fix["r"])) and np.all(np.isnan(fix["v"]))
    good = s.read_gnss(R, V, 61.0)
    assert good["valid"] and np.all(np.isfinite(good["r"]))
    assert np.linalg.norm(good["r"] - R) < 100.0  # errors are metres, not kilometres

    assert not s.read_gnss(R, V, 62.0, powered=False)["valid"]
    # Power cycling restarts the TTFF clock -- it does not resume where it left off.
    assert not s.read_gnss(R, V, 63.0)["valid"]
    assert not s.read_gnss(R, V, 100.0)["valid"]
    assert s.read_gnss(R, V, 130.0)["valid"]


def test_gnss_fix_is_latched_at_the_configured_rate():
    """1 Hz receiver polled at 10 Hz re-serves the same epoch."""
    s = SensorSuite(SensorConfig(seed=1))
    s.read_gnss(R, V, 0.0)
    first = s.read_gnss(R, V, 100.0)
    assert first["valid"]
    for k in range(1, 10):
        assert np.array_equal(s.read_gnss(R, V, 100.0 + 0.1 * k)["r"], first["r"])
    assert not np.array_equal(s.read_gnss(R, V, 101.0)["r"], first["r"])


def test_gnss_gauss_markov_correlation_decays_with_tau():
    """1-sigma budget = sqrt(10^2 + 3^2) = 10.44 m, correlated part decaying
    with tau = 300 s. The white component dilutes the lag-k correlation by
    sigma_gm^2 / (sigma_gm^2 + sigma_w^2) = 100/109, which the prediction must
    include -- otherwise the test would pass with the white term missing."""
    cfg = SensorConfig(seed=11)
    s = SensorSuite(cfg)
    err = []
    for t in range(40_000):
        fix = s.read_gnss(R, V, float(t))  # must be called every second: the
        if fix["valid"]:                   # Gauss-Markov state advances per fix
            err.append(fix["r"] - R)
    err = np.array(err)
    assert len(err) > 39_000
    assert err.std() == pytest.approx(np.hypot(10.0, 3.0), rel=0.15)

    x = err - err.mean(axis=0)
    dilution = 100.0 / 109.0

    def rho(lag):
        num = np.mean([np.dot(x[:-lag, k], x[lag:, k]) for k in range(3)])
        den = np.mean([np.dot(x[:, k], x[:, k]) for k in range(3)]) * (len(x) - lag) / len(x)
        return float(num / den)

    r100, r300, r900 = rho(100), rho(300), rho(900)
    assert r100 == pytest.approx(dilution * np.exp(-100.0 / 300.0), abs=0.10)
    assert r300 == pytest.approx(dilution * np.exp(-1.0), abs=0.12)
    assert r100 > r300 > r900
    assert abs(rho(1500)) < 0.15  # 5 tau: decorrelated

    tau_hat = -300.0 / np.log(r300 / dilution)
    assert 180.0 < tau_hat < 480.0, f"recovered tau {tau_hat:.0f} s"


def test_gnss_velocity_noise_is_white_at_the_configured_sigma():
    cfg = SensorConfig(seed=2, gnss=GnssConfig(ttff_s=0.0))
    s = SensorSuite(cfg)
    dv = np.array([s.read_gnss(R, V, float(t))["v"] - V for t in range(4000)])
    assert dv.std() == pytest.approx(0.1, rel=0.1)
    assert abs(np.corrcoef(dv[:-1, 0], dv[1:, 0])[0, 1]) < 0.06  # uncorrelated fix to fix
