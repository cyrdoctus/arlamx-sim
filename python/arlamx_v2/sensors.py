"""Gyro, accelerometer, magnetometer and GNSS noise/bias models from datasheet YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable

import numpy as np

DEG = np.pi / 180.0
G0 = 9.80665  # m/s^2; the accel datasheet is written in g
MG_TO_T = 1e-7


@dataclass
class GyroConfig:
    arw_dps_rthz: float = 0.0028  # [SPEC] rate noise density, DS-000347
    turn_on_bias_dps: float = 0.5
    bias_tempco_dps_per_C: float = 0.005
    bias_tempco_poly: tuple = ()
    bias_temp_hook: Callable | None = None
    rrw_dps_per_s_rthz: float = 1e-5
    rrw_on_datasheet: bool = False  # [MEASURE] NOT a datasheet number - Allan-fit it in TVAC
    scale_factor_err: float = 0.005  # [SPEC] 0.5 %, drawn uniform +/- per axis per run
    misalign_deg: float = 0.5
    n_bits: int = 16  # [SPEC] ADC width over the selected FS (<=0 disables quantization)
    fs_dps: float = 2000.0  # [SPEC] selected full scale; also the saturation limit


@dataclass
class AccelConfig:
    noise_density_ug_rthz: float = 70.0  # [SPEC] DS-000347
    turn_on_bias_mg: float = 20.0
    bias_tempco_mg_per_C: float = 0.15
    n_bits: int = 16  # [SPEC]
    fs_g: float = 16.0  # [SPEC] selected full scale; also the saturation limit


@dataclass
class MagConfig:
    noise_mG_rms: float = 0.4  # [SPEC] total RMS noise ...
    noise_bw_hz: float = 100.0  # [SPEC] ... quoted at this bandwidth; D = rms/sqrt(bw)
    lsb_mG: float = 0.0625  # [SPEC] 18-bit output resolution
    fs_G: float = 8.0  # [SPEC] +/-8 G full scale; also the saturation limit
    nonlinearity_pct_fs: float = 0.1  # [SPEC] INL, modelled as an odd bow (see _nonlinearity)
    hard_iron_mG: float = 20.0
    soft_iron: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    sens_tempco_pct_per_C: float = 0.06
    dipole_coupling_mG_per_Am2: tuple = ((0.0,) * 3,) * 3

    def __post_init__(self):
        self.soft_iron_m = np.asarray(self.soft_iron, float).reshape(3, 3)
        self.dipole_m = np.asarray(self.dipole_coupling_mG_per_Am2, float).reshape(3, 3)


@dataclass
class GnssConfig:
    fix_rate_hz: float = 1.0  # [SPEC] Orion B16-01 nominal navigation rate
    ttff_s: float = 60.0
    pos_gm_sigma_m: float = 10.0
    pos_gm_tau_s: float = 300.0
    pos_white_m: float = 3.0
    vel_sigma_ms: float = 0.1


@dataclass
class SensorConfig:
    enabled: bool = True
    seed: int = 0
    temp_ref_C: float = 25.0  # [SPEC] temperature the datasheet 'typ' figures are quoted at
    gyro: GyroConfig = field(default_factory=GyroConfig)
    accel: AccelConfig = field(default_factory=AccelConfig)
    mag: MagConfig = field(default_factory=MagConfig)
    gnss: GnssConfig = field(default_factory=GnssConfig)

    @staticmethod
    def from_yaml(path) -> "SensorConfig":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return SensorConfig.from_dict(d)

    @staticmethod
    def from_dict(d) -> "SensorConfig":
        subs = {"gyro": GyroConfig, "accel": AccelConfig, "mag": MagConfig, "gnss": GnssConfig}
        kw = {k: (_build(subs[k], v, k) if k in subs else v) for k, v in d.items()}
        return _build(SensorConfig, kw, "sensors")


def _build(cls, d, where):
    if not isinstance(d, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(d).__name__}")
    known = {f.name for f in fields(cls)}
    bad = sorted(set(d) - known)
    if bad:
        raise ValueError(f"{where}: unknown key(s) {bad}; known: {sorted(known)}")
    return cls(**d)


# sigma = D / sqrt(2 dt): one-sided ASD band-limited to Nyquist (IEEE Std 952-1997, App. B);
# docs/modules/14_reward_v8.md §7.
def _white(rng, density, dt, n=3):
    return rng.normal(0.0, density / np.sqrt(2.0 * dt), n)


def _lsb(fs, n_bits):
    return 0.0 if n_bits <= 0 else 2.0 * fs / float(2**n_bits)


def _digitize(x, fs, lsb):
    x = np.clip(x, -fs, fs)
    return x if lsb <= 0.0 else np.round(x / lsb) * lsb


def _misalign_matrix(rng, sigma_deg):
    e = rng.normal(0.0, sigma_deg * DEG, 3)
    return np.array(
        [[1.0, -e[2], e[1]], [e[2], 1.0, -e[0]], [-e[1], e[0], 1.0]]
    )


def _temp_bias(c, temp_C, temp_ref_C):
    if c.bias_temp_hook is not None:
        return np.broadcast_to(np.asarray(c.bias_temp_hook(temp_C), float), (3,))
    dT = temp_C - temp_ref_C
    if len(c.bias_tempco_poly):
        return np.full(3, float(np.polyval(np.asarray(c.bias_tempco_poly, float), dT)))
    return np.full(3, c.bias_tempco_dps_per_C * dT)


def _nonlinearity(x, fs, pct_fs):
    if pct_fs <= 0.0:
        return np.zeros(3)
    u = np.clip(x / fs, -1.0, 1.0)
    return (0.01 * pct_fs * fs / 0.3849001794597505) * (u - u**3)


def _no_fix():
    return {"valid": False, "r": np.full(3, np.nan), "v": np.full(3, np.nan)}


class SensorSuite:

    def __init__(self, cfg: SensorConfig, seed: int | None = None):
        self.cfg = cfg
        self._seed = int(cfg.seed if seed is None else seed)
        self._run = -1
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed, self._run = int(seed), 0
        else:
            self._run += 1
        ss = np.random.SeedSequence([self._seed, self._run]).spawn(5)
        self.rng_gyro, self.rng_accel, self.rng_mag_a, self.rng_mag_b, self.rng_gnss = (
            np.random.default_rng(s) for s in ss
        )

        g, a, m = self.cfg.gyro, self.cfg.accel, self.cfg.mag
        self.gyro_bias = self.rng_gyro.normal(0.0, g.turn_on_bias_dps, 3)
        self.gyro_sf = self.rng_gyro.uniform(-g.scale_factor_err, g.scale_factor_err, 3)
        self.gyro_M = _misalign_matrix(self.rng_gyro, g.misalign_deg)
        self.gyro_rrw = np.zeros(3)
        self.accel_bias = self.rng_accel.normal(0.0, a.turn_on_bias_mg, 3)
        self.mag_hard_iron = [
            r.normal(0.0, m.hard_iron_mG, 3) for r in (self.rng_mag_a, self.rng_mag_b)
        ]
        self._gnss = {"t_on": None, "t_last": None, "err": np.zeros(3), "fix": None}

    def read_gyro(self, omega_true, dt_s, temp_C=20.0) -> np.ndarray:
        w = np.array(omega_true, float)
        if not self.cfg.enabled:
            return w
        c, dt = self.cfg.gyro, float(dt_s)
        x = (self.gyro_M @ w) / DEG
        x = x * (1.0 + self.gyro_sf)
        self.gyro_rrw += self.rng_gyro.normal(0.0, c.rrw_dps_per_s_rthz * np.sqrt(dt), 3)
        x = x + self.gyro_bias + self.gyro_rrw + _temp_bias(c, temp_C, self.cfg.temp_ref_C)
        x = x + _white(self.rng_gyro, c.arw_dps_rthz, dt)
        return _digitize(x, c.fs_dps, _lsb(c.fs_dps, c.n_bits)) * DEG

    def read_accel(self, accel_true, dt_s, temp_C=20.0) -> np.ndarray:
        a = np.array(accel_true, float)
        if not self.cfg.enabled:
            return a
        c, dt = self.cfg.accel, float(dt_s)
        x = a / G0
        x = x + 1e-3 * (self.accel_bias + c.bias_tempco_mg_per_C * (temp_C - self.cfg.temp_ref_C))
        x = x + _white(self.rng_accel, 1e-6 * c.noise_density_ug_rthz, dt)
        return _digitize(x, c.fs_g, _lsb(c.fs_g, c.n_bits)) * G0

    def read_mag(self, b_true_body, dt_s, temp_C=20.0, dipole_cmd=None):
        b = np.array(b_true_body, float)
        if not self.cfg.enabled:
            return b, b.copy()
        c, dt = self.cfg.mag, float(dt_s)
        fs_mG = c.fs_G * 1e3
        dens = c.noise_mG_rms / np.sqrt(c.noise_bw_hz)
        d = np.zeros(3) if dipole_cmd is None else np.asarray(dipole_cmd, float)
        gain = 1.0 + 0.01 * c.sens_tempco_pct_per_C * (temp_C - self.cfg.temp_ref_C)
        out = []
        for rng, hard_iron in zip((self.rng_mag_a, self.rng_mag_b), self.mag_hard_iron):
            x = (c.soft_iron_m @ b) / MG_TO_T * gain
            x = x + hard_iron + c.dipole_m @ d
            x = x + _nonlinearity(x, fs_mG, c.nonlinearity_pct_fs)
            x = x + _white(rng, dens, dt)
            out.append(_digitize(x, fs_mG, c.lsb_mG) * MG_TO_T)
        return out[0], out[1]

    def read_gnss(self, r_true, v_true, t_s, powered=True) -> dict:
        r, v, t = np.array(r_true, float), np.array(v_true, float), float(t_s)
        if not powered:
            self._gnss.update(t_on=None, t_last=None, err=np.zeros(3), fix=None)
            return _no_fix()
        if not self.cfg.enabled:
            return {"valid": True, "r": r, "v": v}

        c, g = self.cfg.gnss, self._gnss
        if g["t_on"] is None:
            g["t_on"] = t
            g["t_last"] = t
            g["err"] = self.rng_gnss.normal(0.0, c.pos_gm_sigma_m, 3)
            g["fix"] = None
        period = 1.0 / c.fix_rate_hz
        if g["fix"] is None or t - g["t_last"] + 1e-9 >= period:
            dt = period if g["fix"] is None else max(t - g["t_last"], 0.0)
            phi = np.exp(-dt / c.pos_gm_tau_s)
            q = c.pos_gm_sigma_m * np.sqrt(max(1.0 - phi * phi, 0.0))
            g["err"] = phi * g["err"] + self.rng_gnss.normal(0.0, q, 3)
            if t - g["t_on"] + 1e-9 < c.ttff_s:
                g["fix"] = _no_fix()
            else:
                g["fix"] = {
                    "valid": True,
                    "r": r + g["err"] + self.rng_gnss.normal(0.0, c.pos_white_m, 3),
                    "v": v + self.rng_gnss.normal(0.0, c.vel_sigma_ms, 3),
                }
            g["t_last"] = t
        f = g["fix"]
        return {"valid": f["valid"], "r": f["r"].copy(), "v": f["v"].copy()}
