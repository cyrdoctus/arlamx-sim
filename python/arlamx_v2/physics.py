"""Physics presets (fast/standard/high): load, validate, apply to the C++ SimParams."""
from __future__ import annotations

import copy

from arlamx_v2.config import load as load_cfg, merge
from arlamx_v2.paths import resolve_ggm, resolve_wmm

PRESETS = {
    "standard": "physics",
    "default": "physics",
    "fast": "physics_fast",
    "high": "physics_high",
}

GRAVITY_MODELS = ("ggm03s", "twobody")
ATMO_MODELS = ("msis21", "nrlmsise00", "constant")
GSI_MODELS = ("sentman", "cll")
MAG_MODELS = ("wmm", "dipole")
ATT_LAWS = ("mrp", "quaternion")


def load(name_or_path="standard", overrides=None):
    if isinstance(name_or_path, dict):
        cfg = copy.deepcopy(name_or_path)
        if overrides:
            cfg = merge(cfg, overrides)
        return validate(cfg)
    key = str(name_or_path or "standard").strip()
    if key in PRESETS:
        cfg = load_cfg(PRESETS[key], use_cache=False)
    else:
        cfg = load_cfg(key, use_cache=False)
    if overrides:
        cfg = merge(cfg, overrides)
    return validate(cfg)


def validate(cfg):
    cfg = copy.deepcopy(cfg)
    g = cfg.get("gravity") or {}
    a = cfg.get("atmosphere") or {}
    aero = cfg.get("aero") or {}
    mag = cfg.get("magnetics") or {}
    att = cfg.get("attitude") or {}
    srp = cfg.get("srp") or {}
    integ = cfg.get("integrator") or {}

    model = str(g.get("model", "ggm03s")).lower()
    if model not in GRAVITY_MODELS:
        raise ValueError(f"gravity.model must be one of {GRAVITY_MODELS}, got {model!r}")
    deg = int(g.get("degree", 4))
    if deg < 0 or deg > 24:
        raise ValueError(f"gravity.degree must be in 0..24, got {deg}")
    g["model"] = model
    cfg["gravity"] = g

    am = str(a.get("model", "msis21")).lower()
    if am not in ATMO_MODELS:
        raise ValueError(f"atmosphere.model must be one of {ATMO_MODELS}, got {am!r}")
    if am == "constant":
        c = a.get("constant") or {}
        for k in ("rho", "T", "m_bar"):
            if k not in c:
                raise ValueError("atmosphere.constant needs rho, T, m_bar")
    a["model"] = am
    cfg["atmosphere"] = a

    gsi = str(aero.get("gsi", "sentman")).lower()
    if gsi not in GSI_MODELS:
        raise ValueError(f"aero.gsi must be one of {GSI_MODELS}, got {gsi!r}")
    aero["gsi"] = gsi
    cfg["aero"] = aero

    mm = str(mag.get("model", "wmm")).lower()
    if mm not in MAG_MODELS:
        raise ValueError(f"magnetics.model must be one of {MAG_MODELS}, got {mm!r}")
    mag["model"] = mm
    cfg["magnetics"] = mag

    law = str(att.get("law", "mrp")).lower()
    if law not in ATT_LAWS:
        raise ValueError(f"attitude.law must be one of {ATT_LAWS}, got {law!r}")
    att["law"] = law
    ideal = att.get("ideal_torque", False)
    if not isinstance(ideal, bool):
        raise ValueError("attitude.ideal_torque must be true/false")
    att["ideal_torque"] = ideal
    cfg["attitude"] = att

    if "enabled" not in srp:
        raise ValueError("srp.enabled is required")
    from arlamx_v2.sail_optics import PRESETS as OPTICS
    if srp.get("optical") and str(srp.get("optics", "absorber")) not in OPTICS:
        raise ValueError(f"srp.optics must be one of {sorted(OPTICS)}")
    ir = srp.get("ir") or {}
    if ir and not all(k in ir for k in ("ca", "cs", "cd")):
        raise ValueError("srp.ir needs ca, cs, cd")
    rk4 = float(integ.get("rk4_step_s", 2.0))
    if not (0.0 < rk4 <= 120.0):
        raise ValueError("integrator.rk4_step_s must be in (0, 120]")
    return cfg


def msis_version(cfg):
    model = str((cfg.get("atmosphere") or {}).get("model", "msis21")).lower()
    if model == "nrlmsise00":
        return 0
    return 2.1


def apply_to_params(params, cfg, ggm_path=None, wmm_path=None, controller=None):
    g = cfg.get("gravity") or {}
    a = cfg.get("atmosphere") or {}
    aero = cfg.get("aero") or {}
    srp = cfg.get("srp") or {}
    mag = cfg.get("magnetics") or {}
    att = cfg.get("attitude") or {}
    integ = cfg.get("integrator") or {}

    gmodel = str(g.get("model", "ggm03s")).lower()
    params.sh_degree = 0 if gmodel == "twobody" else int(g.get("degree", 4))
    params.lunisolar = bool(g.get("lunisolar", False))
    params.corotating = bool(a.get("corotating", True))
    params.gsi = str(aero.get("gsi", "sentman")).lower()
    params.alpha_E = float(aero.get("alpha_E", 0.93))
    params.alpha_n = float(aero.get("alpha_n", 1.0))
    params.alpha_t = float(aero.get("alpha_t", 1.0))
    params.T_w = float(aero.get("T_w", 300.0))
    params.one_sided_ref = bool(aero.get("one_sided_ref", True))
    params.use_panel_srp = bool(srp.get("enabled", True))
    params.srp_optical = bool(srp.get("optical", False))
    params.Cr = float(srp.get("Cr", 1.8))
    if params.srp_optical:
        from arlamx_v2.sail_optics import PRESETS as OPTICS
        o = OPTICS[str(srp.get("optics", "absorber"))]
        params.srp_ca, params.srp_cs, params.srp_cd = o["ca"], o["cs"], o["cd"]
    params.earth_rad = bool(srp.get("earth", False))
    ir = srp.get("ir") or {}
    if ir:
        params.ir_ca, params.ir_cs, params.ir_cd = float(ir["ca"]), float(ir["cs"]), float(ir["cd"])
    params.gravity_gradient = bool(g.get("gradient_torque", True))
    params.rk4_step_s = float(integ.get("rk4_step_s", 2.0))
    law = str(controller or att.get("law") or "mrp").lower()
    if law not in ATT_LAWS:
        raise ValueError(f"controller/attitude.law must be one of {ATT_LAWS}")
    params.attitude_law = law
    params.ideal_torque = bool(att.get("ideal_torque", False))

    params.ggm_path = str(ggm_path if ggm_path is not None else resolve_ggm())

    mag_model = str(mag.get("model", "wmm")).lower()
    if mag_model == "dipole":
        params.wmm_path = ""
    else:
        params.wmm_path = str(wmm_path if wmm_path is not None else resolve_wmm())
    return params


def gsi_name(cfg):
    return str((cfg.get("aero") or {}).get("gsi", "sentman")).lower()


def constant_atmo(cfg):
    c = (cfg.get("atmosphere") or {}).get("constant") or {}
    return float(c["rho"]), float(c["T"]), float(c["m_bar"])
