"""Data-file resolution (GGM, WMM, hex geom, STL) and the outputs/ layout."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_V2 = Path(__file__).resolve().parents[2]
DATA = ROOT_V2 / "data"


def _env_path(var):
    v = os.environ.get(var, "").strip()
    return Path(v) if v else None


_V17_CANDIDATES = (
    _env_path("ARLAMX_V17"),
    Path("/mnt/storage/ARLAMX_Dev/Thesis/ARLAMX-V1.7"),
    Path.home() / "Documents" / "Code" / "ARLAMX_Dev" / "Thesis" / "ARLAMX-V1.7",
)
V17 = next((p for p in _V17_CANDIDATES if p is not None and p.is_dir()), _V17_CANDIDATES[1])

GGM_CANDIDATES = (
    _env_path("ARLAMX_GGM"),
    DATA / "GGM03S.txt",
    Path("/home/nekolny/Thesis/basilisk/dist3/Basilisk/supportData/LocalGravData/GGM03S.txt"),
    Path("/home/nekolny/Thesis/basilisk/supportData/LocalGravData/GGM03S.txt"),
    V17 / "support" / "GGM03S.txt",
)
WMM_CANDIDATES = (
    _env_path("ARLAMX_WMM"),
    DATA / "WMM.COF",
    Path("/home/nekolny/Thesis/basilisk/dist3/Basilisk/supportData/MagneticField/WMM.COF"),
)
_HEX_CANDIDATES = (
    _env_path("ARLAMX_HEX_GEOM"),
    DATA / "earthcup_hex_v3.geom",
    V17 / "geometry" / "models" / "earthcup_hex_v3.geom",
)
_STL_CANDIDATES = (
    _env_path("ARLAMX_SOLARCAT_STL"),
    DATA / "SolarCat_Assembly.STL",
    V17 / "geometry" / "models" / "SolarCat_Assembly.STL",
)


def _first(cands, default):
    for p in cands:
        if p is not None and str(p) not in ("", ".") and p.is_file():
            return p
    return default


HEX_GEOM = _first(_HEX_CANDIDATES, _HEX_CANDIDATES[-1])
SOLARCAT_STL = _first(_STL_CANDIDATES, _STL_CANDIDATES[-1])


def resolve_ggm() -> str:
    p = _first(GGM_CANDIDATES, None)
    return str(p) if p is not None else ""


def resolve_wmm() -> str:
    p = _first(WMM_CANDIDATES, None)
    return str(p) if p is not None else ""


OUTPUTS = ROOT_V2 / "outputs"
OUTPUTS_OLD = OUTPUTS / ".old"
OUT_SNAPSHOTS = OUTPUTS / "snapshots"
OUT_MODELS = OUTPUTS / "models"
OUT_RESULTS = OUTPUTS / "results"
OUT_TRAINING = OUTPUTS_OLD / "training"
OUT_TUNING = OUTPUTS_OLD / "tuning"
OUT_ANALYSIS = OUTPUTS_OLD / "analysis"
OUT_LOGS = OUTPUTS_OLD / "logs"


def run_dir(name: str) -> Path:
    for base in (OUT_MODELS, OUT_TRAINING, OUT_TUNING,
                 OUTPUTS_OLD / "campaign" / "runs", OUTPUTS_OLD):
        cand = base / name
        if cand.exists():
            return cand
    return OUT_MODELS / name
