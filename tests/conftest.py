import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from arlamx_v2.paths import resolve_ggm, resolve_wmm  # noqa: E402

# v2.1: the data files are resolved through arlamx_v2.paths (env var, repo-local
# data/, then the historical Basilisk install paths) instead of one hard-coded
# machine path, so the gates run on any checkout that ships data/.
GGM = Path(resolve_ggm()) if resolve_ggm() else Path("/nonexistent/GGM03S.txt")
WMM = Path(resolve_wmm()) if resolve_wmm() else Path("/nonexistent/WMM.COF")
SPICE_DIR = Path("/home/nekolny/Thesis/basilisk/dist3/Basilisk/supportData/EphemerisData")


def ggm_path():
    return str(GGM) if GGM.exists() else None


def wmm_path():
    return str(WMM) if WMM.exists() else None
