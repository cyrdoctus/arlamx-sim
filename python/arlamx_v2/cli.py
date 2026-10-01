"""`python -m arlamx_v2 ...` = `python main.py ...`."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2] / "main.py"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    return runpy.run_path(str(MAIN))["main"](argv)


if __name__ == "__main__":
    raise SystemExit(main())
