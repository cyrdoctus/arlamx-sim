"""ARLAMX package: loads the compiled C++ plant as `cpp`."""

import sys
from pathlib import Path

__version__ = "2.7.5"


def _import_cpp():
    try:
        from . import arlamx_cpp as _cpp
        return _cpp
    except ImportError as exc:
        pkg = Path(__file__).resolve().parent
        sos = sorted(pkg.glob("arlamx_cpp*.so")) + sorted(pkg.glob("arlamx_cpp*.pyd"))
        abi = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        matching = [p.name for p in sos if abi in p.name]
        leftovers = [p.name for p in sos if abi not in p.name]
        if leftovers and not matching:
            names = ", ".join(leftovers)
            raise ImportError(
                f"arlamx_cpp ABI mismatch: found {names}, but this interpreter "
                f"is Python {sys.version_info.major}.{sys.version_info.minor} "
                f"({sys.executable}). Activate the arlamx env (Python 3.12) and "
                f"rebuild: mamba activate arlamx && ./build.sh"
            ) from exc
        raise


cpp = _import_cpp()

__all__ = ["cpp", "__version__"]
