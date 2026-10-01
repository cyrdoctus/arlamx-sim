#!/usr/bin/env bash
# ARLAMX plant build. Uses the local mamba env named arlamx (Python 3.12).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

_is_py312() {
  local exe="$1"
  [ -x "$exe" ] || return 1
  "$exe" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null
}

_resolve_py() {
  if [ -n "${ARLAMX_PYTHON:-}" ]; then
    printf '%s\n' "$ARLAMX_PYTHON"
    return 0
  fi
  if [ -n "${MAMBA_ROOT_PREFIX:-}" ] && [ -x "${MAMBA_ROOT_PREFIX}/envs/arlamx/bin/python" ]; then
    printf '%s\n' "${MAMBA_ROOT_PREFIX}/envs/arlamx/bin/python"
    return 0
  fi
  if [ "${CONDA_DEFAULT_ENV:-}" = "arlamx" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return 0
  fi
  local cand
  for cand in \
    "${HOME}/.local/share/mamba/envs/arlamx/bin/python" \
    "${HOME}/miniforge3/envs/arlamx/bin/python" \
    "${HOME}/mambaforge/envs/arlamx/bin/python" \
    "${HOME}/miniconda3/envs/arlamx/bin/python" \
    "${HOME}/micromamba/envs/arlamx/bin/python"
  do
    if [ -x "$cand" ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

if ! PY="$(_resolve_py)"; then
  echo "error: no arlamx interpreter found." >&2
  echo "Use: mamba activate arlamx   or set ARLAMX_PYTHON to that env's python." >&2
  exit 1
fi
if [ ! -e "$PY" ]; then
  echo "error: interpreter path does not exist: $PY" >&2
  echo "If you set ARLAMX_PYTHON, it must be the arlamx env's python binary." >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "error: interpreter is not executable: $PY" >&2
  exit 1
fi
if ! _is_py312 "$PY"; then
  echo "error: resolved interpreter is not Python 3.12: $PY" >&2
  echo "Build only with the arlamx env (Python 3.12). Refuse generic python3 / ThesisMS / arlamx21." >&2
  exit 1
fi

# Use the env next to $PY so an unactivated build still gets that cmake/c++.
_ENV_BIN="$(cd "$(dirname "$PY")" && pwd)"
_ENV_PREFIX="$(cd "${_ENV_BIN}/.." && pwd)"
export PATH="${_ENV_BIN}:${PATH}"
if [ -z "${CMAKE:-}" ]; then
  if [ -x "${_ENV_BIN}/cmake" ]; then
    CMAKE="${_ENV_BIN}/cmake"
  else
    CMAKE="cmake"
  fi
fi
if [ ! -x "$CMAKE" ] && ! command -v "$CMAKE" >/dev/null 2>&1; then
  echo "error: cmake not found (looked at ${_ENV_BIN}/cmake and PATH)." >&2
  exit 1
fi

# A leftover cache from another interpreter (e.g. 3.11 / arlamx21) ignores
# -DPython_EXECUTABLE and FindPython reports the old ABI.
_CACHE="$ROOT/build/CMakeCache.txt"
if [ -f "$_CACHE" ]; then
  _cached="$(sed -n 's/^_Python_EXECUTABLE:INTERNAL=//p' "$_CACHE" | head -n 1 || true)"
  if [ -n "$_cached" ] && [ "$_cached" != "$PY" ]; then
    echo "note: dropping stale CMake cache (cached ${_cached}, now ${PY})" >&2
    rm -rf "$ROOT/build"
  fi
fi
# A renamed/moved checkout (e.g. v2.6 -> v2.7) leaves a cache for the old source dir.
if [ -f "$_CACHE" ]; then
  _src="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$_CACHE" | head -n 1 || true)"
  if [ -n "$_src" ] && [ "$_src" != "$ROOT/cpp" ]; then
    echo "note: dropping CMake cache from another source dir (${_src})" >&2
    rm -rf "$ROOT/build"
  fi
fi

"$CMAKE" -S "$ROOT/cpp" -B "$ROOT/build" \
  -DPython_EXECUTABLE="$PY" \
  -DPython_ROOT_DIR="$_ENV_PREFIX" \
  -DCMAKE_PREFIX_PATH="$_ENV_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build "$ROOT/build" -j"$(nproc)"
echo "Built: $ROOT/python/arlamx_v2/arlamx_cpp*.so  (python: $PY)"
