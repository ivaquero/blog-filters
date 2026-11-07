#!/usr/bin/env bash
# Regenerate every figure that the tutorial notebooks produce.
#
# blog-filters keeps its notebooks as the source of truth: `jupyter-book build
# --execute` (Jupyter Book 2 / MyST) runs each notebook top-to-bottom and
# re-renders the inline outputs. Executed outputs are cached under
# `_build/execute`, so a second run only re-executes notebooks whose code
# changed -- delete that directory to force a full re-run.
#
# Usage:
#   ./scripts/generate_figures.sh              # execute everything, build the site
#   ./scripts/generate_figures.sh --clean      # drop the execution cache first
#
# Requires: `jupyter-book>=2` plus `jupyter-server` and `ipykernel` (see
# requirements.txt), and a Python env with numpy/scipy/matplotlib/pandas
# installed via `pip install -e .`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    echo "[generate_figures] using active env: ${VIRTUAL_ENV:-$CONDA_PREFIX}"
else
    echo "[generate_figures] WARNING: no virtualenv detected; falling back to system python" >&2
fi

command -v jupyter-book >/dev/null 2>&1 || {
    echo "[generate_figures] jupyter-book not found on PATH. Run: pip install -r requirements.txt" >&2
    exit 127
}

if [[ "${1:-}" == "--clean" ]]; then
    echo "[generate_figures] clearing _build/execute"
    if command -v jupyter-book >/dev/null 2>&1; then
        jupyter-book clean --execute
    fi
fi

# MyST builds the whole project as one site; there is no per-notebook export
# target, so a single-notebook re-run is not expressible here. The execution
# cache keeps repeated runs cheap instead.
echo "[generate_figures] building site with notebook execution"
jupyter-book build --html --execute

echo "[generate_figures] done. Open _build/html/index.html to review."
