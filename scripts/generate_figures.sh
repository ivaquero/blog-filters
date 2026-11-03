#!/usr/bin/env bash
# Regenerate every figure that the tutorial notebooks produce.
#
# blog-filters keeps its notebooks as the source of truth: `jupyter-book build`
# executes each notebook top-to-bottom and re-renders the inline outputs. This
# script is a thin wrapper that (a) rebuilds only when invoked, (b) writes the
# rendered HTML to _build/html so authors can eyeball the result, and (c) can
# optionally persist PNGs to images/ when the notebook bodies are switched from
# plt.show() to plt.savefig() (see TODO below).
#
# Usage:
#   ./scripts/generate_figures.sh              # rebuild all notebooks
#   ./scripts/generate_figures.sh filters-kf-basic.ipynb   # single notebook
#
# Requires: `jupyter-book` (see requirements.txt) and a working Python env with
# numpy/scipy/matplotlib/pandas installed via `pip install -e .`.

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

if [[ $# -gt 0 ]]; then
    for nb in "$@"; do
        echo "[generate_figures] executing $nb"
        # TODO: once notebooks save figures to images/*.png via plt.savefig,
        # add `--ExecutePreprocessor.savefigs_to_build=True` here.
        jupyter-book execute --to html --output _build/execute "$nb"
    done
else
    echo "[generate_figures] full rebuild via jupyter-book build ."
    jupyter-book build --all .
fi

echo "[generate_figures] done. Open _build/html/index.html to review."
