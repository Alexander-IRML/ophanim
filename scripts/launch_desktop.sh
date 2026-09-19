#!/usr/bin/env bash
set -eu

project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"

export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
ophanim_python="${OPHANIM_PYTHON:-python3}"
if [ -z "${OPHANIM_PYTHON:-}" ] && [ -x "$project_dir/.venv-shawty/bin/python" ]; then
  ophanim_python="$project_dir/.venv-shawty/bin/python"
fi
exec "$ophanim_python" -m ophanim.desktop \
  --data-dir "$project_dir/var/desktop" \
  "$@"
