#!/usr/bin/env bash
set -eu

project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"

export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m ophanim.desktop \
  --data-dir "$project_dir/var/desktop" \
  "$@"
