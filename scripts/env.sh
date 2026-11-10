#!/usr/bin/env bash
set -euo pipefail
FORESIGHT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export FORESIGHT_ROOT
export PYTHONPATH="$FORESIGHT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NAVSIM_DEVKIT_ROOT="$FORESIGHT_ROOT/navsim"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$FORESIGHT_ROOT/exp}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT; see .env.example}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
cd -- "$FORESIGHT_ROOT"
foresight_run() {
    if [[ "${FORESIGHT_DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}
