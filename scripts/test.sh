#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
# Hydra configuration tests need paths, but do not read data or checkpoints.
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/unused/NAVSIM}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/tmp/foresight-tests}"
exec "${PYTHON:-python}" -B -m unittest discover -s tests -v "$@"
