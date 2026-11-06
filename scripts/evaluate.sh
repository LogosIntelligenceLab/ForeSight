#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
: "${FORESIGHT_EVAL_CKPT:?Set FORESIGHT_EVAL_CKPT to a trained ForeSight checkpoint}"
foresight_run "${PYTHON:-python}" "$NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score.py" \
    train_test_split=navtest agent=foresight_agent worker=sequential \
    "agent.checkpoint_path=$FORESIGHT_EVAL_CKPT" experiment_name=foresight_eval "$@"
