#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
foresight_run "${PYTHON:-python}" "$NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py" \
    agent=foresight_agent agent.build_model=False experiment_name=foresight_cache train_test_split=navtrain "$@"
