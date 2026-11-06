#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
foresight_run "${PYTHON:-python}" "$NAVSIM_DEVKIT_ROOT/planning/script/run_metric_caching.py" \
    train_test_split=navtest "cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache" "$@"
