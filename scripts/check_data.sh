#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
mode="${1:-train}"
case "$mode" in
    train) split=trainval ;;
    evaluate) split=test ;;
    *) printf 'Usage: bash scripts/check_data.sh [train|evaluate]\n' >&2; exit 2 ;;
esac
"${PYTHON:-python}" - "$split" "$mode" <<'PY'
import os
import sys
from pathlib import Path

split, mode = sys.argv[1:]
root = Path(os.environ['OPENSCENE_DATA_ROOT'])
required_dirs = [root / 'navsim_logs' / split, root / 'sensor_blobs' / split,
                 Path(os.environ['NUPLAN_MAPS_ROOT']) / os.environ['NUPLAN_MAP_VERSION']]
required_assets = ['FORESIGHT_EPONA_CKPT', 'FORESIGHT_VAE_CKPT',
                   'FORESIGHT_BACKBONE_CKPT', 'FORESIGHT_PLAN_ANCHOR']
if mode == 'evaluate':
    required_assets.append('FORESIGHT_EVAL_CKPT')
if mode == 'train' and os.environ.get('FORESIGHT_PLANNER_PRETRAIN'):
    required_assets.append('FORESIGHT_PLANNER_PRETRAIN')
missing = [str(p) for p in required_dirs if not p.is_dir() or not any(p.iterdir())]
missing += [key for key in required_assets
            if not os.environ.get(key) or not Path(os.environ[key]).is_file()]
if missing:
    raise SystemExit('Missing or empty data directories / asset settings:\n  ' + '\n  '.join(missing))
print('Required data directories and asset files exist; contents were not validated.')
PY
