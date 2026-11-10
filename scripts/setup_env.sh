#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
python_bin="${PYTHON:-python}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 12), "Use Python 3.10 or 3.11"'
"$python_bin" -m pip install -r requirements.txt
"$python_bin" -m pip install --no-deps 'nuplan-devkit @ git+https://github.com/motional/nuplan-devkit.git@nuplan-devkit-v1.2'
"$python_bin" -m pip install --no-deps -e .
"$python_bin" -c 'import torch, torchvision, timm, hydra, nuplan; print("Environment imports OK; CUDA available:", torch.cuda.is_available())'
