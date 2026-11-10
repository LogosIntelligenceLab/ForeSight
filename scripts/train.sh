#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
NNODES="${NNODES:-${PET_NNODES:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
foresight_run "${TORCHRUN:-torchrun}" \
    --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}" \
    "$NAVSIM_DEVKIT_ROOT/planning/script/run_training.py" \
    agent=foresight_agent experiment_name=foresight_train train_test_split=navtrain split=trainval \
    "trainer.params.max_epochs=$TRAIN_EPOCHS" "agent.config.scheduler_epochs=$TRAIN_EPOCHS" \
    "trainer.params.num_nodes=$NNODES" "dataloader.params.batch_size=${BATCH_SIZE:-8}" \
    "cache_path=$NAVSIM_EXP_ROOT/training_cache" \
    use_cache_without_dataset=True force_cache_computation=False "$@"
