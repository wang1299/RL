#!/bin/bash

set -euo pipefail

ROOT="/root/RL"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

GPUS_CSV="${GPUS:-3,4,5,6,7}"
PER_GPU="${PER_GPU:-10}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS_CSV"

NUM_GPUS="${#GPU_LIST[@]}"
NUM_SHARDS="${NUM_SHARDS:-$((NUM_GPUS * PER_GPU))}"
DATA_DIR="${HM3D_IL_DATA_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_${TIMESTAMP}_sharded_${NUM_SHARDS}}"
RUN_DIR="$ROOT/train_log/hm3d_generate_multi_${TIMESTAMP}"

mkdir -p "$RUN_DIR" "$DATA_DIR"

SUMMARY_FILE="$RUN_DIR/summary.txt"
: > "$SUMMARY_FILE"

echo "TIMESTAMP=$TIMESTAMP" | tee -a "$SUMMARY_FILE"
echo "DATA_DIR=$DATA_DIR" | tee -a "$SUMMARY_FILE"
echo "GPUS=$GPUS_CSV" | tee -a "$SUMMARY_FILE"
echo "PER_GPU=$PER_GPU" | tee -a "$SUMMARY_FILE"
echo "NUM_SHARDS=$NUM_SHARDS" | tee -a "$SUMMARY_FILE"

shard_index=0
for gpu in "${GPU_LIST[@]}"; do
  for _ in $(seq 1 "$PER_GPU"); do
    if [ "$shard_index" -ge "$NUM_SHARDS" ]; then
      break 2
    fi

    log_file="$RUN_DIR/shard_${shard_index}_of_${NUM_SHARDS}_gpu_${gpu}.log"
    pid_file="$RUN_DIR/shard_${shard_index}_of_${NUM_SHARDS}_gpu_${gpu}.pid"

    CUDA_VISIBLE_DEVICES="$gpu" \
    NUM_SHARDS="$NUM_SHARDS" \
    SHARD_INDEX="$shard_index" \
    HM3D_IL_DATA_DIR="$DATA_DIR" \
    HM3D_IL_LOG_FILE="$log_file" \
    HM3D_IL_PID_FILE="$pid_file" \
      /bin/bash "$ROOT/tools/start_hm3d_geometric_generation.sh" >> "$SUMMARY_FILE"

    echo "GPU=$gpu SHARD_INDEX=$shard_index LOG=$log_file PID_FILE=$pid_file" | tee -a "$SUMMARY_FILE"
    shard_index=$((shard_index + 1))
    sleep "${START_STAGGER_SECONDS:-1}"
  done
done

echo "RUN_DIR=$RUN_DIR" | tee -a "$SUMMARY_FILE"
echo "SUMMARY_FILE=$SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
