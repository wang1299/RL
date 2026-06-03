#!/bin/bash

set -euo pipefail

ROOT="/root/RL"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

DATA_DIR="${HM3D_IL_DATA_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_$TIMESTAMP}"
LOG_FILE="${HM3D_IL_LOG_FILE:-$ROOT/train_log/hm3d_il_generate_poi2yaw_geometric_${TIMESTAMP}_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.log}"
PID_FILE="${HM3D_IL_PID_FILE:-$ROOT/train_log/hm3d_il_generate_poi2yaw_geometric_${TIMESTAMP}_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.pid}"

mkdir -p "$ROOT/train_log"
mkdir -p "$DATA_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:/root/GroundingDINO"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

setsid /root/miniconda3/envs/habitat/bin/python -u \
  "$ROOT/ImitationLearning/scripts/generate_hm3d_viewpoint_expert_dataset.py" \
  --conf_path "$ROOT/config" \
  --dataset_root /root/hm3d/scene_datasets/hm3d \
  --output_dir "$DATA_DIR" \
  --poi_dir "$ROOT/pois" \
  --start_yaw_degrees "${START_YAW_DEGREES:-0,180}" \
  --viewpoint_yaw_degrees "${VIEWPOINT_YAW_DEGREES:-0,60,120,180,240,300}" \
  --gpu_id 0 \
  --action_policy geometric \
  --turn_threshold_deg "${TURN_THRESHOLD_DEG:-15}" \
  --num_shards "$NUM_SHARDS" \
  --shard_index "$SHARD_INDEX" \
  --skip_existing \
  --profile_timing \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"

echo "PID=$PID"
echo "LOG=$LOG_FILE"
echo "DATA_DIR=$DATA_DIR"
echo "PID_FILE=$PID_FILE"
echo "SHARD_INDEX=$SHARD_INDEX"
echo "NUM_SHARDS=$NUM_SHARDS"
