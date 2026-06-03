#!/usr/bin/env bash
set -Eeuo pipefail

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
ROOT="/root/RL"

DATA_DIR="${HM3D_IL_DATA_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_20260523_162231_sharded_50}"
FEATURE_DIR="${HM3D_IL_FEATURE_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_20260523_162231_sharded_50_features}"
LOG_FILE="${HM3D_CACHE_LOG_FILE:-$ROOT/train_log/hm3d_rgb_feature_cache_$TS.log}"
PID_FILE="${HM3D_CACHE_PID_FILE:-$ROOT/train_log/hm3d_rgb_feature_cache_$TS.pid}"

mkdir -p "$FEATURE_DIR" "$ROOT/train_log"
printf "%s\n" "$$" > "$PID_FILE"

exec >> "$LOG_FILE" 2>&1

echo "[INFO] Starting HM3D RGB feature cache"
echo "[INFO] timestamp: $TS"
echo "[INFO] pid: $$"
echo "[INFO] data_dir: $DATA_DIR"
echo "[INFO] feature_dir: $FEATURE_DIR"
echo "[INFO] log_file: $LOG_FILE"
echo "[INFO] pid_file: $PID_FILE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="$ROOT:/root/GroundingDINO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] HM3D_CACHE_BATCH_SIZE: ${HM3D_CACHE_BATCH_SIZE:-256}"
echo "[INFO] HM3D_CACHE_DTYPE: ${HM3D_CACHE_DTYPE:-float16}"
echo "[INFO] HM3D_CACHE_NUM_SHARDS: ${HM3D_CACHE_NUM_SHARDS:-1}"
echo "[INFO] HM3D_CACHE_SHARD_INDEX: ${HM3D_CACHE_SHARD_INDEX:-0}"

COMPRESS_ARGS=()
if [[ "${HM3D_CACHE_COMPRESS:-0}" == "1" || "${HM3D_CACHE_COMPRESS:-0}" == "true" || "${HM3D_CACHE_COMPRESS:-0}" == "TRUE" ]]; then
  COMPRESS_ARGS=(--compress)
fi

OVERWRITE_ARGS=()
if [[ "${HM3D_CACHE_OVERWRITE:-0}" == "1" || "${HM3D_CACHE_OVERWRITE:-0}" == "true" || "${HM3D_CACHE_OVERWRITE:-0}" == "TRUE" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

exec /root/miniconda3/envs/habitat/bin/python "$ROOT/ImitationLearning/scripts/cache_hm3d_rgb_features.py" \
  --conf_path "$ROOT/config" \
  --data_dir "$DATA_DIR" \
  --output_dir "$FEATURE_DIR" \
  --batch_size "${HM3D_CACHE_BATCH_SIZE:-256}" \
  --dtype "${HM3D_CACHE_DTYPE:-float16}" \
  --num_shards "${HM3D_CACHE_NUM_SHARDS:-1}" \
  --shard_index "${HM3D_CACHE_SHARD_INDEX:-0}" \
  --limit "${HM3D_CACHE_LIMIT:-0}" \
  --gpu_id "${HM3D_CACHE_GPU_ID:-0}" \
  "${COMPRESS_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"
