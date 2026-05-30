#!/usr/bin/env bash
set -Eeuo pipefail

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
ROOT="/home/wgy/RL"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-3,4,5,6,7}"
PER_GPU="${PER_GPU:-1}"
NUM_SHARDS="${HM3D_CACHE_NUM_SHARDS:-$(( ${#GPU_LIST[@]} * PER_GPU ))}"

DATA_DIR="${HM3D_IL_DATA_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_20260523_162231_sharded_50}"
FEATURE_DIR="${HM3D_IL_FEATURE_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_20260523_162231_sharded_50_features}"
RUN_DIR="${HM3D_CACHE_RUN_DIR:-$ROOT/train_log/hm3d_rgb_feature_cache_multi_$TS}"
mkdir -p "$RUN_DIR" "$FEATURE_DIR" "$ROOT/train_log"

SUMMARY="$RUN_DIR/summary.txt"
{
  echo "[INFO] Starting multi-GPU HM3D RGB feature cache"
  echo "[INFO] timestamp: $TS"
  echo "[INFO] data_dir: $DATA_DIR"
  echo "[INFO] feature_dir: $FEATURE_DIR"
  echo "[INFO] run_dir: $RUN_DIR"
  echo "[INFO] gpus: ${GPU_LIST[*]}"
  echo "[INFO] per_gpu: $PER_GPU"
  echo "[INFO] num_shards: $NUM_SHARDS"
  echo "[INFO] batch_size: ${HM3D_CACHE_BATCH_SIZE:-256}"
  echo "[INFO] dtype: ${HM3D_CACHE_DTYPE:-float16}"
} | tee "$SUMMARY"

shard=0
for gpu in "${GPU_LIST[@]}"; do
  for local_idx in $(seq 1 "$PER_GPU"); do
    if [[ "$shard" -ge "$NUM_SHARDS" ]]; then
      break
    fi
    shard_padded="$(printf "%02d" "$shard")"
    log_file="$RUN_DIR/cache_gpu${gpu}_shard${shard_padded}.log"
    pid_file="$RUN_DIR/cache_gpu${gpu}_shard${shard_padded}.pid"
    echo "[INFO] Launching shard=$shard gpu=$gpu log=$log_file" | tee -a "$SUMMARY"
    CUDA_VISIBLE_DEVICES="$gpu" \
      HM3D_IL_DATA_DIR="$DATA_DIR" \
      HM3D_IL_FEATURE_DIR="$FEATURE_DIR" \
      HM3D_CACHE_LOG_FILE="$log_file" \
      HM3D_CACHE_PID_FILE="$pid_file" \
      HM3D_CACHE_NUM_SHARDS="$NUM_SHARDS" \
      HM3D_CACHE_SHARD_INDEX="$shard" \
      HM3D_CACHE_GPU_ID=0 \
      setsid /bin/bash "$ROOT/tools/cache_hm3d_rgb_features.sh" "${TS}_shard${shard_padded}" >/dev/null 2>&1 &
    echo "$!" > "$pid_file.launcher"
    shard=$((shard + 1))
  done
done

echo "[INFO] Launched $shard cache shards" | tee -a "$SUMMARY"
echo "[INFO] Watch logs: tail -f $RUN_DIR/cache_gpu*_shard*.log" | tee -a "$SUMMARY"
