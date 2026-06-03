#!/usr/bin/env bash
set -Eeuo pipefail

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
ROOT="/root/RL"

DATA_DIR="${HM3D_IL_DATA_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_20260523_162231_sharded_50}"
FEATURE_DIR="${HM3D_IL_FEATURE_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi2yaw_geometric_20260523_162231_sharded_50_features}"
RUN_DIR="${HM3D_CACHE_RUN_DIR:-$ROOT/train_log/hm3d_rgb_feature_cache_multi_$TS}"
LOG_FILE="${HM3D_WATCHER_LOG_FILE:-$ROOT/train_log/hm3d_feature_cache_then_train_$TS.log}"
PID_FILE="${HM3D_WATCHER_PID_FILE:-$ROOT/train_log/hm3d_feature_cache_then_train_$TS.pid}"
SLEEP_SECONDS="${HM3D_WATCHER_SLEEP_SECONDS:-60}"

mkdir -p "$ROOT/train_log"
printf "%s\n" "$$" > "$PID_FILE"
exec >> "$LOG_FILE" 2>&1

echo "[INFO] Starting HM3D feature-cache watcher"
echo "[INFO] timestamp: $TS"
echo "[INFO] pid: $$"
echo "[INFO] data_dir: $DATA_DIR"
echo "[INFO] feature_dir: $FEATURE_DIR"
echo "[INFO] cache_run_dir: $RUN_DIR"
echo "[INFO] watcher_log_file: $LOG_FILE"
echo "[INFO] watcher_pid_file: $PID_FILE"
echo "[INFO] sleep_seconds: $SLEEP_SECONDS"

expected_count="$(find "$DATA_DIR" -name '*.npz' | wc -l)"
echo "[INFO] expected source npz count: $expected_count"

while true; do
  mapfile -t pid_files < <(find "$RUN_DIR" -maxdepth 1 -name 'cache_gpu*_shard*.pid' 2>/dev/null | sort)
  if [[ "${#pid_files[@]}" -eq 0 ]]; then
    echo "[ERROR] No cache pid files found in $RUN_DIR"
    exit 1
  fi

  alive=0
  for pid_file in "${pid_files[@]}"; do
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
      alive=$((alive + 1))
    fi
  done

  feature_count="$(find "$FEATURE_DIR" -name '*.npz' | wc -l)"
  echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') cache_alive=$alive feature_npz=$feature_count/$expected_count"

  if [[ "$alive" -eq 0 ]]; then
    break
  fi
  sleep "$SLEEP_SECONDS"
done

feature_count="$(find "$FEATURE_DIR" -name '*.npz' | wc -l)"
if [[ "$feature_count" -lt "$expected_count" ]]; then
  echo "[ERROR] Cache finished but feature count is incomplete: $feature_count/$expected_count"
  echo "[ERROR] Feature IL was not started. Check shard logs in $RUN_DIR"
  exit 1
fi

TRAIN_TS="${HM3D_IL_TRAIN_TS:-${TS}_feature_il}"
export HM3D_IL_FEATURE_DIR="$FEATURE_DIR"
export HM3D_IL_SAVE_DIR="${HM3D_IL_SAVE_DIR:-$ROOT/components/data/model_weights/hm3d_viewpoint_imitation_features_$TRAIN_TS}"
export HM3D_IL_LOG_FILE="${HM3D_IL_LOG_FILE:-$ROOT/train_log/hm3d_il_train_features_$TRAIN_TS.log}"
export HM3D_IL_PID_FILE="${HM3D_IL_PID_FILE:-$ROOT/train_log/hm3d_il_train_features_$TRAIN_TS.pid}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

echo "[INFO] Cache complete; starting feature IL training"
echo "[INFO] train_timestamp: $TRAIN_TS"
echo "[INFO] train_log_file: $HM3D_IL_LOG_FILE"
echo "[INFO] train_save_dir: $HM3D_IL_SAVE_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

exec /bin/bash "$ROOT/tools/start_hm3d_il_feature.sh" "$TRAIN_TS"
