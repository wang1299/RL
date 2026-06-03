#!/usr/bin/env bash
set -Eeuo pipefail

TS="${1:-$(date +%Y%m%d_%H%M%S)}"
ROOT="/root/RL"

DATA_DIR="${HM3D_IL_DATA_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset}"
SAVE_DIR="${HM3D_IL_SAVE_DIR:-$ROOT/components/data/model_weights/hm3d_viewpoint_imitation_newlogic_$TS}"
LOG_FILE="${HM3D_IL_LOG_FILE:-$ROOT/train_log/hm3d_il_train_newlogic_$TS.log}"
PID_FILE="${HM3D_IL_PID_FILE:-$ROOT/train_log/hm3d_il_train_newlogic_$TS.pid}"

mkdir -p "$SAVE_DIR" "$ROOT/train_log"
printf "%s\n" "$$" > "$PID_FILE"

exec >> "$LOG_FILE" 2>&1

echo "[INFO] Starting HM3D IL new-logic training"
echo "[INFO] timestamp: $TS"
echo "[INFO] pid: $$"
echo "[INFO] data_dir: $DATA_DIR"
echo "[INFO] save_dir: $SAVE_DIR"
echo "[INFO] log_file: $LOG_FILE"
echo "[INFO] pid_file: $PID_FILE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="$ROOT:/root/GroundingDINO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] IL_VAL_SPLIT_MODE: ${IL_VAL_SPLIT_MODE:-file}"
echo "[INFO] IL_TRAIN_SAMPLING_MODE: ${IL_TRAIN_SAMPLING_MODE:-scene_sqrt}"
echo "[INFO] IL_LABEL_SMOOTHING: ${IL_LABEL_SMOOTHING:-0.05}"
echo "[INFO] IL_EARLY_STOPPING_PATIENCE: ${IL_EARLY_STOPPING_PATIENCE:-6}"
echo "[INFO] IL_TRAIN_EPOCH_SAMPLES: ${IL_TRAIN_EPOCH_SAMPLES:-0}"
echo "[INFO] IL_VAL_EPOCH_SAMPLES: ${IL_VAL_EPOCH_SAMPLES:-0}"
echo "[INFO] IL_NUM_WORKERS: ${IL_NUM_WORKERS:-0}"
echo "[INFO] IL_FREEZE_ENCODER: ${IL_FREEZE_ENCODER:-0}"

FREEZE_ENCODER_ARGS=()
if [[ "${IL_FREEZE_ENCODER:-0}" == "1" || "${IL_FREEZE_ENCODER:-0}" == "true" || "${IL_FREEZE_ENCODER:-0}" == "TRUE" ]]; then
  FREEZE_ENCODER_ARGS=(--freeze_encoder)
fi

exec /root/miniconda3/envs/habitat/bin/python "$ROOT/ImitationLearning/train_hm3d_il.py" \
  --conf_path "$ROOT/config" \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --epochs "${IL_EPOCHS:-30}" \
  --batch_size "${IL_BATCH_SIZE:-8}" \
  --seq_len "${IL_SEQ_LEN:-16}" \
  --lr "${IL_LR:-0.0001}" \
  --train_epoch_samples "${IL_TRAIN_EPOCH_SAMPLES:-0}" \
  --val_epoch_samples "${IL_VAL_EPOCH_SAMPLES:-0}" \
  --num_workers "${IL_NUM_WORKERS:-0}" \
  --val_split_mode "${IL_VAL_SPLIT_MODE:-file}" \
  --train_sampling_mode "${IL_TRAIN_SAMPLING_MODE:-scene_sqrt}" \
  --label_smoothing "${IL_LABEL_SMOOTHING:-0.05}" \
  --early_stopping_patience "${IL_EARLY_STOPPING_PATIENCE:-6}" \
  --gpu_id "${IL_GPU_ID:-0}" \
  "${FREEZE_ENCODER_ARGS[@]}"
