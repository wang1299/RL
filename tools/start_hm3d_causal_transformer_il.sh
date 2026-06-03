#!/bin/bash
set -euo pipefail

ROOT="/root/RL"
TIMESTAMP="${1:-$(date +"%Y%m%d_%H%M%S")}"

FEATURE_DIR="${HM3D_IL_FEATURE_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi4yaw_geometric_20260524_172033_features}"
SAVE_DIR="${HM3D_IL_SAVE_DIR:-$ROOT/components/data/model_weights/hm3d_viewpoint_imitation_features_transformer_poi4yaw_causal_$TIMESTAMP}"
LOG_FILE="${HM3D_IL_LOG_FILE:-$ROOT/train_log/hm3d_il_train_features_transformer_poi4yaw_causal_$TIMESTAMP.log}"
PID_FILE="${HM3D_IL_PID_FILE:-$ROOT/train_log/hm3d_il_train_features_transformer_poi4yaw_causal_$TIMESTAMP.pid}"
GPU="${IL_TRAIN_GPU:-3}"

mkdir -p "$ROOT/train_log" "$SAVE_DIR"

cd "$ROOT"

echo "[INFO] Starting causal Transformer HM3D feature IL training"
echo "[INFO] FEATURE_DIR: $FEATURE_DIR"
echo "[INFO] SAVE_DIR: $SAVE_DIR"
echo "[INFO] LOG_FILE: $LOG_FILE"
echo "[INFO] CUDA_VISIBLE_DEVICES: $GPU"
echo "[INFO] epochs=${IL_EPOCHS:-30} batch_size=${IL_BATCH_SIZE:-256} seq_len=${IL_SEQ_LEN:-16} lr=${IL_LR:-0.0001}"
echo "[INFO] train_epoch_samples=${IL_TRAIN_EPOCH_SAMPLES:-120000} val_epoch_samples=${IL_VAL_EPOCH_SAMPLES:-20000}"
echo "[INFO] train_sampling_mode=${IL_TRAIN_SAMPLING_MODE:-scene_sqrt} label_smoothing=${IL_LABEL_SMOOTHING:-0.05}"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT:${PYTHONPATH:-}" \
PYTHONUNBUFFERED=1 \
setsid /root/miniconda3/envs/habitat/bin/python -u "$ROOT/ImitationLearning/train_hm3d_il_features.py" \
    --conf_path "$ROOT/config" \
    --data_dir "$FEATURE_DIR" \
    --save_dir "$SAVE_DIR" \
    --epochs "${IL_EPOCHS:-30}" \
    --batch_size "${IL_BATCH_SIZE:-256}" \
    --seq_len "${IL_SEQ_LEN:-16}" \
    --lr "${IL_LR:-0.0001}" \
    --train_epoch_samples "${IL_TRAIN_EPOCH_SAMPLES:-120000}" \
    --val_epoch_samples "${IL_VAL_EPOCH_SAMPLES:-20000}" \
    --num_workers "${IL_NUM_WORKERS:-4}" \
    --val_split_mode "${IL_VAL_SPLIT_MODE:-file}" \
    --train_sampling_mode "${IL_TRAIN_SAMPLING_MODE:-scene_sqrt}" \
    --label_smoothing "${IL_LABEL_SMOOTHING:-0.05}" \
    --early_stopping_patience "${IL_EARLY_STOPPING_PATIENCE:-4}" \
    --gpu_id 0 \
    --freeze_encoder \
    --use_transformer \
    > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
echo "$SAVE_DIR" > "$ROOT/train_log/hm3d_il_train_features_transformer_poi4yaw_causal_$TIMESTAMP.save_dir"
echo "$LOG_FILE" > "$ROOT/train_log/hm3d_il_train_features_transformer_poi4yaw_causal_$TIMESTAMP.log.path"

echo "[INFO] Training PID: $PID"
echo "[INFO] PID file: $PID_FILE"
echo "[INFO] Tail log: tail -f $LOG_FILE"
