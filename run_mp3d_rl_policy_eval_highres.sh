#!/bin/bash

# MP3D high-resolution diagnostic evaluation.
# This is intentionally separate from run_mp3d_rl_policy_eval.sh so HM3D and the
# default MP3D path keep their original behavior. Use this script when we want
# DINO/GT validation to see a sharper MP3D RGB frame while the policy still
# receives the original 300x300 observation.

set -euo pipefail

export CONF_PATH="${CONF_PATH:-/root/RL/RL_training/sbatch/configs/REINFORCE_Transformer}"
export EVAL_WIDTH="${EVAL_WIDTH:-768}"
export EVAL_HEIGHT="${EVAL_HEIGHT:-768}"
export DINO_MAX_BOX_AREA_RATIO="${DINO_MAX_BOX_AREA_RATIO:-0.70}"
export DINO_MAX_BOX_ASPECT_RATIO="${DINO_MAX_BOX_ASPECT_RATIO:-12.0}"
export MP3D_SPATIAL_FALLBACK_MIN_OVERLAP="${MP3D_SPATIAL_FALLBACK_MIN_OVERLAP:-0.25}"
export MP3D_SAME_LABEL_CENTER_MAX_NORM_DIST="${MP3D_SAME_LABEL_CENTER_MAX_NORM_DIST:-0.35}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p /root/RL/train_log /root/RL/eval_png /root/RL/eval_results

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="/root/RL:/root/GroundingDINO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-rl-eval}"
mkdir -p "$MPLCONFIGDIR"

unset CUDA_VISIBLE_DEVICES
export RL_GPU_IDS="${RL_GPU_IDS:-7}"
export ENV_GPU_IDS="${ENV_GPU_IDS:-4}"
export DINO_DEVICE="${DINO_DEVICE:-cuda:4}"
export DINO_DEVICES="${DINO_DEVICES:-cuda:4}"

SCENE_ID="${SCENE_ID:-zsNo4HB9uLZ}"
DATASET_ROOT="${DATASET_ROOT:-/root/MatterPort3D/mp3d}"
SCENE_DATASET_CONFIG="${SCENE_DATASET_CONFIG:-/root/MatterPort3D/mp3d/mp3d.scene_dataset_config.json}"
POLICY_CHECKPOINT_LOAD="${POLICY_CHECKPOINT_LOAD:-/root/RL/RL_training/runs/model_weights/parallel_train_20260607_130026/REINFORCE_Agent_Transformer/20260607_152447_parallel_train_20260607_130026_BEST_update_0001_ep_00015_score_0p7071_cov_0p6879.pth}"
NUM_STEPS="${NUM_STEPS:-3000}"
TRANSFORMER_CONTEXT_LEN="${TRANSFORMER_CONTEXT_LEN:-16}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="rl_policy_eval_mp3d_${SCENE_ID}_highres_${TIMESTAMP}"
LOG_FILE="/root/RL/train_log/${RUN_TAG}.log"
VIZ_DIR="/root/RL/eval_png/${RUN_TAG}"
CSV_FILE="/root/RL/eval_results/${RUN_TAG}.csv"
mkdir -p "$VIZ_DIR"

echo "[INFO] Starting MP3D high-res RL policy evaluation..."
echo "[INFO] Scene: $SCENE_ID"
echo "[INFO] Config: $CONF_PATH"
echo "[INFO] Policy resolution: Habitat default from config/runner"
echo "[INFO] DINO validation resolution: ${EVAL_WIDTH}x${EVAL_HEIGHT}"
echo "[INFO] MP3D max box area ratio: $DINO_MAX_BOX_AREA_RATIO"
echo "[INFO] MP3D max box aspect ratio: $DINO_MAX_BOX_ASPECT_RATIO"
echo "[INFO] MP3D spatial fallback overlap: $MP3D_SPATIAL_FALLBACK_MIN_OVERLAP"
echo "[INFO] MP3D same-label center max distance: $MP3D_SAME_LABEL_CENTER_MAX_NORM_DIST"
echo "[INFO] MP3D score validation: object-presence/category-agnostic"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Visualization directory: $VIZ_DIR"
echo "[INFO] Evaluation CSV: $CSV_FILE"
echo "[INFO] Policy checkpoint: $POLICY_CHECKPOINT_LOAD"
echo "[INFO] Policy GPU: $RL_GPU_IDS"
echo "[INFO] Env GPU: $ENV_GPU_IDS"
echo "[INFO] DINO devices: $DINO_DEVICES"

RUN_CMD=(
    /root/miniconda3/envs/habitat/bin/python "$SCRIPT_DIR/train_habitat_parallel.py"
    --conf_path "$CONF_PATH" \
    --num_workers 1 \
    --episodes 1 \
    --num_steps "$NUM_STEPS" \
    --gpu_ids "$RL_GPU_IDS" \
    --env_gpu_ids "$ENV_GPU_IDS" \
    --dino_device "$DINO_DEVICE" \
    --dino_devices "$DINO_DEVICES" \
    --use_dino \
    --dataset_root "$DATASET_ROOT" \
    --scene_dataset_config_file "$SCENE_DATASET_CONFIG" \
    --habitat_scene "$SCENE_ID" \
    --save_frames_to "$VIZ_DIR" \
    --save_model_to /root/RL/RL_training/runs/model_weights \
    --policy_checkpoint_load "$POLICY_CHECKPOINT_LOAD" \
    --use_transformer \
    --transformer_context_len "$TRANSFORMER_CONTEXT_LEN" \
    --eval_only \
    --eval_deterministic \
    --eval_no_stop_on_success \
    --eval_env_override dino_width="$EVAL_WIDTH" \
    --eval_env_override dino_height="$EVAL_HEIGHT" \
    --eval_env_override dino_max_box_area_ratio="$DINO_MAX_BOX_AREA_RATIO" \
    --eval_env_override dino_max_box_aspect_ratio="$DINO_MAX_BOX_ASPECT_RATIO" \
    --eval_env_override mp3d_spatial_fallback=true \
    --eval_env_override mp3d_spatial_fallback_min_overlap="$MP3D_SPATIAL_FALLBACK_MIN_OVERLAP" \
    --eval_env_override mp3d_spatial_fallback_center=true \
    --eval_env_override mp3d_same_label_center_fallback=true \
    --eval_env_override mp3d_same_label_center_max_norm_dist="$MP3D_SAME_LABEL_CENTER_MAX_NORM_DIST" \
    --eval_env_override mp3d_category_agnostic_validation=true \
    --eval_output_csv "$CSV_FILE"
)

if [[ "${NO_TEE:-0}" == "1" ]]; then
    "${RUN_CMD[@]}"
else
    "${RUN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
