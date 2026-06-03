#!/bin/bash

# MP3D-only RL policy evaluation.
# Default behavior mirrors the high-coverage MP3D eval run from 20260612:
# REINFORCE_Transformer config, 300x300 Habitat RGB, random start/yaw, 3000 steps.
# Use run_mp3d_rl_policy_eval_highres.sh for 1024x1024 DINO diagnostics.

set -euo pipefail

mkdir -p /root/RL/train_log /root/RL/eval_png /root/RL/eval_results

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="/root/RL:/root/GroundingDINO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-rl-eval}"
mkdir -p "$MPLCONFIGDIR"

# Use physical GPU IDs directly. Policy/update stays on 7; MP3D env + DINO stay on 4.
unset CUDA_VISIBLE_DEVICES
export RL_GPU_IDS="${RL_GPU_IDS:-7}"
export ENV_GPU_IDS="${ENV_GPU_IDS:-4}"
export DINO_DEVICE="${DINO_DEVICE:-cuda:4}"
export DINO_DEVICES="${DINO_DEVICES:-cuda:4}"

SCENE_ID="${SCENE_ID:-zsNo4HB9uLZ}"
DATASET_ROOT="${DATASET_ROOT:-/root/MatterPort3D/mp3d}"
SCENE_DATASET_CONFIG="${SCENE_DATASET_CONFIG:-/root/MatterPort3D/mp3d/mp3d.scene_dataset_config.json}"
POLICY_CHECKPOINT_LOAD="${POLICY_CHECKPOINT_LOAD:-/root/RL/RL_training/runs/model_weights/parallel_train_20260607_130026/REINFORCE_Agent_Transformer/20260607_152447_parallel_train_20260607_130026_BEST_update_0001_ep_00015_score_0p7071_cov_0p6879.pth}"
CONF_PATH="${CONF_PATH:-/root/RL/RL_training/sbatch/configs/REINFORCE_Transformer}"
NUM_STEPS="${NUM_STEPS:-3000}"
TRANSFORMER_CONTEXT_LEN="${TRANSFORMER_CONTEXT_LEN:-16}"
EVAL_NO_PROGRESS_WINDOW="${EVAL_NO_PROGRESS_WINDOW:-160}"
EVAL_NO_PROGRESS_TURN_STEPS="${EVAL_NO_PROGRESS_TURN_STEPS:-3}"
EVAL_NO_PROGRESS_FORWARD_STEPS="${EVAL_NO_PROGRESS_FORWARD_STEPS:-35}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="rl_policy_eval_mp3d_${SCENE_ID}_${TIMESTAMP}"
LOG_FILE="/root/RL/train_log/${RUN_TAG}.log"
VIZ_DIR="/root/RL/eval_png/${RUN_TAG}"
CSV_FILE="/root/RL/eval_results/${RUN_TAG}.csv"
mkdir -p "$VIZ_DIR"

echo "[INFO] Starting MP3D RL policy evaluation..."
echo "[INFO] Scene: $SCENE_ID"
echo "[INFO] Config: $CONF_PATH"
echo "[INFO] Resolution: Habitat default from config/runner"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Visualization directory: $VIZ_DIR"
echo "[INFO] Evaluation CSV: $CSV_FILE"
echo "[INFO] Policy checkpoint: $POLICY_CHECKPOINT_LOAD"
echo "[INFO] Policy GPU: $RL_GPU_IDS"
echo "[INFO] Env GPU: $ENV_GPU_IDS"
echo "[INFO] DINO devices: $DINO_DEVICES"
echo "[INFO] MP3D score validation: object-presence/category-agnostic"
echo "[INFO] Eval no-progress exploration: window=${EVAL_NO_PROGRESS_WINDOW}, turn_steps=${EVAL_NO_PROGRESS_TURN_STEPS}, forward_steps=${EVAL_NO_PROGRESS_FORWARD_STEPS}"

/root/miniconda3/envs/habitat/bin/python /root/RL/train_habitat_parallel.py \
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
    --eval_env_override mp3d_category_agnostic_validation=true \
    --eval_env_override eval_no_progress_explore_enabled=true \
    --eval_env_override eval_no_progress_window="$EVAL_NO_PROGRESS_WINDOW" \
    --eval_env_override eval_no_progress_turn_steps="$EVAL_NO_PROGRESS_TURN_STEPS" \
    --eval_env_override eval_no_progress_forward_steps="$EVAL_NO_PROGRESS_FORWARD_STEPS" \
    --eval_output_csv "$CSV_FILE" \
    2>&1 | tee "$LOG_FILE"
