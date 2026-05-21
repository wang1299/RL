#!/bin/bash

set -euo pipefail

mkdir -p /home/wgy/RL/train_log

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${PYTHONPATH:-}:/home/wgy/RL:/home/wgy/GroundingDINO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use one physical GPU for dataset generation and imitation training by default.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATA_DIR="${HM3D_IL_DATA_DIR:-/home/wgy/RL/components/data/hm3d_viewpoint_il_dataset}"
WEIGHT_DIR="${HM3D_IL_WEIGHT_DIR:-/home/wgy/RL/components/data/model_weights/hm3d_viewpoint_imitation}"
GEN_LOG="/home/wgy/RL/train_log/hm3d_il_generate_${TIMESTAMP}.log"
TRAIN_LOG="/home/wgy/RL/train_log/hm3d_il_train_${TIMESTAMP}.log"

mkdir -p "$DATA_DIR"
mkdir -p "$WEIGHT_DIR"

GENERATOR_SCRIPT="${HM3D_IL_GENERATOR_SCRIPT:-/home/wgy/RL/ImitationLearning/scripts/generate_hm3d_viewpoint_expert_dataset.py}"
EPISODES_PER_SCENE="${EPISODES_PER_SCENE:-8}"
MAX_STEPS="${MAX_STEPS:-900}"
IL_EPOCHS="${IL_EPOCHS:-30}"
IL_BATCH_SIZE="${IL_BATCH_SIZE:-8}"
IL_SEQ_LEN="${IL_SEQ_LEN:-16}"
CANDIDATE_VIEWPOINTS="${CANDIDATE_VIEWPOINTS:-180}"
MAX_COVER_VIEWPOINTS="${MAX_COVER_VIEWPOINTS:-28}"
MIN_SAVE_SCORE="${MIN_SAVE_SCORE:-0.45}"
MIN_SAVE_COVERAGE="${MIN_SAVE_COVERAGE:-0.03}"

echo "[INFO] Generating HM3D expert dataset"
echo "[INFO] DATA_DIR: $DATA_DIR"
echo "[INFO] EPISODES_PER_SCENE: $EPISODES_PER_SCENE"
echo "[INFO] MAX_STEPS: $MAX_STEPS"
echo "[INFO] GENERATOR_SCRIPT: $GENERATOR_SCRIPT"
echo "[INFO] CANDIDATE_VIEWPOINTS: $CANDIDATE_VIEWPOINTS"
echo "[INFO] MIN_SAVE_SCORE: $MIN_SAVE_SCORE"
echo "[INFO] MIN_SAVE_COVERAGE: $MIN_SAVE_COVERAGE"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Generation log: $GEN_LOG"

/root/miniconda3/envs/habitat/bin/python "$GENERATOR_SCRIPT" \
    --conf_path /home/wgy/RL/config \
    --dataset_root /home/wgy/hm3d/scene_datasets/hm3d \
    --output_dir "$DATA_DIR" \
    --episodes_per_scene "$EPISODES_PER_SCENE" \
    --max_steps "$MAX_STEPS" \
    --min_steps 80 \
    --candidate_viewpoints "$CANDIDATE_VIEWPOINTS" \
    --max_cover_viewpoints "$MAX_COVER_VIEWPOINTS" \
    --min_save_score "$MIN_SAVE_SCORE" \
    --min_save_coverage "$MIN_SAVE_COVERAGE" \
    --gpu_id 0 \
    > "$GEN_LOG" 2>&1

echo "[INFO] Training HM3D imitation policy"
echo "[INFO] WEIGHT_DIR: $WEIGHT_DIR"
echo "[INFO] IL_EPOCHS: $IL_EPOCHS"
echo "[INFO] IL_BATCH_SIZE: $IL_BATCH_SIZE"
echo "[INFO] IL_SEQ_LEN: $IL_SEQ_LEN"
echo "[INFO] Training log: $TRAIN_LOG"

/root/miniconda3/envs/habitat/bin/python /home/wgy/RL/ImitationLearning/train_hm3d_il.py \
    --conf_path /home/wgy/RL/config \
    --data_dir "$DATA_DIR" \
    --save_dir "$WEIGHT_DIR" \
    --epochs "$IL_EPOCHS" \
    --batch_size "$IL_BATCH_SIZE" \
    --seq_len "$IL_SEQ_LEN" \
    --gpu_id 0 \
    > "$TRAIN_LOG" 2>&1

echo "[INFO] Done."
echo "[INFO] Best checkpoint: $WEIGHT_DIR/hm3d_imitation_best.pth"
echo "[INFO] Start RL with:"
echo "POLICY_CHECKPOINT_LOAD=$WEIGHT_DIR/hm3d_imitation_best.pth /home/wgy/RL/run_train_parallel.sh"
