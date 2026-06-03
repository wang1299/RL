#!/bin/bash

# Evaluate an HM3D IL policy inside the same parallel Habitat RL environment,
# without any RL update.

mkdir -p /root/RL/train_log
mkdir -p /root/RL/eval_png
mkdir -p /root/RL/eval_results

cd /root/RL || exit 1

export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:/root/RL:/root/GroundingDINO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Use physical GPU IDs 4,5,6,7 directly.
unset CUDA_VISIBLE_DEVICES
export RL_GPU_IDS="${RL_GPU_IDS:-7}"
export DINO_DEVICE="${DINO_DEVICE:-cuda:4}"
export DINO_DEVICES="${DINO_DEVICES:-cuda:4,cuda:5,cuda:6}"
export ENV_GPU_IDS="${ENV_GPU_IDS:-4,5,6}"
export WORKERS_PER_ENV_GPU="${WORKERS_PER_ENV_GPU:-4}"

DEFAULT_POLICY="/root/RL/components/data/model_weights/hm3d_viewpoint_imitation_features_transformer_poi4yaw_20260524_172033/hm3d_imitation_best_balanced.pth"
POLICY_CHECKPOINT_LOAD="${POLICY_CHECKPOINT_LOAD:-$DEFAULT_POLICY}"
ENCODER_CHECKPOINT_OVERRIDE="${ENCODER_CHECKPOINT_OVERRIDE:-}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="hm3d_il_policy_eval_${TIMESTAMP}"
LOG_FILE="/root/RL/train_log/${EXP_NAME}.log"
VIZ_DIR="/root/RL/eval_png/${EXP_NAME}"
CSV_FILE="/root/RL/eval_results/${EXP_NAME}.csv"

HABITAT_SCENES="00016-qk9eeNeR4vw,00017-oEPjPNSPmzL,00023-zepmXAdrpjR,00031-Wo6kuutE9i7,00033-oPj9qMxrDEa,00087-YY8rqV6L6rf,00099-226REUyJh2K,00105-xWvSkKiWQpC,00108-oStKKWkQ1id,00155-iLDo95ZbDJq,00166-RaYrxWt5pR1,00177-VSxVP19Cdyw,00210-j2EJhFEQGCL,00245-741Fdj7NLF9,00250-U3oQjwTuMX8,00251-wsAYBFtQaL7,00254-YMNvYDhK8mB,00255-NGyoyh91xXJ,00269-JNiWU5TZLtt,00299-bdp1XNEdvmW,00304-X6Pct1msZv5,00323-yHLr6bvWsVm,00324-DoSbsoo4EAg,00327-xgLmjqzoAzF,00378-DqJKU7YU7dA,00384-ceJTwFNjqCt,00401-H8rQCnvBgo6,00404-QN2dRqwd84J,00417-nGhNxKrgBPb,00434-L5QEsaVqwrY,00444-sX9xad6ULKc,00466-xAHnY3QzFUN,00506-QVAA6zecMHu,00567-KjZrPggnHm8,00569-YJDUB7hWg9h,00591-JptJPosx1Z6,00598-mt9H8KcxRKD,00612-GsQBY83r3hb,00624-ooq3SnvC79d,00638-iePHCSf119p,00662-aRKASs4e8j1,00669-DNWbUAJYsPy,00680-YmWinf3mhb5,00706-YHmAkqgwe2p,00712-HZ2iMMBsBQ9,00733-GtM3JtRvvvR,00741-w8GiikYuFRk,00745-yX5efd48dLf,00750-E1NrAhMoqvB,00758-HfMobPm86Xn"

EPISODES_PER_SCENE="${EPISODES_PER_SCENE:-1}"
NUM_STEPS="${NUM_STEPS:-3000}"
TRANSFORMER_CONTEXT_LEN="${TRANSFORMER_CONTEXT_LEN:-16}"

mkdir -p "$VIZ_DIR"

EXTRA_ARGS=()
if [ -n "$ENCODER_CHECKPOINT_OVERRIDE" ]; then
    EXTRA_ARGS+=(--encoder_checkpoint_override "$ENCODER_CHECKPOINT_OVERRIDE")
fi

echo "[INFO] Starting HM3D IL policy evaluation..."
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Visualization directory: $VIZ_DIR"
echo "[INFO] CSV file: $CSV_FILE"
echo "[INFO] POLICY_CHECKPOINT_LOAD: $POLICY_CHECKPOINT_LOAD"
if [ -n "$ENCODER_CHECKPOINT_OVERRIDE" ]; then
    echo "[INFO] ENCODER_CHECKPOINT_OVERRIDE: $ENCODER_CHECKPOINT_OVERRIDE"
fi
echo "[INFO] EPISODES_PER_SCENE: $EPISODES_PER_SCENE"
echo "[INFO] NUM_STEPS: $NUM_STEPS"
echo "[INFO] TRANSFORMER_CONTEXT_LEN: $TRANSFORMER_CONTEXT_LEN"
echo "[INFO] WORKERS_PER_ENV_GPU: $WORKERS_PER_ENV_GPU"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] Habitat workers and DINO use physical GPUs 4,5,6; policy/update uses physical GPU 7"

CMD=(/root/miniconda3/envs/habitat/bin/python /root/RL/train_habitat_parallel.py
    --conf_path /root/RL/config \
    --num_workers "$WORKERS_PER_ENV_GPU" \
    --episodes "$EPISODES_PER_SCENE" \
    --num_steps "$NUM_STEPS" \
    --gpu_ids "$RL_GPU_IDS" \
    --env_gpu_ids "$ENV_GPU_IDS" \
    --dino_device "$DINO_DEVICE" \
    --dino_devices "$DINO_DEVICES" \
    --use_dino \
    --dataset_root /root/hm3d/scene_datasets/hm3d \
    --habitat_scenes "$HABITAT_SCENES" \
    --save_frames_to "$VIZ_DIR" \
    --policy_checkpoint_load "$POLICY_CHECKPOINT_LOAD" \
    "${EXTRA_ARGS[@]}" \
    --use_transformer \
    --transformer_context_len "$TRANSFORMER_CONTEXT_LEN" \
    --eval_only \
    --eval_deterministic \
    --eval_output_csv "$CSV_FILE" \
    --no_save_on_exit)

if [[ "${RUN_FOREGROUND:-0}" == "1" || "${RUN_FOREGROUND:-0}" == "true" ]]; then
    echo "[INFO] Running evaluation in foreground"
    "${CMD[@]}" > "$LOG_FILE" 2>&1
    exit $?
fi

setsid "${CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "[INFO] Evaluation process started with PID: $PID"
echo "[INFO] Check logs with: tail -f $LOG_FILE"
echo "[INFO] Kill with: kill -TERM $PID"

echo "$PID" > "$VIZ_DIR/evaluation.pid"
echo "$LOG_FILE" > "$VIZ_DIR/evaluation.log.path"
echo "$CSV_FILE" > "$VIZ_DIR/evaluation.csv.path"

exit 0
