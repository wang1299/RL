#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 TRAIN_PID SAVE_DIR TIMESTAMP TRAIN_LOG" >&2
    exit 2
fi

ROOT="/home/wgy/RL"
TRAIN_PID="$1"
SAVE_DIR="$2"
TIMESTAMP="$3"
TRAIN_LOG="$4"
WATCH_LOG="$ROOT/train_log/hm3d_causal_il_then_eval_$TIMESTAMP.log"

mkdir -p "$ROOT/train_log"

{
    echo "[INFO] Watcher started at $(date)"
    echo "[INFO] TRAIN_PID: $TRAIN_PID"
    echo "[INFO] SAVE_DIR: $SAVE_DIR"
    echo "[INFO] TRAIN_LOG: $TRAIN_LOG"

    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        echo "[INFO] $(date) training still running..."
        sleep "${WATCH_INTERVAL_SECONDS:-300}"
    done

    echo "[INFO] $(date) training process ended; selecting checkpoint"

    CHECKPOINT=""
    for candidate in \
        "$SAVE_DIR/hm3d_imitation_best_balanced.pth" \
        "$SAVE_DIR/hm3d_imitation_best_top1.pth" \
        "$SAVE_DIR/hm3d_imitation_final.pth" \
        "$SAVE_DIR/hm3d_imitation_latest.pth"
    do
        if [ -f "$candidate" ]; then
            CHECKPOINT="$candidate"
            break
        fi
    done

    if [ -z "$CHECKPOINT" ]; then
        echo "[ERROR] No checkpoint found in $SAVE_DIR"
        exit 1
    fi

    echo "[INFO] Starting pure IL policy eval with checkpoint: $CHECKPOINT"
    POLICY_CHECKPOINT_LOAD="$CHECKPOINT" /bin/bash "$ROOT/run_hm3d_il_policy_eval.sh"
    echo "[INFO] Eval launch command finished at $(date)"
} >> "$WATCH_LOG" 2>&1
