#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/RL"
TS="${1:-$(date +%Y%m%d_%H%M%S)}"

RUN_DIR="${HM3D_PIPELINE_RUN_DIR:-$ROOT/train_log/hm3d_poi4yaw_transformer_pipeline_$TS}"
RAW_DIR="${HM3D_IL_DATA_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi4yaw_geometric_$TS}"
FEATURE_DIR="${HM3D_IL_FEATURE_DIR:-$ROOT/components/data/hm3d_viewpoint_il_dataset_poi4yaw_geometric_${TS}_features}"
WEIGHT_DIR="${HM3D_IL_SAVE_DIR:-$ROOT/components/data/model_weights/hm3d_viewpoint_imitation_features_transformer_poi4yaw_$TS}"
PIPELINE_LOG="$RUN_DIR/pipeline.log"
PID_FILE="$RUN_DIR/pipeline.pid"

mkdir -p "$RUN_DIR" "$RAW_DIR" "$FEATURE_DIR" "$WEIGHT_DIR" "$ROOT/train_log"
printf "%s\n" "$$" > "$PID_FILE"
exec >> "$PIPELINE_LOG" 2>&1

export PYTHONPATH="$ROOT:/root/GroundingDINO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

SCENES=(
  00016-qk9eeNeR4vw
  00017-oEPjPNSPmzL
  00023-zepmXAdrpjR
  00031-Wo6kuutE9i7
  00033-oPj9qMxrDEa
  00087-YY8rqV6L6rf
  00099-226REUyJh2K
  00105-xWvSkKiWQpC
  00108-oStKKWkQ1id
  00155-iLDo95ZbDJq
  00166-RaYrxWt5pR1
  00177-VSxVP19Cdyw
  00210-j2EJhFEQGCL
  00245-741Fdj7NLF9
  00250-U3oQjwTuMX8
  00251-wsAYBFtQaL7
  00254-YMNvYDhK8mB
  00255-NGyoyh91xXJ
  00269-JNiWU5TZLtt
  00299-bdp1XNEdvmW
  00304-X6Pct1msZv5
  00323-yHLr6bvWsVm
  00324-DoSbsoo4EAg
  00327-xgLmjqzoAzF
  00378-DqJKU7YU7dA
  00384-ceJTwFNjqCt
  00401-H8rQCnvBgo6
  00404-QN2dRqwd84J
  00417-nGhNxKrgBPb
  00434-L5QEsaVqwrY
  00444-sX9xad6ULKc
  00466-xAHnY3QzFUN
  00506-QVAA6zecMHu
  00567-KjZrPggnHm8
  00569-YJDUB7hWg9h
  00591-JptJPosx1Z6
  00598-mt9H8KcxRKD
  00612-GsQBY83r3hb
  00624-ooq3SnvC79d
  00638-iePHCSf119p
  00662-aRKASs4e8j1
  00669-DNWbUAJYsPy
  00680-YmWinf3mhb5
  00706-YHmAkqgwe2p
  00712-HZ2iMMBsBQ9
  00733-GtM3JtRvvvR
  00741-w8GiikYuFRk
  00745-yX5efd48dLf
  00750-E1NrAhMoqvB
  00758-HfMobPm86Xn
)

GPUS=(0 1 2 3)
GROUP_COUNTS=(13 13 13 11)
START_YAWS="${START_YAW_DEGREES:-0,90,180,270}"
VIEWPOINT_YAWS="${VIEWPOINT_YAW_DEGREES:-0,60,120,180,240,300}"

join_csv() {
  local IFS=,
  echo "$*"
}

log() {
  echo "[$(date '+%F %T')] $*"
}

write_metadata() {
  ROOT="$ROOT" TS="$TS" RAW_DIR="$RAW_DIR" FEATURE_DIR="$FEATURE_DIR" WEIGHT_DIR="$WEIGHT_DIR" RUN_DIR="$RUN_DIR" START_YAWS="$START_YAWS" VIEWPOINT_YAWS="$VIEWPOINT_YAWS" \
  /root/miniconda3/envs/habitat/bin/python - <<'PY'
import json
import os
from pathlib import Path

scenes = [
    "00016-qk9eeNeR4vw", "00017-oEPjPNSPmzL", "00023-zepmXAdrpjR", "00031-Wo6kuutE9i7",
    "00033-oPj9qMxrDEa", "00087-YY8rqV6L6rf", "00099-226REUyJh2K", "00105-xWvSkKiWQpC",
    "00108-oStKKWkQ1id", "00155-iLDo95ZbDJq", "00166-RaYrxWt5pR1", "00177-VSxVP19Cdyw",
    "00210-j2EJhFEQGCL", "00245-741Fdj7NLF9", "00250-U3oQjwTuMX8", "00251-wsAYBFtQaL7",
    "00254-YMNvYDhK8mB", "00255-NGyoyh91xXJ", "00269-JNiWU5TZLtt", "00299-bdp1XNEdvmW",
    "00304-X6Pct1msZv5", "00323-yHLr6bvWsVm", "00324-DoSbsoo4EAg", "00327-xgLmjqzoAzF",
    "00378-DqJKU7YU7dA", "00384-ceJTwFNjqCt", "00401-H8rQCnvBgo6", "00404-QN2dRqwd84J",
    "00417-nGhNxKrgBPb", "00434-L5QEsaVqwrY", "00444-sX9xad6ULKc", "00466-xAHnY3QzFUN",
    "00506-QVAA6zecMHu", "00567-KjZrPggnHm8", "00569-YJDUB7hWg9h", "00591-JptJPosx1Z6",
    "00598-mt9H8KcxRKD", "00612-GsQBY83r3hb", "00624-ooq3SnvC79d", "00638-iePHCSf119p",
    "00662-aRKASs4e8j1", "00669-DNWbUAJYsPy", "00680-YmWinf3mhb5", "00706-YHmAkqgwe2p",
    "00712-HZ2iMMBsBQ9", "00733-GtM3JtRvvvR", "00741-w8GiikYuFRk", "00745-yX5efd48dLf",
    "00750-E1NrAhMoqvB", "00758-HfMobPm86Xn",
]
payload = {
    "created_at": os.popen("date '+%F %T'").read().strip(),
    "timestamp": os.environ["TS"],
    "pipeline": "hm3d_poi4yaw_geometric_features_transformer",
    "scene_count": len(scenes),
    "scenes": scenes,
    "gpu_scene_split": {"0": 13, "1": 13, "2": 13, "3": 11},
    "start_policy": "all_pois_fixed_yaw",
    "start_yaw_degrees": [float(x) for x in os.environ["START_YAWS"].split(",") if x],
    "viewpoint_yaw_degrees": [float(x) for x in os.environ["VIEWPOINT_YAWS"].split(",") if x],
    "action_policy": "geometric",
    "raw_dir": os.environ["RAW_DIR"],
    "feature_dir": os.environ["FEATURE_DIR"],
    "weight_dir": os.environ["WEIGHT_DIR"],
    "run_dir": os.environ["RUN_DIR"],
}
for path in (Path(os.environ["RAW_DIR"]) / "metadata.json", Path(os.environ["FEATURE_DIR"]) / "metadata.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

kill_children() {
  local pids
  pids="$(jobs -pr || true)"
  if [[ -n "$pids" ]]; then
    log "Stopping child processes: $pids"
    kill -TERM $pids 2>/dev/null || true
  fi
}
trap kill_children INT TERM

wait_stage() {
  local stage="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      log "[ERROR] $stage child failed: pid=$pid"
      failed=1
    fi
  done
  return "$failed"
}

log "Starting HM3D POI 4-yaw Transformer IL pipeline"
log "PID=$$"
log "RUN_DIR=$RUN_DIR"
log "RAW_DIR=$RAW_DIR"
log "FEATURE_DIR=$FEATURE_DIR"
log "WEIGHT_DIR=$WEIGHT_DIR"
log "START_YAW_DEGREES=$START_YAWS"
log "VIEWPOINT_YAW_DEGREES=$VIEWPOINT_YAWS"
write_metadata

log "Stage 1/3: generating expert trajectories on GPUs 0,1,2,3"
log "Generation parallelism: one process per scene; GPUs run 13/13/13/11 scene processes concurrently"
gen_pids=()
offset=0
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  count="${GROUP_COUNTS[$idx]}"
  group=("${SCENES[@]:$offset:$count}")
  offset=$((offset + count))
  log "GPU $gpu assigned $count scenes: $(join_csv "${group[@]}")"
  for scene in "${group[@]}"; do
    scene_dir="$RAW_DIR/raw_shards/gpu_${gpu}/${scene}"
    mkdir -p "$scene_dir"
    log_file="$RUN_DIR/generate_gpu${gpu}_${scene}.log"
    log "Launch generation gpu=$gpu scene=$scene log=$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" /root/miniconda3/envs/habitat/bin/python -u \
      "$ROOT/ImitationLearning/scripts/generate_hm3d_viewpoint_expert_dataset.py" \
      --conf_path "$ROOT/config" \
      --dataset_root /root/hm3d/scene_datasets/hm3d \
      --habitat_scene "$scene" \
      --output_dir "$scene_dir" \
      --poi_dir "$ROOT/pois" \
      --start_yaw_degrees "$START_YAWS" \
      --viewpoint_yaw_degrees "$VIEWPOINT_YAWS" \
      --gpu_id 0 \
      --action_policy geometric \
      --turn_threshold_deg "${TURN_THRESHOLD_DEG:-15}" \
      --max_steps "${GEN_MAX_STEPS:-900}" \
      --min_steps "${GEN_MIN_STEPS:-80}" \
      --candidate_viewpoints "${CANDIDATE_VIEWPOINTS:-180}" \
      --max_cover_viewpoints "${MAX_COVER_VIEWPOINTS:-28}" \
      --min_save_score "${MIN_SAVE_SCORE:-0.45}" \
      --min_save_coverage "${MIN_SAVE_COVERAGE:-0.02}" \
      --skip_existing \
      --profile_timing \
      > "$log_file" 2>&1 &
    gen_pids+=("$!")
    echo "${gen_pids[-1]}" > "$RUN_DIR/generate_gpu${gpu}_${scene}.pid"
    sleep "${START_STAGGER_SECONDS:-0.2}"
  done
done

if ! wait_stage "generation" "${gen_pids[@]}"; then
  log "[ERROR] Generation failed; feature cache and IL training will not start"
  exit 1
fi

raw_count="$(find "$RAW_DIR" -name '*.npz' | wc -l)"
log "Generation complete: raw_npz=$raw_count"
if [[ "$raw_count" -le 0 ]]; then
  log "[ERROR] No expert trajectories were generated"
  exit 1
fi

log "Stage 2/3: caching RGB features on GPUs 0,1,2,3"
cache_per_gpu="${HM3D_CACHE_PER_GPU:-5}"
cache_num_shards="${HM3D_CACHE_NUM_SHARDS:-$(( ${#GPUS[@]} * cache_per_gpu ))}"
log "Feature cache parallelism: $cache_num_shards shards total, $cache_per_gpu shards per GPU"
cache_pids=()
for shard_idx in $(seq 0 $((cache_num_shards - 1))); do
  gpu="${GPUS[$((shard_idx % ${#GPUS[@]}))]}"
  log_file="$RUN_DIR/cache_gpu${gpu}_shard${shard_idx}_of_${cache_num_shards}.log"
  log "Launch feature cache gpu=$gpu shard=$shard_idx/$cache_num_shards log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" /root/miniconda3/envs/habitat/bin/python -u \
    "$ROOT/ImitationLearning/scripts/cache_hm3d_rgb_features.py" \
    --conf_path "$ROOT/config" \
    --data_dir "$RAW_DIR" \
    --output_dir "$FEATURE_DIR" \
    --batch_size "${HM3D_CACHE_BATCH_SIZE:-256}" \
    --dtype "${HM3D_CACHE_DTYPE:-float16}" \
    --num_shards "$cache_num_shards" \
    --shard_index "$shard_idx" \
    --gpu_id 0 \
    > "$log_file" 2>&1 &
  cache_pids+=("$!")
  echo "${cache_pids[-1]}" > "$RUN_DIR/cache_gpu${gpu}_shard${shard_idx}.pid"
  sleep "${CACHE_START_STAGGER_SECONDS:-0.5}"
done

if ! wait_stage "feature_cache" "${cache_pids[@]}"; then
  log "[ERROR] Feature cache failed; IL training will not start"
  exit 1
fi

feature_count="$(find "$FEATURE_DIR" -name '*.npz' | wc -l)"
log "Feature cache complete: feature_npz=$feature_count"
if [[ "$feature_count" -le 0 ]]; then
  log "[ERROR] No feature files were generated"
  exit 1
fi

log "Stage 3/3: training Transformer IL"
train_log="$RUN_DIR/train_transformer_il.log"
CUDA_VISIBLE_DEVICES="${IL_TRAIN_GPU:-3}" /root/miniconda3/envs/habitat/bin/python -u \
  "$ROOT/ImitationLearning/train_hm3d_il_features.py" \
  --conf_path "$ROOT/config" \
  --data_dir "$FEATURE_DIR" \
  --save_dir "$WEIGHT_DIR" \
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
  > "$train_log" 2>&1

log "Transformer IL training complete"
log "Best-balanced checkpoint: $WEIGHT_DIR/hm3d_imitation_best_balanced.pth"
log "Final checkpoint: $WEIGHT_DIR/hm3d_imitation_final.pth"
log "Pipeline complete"
