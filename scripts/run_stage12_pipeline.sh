#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_DOCKER="${USE_DOCKER:-auto}"
IN_SOULX_DOCKER="${IN_SOULX_DOCKER:-0}"
RUN_STAGE="${RUN_STAGE:-all}"
PROFILE="${PROFILE:-configs/data/paper_all.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/stage1_paper_all.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/stage2_paper_all.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/soulx/datasets}"
MODEL_ROOT="${MODEL_ROOT:-/data/soulx/models}"
CACHE_ROOT="${CACHE_ROOT:-/data/soulx/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/soulx/outputs}"
PIPELINE_LOG="${PIPELINE_LOG:-$OUTPUT_ROOT/stage12_pipeline.log}"
PIPELINE_TEE="${PIPELINE_TEE:-1}"
STAGE1_MANIFEST_DIR="${STAGE1_MANIFEST_DIR:-manifests/stage1_paper_all}"
STAGE2_MANIFEST_DIR="${STAGE2_MANIFEST_DIR:-manifests/stage2_paper_all}"
ALIGN_DEVICE="${ALIGN_DEVICE:-cuda}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
SKIP_DATASET_VERIFY="${SKIP_DATASET_VERIFY:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

DATASETS_FOR_VERIFY="aishell1,aishell3,wenetspeech,magicdata,commonvoice-cn,emilia-cn,librispeech,gigaspeech,commonvoice-en,emilia-en"

host_uses_docker() {
  if [[ "$USE_DOCKER" == "0" || "$USE_DOCKER" == "false" ]]; then
    return 1
  fi
  if [[ "$USE_DOCKER" == "1" || "$USE_DOCKER" == "true" ]]; then
    return 0
  fi
  command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1
}

if [[ "$IN_SOULX_DOCKER" != "1" ]] && host_uses_docker; then
  mkdir -p "$OUTPUT_ROOT"
  exec > >(tee -a "$PIPELINE_LOG") 2>&1
  echo "[pipeline] host docker mode"
  echo "[pipeline] DATA_ROOT=$DATA_ROOT MODEL_ROOT=$MODEL_ROOT CACHE_ROOT=$CACHE_ROOT OUTPUT_ROOT=$OUTPUT_ROOT"
  docker compose build
  exec docker compose run --rm \
    -e IN_SOULX_DOCKER=1 \
    -e RUN_STAGE="$RUN_STAGE" \
    -e PROFILE="$PROFILE" \
    -e STAGE1_CONFIG="$STAGE1_CONFIG" \
    -e STAGE2_CONFIG="$STAGE2_CONFIG" \
    -e DATA_ROOT=/data/datasets \
    -e MODEL_ROOT=/data/models \
    -e CACHE_ROOT=/data/cache \
    -e OUTPUT_ROOT=/data/outputs \
    -e PIPELINE_LOG=/data/outputs/stage12_pipeline.log \
    -e PIPELINE_TEE=0 \
    -e ALIGN_DEVICE="$ALIGN_DEVICE" \
    -e SKIP_MODEL_DOWNLOAD="$SKIP_MODEL_DOWNLOAD" \
    -e SKIP_DATASET_VERIFY="$SKIP_DATASET_VERIFY" \
    -e FORCE_PREPARE="$FORCE_PREPARE" \
    soulx ./scripts/run_stage12_pipeline.sh
fi

mkdir -p "$(dirname "$PIPELINE_LOG")" "$OUTPUT_ROOT" "$CACHE_ROOT" "$MODEL_ROOT"
if [[ "$PIPELINE_TEE" == "1" ]]; then
  exec > >(tee -a "$PIPELINE_LOG") 2>&1
fi

echo "[pipeline] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[pipeline] mode=$([[ "$IN_SOULX_DOCKER" == "1" ]] && echo docker || echo native) run_stage=$RUN_STAGE"
echo "[pipeline] DATA_ROOT=$DATA_ROOT"
echo "[pipeline] MODEL_ROOT=$MODEL_ROOT"
echo "[pipeline] CACHE_ROOT=$CACHE_ROOT"
echo "[pipeline] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[pipeline] log=$PIPELINE_LOG"

case "$RUN_STAGE" in
  all|prepare|stage1|stage2) ;;
  *)
    echo "[error] RUN_STAGE must be one of: all, prepare, stage1, stage2" >&2
    exit 2
    ;;
esac

if [[ "$IN_SOULX_DOCKER" != "1" && "$USE_DOCKER" != "0" && "$USE_DOCKER" != "false" ]]; then
  echo "[error] Docker is unavailable. Install Docker/NVIDIA Container Toolkit, or run with USE_DOCKER=0 inside a prepared environment." >&2
  exit 2
fi

python - <<'PY'
import torch
print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"[env] gpu{index}={props.name} memory_gib={props.total_memory / 1024**3:.2f}")
PY

if [[ "$SKIP_MODEL_DOWNLOAD" != "1" ]]; then
  echo "[pipeline] model assets check/download"
  python scripts/download_models.py \
    --all \
    --model-root "$MODEL_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --load-check
else
  echo "[pipeline] skip model download"
fi

if [[ "$SKIP_DATASET_VERIFY" != "1" ]]; then
  echo "[pipeline] verify datasets"
  python -m soulx_duplug.data.verify_datasets \
    --data-root "$DATA_ROOT" \
    --datasets "$DATASETS_FOR_VERIFY" \
    --strict
else
  echo "[pipeline] skip dataset verify"
fi

prepare_stage1_manifest() {
  local train_manifest="$STAGE1_MANIFEST_DIR/train.jsonl"
  local dev_manifest="$STAGE1_MANIFEST_DIR/dev.jsonl"
  if [[ "$FORCE_PREPARE" != "1" && -s "$train_manifest" && -s "$dev_manifest" ]]; then
    echo "[pipeline] stage1 manifest exists: $STAGE1_MANIFEST_DIR"
    return
  fi
  echo "[pipeline] build Stage 1 manifest"
  python -m soulx_duplug.data.prepare_stage1_manifest \
    --profile "$PROFILE" \
    --data-root "$DATA_ROOT" \
    --out-dir "$STAGE1_MANIFEST_DIR" \
    --with-audio-metadata \
    --prepare-archives
}

run_if_missing() {
  local output="$1"
  shift
  if [[ "$FORCE_PREPARE" != "1" && -s "$output" ]]; then
    echo "[pipeline] exists: $output"
    return
  fi
  echo "[pipeline] run: $*"
  "$@"
}

combine_alignments() {
  local zh="$1"
  local en="$2"
  local out="$3"
  if [[ "$FORCE_PREPARE" != "1" && -s "$out" ]]; then
    echo "[pipeline] exists: $out"
    return
  fi
  mkdir -p "$(dirname "$out")"
  : > "$out"
  [[ -s "$zh" ]] && cat "$zh" >> "$out"
  [[ -s "$en" ]] && cat "$en" >> "$out"
  echo "[pipeline] wrote combined alignment: $out"
}

prepare_stage2_manifest() {
  mkdir -p "$STAGE2_MANIFEST_DIR"

  run_if_missing "$STAGE2_MANIFEST_DIR/alignments.zh.train.jsonl" \
    python -m soulx_duplug.data.generate_paraformer_alignments \
      --manifest "$STAGE1_MANIFEST_DIR/train.jsonl" \
      --out "$STAGE2_MANIFEST_DIR/alignments.zh.train.jsonl" \
      --language zh

  run_if_missing "$STAGE2_MANIFEST_DIR/alignments.en.train.jsonl" \
    python -m soulx_duplug.data.generate_whisperx_alignments \
      --manifest "$STAGE1_MANIFEST_DIR/train.jsonl" \
      --out "$STAGE2_MANIFEST_DIR/alignments.en.train.jsonl" \
      --language en \
      --device "$ALIGN_DEVICE"

  combine_alignments \
    "$STAGE2_MANIFEST_DIR/alignments.zh.train.jsonl" \
    "$STAGE2_MANIFEST_DIR/alignments.en.train.jsonl" \
    "$STAGE2_MANIFEST_DIR/alignments.train.jsonl"

  run_if_missing "$STAGE2_MANIFEST_DIR/train.jsonl" \
    python -m soulx_duplug.data.stage2_chunks \
      --manifest "$STAGE1_MANIFEST_DIR/train.jsonl" \
      --alignment "$STAGE2_MANIFEST_DIR/alignments.train.jsonl" \
      --out "$STAGE2_MANIFEST_DIR/train.jsonl" \
      --chunk-seconds 0.16

  run_if_missing "$STAGE2_MANIFEST_DIR/alignments.zh.dev.jsonl" \
    python -m soulx_duplug.data.generate_paraformer_alignments \
      --manifest "$STAGE1_MANIFEST_DIR/dev.jsonl" \
      --out "$STAGE2_MANIFEST_DIR/alignments.zh.dev.jsonl" \
      --language zh

  run_if_missing "$STAGE2_MANIFEST_DIR/alignments.en.dev.jsonl" \
    python -m soulx_duplug.data.generate_whisperx_alignments \
      --manifest "$STAGE1_MANIFEST_DIR/dev.jsonl" \
      --out "$STAGE2_MANIFEST_DIR/alignments.en.dev.jsonl" \
      --language en \
      --device "$ALIGN_DEVICE"

  combine_alignments \
    "$STAGE2_MANIFEST_DIR/alignments.zh.dev.jsonl" \
    "$STAGE2_MANIFEST_DIR/alignments.en.dev.jsonl" \
    "$STAGE2_MANIFEST_DIR/alignments.dev.jsonl"

  run_if_missing "$STAGE2_MANIFEST_DIR/dev.jsonl" \
    python -m soulx_duplug.data.stage2_chunks \
      --manifest "$STAGE1_MANIFEST_DIR/dev.jsonl" \
      --alignment "$STAGE2_MANIFEST_DIR/alignments.dev.jsonl" \
      --out "$STAGE2_MANIFEST_DIR/dev.jsonl" \
      --chunk-seconds 0.16
}

case "$RUN_STAGE" in
  all|prepare|stage1)
    prepare_stage1_manifest
    ;;
esac

case "$RUN_STAGE" in
  all|stage1)
    echo "[pipeline] start Stage 1 training"
    ./scripts/train_stage1.sh "$STAGE1_CONFIG"
    ;;
esac

case "$RUN_STAGE" in
  all|prepare|stage2)
    prepare_stage1_manifest
    prepare_stage2_manifest
    ;;
esac

case "$RUN_STAGE" in
  all|stage2)
    echo "[pipeline] start Stage 2 training"
    ./scripts/train_stage2.sh "$STAGE2_CONFIG"
    ;;
esac

echo "[pipeline] complete $(date '+%Y-%m-%d %H:%M:%S')"
