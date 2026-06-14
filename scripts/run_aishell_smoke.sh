#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-dummy}"
SMOKE_DEVICE="${SMOKE_DEVICE:-cpu}"
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d /root/autodl-tmp/datasets/aishell1 || -d /root/autodl-tmp/datasets/aishell3 ]]; then
    DATA_ROOT="/root/autodl-tmp/datasets"
  else
    DATA_ROOT="$ROOT_DIR/datasets"
  fi
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
SMOKE_ROOT="${SMOKE_ROOT:-manifests/aishell_tiny}"
NORMALIZED_ROOT="${NORMALIZED_ROOT:-$ROOT_DIR/datasets/normalized/aishell_tiny}"
TINY_TRAIN="${TINY_TRAIN:-64}"
TINY_DEV="${TINY_DEV:-16}"
PAPER_TRAIN_ALIGN_LIMIT="${PAPER_TRAIN_ALIGN_LIMIT:-8}"
PAPER_DEV_ALIGN_LIMIT="${PAPER_DEV_ALIGN_LIMIT:-2}"
RUN_LOG="${RUN_LOG:-$OUTPUT_ROOT/aishell_smoke_run.log}"

export OUTPUT_ROOT
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RUN_LOG")"
: > "$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[start] MODE=$MODE SMOKE_DEVICE=$SMOKE_DEVICE"
echo "[paths] DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[log] $RUN_LOG"

case "$SMOKE_DEVICE" in
  cpu)
    DUMMY_STAGE1_CONFIG="configs/stage1_aishell_tiny.yaml"
    DUMMY_STAGE2_CONFIG="configs/stage2_aishell_tiny.yaml"
    DUMMY_STAGE1_OUTPUT="$OUTPUT_ROOT/stage1_aishell_tiny"
    DUMMY_STAGE2_OUTPUT="$OUTPUT_ROOT/stage2_aishell_tiny"
    ;;
  cuda)
    python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' || {
      echo "[error] SMOKE_DEVICE=cuda but torch.cuda.is_available() is false" >&2
      exit 1
    }
    DUMMY_STAGE1_CONFIG="configs/stage1_aishell_tiny_cuda.yaml"
    DUMMY_STAGE2_CONFIG="configs/stage2_aishell_tiny_cuda.yaml"
    DUMMY_STAGE1_OUTPUT="$OUTPUT_ROOT/stage1_aishell_tiny_cuda"
    DUMMY_STAGE2_OUTPUT="$OUTPUT_ROOT/stage2_aishell_tiny_cuda"
    ;;
  *)
    echo "Unknown SMOKE_DEVICE=$SMOKE_DEVICE. Use SMOKE_DEVICE=cpu or SMOKE_DEVICE=cuda." >&2
    exit 2
    ;;
esac

require_nonempty() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "[error] $label is empty or missing: $path" >&2
    echo "[hint] Set DATA_ROOT to the directory containing aishell1/ and aishell3/." >&2
    exit 1
  fi
}

prepare_aishell_manifest() {
  python -m soulx_duplug.data.prepare_stage1_manifest \
    --profile configs/data/local_aishell.yaml \
    --data-root "$DATA_ROOT" \
    --out-dir manifests/stage1_aishell \
    --prepare-archives
}

make_tiny_manifests() {
  require_nonempty manifests/stage1_aishell/train.jsonl "AISHELL train manifest"
  require_nonempty manifests/stage1_aishell/dev.jsonl "AISHELL dev manifest"
  mkdir -p "$SMOKE_ROOT/stage1" "$SMOKE_ROOT/stage2"
  head -n "$TINY_TRAIN" manifests/stage1_aishell/train.jsonl > "$SMOKE_ROOT/stage1/train.jsonl"
  head -n "$TINY_DEV" manifests/stage1_aishell/dev.jsonl > "$SMOKE_ROOT/stage1/dev.jsonl"

  python -m soulx_duplug.data.normalize_audio \
    --manifest "$SMOKE_ROOT/stage1/train.jsonl" \
    --out-manifest "$SMOKE_ROOT/stage1/train.normalized.jsonl" \
    --output-root "$NORMALIZED_ROOT" \
    --strict

  python -m soulx_duplug.data.normalize_audio \
    --manifest "$SMOKE_ROOT/stage1/dev.jsonl" \
    --out-manifest "$SMOKE_ROOT/stage1/dev.normalized.jsonl" \
    --output-root "$NORMALIZED_ROOT" \
    --strict

  python -m soulx_duplug.data.stage2_chunks \
    --manifest "$SMOKE_ROOT/stage1/train.normalized.jsonl" \
    --out "$SMOKE_ROOT/stage2/train.jsonl" \
    --allow-uniform-fallback

  python -m soulx_duplug.data.stage2_chunks \
    --manifest "$SMOKE_ROOT/stage1/dev.normalized.jsonl" \
    --out "$SMOKE_ROOT/stage2/dev.jsonl" \
    --allow-uniform-fallback
}

run_dummy_smoke() {
  rm -f "$DUMMY_STAGE1_OUTPUT"/log*.txt "$DUMMY_STAGE2_OUTPUT"/log*.txt
  prepare_aishell_manifest
  make_tiny_manifests

  ./scripts/train_stage1.sh "$DUMMY_STAGE1_CONFIG" \
    --log-file "$DUMMY_STAGE1_OUTPUT/log.txt"

  ./scripts/train_stage2.sh "$DUMMY_STAGE2_CONFIG" \
    --log-file "$DUMMY_STAGE2_OUTPUT/log.txt"

  echo "[done] dummy smoke complete"
  echo "[stage1-log] $DUMMY_STAGE1_OUTPUT/log.txt"
  echo "[stage2-log] $DUMMY_STAGE2_OUTPUT/log.txt"
}

run_paper_smoke() {
  rm -f "$OUTPUT_ROOT/stage1_aishell_paper_smoke/log.txt" "$OUTPUT_ROOT/stage2_aishell_paper_smoke/log.txt"
  prepare_aishell_manifest

  python -m soulx_duplug.train.stage1_asr \
    --config configs/stage1_aishell_paper_smoke.yaml \
    --log-file "$OUTPUT_ROOT/stage1_aishell_paper_smoke/log.txt"

  mkdir -p manifests/stage2_aishell_smoke
  python -m soulx_duplug.data.generate_paraformer_alignments \
    --manifest manifests/stage1_aishell/train.jsonl \
    --out manifests/stage2_aishell_smoke/alignments.train.jsonl \
    --language zh \
    --limit "$PAPER_TRAIN_ALIGN_LIMIT"

  python -m soulx_duplug.data.generate_paraformer_alignments \
    --manifest manifests/stage1_aishell/dev.jsonl \
    --out manifests/stage2_aishell_smoke/alignments.dev.jsonl \
    --language zh \
    --limit "$PAPER_DEV_ALIGN_LIMIT"

  python -m soulx_duplug.data.stage2_chunks \
    --manifest manifests/stage1_aishell/train.jsonl \
    --alignment manifests/stage2_aishell_smoke/alignments.train.jsonl \
    --out manifests/stage2_aishell_smoke/train.jsonl \
    --limit "$PAPER_TRAIN_ALIGN_LIMIT"

  python -m soulx_duplug.data.stage2_chunks \
    --manifest manifests/stage1_aishell/dev.jsonl \
    --alignment manifests/stage2_aishell_smoke/alignments.dev.jsonl \
    --out manifests/stage2_aishell_smoke/dev.jsonl \
    --limit "$PAPER_DEV_ALIGN_LIMIT"

  python -m soulx_duplug.train.stage2_streaming_asr \
    --config configs/stage2_aishell_paper_smoke.yaml \
    --log-file "$OUTPUT_ROOT/stage2_aishell_paper_smoke/log.txt"

  echo "[done] paper smoke complete"
  echo "[stage1-log] $OUTPUT_ROOT/stage1_aishell_paper_smoke/log.txt"
  echo "[stage2-log] $OUTPUT_ROOT/stage2_aishell_paper_smoke/log.txt"
}

case "$MODE" in
  dummy)
    run_dummy_smoke
    ;;
  paper)
    run_paper_smoke
    ;;
  *)
    echo "Unknown MODE=$MODE. Use MODE=dummy or MODE=paper." >&2
    exit 2
    ;;
esac
