#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_STAGE="${RUN_STAGE:-all}"
PROFILE="${PROFILE:-configs/data/paper_all.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/stage1_paper_all.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/stage2_paper_all.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/soulx/datasets}"
MODEL_ROOT="${MODEL_ROOT:-/data/soulx/models}"
CACHE_ROOT="${CACHE_ROOT:-/data/soulx/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/soulx/outputs}"
PIPELINE_LOG="${PIPELINE_LOG:-$OUTPUT_ROOT/stage12_pipeline.log}"
STAGE1_MANIFEST_DIR="${STAGE1_MANIFEST_DIR:-manifests/stage1_paper_all}"
STAGE2_MANIFEST_DIR="${STAGE2_MANIFEST_DIR:-manifests/stage2_paper_all}"
ALIGN_DEVICE="${ALIGN_DEVICE:-cuda}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
SKIP_DATASET_VERIFY="${SKIP_DATASET_VERIFY:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
SETUP_CONDA="${SETUP_CONDA:-1}"
ALLOW_STAGE2_WITHOUT_STAGE1="${ALLOW_STAGE2_WITHOUT_STAGE1:-0}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$CACHE_ROOT/conda_envs/soulx-duplug}"
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.10}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.1.2}"
PYTORCH_CUDA="${PYTORCH_CUDA:-11.8}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"

DATASETS_FOR_VERIFY="aishell1,aishell3,wenetspeech,magicdata,commonvoice-cn,emilia-cn,librispeech,gigaspeech,commonvoice-en,emilia-en"

mkdir -p "$(dirname "$PIPELINE_LOG")" "$OUTPUT_ROOT" "$CACHE_ROOT" "$MODEL_ROOT"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[pipeline] start $(date '+%Y-%m-%d %H:%M:%S')"
echo "[pipeline] mode=conda run_stage=$RUN_STAGE"
echo "[pipeline] DATA_ROOT=$DATA_ROOT"
echo "[pipeline] MODEL_ROOT=$MODEL_ROOT"
echo "[pipeline] CACHE_ROOT=$CACHE_ROOT"
echo "[pipeline] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[pipeline] CONDA_ENV_DIR=$CONDA_ENV_DIR"
echo "[pipeline] log=$PIPELINE_LOG"

case "$RUN_STAGE" in
  all|prepare|stage1|stage2) ;;
  *)
    echo "[error] RUN_STAGE must be one of: all, prepare, stage1, stage2" >&2
    exit 2
    ;;
esac

safe_remove_env_dir() {
  local target
  target="$(cd "$(dirname "$CONDA_ENV_DIR")" && pwd)/$(basename "$CONDA_ENV_DIR")"
  case "$target" in
    "/"|"$HOME"|"$ROOT_DIR"|"$ROOT_DIR/.."|"$DATA_ROOT"|"$MODEL_ROOT"|"$CACHE_ROOT"|"$OUTPUT_ROOT")
      echo "[error] refusing to remove unsafe CONDA_ENV_DIR: $target" >&2
      return 1
      ;;
  esac
  rm -rf "$target"
}

setup_conda_env() {
  if [[ "$SETUP_CONDA" == "0" || "$SETUP_CONDA" == "false" ]]; then
    echo "[env] skip conda setup; using current Python environment"
    return
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda not found. Install Miniconda/Anaconda first, or run with SETUP_CONDA=0 inside a prepared env." >&2
    exit 2
  fi

  mkdir -p "$CACHE_ROOT" "$(dirname "$CONDA_ENV_DIR")"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda_pkgs}"
  mkdir -p "$CONDA_PKGS_DIRS"

  eval "$(conda shell.bash hook)"

  if [[ -d "$CONDA_ENV_DIR/conda-meta" ]]; then
    echo "[env] reuse existing conda env: $CONDA_ENV_DIR"
  else
    if [[ -e "$CONDA_ENV_DIR" ]]; then
      echo "[env] invalid conda env path exists; rebuild: $CONDA_ENV_DIR"
      safe_remove_env_dir
    fi
    echo "[env] create conda env: $CONDA_ENV_DIR python=$CONDA_PYTHON_VERSION"
    conda create -p "$CONDA_ENV_DIR" "python=$CONDA_PYTHON_VERSION" -y
  fi

  conda activate "$CONDA_ENV_DIR"
  echo "[env] active python=$(command -v python)"

  if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    python - <<'PY'
import torch
print(f"[env] torch already installed: {torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
PY
  else
    echo "[env] install PyTorch $PYTORCH_VERSION with CUDA $PYTORCH_CUDA"
    conda install -p "$CONDA_ENV_DIR" -y \
      "pytorch==$PYTORCH_VERSION" \
      "torchaudio==$PYTORCH_VERSION" \
      "pytorch-cuda=$PYTORCH_CUDA" \
      -c pytorch -c nvidia
  fi

  local deps_stamp="$CONDA_ENV_DIR/.soulx_deps_installed"
  local need_deps=0
  if [[ "$INSTALL_DEPS" == "1" || "$INSTALL_DEPS" == "true" ]]; then
    need_deps=1
  elif [[ "$INSTALL_DEPS" == "0" || "$INSTALL_DEPS" == "false" ]]; then
    need_deps=0
  elif [[ ! -f "$deps_stamp" || requirements.txt -nt "$deps_stamp" ]]; then
    need_deps=1
  fi

  if [[ "$need_deps" == "1" ]]; then
    echo "[env] install system-level Python deps into conda env"
    conda install -p "$CONDA_ENV_DIR" -y -c conda-forge ffmpeg libsndfile
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    date '+%Y-%m-%d %H:%M:%S' > "$deps_stamp"
  else
    echo "[env] Python deps already installed; set INSTALL_DEPS=1 to reinstall"
  fi
}

setup_conda_env

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

ensure_stage1_checkpoint_for_stage2() {
  local state_path
  state_path="$(
    python - "$STAGE2_CONFIG" <<'PY'
import sys
from pathlib import Path
from soulx_duplug.config import load_yaml, resolve_path

cfg = load_yaml(Path(sys.argv[1]))
checkpoint = cfg.get("stage1_checkpoint")
print(resolve_path(checkpoint) / "pytorch_model.bin" if checkpoint else "")
PY
  )"
  if [[ -z "$state_path" ]]; then
    echo "[pipeline] stage2 config has no stage1_checkpoint"
    return
  fi
  if [[ -f "$state_path" ]]; then
    echo "[pipeline] Stage 1 checkpoint ready: $state_path"
    return
  fi
  if [[ "$ALLOW_STAGE2_WITHOUT_STAGE1" == "1" ]]; then
    echo "[warn] Stage 1 checkpoint missing, but ALLOW_STAGE2_WITHOUT_STAGE1=1: $state_path"
    return
  fi
  echo "[error] Stage 1 checkpoint missing for Stage 2: $state_path" >&2
  echo "[error] Run Stage 1 first: RUN_STAGE=stage1 ./scripts/run_stage12_pipeline.sh" >&2
  exit 2
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
    ensure_stage1_checkpoint_for_stage2
    echo "[pipeline] start Stage 2 training"
    ./scripts/train_stage2.sh "$STAGE2_CONFIG"
    ;;
esac

echo "[pipeline] complete $(date '+%Y-%m-%d %H:%M:%S')"
