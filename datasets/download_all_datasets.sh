#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$SCRIPT_DIR/.dataset-download-env}"
PROFILE="${PROFILE:-$SCRIPT_DIR/../configs/data/paper_all.yaml}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/download.log}"
CLEANUP_ENV=0

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

cleanup() {
  local exit_code=$?
  trap - EXIT
  set +e
  if [[ "$CLEANUP_ENV" == "1" ]]; then
    conda deactivate >/dev/null 2>&1 || true
    conda env remove -p "$ENV_DIR" -y || rm -rf "$ENV_DIR"
  fi
  if [[ "$exit_code" -eq 0 ]]; then
    echo "[success] 全部数据集下载流程完成"
  else
    echo "[error] 数据集下载流程失败，退出码：$exit_code"
  fi
  echo "[log] 日志文件：$LOG_FILE"
  exit "$exit_code"
}

trap cleanup EXIT

echo "[log] 日志文件：$LOG_FILE"
echo "[start] $(date '+%Y-%m-%d %H:%M:%S')"

if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_xxx" ]]; then
  echo "请先设置 HF_TOKEN：export HF_TOKEN=hf_xxx"
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/download_asr_datasets.py" ]]; then
  echo "找不到 $SCRIPT_DIR/download_asr_datasets.py"
  exit 1
fi

if [[ ! -f "$PROFILE" ]]; then
  echo "找不到 profile：$PROFILE"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "找不到 conda"
  exit 1
fi

if ! command -v aria2c >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y aria2
fi

eval "$(conda shell.bash hook)"

if [[ -d "$ENV_DIR" ]]; then
  conda env remove -p "$ENV_DIR" -y || rm -rf "$ENV_DIR"
fi

conda create -p "$ENV_DIR" python=3.10 -y
CLEANUP_ENV=1
conda activate "$ENV_DIR"
pip install PyYAML huggingface_hub
hf auth login --token "$HF_TOKEN"

python3 "$SCRIPT_DIR/download_asr_datasets.py" \
  --profile "$PROFILE" \
  --out-dir "$OUT_DIR" \
  --dry-run

python3 "$SCRIPT_DIR/download_asr_datasets.py" \
  --profile "$PROFILE" \
  --out-dir "$OUT_DIR" \
  --extract
