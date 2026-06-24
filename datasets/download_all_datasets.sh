#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="${ENV_DIR:-$SCRIPT_DIR/.dataset-download-env}"
PROFILE="${PROFILE:-$SCRIPT_DIR/../configs/data/paper_all.yaml}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/download.log}"
DELETE_ENV_AFTER="${DELETE_ENV_AFTER:-0}"
USING_DOWNLOAD_ENV=0

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

safe_remove_env_dir() {
  local target
  target="$(cd "$(dirname "$ENV_DIR")" && pwd)/$(basename "$ENV_DIR")"
  case "$target" in
    "/"|"$HOME"|"$SCRIPT_DIR"|"$PROJECT_DIR"|"$SCRIPT_DIR/..")
      echo "[error] refusing to remove unsafe ENV_DIR: $target" >&2
      return 1
      ;;
  esac
  rm -rf "$target"
}

cleanup() {
  local exit_code=$?
  trap - EXIT
  set +e
  if [[ "$USING_DOWNLOAD_ENV" == "1" ]]; then
    conda deactivate >/dev/null 2>&1 || true
  fi
  if [[ "$DELETE_ENV_AFTER" == "1" && -d "$ENV_DIR" ]]; then
    conda env remove -p "$ENV_DIR" -y || safe_remove_env_dir
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
echo "[env] ENV_DIR=$ENV_DIR DELETE_ENV_AFTER=$DELETE_ENV_AFTER"
export SOULX_TEE_LOG_FILE="$LOG_FILE"

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

if grep -q "source_dataset: wenetspeech" "$PROFILE"; then
  if [[ -z "${WENETSPEECH_PASSWORD:-}" && -z "${WENETSPEECH_PASSWORD_FILE:-}" ]]; then
    echo "请先申请 WenetSpeech 官方下载密码，然后设置："
    echo "  export WENETSPEECH_PASSWORD='你的密码'"
    echo "或者："
    echo "  export WENETSPEECH_PASSWORD_FILE=/path/to/wenetspeech_password.txt"
    exit 1
  fi
fi

APT_PACKAGES=()
command -v aria2c >/dev/null 2>&1 || APT_PACKAGES+=(aria2)
command -v git >/dev/null 2>&1 || APT_PACKAGES+=(git)
command -v wget >/dev/null 2>&1 || APT_PACKAGES+=(wget)
command -v openssl >/dev/null 2>&1 || APT_PACKAGES+=(openssl)

if (( ${#APT_PACKAGES[@]} > 0 )) && command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y "${APT_PACKAGES[@]}"
fi

eval "$(conda shell.bash hook)"

if [[ -d "$ENV_DIR/conda-meta" ]]; then
  echo "[env] 复用已有下载环境：$ENV_DIR"
else
  if [[ -e "$ENV_DIR" ]]; then
    echo "[env] 发现无效下载环境，重建：$ENV_DIR"
    safe_remove_env_dir
  fi
  echo "[env] 创建下载环境：$ENV_DIR"
  conda create -p "$ENV_DIR" python=3.10 -y
fi

conda activate "$ENV_DIR"
USING_DOWNLOAD_ENV=1

if python -c "import yaml, huggingface_hub" >/dev/null 2>&1; then
  echo "[env] 依赖已存在：PyYAML huggingface_hub"
else
  echo "[env] 安装下载依赖：PyYAML huggingface_hub"
  pip install PyYAML huggingface_hub
fi

hf auth login --token "$HF_TOKEN"

python3 "$SCRIPT_DIR/download_asr_datasets.py" \
  --profile "$PROFILE" \
  --out-dir "$OUT_DIR" \
  --log-file "$LOG_FILE" \
  --dry-run

python3 "$SCRIPT_DIR/download_asr_datasets.py" \
  --profile "$PROFILE" \
  --out-dir "$OUT_DIR" \
  --log-file "$LOG_FILE" \
  --extract
