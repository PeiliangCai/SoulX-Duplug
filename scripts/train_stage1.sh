#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/stage1_paper_all.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

CONFIG_DEVICE="$(
  python -c 'import sys, yaml; cfg=yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}; print(cfg.get("device", ""))' "$CONFIG" 2>/dev/null || true
)"
GPU_COUNT="$(
  python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0
)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[train-stage1] config=$CONFIG"
echo "[train-stage1] config_device=${CONFIG_DEVICE:-auto} visible_gpu_count=$GPU_COUNT"

if [[ "$GPU_COUNT" -gt 1 && "$CONFIG_DEVICE" != "cpu" ]]; then
  exec torchrun --standalone --nproc_per_node "$GPU_COUNT" \
    -m soulx_duplug.train.stage1_asr \
    --config "$CONFIG" \
    "$@"
fi

exec python -m soulx_duplug.train.stage1_asr \
  --config "$CONFIG" \
  "$@"
