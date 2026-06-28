# SoulX-Duplug

本仓库用于复现 SoulX-Duplug 的 Stage 1/2 ASR 训练流程。

## 当前实现是否接近论文

已经尽可能贴近论文公开描述：

- Stage 1：Non-Streaming ASR Pretraining。
- Stage 2：Streaming ASR Adaptation。
- 语音 tokenizer：GLM-4-Voice tokenizer，训练时冻结。
- 语言模型：Qwen3-0.6B。
- Stage 2：160 ms chunk、960 ms look-back、40 ms look-ahead。
- Stage 1：支持从音频前缀开始自回归生成完整转写。
- Stage 2：按 chunk 自由生成 ASR token，直到 `<asr_eos>`。
- 中文对齐：Paraformer。
- 英文对齐：WhisperX。
- 数据：论文列出的中英文 ASR 数据集混合训练。
- 规模筛选：按论文公开的中文约 47,000 小时、英文约 31,000 小时筛选 train split。

论文的最终系统还包含 Stage 3 状态预测 SFT；当前仓库先实现并训练 Stage 1/2。论文没有公开 batch size、学习率、训练步数、精确数据采样权重、每个数据集具体小时配额。代码中这些参数是可复现的工程默认值，不伪装成论文原始超参。

## 环境部署

推荐用 Docker。新服务器只需要有 NVIDIA 驱动、Docker、NVIDIA Container Toolkit。

注意：Docker 命令需要在宿主机执行。如果你已经进入了 AutoDL 这类平台提供的容器，通常不能再执行 `docker compose`，因为容器里没有 Docker daemon。

```bash
cd /root/SoulX-Duplug

export DATA_ROOT=/data/soulx/datasets
export MODEL_ROOT=/data/soulx/models
export CACHE_ROOT=/data/soulx/cache
export OUTPUT_ROOT=/data/soulx/outputs
export HF_TOKEN=hf_xxx
export WENETSPEECH_PASSWORD='你的 WenetSpeech 官方密码'

docker compose build
docker compose run --rm soulx bash
```

如果只想直接在容器里跑命令：

```bash
docker compose run --rm soulx ./scripts/run_aishell_smoke.sh
docker compose run --rm soulx python scripts/download_models.py --all
```

## 一键 Stage 1/2 训练

新服务器上推荐直接用 Docker 一键启动。脚本会构建 Docker 环境、检查/下载模型、验证数据集、生成 manifest、启动 Stage 1，并在 Stage 1 完成后生成 Stage 2 对齐与 chunk manifest，再启动 Stage 2：

```bash
cd /root/SoulX-Duplug

export DATA_ROOT=/data/soulx/datasets
export MODEL_ROOT=/data/soulx/models
export CACHE_ROOT=/data/soulx/cache
export OUTPUT_ROOT=/data/soulx/outputs
export HF_TOKEN=hf_xxx

tmux new -s soulx_train
./scripts/run_stage12_pipeline.sh
```

只跑 Stage 1：

```bash
RUN_STAGE=stage1 ./scripts/run_stage12_pipeline.sh
```

Stage 1 完成后只跑 Stage 2：

```bash
RUN_STAGE=stage2 ./scripts/run_stage12_pipeline.sh
```

一键脚本日志追加写入：

```text
${OUTPUT_ROOT}/stage12_pipeline.log
```

训练中断后重复执行同一条命令会自动从 `${OUTPUT_ROOT}/stage1_paper_all/checkpoints/latest` 或 `${OUTPUT_ROOT}/stage2_paper_all/checkpoints/latest` 恢复。强制从头训练时，在对应配置里设置：

```yaml
training:
  resume_from_checkpoint: false
```

依赖文件只保留：

```text
requirements.txt
```

Docker 仍然需要 `requirements.txt`，它用于安装 transformers、FunASR、WhisperX 等项目依赖。PyTorch/CUDA 不放在 requirements 里，由 Docker 的 PyTorch CUDA 基础镜像提供。
非 Docker 手动安装时，也应先安装匹配服务器 CUDA 的 PyTorch，再执行：

```bash
pip install -r requirements.txt
```

在 AutoDL 容器内通常直接使用非 Docker 方式：

```bash
cd /root/SoulX-Duplug
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
pip install -r requirements.txt
./scripts/run_aishell_smoke.sh
```

多 GPU 训练使用脚本启动。脚本会自动检测当前可见 GPU 数量；4 卡、6 卡、8 卡不需要改命令。需要限制 GPU 时使用 `CUDA_VISIBLE_DEVICES`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_stage1.sh configs/stage1_paper_all.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_stage2.sh configs/stage2_paper_all.yaml
```

## 快速检查当前服务器

当前服务器只有 AISHELL-1/3 时，先跑最小 smoke test，确认 Stage 1/2 不崩溃：

```bash
cd /root/SoulX-Duplug
./scripts/run_aishell_smoke.sh
```

默认 `MODE=dummy`、`SMOKE_DEVICE=cpu`，使用 CPU、dummy tokenizer、tiny AISHELL 子集，只验证：

- 数据读取
- 音频规范化
- Stage 1 训练循环
- Stage 2 chunk manifest
- Stage 2 训练循环
- checkpoint 保存
- log 文件写入

日志位置：

```text
outputs/aishell_smoke_run.log
outputs/stage1_aishell_tiny/log.txt
outputs/stage2_aishell_tiny/log.txt
```

如果要验证 CUDA / 多 GPU DDP 路径：

```bash
SMOKE_DEVICE=cuda ./scripts/run_aishell_smoke.sh
```

CUDA smoke 仍然使用 dummy 小模型，但训练会在 GPU 上执行。多张 GPU 可见时，脚本会自动用 `torchrun` 启动多进程 DDP。日志位置：

```text
outputs/stage1_aishell_tiny_cuda/log.txt
outputs/stage1_aishell_tiny_cuda/log.rank1.txt
outputs/stage2_aishell_tiny_cuda/log.txt
outputs/stage2_aishell_tiny_cuda/log.rank1.txt
```

如果要验证真实 GLM/Qwen/Paraformer 链路：

```bash
export MODEL_ROOT=/root/autodl-tmp/models
export OUTPUT_ROOT=/root/autodl-tmp/outputs
MODE=paper ./scripts/run_aishell_smoke.sh
```

`MODE=paper` 需要本地已经准备好：

- `${MODEL_ROOT}/Qwen3-0.6B`
- `${MODEL_ROOT}/glm-4-voice-tokenizer`
- `${MODEL_ROOT}/GLM-4-Voice`
- FunASR / ModelScope / Paraformer 依赖

## 数据下载

详细下载说明见：

```text
datasets/README.md
```

一键下载入口：

```bash
cd /root/SoulX-Duplug/datasets
export HF_TOKEN=hf_xxx
export WENETSPEECH_PASSWORD='你的 WenetSpeech 官方密码'
./download_all_datasets.sh
```

WenetSpeech 官方 toolkit 会由脚本自动 clone 并在下载后清理，不需要手动保留该仓库。

下载日志：

```text
/root/SoulX-Duplug/datasets/download.log
```

## 论文全量 Stage 1

准备环境变量：

```bash
export DATA_ROOT=/data/soulx/datasets
export MODEL_ROOT=/data/soulx/models
export CACHE_ROOT=/data/soulx/cache
export OUTPUT_ROOT=/data/soulx/outputs
export HF_TOKEN=hf_xxx
export WENETSPEECH_PASSWORD='你的 WenetSpeech 官方密码'
```

下载模型：

```bash
docker compose run --rm soulx python scripts/download_models.py --all
```

下载论文数据集：

```bash
docker compose run --rm soulx python datasets/download_asr_datasets.py --profile configs/data/paper_all.yaml --dry-run
docker compose run --rm soulx python datasets/download_asr_datasets.py --profile configs/data/paper_all.yaml --extract
```

生成 Stage 1 manifest：

```bash
docker compose run --rm soulx python -m soulx_duplug.data.prepare_stage1_manifest \
  --profile configs/data/paper_all.yaml \
  --data-root /data/datasets \
  --out-dir manifests/stage1_paper_all \
  --with-audio-metadata
```

检查实际筛选规模：

```bash
cat manifests/stage1_paper_all/selection.summary.json
```

启动 Stage 1：

```bash
docker compose run --rm soulx ./scripts/train_stage1.sh configs/stage1_paper_all.yaml
```

Stage 1 日志：

```text
${OUTPUT_ROOT}/stage1_paper_all/log.txt
${OUTPUT_ROOT}/stage1_paper_all/log.rank1.txt
${OUTPUT_ROOT}/stage1_paper_all/log.rank2.txt
```

日志内容说明见“训练日志怎么看”。

## 论文全量 Stage 2

生成中文对齐：

```bash
docker compose run --rm soulx python -m soulx_duplug.data.generate_paraformer_alignments \
  --manifest manifests/stage1_paper_all/train.jsonl \
  --out manifests/stage2_paper_all/alignments.zh.train.jsonl \
  --language zh
```

生成英文对齐：

```bash
docker compose run --rm soulx python -m soulx_duplug.data.generate_whisperx_alignments \
  --manifest manifests/stage1_paper_all/train.jsonl \
  --out manifests/stage2_paper_all/alignments.en.train.jsonl \
  --language en
```

合并并生成 Stage 2 train manifest：

```bash
cat manifests/stage2_paper_all/alignments.zh.train.jsonl \
    manifests/stage2_paper_all/alignments.en.train.jsonl \
  > manifests/stage2_paper_all/alignments.train.jsonl

docker compose run --rm soulx python -m soulx_duplug.data.stage2_chunks \
  --manifest manifests/stage1_paper_all/train.jsonl \
  --alignment manifests/stage2_paper_all/alignments.train.jsonl \
  --out manifests/stage2_paper_all/train.jsonl
```

dev split 也按同样方式生成：

```bash
docker compose run --rm soulx python -m soulx_duplug.data.generate_paraformer_alignments \
  --manifest manifests/stage1_paper_all/dev.jsonl \
  --out manifests/stage2_paper_all/alignments.zh.dev.jsonl \
  --language zh

docker compose run --rm soulx python -m soulx_duplug.data.generate_whisperx_alignments \
  --manifest manifests/stage1_paper_all/dev.jsonl \
  --out manifests/stage2_paper_all/alignments.en.dev.jsonl \
  --language en

cat manifests/stage2_paper_all/alignments.zh.dev.jsonl \
    manifests/stage2_paper_all/alignments.en.dev.jsonl \
  > manifests/stage2_paper_all/alignments.dev.jsonl

docker compose run --rm soulx python -m soulx_duplug.data.stage2_chunks \
  --manifest manifests/stage1_paper_all/dev.jsonl \
  --alignment manifests/stage2_paper_all/alignments.dev.jsonl \
  --out manifests/stage2_paper_all/dev.jsonl
```

启动 Stage 2：

```bash
docker compose run --rm soulx ./scripts/train_stage2.sh configs/stage2_paper_all.yaml
```

Stage 2 日志：

```text
${OUTPUT_ROOT}/stage2_paper_all/log.txt
${OUTPUT_ROOT}/stage2_paper_all/log.rank1.txt
${OUTPUT_ROOT}/stage2_paper_all/log.rank2.txt
```

日志内容说明见“训练日志怎么看”。

## 训练日志怎么看

训练日志是按行写入的 JSON 事件，服务器后台运行时主要看这些事件：

```text
logger_ready       日志文件已创建
train_start        训练开始，包含 output_dir、checkpoint_dir
config             当前训练配置完整快照
runtime            Python / PyTorch / CUDA / GPU 信息
distributed_ready  是否多 GPU、world_size、rank、local_rank
data_paths         train/dev manifest 路径
manifest_loaded    数据条数、小时数、语言和数据集统计
tokenizers_ready   文本 tokenizer、语音 tokenizer、词表大小
model_ready        模型类型、总参数量、可训练参数量、显存占用
stage1_checkpoint_loaded Stage 2 已加载 Stage 1 权重及词表扩展信息
training_ready     batch size、world_size、effective_batch_size、学习率、max_steps、eval/save 间隔
checkpoint_resumed 从 checkpoint 恢复训练，包含 checkpoint 路径和 step
checkpoint_resume_skipped 没有发现 latest checkpoint，从头训练
train_step         当前 step 的 train_loss
eval               dev loss、CER/WER
eval_prediction    少量 reference/hypothesis 对照样例
training_curves_updated 训练曲线已更新，包含图片和指标文件路径
training_curves_failed  绘图失败；训练会继续运行
checkpoint_saved   checkpoint 保存位置
train_complete     训练正常结束
train_failed       训练失败，后面会跟 traceback
```

最重要的是：

- `manifest_loaded`：确认训练数据不是 0，数据集和语言分布正确。
- `checkpoint_resumed`：确认重启训练时确实从已有 checkpoint 恢复。
- `train_step`：确认训练在前进，`train_loss` 有正常数值，不是 `nan`。
- `eval`：看 dev loss、中文 `cer_zh`、英文 `wer_en`。
- `eval_prediction`：直接检查模型生成文本是否逐步接近参考文本。
- `training_curves_updated`：确认训练中的曲线图片已经更新。
- `checkpoint_saved`：确认 checkpoint 路径已经保存。
- `train_failed`：如果出现，直接看它后面的 `exception_type`、`message` 和 traceback。

常用查看命令：

```bash
tail -f ${OUTPUT_ROOT}/stage1_paper_all/log.txt
tail -f ${OUTPUT_ROOT}/stage2_paper_all/log.txt
grep '"event": "train_failed"' ${OUTPUT_ROOT}/stage1_paper_all/log.txt
grep '"event": "checkpoint_saved"' ${OUTPUT_ROOT}/stage1_paper_all/log.txt
grep '"event": "checkpoint_resumed"' ${OUTPUT_ROOT}/stage1_paper_all/log.txt
grep '"event": "eval"' ${OUTPUT_ROOT}/stage1_paper_all/log.txt
```

训练期间会自动生成：

```text
${OUTPUT_ROOT}/stage1_paper_all/training_curves.png
${OUTPUT_ROOT}/stage1_paper_all/training_metrics.jsonl
${OUTPUT_ROOT}/stage2_paper_all/training_curves.png
${OUTPUT_ROOT}/stage2_paper_all/training_metrics.jsonl
```

`training_curves.png` 默认每 100 steps 更新，并在每次验证和训练结束时更新。图中包含 train loss、滑动平均 train loss、dev loss，以及当前验证能够提供的 CER/WER。`training_metrics.jsonl` 按运行追加保存原始指标，每次运行由不同的 `run_id` 区分。

训练日志和指标文件都是追加模式；重复启动不会清空旧日志。checkpoint 默认每 5000 steps 保存一次，并在训练结束时再保存一次，`checkpoints/latest` 指向最近一次 checkpoint。

Stage 1/2 的 CER/WER 均来自自由解码，不再使用真实文本前缀计算。Stage 1 从 BOS 开始生成到 EOS；Stage 2 对每个音频 chunk 生成文本到 `<asr_eos>`。`decode_eos_rate` 应逐步接近 1，Stage 2 的 `decode_truncated_chunk_rate` 应逐步接近 0；否则模型可能尚未学会正确结束生成。

配置项：

```yaml
training:
  resume_from_checkpoint: latest
  plot_every: 100
  plot_smoothing_window: 20
  eval_decode_samples_per_language: 25
  eval_log_examples: 3
```

训练期间的 dev loss 使用完整 dev 集；为控制自回归解码成本，CER/WER 默认每种语言取前 25 条 dev 样本。`plot_every: 0` 可关闭图片生成，原始指标仍会写入 JSONL。判断训练是否真实有效时，不能只看 train loss：train loss 持续下降说明模型能够拟合训练数据；dev loss、CER/WER 同时下降才说明验证集效果在改善。CER/WER 越低越好。

训练结束后可独立评估 checkpoint。省略 `--limit` 时评估完整 manifest：

```bash
python -m soulx_duplug.eval.asr \
  --checkpoint ${OUTPUT_ROOT}/stage1_paper_all/checkpoints/latest \
  --manifest manifests/stage1_paper_all/dev.jsonl \
  --limit 100

python -m soulx_duplug.eval.streaming_asr \
  --checkpoint ${OUTPUT_ROOT}/stage2_paper_all/checkpoints/latest \
  --manifest manifests/stage2_paper_all/dev.jsonl \
  --limit 100
```

## 常用验证命令

```bash
python -m py_compile $(find soulx_duplug tests -name '*.py')
python tests/test_text_manifest.py
python tests/test_audio.py
python tests/test_stage1_smoke.py
python tests/test_stage2_smoke.py
python tests/test_qwen_decoding.py
```

## 重要目录

```text
configs/       训练和数据 profile
datasets/      数据下载脚本
manifests/     训练 manifest
scripts/       模型下载和 smoke test 脚本
soulx_duplug/  训练、数据处理、模型代码
tests/         基础测试
```
