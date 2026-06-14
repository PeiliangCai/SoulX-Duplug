FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_ROOT=/data/datasets \
    MODEL_ROOT=/data/models \
    CACHE_ROOT=/data/cache \
    OUTPUT_ROOT=/data/outputs \
    HF_HOME=/data/cache/huggingface \
    MODELSCOPE_CACHE=/data/cache/modelscope

RUN apt-get update && apt-get install -y --no-install-recommends \
    aria2 \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/SoulX-Duplug

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["/bin/bash"]
