from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Protocol

import numpy as np
import torch


class SpeechTokenizerBackend(Protocol):
    vocab_size: int

    def encode(self, waveform: np.ndarray, sample_rate: int) -> torch.LongTensor:
        ...

    def encode_streaming_chunks(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        chunk_seconds: float = 0.16,
        lookback_seconds: float = 0.96,
        lookahead_seconds: float = 0.04,
    ) -> list[torch.LongTensor]:
        ...


@dataclass
class DummySpeechTokenizerBackend:
    """Deterministic local tokenizer used for smoke tests.

    It preserves the paper's 12.5 Hz token rate convention but does not attempt
    to reproduce GLM-4-Voice acoustic units.
    """

    vocab_size: int = 256
    token_rate_hz: float = 12.5

    def encode(self, waveform: np.ndarray, sample_rate: int) -> torch.LongTensor:
        samples_per_token = max(1, int(round(sample_rate / self.token_rate_hz)))
        if waveform.size == 0:
            return torch.zeros(1, dtype=torch.long)
        pad = (-len(waveform)) % samples_per_token
        if pad:
            waveform = np.pad(waveform, (0, pad))
        frames = waveform.reshape(-1, samples_per_token)
        energy = np.clip(np.mean(np.abs(frames), axis=1), 0.0, 1.0)
        signed_mean = np.mean(frames, axis=1)
        buckets = np.floor(energy * (self.vocab_size // 2 - 1)).astype(np.int64)
        tokens = buckets + (signed_mean > 0).astype(np.int64) * (self.vocab_size // 2)
        return torch.from_numpy(tokens.astype(np.int64))

    def encode_streaming_chunks(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        chunk_seconds: float = 0.16,
        lookback_seconds: float = 0.96,
        lookahead_seconds: float = 0.04,
    ) -> list[torch.LongTensor]:
        chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
        num_chunks = max(1, int(np.ceil(len(waveform) / chunk_samples)))
        full_tokens = self.encode(waveform, sample_rate)
        tokens_per_chunk = max(1, int(round(self.token_rate_hz * chunk_seconds)))
        chunks = []
        for idx in range(num_chunks):
            start = idx * tokens_per_chunk
            end = start + tokens_per_chunk
            chunk_tokens = full_tokens[start:end]
            if chunk_tokens.numel() < tokens_per_chunk:
                pad = torch.zeros(tokens_per_chunk - chunk_tokens.numel(), dtype=torch.long)
                chunk_tokens = torch.cat([chunk_tokens, pad], dim=0)
            chunks.append(chunk_tokens)
        return chunks


class Glm4VoiceTokenizerBackend:
    """Frozen GLM-4-Voice tokenizer adapter.

    The official tokenizer is distributed as model/code rather than a stable
    pip API. This adapter supports two common layouts:
    - a Hugging Face compatible checkpoint with `trust_remote_code=True`
    - a cloned GLM-4-Voice repo passed as `code_path`, containing
      `speech_tokenizer/modeling_whisper.py`
    """

    def __init__(
        self,
        model_path: str,
        vocab_size: int | None = None,
        *,
        code_path: str | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        try:
            from transformers import WhisperFeatureExtractor  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GLM-4-Voice tokenizer backend requires the official tokenizer code "
                "and transformers. Use tokenizer.backend=dummy for smoke tests."
            ) from exc
        if code_path:
            sys.path.insert(0, str(Path(code_path).resolve()))
        try:
            from speech_tokenizer.modeling_whisper import WhisperVQEncoder  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GLM-4-Voice tokenizer backend requires code_path to point to the "
                "official GLM-4-Voice repository containing speech_tokenizer/."
            ) from exc
        self.model_path = model_path
        self.vocab_size = vocab_size or 16384
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.bfloat16)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        self.model = WhisperVQEncoder.from_pretrained(model_path).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, waveform: np.ndarray, sample_rate: int) -> torch.LongTensor:
        pooling_kernel_size = int(getattr(self.model.config, "pooling_kernel_size", None) or 1)
        stride = (
            int(self.model.conv1.stride[0])
            * int(self.model.conv2.stride[0])
            * pooling_kernel_size
            * int(self.feature_extractor.hop_length)
        )
        inputs = self.feature_extractor(
            [waveform.astype(np.float32)],
            sampling_rate=sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=stride,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.device.type == "cuda":
            inputs = {
                key: value.to(dtype=self.dtype) if value.is_floating_point() else value
                for key, value in inputs.items()
            }
        output = self.model(**inputs, return_dict=True)
        token_ids = getattr(output, "quantized_token_ids", None)
        if token_ids is None and isinstance(output, dict):
            token_ids = output.get("quantized_token_ids")
        if token_ids is None:
            raise RuntimeError(
                "GLM-4-Voice tokenizer output did not expose quantized_token_ids. "
                "Check the official tokenizer code/model version."
            )
        attention_mask = inputs["attention_mask"][:, :: int(self.model.conv1.stride[0]) * int(self.model.conv2.stride[0])]
        attention_mask = attention_mask[:, ::pooling_kernel_size]
        token_ids = token_ids[0][attention_mask[0].bool()].detach().to("cpu", dtype=torch.long)
        return token_ids

    def encode_streaming_chunks(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        chunk_seconds: float = 0.16,
        lookback_seconds: float = 0.96,
        lookahead_seconds: float = 0.04,
    ) -> list[torch.LongTensor]:
        chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
        num_chunks = max(1, int(np.ceil(len(waveform) / chunk_samples)))
        chunks: list[torch.LongTensor] = []
        for idx in range(num_chunks):
            target_start = idx * chunk_samples
            target_end = min(len(waveform), target_start + chunk_samples)
            window_start = max(0, target_start - int(round(lookback_seconds * sample_rate)))
            window_end = min(len(waveform), target_end + int(round(lookahead_seconds * sample_rate)))
            window = waveform[window_start:window_end]
            tokens = self.encode(window, sample_rate)
            if tokens.numel() >= 3:
                target_tokens = tokens[-3:-1]
            elif tokens.numel() >= 2:
                target_tokens = tokens[-2:]
            elif tokens.numel() == 1:
                target_tokens = torch.cat([tokens, torch.zeros(1, dtype=torch.long)])
            else:
                target_tokens = torch.zeros(2, dtype=torch.long)
            chunks.append(target_tokens.to(dtype=torch.long))
        return chunks


def build_speech_tokenizer(config: dict) -> SpeechTokenizerBackend:
    backend = config.get("backend", "dummy")
    if backend == "dummy":
        return DummySpeechTokenizerBackend(
            vocab_size=int(config.get("vocab_size", 256)),
            token_rate_hz=float(config.get("token_rate_hz", 12.5)),
        )
    if backend == "glm4voice":
        return Glm4VoiceTokenizerBackend(
            model_path=os.path.expandvars(os.path.expanduser(str(config.get("model_path", "")))),
            vocab_size=config.get("vocab_size"),
            code_path=os.path.expandvars(os.path.expanduser(str(config["code_path"]))) if config.get("code_path") else None,
            device=config.get("device"),
            dtype=str(config.get("dtype", "bfloat16")),
        )
    raise ValueError(f"unknown speech tokenizer backend: {backend}")
