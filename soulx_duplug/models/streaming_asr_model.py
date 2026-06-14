from __future__ import annotations

from dataclasses import dataclass
import os

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class InterleavedBatch:
    audio_token_ids: torch.LongTensor
    text_input_ids: torch.LongTensor
    modality: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor


class DummyInterleavedAsrModel(nn.Module):
    def __init__(self, *, audio_vocab_size: int, text_vocab_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.audio_embedding = nn.Embedding(audio_vocab_size, hidden_size)
        self.projector = nn.Linear(hidden_size, hidden_size)
        self.text_embedding = nn.Embedding(text_vocab_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, text_vocab_size)

    def _embed(self, batch: InterleavedBatch) -> torch.Tensor:
        audio_embeds = self.projector(self.audio_embedding(batch.audio_token_ids.clamp_min(0)))
        text_embeds = self.text_embedding(batch.text_input_ids.clamp_min(0))
        return torch.where(batch.modality.unsqueeze(-1) == 1, audio_embeds, text_embeds)

    def forward(self, batch: InterleavedBatch) -> dict[str, torch.Tensor]:
        embeds = self._embed(batch)
        outputs, _ = self.decoder(embeds)
        logits = self.lm_head(outputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.labels.reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}


class QwenInterleavedAsrModel(nn.Module):
    def __init__(self, *, model_name_or_path: str, audio_vocab_size: int, text_vocab_size: int) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Qwen backend requires transformers to be installed.") from exc
        model_path = os.path.expandvars(os.path.expanduser(model_name_or_path))
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if text_vocab_size > int(self.llm.get_input_embeddings().num_embeddings):
            self.llm.resize_token_embeddings(text_vocab_size)
        hidden_size = int(self.llm.config.hidden_size)
        self.audio_embedding = nn.Embedding(audio_vocab_size, hidden_size)
        self.projector = nn.Linear(hidden_size, hidden_size)

    def _embed(self, batch: InterleavedBatch) -> torch.Tensor:
        audio_embeds = self.projector(self.audio_embedding(batch.audio_token_ids.clamp_min(0)))
        text_embeds = self.llm.get_input_embeddings()(batch.text_input_ids.clamp_min(0))
        return torch.where(batch.modality.unsqueeze(-1) == 1, audio_embeds, text_embeds)

    def forward(self, batch: InterleavedBatch) -> dict[str, torch.Tensor]:
        embeds = self._embed(batch)
        outputs = self.llm(inputs_embeds=embeds, attention_mask=batch.attention_mask)
        logits = outputs.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.labels.reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}


def build_interleaved_asr_model(config: dict, *, audio_vocab_size: int, text_vocab_size: int) -> nn.Module:
    backend = config.get("backend", "dummy")
    if backend == "dummy":
        return DummyInterleavedAsrModel(
            audio_vocab_size=audio_vocab_size,
            text_vocab_size=text_vocab_size,
            hidden_size=int(config.get("hidden_size", 128)),
        )
    if backend == "qwen":
        return QwenInterleavedAsrModel(
            model_name_or_path=str(config["model_name_or_path"]),
            audio_vocab_size=audio_vocab_size,
            text_vocab_size=text_vocab_size,
        )
    raise ValueError(f"unknown model backend: {backend}")
