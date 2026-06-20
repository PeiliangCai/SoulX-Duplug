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


@dataclass(frozen=True)
class StreamingGeneration:
    token_ids: list[int]
    chunk_count: int
    eos_count: int
    truncated_chunk_count: int


def _audio_chunks_from_batch(batch: InterleavedBatch, row: int) -> list[torch.LongTensor]:
    valid_length = int(batch.attention_mask[row].sum().item())
    modalities = batch.modality[row, :valid_length].tolist()
    audio_ids = batch.audio_token_ids[row, :valid_length]
    chunks: list[torch.LongTensor] = []
    start: int | None = None
    for index, modality in enumerate(modalities):
        if modality == 1 and start is None:
            start = index
        elif modality != 1 and start is not None:
            chunks.append(audio_ids[start:index])
            start = None
    if start is not None:
        chunks.append(audio_ids[start:valid_length])
    return chunks


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

    @torch.no_grad()
    def generate_streaming(
        self,
        batch: InterleavedBatch,
        *,
        bos_id: int,
        asr_eos_id: int,
        max_new_tokens_per_chunk: int,
    ) -> list[StreamingGeneration]:
        self.eval()
        generations = []
        for row in range(batch.audio_token_ids.shape[0]):
            hidden = None
            generated_ids: list[int] = []
            chunks = _audio_chunks_from_batch(batch, row)
            eos_count = 0
            truncated_count = 0
            for audio_chunk in chunks:
                audio_embeds = self.projector(self.audio_embedding(audio_chunk[None, :]))
                bos_token = torch.tensor([[bos_id]], dtype=torch.long, device=audio_chunk.device)
                bos_embed = self.text_embedding(bos_token)
                outputs, hidden = self.decoder(torch.cat([audio_embeds, bos_embed], dim=1), hidden)
                next_token = int(self.lm_head(outputs[:, -1]).argmax(dim=-1).item())
                for _ in range(max_new_tokens_per_chunk):
                    if next_token == asr_eos_id:
                        eos_count += 1
                        eos_token = torch.tensor([[asr_eos_id]], dtype=torch.long, device=audio_chunk.device)
                        _, hidden = self.decoder(self.text_embedding(eos_token), hidden)
                        break
                    generated_ids.append(next_token)
                    token = torch.tensor([[next_token]], dtype=torch.long, device=audio_chunk.device)
                    outputs, hidden = self.decoder(self.text_embedding(token), hidden)
                    next_token = int(self.lm_head(outputs[:, -1]).argmax(dim=-1).item())
                else:
                    truncated_count += 1
            generations.append(
                StreamingGeneration(
                    token_ids=generated_ids,
                    chunk_count=len(chunks),
                    eos_count=eos_count,
                    truncated_chunk_count=truncated_count,
                )
            )
        return generations


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

    @torch.no_grad()
    def generate_streaming(
        self,
        batch: InterleavedBatch,
        *,
        bos_id: int,
        asr_eos_id: int,
        max_new_tokens_per_chunk: int,
    ) -> list[StreamingGeneration]:
        self.eval()
        generations = []
        for row in range(batch.audio_token_ids.shape[0]):
            past_key_values = None
            sequence_length = 0
            generated_ids: list[int] = []
            chunks = _audio_chunks_from_batch(batch, row)
            eos_count = 0
            truncated_count = 0

            for audio_chunk in chunks:
                audio_embeds = self.projector(self.audio_embedding(audio_chunk[None, :]))
                bos_token = torch.tensor([[bos_id]], dtype=torch.long, device=audio_chunk.device)
                bos_embed = self.llm.get_input_embeddings()(bos_token)
                prefix_embeds = torch.cat([audio_embeds, bos_embed], dim=1)
                prefix_length = prefix_embeds.shape[1]
                attention_mask = torch.ones(
                    (1, sequence_length + prefix_length),
                    dtype=torch.long,
                    device=audio_chunk.device,
                )
                position_ids = torch.arange(
                    sequence_length,
                    sequence_length + prefix_length,
                    dtype=torch.long,
                    device=audio_chunk.device,
                )[None, :]
                outputs = self.llm(
                    inputs_embeds=prefix_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                sequence_length += prefix_length
                next_token = int(outputs.logits[:, -1].argmax(dim=-1).item())

                for _ in range(max_new_tokens_per_chunk):
                    if next_token == asr_eos_id:
                        eos_count += 1
                        eos_token = torch.tensor([[asr_eos_id]], dtype=torch.long, device=audio_chunk.device)
                        eos_embed = self.llm.get_input_embeddings()(eos_token)
                        attention_mask = torch.ones(
                            (1, sequence_length + 1),
                            dtype=torch.long,
                            device=audio_chunk.device,
                        )
                        position_ids = torch.tensor(
                            [[sequence_length]],
                            dtype=torch.long,
                            device=audio_chunk.device,
                        )
                        outputs = self.llm(
                            inputs_embeds=eos_embed,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = outputs.past_key_values
                        sequence_length += 1
                        break
                    generated_ids.append(next_token)
                    token = torch.tensor([[next_token]], dtype=torch.long, device=audio_chunk.device)
                    token_embed = self.llm.get_input_embeddings()(token)
                    attention_mask = torch.ones(
                        (1, sequence_length + 1),
                        dtype=torch.long,
                        device=audio_chunk.device,
                    )
                    position_ids = torch.tensor(
                        [[sequence_length]],
                        dtype=torch.long,
                        device=audio_chunk.device,
                    )
                    outputs = self.llm(
                        inputs_embeds=token_embed,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    sequence_length += 1
                    next_token = int(outputs.logits[:, -1].argmax(dim=-1).item())
                else:
                    truncated_count += 1

            generations.append(
                StreamingGeneration(
                    token_ids=generated_ids,
                    chunk_count=len(chunks),
                    eos_count=eos_count,
                    truncated_chunk_count=truncated_count,
                )
            )
        return generations


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
