from __future__ import annotations

from dataclasses import dataclass
import os

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Stage1Batch:
    audio_tokens: torch.LongTensor
    audio_lengths: torch.LongTensor
    decoder_input_ids: torch.LongTensor
    labels: torch.LongTensor


class DummyStage1AsrModel(nn.Module):
    """Small conditional decoder for local Stage 1 smoke tests."""

    def __init__(self, *, audio_vocab_size: int, text_vocab_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.audio_embedding = nn.Embedding(audio_vocab_size, hidden_size)
        self.projector = nn.Linear(hidden_size, hidden_size)
        self.text_embedding = nn.Embedding(text_vocab_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, text_vocab_size)

    def _audio_context(self, audio_tokens: torch.LongTensor, audio_lengths: torch.LongTensor) -> torch.Tensor:
        embedded = self.audio_embedding(audio_tokens)
        mask = torch.arange(audio_tokens.shape[1], device=audio_tokens.device)[None, :] < audio_lengths[:, None]
        embedded = embedded * mask.unsqueeze(-1)
        denom = audio_lengths.clamp_min(1).to(embedded.dtype).unsqueeze(-1)
        return self.projector(embedded.sum(dim=1) / denom)

    def forward(
        self,
        *,
        audio_tokens: torch.LongTensor,
        audio_lengths: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self._audio_context(audio_tokens, audio_lengths)
        decoder_inputs = self.text_embedding(decoder_input_ids) + context.unsqueeze(1)
        outputs, _ = self.decoder(decoder_inputs)
        logits = self.lm_head(outputs)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return result

    @torch.no_grad()
    def generate(
        self,
        *,
        audio_tokens: torch.LongTensor,
        audio_lengths: torch.LongTensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        self.eval()
        batch_size = audio_tokens.shape[0]
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=audio_tokens.device)
        for _ in range(max_new_tokens):
            logits = self(
                audio_tokens=audio_tokens,
                audio_lengths=audio_lengths,
                decoder_input_ids=generated,
                labels=None,
            )["logits"]
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == eos_id):
                break
        return generated


class QwenStage1AsrModel(nn.Module):
    """Qwen3 causal LM wrapper for the paper-like Stage 1 path."""

    def __init__(self, *, model_name_or_path: str, audio_vocab_size: int, text_vocab_size: int | None = None) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Qwen backend requires transformers to be installed.") from exc
        model_path = os.path.expandvars(os.path.expanduser(model_name_or_path))
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if text_vocab_size is not None and text_vocab_size > int(self.llm.get_input_embeddings().num_embeddings):
            self.llm.resize_token_embeddings(text_vocab_size)
        hidden_size = int(self.llm.config.hidden_size)
        self.audio_embedding = nn.Embedding(audio_vocab_size, hidden_size)
        self.projector = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        *,
        audio_tokens: torch.LongTensor,
        audio_lengths: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        audio_embeds = self.projector(self.audio_embedding(audio_tokens))
        text_embeds = self.llm.get_input_embeddings()(decoder_input_ids)
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        audio_mask = (
            torch.arange(audio_tokens.shape[1], device=audio_tokens.device)[None, :]
            < audio_lengths[:, None]
        ).long()
        if labels is not None:
            prefix_labels = torch.full(
                audio_tokens.shape,
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            text_mask = (labels != -100).long()
            labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            text_mask = torch.ones(decoder_input_ids.shape, dtype=torch.long, device=decoder_input_ids.device)
        attention_mask = torch.cat([audio_mask, text_mask], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        result = {"logits": outputs.logits}
        if labels is not None:
            text_logits = outputs.logits[:, -decoder_input_ids.shape[1] :, :]
            result["loss"] = F.cross_entropy(
                text_logits.reshape(-1, text_logits.shape[-1]),
                labels[:, -decoder_input_ids.shape[1] :].reshape(-1),
                ignore_index=-100,
            )
        return result

    @torch.no_grad()
    def generate(
        self,
        *,
        audio_tokens: torch.LongTensor,
        audio_lengths: torch.LongTensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        self.eval()
        batch_size, max_audio_tokens = audio_tokens.shape
        device = audio_tokens.device
        audio_embeds = self.projector(self.audio_embedding(audio_tokens))
        bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        bos_embeds = self.llm.get_input_embeddings()(bos_tokens)
        inputs_embeds = torch.cat([audio_embeds, bos_embeds], dim=1)

        audio_mask = (
            torch.arange(max_audio_tokens, device=device)[None, :]
            < audio_lengths[:, None]
        ).long()
        attention_mask = torch.cat(
            [audio_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
            dim=1,
        )
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1].argmax(dim=-1)
        generated = [bos_tokens]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, eos_id),
                next_tokens,
            )
            generated.append(next_tokens[:, None])
            finished |= next_tokens.eq(eos_id)
            if bool(finished.all()):
                break

            next_embeds = self.llm.get_input_embeddings()(next_tokens[:, None])
            next_position_ids = attention_mask.sum(dim=-1, keepdim=True)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device),
                ],
                dim=1,
            )
            outputs = self.llm(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_tokens = outputs.logits[:, -1].argmax(dim=-1)

        return torch.cat(generated, dim=1)


def build_stage1_model(config: dict, *, audio_vocab_size: int, text_vocab_size: int) -> nn.Module:
    backend = config.get("backend", "dummy")
    if backend == "dummy":
        return DummyStage1AsrModel(
            audio_vocab_size=audio_vocab_size,
            text_vocab_size=text_vocab_size,
            hidden_size=int(config.get("hidden_size", 128)),
        )
    if backend == "qwen":
        return QwenStage1AsrModel(
            model_name_or_path=str(config["model_name_or_path"]),
            audio_vocab_size=audio_vocab_size,
            text_vocab_size=text_vocab_size,
        )
    raise ValueError(f"unknown model backend: {backend}")
