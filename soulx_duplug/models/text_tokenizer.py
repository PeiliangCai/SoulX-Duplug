from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol


class TextTokenizer(Protocol):
    pad_id: int
    bos_id: int
    eos_id: int
    vocab_size: int

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...

    def save(self, path: str | Path) -> None:
        ...

    def ensure_token(self, token: str) -> int:
        ...


class CharTokenizer:
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        if token_to_id is None:
            token_to_id = {
                self.pad_token: 0,
                self.bos_token: 1,
                self.eos_token: 2,
                self.unk_token: 3,
            }
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for char in text:
                if char not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[char] = idx
                    self.id_to_token[idx] = char

    def ensure_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        chars: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            if token == self.unk_token:
                continue
            if token.startswith("<") and token.endswith(">"):
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"type": "char", "token_to_id": self.token_to_id}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "token_to_id" in data:
            data = data["token_to_id"]
        return cls({str(token): int(idx) for token, idx in data.items()})


class HfTextTokenizer:
    def __init__(self, model_name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Qwen text tokenizer requires transformers to be installed.") from exc
        model_path = os.path.expandvars(os.path.expanduser(model_name_or_path))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def pad_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def bos_id(self) -> int:
        token_id = self.tokenizer.bos_token_id
        return int(token_id if token_id is not None else self.eos_id)

    @property
    def eos_id(self) -> int:
        token_id = self.tokenizer.eos_token_id
        if token_id is None:
            raise RuntimeError("HF tokenizer has no eos_token_id")
        return int(token_id)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return [int(item) for item in ids]

    def ensure_token(self, token: str) -> int:
        if token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        return int(self.tokenizer.convert_tokens_to_ids(token))

    def decode(self, ids: list[int]) -> str:
        filtered = [int(idx) for idx in ids if int(idx) not in {self.pad_id, self.bos_id, self.eos_id}]
        return self.tokenizer.decode(filtered, skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        save_dir = path.parent / "hf_tokenizer"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        payload = {"type": "hf", "path": save_dir.name}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_text_tokenizer(path: str | Path) -> TextTokenizer:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_type = data.get("type", "char")
    if tokenizer_type == "char":
        return CharTokenizer.load(path)
    if tokenizer_type == "hf":
        return HfTextTokenizer(str(path.parent / data["path"]))
    raise ValueError(f"unknown text tokenizer type: {tokenizer_type}")
