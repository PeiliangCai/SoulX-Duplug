from __future__ import annotations

import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_EN_KEEP_RE = re.compile(r"[^a-z0-9' ]+")
_PINYIN_RE = re.compile(r"^[a-züv:]+[1-5]?$", re.IGNORECASE)


def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def normalize_chinese_text(text: str) -> str:
    """Normalize Chinese ASR targets into compact character-level text."""

    text = unicodedata.normalize("NFKC", text)
    kept: list[str] = []
    for char in text:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if is_cjk(char) or char.isascii() and char.isalnum():
            kept.append(char.lower())
        elif category.startswith("P"):
            continue
    return "".join(kept)


def normalize_aishell3_text(text: str) -> str:
    """AISHELL-3 content.txt alternates Chinese characters and pinyin tokens."""

    tokens = []
    for token in _SPACE_RE.split(text.strip()):
        if not token or _PINYIN_RE.match(token):
            continue
        tokens.append(token)
    return normalize_chinese_text("".join(tokens))


def normalize_english_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _EN_KEEP_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_text(text: str, lang: str, dataset: str = "") -> str:
    if dataset == "aishell3":
        return normalize_aishell3_text(text)
    if lang == "en":
        return normalize_english_text(text)
    return normalize_chinese_text(text)
