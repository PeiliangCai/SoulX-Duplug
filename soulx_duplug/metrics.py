from __future__ import annotations

import re
from collections.abc import Sequence


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def cer(ref: str, hyp: str) -> float:
    ref_chars = [c for c in ref if not c.isspace()]
    hyp_chars = [c for c in hyp if not c.isspace()]
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


_WORD_RE = re.compile(r"[a-z0-9']+")


def wer(ref: str, hyp: str) -> float:
    ref_words = _WORD_RE.findall(ref.lower())
    hyp_words = _WORD_RE.findall(hyp.lower())
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)
