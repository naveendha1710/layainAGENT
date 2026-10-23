"""Option A 300-token sliding chunker for raw tool execution outputs.

Because the English ModernBERT-large checkpoint has a 512-token context window
(192 tokens reserved for criteria headers and prompt overhead, leaving ~320
tokens for input state), tool outputs exceeding the budget must be chunked
into <= 300-token windows before evaluation by the Context Firewall gate.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional


# Rough token estimation factor: 1 word ~ 1.33 BPE/ModernBERT subword tokens
# (i.e. 0.75 words per token)
WORDS_PER_TOKEN = 0.75


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Estimate or count the number of tokens in ``text``."""
    if not text:
        return 0
    if tokenizer is not None:
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
        if callable(tokenizer):
            return len(tokenizer(text))
    # Heuristic estimation: split by whitespace / punctuation
    words = re.findall(r"\w+|[^\w\s]", text)
    return max(1, int(len(words) / WORDS_PER_TOKEN))


def sliding_chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 0,
    tokenizer: Optional[Any] = None,
) -> List[str]:
    """Split text into sequential or overlapping <= 300-token chunks.

    Parameters
    ----------
    text:
        Raw input string (e.g. tool execution response).
    chunk_size:
        Maximum token capacity per chunk (default: 300).
    overlap:
        Token overlap between consecutive chunks (default: 0).
    tokenizer:
        Optional tokenizer object with an ``encode`` method or callable.
        When None, a fast regex-based word/token partition is used.

    Returns
    -------
    List[str]
        Ordered list of chunk strings. Returns an empty list if ``text`` is empty.
    """
    if not text or not text.strip():
        return []

    # If already within budget, return single chunk immediately
    estimated_tokens = count_tokens(text, tokenizer=tokenizer)
    if estimated_tokens <= chunk_size:
        return [text]

    # Split text while preserving whitespaces and words
    tokens_and_spaces = re.findall(r"\S+|\s+", text)
    # Filter to non-whitespace tokens for counting
    token_indices = [
        i for i, part in enumerate(tokens_and_spaces) if not part.isspace()
    ]

    total_words = len(token_indices)
    # Map token capacity to word budget when using heuristic estimation
    if tokenizer is None:
        max_words_per_chunk = max(1, int(chunk_size * WORDS_PER_TOKEN))
        word_overlap = max(0, int(overlap * WORDS_PER_TOKEN))
    else:
        max_words_per_chunk = max(1, chunk_size)
        word_overlap = max(0, overlap)

    if total_words <= max_words_per_chunk:
        return [text]

    step = max(1, max_words_per_chunk - word_overlap)
    chunks: List[str] = []

    start_tok = 0
    while start_tok < total_words:
        end_tok = min(start_tok + max_words_per_chunk, total_words)
        first_idx = token_indices[start_tok]
        last_idx = token_indices[end_tok - 1]

        chunk_str = "".join(tokens_and_spaces[first_idx : last_idx + 1]).strip()
        if chunk_str:
            chunks.append(chunk_str)

        if end_tok >= total_words:
            break
        start_tok += step

    return chunks


__all__ = ["count_tokens", "sliding_chunk_text"]
