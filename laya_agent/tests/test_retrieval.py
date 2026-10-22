"""Tests for the read-only FAISS policy retrieval engine.

Covers embedding generation and post-freeze immutability (no test-set leakage).
No model download is required: a deterministic fake encoder is injected.
"""

from __future__ import annotations

import numpy as np
import pytest

from laya_agent.agent.retrieval.embed import EmbeddingEncoder
from laya_agent.agent.retrieval.store import (
    PolicyEntry,
    PolicyStore,
    StoreFrozenError,
)


class FakeEncoder:
    """Deterministic fake encoder: stable hash-based vectors (no model download)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts):  # noqa: ANN001
        self.calls.append(list(texts))
        vecs = []
        for t in texts:
            h = abs(hash(t)) % 1_000_000
            vec = np.array(
                [(h * (i + 1)) % 97 / 96.0 - 0.5 for i in range(self.dim)],
                dtype=np.float32,
            )
            vecs.append(vec)
        return np.vstack(vecs)


def test_embedding_generation_deterministic() -> None:
    encoder = FakeEncoder(dim=8)
    a = encoder.encode(["hello world", "goodbye world"])
    b = encoder.encode(["hello world", "goodbye world"])
    assert a.shape == (2, 8)
    assert a.dtype == np.float32
    assert np.allclose(a, b)


def test_store_top_k_returns_raw_text_chunks() -> None:
    encoder = FakeEncoder(dim=8)
    entries = [
        PolicyEntry(
            rule_id="r1",
            text="banking: never transfer funds without approval",
            suite="banking",
            domain="security",
        ),
        PolicyEntry(
            rule_id="r2",
            text="workspace: allow read-only file access",
            suite="workspace",
            domain="access",
        ),
        PolicyEntry(
            rule_id="r3",
            text="travel: flight booking requires manager sign-off",
            suite="travel",
            domain="security",
        ),
    ]
    store = PolicyStore.from_entries(entries, encoder=encoder)

    results = store.search("banking fund transfer approval", top_k=2)
    assert len(results) == 2
    assert all(
        isinstance(entry, PolicyEntry) and isinstance(score, float)
        for entry, score in results
    )

    texts = store.top_k_texts("workspace file access", top_k=5)
    assert len(texts) == 3  # all entries returned when top_k >= len(entries)


def test_store_freeze_prevents_mutation() -> None:
    encoder = FakeEncoder(dim=8)
    entries = [
        PolicyEntry(rule_id="r1", text="do X", suite="banking", domain="security")
    ]
    store = PolicyStore.from_entries(entries, encoder=encoder)
    assert not store.frozen

    store.freeze()
    assert store.frozen

    with pytest.raises(StoreFrozenError):
        store.add_entry(
            PolicyEntry(rule_id="r2", text="do Y", suite="workspace", domain="access")
        )


def test_store_immutable_during_run() -> None:
    """Simulate a run: freeze before search, confirm no mutation mid-run."""
    encoder = FakeEncoder(dim=8)
    entries = [
        PolicyEntry(rule_id="r1", text="alpha rule", suite="banking", domain="security")
    ]
    store = PolicyStore.from_entries(entries, encoder=encoder)
    store.freeze()

    before = len(store.entries)
    _ = store.search("alpha", top_k=1)
    assert len(store.entries) == before
    assert store.frozen


def test_store_empty_search_returns_empty() -> None:
    store = PolicyStore.from_entries([], encoder=FakeEncoder(dim=8))
    assert store.search("anything", top_k=5) == []


__all__ = ["FakeEncoder"]
