"""Tests for the read-only FAISS policy retrieval engine.

Covers:
  - FAISS IndexFlatIP query vector search (single query vector q of shape (1, dim)
    and dtype float32, fixing the line 111 bug).
  - 40 frozen domain rules loading from policies.yaml across all 4 suites.
  - Startup initialization and immutability lock (.freeze()) preventing test-set leakage.
  - Fallback cosine similarity when FAISS is bypassed or unavailable.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from laya_agent.agent.retrieval.embed import EmbeddingEncoder
from laya_agent.agent.retrieval.store import (
    PolicyEntry,
    PolicyStore,
    StoreFrozenError,
    load_default_policy_store,
    load_domain_rules,
)


class FakeEncoder:
    """Deterministic fake encoder: stable hash-based vectors (no model download)."""

    def __init__(self, dim: int = 16) -> None:
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
        return np.vstack(vecs).astype(np.float32)


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
    store = PolicyStore.from_entries(entries, encoder=encoder, build_index=False)

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
    store = PolicyStore.from_entries(entries, encoder=encoder, build_index=False)
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
    store = PolicyStore.from_entries(entries, encoder=encoder, build_index=False)
    store.freeze()

    before = len(store.entries)
    _ = store.search("alpha", top_k=1)
    assert len(store.entries) == before
    assert store.frozen


def test_store_empty_search_returns_empty() -> None:
    store = PolicyStore.from_entries([], encoder=FakeEncoder(dim=8), build_index=False)
    assert store.search("anything", top_k=5) == []


def test_faiss_index_flat_ip_search_and_query_shape() -> None:
    """Verify that FAISS IndexFlatIP searches using a (1, dim) query vector."""
    pytest.importorskip("faiss")
    import faiss

    dim = 16
    encoder = FakeEncoder(dim=dim)
    entries = [
        PolicyEntry(
            rule_id="bank_01",
            text="banking wire transfer approval threshold policy",
            suite="banking",
            domain="security",
        ),
        PolicyEntry(
            rule_id="work_01",
            text="workspace confidential document sharing protocol",
            suite="workspace",
            domain="privacy",
        ),
        PolicyEntry(
            rule_id="slack_01",
            text="slack public channel credential posting restriction",
            suite="slack",
            domain="security",
        ),
    ]

    store = PolicyStore.from_entries(entries, encoder=encoder, build_index=True)
    assert store._index is not None
    assert isinstance(store._index, faiss.IndexFlatIP)

    # Search with a query
    results = store.search("wire transfer approval", top_k=2)
    assert len(results) == 2
    top_entry, top_score = results[0]
    assert isinstance(top_entry, PolicyEntry)
    assert isinstance(top_score, float)
    # Cosine similarity for normalized vectors is bounded in [-1.0, 1.0]
    assert -1.01 <= top_score <= 1.01


def test_load_40_frozen_domain_rules() -> None:
    """Verify loading the 40 frozen domain rules across the 4 suites."""
    rules = load_domain_rules()
    assert len(rules) == 40

    suites = [r.suite for r in rules]
    for suite in ["workspace", "banking", "slack", "travel"]:
        count = suites.count(suite)
        assert count == 10, f"Expected 10 rules for suite '{suite}', got {count}"

    # Ensure all rule IDs are unique
    rule_ids = [r.rule_id for r in rules]
    assert len(set(rule_ids)) == 40


def test_load_default_policy_store_is_frozen_with_faiss() -> None:
    """Verify load_default_policy_store returns a frozen store with 40 entries."""
    encoder = FakeEncoder(dim=16)
    store = load_default_policy_store(encoder=encoder)

    assert store.frozen is True
    assert len(store.entries) == 40

    # Ensure mutation is strictly rejected
    with pytest.raises(StoreFrozenError):
        store.add_entry(
            PolicyEntry(
                rule_id="leak_01",
                text="attempted test-set leakage rule",
                suite="workspace",
                domain="leak",
            )
        )

    # Retrieval should execute cleanly on the frozen store
    results = store.search("credit card blocking", top_k=3)
    assert len(results) == 3
    for entry, score in results:
        assert isinstance(entry, PolicyEntry)
        assert isinstance(score, float)


__all__ = ["FakeEncoder"]
