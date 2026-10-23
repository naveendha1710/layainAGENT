"""Read-only in-memory policy store over embedded rules.

Built once at startup with BAAI/bge-small-en-v1.5 embeddings over the 40 domain
rules. Once :meth:`freeze` is called, all mutation methods raise
``StoreFrozenError`` to guarantee the policy store cannot leak test-set
information during evaluation.

A real FAISS ``IndexFlatIP`` is populated and used for inner-product
(cosine similarity with L2-normalized vectors). When FAISS is unavailable, the
store falls back to exact cosine similarity over NumPy vectors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from laya_agent.agent.retrieval.embed import EmbeddingEncoder


class StoreFrozenError(RuntimeError):
    """Raised when mutating a frozen (read-only) policy store."""


@dataclass(frozen=True)
class PolicyEntry:
    rule_id: str
    text: str
    suite: str
    domain: str


@dataclass
class PolicyStore:
    """In-memory policy store, read-only after ``freeze()``."""

    entries: List[PolicyEntry] = field(default_factory=list)
    _encoder: Optional[Any] = None
    _index: Optional[Any] = None  # FAISS IndexFlatIP when available
    _vectors: Optional[np.ndarray] = None
    _frozen: bool = False

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[PolicyEntry],
        encoder: Optional[Any] = None,
        index: Optional[Any] = None,
        build_index: bool = True,
    ) -> "PolicyStore":
        store = cls(
            entries=list(entries),
            _encoder=encoder,
            _index=index,
        )
        if build_index and entries and encoder is not None:
            store.build_index()
        return store

    def freeze(self) -> None:
        """Lock the store read-only; subsequent mutations raise StoreFrozenError."""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------------------
    # mutation (rejected once frozen)
    # ------------------------------------------------------------------
    def add_entry(self, entry: PolicyEntry) -> None:
        if self._frozen:
            raise StoreFrozenError(
                "Cannot add entries to a frozen policy store; "
                "freeze() must be called only after the store is final."
            )
        self.entries.append(entry)
        self._invalidate_index()

    def _invalidate_index(self) -> None:
        self._index = None
        self._vectors = None

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """Populate vector embeddings and the FAISS IndexFlatIP index."""
        if not self.entries or self._encoder is None:
            return

        self._vectors = np.vstack(
            [self._encoder.encode([entry.text]) for entry in self.entries]
        ).astype(np.float32)

        try:
            import faiss  # type: ignore

            dim = int(self._vectors.shape[1])
            index = faiss.IndexFlatIP(dim)
            vs = self._vectors.copy().astype(np.float32)
            faiss.normalize_L2(vs)
            index.add(vs)
            self._index = index
        except ImportError:
            self._index = None

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[Tuple[PolicyEntry, float]]:
        """Return top-k ``(entry, similarity)`` pairs by embedding similarity."""
        if not self.entries:
            return []
        if self._encoder is None:
            raise RuntimeError(
                "PolicyStore has no embedding encoder configured; "
                "construct via from_entries(encoder=...)."
            )

        encoded = self._encoder.encode([query])
        query_vec = np.asarray(encoded[0], dtype=np.float32)

        # Rebuild vector matrix and index if entry set changed or unindexed
        if self._vectors is None or len(self._vectors) != len(self.entries):
            self.build_index()

        # Query vector converted to a 2D float32 NumPy array: (1, dim)
        q = query_vec.reshape(1, -1).astype(np.float32)

        # Use FAISS index when available
        index = self._index
        if index is not None:
            try:
                import faiss  # type: ignore

                if isinstance(index, faiss.Index):
                    # Normalize query vector for cosine similarity via inner product
                    faiss.normalize_L2(q)
                    # Pass single query vector q (1, dim) rather than entry matrix vs
                    scores, ids = index.search(q, top_k)
                    results: List[Tuple[PolicyEntry, float]] = []
                    for score, idx in zip(scores[0], ids[0]):
                        if idx == -1:
                            continue
                        results.append((self.entries[int(idx)], float(score)))
                    return results[:top_k]
            except ImportError:
                pass  # Fall through to NumPy cosine similarity

        # Fallback: exact cosine similarity over NumPy vectors
        if self._vectors is None:
            self._vectors = np.vstack(
                [self._encoder.encode([entry.text]) for entry in self.entries]
            ).astype(np.float32)

        norms = np.linalg.norm(self._vectors, axis=1) + 1e-12
        qnorm = np.linalg.norm(query_vec) + 1e-12
        sims = (self._vectors @ query_vec) / (norms * qnorm)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.entries[int(i)], float(sims[i])) for i in top_idx]

    def top_k_texts(self, query: str, top_k: int = 10) -> List[str]:
        """Return the raw text chunks of the top-k results."""
        return [entry.text for entry, _ in self.search(query, top_k=top_k)]


def load_domain_rules(path: Optional[str | Path] = None) -> List[PolicyEntry]:
    """Load the 40 frozen domain rules from policies.yaml."""
    candidates = [
        Path(path) if path else None,
        Path("policies/policies.yaml"),
        Path("laya_agent/policies/policies.yaml"),
        Path(__file__).resolve().parent.parent.parent / "policies" / "policies.yaml",
        Path(__file__).resolve().parent.parent.parent.parent / "policies" / "policies.yaml",
    ]

    target: Optional[Path] = None
    for cand in candidates:
        if cand is not None and cand.is_file():
            target = cand
            break

    if target is None:
        raise FileNotFoundError(
            "Could not locate policies/policies.yaml. Ensure 40 frozen rules are authored."
        )

    with open(target, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules_data = data.get("rules", [])
    entries = [
        PolicyEntry(
            rule_id=r["rule_id"],
            text=r["text"],
            suite=r["suite"],
            domain=r.get("domain", "general"),
        )
        for r in rules_data
    ]
    return entries


def load_default_policy_store(
    policies_path: Optional[str | Path] = None,
    encoder: Optional[Any] = None,
) -> PolicyStore:
    """Load the 40 frozen domain rules, populate the FAISS index, and lock read-only."""
    entries = load_domain_rules(policies_path)
    if encoder is None:
        encoder = EmbeddingEncoder()

    store = PolicyStore.from_entries(entries=entries, encoder=encoder, build_index=True)
    store.freeze()
    return store


__all__ = [
    "PolicyEntry",
    "PolicyStore",
    "StoreFrozenError",
    "load_default_policy_store",
    "load_domain_rules",
]
