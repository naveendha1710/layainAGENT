"""Read-only in-memory policy store over embedded rules.

Built once at startup with BAAI/bge-small-en-v1.5 embeddings over the 40 domain
rules. Once :meth:`freeze` is called, all mutation methods raise
``StoreFrozenError`` to guarantee the policy store cannot leak test-set
information during evaluation.

A real FAISS ``IndexFlatIP`` can be substituted for ``_index``; when FAISS is
unavailable the store falls back to exact cosine similarity over numpy vectors
(correct and deterministic for the 40-rule store).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


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
    ) -> "PolicyStore":
        return cls(
            entries=list(entries),
            _encoder=encoder,
            _index=index,
        )

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

        query_vec = self._encoder.encode([query])[0]  # (dim,)

        # Rebuild the vector matrix if the entry set changed.
        if self._vectors is None or len(self._vectors) != len(self.entries):
            self._vectors = np.vstack(
                [self._encoder.encode([entry.text]) for entry in self.entries]
            )

        # Use a real FAISS index when one was provided; otherwise exact cosine
        # similarity over numpy (equivalent for a small, static rule set).
        index = self._index
        if index is not None:
            try:
                import faiss  # type: ignore

                if isinstance(index, faiss.Index):
                    q = query_vec.reshape(1, -1).astype(np.float32)
                    faiss.normalize_L2(q)
                    vs = self._vectors.astype(np.float32)
                    faiss.normalize_L2(vs)
                    scores, ids = index.search(vs, top_k)  # type: ignore[attr-defined]
                    # Re-map: index was built over self.entries order.
                    results = []
                    for score, idx in zip(scores[0], ids[0]):
                        if idx == -1:
                            continue
                        results.append((self.entries[int(idx)], float(score)))
                    return results[:top_k]
            except ImportError:
                pass  # fall through to numpy cosine

        # Exact cosine similarity (deterministic, no FAISS required).
        norms = np.linalg.norm(self._vectors, axis=1) + 1e-12
        qnorm = np.linalg.norm(query_vec) + 1e-12
        sims = (self._vectors @ query_vec) / (norms * qnorm)
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.entries[int(i)], float(sims[i])) for i in top_idx]

    def top_k_texts(self, query: str, top_k: int = 10) -> List[str]:
        """Return the raw text chunks of the top-k results."""
        return [entry.text for entry, _ in self.search(query, top_k=top_k)]


__all__ = ["PolicyEntry", "PolicyStore", "StoreFrozenError"]
