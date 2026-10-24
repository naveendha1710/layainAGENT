"""Text embedding encoder backed by BAAI/bge-small-en-v1.5.

Uses ``sentence-transformers`` when available; exposes a clean
``EmbeddingEncoder`` interface so the policy store can be tested with a
deterministic fake encoder without downloading a model.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class EmbeddingEncoder:
    """Encode text into dense vectors for FAISS similarity search."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        encoder: Optional[Any] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._encoder = encoder  # dependency injection: fake encoder for tests
        self._model = None  # lazily built real model
        self.normalize = normalize

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array of embeddings."""
        if self._encoder is not None:
            return np.asarray(self._encoder.encode(list(texts)), dtype=np.float32)
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "sentence-transformers is required to build real embeddings. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        vecs = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vecs, dtype=np.float32)


__all__ = ["EmbeddingEncoder"]
