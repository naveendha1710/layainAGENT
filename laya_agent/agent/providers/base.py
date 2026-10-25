"""Provider interface for gate evaluation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseProvider(ABC):
    """Abstract base class for gate evaluation providers."""

    @abstractmethod
    def predict(
        self,
        text: Any,
        schema: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate text or context against a structured gate schema."""
        ...
