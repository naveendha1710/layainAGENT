"""Deterministic mock provider for offline CI and unit tests."""

from __future__ import annotations

from typing import Any, Dict, Optional

from laya_agent.agent.gates.runtime import LayaRuntime
from laya_agent.agent.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Offline deterministic provider utilizing LayaRuntime heuristics."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.runtime = LayaRuntime(device=device, load_weights=False)

    def predict(
        self,
        text: Any,
        schema: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict using deterministic heuristic evaluations."""
        if isinstance(text, dict):
            merged_ctx = dict(text)
            if ctx:
                merged_ctx.update(ctx)
            text_str = str(merged_ctx.get("recent_observation", "") or merged_ctx.get("user_prompt", "") or text)
            return self.runtime.predict(text=text_str, schema=schema, ctx=merged_ctx)

        text_str = text if isinstance(text, str) else str(text)
        return self.runtime.predict(text=text_str, schema=schema, ctx=ctx)

