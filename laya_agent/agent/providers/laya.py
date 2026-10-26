"""Real LAYA Provider wrapping laya.Router with GPU synchronization.

Evaluates dynamic schemas in a single non-autoregressive forward pass (~35 ms)
on CUDA using ModernBERT-large English checkpoint.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from laya_agent.agent.gates.filter import FILTER_SCHEMA
from laya_agent.agent.gates.firewall import FIREWALL_SCHEMA
from laya_agent.agent.gates.router import ROUTER_SCHEMA
from laya_agent.agent.gates.runtime import detect_device, synchronize_device
from laya_agent.agent.gates.scorer import SCORER_SCHEMA
from laya_agent.agent.gates.supervisor import SUPERVISOR_SCHEMA
from laya_agent.agent.providers.base import BaseProvider
from laya_agent.agent.providers.mock import MockProvider

# Cache single shared Router instance per device to avoid duplicate model weight allocations
_ROUTER_CACHE: Dict[str, Any] = {}


def get_router(device: str) -> Any:
    """Retrieve or initialize cached laya.Router instance."""
    global _ROUTER_CACHE
    dev_key = device.lower()
    if dev_key not in _ROUTER_CACHE:
        from laya import Router  # type: ignore

        _ROUTER_CACHE[dev_key] = Router(device=device)
    return _ROUTER_CACHE[dev_key]


class LayaProvider(BaseProvider):
    """Wraps laya.Router from the laya package with GPU timing synchronization."""

    def __init__(
        self,
        device: Optional[str] = None,
        router: Optional[Any] = None,
        fallback_to_mock: bool = True,
        load_weights: Optional[bool] = None,
    ) -> None:
        import torch

        self.device = detect_device(device)
        is_cuda_available = (self.device == "cuda" and torch.cuda.is_available())
        if load_weights is None:
            load_weights = is_cuda_available
        self.load_weights = load_weights
        self.fallback_to_mock = fallback_to_mock
        self._mock_provider: Optional[MockProvider] = None
        self.router = router
        self.gate_latency_ms: float = 0.0

        if self.router is None:
            self._init_router()

    def _init_router(self) -> None:
        """Attempt to instantiate or retrieve laya.Router with fallback to mock."""
        import torch

        is_cuda = (self.device == "cuda" and torch.cuda.is_available())
        try:
            self.router = get_router(self.device)
        except Exception as exc:
            if is_cuda or self.load_weights or not self.fallback_to_mock:
                raise RuntimeError(
                    f"Failed to initialize laya.Router on {self.device}: {exc}. "
                    "Silent fallback to MockProvider is disabled for CUDA live runs."
                ) from exc
            self._mock_provider = MockProvider(device=self.device, load_weights=self.load_weights)

    def predict(
        self,
        text: Any,
        schema: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute non-autoregressive schema inference with CUDA kernel synchronization."""
        ctx = dict(ctx) if ctx is not None else {}
        if isinstance(text, dict):
            merged_ctx = dict(text)
            merged_ctx.update(ctx)
            ctx = merged_ctx
            text = str(ctx.get("recent_observation", "") or ctx.get("user_prompt", "") or text)

        if self.router is None:
            if self._mock_provider is not None:
                t0 = time.perf_counter()
                res = self._mock_provider.predict(text=text, schema=schema, ctx=ctx)
                self.gate_latency_ms = (time.perf_counter() - t0) * 1000.0
                return res
            raise RuntimeError("laya.Router is not initialized and fallback_to_mock is False.")

        # Ensure GPU execution queue is drained prior to timing
        synchronize_device(self.device)
        t0 = time.perf_counter()

        try:
            if hasattr(self.router, "predict"):
                result = self.router.predict(text, schema=schema, ctx=ctx)
            elif callable(self.router):
                result = self.router(text, schema=schema, ctx=ctx)
            else:
                result = {}
        finally:
            # Synchronize CUDA device to record true kernel execution time
            synchronize_device(self.device)
            self.gate_latency_ms = (time.perf_counter() - t0) * 1000.0

        return result if isinstance(result, dict) else {}


__all__ = [
    "FILTER_SCHEMA",
    "FIREWALL_SCHEMA",
    "LayaProvider",
    "ROUTER_SCHEMA",
    "SCORER_SCHEMA",
    "SUPERVISOR_SCHEMA",
    "get_router",
]
