"""Real LAYA Provider wrapping laya.Router with GPU synchronization.

Evaluates dynamic schemas in a single non-autoregressive forward pass (~35 ms)
on CUDA using ModernBERT-large English checkpoint.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional

from laya_agent.agent.gates.filter import FILTER_SCHEMA
from laya_agent.agent.gates.firewall import FIREWALL_SCHEMA
from laya_agent.agent.gates.router import ROUTER_SCHEMA
from laya_agent.agent.gates.scorer import SCORER_SCHEMA
from laya_agent.agent.gates.supervisor import SUPERVISOR_SCHEMA
from laya_agent.agent.providers.base import BaseProvider
from laya_agent.agent.providers.mock import MockProvider

try:
    from laya import Router  # type: ignore
except ImportError:
    Router = None

# Cache single shared Router instance per device to avoid duplicate model weight allocations
_ROUTER_CACHE: Dict[str, Any] = {}


def get_router(device: str) -> Any:
    """Retrieve or initialize cached laya.Router instance."""
    global _ROUTER_CACHE
    dev_key = device.lower()
    if dev_key not in _ROUTER_CACHE:
        if Router is None:
            from laya import Router as ImportedRouter  # type: ignore
            _ROUTER_CACHE[dev_key] = ImportedRouter(device=device)
        else:
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
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
        except ImportError:
            cuda_avail = False

        if device is None:
            device = "cuda" if cuda_avail else "cpu"
        self.device = str(device).lower()

        is_cuda = (self.device == "cuda" and cuda_avail)
        if load_weights is None:
            load_weights = is_cuda
        self.load_weights = load_weights
        self.fallback_to_mock = fallback_to_mock
        self._mock_provider: Optional[MockProvider] = None
        self.router = router
        self.gate_latency_ms: float = 0.0

        if self.router is None:
            self._init_router()

    def _init_router(self) -> None:
        """Attempt to instantiate or retrieve laya.Router with fallback to mock."""
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
        except ImportError:
            cuda_avail = False

        is_cuda = (self.device == "cuda" and cuda_avail)
        try:
            self.router = get_router(self.device)
        except (ImportError, ModuleNotFoundError) as exc:
            if is_cuda or self.load_weights or not self.fallback_to_mock:
                raise RuntimeError(
                    "The 'laya' package from NandhaKishorM/laya is required on CUDA or when load_weights=True."
                ) from exc
            warnings.warn("The 'laya' package is not installed; falling back to MockProvider.")
            self._mock_provider = MockProvider(device=self.device, load_weights=self.load_weights)
        except Exception as exc:
            if is_cuda or self.load_weights or not self.fallback_to_mock:
                raise RuntimeError(
                    f"Failed to initialize laya.Router on {self.device}: {exc}. "
                    "Silent fallback to MockProvider is disabled for CUDA live runs."
                ) from exc
            warnings.warn(f"Failed to initialize laya.Router ({exc}); falling back to MockProvider.")
            self._mock_provider = MockProvider(device=self.device, load_weights=self.load_weights)

    def predict(
        self,
        text: Any = None,
        schema: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute non-autoregressive schema inference with CUDA kernel synchronization."""
        # Support both predict(schema, text) and predict(text, schema)
        if isinstance(text, dict) and isinstance(schema, str):
            schema, text = text, schema
        elif schema is None and isinstance(text, dict) and any(k in text for k in ("model_tier", "applies", "action", "injection_detected", "status")):
            schema = text
            text = kwargs.get("text", "")
        elif schema is None and "schema" in kwargs:
            schema = kwargs["schema"]
            text = text if text is not None else kwargs.get("text", "")

        ctx = dict(ctx) if ctx is not None else {}
        if isinstance(text, dict):
            merged_ctx = dict(text)
            merged_ctx.update(ctx)
            ctx = merged_ctx
            text_str = str(ctx.get("recent_observation", "") or ctx.get("user_prompt", "") or text)
        else:
            text_str = "" if text is None else str(text)

        if self.router is None:
            if self._mock_provider is not None:
                t0 = time.perf_counter()
                res = self._mock_provider.predict(text=text_str, schema=schema, ctx=ctx)
                self.gate_latency_ms = (time.perf_counter() - t0) * 1000.0
                if isinstance(res, dict):
                    res["latency_ms"] = self.gate_latency_ms
                return res
            raise RuntimeError("laya.Router is not initialized and fallback_to_mock is False.")

        if self.device == "cuda":
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
        t0 = time.perf_counter()

        try:
            if hasattr(self.router, "predict"):
                try:
                    result = self.router.predict(schema=schema, text=text_str)
                except TypeError:
                    result = self.router.predict(text_str, schema=schema)
            elif callable(self.router):
                try:
                    result = self.router(schema=schema, text=text_str)
                except TypeError:
                    result = self.router(text_str, schema=schema)
            else:
                result = {}
        finally:
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.synchronize()
                except Exception:
                    pass
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.gate_latency_ms = latency_ms

        if isinstance(result, dict):
            result["latency_ms"] = self.gate_latency_ms
            return result
        return {"latency_ms": self.gate_latency_ms}


__all__ = [
    "FILTER_SCHEMA",
    "FIREWALL_SCHEMA",
    "LayaProvider",
    "ROUTER_SCHEMA",
    "SCORER_SCHEMA",
    "SUPERVISOR_SCHEMA",
    "get_router",
]
