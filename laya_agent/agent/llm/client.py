"""Dual-tier hosted LLM client targeting the NVIDIA Build API.

OpenAI-compatible chat-completion client
(``https://integrate.api.nvidia.com/v1``) with exponential-backoff-plus-jitter
retry on HTTP 429/5xx, exact input/output token counting, and per-call latency
tracking (``llm_latency_ms``).

The ``openai`` SDK is imported lazily so this module imports cleanly in
environments where the hosted-LLM dependencies are not yet installed.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


class LLMError(RuntimeError):
    """Base exception for LLM client failures."""


class LLMRateLimitError(LLMError):
    """HTTP 429 rate-limit responses after retries are exhausted."""


class LLMProviderError(LLMError):
    """HTTP 5xx or other provider failures after retries are exhausted."""


@dataclass(frozen=True)
class LLMResponse:
    """Result of a single LLM chat completion."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def _import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LLMError(
            "The 'openai' package is required to call the NVIDIA Build API. "
            "Install it with: pip install openai>=1.0"
        ) from exc
    return OpenAI


def _is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a transient 429/5xx provider error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        return status == 429 or 500 <= status < 600
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or any(code in msg for code in ("500", "502", "503", "504"))
    )


def _extract_usage(response: Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an OpenAI response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    return (
        int(getattr(usage, "prompt_tokens", 0)),
        int(getattr(usage, "completion_tokens", 0)),
    )


class NVIDIABuildClient:
    """OpenAI-compatible chat-completion client for NVIDIA Build API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nemotron-3.5-super-vl-120b-a12b",
        small_model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        initial_backoff_ms: int = 250,
        max_backoff_ms: int = 30000,
        client: Optional[Any] = None,
    ) -> None:
        # Resolve API key from argument, environment, or .env file
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        if not resolved_key:
            # Check local .env files
            for p in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent.parent / ".env"):
                if p.exists() and p.is_file():
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("NVIDIA_API_KEY="):
                                resolved_key = line.split("=", 1)[1].strip().strip("'\"")
                                break
                if resolved_key:
                    break

        self.api_key = resolved_key
        self.base_url = base_url
        self.model = model
        self.small_model = small_model or os.environ.get("SMALL_TIER_MODEL")
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_backoff_ms = initial_backoff_ms
        self.max_backoff_ms = max_backoff_ms
        self._client = client  # dependency injection for tests
        self._openai = None  # resolved lazily

    def get_model_for_tier(self, tier: str) -> str:
        """Resolve model name for the requested tier ('small' vs 'big')."""
        if tier.lower() == "small" and self.small_model:
            return self.small_model
        return self.model

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._openai is None:
            OpenAI = _import_openai()
            self._openai = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._openai

    def chat_complete(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Run a chat completion with retry/backoff and latency tracking."""
        model = model or self.model
        temperature = self.temperature if temperature is None else temperature
        start = time.perf_counter()
        last_error: Optional[Exception] = None
        backoff_ms = self.initial_backoff_ms

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    temperature=temperature,
                    timeout=timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                input_tokens, output_tokens = _extract_usage(response)
                content = ""
                try:
                    content = response.choices[0].message.content or ""
                except (AttributeError, IndexError):
                    content = ""
                return LLMResponse(
                    content=content,
                    model=getattr(response, "model", model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    raw={"response": response},
                )
            except Exception as exc:  # noqa: BLE001 - retry/backoff on provider errors
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable(exc):
                    break
                sleep_s = min(backoff_ms / 1000.0, self.max_backoff_ms / 1000.0)
                # jitter: sleep in [sleep_s, 2 * sleep_s]
                time.sleep(sleep_s + random.random() * sleep_s)
                backoff_ms = min(backoff_ms * 2, self.max_backoff_ms)

        latency_ms = (time.perf_counter() - start) * 1000.0
        raise LLMRateLimitError(
            f"LLM call failed after {self.max_retries + 1} attempts: {last_error}"
        )


__all__ = [
    "LLMError",
    "LLMRateLimitError",
    "LLMProviderError",
    "LLMResponse",
    "NVIDIABuildClient",
]
