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


try:
    from openai import APIStatusError, OpenAI  # type: ignore
except ImportError:
    APIStatusError = None
    OpenAI = None


def _import_openai():
    global OpenAI
    if OpenAI is None:
        try:
            from openai import OpenAI as _OpenAI  # type: ignore
            OpenAI = _OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LLMError(
                "The 'openai' package is required to call the NVIDIA Build API. "
                "Install it with: pip install openai>=1.0"
            ) from exc
    return OpenAI


def _is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a transient 404/429/5xx provider error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        return status in (404, 429, 500, 502, 503, 504) or (isinstance(status, int) and 500 <= status < 600)
    msg = str(exc).lower()
    return (
        "404" in msg
        or "not found" in msg
        or "429" in msg
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
        base_url: Optional[str] = None,
        big_tier_model: Optional[str] = None,
        small_tier_model: Optional[str] = None,
        model: Optional[str] = None,
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
        raw_base_url = (base_url or os.environ.get("NVIDIA_BUILD_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
        if not raw_base_url.endswith("/v1"):
            raw_base_url += "/v1"
        self.base_url = raw_base_url

        # Resolve explicit big and small tier models strictly without silent arbitrary defaults
        big = (
            big_tier_model
            or model
            or os.environ.get("BIG_TIER_MODEL")
            or os.environ.get("big_tier_model")
        )
        small = (
            small_tier_model
            or small_model
            or os.environ.get("SMALL_TIER_MODEL")
            or os.environ.get("small_tier_model")
        )

        if not big or not small:
            # Fall back strictly to arms.yaml configuration
            try:
                from laya_agent.configs.loader import load_arms_config

                cfg = load_arms_config()
                if not big:
                    big = cfg.get("big_tier_model") or cfg.get("BIG_TIER_MODEL")
                if not small:
                    small = cfg.get("small_tier_model") or cfg.get("SMALL_TIER_MODEL")
            except Exception:
                pass

        if not big or not small:
            raise ValueError(
                "Research evaluation requires explicit BIG_TIER_MODEL and SMALL_TIER_MODEL. "
                "Silent model fallbacks are disallowed."
            )

        self.big_tier_model = str(big).strip()
        self.small_tier_model = str(small).strip()
        if not self.big_tier_model or not self.small_tier_model:
            raise ValueError(
                "Research evaluation requires explicit BIG_TIER_MODEL and SMALL_TIER_MODEL. "
                "Silent model fallbacks are disallowed."
            )

        self.model = self.big_tier_model
        self.small_model = self.small_tier_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_backoff_ms = initial_backoff_ms
        self.max_backoff_ms = max_backoff_ms
        self._client = client  # dependency injection for tests
        self._openai = None  # resolved lazily

    def get_model_for_tier(self, tier: str) -> str:
        """Resolve model name for the requested tier ('small' vs 'big')."""
        if tier and tier.lower() == "small":
            return self.small_tier_model
        if tier and tier.lower() == "big":
            return self.big_tier_model
        raise ValueError(f"Unspecified or invalid model parameter: {tier}")

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
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run a chat completion with retry/backoff and latency tracking."""
        # Require model parameter (or allow tier kwarg), strictly mapped without silent fallbacks
        req_model = model if model is not None else kwargs.get("tier")
        if not req_model:
            raise ValueError(f"Unspecified or invalid model parameter: {model}")

        if req_model == "big":
            target_model = self.big_tier_model
        elif req_model == "small":
            target_model = self.small_tier_model
        elif req_model == self.big_tier_model:
            target_model = self.big_tier_model
        elif req_model == self.small_tier_model:
            target_model = self.small_tier_model
        else:
            raise ValueError(f"Unspecified or invalid model parameter: {model}")

        temperature = self.temperature if temperature is None else temperature
        timeout = timeout if timeout is not None else kwargs.get("timeout")
        eff_max_tokens = max_tokens if max_tokens is not None else kwargs.get("max_tokens", 512)
        start = time.perf_counter()
        last_error: Optional[Exception] = None
        initial_delay = max(0.05, self.initial_backoff_ms / 1000.0)
        backoff_factor = 2.0
        max_delay = self.max_backoff_ms / 1000.0

        for attempt_idx in range(self.max_retries + 1):
            try:
                create_kwargs: Dict[str, Any] = {
                    "model": target_model,
                    "messages": list(messages),
                    "temperature": temperature,
                }
                if timeout is not None:
                    create_kwargs["timeout"] = timeout
                if eff_max_tokens is not None:
                    create_kwargs["max_tokens"] = eff_max_tokens
                response = self.client.chat.completions.create(**create_kwargs)
                latency_ms = (time.perf_counter() - start) * 1000.0
                input_tokens, output_tokens = _extract_usage(response)
                content = ""
                try:
                    content = response.choices[0].message.content or ""
                except (AttributeError, IndexError):
                    content = ""

                resp_model = getattr(response, "model", None)
                if not resp_model or resp_model in ("default", "big", "small", "mock-default-model"):
                    resolved_model = target_model
                else:
                    resolved_model = resp_model

                return LLMResponse(
                    content=content,
                    model=resolved_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    raw={"response": response},
                )
            except Exception as exc:  # noqa: BLE001 - retry/backoff on provider errors
                last_error = exc
                err = exc
                status_code = getattr(err, "status_code", None) or getattr(err, "status", None)

                # Check if error is transient/retryable gateway error (404, 429, 500, 502, 503, 504)
                is_status_err = (APIStatusError is not None and isinstance(err, APIStatusError)) or (status_code is not None)
                is_transient_status = (status_code in (404, 429, 500, 502, 503, 504)) or (isinstance(status_code, int) and 500 <= status_code < 600)

                if is_status_err:
                    if not is_transient_status:
                        raise err
                elif not _is_retryable(err):
                    raise err

                attempt = attempt_idx + 1
                if attempt > self.max_retries:
                    raise err

                if not hasattr(err, "status_code"):
                    try:
                        setattr(err, "status_code", status_code or 503)
                    except Exception:
                        pass

                # Exponential backoff with jitter: delay = initial_delay * (backoff_factor ** (attempt - 1)) + jitter
                base_delay = min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)
                jitter = random.uniform(0.1, 0.5) * base_delay
                delay = min(base_delay + jitter, max_delay)

                code_val = getattr(err, "status_code", status_code or "ERROR")
                print(
                    f"  [Warn] NVIDIA NIM transient HTTP {code_val}. Retrying in {delay:.1f}s (attempt {attempt}/{self.max_retries}) ...",
                    flush=True,
                )
                time.sleep(delay)

        raise last_error if last_error is not None else RuntimeError("LLM call failed.")


__all__ = [
    "LLMError",
    "LLMRateLimitError",
    "LLMProviderError",
    "LLMResponse",
    "NVIDIABuildClient",
]
