"""Standard ``Gate`` and ``GateResult`` abstraction shared by all five LAYA gates.

All gates execute locally on GPU via ``laya.Router`` (ModernBERT-large) and are
hooked into the unified ReAct loop. A gate is active or inactive strictly through
its ``enabled`` flag, which is driven by ``configs/arms.yaml`` ("build once,
toggle via config").

Fail-open contract:
  * Utility gates (scorer, filter, router, supervisor, firewall relevance)
    fail open to their safe default when ``confidence < tau``.
  * The firewall injection check (``noul`` primitive) redacts rather than
    fails open; that redaction policy lives in ``firewall.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import time


class Decision(str, Enum):
    """Standardized gate decisions shared across all five LAYA gates.

    Individual gates consume the subset relevant to their schema primitive
    (``choice`` or ``noul``).
    """

    # Supervisor / Router / Firewall relevance
    CONTINUE = "continue"
    NUDGE = "nudge"
    STOP = "stop"

    # Firewall relevance / Filter
    KEEP = "keep"
    DROP = "drop"

    # Router
    BYPASS_LLM = "bypass_llm"
    CALL_LLM = "call_llm"

    # Scorer model tier
    SMALL = "small"
    BIG = "big"

    # Firewall injection redaction
    REDACT = "redact"
    PASS = "pass"


@dataclass
class GateResult:
    """Standardized result returned by every LAYA gate evaluation."""

    decision: Decision
    confidence: float
    latency_ms: float
    fallback_used: bool
    raw_output: Dict[str, Any] = field(default_factory=dict)


class Gate(ABC):
    """Abstract base class for all LAYA decision gates.

    Subclasses implement ``_evaluate`` returning a ``GateResult``. The base
    ``__call__`` handles:

    * disabled (pass-through) shortcut,
    * GPU-synchronized timing (``torch.cuda.synchronize()`` before stopping the
      timer, per the timing-precision requirement),
    * exception-safe fallback to ``default()`` with ``fallback_used=True``,
    * fail-open on ``confidence < tau``.
    """

    def __init__(
        self,
        name: str,
        tau: float,
        enabled: bool = True,
        device: str = "cuda",
    ) -> None:
        self.name = name
        self.tau = float(tau)
        self.enabled = bool(enabled)
        self.device = device

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _evaluate(self, ctx: Dict[str, Any]) -> GateResult:
        """Run the gate's underlying model/heuristic on ``ctx``.

        Must not handle exceptions itself — the base ``__call__`` does that.
        """
        ...

    def default(self, ctx: Dict[str, Any]) -> GateResult:
        """Fail-open default returned when ``confidence < tau`` or on exception.

        Subclasses override to return their gate-specific safe default
        (scorer->BIG, filter->KEEP, router->CALL_LLM, supervisor->CONTINUE).
        """
        return GateResult(
            decision=Decision.CALL_LLM,
            confidence=0.0,
            latency_ms=0.0,
            fallback_used=True,
            raw_output={"reason": "fail_open_default", "gate": self.name},
        )

    # ------------------------------------------------------------------
    # Public entry point (timing + exception safety + fail-open)
    # ------------------------------------------------------------------
    def __call__(self, ctx: Dict[str, Any]) -> GateResult:
        """Evaluate the gate with timing, exception safety, and fail-open."""
        # Disabled gate: pass-through, no latency, no fallback.
        if not self.enabled:
            default_res = self.default(ctx)
            return GateResult(
                decision=default_res.decision,
                confidence=1.0,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"reason": "gate_disabled", "gate": self.name},
            )

        start = time.perf_counter()
        try:
            result = self._evaluate(ctx)
        except Exception:  # noqa: BLE001 - exception-safe fallback is required
            import sys
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = self.default(ctx)
            result.fallback_used = True

        # Synchronize GPU before stopping the timer (timing-precision rule).
        try:
            import torch  # lazy import: no hard torch dep at import time

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result.latency_ms = elapsed_ms

        # Fail-open on low confidence.
        if result.confidence < self.tau:
            result = self.default(ctx)
            result.fallback_used = True
            result.latency_ms = elapsed_ms

        return result

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"tau={self.tau}, enabled={self.enabled}, device={self.device!r})"
        )


__all__ = ["Decision", "GateResult", "Gate"]
