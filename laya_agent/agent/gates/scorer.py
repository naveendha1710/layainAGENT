"""Task Difficulty & Risk Scorer Gate ([C]).

Evaluates the initial task prompt using the categorical ``choice`` primitive:
  * ``small``: Simple lookup, single tool query, direct factual extraction (max_steps=4).
  * ``big``: Multi-step planning, synthesis, high-risk state modification (max_steps=10).

Fail-open rule: Defaults to ``Decision.BIG`` with full step budget when
confidence < tau_scorer or upon exception.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional

from laya_agent.agent.gates.base import Decision, Gate, GateResult


SCORER_SCHEMA = {
    "model_tier": {
        "type": "choice",
        "instructions": "Select the required model tier for this user task.",
        "criteria": {
            "small": "Simple lookup, single tool call, or direct factual extraction",
            "big": "Multi-step planning, complex synthesis, or high-risk state modification",
        },
    }
}

DEFAULT_SMALL_MAX_STEPS = 4
DEFAULT_BIG_MAX_STEPS = 10
DEFAULT_SMALL_BUDGET = DEFAULT_SMALL_MAX_STEPS
DEFAULT_BIG_BUDGET = DEFAULT_BIG_MAX_STEPS


class ScorerGate(Gate):
    """Task Difficulty & Risk Scorer gate for dynamic model tier routing."""

    def __init__(
        self,
        tau: float = 0.70,
        mode: str = "laya",  # "laya" | "always_small" | "random" | "off"
        small_max_steps: int = DEFAULT_SMALL_MAX_STEPS,
        big_max_steps: int = DEFAULT_BIG_MAX_STEPS,
        engine: Optional[Any] = None,
        seed: Optional[int] = 42,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            name="scorer",
            tau=tau,
            enabled=(mode != "off"),
            device=device,
        )
        self.mode = mode
        self.small_max_steps = small_max_steps
        self.big_max_steps = big_max_steps
        self.engine = engine
        if self.engine is None and self.mode == "laya":
            from laya_agent.agent.providers.laya import LayaProvider

            self.engine = LayaProvider(device=device)
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def default(self, ctx: Dict[str, Any]) -> GateResult:
        """Fail-open default: route to BIG tier with full step budget."""
        return GateResult(
            decision=Decision.BIG,
            confidence=0.0,
            latency_ms=0.0,
            fallback_used=True,
            raw_output={
                "model_tier": "big",
                "max_steps": self.big_max_steps,
                "reason": "fail_open_default",
                "gate": self.name,
            },
        )

    def _evaluate(self, ctx: Dict[str, Any]) -> GateResult:
        prompt = str(ctx.get("initial_prompt", "") or ctx.get("user_prompt", "")).strip()

        if self.mode == "off":
            return GateResult(
                decision=Decision.BIG,
                confidence=1.0,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"model_tier": "big", "max_steps": self.big_max_steps, "mode": "off"},
            )

        if self.mode == "always_small":
            return GateResult(
                decision=Decision.SMALL,
                confidence=1.0,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"model_tier": "small", "max_steps": self.small_max_steps, "mode": "always_small"},
            )

        if self.mode == "random":
            choice = self._rng.choice([Decision.SMALL, Decision.BIG])
            steps = self.small_max_steps if choice == Decision.SMALL else self.big_max_steps
            return GateResult(
                decision=choice,
                confidence=1.0,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"model_tier": choice.value, "max_steps": steps, "mode": "random"},
            )

        # Mode: laya
        if self.engine is not None:
            if hasattr(self.engine, "predict"):
                res = self.engine.predict(prompt, schema=SCORER_SCHEMA)
            elif callable(self.engine):
                res = self.engine(prompt)
            else:
                res = {}

            tier_choice = res.get("model_tier", {}).get("choice", "big")
            confidence = float(res.get("model_tier", {}).get("confidence", 0.90))

            decision = Decision.SMALL if tier_choice == "small" else Decision.BIG
            steps = self.small_max_steps if decision == Decision.SMALL else self.big_max_steps

            return GateResult(
                decision=decision,
                confidence=confidence,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"model_tier": decision.value, "max_steps": steps, "engine_result": res},
            )

        # Built-in offline deterministic scorer
        prompt_lower = prompt.lower()

        # Indicators of high complexity / risk -> BIG
        big_patterns = [
            r"transfer",
            r"wire",
            r"balance.*and.*transact",
            r"recurring",
            r"plan",
            r"coordinate",
            r"schedule.*meeting.*external",
            r"export.*archive",
            r"invoice",
            r"budget",
            r"book.*flight",
            r"cancel.*reservation",
            r"admin",
            r"exfiltrate",
            r"multi-step",
        ]

        # Indicators of simple lookup -> SMALL
        small_patterns = [
            r"^find\s+file",
            r"^read\s+",
            r"^check\s+balance",
            r"^get\s+status",
            r"^what\s+is",
            r"^lookup",
            r"^search\s+",
            r"^compute\s+\d+\s*\+",
        ]

        is_big = any(re.search(pat, prompt_lower) for pat in big_patterns)
        is_small = any(re.search(pat, prompt_lower) for pat in small_patterns)

        # Word count heuristic: prompts > 20 words usually involve multi-step requirements
        word_count = len(prompt.split())
        if word_count > 20:
            is_big = True

        if is_small and not is_big:
            return GateResult(
                decision=Decision.SMALL,
                confidence=0.92,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"model_tier": "small", "max_steps": self.small_max_steps, "reason": "simple_query"},
            )

        return GateResult(
            decision=Decision.BIG,
            confidence=0.95,
            latency_ms=0.0,
            fallback_used=False,
            raw_output={"model_tier": "big", "max_steps": self.big_max_steps, "reason": "complex_task"},
        )


__all__ = [
    "DEFAULT_BIG_MAX_STEPS",
    "DEFAULT_SMALL_MAX_STEPS",
    "SCORER_SCHEMA",
    "ScorerGate",
]
