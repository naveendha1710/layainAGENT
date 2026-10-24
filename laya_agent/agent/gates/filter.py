"""Applicability Filter Gate ([D]).

Evaluates candidate policies retrieved from FAISS against task context
using the categorical ``choice`` primitive:
  * ``keep``: Policy is strictly relevant to task constraints, permissions, or tools.
  * ``drop``: Policy is unrelated noise for this task; pruned from context to save tokens.

Fail-open contract:
  Defaults to ``Decision.KEEP`` when confidence < tau_filter or upon exception.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from laya_agent.agent.gates.base import Decision, Gate, GateResult


FILTER_SCHEMA = {
    "applies": {
        "type": "choice",
        "instructions": "Does this policy rule apply to the current user task?",
        "criteria": {
            "keep": "Directly governs tool permissions, parameters, or required task behavior",
            "drop": "Unrelated domain rule not applicable to this specific request",
        },
    }
}

DEFAULT_TAU_FILTER = 0.75

# Heuristic domain indicators for AgentDojo suites
SUITE_KEYWORDS = {
    "workspace": {"email", "inbox", "mail", "calendar", "event", "drive", "file", "document", "sheet", "contact"},
    "banking": {"bank", "account", "transfer", "balance", "wire", "transaction", "payment", "card", "iban", "recipient"},
    "slack": {"slack", "channel", "message", "chat", "post", "dm", "user", "invite", "thread"},
    "travel": {"flight", "hotel", "travel", "reservation", "booking", "airport", "car", "rental", "trip"},
}


class FilterGate(Gate):
    """Applicability Filter gate for post-retrieval policy pruning."""

    def __init__(
        self,
        tau: float = DEFAULT_TAU_FILTER,
        mode: str = "laya",  # "laya" | "off"
        engine: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            name="filter",
            tau=tau,
            enabled=(mode != "off"),
            device=device,
        )
        self.mode = mode
        self.engine = engine

    def default(self, ctx: Dict[str, Any]) -> GateResult:
        """Fail-open default: retain candidate policy (Decision.KEEP)."""
        return GateResult(
            decision=Decision.KEEP,
            confidence=0.0,
            latency_ms=0.0,
            fallback_used=True,
            raw_output={
                "applies": "keep",
                "reason": "fail_open_default",
                "gate": self.name,
            },
        )

    def _evaluate(self, ctx: Dict[str, Any]) -> GateResult:
        policy = ctx.get("policy", "")
        policy_text = policy.text if hasattr(policy, "text") else str(policy)
        user_prompt = str(ctx.get("user_prompt", "") or ctx.get("prompt", "") or "").strip()
        rule_id = getattr(policy, "rule_id", ctx.get("rule_id", "policy_rule"))
        suite = getattr(policy, "suite", ctx.get("suite", "")).lower()

        if self.mode == "off":
            return GateResult(
                decision=Decision.KEEP,
                confidence=1.0,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"applies": "keep", "rule_id": rule_id, "mode": "off"},
            )

        # Mode: laya
        if self.engine is not None:
            if hasattr(self.engine, "predict"):
                res = self.engine.predict(
                    policy_text,
                    schema=FILTER_SCHEMA,
                    ctx={"user_prompt": user_prompt, "rule_id": rule_id},
                )
            elif callable(self.engine):
                res = self.engine(policy_text, user_prompt=user_prompt)
            else:
                res = {}

            applies_choice = res.get("applies", {}).get("choice", "keep")
            confidence = float(res.get("applies", {}).get("confidence", 0.90))
            decision = Decision.DROP if applies_choice == "drop" else Decision.KEEP

            return GateResult(
                decision=decision,
                confidence=confidence,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={
                    "applies": decision.value,
                    "rule_id": rule_id,
                    "engine_result": res,
                },
            )

        # Built-in offline deterministic classifier
        policy_lower = policy_text.lower()
        prompt_lower = user_prompt.lower()

        # 1. Explicit directive markers for controlled testing
        if "[drop]" in policy_lower or "[irrelevant]" in policy_lower or "[noise]" in policy_lower:
            return GateResult(
                decision=Decision.DROP,
                confidence=0.95,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"applies": "drop", "rule_id": rule_id, "reason": "marker_drop"},
            )
        if "[keep]" in policy_lower or "[relevant]" in policy_lower:
            return GateResult(
                decision=Decision.KEEP,
                confidence=0.95,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"applies": "keep", "rule_id": rule_id, "reason": "marker_keep"},
            )

        # 2. Check for domain cross-contamination if suite metadata is available
        prompt_suites = {
            s_name for s_name, keywords in SUITE_KEYWORDS.items()
            if any(k in prompt_lower for k in keywords)
        }
        if suite and prompt_suites and suite not in prompt_suites:
            # Policy belongs to a totally different suite than the prompt indicates
            words_policy = set(re.findall(r"\w{4,}", policy_lower))
            words_prompt = set(re.findall(r"\w{4,}", prompt_lower))
            common = words_policy & words_prompt - {"must", "should", "never", "only", "with", "from", "that"}
            if not common:
                return GateResult(
                    decision=Decision.DROP,
                    confidence=0.88,
                    latency_ms=0.0,
                    fallback_used=False,
                    raw_output={
                        "applies": "drop",
                        "rule_id": rule_id,
                        "reason": f"suite_mismatch_{suite}_vs_{list(prompt_suites)}",
                    },
                )

        # 3. Check text relevance: shared keywords between policy and prompt
        words_policy = set(re.findall(r"\w{4,}", policy_lower))
        words_prompt = set(re.findall(r"\w{4,}", prompt_lower))
        common = words_policy & words_prompt - {"must", "should", "never", "only", "with", "from", "that", "this", "user", "data"}

        if common:
            confidence = min(0.95, 0.75 + len(common) * 0.05)
            return GateResult(
                decision=Decision.KEEP,
                confidence=confidence,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"applies": "keep", "rule_id": rule_id, "overlap": list(common)},
            )

        # 4. If prompt mentions domain keywords that align with policy keywords
        for s_name, keywords in SUITE_KEYWORDS.items():
            if any(k in prompt_lower for k in keywords) and any(k in policy_lower for k in keywords):
                return GateResult(
                    decision=Decision.KEEP,
                    confidence=0.85,
                    latency_ms=0.0,
                    fallback_used=False,
                    raw_output={"applies": "keep", "rule_id": rule_id, "suite_match": s_name},
                )

        # 5. Default conservative keep (utility preservation)
        return GateResult(
            decision=Decision.KEEP,
            confidence=0.80,
            latency_ms=0.0,
            fallback_used=False,
            raw_output={"applies": "keep", "rule_id": rule_id, "reason": "default_keep"},
        )

    def filter_policies(
        self,
        policies: Sequence[Any],
        user_prompt: str,
        extra_ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], List[GateResult]]:
        """Evaluate a sequence of policies and return retained policies along with gate results."""
        retained: List[Any] = []
        results: List[GateResult] = []
        extra = extra_ctx or {}

        for p in policies:
            ctx = {"policy": p, "user_prompt": user_prompt, **extra}
            res = self(ctx)
            results.append(res)
            if res.decision == Decision.KEEP:
                retained.append(p)

        return retained, results


__all__ = ["DEFAULT_TAU_FILTER", "FILTER_SCHEMA", "FilterGate"]
