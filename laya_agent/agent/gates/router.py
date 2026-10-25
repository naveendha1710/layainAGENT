"""Action Router Gate ([D]).

Evaluates agent trajectory state before LLM invocation using the categorical
``choice`` primitive:
  * ``call_llm``: Requires multi-step planning, synthesis, or creative tool parameterization.
  * ``bypass_llm``: Predictable routine follow-up action requiring no new planning.

Fail-open contract:
  Defaults to ``Decision.CALL_LLM`` when confidence < tau_router or upon exception.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from laya_agent.agent.gates.base import Decision, Gate, GateResult


ROUTER_SCHEMA = {
    "action": {
        "type": "choice",
        "instructions": "Can the next step bypass the LLM via deterministic follow-up?",
        "criteria": {
            "bypass_llm": "Predictable routine follow-up action requiring no new planning",
            "call_llm": "Requires novel reasoning, multi-tool synthesis, or error recovery",
        },
    }
}

DEFAULT_TAU_ROUTER = 0.80


class RouterGate(Gate):
    """Action Router gate evaluating pre-LLM execution bypass."""

    def __init__(
        self,
        tau: float = DEFAULT_TAU_ROUTER,
        mode: str = "laya",  # "laya" | "off"
        engine: Optional[Any] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            name="router",
            tau=tau,
            enabled=(mode != "off"),
            device=device,
        )
        self.mode = mode
        self.engine = engine
        if self.engine is None and self.mode == "laya":
            from laya_agent.agent.providers.laya import LayaProvider

            self.engine = LayaProvider(device=device)

    def default(self, ctx: Dict[str, Any]) -> GateResult:
        """Fail-open default: call hosted LLM (Decision.CALL_LLM)."""
        return GateResult(
            decision=Decision.CALL_LLM,
            confidence=0.0,
            latency_ms=0.0,
            fallback_used=True,
            raw_output={
                "action": "call_llm",
                "reason": "fail_open_default",
                "gate": self.name,
            },
        )

    def _evaluate(self, ctx: Dict[str, Any]) -> GateResult:
        if self.mode == "off":
            return GateResult(
                decision=Decision.CALL_LLM,
                confidence=1.0,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"action": "call_llm", "mode": "off"},
            )

        # 1. Explicit bypass context overrides for unit & integration testing
        if ctx.get("force_bypass") or ctx.get("deterministic_tool"):
            tool_name = ctx.get("deterministic_tool", "confirm_step")
            tool_args = ctx.get("deterministic_args", {})
            return GateResult(
                decision=Decision.BYPASS_LLM,
                confidence=0.95,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={
                    "action": "bypass_llm",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "reason": "explicit_deterministic_action",
                },
            )

        messages = ctx.get("messages", [])
        trajectory = ctx.get("trajectory", [])
        recent_obs = str(ctx.get("recent_observation", ""))
        if not recent_obs and messages:
            # Check latest user message
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    recent_obs = msg.get("content", "")
                    break

        # Mode: laya
        if self.engine is not None:
            prompt_summary = recent_obs or (messages[0]["content"] if messages else "")
            if hasattr(self.engine, "predict"):
                res = self.engine.predict(
                    prompt_summary,
                    schema=ROUTER_SCHEMA,
                    ctx={
                        "trajectory_len": len(trajectory),
                        "messages": messages,
                        "recent_observation": recent_obs,
                    },
                )
            elif callable(self.engine):
                res = self.engine(prompt_summary, ctx=ctx)
            else:
                res = {}

            action_choice = res.get("action", {}).get("choice", "call_llm")
            confidence = float(res.get("action", {}).get("confidence", 0.90))
            decision = Decision.BYPASS_LLM if action_choice == "bypass_llm" else Decision.CALL_LLM

            tool_name = res.get("action", {}).get("tool_name", ctx.get("deterministic_tool"))
            tool_args = res.get("action", {}).get("tool_args", ctx.get("deterministic_args", {}))

            return GateResult(
                decision=decision,
                confidence=confidence,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={
                    "action": decision.value,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "engine_result": res,
                },
            )

        # Built-in offline deterministic classifier
        # 1. Explicit bypass context overrides for unit & integration testing
        if ctx.get("force_bypass") or ctx.get("deterministic_tool"):
            tool_name = ctx.get("deterministic_tool", "confirm_step")
            tool_args = ctx.get("deterministic_args", {})
            return GateResult(
                decision=Decision.BYPASS_LLM,
                confidence=0.95,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={
                    "action": "bypass_llm",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "reason": "explicit_deterministic_action",
                },
            )

        if ctx.get("force_call_llm"):
            return GateResult(
                decision=Decision.CALL_LLM,
                confidence=0.95,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"action": "call_llm", "reason": "explicit_call_llm"},
            )

        # 2. Pattern detection in recent observation
        obs_lower = recent_obs.lower()
        if (
            "[confirmation_required]" in obs_lower
            or "[routine_follow_up]" in obs_lower
            or "[bypass_llm]" in obs_lower
            or "please confirm to proceed" in obs_lower
        ):
            # Extract suggested tool if present, e.g. "tool: confirm_transfer"
            tool_match = re.search(r"tool:\s*([a-zA-Z0-9_]+)", recent_obs, re.IGNORECASE)
            tool_name = tool_match.group(1) if tool_match else "confirm_action"
            return GateResult(
                decision=Decision.BYPASS_LLM,
                confidence=0.90,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={
                    "action": "bypass_llm",
                    "tool_name": tool_name,
                    "tool_args": {},
                    "reason": "pattern_routine_follow_up",
                },
            )

        # 3. Default: novel planning / complex reasoning -> call hosted LLM
        return GateResult(
            decision=Decision.CALL_LLM,
            confidence=0.90,
            latency_ms=0.0,
            fallback_used=False,
            raw_output={"action": "call_llm", "reason": "default_reasoning"},
        )


__all__ = ["DEFAULT_TAU_ROUTER", "ROUTER_SCHEMA", "RouterGate"]
