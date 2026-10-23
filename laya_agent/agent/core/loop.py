"""Unified ReAct controller (Plan -> Act -> Observe) with gate hook points.

All five gate insertion points (scorer, filter, router, firewall, supervisor)
are present in every arm but active only when their mode flag in
``configs/arms.yaml`` is not ``off`` ("build once, toggle via config").
"""

from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from laya_agent.agent.core.state import AgentState, StepRecord
from laya_agent.agent.gates.base import Decision, Gate, GateResult


@dataclass
class GateHook:
    """A single conditional gate insertion point in the ReAct loop."""

    name: str
    gate: Optional[Gate]
    ctx_builder: Callable[[AgentState], Dict[str, Any]]

    def evaluate(self, state: AgentState, **extra_ctx: Any) -> Optional[GateResult]:
        if self.gate is None or not self.gate.enabled:
            return None
        ctx = self.ctx_builder(state)
        ctx.update(extra_ctx)
        return self.gate(ctx)


@dataclass
class LoopConfig:
    max_steps: int = 10
    temperature: float = 0.0


@dataclass
class ToolResult:
    output: str
    ok: bool
    latency_ms: float = 0.0


class ReActLoop:
    """Plan-Act-Observe loop with optional LAYA gate hooks."""

    def __init__(
        self,
        *,
        state: AgentState,
        hooks: Dict[str, GateHook],
        llm_client: Any,
        tool_registry: Any,
        config: Optional[LoopConfig] = None,
    ) -> None:
        self.state = state
        self.hooks = hooks
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config or LoopConfig(max_steps=state.max_steps)

    # ------------------------------------------------------------------
    # gate helpers
    # ------------------------------------------------------------------
    def _run_hook(self, name: str, **extra_ctx: Any) -> Optional[GateResult]:
        hook = self.hooks.get(name)
        if hook is None:
            return None
        return hook.evaluate(self.state, **extra_ctx)

    # ------------------------------------------------------------------
    # main cycle
    # ------------------------------------------------------------------
    def run(self, initial_prompt: str) -> AgentState:
        self.state.messages.append({"role": "user", "content": initial_prompt})

        # [C] Scorer gate (pre-loop): selects tier / step budget.
        scorer_result = self._run_hook("scorer", initial_prompt=initial_prompt)
        if scorer_result is not None:
            self.state.record_step(StepRecord(
                step=self.state.step,
                module="scorer",
                decision=scorer_result.decision.value,
                confidence=scorer_result.confidence,
                fallback_used=scorer_result.fallback_used,
                latency_ms=scorer_result.latency_ms,
                raw=scorer_result.raw_output,
            ))
            self.state.model_tier = scorer_result.decision.value
            max_steps = scorer_result.raw_output.get("max_steps")
            if max_steps is not None:
                self.state.max_steps = int(max_steps)
            elif scorer_result.decision == Decision.SMALL:
                self.state.max_steps = 4

        while self.state.can_continue():
            # --- PLAN / reason via LLM (unless router says bypass) ---
            router_result = self._run_hook("router")
            call_llm = True
            if router_result is not None:
                self.state.record_step(StepRecord(
                    step=self.state.step,
                    module="router",
                    decision=router_result.decision.value,
                    confidence=router_result.confidence,
                    fallback_used=router_result.fallback_used,
                    latency_ms=router_result.latency_ms,
                    raw=router_result.raw_output,
                ))
                call_llm = (router_result.decision == Decision.CALL_LLM)

            tool_name: Optional[str] = None
            tool_args: Dict[str, Any] = {}

            if call_llm:
                model_kwarg = {}
                if hasattr(self.llm_client, "get_model_for_tier"):
                    model_kwarg["model"] = self.llm_client.get_model_for_tier(self.state.model_tier)
                elif hasattr(self.llm_client, "small_model") and self.state.model_tier == "small":
                    model_kwarg["model"] = self.llm_client.small_model

                t0 = time.perf_counter()
                response = self.llm_client.chat_complete(self.state.messages, **model_kwarg)
                llm_latency_ms = (time.perf_counter() - t0) * 1000.0
                self.state.record_step(StepRecord(
                    step=self.state.step,
                    module="llm",
                    decision=response.model,
                    confidence=1.0,
                    fallback_used=False,
                    latency_ms=llm_latency_ms,
                    llm_latency_ms=llm_latency_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ))
                self.state.messages.append(
                    {"role": "assistant", "content": response.content}
                )

                parsed = self._parse_tool_call(response.content)
                if parsed is None:
                    # No tool call -> final answer.
                    self.state.finished = True
                    self.state.final_answer = response.content
                    self.state.stop_reason = "final_answer"
                    break
                tool_name, tool_args = parsed
            else:
                # [D] Router bypass: deterministic follow-up, no LLM call.
                tool_name, tool_args = self._deterministic_follow_up()

            # --- ACT / execute tool ---
            tool_result = self._execute_tool(tool_name, tool_args)

            # --- OBSERVE: firewall on tool output ---
            user_prompt = self.state.messages[0]["content"] if self.state.messages else ""
            firewall_result = self._run_hook(
                "firewall",
                tool_output=tool_result.output,
                tool_name=tool_name,
                user_prompt=user_prompt,
            )
            observation = str(tool_result.output)
            if firewall_result is not None:
                self.state.record_step(StepRecord(
                    step=self.state.step,
                    module="firewall",
                    decision=firewall_result.decision.value,
                    confidence=firewall_result.confidence,
                    fallback_used=firewall_result.fallback_used,
                    latency_ms=firewall_result.latency_ms,
                    tool=tool_name,
                    tool_ok=tool_result.ok,
                    raw=firewall_result.raw_output,
                ))
                if firewall_result.decision == Decision.REDACT:
                    observation = "[FIREWALL: Tool output redacted due to suspected injection]"
                elif "sanitized_output" in firewall_result.raw_output:
                    observation = str(firewall_result.raw_output["sanitized_output"])

            self.state.messages.append({"role": "user", "content": observation})
            self.state.record_step(StepRecord(
                step=self.state.step,
                module="tool",
                decision=tool_name,
                confidence=1.0,
                fallback_used=not tool_result.ok,
                latency_ms=tool_result.latency_ms,
                tool=tool_name,
                tool_ok=tool_result.ok,
            ))

            # --- Supervisor gate (post-step) ---
            supervisor_result = self._run_hook(
                "supervisor",
                recent_observation=observation,
                trajectory=self.state.trajectory,
            )
            if supervisor_result is not None:
                self.state.record_step(StepRecord(
                    step=self.state.step,
                    module="supervisor",
                    decision=supervisor_result.decision.value,
                    confidence=supervisor_result.confidence,
                    fallback_used=supervisor_result.fallback_used,
                    latency_ms=supervisor_result.latency_ms,
                    raw=supervisor_result.raw_output,
                ))
                if supervisor_result.decision == Decision.STOP:
                    self.state.finished = True
                    self.state.stop_reason = "supervisor_stop"
                    break
                if supervisor_result.decision == Decision.NUDGE:
                    self.state.messages.append({
                        "role": "user",
                        "content": (
                            "[SUPERVISOR: nudge -- recover from loop/drift "
                            "and continue toward the goal.]"
                        ),
                    })

        if not self.state.finished:
            self.state.finished = True
            self.state.stop_reason = "max_steps"
        return self.state

    # ------------------------------------------------------------------
    # tool parsing / execution
    # ------------------------------------------------------------------
    def _parse_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parse ``TOOL: <name> <json_args>`` from LLM text."""
        marker = "TOOL:"
        idx = content.find(marker)
        if idx < 0:
            return None
        line = content[idx + len(marker):].strip()
        parts = line.split(None, 1)
        name = parts[0]
        args: Dict[str, Any] = {}
        if len(parts) > 1:
            try:
                args = _json.loads(parts[1])
            except _json.JSONDecodeError:
                args = {"_raw": parts[1]}
        return name, args

    def _deterministic_follow_up(self) -> Tuple[str, Dict[str, Any]]:
        """Router bypass: deterministic no-op follow-up requiring no LLM."""
        return "noop", {}

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        try:
            result = self.tool_registry.execute(name, **args)
            return ToolResult(output=str(result), ok=True, latency_ms=(time.perf_counter() - t0) * 1000.0)
        except Exception as exc:  # noqa: BLE001 - capture tool errors as observations
            return ToolResult(output=str(exc), ok=False, latency_ms=(time.perf_counter() - t0) * 1000.0)


__all__ = ["GateHook", "LoopConfig", "ReActLoop", "ToolResult"]
