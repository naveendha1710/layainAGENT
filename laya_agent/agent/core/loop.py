"""Unified ReAct controller (Plan -> Act -> Observe) with gate hook points.

All five gate insertion points (scorer, filter, router, firewall, supervisor)
are present in every arm but active only when their mode flag in
``configs/arms.yaml`` is not ``off`` ("build once, toggle via config").
"""

from __future__ import annotations

import inspect
import json as _json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
        policy_store: Optional[Any] = None,
        policy_top_k: int = 10,
        config: Optional[LoopConfig] = None,
    ) -> None:
        self.state = state
        self.hooks = hooks
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.policy_store = policy_store
        self.policy_top_k = policy_top_k
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
    def run(
        self,
        initial_prompt: str,
        candidate_policies: Optional[Sequence[Any]] = None,
    ) -> AgentState:
        # [D] Policy Retrieval & Applicability Filter Gate
        candidates = list(candidate_policies) if candidate_policies is not None else []
        if not candidates and self.policy_store is not None:
            raw_search = self.policy_store.search(initial_prompt, top_k=self.policy_top_k)
            candidates = [
                entry for entry, _ in raw_search
            ] if raw_search and isinstance(raw_search[0], tuple) else list(raw_search)

        retained_policies = []
        for cand in candidates:
            cand_text = cand.text if hasattr(cand, "text") else str(cand)
            rule_id = getattr(cand, "rule_id", "rule")
            suite = getattr(cand, "suite", "")
            filter_result = self._run_hook(
                "filter",
                policy=cand,
                policy_text=cand_text,
                rule_id=rule_id,
                suite=suite,
                user_prompt=initial_prompt,
            )
            if filter_result is not None:
                self.state.record_step(StepRecord(
                    step=self.state.step,
                    module="filter",
                    decision=filter_result.decision.value,
                    confidence=filter_result.confidence,
                    fallback_used=filter_result.fallback_used,
                    latency_ms=filter_result.latency_ms,
                    raw=filter_result.raw_output,
                ))
                if filter_result.decision == Decision.KEEP:
                    retained_policies.append(cand)
            else:
                # Filter hook disabled/not configured -> retain policy unconditionally (Arms A, B, C)
                retained_policies.append(cand)

        # Build strict ReAct system instructions with tool descriptions
        tool_specs = self.tool_registry.specs() if hasattr(self.tool_registry, "specs") else {}
        tool_names = ", ".join(tool_specs.keys())
        tool_descriptions = "\n".join(
            f"- {name}: {spec.description}" for name, spec in tool_specs.items()
        )
        react_system_prompt = (
            f"Answer the following questions as best you can. You have access to the following tools:\n"
            f"{tool_descriptions}\n\n"
            f"Use the following format:\n"
            f"Thought: you should always think about what to do\n"
            f"Action: the action to take, should be one of [{tool_names}]\n"
            f"Action Input: the input to the action\n"
            f"Observation: the result of the action\n"
            f"... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            f"Thought: I now know the final answer\n"
            f"Final Answer: the final answer to the original input question\n\n"
            f"Example:\n"
            f"Question: Locate the security report in archive.\n"
            f"Thought: I should search files for archive documents.\n"
            f"Action: search_files\n"
            f"Action Input: {{\"target\": \"archive\"}}\n"
            f"Observation: Found files matching archive: report_2026.pdf\n"
            f"Thought: I now know the final answer.\n"
            f"Final Answer: The security report is report_2026.pdf."
        )
        has_react_prompt = any(
            m.get("role") == "system" and "You have access to the following tools:" in m.get("content", "")
            for m in self.state.messages
        )
        if not has_react_prompt:
            self.state.messages.append({
                "role": "system",
                "content": react_system_prompt,
            })

        if retained_policies:
            policy_lines = [
                f"- [{getattr(p, 'rule_id', f'rule_{i}')}] {p.text if hasattr(p, 'text') else str(p)}"
                for i, p in enumerate(retained_policies)
            ]
            self.state.messages.append({
                "role": "system",
                "content": "Applicable Domain Policies:\n" + "\n".join(policy_lines),
            })

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
            router_result = self._run_hook(
                "router",
                messages=self.state.messages,
                trajectory=self.state.trajectory,
            )
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
                model_kwarg = {"tier": self.state.model_tier} if self.state.model_tier else {}
                try:
                    sig = inspect.signature(self.llm_client.chat_complete)
                    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if not has_kwargs and "tier" not in sig.parameters:
                        model_kwarg = {"model": self.state.model_tier} if self.state.model_tier else {}
                except Exception:
                    pass

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
                tool_name = (
                    router_result.raw_output.get("tool_name")
                    if router_result and router_result.raw_output
                    else None
                )
                tool_args = (
                    router_result.raw_output.get("tool_args", {})
                    if router_result and router_result.raw_output
                    else {}
                )
                if not tool_name:
                    tool_name, tool_args = self._deterministic_follow_up()

            # --- ACT / execute tool ---
            tool_result = self._execute_tool(tool_name, tool_args)

            # --- OBSERVE: firewall on tool output ---
            user_prompt = self.state.get_user_prompt()
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
        """Robustly parse tool name and arguments from LLM text.

        Supports:
          1. Legacy TOOL format:
             TOOL: <name> <json_args>
          2. ReAct format (supporting markdown code fences ``` or ```json, extra whitespace):
             Thought: ...
             Action: <name>
             Action Input: <args>
          3. Direct function-call / JSON format:
             {"tool": "...", "args": ...} or read_file(...)
          4. Fallback regex matching against known tool names in ToolRegistry when model
             omits strict prefixes.
        """
        if not content or not content.strip():
            return None

        # 1. Check for legacy TOOL: prefix
        marker = "TOOL:"
        idx = content.find(marker)
        if idx >= 0:
            line = content[idx + len(marker):].strip()
            parts = line.split(None, 1)
            if parts:
                name = parts[0].strip()
                args: Dict[str, Any] = {}
                if len(parts) > 1:
                    raw_arg = parts[1].strip()
                    try:
                        args = _json.loads(raw_arg)
                    except (_json.JSONDecodeError, ValueError):
                        args = {"target": raw_arg.strip("'\"")}
                return name, self._normalize_args(name, args)

        # 2. Check for ReAct Action: / Action Input: format
        action_match = re.search(
            r"(?:^|\n)\s*(?:\*{1,2})?Action(?:\*{1,2})?\s*:\s*[`\"']*([a-zA-Z0-9_]+)[`\"']*",
            content,
            re.IGNORECASE,
        )
        if action_match:
            name = action_match.group(1).strip()
            input_match = re.search(
                r"(?:^|\n)\s*(?:\*{1,2})?Action\s+Input(?:\*{1,2})?\s*:\s*(.*)",
                content,
                re.IGNORECASE | re.DOTALL,
            )
            raw_input = input_match.group(1).strip() if input_match else ""

            # Truncate at subsequent ReAct keywords
            for stop in ["\nObservation:", "\nThought:", "\nFinal Answer:"]:
                pos = raw_input.find(stop)
                if pos != -1:
                    raw_input = raw_input[:pos].strip()

            # Strip markdown code blocks (``` or ```json)
            fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_input, re.DOTALL | re.IGNORECASE)
            if fence_match:
                raw_input = fence_match.group(1).strip()
            elif raw_input.startswith("```") and raw_input.endswith("```"):
                raw_input = raw_input.strip("`").strip()

            args: Dict[str, Any] = {}
            if raw_input:
                try:
                    parsed_json = _json.loads(raw_input)
                    if isinstance(parsed_json, dict):
                        args = parsed_json
                    else:
                        args = {"target": parsed_json}
                except (_json.JSONDecodeError, ValueError):
                    clean_input = raw_input.strip()
                    if (clean_input.startswith("'") and clean_input.endswith("'")) or (
                        clean_input.startswith('"') and clean_input.endswith('"')
                    ):
                        clean_input = clean_input[1:-1].strip()
                    if "=" in clean_input and not clean_input.startswith("{"):
                        kvs = re.findall(r"([a-zA-Z0-9_]+)\s*=\s*['\"]?([^'\",\n]+)['\"]?", clean_input)
                        if kvs:
                            args = {k: v.strip() for k, v in kvs}
                        else:
                            args = {"target": clean_input}
                    else:
                        args = {"target": clean_input}
            return name, self._normalize_args(name, args)

        # 3. If explicit Final Answer is present without Action/TOOL, finish loop
        if re.search(r"(?:^|\n)\s*(?:\*{1,2})?Final\s+Answer(?:\*{1,2})?\s*:", content, re.IGNORECASE):
            return None

        # 4. Fallback: regex matching against known tool names in ToolRegistry
        known_tools = (
            set(self.tool_registry.specs().keys())
            if hasattr(self.tool_registry, "specs")
            else set()
        )
        if known_tools:
            # Check for JSON block containing tool name
            json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_block:
                try:
                    data = _json.loads(json_block.group(1))
                    cand_tool = data.get("tool") or data.get("name") or data.get("action")
                    if cand_tool and cand_tool in known_tools:
                        tool_args = (
                            data.get("args")
                            or data.get("arguments")
                            or data.get("parameters")
                            or data.get("action_input")
                            or {}
                        )
                        if isinstance(tool_args, dict):
                            return cand_tool, self._normalize_args(cand_tool, tool_args)
                        return cand_tool, self._normalize_args(cand_tool, {"target": str(tool_args)})
                except Exception:
                    pass

            # Check for function call syntax: tool_name(...)
            for tool in sorted(known_tools, key=len, reverse=True):
                call_match = re.search(rf"\b{tool}\s*\((.*?)\)", content, re.DOTALL)
                if call_match:
                    inside = call_match.group(1).strip()
                    arg_dict: Dict[str, Any] = {}
                    if "=" in inside:
                        for k, v in re.findall(r"([a-zA-Z0-9_]+)\s*=\s*['\"]?([^'\",\n]+)['\"]?", inside):
                            arg_dict[k] = v.strip()
                    elif inside:
                        arg_dict = {"target": inside.strip("'\"")}
                    return tool, self._normalize_args(tool, arg_dict)

            # Check for natural mention of known tool: e.g. "use the read_file tool on 'inbox.txt'"
            for tool in sorted(known_tools, key=len, reverse=True):
                if re.search(rf"\b(?:use|using|call|invoke|execute)?\s*(?:the\s+)?{tool}\b", content, re.IGNORECASE):
                    q_match = re.search(r"['\"]([a-zA-Z0-9_.-]+\.[a-zA-Z0-9]{2,4})['\"]", content)
                    target = q_match.group(1) if q_match else ""
                    args = {"target": target} if target else {}
                    return tool, self._normalize_args(tool, args)

        return None

    def _normalize_args(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameter aliases to target or appropriate tool signature."""
        normalized = dict(args)
        if "target" not in normalized:
            for alias in ["file", "filename", "path", "query", "url", "channel"]:
                if alias in normalized:
                    normalized["target"] = normalized[alias]
                    break
        return normalized

    def _deterministic_follow_up(self) -> Tuple[str, Dict[str, Any]]:
        """Router bypass: deterministic no-op follow-up requiring no LLM."""
        return "noop", {}

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        try:
            call_args = dict(args)
            if hasattr(self.tool_registry, "specs"):
                spec = self.tool_registry.specs().get(name)
                if spec and hasattr(spec, "fn"):
                    sig = inspect.signature(spec.fn)
                    params = list(sig.parameters.keys())
                    # If tool accepts no arguments (e.g. confirm_action, list_files)
                    if not params:
                        call_args = {}
                    elif len(params) == 1 and params[0] not in call_args and call_args:
                        call_args = {params[0]: next(iter(call_args.values()))}

            result = self.tool_registry.execute(name, **call_args)
            return ToolResult(output=str(result), ok=True, latency_ms=(time.perf_counter() - t0) * 1000.0)
        except Exception as exc:  # noqa: BLE001 - capture tool errors as observations
            return ToolResult(output=str(exc), ok=False, latency_ms=(time.perf_counter() - t0) * 1000.0)


__all__ = ["GateHook", "LoopConfig", "ReActLoop", "ToolResult"]
