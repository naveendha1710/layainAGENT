"""Tests for the Arm A ReAct baseline agent.

Covers step progression, tool execution, max-step termination, and an
end-to-end smoke run writing schema-valid JSONL to eval_arm_A_smoke.jsonl.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

from laya_agent.agent.core.baseline import ArmABaseline
from laya_agent.agent.core.state import AgentState
from laya_agent.agent.llm.client import LLMResponse
from laya_agent.tracing.trace import TraceLogger
from laya_agent.agent.tools.base import ToolRegistry
from laya_agent.eval.runner import ArmARunner


class FakeLLMClient:
    """Deterministic fake LLM: one TOOL call then a final answer.

    Mirrors the production ``NVIDIABuildClient.chat_complete`` contract by
    returning an ``LLMResponse`` with input/output tokens and latency.
    """

    def __init__(self, tool_call: str, final: str) -> None:
        self.tool_call = tool_call
        self.final = final
        self.calls = 0

    def chat_complete(self, messages, *, model=None, temperature=None, timeout=None, **kwargs):  # noqa: ARG002,ANN001
        import time as _time

        t0 = _time.perf_counter()
        self.calls += 1
        content = self.tool_call if self.calls == 1 else self.final
        return LLMResponse(
            content=content,
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=(_time.perf_counter() - t0) * 1000.0,
        )


class AlwaysToolLLMClient(FakeLLMClient):
    """Fake LLM that always emits a TOOL call (forces max-step termination)."""

    def chat_complete(self, messages, *, model=None, temperature=None, timeout=None, **kwargs):  # noqa: ARG002,ANN001
        import time as _time

        t0 = _time.perf_counter()
        self.calls += 1
        return LLMResponse(
            content=self.tool_call,
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=(_time.perf_counter() - t0) * 1000.0,
        )


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("add", "add two numbers", lambda a, b: a + b)
    registry.register("noop", "deterministic no-op", lambda: "done")
    return registry


def test_react_step_progression_and_tool_execution() -> None:
    state = AgentState(
        task_id="t1", pair_id="p1", split="dev", arm="A", run_idx=0, max_steps=5
    )
    registry = _make_registry()
    llm = FakeLLMClient('TOOL: add {"a": 2, "b": 3}', "The answer is 5")
    baseline = ArmABaseline(state, llm, registry, max_steps=5)

    result = baseline.run("Compute 2 + 3")

    assert result.final_answer == "The answer is 5"
    assert result.stop_reason == "final_answer"
    assert result.total_llm_calls == 2
    # One LLM step + one tool step recorded.
    modules = [rec.module for rec in result.trajectory]
    assert "llm" in modules and "tool" in modules
    assert result.trajectory[-1].module == "llm"  # final answer step


def test_max_step_termination() -> None:
    state = AgentState(
        task_id="t2", pair_id="p2", split="dev", arm="A", run_idx=0, max_steps=2
    )
    registry = _make_registry()
    llm = AlwaysToolLLMClient('TOOL: add {"a": 1, "b": 1}', "unreachable")
    baseline = ArmABaseline(state, llm, registry, max_steps=2)

    result = baseline.run("Keep calling tools")

    assert result.stop_reason == "max_steps"
    assert result.finished
    assert result.step == 2  # hit the step budget


def test_unknown_tool_is_captured_as_observation() -> None:
    state = AgentState(
        task_id="t3", pair_id="p3", split="dev", arm="A", run_idx=0, max_steps=5
    )
    registry = _make_registry()
    llm = FakeLLMClient('TOOL: nonexistent {"x": 1}', "done after error")
    baseline = ArmABaseline(state, llm, registry, max_steps=5)

    result = baseline.run("Call a missing tool")

    # The unknown-tool error becomes an observation, the loop continues, then
    # the LLM gives a final answer.
    assert result.final_answer == "done after error"
    assert result.stop_reason == "final_answer"
    tool_records = [rec for rec in result.trajectory if rec.module == "tool"]
    assert tool_records and not tool_records[0].tool_ok


def test_smoke_run_emits_schema_valid_jsonl() -> None:
    """End-to-end smoke run: write eval_arm_A_smoke.jsonl, 100% schema compliance."""
    state = AgentState(
        task_id="smoke-1", pair_id="smoke-pair-1", split="dev", arm="A", run_idx=0,
        max_steps=5,
    )
    registry = _make_registry()
    llm = FakeLLMClient('TOOL: add {"a": 4, "b": 6}', "The sum is 10")

    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as fh:
        path = fh.name
    try:
        runner = ArmARunner(trace_path=path)
        result = runner.run_task(
            task_id=state.task_id,
            pair_id=state.pair_id,
            split=state.split,
            arm=state.arm,
            run_idx=state.run_idx,
            initial_prompt="Compute 4 + 6",
            tool_registry=registry,
            llm_client=llm,
            max_steps=5,
        )
        assert result.final_answer == "The sum is 10"

        # Validate every written line against the 19-field schema.
        with open(path, "r", encoding="utf-8") as fh:
            lines = [line for line in fh.read().splitlines() if line.strip()]

        assert len(lines) == len(result.trajectory)
        for line in lines:
            record = TraceLogger.validate_line(line)
            assert record.task_id == "smoke-1"
            assert record.arm == "A"
            assert record.split == "dev"
            assert record.commit_hash  # git provenance pinned
            assert isinstance(record.confidence, float)
            assert 0.0 <= record.confidence <= 1.0
    finally:
        os.remove(path)


def test_react_format_parsing_and_markdown_fences() -> None:
    """Verify Action/Action Input extraction with code blocks and extra whitespace."""
    state = AgentState(
        task_id="t-react", pair_id="p-react", split="dev", arm="A", run_idx=0, max_steps=5
    )
    registry = _make_registry()
    react_tool_call = (
        "Thought: I should add 10 and 20.\n"
        "Action: add\n"
        "Action Input: ```json\n"
        '{"a": 10, "b": 20}\n'
        "```"
    )
    llm = FakeLLMClient(react_tool_call, "Thought: Finished.\nFinal Answer: 30")
    baseline = ArmABaseline(state, llm, registry, max_steps=5)

    result = baseline.run("Add 10 and 20")
    assert result.finished
    assert result.stop_reason == "final_answer"
    assert "Final Answer: 30" in result.final_answer
    tool_rec = [r for r in result.trajectory if r.module == "tool"]
    assert len(tool_rec) == 1
    assert tool_rec[0].decision == "add"
    assert tool_rec[0].tool_ok


def test_react_fallback_regex_tool_call() -> None:
    """Verify tool call extraction when model omits Action/Action Input prefixes."""
    state = AgentState(
        task_id="t-fallback", pair_id="p-fallback", split="dev", arm="A", run_idx=0, max_steps=5
    )
    registry = _make_registry()
    natural_tool_call = (
        "I will calculate this by calling add(a=15, b=25) to find the total."
    )
    llm = FakeLLMClient(natural_tool_call, "Final Answer: The total is 40.")
    baseline = ArmABaseline(state, llm, registry, max_steps=5)

    result = baseline.run("Calculate 15 + 25")
    assert result.finished
    assert result.stop_reason == "final_answer"
    tool_rec = [r for r in result.trajectory if r.module == "tool"]
    assert len(tool_rec) == 1
    assert tool_rec[0].decision == "add"
    assert tool_rec[0].tool_ok


__all__ = ["FakeLLMClient", "AlwaysToolLLMClient"]

