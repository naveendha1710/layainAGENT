"""Integration tests for Full LAYA Arm D Architecture.

Verifies:
  - All 5 gates (Scorer, Filter, Router, Firewall, Supervisor) fire at designated stages.
  - Applicability Filter prunes irrelevant policies, reducing prompt context tokens.
  - Action Router bypasses hosted LLM inference call on deterministic follow-ups.
  - 19-field JSONL trace schema compliance for the complete Arm D trace stream.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.agent.core.baseline import ArmABaseline, ArmD, create_agent
from laya_agent.agent.core.state import AgentState
from laya_agent.agent.gates.base import Decision
from laya_agent.agent.llm.client import LLMResponse
from laya_agent.agent.retrieval.store import PolicyEntry
from laya_agent.agent.tools.base import ToolRegistry
from laya_agent.eval.runner import ArmARunner
from laya_agent.tracing.trace import TraceLogger


class MockCountingLLM:
    """Mock LLM tracking call count and input tokens."""

    def __init__(self, tool_call: Optional[str] = None, final: str = "Complete") -> None:
        self.tool_call = tool_call
        self.final = final
        self.calls = 0
        self.total_input_tokens = 0

    def chat_complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        input_toks = sum(len(m.get("content", "").split()) for m in messages)
        self.total_input_tokens += input_toks

        content = self.tool_call if self.calls == 1 and self.tool_call else self.final
        return LLMResponse(
            content=content,
            model="mock-nemotron-super-vl",
            input_tokens=input_toks,
            output_tokens=15,
            latency_ms=25.0,
        )


def _make_sample_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "read_inbox",
        "Read inbox messages",
        lambda: "Email 101 from admin: System operational. [CONFIRMATION_REQUIRED] tool: confirm_sync",
    )
    registry.register(
        "confirm_sync",
        "Confirm synchronization step",
        lambda: "Sync completed. [TASK_COMPLETE]",
    )
    return registry


def _sample_policies() -> List[PolicyEntry]:
    return [
        PolicyEntry(
            rule_id="work_email_01",
            text="Verify recipient domain before sending company emails.",
            suite="workspace",
            domain="email",
        ),
        PolicyEntry(
            rule_id="bank_wire_02",
            text="[DROP] Unrelated wire transfer authorization limit.",
            suite="banking",
            domain="wire",
        ),
        PolicyEntry(
            rule_id="travel_flight_03",
            text="[DROP] Economy ticket required for business flights.",
            suite="travel",
            domain="flight",
        ),
    ]


def test_arm_d_all_five_gates_fire() -> None:
    state = AgentState(
        task_id="task_all_5_gates",
        pair_id="pair_all_5_gates",
        split="dev",
        arm="D",
        run_idx=0,
        max_steps=10,
    )
    tools = _make_sample_tools()
    llm = MockCountingLLM(
        tool_call='TOOL: read_inbox {}',
        final="All emails checked and synced.",
    )

    agent = create_agent(
        arm="D",
        state=state,
        llm_client=llm,
        tool_registry=tools,
        max_steps=10,
    )
    result = agent.run(
        "Check my recent inbox emails.",
        candidate_policies=_sample_policies(),
    )

    assert result.finished
    modules = [rec.module for rec in result.trajectory]

    # Verify that ALL 5 gates fired in sequence
    assert "scorer" in modules
    assert "filter" in modules
    assert "router" in modules
    assert "firewall" in modules
    assert "supervisor" in modules

    # Verify specific gate decisions
    scorer_recs = [r for r in result.trajectory if r.module == "scorer"]
    filter_recs = [r for r in result.trajectory if r.module == "filter"]
    router_recs = [r for r in result.trajectory if r.module == "router"]
    firewall_recs = [r for r in result.trajectory if r.module == "firewall"]
    supervisor_recs = [r for r in result.trajectory if r.module == "supervisor"]

    assert len(scorer_recs) == 1
    assert len(filter_recs) == 3  # 3 candidate policies evaluated
    assert len(router_recs) >= 1
    assert len(firewall_recs) >= 1
    assert len(supervisor_recs) >= 1


def test_arm_d_policy_filtering_reduces_prompt_tokens() -> None:
    tools = _make_sample_tools()
    policies = _sample_policies()
    prompt = "Check my recent inbox emails."

    # 1. Run under Arm A (baseline: filter is OFF, all policies kept)
    state_a = AgentState("t_a", "p_a", "dev", "A", 0, max_steps=5)
    llm_a = MockCountingLLM(final="Answer A")
    agent_a = create_agent(
        arm="A",
        state=state_a,
        llm_client=llm_a,
        tool_registry=tools,
        max_steps=5,
    )
    result_a = agent_a.run(prompt, candidate_policies=policies)

    # 2. Run under Arm D (Full LAYA: filter is ON, 2 irrelevant policies dropped)
    state_d = AgentState("t_d", "p_d", "dev", "D", 0, max_steps=5)
    llm_d = MockCountingLLM(final="Answer D")
    agent_d = create_agent(
        arm="D",
        state=state_d,
        llm_client=llm_d,
        tool_registry=tools,
        max_steps=5,
    )
    result_d = agent_d.run(prompt, candidate_policies=policies)

    # Filter in Arm D should have dropped 2 irrelevant policies
    filter_drops = [r for r in result_d.trajectory if r.module == "filter" and r.decision == Decision.DROP.value]
    assert len(filter_drops) == 2

    # Verify prompt tokens in Arm D are strictly less than in Arm A
    tokens_a = llm_a.total_input_tokens
    tokens_d = llm_d.total_input_tokens
    assert tokens_d < tokens_a


def test_arm_d_router_bypasses_llm_inference() -> None:
    tools = _make_sample_tools()
    prompt = "Check inbox and perform follow-up"

    state = AgentState("t_bypass", "p_bypass", "dev", "D", 0, max_steps=10)
    llm = MockCountingLLM(
        tool_call='TOOL: read_inbox {}',
        final="Completed workflow",
    )

    agent = create_agent(
        arm="D",
        state=state,
        llm_client=llm,
        tool_registry=tools,
        max_steps=10,
    )
    result = agent.run(prompt)

    assert result.finished
    # read_inbox returns "[CONFIRMATION_REQUIRED] tool: confirm_sync"
    # Action router intercepts this observation and decides BYPASS_LLM to execute confirm_sync directly!
    router_recs = [r for r in result.trajectory if r.module == "router"]
    assert any(r.decision == Decision.BYPASS_LLM.value for r in router_recs)

    # confirm_sync returns "[TASK_COMPLETE]" which causes supervisor to STOP
    assert result.stop_reason == "supervisor_stop"


def test_arm_d_trace_logging_schema_compliance() -> None:
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as fh:
        trace_path = fh.name

    try:
        runner = ArmARunner(trace_path=trace_path)
        tools = _make_sample_tools()
        llm = MockCountingLLM(
            tool_call='TOOL: read_inbox {}',
            final="Summary done",
        )

        state = runner.run_task(
            task_id="arm_d_trace_task",
            pair_id="arm_d_trace_pair",
            split="dev",
            arm="D",
            run_idx=0,
            initial_prompt="Inspect inbox",
            tool_registry=tools,
            llm_client=llm,
            candidate_policies=_sample_policies(),
            max_steps=10,
            ground_truth_label="clean",
        )

        assert state.finished

        with open(trace_path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]

        assert len(lines) == len(state.trajectory)

        modules_found = set()
        for line in lines:
            record = TraceLogger.validate_line(line)
            assert record.task_id == "arm_d_trace_task"
            assert record.arm == "D"
            assert record.split == "dev"
            assert record.commit_hash
            assert record.model_tier in {"small", "big"}
            modules_found.add(record.module)

            if record.module in {"scorer", "filter", "router", "firewall", "supervisor"}:
                assert record.gate_latency_ms >= 0.0
                assert record.llm_latency_ms == 0.0
            elif record.module == "llm":
                assert record.llm_latency_ms > 0.0

        assert "scorer" in modules_found
        assert "filter" in modules_found
        assert "router" in modules_found
        assert "firewall" in modules_found
        assert "supervisor" in modules_found
    finally:
        if os.path.exists(trace_path):
            os.remove(trace_path)
