"""Unit tests for the Supervisor gate."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.agent.core.state import StepRecord
from laya_agent.agent.gates.base import Decision
from laya_agent.agent.gates.supervisor import SUPERVISOR_SCHEMA, SupervisorGate


def test_supervisor_continue_on_normal_progress() -> None:
    gate = SupervisorGate(tau=0.70, mode="laya")

    records = [
        StepRecord(step=0, module="llm", decision="nemotron", confidence=1.0, fallback_used=False, latency_ms=10.0),
        StepRecord(step=1, module="tool", decision="get_balance", confidence=1.0, fallback_used=False, latency_ms=5.0, tool="get_balance"),
    ]
    ctx = {"trajectory": records, "recent_observation": "Balance: $1,200"}
    result = gate(ctx)

    assert result.decision == Decision.CONTINUE
    assert not result.fallback_used
    assert result.confidence >= 0.70


def test_supervisor_nudge_on_repeated_loop() -> None:
    gate = SupervisorGate(tau=0.70, mode="laya")

    # 3 identical consecutive tool calls
    records = [
        StepRecord(step=0, module="tool", decision="search", confidence=1.0, fallback_used=False, latency_ms=5.0, tool="search"),
        StepRecord(step=1, module="tool", decision="search", confidence=1.0, fallback_used=False, latency_ms=5.0, tool="search"),
        StepRecord(step=2, module="tool", decision="search", confidence=1.0, fallback_used=False, latency_ms=5.0, tool="search"),
    ]
    ctx = {"trajectory": records, "recent_observation": "No results"}
    result = gate(ctx)

    assert result.decision == Decision.NUDGE
    assert result.confidence >= 0.90
    assert result.raw_output.get("reason") == "detected_repeated_tool_loop"


def test_supervisor_stop_on_goal_completion() -> None:
    gate = SupervisorGate(tau=0.70, mode="laya")

    records = [
        StepRecord(step=0, module="tool", decision="book_flight", confidence=1.0, fallback_used=False, latency_ms=5.0, tool="book_flight"),
    ]
    ctx = {
        "trajectory": records,
        "recent_observation": "Flight reservation confirmed. [TASK_COMPLETE]",
    }
    result = gate(ctx)

    assert result.decision == Decision.STOP
    assert result.confidence >= 0.90


def test_supervisor_fail_open_on_low_confidence() -> None:
    class LowConfidenceEngine:
        def predict(self, *args, **kwargs):
            return {
                "status": {
                    "choice": "nudge",
                    "confidence": 0.40,  # Below tau_supervisor (0.70)
                }
            }

    gate = SupervisorGate(tau=0.70, mode="laya", engine=LowConfidenceEngine())
    ctx = {"trajectory": [], "recent_observation": "something"}

    result = gate(ctx)
    # Fails open to CONTINUE with fallback_used=True
    assert result.decision == Decision.CONTINUE
    assert result.fallback_used is True


def test_supervisor_fail_open_on_exception() -> None:
    class FailingEngine:
        def predict(self, *args, **kwargs):
            raise ConnectionError("Network dropped during inference")

    gate = SupervisorGate(tau=0.70, mode="laya", engine=FailingEngine())
    ctx = {"trajectory": [], "recent_observation": "something"}

    result = gate(ctx)
    assert result.decision == Decision.CONTINUE
    assert result.fallback_used is True
