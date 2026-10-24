"""Unit tests for Action Router Gate ([D]).

Verifies:
  - Novel reasoning and complex steps trigger Decision.CALL_LLM.
  - Predictable routine follow-ups trigger Decision.BYPASS_LLM with tool action.
  - Fail-open to Decision.CALL_LLM when confidence < tau_router (0.80).
  - Fail-open to Decision.CALL_LLM on engine exception.
  - Disabled mode (mode="off") always returns Decision.CALL_LLM.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.agent.gates.base import Decision
from laya_agent.agent.gates.router import DEFAULT_TAU_ROUTER, ROUTER_SCHEMA, RouterGate


class MockRouterEngine:
    """Mock engine simulating a ModernBERT classifier for Action Router gate."""

    def __init__(
        self,
        choice: str = "call_llm",
        confidence: float = 0.90,
        tool_name: str = "noop",
        tool_args: Dict[str, Any] = None,
        should_raise: bool = False,
    ) -> None:
        self.choice = choice
        self.confidence = confidence
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.should_raise = should_raise
        self.calls = 0

    def predict(self, text: str, schema: Any = None, ctx: Any = None) -> Dict[str, Any]:
        self.calls += 1
        if self.should_raise:
            raise RuntimeError("Engine evaluation failure")
        return {
            "action": {
                "choice": self.choice,
                "confidence": self.confidence,
                "tool_name": self.tool_name,
                "tool_args": self.tool_args,
            }
        }


def test_router_call_llm_for_novel_reasoning() -> None:
    gate = RouterGate(tau=0.80)
    ctx = {
        "messages": [{"role": "user", "content": "Analyze the financial portfolio and find discrepancies."}],
        "trajectory": [],
    }
    result = gate(ctx)

    assert result.decision == Decision.CALL_LLM
    assert result.confidence >= 0.80
    assert not result.fallback_used
    assert result.raw_output["action"] == "call_llm"


def test_router_bypass_llm_for_routine_follow_up() -> None:
    gate = RouterGate(tau=0.80)
    ctx = {
        "messages": [
            {"role": "user", "content": "Transfer $100 to Alice"},
            {"role": "assistant", "content": "TOOL: prepare_transfer {\"amount\": 100}"},
            {"role": "user", "content": "Transfer staged. [CONFIRMATION_REQUIRED] tool: confirm_transfer"},
        ],
        "trajectory": [],
    }
    result = gate(ctx)

    assert result.decision == Decision.BYPASS_LLM
    assert result.confidence >= 0.80
    assert not result.fallback_used
    assert result.raw_output["action"] == "bypass_llm"
    assert result.raw_output["tool_name"] == "confirm_transfer"


def test_router_explicit_deterministic_action() -> None:
    gate = RouterGate(tau=0.80)
    ctx = {
        "force_bypass": True,
        "deterministic_tool": "send_receipt",
        "deterministic_args": {"user_id": 42},
    }
    result = gate(ctx)

    assert result.decision == Decision.BYPASS_LLM
    assert not result.fallback_used
    assert result.raw_output["tool_name"] == "send_receipt"
    assert result.raw_output["tool_args"] == {"user_id": 42}


def test_router_fail_open_on_low_confidence() -> None:
    # Engine selects bypass_llm with low confidence (0.65 < 0.80)
    engine = MockRouterEngine(choice="bypass_llm", confidence=0.65)
    gate = RouterGate(tau=0.80, engine=engine)

    ctx = {"messages": [{"role": "user", "content": "Run action"}]}
    result = gate(ctx)

    # Must fail open to CALL_LLM
    assert result.decision == Decision.CALL_LLM
    assert result.fallback_used is True
    assert result.raw_output["reason"] == "fail_open_default"


def test_router_fail_open_on_exception() -> None:
    engine = MockRouterEngine(should_raise=True)
    gate = RouterGate(tau=0.80, engine=engine)

    ctx = {"messages": [{"role": "user", "content": "Run action"}]}
    result = gate(ctx)

    assert result.decision == Decision.CALL_LLM
    assert result.fallback_used is True
    assert result.raw_output["reason"] == "fail_open_default"


def test_router_disabled_mode() -> None:
    gate = RouterGate(mode="off")
    assert not gate.enabled

    ctx = {"force_bypass": True}
    result = gate(ctx)

    assert result.decision == Decision.CALL_LLM
    assert not result.fallback_used
