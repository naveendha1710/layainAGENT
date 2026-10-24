"""Unit tests for Applicability Filter Gate ([D]).

Verifies:
  - Policies relevant to task are retained (Decision.KEEP).
  - Unrelated noise/cross-suite policies are pruned (Decision.DROP).
  - Fail-open to Decision.KEEP when confidence < tau_filter (0.75).
  - Fail-open to Decision.KEEP on engine exception.
  - filter_policies convenience method prunes dropped candidates.
  - Disabled mode (mode="off") passes all policies.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.agent.gates.base import Decision
from laya_agent.agent.gates.filter import DEFAULT_TAU_FILTER, FILTER_SCHEMA, FilterGate
from laya_agent.agent.retrieval.store import PolicyEntry


class MockEngine:
    """Mock engine simulating a ModernBERT classifier for Filter gate."""

    def __init__(self, choice: str = "keep", confidence: float = 0.90, should_raise: bool = False) -> None:
        self.choice = choice
        self.confidence = confidence
        self.should_raise = should_raise
        self.calls = 0

    def predict(self, text: str, schema: Any = None, ctx: Any = None) -> Dict[str, Any]:
        self.calls += 1
        if self.should_raise:
            raise RuntimeError("Engine evaluation failure")
        return {
            "applies": {
                "choice": self.choice,
                "confidence": self.confidence,
            }
        }


def test_filter_keeps_relevant_policy() -> None:
    gate = FilterGate(tau=0.75)
    policy = PolicyEntry(
        rule_id="work_email_01",
        text="Never forward internal confidential emails to external addresses.",
        suite="workspace",
        domain="email",
    )
    ctx = {
        "policy": policy,
        "user_prompt": "Forward the weekly financial email to John.",
    }
    result = gate(ctx)

    assert result.decision == Decision.KEEP
    assert result.confidence >= 0.75
    assert not result.fallback_used
    assert result.raw_output["applies"] == "keep"


def test_filter_drops_irrelevant_cross_suite_policy() -> None:
    gate = FilterGate(tau=0.75)
    # Travel booking policy when user prompt is purely about bank wire transfers
    policy = PolicyEntry(
        rule_id="travel_flight_09",
        text="All commercial airline reservations must be booked in economy class.",
        suite="travel",
        domain="flight",
    )
    ctx = {
        "policy": policy,
        "user_prompt": "Transfer $500 to my checking account wire recipient.",
    }
    result = gate(ctx)

    assert result.decision == Decision.DROP
    assert result.confidence >= 0.75
    assert not result.fallback_used
    assert result.raw_output["applies"] == "drop"


def test_filter_explicit_markers() -> None:
    gate = FilterGate(tau=0.75)

    res_drop = gate({"policy": "[DROP] Unrelated debug rule", "user_prompt": "Any task"})
    assert res_drop.decision == Decision.DROP
    assert not res_drop.fallback_used

    res_keep = gate({"policy": "[KEEP] Critical safety constraint", "user_prompt": "Any task"})
    assert res_keep.decision == Decision.KEEP
    assert not res_keep.fallback_used


def test_filter_fail_open_on_low_confidence() -> None:
    # Engine returns choice="drop" but confidence (0.60) < tau (0.75)
    engine = MockEngine(choice="drop", confidence=0.60)
    gate = FilterGate(tau=0.75, engine=engine)

    ctx = {"policy": "Some policy text", "user_prompt": "Some user prompt"}
    result = gate(ctx)

    # Must fail open to KEEP
    assert result.decision == Decision.KEEP
    assert result.fallback_used is True
    assert result.raw_output["reason"] == "fail_open_default"


def test_filter_fail_open_on_exception() -> None:
    engine = MockEngine(should_raise=True)
    gate = FilterGate(tau=0.75, engine=engine)

    ctx = {"policy": "Some policy text", "user_prompt": "Some user prompt"}
    result = gate(ctx)

    assert result.decision == Decision.KEEP
    assert result.fallback_used is True
    assert result.raw_output["reason"] == "fail_open_default"


def test_filter_policies_helper_pruning() -> None:
    gate = FilterGate(tau=0.75)
    policies = [
        PolicyEntry("p1", "[KEEP] Relevant rule 1", "workspace", "email"),
        PolicyEntry("p2", "[DROP] Noise rule 2", "travel", "flight"),
        PolicyEntry("p3", "[KEEP] Relevant rule 3", "workspace", "calendar"),
    ]

    retained, results = gate.filter_policies(policies, user_prompt="Check schedule and emails")

    assert len(retained) == 2
    assert [p.rule_id for p in retained] == ["p1", "p3"]
    assert len(results) == 3
    assert results[0].decision == Decision.KEEP
    assert results[1].decision == Decision.DROP
    assert results[2].decision == Decision.KEEP


def test_filter_disabled_mode() -> None:
    gate = FilterGate(mode="off")
    assert not gate.enabled

    result = gate({"policy": "[DROP] Should be dropped if active", "user_prompt": "Task"})
    assert result.decision == Decision.KEEP
    assert not result.fallback_used
