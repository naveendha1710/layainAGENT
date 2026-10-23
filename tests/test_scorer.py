"""Unit tests for the Task Difficulty & Risk Scorer Gate ([C])."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.agent.gates.base import Decision
from laya_agent.agent.gates.scorer import (
    DEFAULT_BIG_MAX_STEPS,
    DEFAULT_SMALL_MAX_STEPS,
    SCORER_SCHEMA,
    ScorerGate,
)


def test_scorer_simple_prompt_classifies_small() -> None:
    gate = ScorerGate(tau=0.70, mode="laya")
    ctx = {"initial_prompt": "Find file readme.txt"}
    result = gate(ctx)

    assert result.decision == Decision.SMALL
    assert result.confidence >= 0.70
    assert not result.fallback_used
    assert result.raw_output["model_tier"] == "small"
    assert result.raw_output["max_steps"] == DEFAULT_SMALL_MAX_STEPS


def test_scorer_complex_prompt_classifies_big() -> None:
    gate = ScorerGate(tau=0.70, mode="laya")
    ctx = {
        "initial_prompt": (
            "Transfer $15,000 to vendor IBAN account and coordinate wire "
            "confirmation with external auditor."
        )
    }
    result = gate(ctx)

    assert result.decision == Decision.BIG
    assert result.confidence >= 0.70
    assert not result.fallback_used
    assert result.raw_output["model_tier"] == "big"
    assert result.raw_output["max_steps"] == DEFAULT_BIG_MAX_STEPS


def test_scorer_fail_open_on_low_confidence() -> None:
    class LowConfidenceEngine:
        def predict(self, *args, **kwargs):
            return {
                "model_tier": {
                    "choice": "small",
                    "confidence": 0.45,  # Below tau_scorer (0.70)
                }
            }

    gate = ScorerGate(tau=0.70, mode="laya", engine=LowConfidenceEngine())
    ctx = {"initial_prompt": "Ambiguous task request"}
    result = gate(ctx)

    # Fail open to BIG tier with full budget
    assert result.decision == Decision.BIG
    assert result.fallback_used is True
    assert result.raw_output["model_tier"] == "big"
    assert result.raw_output["max_steps"] == DEFAULT_BIG_MAX_STEPS


def test_scorer_fail_open_on_exception() -> None:
    class CrashingEngine:
        def predict(self, *args, **kwargs):
            raise TimeoutError("Model inference timeout")

    gate = ScorerGate(tau=0.70, mode="laya", engine=CrashingEngine())
    ctx = {"initial_prompt": "Any task"}
    result = gate(ctx)

    assert result.decision == Decision.BIG
    assert result.fallback_used is True
    assert result.raw_output["model_tier"] == "big"


def test_scorer_always_small_mode() -> None:
    gate = ScorerGate(mode="always_small")

    # Even a complex transfer prompt must be routed to small
    ctx = {"initial_prompt": "Transfer $1,000,000 international wire"}
    result = gate(ctx)

    assert result.decision == Decision.SMALL
    assert not result.fallback_used
    assert result.raw_output["model_tier"] == "small"
    assert result.raw_output["max_steps"] == DEFAULT_SMALL_MAX_STEPS


def test_scorer_random_mode() -> None:
    gate_1 = ScorerGate(mode="random", seed=42)
    gate_2 = ScorerGate(mode="random", seed=42)

    ctx = {"initial_prompt": "Some task prompt"}
    res_1 = gate_1(ctx)
    res_2 = gate_2(ctx)

    assert res_1.decision in {Decision.SMALL, Decision.BIG}
    # Deterministic given same seed
    assert res_1.decision == res_2.decision
    assert not res_1.fallback_used


def test_scorer_disabled_mode() -> None:
    gate = ScorerGate(mode="off")
    assert not gate.enabled

    ctx = {"initial_prompt": "Find file notes.txt"}
    result = gate(ctx)

    assert result.decision in {Decision.BIG, Decision.CALL_LLM}
    assert result.latency_ms == 0.0
    assert not result.fallback_used
