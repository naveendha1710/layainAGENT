"""Unit tests for the evaluation metrics engine (laya_agent/analysis/metrics.py).

Verifies:
  - Metric calculations (ASR, FBR, Benign Utility, Utility Under Attack).
  - Gate precision, recall, and F1 evaluations against ground truth labels.
  - Token and dollar cost calculation based on dual-tier model pricing.
  - Latency percentiles (p50, p95, p99) for gates and LLM steps.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.analysis.metrics import ArmMetrics, MetricsEngine, RunMetrics


def _make_sample_step_records(
    arm: str = "D",
    pair_id: str = "task_01",
    task_id: str = "task_01_benign",
    is_injection: bool = False,
    firewall_redacted: bool = False,
    input_tokens: int = 100,
    output_tokens: int = 20,
    model_tier: str = "big",
) -> List[Dict[str, Any]]:
    records = []

    # 1. Scorer step
    records.append({
        "task_id": task_id,
        "pair_id": pair_id,
        "split": "dev",
        "arm": arm,
        "run_idx": 0,
        "step": 0,
        "module": "scorer",
        "model_tier": model_tier,
        "decision": model_tier,
        "confidence": 0.95,
        "fallback_used": False,
        "gate_latency_ms": 5.0,
        "llm_latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool": None,
        "tool_ok": None,
        "ground_truth_label": None,
        "commit_hash": "abc1234",
    })

    # 2. LLM step
    records.append({
        "task_id": task_id,
        "pair_id": pair_id,
        "split": "dev",
        "arm": arm,
        "run_idx": 0,
        "step": 1,
        "module": "llm",
        "model_tier": model_tier,
        "decision": "mock-model",
        "confidence": 1.0,
        "fallback_used": False,
        "gate_latency_ms": 0.0,
        "llm_latency_ms": 25.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool": None,
        "tool_ok": None,
        "ground_truth_label": None,
        "commit_hash": "abc1234",
    })

    # 3. Tool step
    records.append({
        "task_id": task_id,
        "pair_id": pair_id,
        "split": "dev",
        "arm": arm,
        "run_idx": 0,
        "step": 2,
        "module": "tool",
        "model_tier": model_tier,
        "decision": "read_inbox",
        "confidence": 1.0,
        "fallback_used": False,
        "gate_latency_ms": 0.0,
        "llm_latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool": "read_inbox",
        "tool_ok": True,
        "ground_truth_label": "injection" if is_injection else "clean",
        "commit_hash": "abc1234",
    })

    # 4. Firewall step
    firewall_dec = "redact" if firewall_redacted else "pass"
    records.append({
        "task_id": task_id,
        "pair_id": pair_id,
        "split": "dev",
        "arm": arm,
        "run_idx": 0,
        "step": 3,
        "module": "firewall",
        "model_tier": model_tier,
        "decision": firewall_dec,
        "confidence": 0.92,
        "fallback_used": False,
        "gate_latency_ms": 8.0,
        "llm_latency_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool": "read_inbox",
        "tool_ok": True,
        "ground_truth_label": "injection" if is_injection else "clean",
        "commit_hash": "abc1234",
    })

    return records


def test_metrics_analyze_benign_run() -> None:
    engine = MetricsEngine()
    recs = _make_sample_step_records(
        task_id="task_01_benign",
        is_injection=False,
        firewall_redacted=False,
    )
    run_m = engine.analyze_run(recs)

    assert run_m.is_benign is True
    assert run_m.is_injection is False
    assert run_m.condition == "benign"
    assert run_m.utility == 1.0
    assert run_m.false_block == 0.0
    assert run_m.attack_success is None


def test_metrics_analyze_benign_false_block() -> None:
    engine = MetricsEngine()
    # Benign run where firewall falsely intervened (redact)
    recs = _make_sample_step_records(
        task_id="task_01_benign",
        is_injection=False,
        firewall_redacted=True,
    )
    run_m = engine.analyze_run(recs)

    assert run_m.is_benign is True
    assert run_m.false_block == 1.0


def test_metrics_analyze_injection_runs_blocked_vs_penetrated() -> None:
    engine = MetricsEngine()

    # 1. Attack successfully blocked by firewall
    recs_blocked = _make_sample_step_records(
        task_id="task_01_inj_1",
        is_injection=True,
        firewall_redacted=True,
    )
    run_blocked = engine.analyze_run(recs_blocked)
    assert run_blocked.is_injection is True
    assert run_blocked.attack_success == 0.0  # blocked

    # 2. Attack penetrated unredacted
    recs_penetrated = _make_sample_step_records(
        task_id="task_01_inj_2",
        is_injection=True,
        firewall_redacted=False,
    )
    run_penetrated = engine.analyze_run(recs_penetrated)
    assert run_penetrated.is_injection is True
    assert run_penetrated.attack_success == 1.0  # penetrated


def test_metrics_cost_calculation() -> None:
    pricing = {
        "big": {"input_per_1m": 1.0, "output_per_1m": 2.0},
        "small": {"input_per_1m": 0.1, "output_per_1m": 0.2},
    }
    engine = MetricsEngine(pricing=pricing)

    # 1,000,000 input tokens + 500,000 output tokens on Big Tier
    # Cost = 1.0 * 1.0 + 0.5 * 2.0 = $2.00
    cost_big = engine.compute_cost(1_000_000, 500_000, model_tier="big")
    assert abs(cost_big - 2.0) < 1e-6

    # Same tokens on Small Tier: 1.0 * 0.1 + 0.5 * 0.2 = $0.20
    cost_small = engine.compute_cost(1_000_000, 500_000, model_tier="small")
    assert abs(cost_small - 0.20) < 1e-6


def test_metrics_trace_aggregation_and_latency_percentiles() -> None:
    engine = MetricsEngine()
    records = []

    # Triplet for task_01: 1 benign + 2 injections
    # Benign: utility 1.0, no false block
    records.extend(_make_sample_step_records("D", "pair_01", "pair_01_benign", False, False))
    # Inj 1: blocked by firewall (ASR=0.0)
    records.extend(_make_sample_step_records("D", "pair_01", "pair_01_inj_1", True, True))
    # Inj 2: blocked by firewall (ASR=0.0)
    records.extend(_make_sample_step_records("D", "pair_01", "pair_01_inj_2", True, True))

    arm_metrics = engine.analyze_trace(records)

    assert "D" in arm_metrics
    m_d = arm_metrics["D"]

    assert m_d.n_tasks == 1
    assert m_d.n_runs == 3
    assert m_d.benign_utility == 1.0
    assert m_d.attack_utility == 1.0
    assert m_d.asr == 0.0
    assert m_d.fbr == 0.0

    # Latencies
    assert m_d.gate_latency_p50 > 0.0
    assert m_d.gate_latency_p95 >= m_d.gate_latency_p50
    assert m_d.llm_latency_p50 == 25.0

    # Gate breakdown
    assert "firewall" in m_d.gate_metrics
    fw = m_d.gate_metrics["firewall"]
    assert fw.total_evaluations == 3
    assert fw.active_firings == 2  # 2 redactions
    assert fw.true_positives == 2
    assert fw.false_positives == 0
    assert fw.precision == 1.0
    assert fw.recall == 1.0
    assert fw.f1 == 1.0
