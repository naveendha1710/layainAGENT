"""Unit tests for paired cluster bootstrap engine (laya_agent/analysis/bootstrap.py).

Verifies:
  - 95% Confidence Intervals contain or bound the point estimate properly.
  - Strict determinism under fixed seed.
  - Paired delta computation against baseline Arm A.
  - Two-sided empirical p-value computation.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.analysis.bootstrap import PairedBootstrapEngine
from laya_agent.analysis.metrics import ArmMetrics, TaskMetrics


def _build_synthetic_arm_metrics() -> dict[str, ArmMetrics]:
    """Build synthetic task-level metrics for Arm A and Arm D across 10 tasks."""
    arm_a = ArmMetrics(arm="A", n_tasks=10)
    arm_d = ArmMetrics(arm="D", n_tasks=10)

    for i in range(10):
        pid = f"pair_{i:02d}"
        # Arm A has high ASR (attacks succeed), high cost
        tm_a = TaskMetrics(
            arm="A",
            pair_id=pid,
            benign_utility=1.0,
            attack_utility=0.90,
            overall_utility=0.93,
            asr=0.85,  # 85% ASR
            fbr=0.0,
            mean_input_tokens=1000.0,
            mean_output_tokens=150.0,
            mean_llm_calls=3.0,
            mean_cost_usd=0.0050,
        )
        arm_a.task_metrics[pid] = tm_a

        # Arm D defends against injections (low ASR), lower cost
        tm_d = TaskMetrics(
            arm="D",
            pair_id=pid,
            benign_utility=1.0,
            attack_utility=0.90,
            overall_utility=0.93,
            asr=0.05,  # 5% ASR (defended)
            fbr=0.0,
            mean_input_tokens=600.0,  # tokens pruned
            mean_output_tokens=100.0,
            mean_llm_calls=2.0,       # router bypassed
            mean_cost_usd=0.0020,     # cost reduced
        )
        arm_d.task_metrics[pid] = tm_d

    return {"A": arm_a, "D": arm_d}


def test_bootstrap_ci_bounds() -> None:
    data = _build_synthetic_arm_metrics()
    engine = PairedBootstrapEngine(n_boot=1000, seed=42, baseline_arm="A")
    results = engine.run_bootstrap(data)

    assert "A" in results
    assert "D" in results

    res_d = results["D"]
    asr_ci = res_d.metrics["asr"]

    # Point estimate must be between (or equal to) CI bounds
    assert asr_ci.ci_low <= asr_ci.point_estimate <= asr_ci.ci_high
    assert asr_ci.point_estimate == pytest.approx(0.05, abs=1e-4)


def test_bootstrap_determinism_under_fixed_seed() -> None:
    data = _build_synthetic_arm_metrics()

    engine_1 = PairedBootstrapEngine(n_boot=1000, seed=12345, baseline_arm="A")
    results_1 = engine_1.run_bootstrap(data)

    engine_2 = PairedBootstrapEngine(n_boot=1000, seed=12345, baseline_arm="A")
    results_2 = engine_2.run_bootstrap(data)

    for metric_name in ["asr", "mean_cost_usd", "benign_utility"]:
        ci1 = results_1["D"].metrics[metric_name]
        ci2 = results_2["D"].metrics[metric_name]
        assert ci1.ci_low == ci2.ci_low
        assert ci1.ci_high == ci2.ci_high
        assert ci1.delta_ci_low == ci2.delta_ci_low
        assert ci1.delta_ci_high == ci2.delta_ci_high
        assert ci1.p_value == ci2.p_value


def test_bootstrap_paired_delta_and_p_value() -> None:
    data = _build_synthetic_arm_metrics()
    engine = PairedBootstrapEngine(n_boot=2000, seed=42, baseline_arm="A")
    results = engine.run_bootstrap(data)

    res_d = results["D"]
    asr_ci = res_d.metrics["asr"]

    # Delta for ASR = Arm D (0.05) - Arm A (0.85) = -0.80
    assert asr_ci.delta_to_baseline == pytest.approx(-0.80, abs=1e-4)
    assert asr_ci.delta_ci_high < 0.0  # Significant reduction in ASR
    assert asr_ci.p_value is not None
    assert asr_ci.p_value < 0.05  # Statistically significant difference

    # Delta for cost: Arm D (0.0020) - Arm A (0.0050) = -0.0030
    cost_ci = res_d.metrics["mean_cost_usd"]
    assert cost_ci.delta_to_baseline == pytest.approx(-0.0030, abs=1e-4)
    assert cost_ci.delta_ci_high < 0.0
