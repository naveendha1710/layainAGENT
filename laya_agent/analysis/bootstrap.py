"""Paired cluster bootstrap engine for user-task triplets.

Resamples clusters defined by `pair_id` (1 benign + 2 matched injections)
to preserve intra-task correlations and computes:
  * 95% Confidence Intervals (point estimate, ci_low, ci_high) for all metrics.
  * Paired difference distributions: Delta = Arm X - Baseline Arm A.
  * Two-sided empirical p-values for hypothesis testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from laya_agent.analysis.metrics import ArmMetrics, TaskMetrics


@dataclass
class BootstrapCI:
    """95% Confidence Interval and paired comparison metrics."""

    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    std_err: float
    delta_to_baseline: Optional[float] = None
    delta_ci_low: Optional[float] = None
    delta_ci_high: Optional[float] = None
    p_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "ci_95": [self.ci_low, self.ci_high],
            "std_err": self.std_err,
            "delta_to_baseline": self.delta_to_baseline,
            "delta_ci_95": [self.delta_ci_low, self.delta_ci_high] if self.delta_ci_low is not None else None,
            "p_value": self.p_value,
        }


@dataclass
class ArmBootstrapSummary:
    """Bootstrap confidence intervals across all primary metrics for an arm."""

    arm: str
    n_tasks: int
    metrics: Dict[str, BootstrapCI]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "n_tasks": self.n_tasks,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


PRIMARY_METRICS: Sequence[str] = (
    "benign_utility",
    "attack_utility",
    "overall_utility",
    "asr",
    "fbr",
    "mean_input_tokens",
    "mean_output_tokens",
    "mean_llm_calls",
    "mean_cost_usd",
)


class PairedBootstrapEngine:
    """Paired cluster bootstrap resampler over user-task triplets."""

    def __init__(
        self,
        n_boot: int = 2000,
        seed: int = 42,
        baseline_arm: str = "A",
    ) -> None:
        self.n_boot = int(n_boot)
        self.seed = int(seed)
        self.baseline_arm = baseline_arm

    def bootstrap_metric(
        self,
        vals: np.ndarray,
        idx_matrix: np.ndarray,
        baseline_vals: Optional[np.ndarray] = None,
        metric_name: str = "metric",
    ) -> BootstrapCI:
        """Compute bootstrap CI and paired delta given a pre-computed index matrix."""
        n = len(vals)
        if n == 0:
            return BootstrapCI(metric=metric_name, point_estimate=0.0, ci_low=0.0, ci_high=0.0, std_err=0.0)

        point_est = float(np.mean(vals))
        boot_means = vals[idx_matrix].mean(axis=1)

        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        std_err = float(np.std(boot_means, ddof=1)) if len(boot_means) > 1 else 0.0

        delta_est: Optional[float] = None
        d_low: Optional[float] = None
        d_high: Optional[float] = None
        p_val: Optional[float] = None

        if baseline_vals is not None and len(baseline_vals) == n:
            diffs = vals - baseline_vals
            delta_est = float(np.mean(diffs))
            boot_diffs = diffs[idx_matrix].mean(axis=1)
            d_low = float(np.percentile(boot_diffs, 2.5))
            d_high = float(np.percentile(boot_diffs, 97.5))

            # Two-sided empirical p-value for H0: Delta = 0
            n_b = len(boot_diffs)
            p_le = np.sum(boot_diffs <= 0) / n_b
            p_ge = np.sum(boot_diffs >= 0) / n_b
            raw_p = 2.0 * min(p_le, p_ge)
            p_val = max(1.0 / n_b, min(1.0, float(raw_p)))

        return BootstrapCI(
            metric=metric_name,
            point_estimate=point_est,
            ci_low=ci_low,
            ci_high=ci_high,
            std_err=std_err,
            delta_to_baseline=delta_est,
            delta_ci_low=d_low,
            delta_ci_high=d_high,
            p_value=p_val,
        )

    def run_bootstrap(
        self,
        arm_metrics: Dict[str, ArmMetrics],
        metrics_to_compute: Optional[Sequence[str]] = None,
    ) -> Dict[str, ArmBootstrapSummary]:
        """Execute paired cluster bootstrap resampling across all arms."""
        if not arm_metrics:
            return {}

        metrics_list = list(metrics_to_compute or PRIMARY_METRICS)

        # 1. Identify common user task clusters (pair_id)
        arm_names = list(arm_metrics.keys())
        all_task_ids = set()
        for arm_m in arm_metrics.values():
            all_task_ids.update(arm_m.task_metrics.keys())
        pair_ids = sorted(all_task_ids)
        n_tasks = len(pair_ids)

        if n_tasks == 0:
            return {}

        # 2. Pre-generate deterministic cluster index matrix: shape (n_boot, n_tasks)
        rng = np.random.RandomState(self.seed)
        idx_matrix = rng.randint(0, n_tasks, size=(self.n_boot, n_tasks))

        # 3. Extract baseline arm task vectors if baseline is present
        baseline_arm_key = self.baseline_arm
        if baseline_arm_key not in arm_metrics and "A" in arm_metrics:
            baseline_arm_key = "A"

        baseline_vectors: Dict[str, np.ndarray] = {}
        if baseline_arm_key in arm_metrics:
            base_tm = arm_metrics[baseline_arm_key].task_metrics
            for m in metrics_list:
                baseline_vectors[m] = np.array([
                    getattr(base_tm.get(pid, TaskMetrics(arm=baseline_arm_key, pair_id=pid)), m, 0.0)
                    for pid in pair_ids
                ], dtype=float)

        # 4. Compute bootstrap distributions for every arm
        summaries: Dict[str, ArmBootstrapSummary] = {}

        for arm, arm_m in arm_metrics.items():
            arm_tm = arm_m.task_metrics
            arm_cis: Dict[str, BootstrapCI] = {}

            is_baseline = (arm == baseline_arm_key)

            for m in metrics_list:
                vals = np.array([
                    getattr(arm_tm.get(pid, TaskMetrics(arm=arm, pair_id=pid)), m, 0.0)
                    for pid in pair_ids
                ], dtype=float)

                base_vals = None if is_baseline else baseline_vectors.get(m)
                ci = self.bootstrap_metric(vals, idx_matrix, baseline_vals=base_vals, metric_name=m)
                arm_cis[m] = ci

            summaries[arm] = ArmBootstrapSummary(arm=arm, n_tasks=n_tasks, metrics=arm_cis)

        return summaries


__all__ = ["ArmBootstrapSummary", "BootstrapCI", "PRIMARY_METRICS", "PairedBootstrapEngine"]
