"""Evaluation analysis suite: metrics engine, bootstrap statistics, and paper tables."""

from laya_agent.analysis.metrics import ArmMetrics, MetricsEngine, RunMetrics, TaskMetrics
from laya_agent.analysis.bootstrap import BootstrapCI, PairedBootstrapEngine
from laya_agent.analysis.tables import generate_ablation_table, generate_gate_breakdown_table, save_all_tables

__all__ = [
    "ArmMetrics",
    "BootstrapCI",
    "MetricsEngine",
    "PairedBootstrapEngine",
    "RunMetrics",
    "TaskMetrics",
    "generate_ablation_table",
    "generate_gate_breakdown_table",
    "generate_figures",
    "plot_fig1_ablation_asr_utility",
    "plot_fig2_gate_latency_cdf",
    "plot_fig3_call_reduction",
    "save_all_tables",
]


def __getattr__(name: str):
    if name in {
        "generate_figures",
        "plot_fig1_ablation_asr_utility",
        "plot_fig2_gate_latency_cdf",
        "plot_fig3_call_reduction",
    }:
        import laya_agent.analysis.visualize as viz

        return getattr(viz, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
