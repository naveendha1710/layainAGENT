"""Publication table generators outputting formatted Markdown and LaTeX (booktabs).

Generates:
  * Table 1: Main Ablation Matrix (Arms A, B_regex, B, C_always_small, C_random, C, D)
    showing Benign Utility, Attack Utility, ASR, Cost/Task ($), LLM Calls, and Latency with 95% CIs.
  * Table 2: Per-Gate Firing Rate, Fallback Rate, Precision, Recall, and Latency Breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from laya_agent.analysis.bootstrap import ArmBootstrapSummary
from laya_agent.analysis.metrics import ArmMetrics, GateMetrics

ARM_ORDER: Sequence[str] = (
    "A",
    "B_regex",
    "B",
    "C_always_small",
    "C_random",
    "C",
    "D",
)

GATE_SCHEMAS: Dict[str, str] = {
    "scorer": "choice (small/big)",
    "filter": "choice (keep/drop)",
    "router": "choice (bypass/call)",
    "firewall": "noul + choice",
    "supervisor": "choice (cont/nudge/stop)",
}


def _fmt_pct_ci(ci: Optional[Any], scale: float = 100.0) -> str:
    if ci is None:
        return "-"
    pt = ci.point_estimate * scale
    low = ci.ci_low * scale
    high = ci.ci_high * scale
    return f"{pt:.1f} [{low:.1f}, {high:.1f}]"


def _fmt_num_ci(ci: Optional[Any], fmt: str = ".4f") -> str:
    if ci is None:
        return "-"
    pt = ci.point_estimate
    low = ci.ci_low
    high = ci.ci_high
    return f"{pt:{fmt}} [{low:{fmt}}, {high:{fmt}}]"


def generate_ablation_markdown(
    arm_metrics: Dict[str, ArmMetrics],
    bootstrap_summaries: Optional[Dict[str, ArmBootstrapSummary]] = None,
) -> str:
    """Generate Table 1: Main Ablation Matrix in GitHub Flavored Markdown."""
    lines = [
        "| Arm | Benign Utility (%) | Attack Utility (%) | ASR (%) | Cost / Task ($) | LLM Calls | Gate Latency (p50/p95 ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_arms = list(arm_metrics.keys())
    ordered_arms = [a for a in ARM_ORDER if a in all_arms] + [a for a in all_arms if a not in ARM_ORDER]

    for arm in ordered_arms:
        am = arm_metrics[arm]
        bs = bootstrap_summaries.get(arm) if bootstrap_summaries else None

        if bs:
            b_util = _fmt_pct_ci(bs.metrics.get("benign_utility"))
            a_util = _fmt_pct_ci(bs.metrics.get("attack_utility"))
            asr = _fmt_pct_ci(bs.metrics.get("asr"))
            cost = f"${_fmt_num_ci(bs.metrics.get('mean_cost_usd'), '.4f')}"
            calls = f"{bs.metrics.get('mean_llm_calls').point_estimate:.1f}" if bs.metrics.get("mean_llm_calls") else f"{am.mean_llm_calls:.1f}"
        else:
            b_util = f"{am.benign_utility * 100.0:.1f}"
            a_util = f"{am.attack_utility * 100.0:.1f}"
            asr = f"{am.asr * 100.0:.1f}"
            cost = f"${am.mean_cost_usd:.4f}"
            calls = f"{am.mean_llm_calls:.1f}"

        gate_lat = f"{am.gate_latency_p50:.1f} / {am.gate_latency_p95:.1f}"
        lines.append(f"| **{arm}** | {b_util} | {a_util} | {asr} | {cost} | {calls} | {gate_lat} |")

    return "\n".join(lines)


def generate_ablation_latex(
    arm_metrics: Dict[str, ArmMetrics],
    bootstrap_summaries: Optional[Dict[str, ArmBootstrapSummary]] = None,
) -> str:
    """Generate Table 1: Main Ablation Matrix in publication-ready LaTeX (booktabs)."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Arm} & \textbf{Benign Util (\%)} & \textbf{Attack Util (\%)} & \textbf{ASR (\%)} & \textbf{Cost / Task (\$)} & \textbf{LLM Calls} & \textbf{Gate Latency (ms)} \\",
        r"\midrule",
    ]

    all_arms = list(arm_metrics.keys())
    ordered_arms = [a for a in ARM_ORDER if a in all_arms] + [a for a in all_arms if a not in ARM_ORDER]

    for arm in ordered_arms:
        am = arm_metrics[arm]
        bs = bootstrap_summaries.get(arm) if bootstrap_summaries else None
        arm_escaped = arm.replace("_", r"\_")

        if bs:
            b_util = _fmt_pct_ci(bs.metrics.get("benign_utility"))
            a_util = _fmt_pct_ci(bs.metrics.get("attack_utility"))
            asr = _fmt_pct_ci(bs.metrics.get("asr"))
            cost = f"\\${_fmt_num_ci(bs.metrics.get('mean_cost_usd'), '.4f')}"
            calls = f"{bs.metrics.get('mean_llm_calls').point_estimate:.1f}" if bs.metrics.get("mean_llm_calls") else f"{am.mean_llm_calls:.1f}"
        else:
            b_util = f"{am.benign_utility * 100.0:.1f}"
            a_util = f"{am.attack_utility * 100.0:.1f}"
            asr = f"{am.asr * 100.0:.1f}"
            cost = f"\\${am.mean_cost_usd:.4f}"
            calls = f"{am.mean_llm_calls:.1f}"

        gate_lat = f"{am.gate_latency_p50:.1f} / {am.gate_latency_p95:.1f}"
        lines.append(
            f"\\textbf{{{arm_escaped}}} & {b_util} & {a_util} & {asr} & {cost} & {calls} & {gate_lat} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Main Ablation Matrix across experimental arms. All point estimates are reported alongside 95\% paired cluster bootstrap confidence intervals.}",
        r"\label{tab:main_ablation}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_gate_breakdown_markdown(arm_metrics: Dict[str, ArmMetrics], target_arm: str = "D") -> str:
    """Generate Table 2: Per-Gate Firing, Fallback, and Latency Breakdown in Markdown."""
    am = arm_metrics.get(target_arm) or next(iter(arm_metrics.values())) if arm_metrics else None
    if not am:
        return "No gate metrics available."

    lines = [
        f"### Per-Gate Breakdown (Arm {am.arm})",
        "",
        "| Gate | Schema Primitive | Firing Rate (%) | Fallback Rate (%) | Precision | Recall | F1 | Latency p50 (ms) | Latency p95 (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for g_name in ["scorer", "filter", "router", "firewall", "supervisor"]:
        gm = am.gate_metrics.get(g_name, GateMetrics(gate=g_name))
        schema = GATE_SCHEMAS.get(g_name, "choice")
        f_rate = f"{gm.firing_rate * 100.0:.1f}"
        fb_rate = f"{gm.fallback_rate * 100.0:.1f}"
        p = f"{gm.precision:.2f}"
        r = f"{gm.recall:.2f}"
        f1 = f"{gm.f1:.2f}"
        p50 = f"{gm.latency_p50:.1f}"
        p95 = f"{gm.latency_p95:.1f}"
        lines.append(f"| **{g_name.capitalize()}** | `{schema}` | {f_rate} | {fb_rate} | {p} | {r} | {f1} | {p50} | {p95} |")

    return "\n".join(lines)


def generate_gate_breakdown_latex(arm_metrics: Dict[str, ArmMetrics], target_arm: str = "D") -> str:
    """Generate Table 2: Per-Gate Firing, Fallback, and Latency Breakdown in LaTeX."""
    am = arm_metrics.get(target_arm) or next(iter(arm_metrics.values())) if arm_metrics else None
    if not am:
        return "% No gate metrics available."

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Gate} & \textbf{Schema} & \textbf{Firing (\%)} & \textbf{Fallback (\%)} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{p50 (ms)} & \textbf{p95 (ms)} \\",
        r"\midrule",
    ]

    for g_name in ["scorer", "filter", "router", "firewall", "supervisor"]:
        gm = am.gate_metrics.get(g_name, GateMetrics(gate=g_name))
        schema = GATE_SCHEMAS.get(g_name, "choice").replace("_", r"\_")
        f_rate = f"{gm.firing_rate * 100.0:.1f}"
        fb_rate = f"{gm.fallback_rate * 100.0:.1f}"
        p = f"{gm.precision:.2f}"
        r = f"{gm.recall:.2f}"
        f1 = f"{gm.f1:.2f}"
        p50 = f"{gm.latency_p50:.1f}"
        p95 = f"{gm.latency_p95:.1f}"
        lines.append(
            f"\\textbf{{{g_name.capitalize()}}} & {schema} & {f_rate} & {fb_rate} & {p} & {r} & {f1} & {p50} & {p95} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Per-Gate Firing, Fallback, and Latency Breakdown for Arm {am.arm}.}}",
        r"\label{tab:gate_breakdown}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_ablation_table(
    arm_metrics: Dict[str, ArmMetrics],
    bootstrap_summaries: Optional[Dict[str, ArmBootstrapSummary]] = None,
    output_format: str = "markdown",
) -> str:
    if output_format.lower() == "latex":
        return generate_ablation_latex(arm_metrics, bootstrap_summaries)
    return generate_ablation_markdown(arm_metrics, bootstrap_summaries)


def generate_gate_breakdown_table(
    arm_metrics: Dict[str, ArmMetrics],
    target_arm: str = "D",
    output_format: str = "markdown",
) -> str:
    if output_format.lower() == "latex":
        return generate_gate_breakdown_latex(arm_metrics, target_arm=target_arm)
    return generate_gate_breakdown_markdown(arm_metrics, target_arm=target_arm)


def save_all_tables(
    arm_metrics: Dict[str, ArmMetrics],
    bootstrap_summaries: Optional[Dict[str, ArmBootstrapSummary]] = None,
    output_dir: Union[str, Path] = "results",
) -> Dict[str, Path]:
    """Save ablation and gate breakdown tables in both Markdown and LaTeX formats."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files_written: Dict[str, Path] = {}

    # 1. Summary JSON
    summary_data = {
        "metrics": {arm: am.to_dict() for arm, am in arm_metrics.items()},
        "bootstrap": {arm: bs.to_dict() for arm, bs in (bootstrap_summaries or {}).items()},
    }
    json_path = out / "summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary_data, fh, indent=2)
    files_written["summary_json"] = json_path

    # 2. Ablation Markdown
    ablation_md = generate_ablation_markdown(arm_metrics, bootstrap_summaries)
    md_path = out / "ablation_table.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(ablation_md + "\n")
    files_written["ablation_md"] = md_path

    # 3. Ablation LaTeX
    ablation_tex = generate_ablation_latex(arm_metrics, bootstrap_summaries)
    tex_path = out / "ablation_table.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(ablation_tex + "\n")
    files_written["ablation_tex"] = tex_path

    # 4. Gate Breakdown Markdown
    gate_md = generate_gate_breakdown_markdown(arm_metrics)
    gate_md_path = out / "gate_breakdown.md"
    with open(gate_md_path, "w", encoding="utf-8") as fh:
        fh.write(gate_md + "\n")
    files_written["gate_md"] = gate_md_path

    # 5. Gate Breakdown LaTeX
    gate_tex = generate_gate_breakdown_latex(arm_metrics)
    gate_tex_path = out / "gate_breakdown.tex"
    with open(gate_tex_path, "w", encoding="utf-8") as fh:
        fh.write(gate_tex + "\n")
    files_written["gate_tex"] = gate_tex_path

    return files_written


__all__ = [
    "ARM_ORDER",
    "GATE_SCHEMAS",
    "generate_ablation_latex",
    "generate_ablation_markdown",
    "generate_ablation_table",
    "generate_gate_breakdown_latex",
    "generate_gate_breakdown_markdown",
    "generate_gate_breakdown_table",
    "save_all_tables",
]
