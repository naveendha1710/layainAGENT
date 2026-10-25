"""Test Set Evaluation & 10,000-Resample Publication Campaign Execution.

Executes the benchmark evaluation across the 50 Test user tasks for all 7 arms:
  Core Arms:     A, B, C, D (600 runs)
  Ablation Arms: B_regex, C_always_small, C_random (450 runs)
Totaling 1,050 runs (7 arms * 150 runs) using canonical frozen splits, frozen policy store,
local ModernBERT gate runtime, and dual-tier clients.

Outputs:
  * Atomic JSONL traces: results/test/arm_{arm}.jsonl
  * Combined trace: results/test/eval_campaign_test.jsonl
  * Summary metrics & bootstrap: results/test/summary_metrics.json
  * Publication tables: results/test/ablation_table.{md,tex}, results/test/gate_breakdown.{md,tex}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from laya_agent.agent.gates.runtime import LayaRuntime
from laya_agent.agent.retrieval.store import load_default_policy_store
from laya_agent.analysis.bootstrap import PairedBootstrapEngine
from laya_agent.analysis.metrics import MetricsEngine
from laya_agent.analysis.tables import ARM_ORDER, save_all_tables
from laya_agent.eval.adapters.agentdojo import TaskRunSpec
from laya_agent.eval.campaign import MultiArmCampaign
from laya_agent.eval.splits import generate_default_splits, load_splits


def run_test_campaign(
    output_dir: Path | str = "results/test",
    arms: Sequence[str] = ARM_ORDER,
    splits_file: Path | str = "splits.json",
    n_boot: int = 10000,
    seed: int = 42,
    max_steps: int = 10,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Execute the 1,050-run test campaign across all 7 arms with B=10,000 bootstrap."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("      LAYA Agent Architecture: Final Test Evaluation Campaign         ")
    print("=====================================================================")
    print(f"Target Split:      test (50 user tasks = 150 runs/arm)")
    print(f"Experiment Arms:   {list(arms)} ({len(arms) * 150} total runs)")
    print(f"Output Directory:  {out_path}")
    print(f"Bootstrap Seed:    {seed} (B={n_boot} resamples)")
    print(f"Max Step Budget:   {max_steps}")
    print("---------------------------------------------------------------------")

    # 1. Load canonical Test tasks
    print("\n[1/5] Loading canonical Test tasks...")
    splits_path = Path(splits_file)
    if splits_path.exists():
        split_result = load_splits(splits_path)
        print(f"      Loaded splits from {splits_path}")
    else:
        print(f"      Splits file {splits_path} not found. Generating default splits...")
        split_result = generate_default_splits(seed=seed)

    test_runs = split_result.test_runs
    print(f"      Test Tasks: {len(split_result.test_user_tasks)} user tasks, {len(test_runs)} runs")
    assert len(split_result.test_user_tasks) == 50, f"Expected 50 test tasks, got {len(split_result.test_user_tasks)}"
    assert len(test_runs) == 150, f"Expected 150 test runs, got {len(test_runs)}"

    # 2. Initialize Policy Store & ModernBERT Gate Runtime
    print("\n[2/5] Initializing Policy Store & Local Gate Runtime...")
    policy_store = load_default_policy_store()
    print(f"      Policy store ready: {len(policy_store.entries)} rules indexed in FAISS")

    runtime = LayaRuntime()
    print(f"      LayaRuntime initialized on device: {runtime.device}")

    # 3. Launch Multi-Arm Campaign with Checkpoint-Resume
    print(f"\n[3/5] Executing multi-arm evaluation campaign across arms {list(arms)}...")
    campaign = MultiArmCampaign(
        output_dir=out_path,
        arms=arms,
        split="test",
        seed=seed,
        max_steps=max_steps,
        n_boot=n_boot,
    )

    t0 = time.perf_counter()
    campaign_result = campaign.run(
        tasks=test_runs,
        policy_store=policy_store,
        runtime=runtime,
        force_rerun=force_rerun,
    )
    duration = time.perf_counter() - t0
    print(f"      Campaign completed in {duration:.2f}s.")
    print(f"      Executed runs: {campaign_result['executed']}, Skipped runs: {campaign_result['skipped']}")

    # 4. Verify Per-Arm Trace Files & Checkpoint-Resume
    print("\n[4/5] Verifying trace files and checkpoint-resume integrity...")
    arm_traces = campaign_result["arm_traces"]
    for arm in arms:
        a_path = arm_traces[arm]
        assert a_path.exists(), f"Per-arm trace file missing: {a_path}"
        with open(a_path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        print(f"      * Arm {arm:<14} Trace: {a_path} ({len(lines)} records)")

    # Test checkpoint-resume: running again without force_rerun should skip 100% of tasks
    resume_check = campaign.run(
        tasks=test_runs,
        policy_store=policy_store,
        runtime=runtime,
        force_rerun=False,
    )
    print(f"      Checkpoint-Resume Verification: Executed={resume_check['executed']}, Skipped={resume_check['skipped']}")
    assert resume_check["executed"] == 0, "Checkpoint resume failed: executed runs when all should be skipped."
    assert resume_check["skipped"] == len(arms) * len(test_runs), "Checkpoint resume skipped count mismatch."

    # 5. Gate Diagnostics & Calibration Analysis
    print("\n[5/5] Computing Gate Diagnostics & Publication Bundle...")
    arm_metrics = campaign_result["metrics"]
    bootstrap = campaign_result["bootstrap"]
    saved_files = campaign_result["files"]

    print("\n======================================================================================================================")
    print("                                      FINAL TEST SET EVALUATION ABLATION MATRIX                                      ")
    print("======================================================================================================================")
    header = (
        f"{'Arm':<15} | {'Benign Ut.':<10} | {'Attack Ut.':<10} | {'ASR':<8} | {'FBR':<8} | "
        f"{'Cost/Task':<10} | {'LLM Calls':<10} | {'In Tokens':<10} | {'Gate p50':<9} | {'Gate p95':<9}"
    )
    print(header)
    print("-" * len(header))

    for arm in arms:
        m = arm_metrics.get(arm)
        if not m:
            continue
        print(
            f"{arm:<15} | "
            f"{m.benign_utility * 100:>9.1f}% | "
            f"{m.attack_utility * 100:>9.1f}% | "
            f"{m.asr * 100:>7.1f}% | "
            f"{m.fbr * 100:>7.2f}% | "
            f"${m.mean_cost_usd:>8.4f} | "
            f"{m.mean_llm_calls:>10.2f} | "
            f"{m.mean_input_tokens:>10.1f} | "
            f"{m.gate_latency_p50:>7.2f}ms | "
            f"{m.gate_latency_p95:>7.2f}ms"
        )

    # Detailed Gate Diagnostics for Arm D
    m_a = arm_metrics.get("A")
    m_d = arm_metrics.get("D")
    if m_d:
        print("\n--- Gate Behavioral Breakdown (Arm D) ---")
        gm = m_d.gate_metrics

        if "scorer" in gm:
            scorer_gm = gm["scorer"]
            print(f"  * Scorer Gate:")
            print(f"      Total Evaluated: {scorer_gm.total_evaluations}")
            print(f"      Firing Rate:     {scorer_gm.firing_rate * 100:.1f}%")
            print(f"      Fallback Rate:   {scorer_gm.fallback_rate * 100:.1f}%")
            print(f"      Latency (p50):   {scorer_gm.latency_p50:.2f} ms")

        if "filter" in gm:
            filter_gm = gm["filter"]
            token_diff = m_a.mean_input_tokens - m_d.mean_input_tokens if m_a else 0
            token_pct = (token_diff / m_a.mean_input_tokens * 100.0) if m_a and m_a.mean_input_tokens > 0 else 0.0
            print(f"  * Applicability Filter Gate:")
            print(f"      Rules Evaluated: {filter_gm.total_evaluations}")
            print(f"      Rules Dropped:   {filter_gm.active_firings}")
            print(f"      Token Savings:   {token_diff:.1f} tokens/task ({token_pct:.1f}%)")
            print(f"      Latency (p50):   {filter_gm.latency_p50:.2f} ms")

        if "router" in gm:
            router_gm = gm["router"]
            llm_call_saved = (m_a.mean_llm_calls - m_d.mean_llm_calls) if m_a else 0
            print(f"  * Action Router Gate:")
            print(f"      Steps Evaluated: {router_gm.total_evaluations}")
            print(f"      LLM Bypasses:    {router_gm.active_firings}")
            print(f"      Calls Saved:     {llm_call_saved:.2f} hosted LLM calls/task")
            print(f"      Latency (p50):   {router_gm.latency_p50:.2f} ms")

        if "firewall" in gm:
            fw_gm = gm["firewall"]
            asr_reduction = (m_a.asr - m_d.asr) * 100.0 if m_a else 0.0
            print(f"  * Context Firewall Gate:")
            print(f"      Chunks Scanned:  {fw_gm.total_evaluations}")
            print(f"      Redactions:      {fw_gm.active_firings}")
            print(f"      ASR Reduction:   {asr_reduction:.1f}% (Arm A: {m_a.asr*100:.1f}% -> Arm D: {m_d.asr*100:.1f}%)")
            print(f"      False-Block (FBR): {m_d.fbr * 100:.2f}% (Target: < 5.0%)")
            assert m_d.fbr < 0.05, f"False-Block Rate exceeded 5%: {m_d.fbr * 100:.2f}%"
            print(f"      Latency (p50):   {fw_gm.latency_p50:.2f} ms")

        if "supervisor" in gm:
            sup_gm = gm["supervisor"]
            print(f"  * Supervisor Gate:")
            print(f"      Steps Evaluated: {sup_gm.total_evaluations}")
            print(f"      Interventions:   {sup_gm.active_firings}")
            print(f"      Fallback Rate:   {sup_gm.fallback_rate * 100:.1f}%")
            print(f"      Latency (p50):   {sup_gm.latency_p50:.2f} ms")

    print("\n--- Generated Publication Artifacts ---")
    for k, v in saved_files.items():
        print(f"  * {k:<15}: {v}")

    print("=====================================================================")
    print("Final Test Set Evaluation Campaign successfully completed & verified!")
    print("=====================================================================\n")

    return {
        "campaign_result": campaign_result,
        "arm_metrics": arm_metrics,
        "bootstrap": bootstrap,
        "saved_files": saved_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1,050-Run Test Set Multi-Arm Campaign with B=10,000 Bootstrap")
    parser.add_argument("--output-dir", default="results/test", help="Directory to save traces and tables")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARM_ORDER),
        help=f"Arms to evaluate (default: {list(ARM_ORDER)})",
    )
    parser.add_argument("--splits-file", default="splits.json", help="Path to splits specification file")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation and bootstrap seed")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps per task")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run and ignore checkpoints")

    args = parser.parse_args()
    run_test_campaign(
        output_dir=args.output_dir,
        arms=args.arms,
        splits_file=args.splits_file,
        n_boot=args.n_boot,
        seed=args.seed,
        max_steps=args.max_steps,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
