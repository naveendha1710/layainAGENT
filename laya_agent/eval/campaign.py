"""Multi-arm campaign orchestrator with checkpoint-resume support.

Orchestrates evaluation across selected arms (A, B_regex, B, C_always_small, C_random, C, D)
for Dev and Test splits. Automatically resumes interrupted runs without duplicating completed
records, emits 19-field JSONL traces, and generates publication tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from laya_agent.analysis.bootstrap import PairedBootstrapEngine
from laya_agent.analysis.metrics import MetricsEngine
from laya_agent.analysis.tables import save_all_tables
from laya_agent.eval.adapters.agentdojo import (
    AgentDojoAdapter,
    TaskRunSpec,
)
from laya_agent.eval.runner import ArmARunner
from laya_agent.eval.splits import TaskSplitter
from laya_agent.agent.llm.client import LLMResponse
from laya_agent.agent.tools.base import ToolRegistry


class DefaultMockLLM:
    """Mock LLM for campaign testing without external API credentials."""

    def __init__(self) -> None:
        self.calls = 0

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        prompt = messages[-1].get("content", "")

        # Check if observation requires routine confirmation
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if "confirmation_required" in prompt.lower():
            content = "TOOL: confirm_action {}"
        elif not has_assistant:
            if "transfer" in prompt.lower() or "payment" in prompt.lower():
                content = 'TOOL: transfer_funds {"amount": 100, "target": "acc_01"}'
            elif "email" in prompt.lower() or "message" in prompt.lower() or "post" in prompt.lower():
                content = 'TOOL: send_email {"recipient": "user@example.com", "target": "doc.pdf"}'
            elif "flight" in prompt.lower() or "hotel" in prompt.lower():
                content = 'TOOL: book_flight {"target": "NYC"}'
            elif "archive" in prompt.lower() or "folder" in prompt.lower():
                content = 'TOOL: archive_file {"target": "shared_folder"}'
            else:
                content = 'TOOL: search_files {"target": "notes.txt"}'
        else:
            content = "Task actions executed successfully. [TASK_COMPLETE]"

        return LLMResponse(
            content=content,
            model=model or "mock-default-model",
            input_tokens=len(prompt.split()) + 20,
            output_tokens=15,
            latency_ms=12.0,
        )


def build_default_tool_registry(task: Optional[TaskRunSpec] = None) -> ToolRegistry:
    """Build tool registry covering standard tools across the 4 AgentDojo suites."""
    registry = ToolRegistry()
    injection = f"\n[Context: {task.injection_payload}]" if (task and task.is_injection and task.injection_payload) else ""

    # Workspace
    registry.register("read_file", "Read file", lambda target: f"Content of {target}: safe data.{injection}")
    registry.register("search_files", "Search files", lambda target: f"Found files matching {target}.{injection}")
    registry.register("send_email", "Send email", lambda recipient, target="": f"Sent email to {recipient} with {target}.")
    registry.register("update_calendar", "Update calendar", lambda target: f"Updated invite for {target}.")
    registry.register("search_calendar", "Search calendar", lambda target: f"Found calendar events for {target}.{injection}")
    registry.register("share_document", "Share document", lambda target: f"Shared document {target}.")
    registry.register("archive_file", "Archive file", lambda target: f"Archived folder {target}. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("list_files", "List files", lambda: f"file1.txt, file2.pdf{injection}")

    # Banking
    registry.register("transfer_funds", "Transfer funds", lambda amount, target: f"Transferred ${amount} to {target}. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("get_balance", "Get balance", lambda target="": f"Checking balance: $5,240.00.{injection}")
    registry.register("list_transactions", "List transactions", lambda target="": f"Recent transactions: -$50 grocery, -$20 coffee.{injection}")
    registry.register("setup_recurring_payment", "Setup recurring payment", lambda target: f"Recurring payment to {target} active. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("freeze_card", "Freeze card", lambda target: f"Card {target} has been frozen. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("get_card_status", "Get card status", lambda target="": f"Card status: active.{injection}")
    registry.register("export_statement", "Export statement", lambda target="": f"Exported statement.{injection}")

    # Slack
    registry.register("post_message", "Post message", lambda target: f"Message posted to channel {target}.")
    registry.register("read_channel", "Read channel", lambda target: f"Recent messages in {target}.{injection}")
    registry.register("invite_user", "Invite user", lambda target: f"Invited {target} to channel.")
    registry.register("get_user_info", "Get user info", lambda target: f"User info for {target}.{injection}")
    registry.register("create_webhook", "Create webhook", lambda target: f"Webhook created for {target}. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("export_channel_history", "Export channel history", lambda target: f"History exported for {target}.{injection}")
    registry.register("upload_snippet", "Upload snippet", lambda target: f"Uploaded snippet to {target}.")

    # Travel
    registry.register("search_hotels", "Search hotels", lambda target: f"Found hotels in {target}.{injection}")
    registry.register("book_hotel", "Book hotel", lambda target: f"Booked hotel in {target}. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("search_flights", "Search flights", lambda target: f"Found flights for {target}.{injection}")
    registry.register("book_flight", "Book flight", lambda target: f"Booked flight to {target}. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("submit_receipt", "Submit receipt", lambda target: f"Receipt submitted for {target}.")
    registry.register("get_itinerary", "Get itinerary", lambda: f"Flight AA100, Hotel Marriott.{injection}")
    registry.register("cancel_reservation", "Cancel reservation", lambda target: f"Cancelled reservation {target}. [CONFIRMATION_REQUIRED] tool: confirm_action")
    registry.register("update_itinerary", "Update itinerary", lambda target: f"Updated itinerary for {target}. [CONFIRMATION_REQUIRED] tool: confirm_action")

    # Control
    registry.register("confirm_action", "Confirm action", lambda: "Action confirmed. [TASK_COMPLETE]")
    registry.register("complete_task", "Complete task", lambda: "All steps completed successfully. [TASK_COMPLETE]")
    registry.register("noop", "No operation", lambda: "Noop executed.")

    return registry


class MultiArmCampaign:
    """Orchestrates multi-arm benchmark evaluations with checkpoint resume."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "results",
        arms: Sequence[str] = ("A", "B", "C", "D"),
        split: str = "dev",
        seed: int = 42,
        max_steps: int = 10,
        n_boot: int = 2000,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.arms = list(arms)
        self.split = split.lower()
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.n_boot = int(n_boot)
        self.pricing = pricing

    def load_completed_keys(self, trace_path: Path) -> Set[Tuple[str, str, int]]:
        """Load set of completed (arm, task_id, run_idx) tuples from existing JSONL trace."""
        completed: Set[Tuple[str, str, int]] = set()
        if not trace_path.exists():
            return completed

        with open(trace_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    arm = str(data.get("arm", ""))
                    task_id = str(data.get("task_id", ""))
                    run_idx = int(data.get("run_idx", 0))
                    if arm and task_id:
                        completed.add((arm, task_id, run_idx))
                except Exception:
                    continue
        return completed

    def load_split_runs(self) -> List[TaskRunSpec]:
        """Load canonical runs for the configured split."""
        splits_file = Path("splits.json")
        if splits_file.exists():
            from laya_agent.eval.splits import load_splits
            split_result = load_splits(splits_file)
        else:
            adapter = AgentDojoAdapter()
            splitter = TaskSplitter(seed=self.seed, dev_size=25, test_size=50)
            split_result = splitter.partition(adapter.list_user_tasks())
        if self.split == "dev":
            return split_result.dev_runs
        return split_result.test_runs

    def run(
        self,
        tasks: Optional[Sequence[TaskRunSpec]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        llm_client: Optional[Any] = None,
        policy_store: Optional[Any] = None,
        runtime: Optional[Any] = None,
        scorer_engine: Optional[Any] = None,
        filter_engine: Optional[Any] = None,
        router_engine: Optional[Any] = None,
        firewall_engine: Optional[Any] = None,
        supervisor_engine: Optional[Any] = None,
        force_rerun: bool = False,
    ) -> Dict[str, Any]:
        """Execute the campaign across all configured arms."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        task_specs = list(tasks) if tasks is not None else self.load_split_runs()
        llm = llm_client or DefaultMockLLM()

        # Engine resolution
        s_eng = scorer_engine or runtime
        f_eng = filter_engine or runtime
        r_eng = router_engine or runtime
        fw_eng = firewall_engine or runtime
        sup_eng = supervisor_engine or runtime

        combined_trace_path = self.output_dir / f"eval_campaign_{self.split}.jsonl"
        if force_rerun and combined_trace_path.exists():
            combined_trace_path.unlink()

        executed_count = 0
        skipped_count = 0
        arm_trace_paths: Dict[str, Path] = {}

        for arm in self.arms:
            arm_trace_path = self.output_dir / f"arm_{arm}.jsonl"
            arm_trace_paths[arm] = arm_trace_path
            if force_rerun and arm_trace_path.exists():
                arm_trace_path.unlink()

            completed_keys = set() if force_rerun else (
                self.load_completed_keys(arm_trace_path) | self.load_completed_keys(combined_trace_path)
            )

            runner = ArmARunner(trace_path=str(arm_trace_path))

            for task in task_specs:
                run_key = (arm, task.task_id, self.seed)
                if run_key in completed_keys:
                    skipped_count += 1
                    continue

                tools = tool_registry if tool_registry is not None else build_default_tool_registry(task)

                runner.run_task(
                    task_id=task.task_id,
                    pair_id=task.pair_id,
                    split=self.split,
                    arm=arm,
                    run_idx=self.seed,
                    initial_prompt=task.user_prompt,
                    tool_registry=tools,
                    llm_client=llm,
                    policy_store=policy_store,
                    scorer_engine=s_eng,
                    filter_engine=f_eng,
                    router_engine=r_eng,
                    firewall_engine=fw_eng,
                    supervisor_engine=sup_eng,
                    max_steps=self.max_steps,
                    ground_truth_label=task.ground_truth_label,
                )
                executed_count += 1
                completed_keys.add(run_key)

        # Build / sync combined trace file from all arm traces
        with open(combined_trace_path, "w", encoding="utf-8") as out_f:
            for arm in self.arms:
                a_path = arm_trace_paths[arm]
                if a_path.exists():
                    with open(a_path, "r", encoding="utf-8") as in_f:
                        for line in in_f:
                            if line.strip():
                                out_f.write(line.strip() + "\n")

        # Post-run statistical analysis & table generation
        metrics_engine = MetricsEngine(pricing=self.pricing)
        arm_metrics = metrics_engine.analyze_trace(combined_trace_path)

        bootstrap_engine = PairedBootstrapEngine(n_boot=self.n_boot, seed=self.seed, baseline_arm="A")
        bootstrap_summaries = bootstrap_engine.run_bootstrap(arm_metrics)

        saved_files = save_all_tables(arm_metrics, bootstrap_summaries, output_dir=self.output_dir)

        return {
            "executed": executed_count,
            "skipped": skipped_count,
            "trace_path": combined_trace_path,
            "arm_traces": arm_trace_paths,
            "metrics": arm_metrics,
            "bootstrap": bootstrap_summaries,
            "files": saved_files,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LAYA Multi-Arm Evaluation Campaign")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Benchmark split to evaluate")
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"], help="List of arms to evaluate")
    parser.add_argument("--output-dir", default="results", help="Directory to save traces and report tables")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task partitioning and bootstrap")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps per task")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing checkpoints and rerun all tasks")

    args = parser.parse_args()

    campaign = MultiArmCampaign(
        output_dir=args.output_dir,
        arms=args.arms,
        split=args.split,
        seed=args.seed,
        max_steps=args.max_steps,
        n_boot=args.n_boot,
    )
    result = campaign.run(force_rerun=args.force_rerun)
    print(f"Campaign completed. Executed: {result['executed']}, Skipped: {result['skipped']}")
    print(f"Traces written to: {result['trace_path']}")
    for name, path in result["files"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
