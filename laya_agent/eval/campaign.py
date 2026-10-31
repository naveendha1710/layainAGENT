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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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

    def get_model_for_tier(self, tier: str) -> str:
        if tier and tier.lower() == "small":
            return os.environ.get("SMALL_TIER_MODEL", "meta/llama-3.1-8b-instruct")
        return os.environ.get("BIG_TIER_MODEL", "meta/llama-3.1-70b-instruct")

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tier: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        prompt = messages[-1].get("content", "")

        effective_tier = tier or (model if model in ("big", "small") else None)
        if effective_tier:
            target_model = self.get_model_for_tier(effective_tier)
        elif model and model not in ("big", "small", "mock-default-model"):
            target_model = model
        else:
            target_model = self.get_model_for_tier("big")

        # Check if observation requires routine confirmation
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if "confirmation_required" in prompt.lower():
            content = "TOOL: confirm_action {}"
        elif not has_assistant:
            if "read_file" in prompt.lower() or "inbox" in prompt.lower() or "passcode" in prompt.lower():
                content = 'TOOL: read_file {"target": "inbox.txt"}'
            elif "transfer" in prompt.lower() or "payment" in prompt.lower():
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
            model=target_model,
            input_tokens=len(prompt.split()) + 20,
            output_tokens=15,
            latency_ms=12.0,
        )


from laya_agent.eval.sandbox import EnvironmentSandbox


def build_sandbox_tool_registry(task: Optional[TaskRunSpec] = None) -> Tuple[ToolRegistry, EnvironmentSandbox]:
    """Build stateful sandbox and corresponding ToolRegistry for the task."""
    task_id = task.task_id if task else "task_01"
    suite = task.suite if task else "workspace"
    inj_payload = task.injection_payload if (task and task.is_injection) else None
    inj_goal = getattr(task, "injection_goal", {}) if (task and task.is_injection) else {}
    expected = getattr(task, "expected_benign_tools", []) if task else []
    sandbox = EnvironmentSandbox(
        suite=suite,
        task_id=task_id,
        injection_payload=inj_payload,
        injection_goal=inj_goal,
        expected_benign_tools=expected,
    )
    registry = sandbox.get_tool_registry()
    registry.sandbox = sandbox
    return registry, sandbox


def build_default_tool_registry(task: Optional[TaskRunSpec] = None) -> ToolRegistry:
    """Build tool registry covering standard tools across the 4 AgentDojo suites."""
    registry, _ = build_sandbox_tool_registry(task)
    return registry


ALL_ARMS: List[str] = ["A", "B_regex", "B", "C_always_small", "C_random", "C", "D"]


class MultiArmCampaign:
    """Orchestrates multi-arm benchmark evaluations with checkpoint resume."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "results",
        arms: Sequence[str] = ("A", "B_regex", "B", "C_always_small", "C_random", "C", "D"),
        split: str = "dev",
        seed: int = 42,
        max_steps: int = 10,
        n_boot: int = 2000,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
        load_weights: Optional[bool] = None,
        allow_mock_llm: bool = False,
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        if list(arms) == ["all"] or "all" in arms:
            self.arms = list(ALL_ARMS)
        else:
            self.arms = list(arms)
        self.split = split.lower()
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.n_boot = int(n_boot)
        self.pricing = pricing
        self.load_weights = load_weights
        self.allow_mock_llm = bool(allow_mock_llm)
        self.verbose = bool(verbose)

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
        load_weights: Optional[bool] = None,
        force_rerun: bool = False,
        max_tasks: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute the campaign across all configured arms."""
        is_verbose = self.verbose if verbose is None else bool(verbose)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        task_specs = list(tasks) if tasks is not None else self.load_split_runs()
        if max_tasks is not None and max_tasks > 0:
            seen_pairs: Set[str] = set()
            filtered_specs: List[TaskRunSpec] = []
            for spec in task_specs:
                if spec.pair_id not in seen_pairs:
                    if len(seen_pairs) >= max_tasks:
                        continue
                    seen_pairs.add(spec.pair_id)
                filtered_specs.append(spec)
            task_specs = filtered_specs

        # Resolve LLM client: enforce NVIDIABuildClient unless mock explicitly permitted
        if llm_client is not None:
            llm = llm_client
        else:
            api_key = os.environ.get("NVIDIA_API_KEY", "")
            if not api_key:
                for p in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent.parent / ".env"):
                    if p.exists() and p.is_file():
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("NVIDIA_API_KEY="):
                                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                                    break
                    if api_key:
                        break
            if api_key:
                from laya_agent.agent.llm.client import NVIDIABuildClient

                llm = NVIDIABuildClient(api_key=api_key)
            elif self.allow_mock_llm:
                llm = DefaultMockLLM()
            else:
                raise RuntimeError(
                    "NVIDIA_API_KEY environment variable is required for campaign execution. "
                    "Pass --allow-mock-llm only for offline unit/wiring tests."
                )

        if hasattr(llm, "get_model_for_tier"):
            big_model = llm.get_model_for_tier("big")
            small_model = llm.get_model_for_tier("small")
        else:
            big_model = getattr(llm, "model", "default")
            small_model = getattr(llm, "small_model", "default")

        effective_load_weights = load_weights if load_weights is not None else self.load_weights

        # Device & Provider discovery & logging
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
        except ImportError:
            cuda_avail = False

        device = "cuda" if cuda_avail else "cpu"
        provider_name = "LayaProvider" if cuda_avail else "MockProvider"

        unique_pairs = len(set(spec.pair_id for spec in task_specs))
        runs_per_arm = len(task_specs)
        total_runs = len(self.arms) * runs_per_arm
        triplets_str = f"Triplets: {runs_per_arm} runs/arm" if runs_per_arm == unique_pairs * 3 else f"{runs_per_arm} runs/arm"

        print("=" * 70, flush=True)
        print("LAYA Agent Evaluation Campaign", flush=True)
        print(f"Split: {self.split} | Arms: {len(self.arms)} | Tasks: {unique_pairs} ({triplets_str}) | Total Runs: {total_runs}", flush=True)
        print(f"LLM Client: {type(llm).__name__} | Device: {device}", flush=True)
        print("=" * 70, flush=True)

        # Engine resolution
        if runtime is None and cuda_avail:
            from laya_agent.agent.gates.runtime import LayaRuntime
            runtime = LayaRuntime(load_weights=effective_load_weights)

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
        run_num = 0

        for arm in self.arms:
            arm_trace_path = self.output_dir / f"arm_{arm}.jsonl"
            arm_trace_paths[arm] = arm_trace_path
            if force_rerun and arm_trace_path.exists():
                arm_trace_path.unlink()

            completed_keys = set() if force_rerun else (
                self.load_completed_keys(arm_trace_path) | self.load_completed_keys(combined_trace_path)
            )

            runner = ArmARunner(trace_path=str(arm_trace_path))

            if effective_load_weights is not None:
                arm_load_weights = effective_load_weights
            else:
                import torch
                # Default to load_weights=True on CUDA for live gate arms B, C, D
                arm_load_weights = True if (torch.cuda.is_available() and arm.upper() in {"B", "B_REGEX", "C", "C_ALWAYS_SMALL", "C_RANDOM", "D"}) else False

            for task in task_specs:
                run_num += 1
                run_key = (arm, task.task_id, self.seed)
                condition = task.condition or ("injection" if task.is_injection else "benign")

                if run_key in completed_keys:
                    skipped_count += 1
                    if is_verbose:
                        print(f"[{run_num}/{total_runs}] Arm {arm} | Task: {task.task_id} ({condition}) ... SKIPPED (checkpoint)", flush=True)
                    continue

                if not is_verbose:
                    print(f"[{run_num}/{total_runs}] Arm {arm} | Task: {task.task_id} ({condition}) ... ", end="", flush=True)
                else:
                    print(f"[{run_num}/{total_runs}] Arm {arm} | Task: {task.task_id} ({condition}) ...", flush=True)

                if tool_registry is not None:
                    tools = tool_registry
                    sb = getattr(tools, "sandbox", None)
                else:
                    tools, sb = build_sandbox_tool_registry(task)

                t_start = time.perf_counter()
                task_state = runner.run_task(
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
                    load_weights=arm_load_weights,
                    sandbox=sb,
                    expected_benign_tools=getattr(task, "expected_benign_tools", []),
                    verbose=is_verbose,
                )
                elapsed_s = time.perf_counter() - t_start
                executed_count += 1
                completed_keys.add(run_key)

                # Determine outcome label
                outcome_label = getattr(task_state, "outcome_label", None)
                if task.is_injection:
                    if outcome_label == "attack_succeeded":
                        outcome_str = "ASR: PENETRATED"
                    else:
                        outcome_str = "ASR: BLOCKED"
                else:
                    if outcome_label == "task_completed":
                        outcome_str = "Utility: PASS"
                    else:
                        outcome_str = "Utility: FAIL"

                # Calculate run cost
                in_tok = task_state.total_input_tokens
                out_tok = task_state.total_output_tokens
                tier = task_state.model_tier or "big"
                pricing = self.pricing or {}
                tier_sheet = pricing.get(tier.lower(), pricing.get("big", {"input_per_1m": 0.8, "output_per_1m": 2.4}))
                cost_usd = (in_tok * (tier_sheet.get("input_per_1m", 0.0) / 1_000_000.0)) + (out_tok * (tier_sheet.get("output_per_1m", 0.0) / 1_000_000.0))

                turns_count = task_state.turn or 1
                turns_label = f"{turns_count} turn" if turns_count == 1 else f"{turns_count} turns"

                if not is_verbose:
                    print(f"DONE ({elapsed_s:.1f}s | {turns_label} | Cost: ${cost_usd:.4f} | {outcome_str})", flush=True)
                else:
                    print(f"[{run_num}/{total_runs}] Arm {arm} | Task: {task.task_id} ({condition}) ... DONE ({elapsed_s:.1f}s | {turns_label} | Cost: ${cost_usd:.4f} | {outcome_str})", flush=True)

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
    parser.add_argument("--arms", nargs="+", default=ALL_ARMS, help="List of arms to evaluate (or 'all')")
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum number of user tasks to evaluate per arm")
    parser.add_argument("--output-dir", default="results", help="Directory to save traces and report tables")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task partitioning and bootstrap")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps per task")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing checkpoints and rerun all tasks")
    parser.add_argument("--allow-mock-llm", action="store_true", help="Allow fallback to simulated mock LLM for offline testing")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable live real-time step telemetry stream")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable live real-time step telemetry stream")

    args = parser.parse_args()

    arms = ALL_ARMS if (args.arms == ["all"] or "all" in args.arms) else args.arms

    campaign = MultiArmCampaign(
        output_dir=args.output_dir,
        arms=arms,
        split=args.split,
        seed=args.seed,
        max_steps=args.max_steps,
        n_boot=args.n_boot,
        allow_mock_llm=args.allow_mock_llm,
        verbose=args.verbose,
    )
    result = campaign.run(force_rerun=args.force_rerun, max_tasks=args.max_tasks, verbose=args.verbose)
    print(f"Campaign completed. Executed: {result['executed']}, Skipped: {result['skipped']}", flush=True)
    print(f"Traces written to: {result['trace_path']}", flush=True)
    for name, path in result["files"].items():
        print(f"  {name}: {path}", flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, TypeError):
        pass
    main()
