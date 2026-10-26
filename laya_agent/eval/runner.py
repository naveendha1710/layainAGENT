"""Batch runner supporting Arm A, B, C, and D evaluation runs.

Executes tasks end-to-end and emits 19-field JSONL trace records via
``tracing/trace.py``, including the ``commit_hash`` provenance field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from laya_agent.agent.core.baseline import (
    ArmABaseline,
    ArmB,
    ArmC,
    ArmD,
    create_agent,
)
from laya_agent.agent.core.state import AgentState
from laya_agent.agent.tools.base import ToolRegistry
from laya_agent.tracing.trace import TraceLogger


class ArmARunner:
    """Batch executor for Arm A, B, C, and D tasks."""

    def __init__(self, *, trace_path: str = "eval_arm_A_smoke.jsonl") -> None:
        self.trace_path = trace_path
        self.logger = TraceLogger(trace_path)

    def run_task(
        self,
        *,
        task_id: str,
        pair_id: str,
        split: str,
        arm: str = "A",
        run_idx: int = 0,
        initial_prompt: str,
        tool_registry: ToolRegistry,
        llm_client: Any,
        policy_store: Optional[Any] = None,
        policy_top_k: int = 10,
        candidate_policies: Optional[Sequence[Any]] = None,
        max_steps: int = 10,
        scorer_engine: Optional[Any] = None,
        filter_engine: Optional[Any] = None,
        router_engine: Optional[Any] = None,
        firewall_engine: Optional[Any] = None,
        supervisor_engine: Optional[Any] = None,
        tau_scorer: float = 0.70,
        tau_filter: float = 0.75,
        tau_router: float = 0.80,
        tau_inj: float = 0.90,
        tau_rel: float = 0.70,
        tau_supervisor: float = 0.70,
        seed: Optional[int] = 42,
        ground_truth_label: Optional[str] = None,
        load_weights: Optional[bool] = None,
    ) -> AgentState:
        state = AgentState(
            task_id=task_id,
            pair_id=pair_id,
            split=split,
            arm=arm,
            run_idx=run_idx,
            max_steps=max_steps,
        )
        agent = create_agent(
            arm=arm,
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            scorer_engine=scorer_engine,
            filter_engine=filter_engine,
            router_engine=router_engine,
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_scorer=tau_scorer,
            tau_filter=tau_filter,
            tau_router=tau_router,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
            seed=seed,
            load_weights=load_weights,
        )
        agent.run(initial_prompt, candidate_policies=candidate_policies)

        # Emit one trace record per trajectory step.
        ctx = state.to_trace_ctx()
        for rec in state.trajectory:
            gate_latency = (
                rec.latency_ms
                if rec.module in {"scorer", "filter", "router", "firewall", "supervisor"}
                else 0.0
            )
            llm_latency = rec.llm_latency_ms if rec.module == "llm" else 0.0
            gt_label = rec.raw.get("ground_truth_label") or ground_truth_label

            self.logger.log(
                task_id=ctx["task_id"],
                pair_id=ctx["pair_id"],
                split=ctx["split"],
                arm=ctx["arm"],
                run_idx=ctx["run_idx"],
                step=rec.step,
                module=rec.module,
                model_tier=state.model_tier,
                decision=rec.decision,
                confidence=rec.confidence,
                fallback_used=rec.fallback_used,
                gate_latency_ms=gate_latency,
                llm_latency_ms=llm_latency,
                input_tokens=rec.input_tokens,
                output_tokens=rec.output_tokens,
                tool=rec.tool,
                tool_ok=rec.tool_ok,
                ground_truth_label=gt_label,
            )
        return state

    def run_batch(
        self,
        tasks: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[AgentState]:
        return [self.run_task(**{**t, **kwargs}) for t in tasks]


__all__ = ["ArmARunner"]
