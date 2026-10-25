"""Trajectory state, token counters, and step budgets for the ReAct loop.

Carried through one task run and consumed by the gate hooks and trace logger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class StepRecord:
    """A single recorded step inside a task trajectory."""

    step: int
    module: str  # scorer|filter|router|firewall|supervisor|llm|tool
    decision: str
    confidence: float
    fallback_used: bool
    latency_ms: float
    llm_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool: Optional[str] = None
    tool_ok: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000.0)


@dataclass
class AgentState:
    """Mutable trajectory state for one task run."""

    task_id: str
    pair_id: str
    split: str
    arm: str
    run_idx: int

    max_steps: int = 10
    model_tier: str = "big"

    step: int = 0
    loop_step: int = 0
    finished: bool = False
    final_answer: Optional[str] = None
    stop_reason: Optional[str] = None

    messages: List[Dict[str, str]] = field(default_factory=list)
    trajectory: List[StepRecord] = field(default_factory=list)

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_calls: int = 0
    total_gate_latency_ms: float = 0.0

    def get_user_prompt(self) -> str:
        """Find the original user prompt from messages."""
        for m in self.messages:
            if m.get("role") == "user":
                return m.get("content", "")
        return self.messages[0]["content"] if self.messages else ""

    def record_step(self, record: StepRecord) -> None:
        self.trajectory.append(record)
        self.step += 1
        if record.module not in {"scorer", "filter"}:
            self.loop_step += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        if record.module == "llm":
            self.total_llm_calls += 1
        self.total_gate_latency_ms += record.latency_ms

    def can_continue(self) -> bool:
        return (not self.finished) and self.loop_step < self.max_steps

    def to_trace_ctx(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pair_id": self.pair_id,
            "split": self.split,
            "arm": self.arm,
            "run_idx": self.run_idx,
        }


__all__ = ["AgentState", "StepRecord"]
