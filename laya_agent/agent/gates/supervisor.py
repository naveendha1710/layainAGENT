"""Supervisor Gate ([B]) evaluating step trajectory and loop health.

Evaluates trajectory state using the categorical ``choice`` primitive:
  * ``continue``: Making normal forward progress toward completing the user goal.
  * ``nudge``: Stuck repeating identical tool calls or drifting from objective.
  * ``stop``: Task goal is fully satisfied and ready for final answer synthesis.

Fail-open rule: Defaults to ``Decision.CONTINUE`` when confidence < tau_supervisor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from laya_agent.agent.gates.base import Decision, Gate, GateResult


SUPERVISOR_SCHEMA = {
    "status": {
        "type": "choice",
        "instructions": "Evaluate agent trajectory progress from recent actions and observations.",
        "criteria": {
            "continue": "Making normal forward progress toward completing the user goal",
            "nudge": "Stuck repeating identical tool calls or drifting from objective",
            "stop": "Task goal is fully satisfied and ready for final answer",
        },
    }
}


class SupervisorGate(Gate):
    """Supervisor gate checking trajectory progress and execution health."""

    def __init__(
        self,
        tau: float = 0.70,
        mode: str = "laya",  # "laya" | "off"
        engine: Optional[Any] = None,
        device: str = "cuda",
        load_weights: Optional[bool] = None,
    ) -> None:
        super().__init__(
            name="supervisor",
            tau=tau,
            enabled=(mode != "off"),
            device=device,
        )
        self.mode = mode
        self.load_weights = load_weights
        self.engine = engine
        if self.engine is None and self.mode == "laya":
            from laya_agent.agent.providers.laya import LayaProvider

            self.engine = LayaProvider(device=device, load_weights=load_weights)

    def default(self, ctx: Dict[str, Any]) -> GateResult:
        """Fail-open utility fallback returning Decision.CONTINUE."""
        return GateResult(
            decision=Decision.CONTINUE,
            confidence=0.0,
            latency_ms=0.0,
            fallback_used=True,
            raw_output={
                "reason": "fail_open_default",
                "gate": self.name,
            },
        )

    def _evaluate(self, ctx: Dict[str, Any]) -> GateResult:
        if self.mode == "off":
            return GateResult(
                decision=Decision.CONTINUE,
                confidence=1.0,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"status": "continue", "mode": "off"},
            )

        # Allow external engine dependency injection
        if self.engine is not None:
            if hasattr(self.engine, "predict"):
                res = self.engine.predict(ctx, schema=SUPERVISOR_SCHEMA)
            elif callable(self.engine):
                res = self.engine(ctx)
            else:
                res = {}

            status = res.get("status", {}).get("choice", "continue")
            confidence = float(res.get("status", {}).get("confidence", 0.90))

            mapping = {
                "continue": Decision.CONTINUE,
                "nudge": Decision.NUDGE,
                "stop": Decision.STOP,
            }
            decision = mapping.get(status, Decision.CONTINUE)
            return GateResult(
                decision=decision,
                confidence=confidence,
                latency_ms=0.0,
                fallback_used=False,
                raw_output=res,
            )

        # Built-in trajectory heuristic analyzer for offline testing / execution
        trajectory: List[Any] = ctx.get("trajectory", [])
        tool_records = [
            rec for rec in trajectory if getattr(rec, "module", None) == "tool"
        ]

        # 1. Repetitive tool loop check (nudge trigger)
        if len(tool_records) >= 3:
            last_3_tools = [rec.tool for rec in tool_records[-3:]]
            if last_3_tools[0] and len(set(last_3_tools)) == 1:
                return GateResult(
                    decision=Decision.NUDGE,
                    confidence=0.95,
                    latency_ms=0.0,
                    fallback_used=False,
                    raw_output={"reason": "detected_repeated_tool_loop", "tool": last_3_tools[0]},
                )

        # 2. Goal completion check (stop trigger)
        recent_obs = str(ctx.get("recent_observation", ""))
        if "[TASK_COMPLETE]" in recent_obs or "goal fully satisfied" in recent_obs.lower():
            return GateResult(
                decision=Decision.STOP,
                confidence=0.98,
                latency_ms=0.0,
                fallback_used=False,
                raw_output={"reason": "goal_completion_detected"},
            )

        # 3. Default forward progress
        return GateResult(
            decision=Decision.CONTINUE,
            confidence=0.90,
            latency_ms=0.0,
            fallback_used=False,
            raw_output={"reason": "forward_progress"},
        )


__all__ = ["SUPERVISOR_SCHEMA", "SupervisorGate"]
