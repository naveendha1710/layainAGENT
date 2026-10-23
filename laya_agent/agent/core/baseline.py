"""Arm A baseline: ReAct agent with all gates bypassed (off).

Standard prompt assembly + direct tool execution against the hosted Big Tier
model. No LAYA gates are active; this is the absolute reference point for all
subsequent gate evaluations.
"""

from __future__ import annotations

from typing import Any

from laya_agent.agent.core.loop import ReActLoop
from laya_agent.agent.core.state import AgentState


class ArmABaseline:
    """Unaugmented ReAct baseline (Arm A)."""

    def __init__(
        self,
        state: AgentState,
        llm_client: Any,
        tool_registry: Any,
        max_steps: int = 10,
    ) -> None:
        self.state = state
        self.state.max_steps = max_steps
        self.loop = ReActLoop(
            state=state,
            hooks={},  # all gates off for Arm A
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

    def run(self, initial_prompt: str) -> AgentState:
        return self.loop.run(initial_prompt)


__all__ = ["ArmABaseline"]
