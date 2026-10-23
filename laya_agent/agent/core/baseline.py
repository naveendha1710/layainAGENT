"""Agent controllers for Arm A and Arm B ("build once, toggle via config").

Arm A: Unaugmented ReAct baseline (all gates off).
Arm B_regex: Heuristic security baseline (firewall regex, supervisor off).
Arm B: LAYA Safety & Control (Context Firewall laya + Supervisor laya).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from laya_agent.agent.core.loop import GateHook, ReActLoop
from laya_agent.agent.core.state import AgentState
from laya_agent.agent.gates.firewall import ContextFirewall
from laya_agent.agent.gates.supervisor import SupervisorGate


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


class ArmB:
    """Arm B: Safety & Control (Context Firewall + Supervisor)."""

    def __init__(
        self,
        state: AgentState,
        llm_client: Any,
        tool_registry: Any,
        max_steps: int = 10,
        firewall_mode: str = "laya",
        supervisor_mode: str = "laya",
        firewall_engine: Optional[Any] = None,
        supervisor_engine: Optional[Any] = None,
        tau_inj: float = 0.90,
        tau_rel: float = 0.70,
        tau_supervisor: float = 0.70,
    ) -> None:
        self.state = state
        self.state.max_steps = max_steps

        firewall_gate = ContextFirewall(
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            mode=firewall_mode,
            engine=firewall_engine,
        )
        supervisor_gate = SupervisorGate(
            tau=tau_supervisor,
            mode=supervisor_mode,
            engine=supervisor_engine,
        )

        hooks: Dict[str, GateHook] = {
            "firewall": GateHook(
                name="firewall",
                gate=firewall_gate,
                ctx_builder=lambda st: {
                    "state": st,
                    "user_prompt": st.messages[0]["content"] if st.messages else "",
                },
            ),
            "supervisor": GateHook(
                name="supervisor",
                gate=supervisor_gate,
                ctx_builder=lambda st: {
                    "state": st,
                    "trajectory": st.trajectory,
                },
            ),
        }

        self.loop = ReActLoop(
            state=state,
            hooks=hooks,
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

    def run(self, initial_prompt: str) -> AgentState:
        return self.loop.run(initial_prompt)


def create_agent(
    arm: str,
    state: AgentState,
    llm_client: Any,
    tool_registry: Any,
    max_steps: int = 10,
    firewall_engine: Optional[Any] = None,
    supervisor_engine: Optional[Any] = None,
    tau_inj: float = 0.90,
    tau_rel: float = 0.70,
    tau_supervisor: float = 0.70,
) -> Any:
    """Factory creating the appropriate agent controller for the requested arm."""
    arm_upper = arm.upper() if arm in {"A", "B", "C", "D"} else arm

    if arm_upper == "A":
        return ArmABaseline(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            max_steps=max_steps,
        )
    elif arm_upper == "B_REGEX":
        return ArmB(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            max_steps=max_steps,
            firewall_mode="regex",
            supervisor_mode="off",
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
        )
    elif arm_upper == "B":
        return ArmB(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            max_steps=max_steps,
            firewall_mode="laya",
            supervisor_mode="laya",
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
        )
    else:
        # Default fallback to Arm B for safety
        return ArmB(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            max_steps=max_steps,
            firewall_mode="laya",
            supervisor_mode="laya",
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
        )


__all__ = ["ArmABaseline", "ArmB", "create_agent"]
