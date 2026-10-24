"""Agent controllers for Arms A, B, C, and D ("build once, toggle via config").

Arm A: Unaugmented ReAct baseline (all gates off).
Arm B_regex: Heuristic security baseline (firewall regex, supervisor off).
Arm B: LAYA Safety & Control (Context Firewall laya + Supervisor laya).
Arm C_always_small: Static cost baseline (scorer always_small, firewall laya, supervisor laya).
Arm C_random: Random routing baseline (scorer random, firewall laya, supervisor laya).
Arm C: LAYA Cost Routing (scorer laya, firewall laya, supervisor laya).
Arm D: Full LAYA Architecture (scorer laya, filter laya, router laya, firewall laya, supervisor laya).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from laya_agent.agent.core.loop import GateHook, ReActLoop
from laya_agent.agent.core.state import AgentState
from laya_agent.agent.gates.filter import FilterGate
from laya_agent.agent.gates.firewall import ContextFirewall
from laya_agent.agent.gates.router import RouterGate
from laya_agent.agent.gates.scorer import ScorerGate
from laya_agent.agent.gates.supervisor import SupervisorGate


class ArmABaseline:
    """Unaugmented ReAct baseline (Arm A)."""

    def __init__(
        self,
        state: AgentState,
        llm_client: Any,
        tool_registry: Any,
        policy_store: Optional[Any] = None,
        policy_top_k: int = 10,
        max_steps: int = 10,
    ) -> None:
        self.state = state
        self.state.max_steps = max_steps
        self.loop = ReActLoop(
            state=state,
            hooks={},  # all gates off for Arm A
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
        )

    def run(
        self,
        initial_prompt: str,
        candidate_policies: Optional[Sequence[Any]] = None,
    ) -> AgentState:
        return self.loop.run(initial_prompt, candidate_policies=candidate_policies)


class ArmB:
    """Arm B: Safety & Control (Context Firewall + Supervisor)."""

    def __init__(
        self,
        state: AgentState,
        llm_client: Any,
        tool_registry: Any,
        policy_store: Optional[Any] = None,
        policy_top_k: int = 10,
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
            policy_store=policy_store,
            policy_top_k=policy_top_k,
        )

    def run(
        self,
        initial_prompt: str,
        candidate_policies: Optional[Sequence[Any]] = None,
    ) -> AgentState:
        return self.loop.run(initial_prompt, candidate_policies=candidate_policies)


class ArmC:
    """Arm C: Cost Routing (Scorer + Context Firewall + Supervisor)."""

    def __init__(
        self,
        state: AgentState,
        llm_client: Any,
        tool_registry: Any,
        policy_store: Optional[Any] = None,
        policy_top_k: int = 10,
        max_steps: int = 10,
        scorer_mode: str = "laya",
        firewall_mode: str = "laya",
        supervisor_mode: str = "laya",
        scorer_engine: Optional[Any] = None,
        firewall_engine: Optional[Any] = None,
        supervisor_engine: Optional[Any] = None,
        tau_scorer: float = 0.70,
        tau_inj: float = 0.90,
        tau_rel: float = 0.70,
        tau_supervisor: float = 0.70,
        seed: Optional[int] = 42,
    ) -> None:
        self.state = state
        self.state.max_steps = max_steps

        scorer_gate = ScorerGate(
            tau=tau_scorer,
            mode=scorer_mode,
            engine=scorer_engine,
            seed=seed,
        )
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
            "scorer": GateHook(
                name="scorer",
                gate=scorer_gate,
                ctx_builder=lambda st: {
                    "state": st,
                    "initial_prompt": st.messages[0]["content"] if st.messages else "",
                },
            ),
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
            policy_store=policy_store,
            policy_top_k=policy_top_k,
        )

    def run(
        self,
        initial_prompt: str,
        candidate_policies: Optional[Sequence[Any]] = None,
    ) -> AgentState:
        return self.loop.run(initial_prompt, candidate_policies=candidate_policies)


class ArmD:
    """Arm D: Full LAYA Architecture (Scorer + Filter + Router + Firewall + Supervisor)."""

    def __init__(
        self,
        state: AgentState,
        llm_client: Any,
        tool_registry: Any,
        policy_store: Optional[Any] = None,
        policy_top_k: int = 10,
        max_steps: int = 10,
        scorer_mode: str = "laya",
        filter_mode: str = "laya",
        router_mode: str = "laya",
        firewall_mode: str = "laya",
        supervisor_mode: str = "laya",
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
    ) -> None:
        self.state = state
        self.state.max_steps = max_steps

        scorer_gate = ScorerGate(
            tau=tau_scorer,
            mode=scorer_mode,
            engine=scorer_engine,
            seed=seed,
        )
        filter_gate = FilterGate(
            tau=tau_filter,
            mode=filter_mode,
            engine=filter_engine,
        )
        router_gate = RouterGate(
            tau=tau_router,
            mode=router_mode,
            engine=router_engine,
        )
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
            "scorer": GateHook(
                name="scorer",
                gate=scorer_gate,
                ctx_builder=lambda st: {
                    "state": st,
                    "initial_prompt": st.messages[0]["content"] if st.messages else "",
                },
            ),
            "filter": GateHook(
                name="filter",
                gate=filter_gate,
                ctx_builder=lambda st: {
                    "state": st,
                },
            ),
            "router": GateHook(
                name="router",
                gate=router_gate,
                ctx_builder=lambda st: {
                    "state": st,
                    "messages": st.messages,
                    "trajectory": st.trajectory,
                },
            ),
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
            policy_store=policy_store,
            policy_top_k=policy_top_k,
        )

    def run(
        self,
        initial_prompt: str,
        candidate_policies: Optional[Sequence[Any]] = None,
    ) -> AgentState:
        return self.loop.run(initial_prompt, candidate_policies=candidate_policies)


def create_agent(
    arm: str,
    state: AgentState,
    llm_client: Any,
    tool_registry: Any,
    policy_store: Optional[Any] = None,
    policy_top_k: int = 10,
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
) -> Any:
    """Factory creating the appropriate agent controller for the requested arm."""
    arm_key = arm.strip()

    if arm_key in {"A", "a"}:
        return ArmABaseline(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
        )
    elif arm_key.lower() == "b_regex":
        return ArmB(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            firewall_mode="regex",
            supervisor_mode="off",
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
        )
    elif arm_key in {"B", "b"}:
        return ArmB(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            firewall_mode="laya",
            supervisor_mode="laya",
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
        )
    elif arm_key.lower() == "c_always_small":
        return ArmC(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            scorer_mode="always_small",
            firewall_mode="laya",
            supervisor_mode="laya",
            scorer_engine=scorer_engine,
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_scorer=tau_scorer,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
            seed=seed,
        )
    elif arm_key.lower() == "c_random":
        return ArmC(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            scorer_mode="random",
            firewall_mode="laya",
            supervisor_mode="laya",
            scorer_engine=scorer_engine,
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_scorer=tau_scorer,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
            seed=seed,
        )
    elif arm_key in {"C", "c"}:
        return ArmC(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            scorer_mode="laya",
            firewall_mode="laya",
            supervisor_mode="laya",
            scorer_engine=scorer_engine,
            firewall_engine=firewall_engine,
            supervisor_engine=supervisor_engine,
            tau_scorer=tau_scorer,
            tau_inj=tau_inj,
            tau_rel=tau_rel,
            tau_supervisor=tau_supervisor,
            seed=seed,
        )
    elif arm_key in {"D", "d"}:
        return ArmD(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            scorer_mode="laya",
            filter_mode="laya",
            router_mode="laya",
            firewall_mode="laya",
            supervisor_mode="laya",
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
        )
    else:
        # Default fallback to Arm D (Full LAYA)
        return ArmD(
            state=state,
            llm_client=llm_client,
            tool_registry=tool_registry,
            policy_store=policy_store,
            policy_top_k=policy_top_k,
            max_steps=max_steps,
            scorer_mode="laya",
            filter_mode="laya",
            router_mode="laya",
            firewall_mode="laya",
            supervisor_mode="laya",
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
        )


__all__ = ["ArmABaseline", "ArmB", "ArmC", "ArmD", "create_agent"]
