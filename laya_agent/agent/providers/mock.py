"""Deterministic mock provider for offline CI and unit tests."""

from __future__ import annotations

import re
import time
import types
from typing import Any, Dict, List, Optional

from laya_agent.agent.gates.filter import FILTER_SCHEMA
from laya_agent.agent.gates.firewall import FIREWALL_SCHEMA, REGEX_INJECTION_PATTERN
from laya_agent.agent.gates.router import ROUTER_SCHEMA
from laya_agent.agent.gates.scorer import SCORER_SCHEMA
from laya_agent.agent.gates.supervisor import SUPERVISOR_SCHEMA
from laya_agent.agent.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Offline deterministic provider utilizing schema-based heuristics."""

    def __init__(self, device: str = "cpu", load_weights: Optional[bool] = None) -> None:
        self.device = device
        self.load_weights = False if load_weights is None else load_weights
        self.runtime = types.SimpleNamespace(load_weights=self.load_weights, device=self.device)
        self.gate_latency_ms: float = 0.0
        self._last_supervisor_reason: str = "normal_progress"

    def predict(
        self,
        text: Any = None,
        schema: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Predict using deterministic heuristic evaluations."""
        # Support flexible argument orders: predict(schema, text) or predict(text, schema)
        if isinstance(text, dict) and isinstance(schema, str):
            schema, text = text, schema
        elif schema is None and isinstance(text, dict) and any(k in text for k in ("model_tier", "applies", "action", "injection_detected", "status")):
            schema = text
            text = kwargs.get("text", "")
        elif schema is None and "schema" in kwargs:
            schema = kwargs["schema"]
            text = text if text is not None else kwargs.get("text", "")

        ctx = dict(ctx) if ctx is not None else {}
        if isinstance(text, dict):
            merged_ctx = dict(text)
            merged_ctx.update(ctx)
            ctx = merged_ctx
            text_str = str(ctx.get("recent_observation", "") or ctx.get("user_prompt", "") or text)
        else:
            text_str = "" if text is None else str(text)

        t0 = time.perf_counter()
        res: Dict[str, Any] = {}

        # 1. Task Scorer Schema
        if schema == SCORER_SCHEMA or "model_tier" in (schema or {}):
            prompt_lower = text_str.lower()
            big_patterns = [
                r"transfer",
                r"wire",
                r"balance.*and.*transact",
                r"recurring",
                r"plan",
                r"coordinate",
                r"schedule.*meeting.*external",
                r"export.*archive",
                r"invoice",
                r"budget",
                r"book.*flight",
                r"cancel.*reservation",
                r"admin",
                r"exfiltrate",
                r"multi-step",
                r"follow-up",
                r"perform.*follow",
            ]
            small_patterns = [
                r"^find\s+file",
                r"^read\s+",
                r"^check\s+balance",
                r"^get\s+status",
                r"^what\s+is",
                r"^look\s*up",
                r"^search\s+",
                r"^compute\s+\d+\s*\+",
                r"^simple\s+query",
            ]
            is_big = any(re.search(pat, prompt_lower) for pat in big_patterns)
            is_small = any(re.search(pat, prompt_lower) for pat in small_patterns)
            if len(text_str.split()) > 20:
                is_big = True

            tier = "small" if (is_small and not is_big) else "big"
            conf = 0.92 if tier == "small" else 0.95
            res = {"model_tier": {"choice": tier, "confidence": conf}}

        # 2. Applicability Filter Schema
        elif schema == FILTER_SCHEMA or "applies" in (schema or {}):
            p_lower = text_str.lower()
            prompt_lower = str(ctx.get("user_prompt", "")).lower()
            suite = str(ctx.get("suite", "")).lower()

            if "[drop]" in p_lower or "[noise]" in p_lower or "[irrelevant]" in p_lower:
                res = {"applies": {"choice": "drop", "confidence": 0.95}}
            elif "[keep]" in p_lower or "[relevant]" in p_lower:
                res = {"applies": {"choice": "keep", "confidence": 0.95}}
            else:
                words_policy = set(re.findall(r"\w{4,}", p_lower))
                words_prompt = set(re.findall(r"\w{4,}", prompt_lower))
                overlap = words_policy & words_prompt - {"must", "should", "never", "only", "from", "with"}
                if overlap:
                    res = {"applies": {"choice": "keep", "confidence": 0.90}}
                else:
                    res = {"applies": {"choice": "drop" if suite else "keep", "confidence": 0.85}}

        # 3. Action Router Schema
        elif schema == ROUTER_SCHEMA or "action" in (schema or {}):
            obs_lower = text_str.lower()
            if "confirmation_required" in obs_lower or "[confirmation_required]" in obs_lower:
                match = re.search(r"tool:\s*([a-zA-Z0-9_-]+)", text_str, re.IGNORECASE)
                t_name = match.group(1) if match else "confirm_action"
                res = {
                    "action": {
                        "choice": "bypass_llm",
                        "confidence": 0.95,
                        "tool_name": t_name,
                        "tool_args": {},
                    }
                }
            elif "routine follow-up" in obs_lower or "[routine_follow_up]" in obs_lower:
                res = {
                    "action": {
                        "choice": "bypass_llm",
                        "confidence": 0.90,
                        "tool_name": "noop",
                        "tool_args": {},
                    }
                }
            else:
                res = {
                    "action": {
                        "choice": "call_llm",
                        "confidence": 0.90,
                        "tool_name": None,
                        "tool_args": {},
                    }
                }

        # 4. Context Firewall Schema
        elif schema == FIREWALL_SCHEMA or "injection_detected" in (schema or {}):
            inj_score = 0.0
            if REGEX_INJECTION_PATTERN.search(text_str) or "attacker.com" in text_str.lower() or "injection" in text_str.lower():
                inj_score = 0.99
            else:
                inj_score = 0.05

            rel_choice = "keep"
            rel_conf = 0.90
            if "[noise]" in text_str.lower() or "unrelated boilerplate" in text_str.lower():
                rel_choice = "drop"
                rel_conf = 0.95

            res = {
                "injection_detected": {"score": inj_score},
                "relevance": {"choice": rel_choice, "confidence": rel_conf},
            }

        # 5. Supervisor Schema
        elif schema == SUPERVISOR_SCHEMA or "status" in (schema or {}):
            traj: List[Any] = ctx.get("trajectory", []) if isinstance(ctx, dict) else []
            tool_records = [rec for rec in traj if getattr(rec, "module", None) == "tool"]

            if len(tool_records) >= 3:
                last_3 = [getattr(rec, "tool", None) for rec in tool_records[-3:]]
                if last_3[0] and len(set(last_3)) == 1:
                    self._last_supervisor_reason = "detected_repeated_tool_loop"
                    res = {
                        "status": {"choice": "nudge", "confidence": 0.95},
                        "reason": self._last_supervisor_reason,
                    }

            if not res:
                if "[TASK_COMPLETE]" in text_str or "goal fully satisfied" in text_str.lower() or "workflow finished" in text_str.lower():
                    self._last_supervisor_reason = "goal_fully_satisfied"
                    res = {
                        "status": {"choice": "stop", "confidence": 0.98},
                        "reason": self._last_supervisor_reason,
                    }
                else:
                    self._last_supervisor_reason = "normal_progress"
                    res = {
                        "status": {"choice": "continue", "confidence": 0.90},
                        "reason": self._last_supervisor_reason,
                    }

        self.gate_latency_ms = (time.perf_counter() - t0) * 1000.0
        res["latency_ms"] = self.gate_latency_ms
        return res


__all__ = ["MockProvider"]

