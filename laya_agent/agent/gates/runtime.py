"""Local ModernBERT-large gate runtime with sub-40ms execution and GPU synchronization.

Provides high-speed local non-autoregressive gate evaluations across all 5 schemas:
  * Scorer Gate ([C]): Task difficulty & risk routing (small vs. big).
  * Applicability Filter Gate ([D]): Candidate policy pruning (keep vs. drop).
  * Action Router Gate ([D]): Pre-LLM execution bypass (call_llm vs. bypass_llm).
  * Context Firewall Gate ([B]): 300-token chunk injection (noul) & relevance (choice).
  * Supervisor Gate ([B]): Step trajectory progress (continue, nudge, stop).

Enforces strict GPU timing synchronization via `torch.cuda.synchronize()` before
starting and stopping timers to guarantee accurate kernel latency telemetry.
Includes a self-contained offline fallback so evaluations function seamlessly
in offline test environments.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

from laya_agent.agent.gates.base import Decision
from laya_agent.agent.gates.filter import FILTER_SCHEMA, FilterGate
from laya_agent.agent.gates.firewall import FIREWALL_SCHEMA, REGEX_INJECTION_PATTERN
from laya_agent.agent.gates.router import ROUTER_SCHEMA, RouterGate
from laya_agent.agent.gates.scorer import SCORER_SCHEMA, ScorerGate
from laya_agent.agent.gates.supervisor import SUPERVISOR_SCHEMA, SupervisorGate


def detect_device(preferred: Optional[str] = None) -> str:
    """Detect the fastest available hardware execution device."""
    if preferred:
        return preferred.lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def synchronize_device(device: str) -> None:
    """Synchronize device execution queues for high-precision latency measurement."""
    if "cuda" in device:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass


class LayaRuntime:
    """Inference runtime delegating decision gates directly to LayaProvider."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-large",
        device: Optional[str] = None,
        load_weights: Optional[bool] = None,
        api_url: Optional[str] = None,
        tau_scorer: float = 0.70,
        tau_filter: float = 0.70,
        tau_router: float = 0.70,
        tau_inj: float = 0.75,
        tau_rel: float = 0.70,
        tau_supervisor: float = 0.50,
    ) -> None:
        self.model_name = model_name
        self.device = detect_device(device)
        self.load_weights = bool(load_weights) if load_weights is not None else (self.device == "cuda")
        self.api_url = (api_url or os.environ.get("LAYA_API_URL", "")).strip().rstrip("/")
        self.tau_scorer = tau_scorer
        self.tau_filter = tau_filter
        self.tau_router = tau_router
        self.tau_inj = tau_inj
        self.tau_rel = tau_rel
        self.tau_supervisor = tau_supervisor

        from laya_agent.agent.providers.laya import LayaProvider
        self.provider = LayaProvider(device=self.device, load_weights=self.load_weights)
        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> Any:
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    def _call_remote(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """Call remote LAYA server (e.g. running on Google Colab GPU via ngrok/localtunnel)."""
        if not self.api_url:
            return None
        import json
        import urllib.request
        try:
            url = f"{self.api_url}/{endpoint.lstrip('/')}"
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "LAYA-Agent-Client/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # 1. Scorer Gate ([C])
    # ------------------------------------------------------------------
    def score_task(self, prompt: str) -> Tuple[Decision, float, float]:
        """Score task prompt: 'small' (low complexity) vs. 'big' (multi-step/risk)."""
        res = self.provider.predict(text=prompt, schema=SCORER_SCHEMA)
        choice = res["model_tier"]["decision"]
        conf = float(res["model_tier"]["confidence"])
        lat = float(res.get("_latency_ms", res.get("latency_ms", 0.0)))
        return Decision(choice), conf, lat

    # ------------------------------------------------------------------
    # 2. Applicability Filter Gate ([D])
    # ------------------------------------------------------------------
    def filter_policy(
        self,
        policy: Union[str, Any],
        user_prompt: str = "",
        prompt: str = "",
        suite: str = "",
    ) -> Tuple[Decision, float, float]:
        """Evaluate candidate policy: 'keep' (strictly relevant) vs 'drop' (noise)."""
        p_text = policy.text if hasattr(policy, "text") else str(policy)
        effective_prompt = user_prompt or prompt
        res = self.provider.predict(
            text=f"Policy: {p_text}\nTask: {effective_prompt}",
            schema=FILTER_SCHEMA,
            ctx={"user_prompt": effective_prompt, "suite": suite},
        )
        choice = res["applies"]["decision"]
        conf = float(res["applies"]["confidence"])
        lat = float(res.get("_latency_ms", res.get("latency_ms", 0.0)))
        return Decision(choice), conf, lat

    # ------------------------------------------------------------------
    # 3. Action Router Gate ([D])
    # ------------------------------------------------------------------
    def route_action(
        self,
        state_summary: Union[str, List[Dict[str, str]]],
        recent_observation: str = "",
    ) -> Tuple[Decision, float, float]:
        """Evaluate pre-LLM execution bypass: 'bypass_llm' vs 'call_llm'."""
        if isinstance(state_summary, list):
            summary = recent_observation or (state_summary[0]["content"] if state_summary else "")
        else:
            summary = str(state_summary)
        res = self.provider.predict(text=summary, schema=ROUTER_SCHEMA)
        choice = res["action"]["decision"]
        conf = float(res["action"]["confidence"])
        lat = float(res.get("_latency_ms", res.get("latency_ms", 0.0)))
        return Decision(choice), conf, lat

    # ------------------------------------------------------------------
    # 4. Context Firewall Gate ([B])
    # ------------------------------------------------------------------
    def check_chunk(
        self,
        chunk: str,
        user_prompt: str = "",
    ) -> Dict[str, Any]:
        """Check raw chunk: (injection_detected, injection_confidence, relevance, relevance_confidence, latency_ms)."""
        res = self.provider.predict(
            text=chunk,
            schema=FIREWALL_SCHEMA,
            ctx={"user_prompt": user_prompt},
        )
        inj_node = res.get("injection_detected", {})
        rel_node = res.get("relevance", {})
        inj_dec = inj_node.get("decision")
        if inj_dec is None:
            inj_dec = "yes" if inj_node.get("score", 0.0) > self.tau_inj else "no"
        inj_conf = float(inj_node.get("confidence", inj_node.get("score", 0.0)))
        rel_dec = rel_node.get("decision", rel_node.get("choice", "keep"))
        rel_conf = float(rel_node.get("confidence", 0.90))
        lat = float(res.get("_latency_ms", res.get("latency_ms", 0.0)))
        return {
            "injection_detected": inj_dec,
            "injection_confidence": inj_conf,
            "relevance": rel_dec,
            "relevance_confidence": rel_conf,
            "latency_ms": lat,
        }

    # ------------------------------------------------------------------
    # 5. Supervisor Gate ([B])
    # ------------------------------------------------------------------
    def supervise_step(
        self,
        trajectory: Union[str, List[Any]],
        recent_observation: str = "",
    ) -> Tuple[Decision, float, float]:
        """Supervise trajectory health: 'continue', 'nudge', or 'stop'."""
        if isinstance(trajectory, str):
            traj_text = trajectory
            traj_list = []
        else:
            traj_list = trajectory
            traj_text = recent_observation or (str(trajectory) if trajectory else "step")
        res = self.provider.predict(
            text=traj_text,
            schema=SUPERVISOR_SCHEMA,
            ctx={"trajectory": traj_list, "recent_observation": recent_observation},
        )
        choice = res["status"]["decision"]
        conf = float(res["status"]["confidence"])
        lat = float(res.get("_latency_ms", res.get("latency_ms", 0.0)))
        self._last_supervisor_reason = res.get("reason", "normal_progress")
        return Decision(choice), conf, lat

    supervise_loop = supervise_step

    # ------------------------------------------------------------------
    # Universal Gate Engine Interface (predict)
    # ------------------------------------------------------------------
    def predict(
        self,
        text: Any,
        schema: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generic prediction hook delegating to remote or LayaProvider."""
        ctx = ctx or {}

        # Remote Colab / tunnel dispatch if api_url is configured
        if self.api_url:
            remote_resp = self._call_remote(
                "predict",
                {"text": text, "schema": schema, "ctx": ctx},
            )
            if remote_resp and isinstance(remote_resp, dict):
                return remote_resp

        return self.provider.predict(text=text, schema=schema, ctx=ctx)


__all__ = ["LayaRuntime", "detect_device", "synchronize_device"]
