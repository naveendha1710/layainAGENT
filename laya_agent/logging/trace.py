"""Step-level JSONL telemetry logger with schema validation and git-commit provenance.

Implements the strict trace schema documented in ``docs/system_architecture.md``
(Section 5). Every gate evaluation and LLM/tool step writes one flat JSONL
record. Records are validated with Pydantic before atomic append.

The trace also pins the exact ``git rev-parse HEAD`` commit hash (Section 6:
Trace & Run Versioning) so every benchmark run is strictly reproducible.

Trace schema (18 documented fields + commit_hash provenance field):

  1. task_id            str     Unique AgentDojo user task identifier
  2. pair_id            str     Identifier linking benign run + 2 injection twins
  3. split              str     "dev" or "test"
  4. arm                str     Experiment arm id
  5. run_idx            int     Seed/run index
  6. step               int     0-indexed step number within trajectory
  7. module             str     scorer|filter|router|firewall|supervisor|llm|tool
  8. model_tier         str     "big"|"small"|"none"
  9. decision           str     standardized Decision enum value
  10. confidence        float    calibrated LAYA confidence in [0.0, 1.0]
  11. fallback_used     bool    True if confidence < tau or on provider exception
  12. gate_latency_ms   float   synchronized GPU gate execution time (ms)
  13. llm_latency_ms    float   hosted LLM API round-trip time (ms)
  14. input_tokens      int     prompt/context tokens billed this step
  15. output_tokens     int     completion tokens billed this step
  16. tool              str|None name of executed tool, if applicable
  17. tool_ok           bool|None whether the tool executed without runtime error
  18. ground_truth_label str|None step-level oracle label
  19. commit_hash       str     git rev-parse HEAD (run provenance)
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


def get_commit_hash() -> str:
    """Return ``git rev-parse HEAD`` for the current working tree.

    Falls back to ``"unknown"`` when not run inside a git repository, so the
    logger never crashes a run outside of version control (e.g. Colab notebooks
    that copy the source tree without a .git directory).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd(),
        )
        return result.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


# Captured once at import time so every record in a run shares the same commit.
COMMIT_HASH: str = get_commit_hash()


class TraceRecord(BaseModel):
    """Single validated step-level JSONL trace record."""

    task_id: str
    pair_id: str
    split: str
    arm: str
    run_idx: int
    step: int
    module: str
    model_tier: str
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_used: bool
    gate_latency_ms: float
    llm_latency_ms: float
    input_tokens: int
    output_tokens: int
    tool: Optional[str] = None
    tool_ok: Optional[bool] = None
    ground_truth_label: Optional[str] = None
    commit_hash: str = COMMIT_HASH

    @field_validator("split")
    def _split_must_be_dev_or_test(cls, v: str) -> str:
        if v not in ("dev", "test"):
            raise ValueError('split must be "dev" or "test"')
        return v

    @field_validator("module")
    def _module_must_be_valid(cls, v: str) -> str:
        allowed = {
            "scorer", "filter", "router", "firewall", "supervisor",
            "llm", "tool",
        }
        if v not in allowed:
            raise ValueError(f"module must be one of {sorted(allowed)}")
        return v

    @field_validator("model_tier")
    def _tier_must_be_valid(cls, v: str) -> str:
        if v not in ("big", "small", "none"):
            raise ValueError('model_tier must be "big", "small", or "none"')
        return v

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line (sorted keys for deterministic output)."""
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))


class TraceLogger:
    """Thread-safe, atomic-appending JSONL step logger.

    Each record is validated against the TraceRecord schema before being written.
    Writes are guarded by a lock and flushed immediately so a crash mid-run cannot
    lose a trailing record (atomic per-line appends).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        # Ensure the parent directory exists.
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

    def log(
        self,
        *,
        task_id: str,
        pair_id: str,
        split: str,
        arm: str,
        run_idx: int,
        step: int,
        module: str,
        model_tier: str,
        decision: str,
        confidence: float,
        fallback_used: bool,
        gate_latency_ms: float,
        llm_latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        tool: Optional[str] = None,
        tool_ok: Optional[bool] = None,
        ground_truth_label: Optional[str] = None,
    ) -> None:
        """Validate and append one trace record to the JSONL file."""
        record = TraceRecord(
            task_id=task_id,
            pair_id=pair_id,
            split=split,
            arm=arm,
            run_idx=run_idx,
            step=step,
            module=module,
            model_tier=model_tier,
            decision=decision,
            confidence=confidence,
            fallback_used=fallback_used,
            gate_latency_ms=gate_latency_ms,
            llm_latency_ms=llm_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool=tool,
            tool_ok=tool_ok,
            ground_truth_label=ground_truth_label,
        )
        line = record.to_jsonl()
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    @classmethod
    def validate_line(cls, line: str) -> TraceRecord:
        """Parse and validate a single JSONL line, returning the record."""
        return TraceRecord(**json.loads(line))


__all__ = [
    "TraceRecord",
    "TraceLogger",
    "get_commit_hash",
    "COMMIT_HASH",
    "ValidationError",
]
