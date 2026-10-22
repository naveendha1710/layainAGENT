"""Unit test for the step-level JSONL trace schema.

Verifies that dummy trace records serialize to valid JSONL, validate against the
Pydantic schema, capture the git commit hash, and round-trip through
``TraceLogger.validate_line``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# Ensure the project root (parent of the ``laya_agent`` package) is importable
# regardless of how pytest is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest  # noqa: E402

from laya_agent.logging.trace import (  # noqa: E402
    COMMIT_HASH,
    TraceLogger,
    TraceRecord,
    ValidationError,
)


def _dummy_record() -> TraceRecord:
    return TraceRecord(
        task_id="task_001",
        pair_id="pair_001",
        split="dev",
        arm="A",
        run_idx=0,
        step=0,
        module="scorer",
        model_tier="big",
        decision="big",
        confidence=0.95,
        fallback_used=False,
        gate_latency_ms=35.3,
        llm_latency_ms=0.0,
        input_tokens=120,
        output_tokens=0,
        tool=None,
        tool_ok=None,
        ground_truth_label=None,
    )


def test_trace_record_serializes_to_jsonl() -> None:
    record = _dummy_record()
    line = record.to_jsonl()
    assert isinstance(line, str)
    # Must be parseable JSON.
    payload = json.loads(line)
    assert payload["task_id"] == "task_001"
    assert payload["arm"] == "A"
    assert payload["commit_hash"] == COMMIT_HASH


def test_trace_record_validates_all_19_fields() -> None:
    record = _dummy_record()
    # All documented fields plus commit_hash must be present.
    expected = {
        "task_id", "pair_id", "split", "arm", "run_idx", "step", "module",
        "model_tier", "decision", "confidence", "fallback_used",
        "gate_latency_ms", "llm_latency_ms", "input_tokens", "output_tokens",
        "tool", "tool_ok", "ground_truth_label", "commit_hash",
    }
    assert set(record.model_dump().keys()) == expected


def test_trace_record_rejects_invalid_split() -> None:
    with pytest.raises(ValidationError):
        TraceRecord(
            task_id="t", pair_id="p", split="staging", arm="A", run_idx=0,
            step=0, module="llm", model_tier="big", decision="call_llm",
            confidence=1.0, fallback_used=False, gate_latency_ms=0.0,
            llm_latency_ms=100.0, input_tokens=10, output_tokens=5,
        )


def test_trace_record_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        TraceRecord(
            task_id="t", pair_id="p", split="dev", arm="A", run_idx=0, step=0,
            module="llm", model_tier="big", decision="call_llm", confidence=1.5,
            fallback_used=False, gate_latency_ms=0.0, llm_latency_ms=100.0,
            input_tokens=10, output_tokens=5,
        )


def test_trace_logger_round_trip() -> None:
    record = _dummy_record()
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as fh:
        path = fh.name
    try:
        logger = TraceLogger(path)
        logger.log(
            task_id=record.task_id,
            pair_id=record.pair_id,
            split=record.split,
            arm=record.arm,
            run_idx=record.run_idx,
            step=record.step,
            module=record.module,
            model_tier=record.model_tier,
            decision=record.decision,
            confidence=record.confidence,
            fallback_used=record.fallback_used,
            gate_latency_ms=record.gate_latency_ms,
            llm_latency_ms=record.llm_latency_ms,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
        )
        with open(path, "r", encoding="utf-8") as fh:
            written = fh.read().strip()
        # Round-trip: the written line must re-validate against the schema.
        reparsed = TraceLogger.validate_line(written)
        assert reparsed.task_id == "task_001"
        assert reparsed.commit_hash == COMMIT_HASH
        assert reparsed.confidence == 0.95
    finally:
        os.remove(path)


def test_commit_hash_is_captured() -> None:
    # commit_hash must be a non-empty string pinned from `git rev-parse HEAD`
    # (or "unknown" when outside a git repository).
    assert isinstance(COMMIT_HASH, str)
    assert len(COMMIT_HASH) > 0
