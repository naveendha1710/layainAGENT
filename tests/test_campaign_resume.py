"""Unit tests for campaign checkpointing resumption (--resume) and max_tokens bounding."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.agent.core.loop import GateHook, LoopConfig, ReActLoop
from laya_agent.agent.core.state import AgentState
from laya_agent.eval.adapters.agentdojo import TaskRunSpec
from laya_agent.eval.campaign import MultiArmCampaign, DefaultMockLLM


def test_react_loop_defaults_max_tokens_512() -> None:
    state = AgentState(
        task_id="test_task",
        pair_id="test_pair",
        split="dev",
        arm="A",
        run_idx=0,
        max_steps=5,
    )
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "Task finished. [TASK_COMPLETE]"
    mock_resp.model = "nvidia/nemotron-3.5-lightning-30b-a3b"
    mock_resp.input_tokens = 20
    mock_resp.output_tokens = 10
    mock_llm.chat_complete.return_value = mock_resp

    loop = ReActLoop(
        state=state,
        hooks={},
        llm_client=mock_llm,
        tool_registry=MagicMock(),
    )
    res_state = loop.run("Do something")

    assert res_state.finished
    assert mock_llm.chat_complete.called
    call_kwargs = mock_llm.chat_complete.call_args.kwargs
    assert call_kwargs.get("max_tokens") == 512


def test_campaign_resume_skips_completed_tasks(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    mock_task = TaskRunSpec(
        task_id="task_resume_test_benign",
        pair_id="pair_resume_test",
        suite="travel",
        user_prompt="Book flight",
        condition="benign",
        is_injection=False,
    )

    campaign = MultiArmCampaign(
        output_dir=str(tmp_path),
        arms=["A"],
        split="dev",
        allow_mock_llm=True,
        verbose=False,
        resume=True,
    )

    # First run: should execute 1 task
    res1 = campaign.run(tasks=[mock_task], force_rerun=False, resume=True)
    assert res1["executed"] == 1
    assert res1["skipped"] == 0

    trace_file = tmp_path / "eval_campaign_dev.jsonl"
    assert trace_file.exists()

    # Second run with resume=True: should skip the task
    capsys.readouterr()  # clear buffer
    res2 = campaign.run(tasks=[mock_task], force_rerun=False, resume=True)
    assert res2["executed"] == 0
    assert res2["skipped"] == 1

    captured = capsys.readouterr()
    assert "[Skip] Arm A | Task: task_resume_test_benign (benign) ... ALREADY COMPLETED" in captured.out

    # Third run with force_rerun=True: should rerun the task
    res3 = campaign.run(tasks=[mock_task], force_rerun=True, resume=False)
    assert res3["executed"] == 1
    assert res3["skipped"] == 0
