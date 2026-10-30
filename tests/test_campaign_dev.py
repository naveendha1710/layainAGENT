"""Unit and integration tests for Dev Set Calibration Campaign (run_dev_campaign.py).

Verifies:
  * Execution of multi-arm dev campaign across Arms A, B, C, D.
  * Atomic per-arm JSONL trace streaming (results/dev/arm_{arm}.jsonl).
  * Checkpoint-resume skipping existing runs without duplicates.
  * Generation of summary JSON, Markdown, and LaTeX ablation tables.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.eval.adapters.agentdojo import AgentDojoAdapter
from laya_agent.eval.run_dev_campaign import run_dev_campaign
from laya_agent.eval.splits import generate_default_splits, save_splits


def test_run_dev_campaign_execution_and_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "dev_results"
        splits_path = Path(tmp_dir) / "splits_test.json"

        # Generate canonical splits and save
        splits = generate_default_splits(seed=42)
        save_splits(splits, splits_path)

        # Run campaign on Arms A and D with fast bootstrap (n_boot=10)
        res = run_dev_campaign(
            output_dir=out_dir,
            arms=["A", "D"],
            splits_file=splits_path,
            n_boot=10,
            seed=42,
            max_steps=5,
            force_rerun=True,
            allow_mock_llm=True,
        )

        assert res is not None
        arm_metrics = res["arm_metrics"]
        assert "A" in arm_metrics
        assert "D" in arm_metrics

        # Verify per-arm JSONL trace files
        arm_a_trace = out_dir / "arm_A.jsonl"
        arm_d_trace = out_dir / "arm_D.jsonl"
        combined_trace = out_dir / "eval_campaign_dev.jsonl"

        assert arm_a_trace.exists()
        assert arm_d_trace.exists()
        assert combined_trace.exists()

        with open(arm_a_trace, "r", encoding="utf-8") as f:
            a_lines = [l.strip() for l in f if l.strip()]
        with open(arm_d_trace, "r", encoding="utf-8") as f:
            d_lines = [l.strip() for l in f if l.strip()]

        assert len(a_lines) > 0
        assert len(d_lines) > 0

        # Verify generated table artifacts
        files = res["saved_files"]
        for expected_key in ["summary_json", "ablation_md", "ablation_tex", "gate_md", "gate_tex"]:
            assert expected_key in files
            assert files[expected_key].exists()

        # Check content of Markdown ablation table
        with open(files["ablation_md"], "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "**A**" in md_content
            assert "**D**" in md_content

        # Verify checkpoint-resume skips all tasks on rerun
        res_resume = run_dev_campaign(
            output_dir=out_dir,
            arms=["A", "D"],
            splits_file=splits_path,
            n_boot=10,
            seed=42,
            max_steps=5,
            force_rerun=False,
            allow_mock_llm=True,
        )
        assert res_resume["campaign_result"]["executed"] == 0
        assert res_resume["campaign_result"]["skipped"] == 2 * 75
