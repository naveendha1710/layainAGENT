"""Unit and integration tests for Test Set Evaluation Campaign (run_test_campaign.py).

Verifies:
  * Execution of multi-arm test campaign across core & ablation arms.
  * Atomic per-arm JSONL trace streaming (results/test/arm_{arm}.jsonl).
  * Checkpoint-resume skipping existing runs without duplicates.
  * Generation of publication asset bundle: summary JSON, Markdown, and LaTeX ablation tables.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.eval.adapters.agentdojo import AgentDojoAdapter
from laya_agent.eval.run_test_campaign import run_test_campaign
from laya_agent.eval.splits import generate_default_splits, save_splits


def test_run_test_campaign_execution_and_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "test_results"
        splits_path = Path(tmp_dir) / "splits_test.json"

        # Generate canonical splits and save
        splits = generate_default_splits(seed=42)
        save_splits(splits, splits_path)

        # Run campaign on subset of arms with fast bootstrap (n_boot=10)
        test_arms = ["A", "B", "D"]
        res = run_test_campaign(
            output_dir=out_dir,
            arms=test_arms,
            splits_file=splits_path,
            n_boot=10,
            seed=42,
            max_steps=5,
            force_rerun=True,
        )

        assert res is not None
        arm_metrics = res["arm_metrics"]
        for arm in test_arms:
            assert arm in arm_metrics

        # Verify per-arm JSONL trace files
        for arm in test_arms:
            arm_trace = out_dir / f"arm_{arm}.jsonl"
            assert arm_trace.exists()
            with open(arm_trace, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) > 0

        combined_trace = out_dir / "eval_campaign_test.jsonl"
        assert combined_trace.exists()

        # Verify generated table artifacts
        files = res["saved_files"]
        for expected_key in ["summary_json", "ablation_md", "ablation_tex", "gate_md", "gate_tex"]:
            assert expected_key in files
            assert files[expected_key].exists()

        # Check content of Markdown ablation table
        with open(files["ablation_md"], "r", encoding="utf-8") as f:
            md_content = f.read()
            for arm in test_arms:
                assert f"**{arm}**" in md_content

        # Verify checkpoint-resume skips all tasks on rerun without force_rerun
        res_cached = run_test_campaign(
            output_dir=out_dir,
            arms=test_arms,
            splits_file=splits_path,
            n_boot=10,
            seed=42,
            max_steps=5,
            force_rerun=False,
        )
        assert res_cached["campaign_result"]["executed"] == 0
        assert res_cached["campaign_result"]["skipped"] == len(test_arms) * 150
