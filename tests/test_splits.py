"""Tests for AgentDojo adapter, task splitting, and pairing ratios.

Covers:
  - AgentDojoAdapter generating tasks across the 4 suites (workspace, banking, slack, travel).
  - 1:2 benign-to-injection pairing ratio enforcement per user task.
  - Deterministic 25 Dev / 50 Test user-task partition.
  - Total run counts (75 Dev runs, 150 Test runs = 225 runs total).
  - Complete split isolation (zero cross-contamination between Dev and Test).
  - JSON serialization and deserialization roundtrip.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.eval.adapters.agentdojo import (
    SUITES,
    AgentDojoAdapter,
    TaskRunSpec,
    UserTaskTriplet,
)
from laya_agent.eval.splits import (
    SplitResult,
    TaskSplitter,
    generate_default_splits,
    load_splits,
    save_splits,
)


def test_agentdojo_adapter_task_generation() -> None:
    adapter = AgentDojoAdapter()
    user_tasks = adapter.list_user_tasks()

    # Must provide at least 75 tasks to cover 25 Dev + 50 Test
    assert len(user_tasks) >= 75

    # Check that all 4 suites are present
    suites_present = {t.suite for t in user_tasks}
    assert suites_present == set(SUITES)

    # Every user task must strictly have 1 benign and 2 injection runs
    for triplet in user_tasks:
        runs = triplet.runs
        assert len(runs) == 3

        benign = triplet.benign
        assert not benign.is_injection
        assert benign.condition == "benign"
        assert benign.ground_truth_label == "benign"

        inj1 = triplet.injection_1
        assert inj1.is_injection
        assert inj1.condition == "injection_1"
        assert inj1.ground_truth_label == "injection"
        assert inj1.injection_payload is not None

        inj2 = triplet.injection_2
        assert inj2.is_injection
        assert inj2.condition == "injection_2"
        assert inj2.ground_truth_label == "injection"
        assert inj2.injection_payload is not None

        # All 3 runs share the same pair_id
        assert benign.pair_id == inj1.pair_id == inj2.pair_id


def test_task_splitter_counts_and_ratios() -> None:
    split = generate_default_splits(seed=42)

    # Exactly 25 Dev user tasks and 50 Test user tasks
    assert len(split.dev_user_tasks) == 25
    assert len(split.test_user_tasks) == 50

    # 1:2 ratio -> 25 * 3 = 75 Dev runs, 50 * 3 = 150 Test runs
    assert len(split.dev_runs) == 75
    assert len(split.test_runs) == 150

    # Verify counts per condition in Dev
    dev_benign = [r for r in split.dev_runs if r.condition == "benign"]
    dev_inj1 = [r for r in split.dev_runs if r.condition == "injection_1"]
    dev_inj2 = [r for r in split.dev_runs if r.condition == "injection_2"]
    assert len(dev_benign) == 25
    assert len(dev_inj1) == 25
    assert len(dev_inj2) == 25

    # Verify counts per condition in Test
    test_benign = [r for r in split.test_runs if r.condition == "benign"]
    test_inj1 = [r for r in split.test_runs if r.condition == "injection_1"]
    test_inj2 = [r for r in split.test_runs if r.condition == "injection_2"]
    assert len(test_benign) == 50
    assert len(test_inj1) == 50
    assert len(test_inj2) == 50


def test_task_splitter_zero_cross_contamination() -> None:
    """Verify zero overlap between Dev and Test user tasks."""
    split = generate_default_splits(seed=42)

    dev_user_ids = {t.user_task_id for t in split.dev_user_tasks}
    test_user_ids = {t.user_task_id for t in split.test_user_tasks}
    assert len(dev_user_ids.intersection(test_user_ids)) == 0

    dev_run_ids = {r.task_id for r in split.dev_runs}
    test_run_ids = {r.task_id for r in split.test_runs}
    assert len(dev_run_ids.intersection(test_run_ids)) == 0


def test_task_splitter_determinism() -> None:
    """Verify that same seed yields identical task assignments in exact order."""
    split1 = generate_default_splits(seed=42)
    split2 = generate_default_splits(seed=42)

    dev1_ids = [t.user_task_id for t in split1.dev_user_tasks]
    dev2_ids = [t.user_task_id for t in split2.dev_user_tasks]
    assert dev1_ids == dev2_ids

    test1_ids = [t.user_task_id for t in split1.test_user_tasks]
    test2_ids = [t.user_task_id for t in split2.test_user_tasks]
    assert test1_ids == test2_ids


def test_task_splitter_insufficient_tasks_raises() -> None:
    splitter = TaskSplitter(seed=42, dev_size=25, test_size=50)
    adapter = AgentDojoAdapter()
    tasks = adapter.list_user_tasks()[:10]  # Only 10 tasks, need 75

    with pytest.raises(ValueError, match="Insufficient user tasks"):
        splitter.partition(tasks)


def test_splits_json_serialization_roundtrip() -> None:
    split = generate_default_splits(seed=42)

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        temp_path = fh.name

    try:
        save_splits(split, temp_path)
        reloaded = load_splits(temp_path)

        assert len(reloaded.dev_user_tasks) == 25
        assert len(reloaded.test_user_tasks) == 50
        assert len(reloaded.dev_runs) == 75
        assert len(reloaded.test_runs) == 150

        # Ensure content matches
        original_dev_ids = [t.user_task_id for t in split.dev_user_tasks]
        reloaded_dev_ids = [t.user_task_id for t in reloaded.dev_user_tasks]
        assert original_dev_ids == reloaded_dev_ids
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
