"""Deterministic 25 Dev / 50 Test user-task splitter with 1:2 pairing ratio.

Enforces:
  * Exactly 25 Dev user tasks (75 execution runs).
  * Exactly 50 Test user tasks (150 execution runs).
  * Exactly 1 clean benign run + 2 matched injection twins per user task.
  * Deterministic pseudo-random partitioning seeded by seed=42.
  * Zero cross-split contamination: all runs of a user task share the same split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Sequence

from laya_agent.eval.adapters.agentdojo import (
    AgentDojoAdapter,
    TaskRunSpec,
    UserTaskTriplet,
)


@dataclass
class SplitResult:
    """Container for dev and test splits."""

    dev_user_tasks: List[UserTaskTriplet]
    test_user_tasks: List[UserTaskTriplet]

    @property
    def dev_runs(self) -> List[TaskRunSpec]:
        """All individual execution runs in the Dev split (25 * 3 = 75 runs)."""
        runs = []
        for triplet in self.dev_user_tasks:
            runs.extend(triplet.runs)
        return runs

    @property
    def test_runs(self) -> List[TaskRunSpec]:
        """All individual execution runs in the Test split (50 * 3 = 150 runs)."""
        runs = []
        for triplet in self.test_user_tasks:
            runs.extend(triplet.runs)
        return runs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "dev_user_tasks_count": len(self.dev_user_tasks),
                "test_user_tasks_count": len(self.test_user_tasks),
                "dev_runs_count": len(self.dev_runs),
                "test_runs_count": len(self.test_runs),
                "ratio": "1_benign_to_2_injections",
            },
            "dev": [
                {
                    "user_task_id": t.user_task_id,
                    "suite": t.suite,
                    "runs": [r.to_dict() for r in t.runs],
                }
                for t in self.dev_user_tasks
            ],
            "test": [
                {
                    "user_task_id": t.user_task_id,
                    "suite": t.suite,
                    "runs": [r.to_dict() for r in t.runs],
                }
                for t in self.test_user_tasks
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitResult":
        def parse_triplet(item: Dict[str, Any]) -> UserTaskTriplet:
            runs = [TaskRunSpec.from_dict(r) for r in item["runs"]]
            benign = next(r for r in runs if r.condition == "benign")
            inj1 = next(r for r in runs if r.condition == "injection_1")
            inj2 = next(r for r in runs if r.condition == "injection_2")
            return UserTaskTriplet(
                user_task_id=item["user_task_id"],
                suite=item["suite"],
                benign=benign,
                injection_1=inj1,
                injection_2=inj2,
            )

        dev_tasks = [parse_triplet(item) for item in data.get("dev", [])]
        test_tasks = [parse_triplet(item) for item in data.get("test", [])]
        return cls(dev_user_tasks=dev_tasks, test_user_tasks=test_tasks)


class TaskSplitter:
    """Deterministic partitioner for AgentDojo user tasks."""

    def __init__(
        self,
        seed: int = 42,
        dev_size: int = 25,
        test_size: int = 50,
    ) -> None:
        self.seed = seed
        self.dev_size = dev_size
        self.test_size = test_size

    def partition(
        self,
        user_tasks: Sequence[UserTaskTriplet],
    ) -> SplitResult:
        """Deterministically partition user tasks into Dev (25) and Test (50)."""
        required = self.dev_size + self.test_size
        if len(user_tasks) < required:
            raise ValueError(
                f"Insufficient user tasks: got {len(user_tasks)}, "
                f"expected at least {required} ({self.dev_size} dev + {self.test_size} test)."
            )

        # Sort canonically by user_task_id before shuffling with pinned seed
        sorted_tasks = sorted(user_tasks, key=lambda t: t.user_task_id)
        rng = Random(self.seed)
        shuffled = list(sorted_tasks)
        rng.shuffle(shuffled)

        dev_tasks = shuffled[: self.dev_size]
        test_tasks = shuffled[self.dev_size : self.dev_size + self.test_size]

        split_result = SplitResult(
            dev_user_tasks=dev_tasks,
            test_user_tasks=test_tasks,
        )
        self.validate(split_result)
        return split_result

    def validate(self, split: SplitResult) -> None:
        """Validate split invariants: task counts, run counts, ratio, and isolation."""
        if len(split.dev_user_tasks) != self.dev_size:
            raise ValueError(
                f"Dev task count mismatch: expected {self.dev_size}, got {len(split.dev_user_tasks)}"
            )
        if len(split.test_user_tasks) != self.test_size:
            raise ValueError(
                f"Test task count mismatch: expected {self.test_size}, got {len(split.test_user_tasks)}"
            )

        # Verify no cross-contamination between dev and test user tasks
        dev_ids = {t.user_task_id for t in split.dev_user_tasks}
        test_ids = {t.user_task_id for t in split.test_user_tasks}
        overlap = dev_ids.intersection(test_ids)
        if overlap:
            raise ValueError(f"Task leakage detected between Dev and Test splits: {overlap}")

        # Verify exact 1:2 ratio and run counts
        expected_dev_runs = self.dev_size * 3
        expected_test_runs = self.test_size * 3

        if len(split.dev_runs) != expected_dev_runs:
            raise ValueError(
                f"Dev runs count mismatch: expected {expected_dev_runs}, got {len(split.dev_runs)}"
            )
        if len(split.test_runs) != expected_test_runs:
            raise ValueError(
                f"Test runs count mismatch: expected {expected_test_runs}, got {len(split.test_runs)}"
            )

        for triplet in split.dev_user_tasks + split.test_user_tasks:
            runs = triplet.runs
            benign_count = sum(1 for r in runs if not r.is_injection and r.condition == "benign")
            injection_count = sum(1 for r in runs if r.is_injection and r.condition.startswith("injection_"))
            if benign_count != 1 or injection_count != 2:
                raise ValueError(
                    f"User task {triplet.user_task_id} violates 1:2 ratio: "
                    f"benign={benign_count}, injection={injection_count}"
                )


def generate_default_splits(
    seed: int = 42,
    adapter: Optional[AgentDojoAdapter] = None,
) -> SplitResult:
    """Generate the canonical frozen 25 Dev / 50 Test split."""
    if adapter is None:
        adapter = AgentDojoAdapter()
    tasks = adapter.list_user_tasks()
    splitter = TaskSplitter(seed=seed, dev_size=25, test_size=50)
    return splitter.partition(tasks)


def save_splits(split: SplitResult, filepath: str | Path = "splits.json") -> None:
    """Save split specification to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2)


def load_splits(filepath: str | Path = "splits.json") -> SplitResult:
    """Load split specification from a JSON file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SplitResult.from_dict(data)


__all__ = [
    "SplitResult",
    "TaskSplitter",
    "generate_default_splits",
    "load_splits",
    "save_splits",
]
