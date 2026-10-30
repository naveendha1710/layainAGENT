"""Unit tests for the publication visualization module (laya_agent.analysis.visualize)."""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

from laya_agent.analysis.visualize import (
    DEFAULT_ARMS_ORDER,
    extract_arm_statistics,
    generate_figures,
    load_trace_data,
    plot_fig1_ablation_asr_utility,
    plot_fig2_gate_latency_cdf,
    plot_fig3_call_reduction,
)


@pytest.fixture
def sample_trace_and_summary(tmp_path: Path):
    """Create temporary trace file and summary_metrics.json for testing."""
    trace_file = tmp_path / "eval_campaign_test.jsonl"
    summary_file = tmp_path / "summary_metrics.json"

    # Write dummy trace lines
    trace_records = [
        {
            "arm": "A",
            "task_id": "task_001",
            "run_idx": 0,
            "module": "llm",
            "gate_latency_ms": 0.0,
            "status": "success",
            "tier_selected": "big",
            "llm_calls": 2,
        },
        {
            "arm": "D",
            "task_id": "task_001",
            "run_idx": 0,
            "module": "scorer",
            "gate_latency_ms": 32.5,
            "tier_selected": "small",
        },
        {
            "arm": "D",
            "task_id": "task_001",
            "run_idx": 0,
            "module": "router",
            "gate_latency_ms": 34.1,
            "tool_call": "read_file",
        },
        {
            "arm": "D",
            "task_id": "task_001",
            "run_idx": 0,
            "module": "firewall",
            "gate_latency_ms": 31.8,
            "injection_detected": False,
        },
        {
            "arm": "D",
            "task_id": "task_001",
            "run_idx": 0,
            "module": "supervisor",
            "gate_latency_ms": 33.0,
            "loop_terminated": False,
        },
        {"malformed": "json line without standard fields"},
    ]

    with open(trace_file, "w", encoding="utf-8") as fh:
        for rec in trace_records:
            fh.write(json.dumps(rec) + "\n")
        fh.write("invalid json string\n")

    summary_data = {
        "arms": {
            "A": {"asr": 0.95, "benign_utility": 0.88, "llm_calls_per_task": 2.4},
            "B_regex": {"asr": 0.70, "benign_utility": 0.85, "llm_calls_per_task": 2.1},
            "B": {"asr": 0.15, "benign_utility": 0.86, "llm_calls_per_task": 2.0},
            "C_always_small": {"asr": 0.95, "benign_utility": 0.65, "llm_calls_per_task": 1.0},
            "C_random": {"asr": 0.95, "benign_utility": 0.78, "llm_calls_per_task": 1.7},
            "C": {"asr": 0.95, "benign_utility": 0.87, "llm_calls_per_task": 1.4},
            "D": {"asr": 0.08, "benign_utility": 0.87, "llm_calls_per_task": 1.3},
        }
    }

    with open(summary_file, "w", encoding="utf-8") as fh:
        json.dump(summary_data, fh)

    return trace_file, summary_file


def test_load_trace_data(sample_trace_and_summary):
    trace_file, _ = sample_trace_and_summary
    records, latencies, total_runs = load_trace_data(trace_file)

    assert len(records) == 6  # 6 valid JSON objects
    assert len(latencies) == 4  # scorer, router, firewall, supervisor
    assert all(30.0 < lat < 36.0 for lat in latencies)
    assert total_runs == 2  # 2 distinct (arm, task_id, run_idx): Arm A and Arm D


def test_load_trace_data_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_trace_data(tmp_path / "non_existent.jsonl")


def test_extract_arm_statistics_from_summary(sample_trace_and_summary):
    trace_file, summary_file = sample_trace_and_summary
    with open(summary_file, "r", encoding="utf-8") as fh:
        summary_data = json.load(fh)

    stats = extract_arm_statistics(summary_data, input_path=trace_file)

    for arm in DEFAULT_ARMS_ORDER:
        assert arm in stats
        assert "asr" in stats[arm]
        assert "benign_utility" in stats[arm]
        assert "llm_calls_per_task" in stats[arm]

    # Verification of scaling to percentages
    assert stats["A"]["asr"] == pytest.approx(95.0)
    assert stats["D"]["asr"] == pytest.approx(8.0)
    assert stats["D"]["benign_utility"] == pytest.approx(87.0)
    assert stats["D"]["llm_calls_per_task"] == pytest.approx(1.3)


def test_extract_arm_statistics_with_nested_ci_format():
    nested_summary = {
        "metrics": {
            "A": {
                "asr": {"point_estimate": 0.92, "mean": 0.92},
                "benign_utility": {"point_estimate": 0.89, "mean": 0.89},
                "mean_llm_calls": {"point_estimate": 2.5, "mean": 2.5},
            },
            "D": {
                "asr": {"point_estimate": 0.05, "mean": 0.05},
                "benign_utility": {"point_estimate": 0.88, "mean": 0.88},
                "mean_llm_calls": {"point_estimate": 1.2, "mean": 1.2},
            },
        }
    }
    stats = extract_arm_statistics(nested_summary)
    assert stats["A"]["asr"] == pytest.approx(92.0)
    assert stats["D"]["asr"] == pytest.approx(5.0)
    assert stats["D"]["llm_calls_per_task"] == pytest.approx(1.2)


def test_individual_plotters(tmp_path: Path):
    dummy_stats = {
        arm: {"asr": 10.0, "benign_utility": 85.0, "llm_calls_per_task": 1.5}
        for arm in DEFAULT_ARMS_ORDER
    }

    fig1_png, fig1_pdf = plot_fig1_ablation_asr_utility(dummy_stats, tmp_path)
    assert fig1_png.exists() and fig1_png.stat().st_size > 0
    assert fig1_pdf.exists() and fig1_pdf.stat().st_size > 0

    latencies = [31.5, 32.8, 33.2, 34.0, 35.1, 42.0]
    fig2_png, fig2_pdf = plot_fig2_gate_latency_cdf(latencies, tmp_path)
    assert fig2_png.exists() and fig2_png.stat().st_size > 0
    assert fig2_pdf.exists() and fig2_pdf.stat().st_size > 0

    fig3_png, fig3_pdf = plot_fig3_call_reduction(dummy_stats, tmp_path)
    assert fig3_png.exists() and fig3_png.stat().st_size > 0
    assert fig3_pdf.exists() and fig3_pdf.stat().st_size > 0


def test_generate_figures_end_to_end(sample_trace_and_summary, tmp_path: Path):
    trace_file, summary_file = sample_trace_and_summary
    output_dir = tmp_path / "figures_out"

    generated = generate_figures(
        input_path=trace_file,
        output_dir=output_dir,
        summary_path=summary_file,
        create_zip=True,
    )

    assert "fig1_png" in generated and generated["fig1_png"].exists()
    assert "fig1_pdf" in generated and generated["fig1_pdf"].exists()
    assert "fig2_png" in generated and generated["fig2_png"].exists()
    assert "fig2_pdf" in generated and generated["fig2_pdf"].exists()
    assert "fig3_png" in generated and generated["fig3_png"].exists()
    assert "fig3_pdf" in generated and generated["fig3_pdf"].exists()
    assert "figures_zip" in generated and generated["figures_zip"].exists()


def test_cli_execution(sample_trace_and_summary, tmp_path: Path):
    trace_file, summary_file = sample_trace_and_summary
    output_dir = tmp_path / "cli_figures"

    cmd = [
        sys.executable,
        "-m",
        "laya_agent.analysis.visualize",
        "--input",
        str(trace_file),
        "--summary",
        str(summary_file),
        "--output",
        str(output_dir),
        "--zip",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"CLI failed with stderr: {proc.stderr}"
    assert "Figures successfully generated" in proc.stdout
    assert (output_dir / "fig1_ablation_asr_utility.png").exists()
    assert (output_dir / "fig2_gate_latency_cdf.pdf").exists()
    assert (output_dir / "fig3_call_reduction.png").exists()
