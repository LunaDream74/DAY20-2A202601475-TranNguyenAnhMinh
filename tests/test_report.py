import pytest

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.report import render_dataset_report, render_markdown_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report
    assert "structural proxy" in report


def test_dataset_report_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="At least one"):
        render_dataset_report([], "mock.jsonl")
