"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, DatasetCaseResult


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render comparable metrics and their interpretation caveat to Markdown."""

    caveat = (
        "Quality is an inspectable structural proxy, not a factuality verdict. Cost estimates "
        "cover model tokens and exclude provider web-search tool fees."
    )
    interpretation = (
        "Compare the extra latency and tokens of role separation against citation coverage and "
        "answer quality. Review traces and outputs manually before drawing conclusions."
    )
    lines = [
        "# Benchmark Report",
        "",
        f"> {caveat}",
        "",
        "| Run | Latency (s) | Input tok. | Output tok. | Cost "
        "(USD) | Quality | Citation cov. | Failure rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {item.input_tokens} "
            f"| {item.output_tokens} | {cost} | {quality} | {citation} | {failure} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
        ]
    )
    return "\n".join(lines) + "\n"


def render_dataset_report(results: list[DatasetCaseResult], dataset_name: str) -> str:
    """Render per-case and aggregate results for one system and dataset."""

    if not results:
        raise ValueError("At least one dataset result is required")
    count = len(results)
    average_concepts = sum(item.concept_coverage for item in results) / count
    average_quality = sum(item.metrics.quality_score or 0 for item in results) / count
    average_latency = sum(item.metrics.latency_seconds for item in results) / count
    total_cost = sum(item.metrics.estimated_cost_usd or 0 for item in results)
    failures = sum(1 for item in results if item.metrics.failure_rate)
    lines = [
        "# Synthetic Dataset Evaluation",
        "",
        f"> Dataset: `{dataset_name}`. These cases and expectations are synthetic, "
        "not ground truth.",
        "",
        f"System: **{results[0].run_name}** | Cases: **{count}** | "
        f"Mean concept coverage: **{average_concepts:.0%}** | "
        f"Mean structural quality: **{average_quality:.1f}/10**",
        "",
        f"Mean latency: **{average_latency:.2f}s** | Estimated total model-token cost: "
        f"**${total_cost:.4f}** | Failed cases: **{failures}/{count}**",
        "",
        "| Case | Category | Difficulty | Concepts | Quality | Citations | Latency | Failure |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        quality = item.metrics.quality_score or 0
        citations = item.metrics.citation_coverage or 0
        failure = item.metrics.failure_rate or 0
        lines.append(
            f"| {item.case_id} | {item.category} | {item.difficulty} | "
            f"{item.concept_coverage:.0%} | {quality:.1f} | {citations:.0%} | "
            f"{item.metrics.latency_seconds:.2f}s | {failure:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Scoring limits",
            "",
            "Concept coverage uses weighted phrase/alias matching. Structural quality measures "
            "answer presence, length, source diversity, and source usage. Neither establishes "
            "factual correctness; inspect the saved per-case states and apply human review.",
        ]
    )
    return "\n".join(lines) + "\n"
