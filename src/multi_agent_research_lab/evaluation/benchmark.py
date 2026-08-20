"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure one run using a transparent, deterministic quality proxy.

    The quality score is not a factuality judge. It rewards answer presence/depth, citation
    coverage, and source diversity so learners can inspect exactly how it was calculated.
    """

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(f"{type(exc).__name__}: {exc}")
    latency = perf_counter() - started
    citation_coverage = _citation_coverage(state)
    quality_score = _quality_score(state, citation_coverage)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        estimated_cost_usd=state.estimated_cost_usd,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if state.errors or not state.final_answer else 0.0,
        notes="Heuristic quality proxy; estimated model-token cost excludes web-search tool fees.",
    )
    return state, metrics


def _citation_coverage(state: ResearchState) -> float:
    if not state.sources or not state.final_answer:
        return 0.0
    numbered = {int(value) for value in re.findall(r"\[(\d+)\]", state.final_answer)}
    cited_urls = {
        index
        for index, source in enumerate(state.sources, start=1)
        if source.url and source.url in state.final_answer
    }
    valid = {index for index in numbered | cited_urls if 1 <= index <= len(state.sources)}
    return len(valid) / len(state.sources)


def _quality_score(state: ResearchState, citation_coverage: float) -> float:
    if not state.final_answer:
        return 0.0
    word_count = len(state.final_answer.split())
    answer_presence = 2.0
    depth = min(word_count / 250, 1.0) * 3.0
    citations = citation_coverage * 3.0
    source_diversity = min(len(state.sources) / 3, 1.0) * 2.0
    return round(answer_presence + depth + citations + source_diversity, 1)
