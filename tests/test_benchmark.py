from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def test_benchmark_calculates_inspectable_metrics() -> None:
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.sources = [
            SourceDocument(title="A", url="https://a.example", snippet="A"),
            SourceDocument(title="B", url="https://b.example", snippet="B"),
        ]
        state.final_answer = "A supported conclusion [1]. Another supported conclusion [2]."
        state.input_tokens = 100
        state.output_tokens = 50
        state.estimated_cost_usd = 0.01
        return state

    _, metrics = run_benchmark("test", "Explain benchmark metrics", runner)

    assert metrics.citation_coverage == 1.0
    assert metrics.input_tokens == 100
    assert metrics.estimated_cost_usd == 0.01
    assert metrics.failure_rate == 0.0
    assert metrics.quality_score is not None
