from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class FakeLLMClient(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, system_prompt: str, user_prompt: str, *, web_search: bool = False
    ) -> LLMResponse:
        self.calls += 1
        if web_search:
            return LLMResponse(
                content="Grounded research notes.",
                input_tokens=10,
                output_tokens=20,
                cost_usd=0.001,
                sources=[
                    SourceDocument(
                        title="Primary source",
                        url="https://example.com/source",
                        snippet="Evidence",
                    )
                ],
            )
        if "Analyst" in system_prompt:
            return LLMResponse(content="The evidence supports the main claim.", output_tokens=10)
        return LLMResponse(content="Final supported answer [1].", output_tokens=10)


def test_supervisor_routes_from_missing_artifacts() -> None:
    settings = Settings(_env_file=None, MAX_ITERATIONS=6)
    supervisor = SupervisorAgent(settings)
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    supervisor.run(state)
    assert state.next_agent == "researcher"

    state.research_notes = "notes"
    supervisor.run(state)
    assert state.next_agent == "analyst"

    state.analysis_notes = "analysis"
    supervisor.run(state)
    assert state.next_agent == "writer"


def test_workflow_runs_expected_handoffs_with_fake_model() -> None:
    fake = FakeLLMClient()
    settings = Settings(_env_file=None, MAX_ITERATIONS=6, TIMEOUT_SECONDS=30)
    workflow = MultiAgentWorkflow(
        settings=settings,
        supervisor=SupervisorAgent(settings),
        researcher=ResearcherAgent(SearchClient(fake)),
        analyst=AnalystAgent(fake),
        writer=WriterAgent(fake),
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    result = workflow.run(state)

    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert result.final_answer == "Final supported answer [1]."
    assert [item.agent for item in result.agent_results] == ["researcher", "analyst", "writer"]
    assert result.input_tokens == 10
    assert result.output_tokens == 40
    assert not result.errors


def test_search_removes_blocks_backed_by_discarded_urls() -> None:
    sources = [SourceDocument(title="Kept", url="https://kept.example", snippet="Evidence")]
    content = (
        "Supported claim (https://kept.example)\n\n"
        "Discarded claim (https://discarded.example)\n\nGeneral caveat."
    )

    result = SearchClient._keep_grounded_blocks(content, sources)

    assert "Supported claim" in result
    assert "Discarded claim" not in result
    assert "General caveat" in result


def test_writer_labels_invalid_citation() -> None:
    class InvalidWriterClient(FakeLLMClient):
        def complete(
            self, system_prompt: str, user_prompt: str, *, web_search: bool = False
        ) -> LLMResponse:
            return LLMResponse(content="Supported [1], but invalid [2].")

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.sources = [SourceDocument(title="A", url="https://a.example", snippet="A")]

    result = WriterAgent(InvalidWriterClient()).run(state)

    assert result.final_answer == "Supported [1], but invalid [citation unavailable]."
    assert result.errors == ["Writer returned invalid citations: [2]"]
