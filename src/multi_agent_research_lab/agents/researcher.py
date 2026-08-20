"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate sources and grounded notes through the configured search service."""

        result = self.search_client.search(state.request.query, state.request.max_sources)
        if not result.sources:
            raise ValidationError("Researcher received no provider-verified web sources")
        state.sources = result.sources
        state.research_notes = result.synthesis
        state.record_usage(result.input_tokens, result.output_tokens, result.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=result.synthesis,
                metadata={
                    "source_count": len(result.sources),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cost_usd": result.cost_usd,
                },
            )
        )
        return state
