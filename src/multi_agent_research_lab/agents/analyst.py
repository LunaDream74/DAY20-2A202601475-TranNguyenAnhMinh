"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Compare evidence, identify conclusions, and expose uncertainty."""

        if not state.research_notes:
            raise ValidationError("Analyst requires research_notes")
        response = self.llm_client.complete(
            "You are the Analyst in a research team. Do not search or write the final answer. "
            "Evaluate only the supplied evidence. Identify agreements, conflicts, weak evidence, "
            "and the strongest supported conclusions.",
            f"Question: {state.request.query}\n\nResearch notes:\n{state.research_notes}",
        )
        state.analysis_notes = response.content
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
