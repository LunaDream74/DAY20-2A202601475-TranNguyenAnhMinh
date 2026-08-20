"""Writer agent skeleton."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize the final response while preserving source traceability."""

        if not state.analysis_notes or not state.research_notes:
            raise ValidationError("Writer requires research_notes and analysis_notes")
        source_list = "\n".join(
            f"[{index}] {source.title} - {source.url or 'no URL'}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            "You are the Writer in a research team. Answer the original question for the stated "
            "audience. Use only the supplied research and analysis. Cite factual claims with [n] "
            "matching the source list. Never use a citation number absent from that list. Clearly "
            "label uncertainty; never invent citations.",
            f"Question: {state.request.query}\nAudience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\nAnalysis:\n{state.analysis_notes}\n\n"
            f"Source list:\n{source_list or 'No URL sources were returned.'}",
        )
        cited = {int(value) for value in re.findall(r"\[(\d+)\]", response.content)}
        valid = {index for index in cited if 1 <= index <= len(state.sources)}
        content = response.content
        if state.sources and not valid:
            state.errors.append("Writer returned no valid [n] source citation")
        if cited - valid:
            state.errors.append(f"Writer returned invalid citations: {sorted(cited - valid)}")
            for index in cited - valid:
                content = content.replace(f"[{index}]", "[citation unavailable]")
        state.final_answer = content
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
