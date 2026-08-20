"""Optional deterministic critic for inspecting final-answer grounding."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Append transparent checks without making another model call."""

        if not state.final_answer:
            raise ValidationError("Critic requires final_answer")
        cited = {int(value) for value in re.findall(r"\[(\d+)\]", state.final_answer)}
        valid = {index for index in cited if 1 <= index <= len(state.sources)}
        invalid = sorted(cited - valid)
        findings = (
            f"Cited {len(valid)}/{len(state.sources)} available sources; "
            f"invalid citation indices: {invalid or 'none'}."
        )
        state.agent_results.append(
            AgentResult(agent=AgentName.CRITIC, content=findings, metadata={"invalid": invalid})
        )
        state.add_trace_event("critic_check", {"findings": findings})
        return state
