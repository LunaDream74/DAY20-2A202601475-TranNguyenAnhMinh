"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Choose the next missing artifact using an inspectable routing policy."""

        if state.iteration >= self.settings.max_iterations:
            route = "done"
            if state.final_answer is None:
                state.errors.append("Stopped after reaching MAX_ITERATIONS")
        elif not state.research_notes:
            route = "researcher"
        elif not state.analysis_notes:
            route = "analyst"
        elif not state.final_answer:
            route = "writer"
        else:
            route = "done"

        state.next_agent = route
        state.record_route(route)
        state.add_trace_event(
            "routing_decision",
            {
                "next": route,
                "iteration": state.iteration,
                "has_research": bool(state.research_notes),
                "has_analysis": bool(state.analysis_notes),
                "has_answer": bool(state.final_answer),
            },
        )
        return state
