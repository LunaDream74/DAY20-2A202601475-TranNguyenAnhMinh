"""Inspectable LangGraph orchestration for the research team."""

from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class GraphState(TypedDict):
    """LangGraph envelope; the Pydantic model remains the shared domain state."""

    research: ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self._deadline: float | None = None

    def _node(self, agent: BaseAgent, payload: GraphState) -> GraphState:
        if self._deadline is not None and perf_counter() > self._deadline:
            raise AgentExecutionError("Workflow exceeded TIMEOUT_SECONDS")
        state = payload["research"]
        with trace_span(f"agent.{agent.name}", {"iteration": state.iteration}) as span:
            state = agent.run(state)
        state.add_trace_event("agent_span", span)
        return {"research": state}

    @staticmethod
    def _route(payload: GraphState) -> str:
        return payload["research"].next_agent

    def build(self) -> Any:
        """Compile nodes, worker-to-supervisor edges, and conditional routing.

        The Supervisor is deliberately deterministic: learners can inspect exactly why
        each branch was selected instead of debugging an opaque LLM router.
        """

        graph = StateGraph(GraphState)
        graph.add_node("supervisor", lambda state: self._node(self.supervisor, state))
        graph.add_node("researcher", lambda state: self._node(self.researcher, state))
        graph.add_node("analyst", lambda state: self._node(self.analyst, state))
        graph.add_node("writer", lambda state: self._node(self.writer, state))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route,
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "done": END},
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph with iteration, timeout, and graceful-failure guardrails."""

        self._deadline = perf_counter() + self.settings.timeout_seconds
        try:
            result: GraphState = self.build().invoke(
                {"research": state},
                config={"recursion_limit": self.settings.max_iterations * 3 + 2},
            )
            return result["research"]
        except Exception as exc:
            state.errors.append(str(exc))
            state.add_trace_event(
                "workflow_error", {"type": type(exc).__name__, "message": str(exc)}
            )
            if state.final_answer is None:
                state.final_answer = (
                    "The research workflow could not complete safely. Inspect `errors` and `trace` "
                    "in the returned state before retrying."
                )
            return state
        finally:
            self._deadline = None
