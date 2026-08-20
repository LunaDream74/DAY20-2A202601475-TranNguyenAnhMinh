"""Public schemas exchanged between CLI, agents, and evaluators."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentName(StrEnum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=5)
    max_sources: int = Field(default=5, ge=1, le=20)
    audience: str = "technical learners"


class AgentResult(BaseModel):
    agent: AgentName
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    title: str
    url: str | None = None
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    run_name: str
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    quality_score: float | None = Field(default=None, ge=0, le=10)
    citation_coverage: float | None = Field(default=None, ge=0, le=1)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    notes: str = ""


class ConceptCriterion(BaseModel):
    """One weighted concept expected in a synthetic evaluation answer."""

    name: str = Field(..., min_length=2)
    aliases: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0)


class EvaluationCase(BaseModel):
    """A synthetic research prompt with inspectable scoring expectations."""

    id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    query: str = Field(..., min_length=5)
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    audience: str = "technical learners"
    concepts: list[ConceptCriterion] = Field(..., min_length=1)
    reference_points: list[str] = Field(..., min_length=1)
    evaluation_notes: str = ""


class DatasetCaseResult(BaseModel):
    """Automatic scores and normal benchmark metrics for one dataset case."""

    case_id: str
    category: str
    difficulty: str
    run_name: str
    concept_coverage: float = Field(..., ge=0, le=1)
    matched_concepts: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    metrics: BenchmarkMetrics
