"""Human-adjudicated, frozen-evidence evaluation for the research workflows."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.dataset import DatasetFormatError
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient, SearchResponse

Verdict = Literal["supported", "contradicted", "insufficient_evidence", "absent"]
CitationStatus = Literal["valid", "invalid", "missing", "not_applicable"]
ItemType = Literal["required", "optional", "forbidden", "caveat"]


class FrozenEvidence(BaseModel):
    id: str = Field(pattern=r"^ev_[a-z0-9_]+$")
    title: str
    url: str
    retrieved_at: date
    excerpt: str = Field(min_length=20)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_type: Literal["primary", "secondary"] = "primary"


class GoldClaim(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    text: str = Field(min_length=5)
    evidence_ids: list[str] = Field(min_length=1)


class GoldCase(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    query: str = Field(min_length=5)
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    audience: str = "technical learners"
    evidence_ids: list[str] = Field(min_length=1)
    required_claims: list[GoldClaim] = Field(min_length=1)
    optional_claims: list[GoldClaim] = Field(default_factory=list)
    forbidden_claims: list[GoldClaim] = Field(default_factory=list)
    required_caveats: list[GoldClaim] = Field(default_factory=list)
    status: Literal["pending_human", "approved", "rejected"] = "pending_human"
    drafted_by: str
    human_reviewer_id: str | None = None
    reviewed_at: datetime | None = None


class GoldDataset(BaseModel):
    name: str
    version: str
    description: str
    evidence: list[FrozenEvidence] = Field(min_length=1)
    cases: list[GoldCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> GoldDataset:
        evidence_ids = [item.id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Duplicate evidence id")
        case_ids = [item.id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Duplicate case id")
        known = set(evidence_ids)
        for evidence in self.evidence:
            digest = hashlib.sha256(evidence.excerpt.encode()).hexdigest()
            if digest != evidence.content_sha256:
                raise ValueError(f"Evidence hash mismatch: {evidence.id}")
        for case in self.cases:
            if not set(case.evidence_ids) <= known:
                raise ValueError(f"Case {case.id} references unknown evidence")
            claims = (
                case.required_claims
                + case.optional_claims
                + case.forbidden_claims
                + case.required_caveats
            )
            claim_ids = [claim.id for claim in claims]
            if len(claim_ids) != len(set(claim_ids)):
                raise ValueError(f"Case {case.id} has duplicate claim ids")
            for claim in claims:
                if not set(claim.evidence_ids) <= set(case.evidence_ids):
                    raise ValueError(f"Claim {claim.id} references evidence outside its case")
            if case.status == "approved" and (not case.human_reviewer_id or not case.reviewed_at):
                raise ValueError(f"Approved case {case.id} lacks review provenance")
        return self


class RubricDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    item_type: ItemType
    verdict: Verdict
    citation_status: CitationStatus


class ExtraClaimDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    verdict: Literal["supported", "contradicted", "insufficient_evidence"]
    citation_status: Literal["valid", "invalid", "missing"]


class JudgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blind_output_id: str
    decisions: list[RubricDecision]
    extra_claims: list[ExtraClaimDecision]
    notes: str


class HumanReview(BaseModel):
    reviewer_id: str
    evaluation: JudgeEvaluation


class GoldMetrics(BaseModel):
    factual_precision: float = Field(ge=0, le=1)
    required_claim_recall: float = Field(ge=0, le=1)
    grounded_f1: float = Field(ge=0, le=1)
    contradiction_rate: float = Field(ge=0, le=1)
    citation_precision: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)
    caveat_coverage: float = Field(ge=0, le=1)
    forbidden_claim_violations: int = Field(ge=0)


class GoldRunRecord(BaseModel):
    blind_output_id: str
    case_id: str
    mode: Literal["baseline", "multi-agent"]
    repetition: int = Field(ge=1)
    answer: str
    state: dict[str, object]
    judgment: JudgeEvaluation
    metrics: GoldMetrics
    system_model: str
    judge_model: str
    dataset_sha256: str
    seed: int
    latency_seconds: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    judge_input_tokens: int = Field(ge=0)
    judge_output_tokens: int = Field(ge=0)
    judge_estimated_cost_usd: float = Field(ge=0)
    created_at: datetime


class CalibrationSummary(BaseModel):
    comparisons: int
    exact_agreement: float
    cohen_kappa: float
    trusted: bool


def load_gold_dataset(path: Path, *, require_approved: bool = False) -> GoldDataset:
    """Load and fully cross-validate the frozen dataset."""

    try:
        dataset = GoldDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetFormatError(f"Invalid gold dataset {path}: {exc}") from exc
    pending = [case.id for case in dataset.cases if case.status != "approved"]
    if require_approved and pending:
        raise DatasetFormatError(
            "Gold evaluation requires approved cases; pending: " + ", ".join(pending)
        )
    return dataset


def dataset_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_for_case(dataset: GoldDataset, case: GoldCase) -> list[FrozenEvidence]:
    by_id = {item.id: item for item in dataset.evidence}
    return [by_id[item_id] for item_id in case.evidence_ids]


def render_frozen_evidence(dataset: GoldDataset, case: GoldCase) -> str:
    """Render evidence only: rubric labels and expected claims never reach candidates."""

    blocks = []
    for index, item in enumerate(evidence_for_case(dataset, case), start=1):
        blocks.append(f"[{index}] {item.title}\nURL: {item.url}\nExcerpt: {item.excerpt}")
    return "\n\n".join(blocks)


class FrozenSearchClient(SearchClient):
    """A search-compatible adapter that never accesses the network."""

    def __init__(self, dataset: GoldDataset, case: GoldCase) -> None:
        self.dataset = dataset
        self.case = case

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        items = evidence_for_case(self.dataset, self.case)[:max_results]
        sources = [
            SourceDocument(
                title=item.title, url=item.url, snippet=item.excerpt, metadata={"frozen": True}
            )
            for item in items
        ]
        return SearchResponse(
            synthesis=render_frozen_evidence(self.dataset, self.case), sources=sources
        )


def run_frozen_baseline(dataset: GoldDataset, case: GoldCase, client: LLMClient) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=case.query, audience=case.audience))
    evidence = render_frozen_evidence(dataset, case)
    response = client.complete(
        "Answer using only the supplied frozen evidence. Cite factual claims with [n]. "
        "State uncertainty or limitations; do not use outside knowledge.",
        f"Question: {case.query}\nAudience: {case.audience}\n\nFrozen evidence:\n{evidence}",
    )
    state.final_answer = response.content
    state.sources = FrozenSearchClient(dataset, case).search(case.query).sources
    state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
    state.agent_results.append(AgentResult(agent=AgentName.WRITER, content=response.content))
    state.route_history = ["single_agent", "done"]
    return state


def run_frozen_multi(
    dataset: GoldDataset, case: GoldCase, client: LLMClient, settings: Settings
) -> ResearchState:
    workflow = MultiAgentWorkflow(
        settings=settings,
        researcher=ResearcherAgent(FrozenSearchClient(dataset, case)),
        analyst=AnalystAgent(client),
        writer=WriterAgent(client),
    )
    state = ResearchState(request=ResearchQuery(query=case.query, audience=case.audience))
    return workflow.run(state)


def rubric(case: GoldCase) -> list[tuple[ItemType, GoldClaim]]:
    return [
        *(("required", item) for item in case.required_claims),
        *(("optional", item) for item in case.optional_claims),
        *(("forbidden", item) for item in case.forbidden_claims),
        *(("caveat", item) for item in case.required_caveats),
    ]


def judge_answer(
    dataset: GoldDataset,
    case: GoldCase,
    answer: str,
    blind_output_id: str,
    client: LLMClient,
) -> tuple[JudgeEvaluation, LLMResponse]:
    items = "\n".join(f"- {kind} | {claim.id} | {claim.text}" for kind, claim in rubric(case))
    result, usage = client.complete_structured(
        "You are a strict claim-level evaluator. Use only the frozen evidence. For every rubric "
        "item return one decision. 'supported' means both stated and entailed by evidence; "
        "'absent' means not stated. Assess citations separately. Record other factual claims as "
        "extra_claims. For a forbidden item, use 'contradicted' only when the answer asserts the "
        "forbidden claim; use 'absent' when it omits or explicitly rejects it. Do not infer which "
        "system produced the answer.",
        f"Blind output id: {blind_output_id}\nQuestion: {case.query}\n\nEvidence:\n"
        f"{render_frozen_evidence(dataset, case)}\n\nRubric:\n{items}\n\nAnswer:\n{answer}",
        JudgeEvaluation,
    )
    validate_judgment(case, result, blind_output_id)
    return result, usage


def validate_judgment(case: GoldCase, judgment: JudgeEvaluation, blind_output_id: str) -> None:
    expected = {(claim.id, kind) for kind, claim in rubric(case)}
    actual = {(item.item_id, item.item_type) for item in judgment.decisions}
    if judgment.blind_output_id != blind_output_id:
        raise ValueError("Judge returned the wrong blind_output_id")
    if actual != expected or len(judgment.decisions) != len(expected):
        raise ValueError("Judge decisions do not exactly match the case rubric")


def score_judgment(case: GoldCase, judgment: JudgeEvaluation) -> GoldMetrics:
    fixed = judgment.decisions
    present = [item for item in fixed if item.verdict != "absent"]
    extra = judgment.extra_claims
    factual_count = len(present) + len(extra)
    supported = sum(item.verdict == "supported" for item in present) + sum(
        item.verdict == "supported" for item in extra
    )
    precision = supported / factual_count if factual_count else 0.0
    required_ids = {item.id for item in case.required_claims}
    caveat_ids = {item.id for item in case.required_caveats}
    forbidden_ids = {item.id for item in case.forbidden_claims}
    required_supported = sum(
        item.item_id in required_ids and item.verdict == "supported" for item in fixed
    )
    recall = required_supported / len(required_ids)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    contradicted = sum(item.verdict == "contradicted" for item in present) + sum(
        item.verdict == "contradicted" for item in extra
    )
    citation_statuses = [item.citation_status for item in present] + [
        item.citation_status for item in extra
    ]
    cited = [status for status in citation_statuses if status in {"valid", "invalid"}]
    valid_citations = cited.count("valid")
    required_valid = sum(
        item.item_id in required_ids and item.citation_status == "valid" for item in fixed
    )
    caveats = sum(item.item_id in caveat_ids and item.verdict == "supported" for item in fixed)
    forbidden = sum(item.item_id in forbidden_ids and item.verdict != "absent" for item in fixed)
    return GoldMetrics(
        factual_precision=precision,
        required_claim_recall=recall,
        grounded_f1=f1,
        contradiction_rate=contradicted / factual_count if factual_count else 0.0,
        citation_precision=valid_citations / len(cited) if cited else 0.0,
        citation_recall=required_valid / len(required_ids),
        caveat_coverage=caveats / len(caveat_ids) if caveat_ids else 1.0,
        forbidden_claim_violations=forbidden,
    )


def blind_id(seed: int, case_id: str, mode: str, repetition: int) -> str:
    raw = f"{seed}:{case_id}:{mode}:{repetition}".encode()
    return "out_" + hashlib.sha256(raw).hexdigest()[:16]


def calibration(llm: list[JudgeEvaluation], human: list[JudgeEvaluation]) -> CalibrationSummary:
    llm_by_id = {item.blind_output_id: item for item in llm}
    pairs: list[tuple[str, str]] = []
    for review in human:
        predicted = llm_by_id.get(review.blind_output_id)
        if not predicted:
            continue
        expected_by_item = {item.item_id: item.verdict for item in review.decisions}
        for item in predicted.decisions:
            if item.item_id in expected_by_item:
                pairs.append((item.verdict, expected_by_item[item.item_id]))
    if not pairs:
        raise ValueError("No matching LLM/human decisions")
    agreement = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = (
        float(
            sum(
                left_counts[label] * right_counts[label]
                for label in set(left_counts | right_counts)
            )
        )
        / len(pairs) ** 2
    )
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else 1.0
    return CalibrationSummary(
        comparisons=len(pairs),
        exact_agreement=agreement,
        cohen_kappa=kappa,
        trusted=agreement >= 0.80 and kappa >= 0.70,
    )


def paired_bootstrap_ci(
    baseline: dict[str, float], multi: dict[str, float], *, seed: int, samples: int = 10_000
) -> tuple[float, float, float]:
    case_ids = sorted(set(baseline) & set(multi))
    if not case_ids:
        raise ValueError("No paired cases")
    deltas = [multi[item] - baseline[item] for item in case_ids]
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    return mean(deltas), boot[int(samples * 0.025)], boot[min(int(samples * 0.975), samples - 1)]


def new_run_record(
    *,
    dataset_path: Path,
    case: GoldCase,
    mode: Literal["baseline", "multi-agent"],
    repetition: int,
    state: ResearchState,
    judgment: JudgeEvaluation,
    system_model: str,
    judge_model: str,
    seed: int,
    latency_seconds: float,
    judge_usage: LLMResponse,
) -> GoldRunRecord:
    return GoldRunRecord(
        blind_output_id=judgment.blind_output_id,
        case_id=case.id,
        mode=mode,
        repetition=repetition,
        answer=state.final_answer or "",
        state=state.model_dump(mode="json"),
        judgment=judgment,
        metrics=score_judgment(case, judgment),
        system_model=system_model,
        judge_model=judge_model,
        dataset_sha256=dataset_digest(dataset_path),
        seed=seed,
        latency_seconds=latency_seconds,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        estimated_cost_usd=state.estimated_cost_usd,
        judge_input_tokens=judge_usage.input_tokens or 0,
        judge_output_tokens=judge_usage.output_tokens or 0,
        judge_estimated_cost_usd=judge_usage.cost_usd or 0.0,
        created_at=datetime.now(UTC),
    )


def aggregate_case_metric(records: list[GoldRunRecord], metric: str) -> dict[str, float]:
    """Average repetitions within each case before system comparison."""

    grouped: dict[str, list[float]] = {}
    for record in records:
        value = getattr(record.metrics, metric)
        grouped.setdefault(record.case_id, []).append(float(value))
    return {case_id: mean(values) for case_id, values in grouped.items()}


def render_gold_report(records: list[GoldRunRecord], *, seed: int) -> str:
    if not records:
        raise ValueError("At least one gold run is required")
    metric_names = [
        "grounded_f1",
        "factual_precision",
        "required_claim_recall",
        "contradiction_rate",
        "citation_precision",
        "citation_recall",
        "caveat_coverage",
    ]
    modes = {record.mode for record in records}
    lines = [
        "# Human-Adjudicated Gold Evaluation",
        "",
        "> This is a lab-created gold set, not an official benchmark. Results remain "
        "provisional until judge calibration passes 80% exact agreement and Cohen's kappa 0.70.",
        "",
        "| Metric | Baseline | Multi-agent | Paired delta (95% bootstrap CI) |",
        "|---|---:|---:|---:|",
    ]
    for metric in metric_names:
        per_mode = {
            mode: aggregate_case_metric([item for item in records if item.mode == mode], metric)
            for mode in modes
        }
        base = per_mode.get("baseline", {})
        multi = per_mode.get("multi-agent", {})
        base_mean = mean(base.values()) if base else 0.0
        multi_mean = mean(multi.values()) if multi else 0.0
        comparison = "n/a"
        if base and multi:
            delta, low, high = paired_bootstrap_ci(base, multi, seed=seed)
            comparison = f"{delta:+.1%} ({low:+.1%}, {high:+.1%})"
        lines.append(f"| {metric} | {base_mean:.1%} | {multi_mean:.1%} | {comparison} |")
    lines.extend(
        [
            "",
            "## Operational metrics",
            "",
            "| System | Mean latency | Total system tokens | Total judge tokens | "
            "Estimated total cost |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in sorted(modes):
        selected = [item for item in records if item.mode == mode]
        total_cost = sum(
            item.estimated_cost_usd + item.judge_estimated_cost_usd for item in selected
        )
        lines.append(
            f"| {mode} | {mean(item.latency_seconds for item in selected):.2f}s | "
            f"{sum(item.input_tokens + item.output_tokens for item in selected)} | "
            f"{sum(item.judge_input_tokens + item.judge_output_tokens for item in selected)} | "
            f"${total_cost:.4f} |"
        )
    lines.extend(
        [
            "",
            "A system is not declared superior unless the paired interval excludes zero. "
            "Inspect claim-level judgments and human calibration before interpreting aggregates.",
        ]
    )
    return "\n".join(lines) + "\n"
