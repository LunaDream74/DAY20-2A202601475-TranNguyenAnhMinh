"""Loading and scoring for the synthetic research evaluation dataset."""

import json
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import DatasetCaseResult, EvaluationCase
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import Runner, run_benchmark


class DatasetFormatError(LabError):
    """Raised when a JSONL row does not satisfy the evaluation schema."""


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load validated cases from JSONL and reject duplicate identifiers."""

    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            case = EvaluationCase.model_validate_json(raw_line)
        except (ValidationError, ValueError) as exc:
            raise DatasetFormatError(f"Invalid dataset row {line_number}: {exc}") from exc
        if case.id in seen:
            raise DatasetFormatError(f"Duplicate case id at row {line_number}: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetFormatError(f"Dataset is empty: {path}")
    return cases


def concept_coverage(
    case: EvaluationCase, answer: str | None
) -> tuple[float, list[str], list[str]]:
    """Score weighted concept presence using declared names and aliases."""

    normalized_answer = _normalize(answer or "")
    matched: list[str] = []
    missing: list[str] = []
    matched_weight = 0.0
    total_weight = sum(concept.weight for concept in case.concepts)
    for concept in case.concepts:
        terms = [concept.name, *concept.aliases]
        if any(_normalize(term) in normalized_answer for term in terms):
            matched.append(concept.name)
            matched_weight += concept.weight
        else:
            missing.append(concept.name)
    return matched_weight / total_weight, matched, missing


def evaluate_dataset(
    cases: Iterable[EvaluationCase], run_name: str, runner: Runner
) -> tuple[list[ResearchState], list[DatasetCaseResult]]:
    """Run and score every case while retaining states for local audit artifacts."""

    states: list[ResearchState] = []
    results: list[DatasetCaseResult] = []
    for case in cases:
        state, metrics = run_benchmark(run_name, case.query, runner)
        coverage, matched, missing = concept_coverage(case, state.final_answer)
        states.append(state)
        results.append(
            DatasetCaseResult(
                case_id=case.id,
                category=case.category,
                difficulty=case.difficulty,
                run_name=run_name,
                concept_coverage=coverage,
                matched_concepts=matched,
                missing_concepts=missing,
                metrics=metrics,
            )
        )
    return states, results


def serialize_case_run(
    case: EvaluationCase, state: ResearchState, result: DatasetCaseResult
) -> str:
    """Serialize inputs, expectations, state, and scores for later auditing."""

    payload = {
        "case": case.model_dump(),
        "state": state.model_dump(),
        "result": result.model_dump(),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
