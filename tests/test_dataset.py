from pathlib import Path

import pytest

from multi_agent_research_lab.core.schemas import ConceptCriterion, EvaluationCase, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.dataset import (
    DatasetFormatError,
    concept_coverage,
    evaluate_dataset,
    load_evaluation_cases,
    serialize_case_run,
)
from multi_agent_research_lab.evaluation.report import render_dataset_report


def make_case() -> EvaluationCase:
    return EvaluationCase(
        id="test_case",
        query="Explain a safe agent workflow",
        category="testing",
        difficulty="easy",
        concepts=[
            ConceptCriterion(name="shared state", aliases=["state object"], weight=2),
            ConceptCriterion(name="timeout", weight=1),
        ],
        reference_points=["Use explicit state and a timeout."],
    )


def test_repository_dataset_loads_unique_cases() -> None:
    cases = load_evaluation_cases(Path("datasets/mock_research_eval.jsonl"))

    assert len(cases) == 6
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.reference_points for case in cases)


def test_dataset_loader_reports_invalid_row() -> None:
    with pytest.raises(DatasetFormatError, match="row 1"):
        load_evaluation_cases(Path("tests/fixtures/invalid_eval.jsonl"))


def test_concept_coverage_uses_aliases_and_weights() -> None:
    coverage, matched, missing = concept_coverage(
        make_case(), "The state object should be passed between every node."
    )

    assert coverage == pytest.approx(2 / 3)
    assert matched == ["shared state"]
    assert missing == ["timeout"]


def test_dataset_run_can_be_audited_and_rendered() -> None:
    case = make_case()

    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.final_answer = "Use shared state and enforce a timeout."
        return state

    states, results = evaluate_dataset([case], "fake", runner)
    raw = serialize_case_run(case, states[0], results[0])
    report = render_dataset_report(results, "mock.jsonl")

    assert results[0].concept_coverage == 1.0
    assert '"reference_points"' in raw
    assert '"final_answer"' in raw
    assert "Synthetic Dataset Evaluation" in report
    assert "100%" in report
