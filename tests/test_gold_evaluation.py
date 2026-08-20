import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.evaluation.dataset import DatasetFormatError
from multi_agent_research_lab.evaluation.gold import (
    ExtraClaimDecision,
    GoldCase,
    GoldRunRecord,
    HumanReview,
    JudgeEvaluation,
    RubricDecision,
    blind_id,
    calibration,
    load_gold_dataset,
    new_run_record,
    paired_bootstrap_ci,
    render_frozen_evidence,
    render_gold_report,
    rubric,
    run_frozen_baseline,
    run_frozen_multi,
    score_judgment,
    validate_judgment,
)
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse

DATASET = Path("datasets/gold_research_eval.json")


class FrozenFakeClient(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(
        self, system_prompt: str, user_prompt: str, *, web_search: bool = False
    ) -> LLMResponse:
        self.prompts.append(user_prompt)
        if "Research notes:" in user_prompt:
            return LLMResponse(content="Analysis grounded in the supplied evidence.")
        return LLMResponse(content="A supported answer [1].", input_tokens=3, output_tokens=4)


def _judgment(case_index: int = 0) -> tuple[GoldCase, JudgeEvaluation]:
    dataset = load_gold_dataset(DATASET)
    case = dataset.cases[case_index]
    decisions = [
        RubricDecision(
            item_id=claim.id,
            item_type=kind,
            verdict="absent" if kind == "forbidden" else "supported",
            citation_status="not_applicable" if kind == "forbidden" else "valid",
        )
        for kind, claim in rubric(case)
    ]
    return case, JudgeEvaluation(
        blind_output_id="out_test", decisions=decisions, extra_claims=[], notes=""
    )


def test_reviewed_dataset_has_twelve_balanced_cases_and_valid_hashes() -> None:
    dataset = load_gold_dataset(DATASET)
    categories = {case.category for case in dataset.cases}

    assert len(dataset.cases) == 12
    assert all(
        sum(item.category == category for item in dataset.cases) == 2 for category in categories
    )
    assert all(case.status == "approved" for case in dataset.cases)
    assert all(case.human_reviewer_id and case.reviewed_at for case in dataset.cases)


def test_unapproved_dataset_cannot_be_evaluated(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    raw["cases"][0]["status"] = "pending_human"
    unapproved = json.dumps(raw)
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: unapproved)

    with pytest.raises(DatasetFormatError, match="requires approved cases"):
        load_gold_dataset(DATASET, require_approved=True)


def test_candidate_evidence_excludes_hidden_rubric() -> None:
    dataset = load_gold_dataset(DATASET)
    case = dataset.cases[0]
    rendered = render_frozen_evidence(dataset, case)

    assert "Building effective agents" in rendered
    assert case.required_claims[0].id not in rendered
    assert "forbidden_claims" not in rendered


def test_baseline_and_multi_use_only_frozen_evidence() -> None:
    dataset = load_gold_dataset(DATASET)
    case = dataset.cases[0]
    fake = FrozenFakeClient()
    settings = Settings(_env_file=None, OPENAI_MODEL="system-model")

    baseline = run_frozen_baseline(dataset, case, fake)
    multi = run_frozen_multi(dataset, case, fake, settings)

    assert baseline.sources == multi.sources
    assert all(source.metadata["frozen"] for source in multi.sources)
    assert baseline.final_answer
    assert multi.final_answer


def test_claim_metrics_have_inspectable_denominators() -> None:
    case, judgment = _judgment()
    judgment.extra_claims.append(
        ExtraClaimDecision(
            claim="Unsupported extra", verdict="insufficient_evidence", citation_status="missing"
        )
    )

    metrics = score_judgment(case, judgment)

    assert metrics.required_claim_recall == 1.0
    assert metrics.forbidden_claim_violations == 0
    assert metrics.factual_precision < 1.0
    assert metrics.caveat_coverage == 1.0


def test_judgment_must_cover_rubric_exactly_once() -> None:
    case, judgment = _judgment()
    judgment.decisions.pop()

    with pytest.raises(ValueError, match="exactly match"):
        validate_judgment(case, judgment, "out_test")


def test_calibration_threshold_and_kappa() -> None:
    _, judgment = _judgment()
    copied = JudgeEvaluation.model_validate(judgment.model_dump())
    review = HumanReview(reviewer_id="reviewer-1", evaluation=copied)

    summary = calibration([judgment], [review.evaluation])

    assert summary.exact_agreement == 1.0
    assert summary.cohen_kappa == 1.0
    assert summary.trusted


def test_blinding_and_bootstrap_are_deterministic() -> None:
    first = blind_id(7, "case", "baseline", 1)
    second = blind_id(7, "case", "multi-agent", 1)
    data_a = {"a": 0.2, "b": 0.4, "c": 0.6}
    data_b = {"a": 0.3, "b": 0.6, "c": 0.9}

    assert first.startswith("out_") and first != second
    assert paired_bootstrap_ci(data_a, data_b, seed=7, samples=1000) == paired_bootstrap_ci(
        data_a, data_b, seed=7, samples=1000
    )


def test_run_record_and_report_include_reproducibility_and_operations() -> None:
    dataset = load_gold_dataset(DATASET)
    case, judgment = _judgment()
    state = run_frozen_baseline(dataset, case, FrozenFakeClient())
    record = new_run_record(
        dataset_path=DATASET,
        case=case,
        mode="baseline",
        repetition=1,
        state=state,
        judgment=judgment,
        system_model="system-model",
        judge_model="judge-model",
        seed=7,
        latency_seconds=0.25,
        judge_usage=LLMResponse(content="{}", input_tokens=2, output_tokens=3, cost_usd=0.01),
    )

    report = render_gold_report([record], seed=7)

    assert len(record.dataset_sha256) == 64
    assert record.judge_model == "judge-model"
    assert "Operational metrics" in report
    assert "provisional" in report


def test_report_uses_paired_case_means() -> None:
    _, judgment = _judgment()
    base = {
        "blind_output_id": "out_a",
        "case_id": "a",
        "repetition": 1,
        "answer": "answer",
        "state": {},
        "judgment": judgment,
        "metrics": score_judgment(_judgment()[0], judgment),
        "system_model": "system",
        "judge_model": "judge",
        "dataset_sha256": "a" * 64,
        "seed": 7,
        "latency_seconds": 1.0,
        "input_tokens": 1,
        "output_tokens": 1,
        "estimated_cost_usd": 0.0,
        "judge_input_tokens": 1,
        "judge_output_tokens": 1,
        "judge_estimated_cost_usd": 0.0,
        "created_at": datetime.now(UTC),
    }
    baseline = GoldRunRecord(mode="baseline", **base)
    multi = GoldRunRecord(mode="multi-agent", **{**base, "blind_output_id": "out_b"})

    report = render_gold_report([baseline, multi], seed=7)

    assert "Paired delta" in report
    assert "+0.0%" in report
