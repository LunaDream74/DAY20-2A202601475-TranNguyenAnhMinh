import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

import multi_agent_research_lab.cli as cli
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.evaluation.gold import (
    GoldDataset,
    GoldRunRecord,
    HumanReview,
    JudgeEvaluation,
    RubricDecision,
    load_gold_dataset,
    rubric,
    run_frozen_baseline,
    score_judgment,
)
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse

DATASET = Path("datasets/gold_research_eval.json")
runner = CliRunner()


class FakeClient(LLMClient):
    def __init__(self) -> None:
        pass

    def complete(
        self, system_prompt: str, user_prompt: str, *, web_search: bool = False
    ) -> LLMResponse:
        return LLMResponse(content="Frozen answer [1].", input_tokens=2, output_tokens=2)


def _approved_dataset() -> GoldDataset:
    dataset = load_gold_dataset(DATASET).model_copy(deep=True)
    for case in dataset.cases:
        case.status = "approved"
        case.human_reviewer_id = "reviewer-1"
        case.reviewed_at = datetime.now(UTC)
    return dataset


def _record(dataset: GoldDataset) -> GoldRunRecord:
    case = dataset.cases[0]
    decisions = [
        RubricDecision(
            item_id=claim.id,
            item_type=kind,
            verdict="absent" if kind == "forbidden" else "supported",
            citation_status="not_applicable" if kind == "forbidden" else "valid",
        )
        for kind, claim in rubric(case)
    ]
    judgment = JudgeEvaluation(
        blind_output_id="out_review", decisions=decisions, extra_claims=[], notes=""
    )
    state = run_frozen_baseline(dataset, case, FakeClient())
    return GoldRunRecord(
        blind_output_id="out_review",
        case_id=case.id,
        mode="baseline",
        repetition=1,
        answer=state.final_answer or "",
        state=state.model_dump(mode="json"),
        judgment=judgment,
        metrics=score_judgment(case, judgment),
        system_model="system",
        judge_model="judge",
        dataset_sha256="a" * 64,
        seed=7,
        latency_seconds=0.1,
        input_tokens=2,
        output_tokens=2,
        estimated_cost_usd=0,
        judge_input_tokens=2,
        judge_output_tokens=2,
        judge_estimated_cost_usd=0,
        created_at=datetime.now(UTC),
    )


def test_validate_gold_cli_reports_approved_dataset() -> None:
    result = runner.invoke(cli.app, ["validate-gold-dataset", "--dataset", str(DATASET)])

    assert result.exit_code == 0
    assert "12 cases" in result.stdout
    assert "approved 12/12" in result.stdout


def test_baseline_runner_records_missing_verified_sources() -> None:
    state = cli.run_baseline("Explain a research workflow", FakeClient())

    assert state.final_answer and state.final_answer.startswith("GROUNDING WARNING")
    assert state.route_history == ["single_agent", "done"]
    assert state.errors


def test_evaluate_gold_rejects_missing_or_same_judge(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "_init", lambda: None)
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(_env_file=None, OPENAI_MODEL="same", EVAL_JUDGE_MODEL="same"),
    )

    result = runner.invoke(cli.app, ["evaluate-gold"])

    assert result.exit_code != 0
    assert "must differ" in result.output


def test_evaluate_gold_requires_openrouter_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "_init", lambda: None)
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            OPENAI_API_KEY="judge-key",
            SYSTEM_PROVIDER="openrouter",
            SYSTEM_MODEL="mistralai/mistral-7b-instruct:free",
            EVAL_JUDGE_MODEL="gpt-4o-mini",
        ),
    )

    result = runner.invoke(cli.app, ["evaluate-gold"])

    assert result.exit_code != 0
    assert "OPENROUTER_API_KEY" in result.output


def test_export_review_packet_is_blinded(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dataset = _approved_dataset()
    record = _record(dataset)
    other_seed = record.model_copy(
        update={"blind_output_id": "out_other_seed", "seed": record.seed + 1}
    )
    monkeypatch.setattr(cli, "load_gold_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(cli, "_load_run_records", lambda path: [record, other_seed])
    output = Path("reports/gold/annotations/test_packet.jsonl")

    result = runner.invoke(
        cli.app,
        ["export-review-packet", "--output", str(output), "--seed", str(record.seed)],
    )

    try:
        text = output.read_text(encoding="utf-8")
        assert result.exit_code == 0
        assert "out_review" in text
        assert "out_other_seed" not in text
        assert '"mode"' not in text
        assert '"human_review"' in text
    finally:
        output.unlink(missing_ok=True)


def test_import_human_labels_calibrates(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dataset = _approved_dataset()
    record = _record(dataset)
    review = HumanReview(reviewer_id="human-1", evaluation=record.judgment)
    review_packet = review.model_dump(mode="json")
    review_packet["evaluation"]["extra_claims"] = ["A human-noted unscored claim."]
    labels = Path("reports/gold/annotations/test_labels.jsonl")
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_text(
        json.dumps({"human_review": review_packet}) + "\n",
        encoding="utf-8",
    )
    written: dict[str, str] = {}

    class FakeStore:
        def write_text(self, relative_path: str, content: str) -> Path:
            written[relative_path] = content
            return Path("reports") / relative_path

    monkeypatch.setattr(cli, "load_gold_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(cli, "_load_run_records", lambda path: [record])
    monkeypatch.setattr(cli, "LocalArtifactStore", FakeStore)

    try:
        result = runner.invoke(
            cli.app,
            ["import-human-labels", "--labels", str(labels), "--seed", str(record.seed)],
        )

        assert result.exit_code == 0
        assert '"trusted": true' in result.stdout
        assert "Excluded 1 text-only extra claims" in result.stdout
        assert "gold/calibration.json" in written
    finally:
        labels.unlink(missing_ok=True)
