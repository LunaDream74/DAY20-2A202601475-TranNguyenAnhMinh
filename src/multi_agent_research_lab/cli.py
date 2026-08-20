"""Command-line entrypoint for running and comparing the research workflows."""

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import Runner, run_benchmark
from multi_agent_research_lab.evaluation.dataset import (
    DatasetFormatError,
    load_evaluation_cases,
    serialize_case_run,
)
from multi_agent_research_lab.evaluation.dataset import (
    evaluate_dataset as run_evaluation_dataset,
)
from multi_agent_research_lab.evaluation.gold import (
    GoldDataset,
    GoldRunRecord,
    HumanReview,
    blind_id,
    calibration,
    evidence_for_case,
    judge_answer,
    load_gold_dataset,
    new_run_record,
    render_gold_report,
    rubric,
    run_frozen_baseline,
    run_frozen_multi,
    validate_judgment,
)
from multi_agent_research_lab.evaluation.report import render_dataset_report, render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import configure_provider_tracing, trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_provider_tracing(settings)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _load_gold_or_exit(path: Path, *, require_approved: bool) -> GoldDataset:
    try:
        return load_gold_dataset(path, require_approved=require_approved)
    except DatasetFormatError as exc:
        console.print(Panel.fit(str(exc), title="Gold Dataset Error", style="red"))
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run one web-enabled model call that researches, analyzes, and writes."""

    _init()
    state = run_baseline(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


def run_baseline(query: str, llm_client: LLMClient | None = None) -> ResearchState:
    """Reusable single-agent runner used by both the CLI and benchmark."""

    state = ResearchState(request=_parse_query(query))
    client = llm_client or LLMClient()
    with trace_span("baseline.single_agent", {"query": query}) as span:
        response = client.complete(
            "You are a single-agent research assistant. Search, evaluate evidence, and write a "
            "clear answer for technical learners in one pass. Prefer primary sources, cite claims, "
            "and identify uncertainty. Never invent URLs.",
            query,
            web_search=True,
        )
        state.final_answer = response.content
        state.sources = response.sources
        if not response.sources:
            state.errors.append("Baseline returned no provider-verified web citations")
            state.final_answer = (
                "GROUNDING WARNING: No provider-verified sources were returned. Treat the "
                f"following answer as unverified.\n\n{response.content}"
            )
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"mode": "single_agent"},
            )
        )
    state.add_trace_event("agent_span", span)
    state.route_history = ["single_agent", "done"]
    return state


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run Supervisor -> Researcher -> Analyst -> Writer."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Query used for both systems")],
) -> None:
    """Compare single-agent and multi-agent runs and write a Markdown report."""

    _init()
    _, baseline_metrics = run_benchmark("single-agent", query, run_baseline)

    def run_multi(item: str) -> ResearchState:
        return MultiAgentWorkflow().run(ResearchState(request=_parse_query(item)))

    _, multi_metrics = run_benchmark("multi-agent", query, run_multi)
    report = render_markdown_report([baseline_metrics, multi_metrics])
    path = LocalArtifactStore().write_text("benchmark_report.md", report)
    console.print(report)
    console.print(f"Saved report to {path}")


@app.command("evaluate-dataset")
def evaluate_dataset_command(
    dataset: Annotated[Path, typer.Option(help="Synthetic JSONL dataset path")] = Path(
        "datasets/mock_research_eval.jsonl"
    ),
    mode: Annotated[str, typer.Option(help="baseline or multi-agent")] = "multi-agent",
    limit: Annotated[int, typer.Option(min=0, help="Cases to run; 0 means all")] = 0,
) -> None:
    """Run one system over validated mock cases and persist auditable states."""

    _init()
    cases = load_evaluation_cases(dataset)
    if limit:
        cases = cases[:limit]

    def run_dataset_baseline(item: str) -> ResearchState:
        return run_baseline(item)

    def run_dataset_multi(item: str) -> ResearchState:
        return MultiAgentWorkflow().run(ResearchState(request=_parse_query(item)))

    runner: Runner
    if mode == "baseline":
        runner = run_dataset_baseline
    elif mode == "multi-agent":
        runner = run_dataset_multi
    else:
        console.print(Panel.fit("Mode must be baseline or multi-agent", style="red"))
        raise typer.Exit(code=1)

    states, results = run_evaluation_dataset(cases, mode, runner)
    store = LocalArtifactStore()
    for case, state, result in zip(cases, states, results, strict=True):
        store.write_text(
            f"runs/{mode}/{case.id}.json",
            serialize_case_run(case, state, result),
        )
    report = render_dataset_report(results, dataset.name)
    report_name = f"dataset_{mode.replace('-', '_')}_report.md"
    path = store.write_text(report_name, report)
    console.print(report)
    console.print(f"Saved report to {path}; raw case states to reports/runs/{mode}/")


@app.command("validate-gold-dataset")
def validate_gold_dataset_command(
    dataset: Annotated[Path, typer.Option(help="Gold dataset JSON path")] = Path(
        "datasets/gold_research_eval.json"
    ),
    require_approved: Annotated[bool, typer.Option(help="Reject pending cases")] = False,
) -> None:
    """Check hashes, references, rubric IDs, and human-approval provenance."""

    gold = _load_gold_or_exit(dataset, require_approved=require_approved)
    approved = sum(case.status == "approved" for case in gold.cases)
    console.print(
        f"Valid: {len(gold.cases)} cases, {len(gold.evidence)} frozen evidence records; "
        f"approved {approved}/{len(gold.cases)}."
    )


@app.command("evaluate-gold")
def evaluate_gold_command(
    dataset: Annotated[Path, typer.Option(help="Approved gold dataset JSON path")] = Path(
        "datasets/gold_research_eval.json"
    ),
    mode: Annotated[str, typer.Option(help="baseline, multi-agent, or both")] = "both",
    repetitions: Annotated[int, typer.Option(min=1, max=10)] = 3,
    limit: Annotated[int, typer.Option(min=0, help="Cases to run; 0 means all")] = 0,
    seed: Annotated[int, typer.Option()] = 20260820,
    resume: Annotated[
        bool,
        typer.Option(help="Reuse matching immutable runs and evaluate only missing ones"),
    ] = False,
) -> None:
    """Run blinded, repeated frozen-evidence evaluation and a separate model judge."""

    _init()
    settings = get_settings()
    if not settings.eval_judge_model:
        raise typer.BadParameter("Set EVAL_JUDGE_MODEL before gold evaluation")
    system_model = settings.system_model or settings.openai_model
    if settings.eval_judge_model == system_model:
        raise typer.BadParameter("EVAL_JUDGE_MODEL must differ from SYSTEM_MODEL")
    if settings.system_provider == "openrouter" and not settings.openrouter_api_key:
        raise typer.BadParameter("Set OPENROUTER_API_KEY for the OpenRouter system model")
    modes: list[str]
    if mode == "both":
        modes = ["baseline", "multi-agent"]
    elif mode in {"baseline", "multi-agent"}:
        modes = [mode]
    else:
        raise typer.BadParameter("mode must be baseline, multi-agent, or both")
    gold = _load_gold_or_exit(dataset, require_approved=True)
    selected_cases = gold.cases[:limit] if limit else gold.cases
    system_client = LLMClient(
        settings,
        provider=settings.system_provider,
        model=system_model,
    )
    judge_client = LLMClient(
        settings,
        provider=settings.eval_judge_provider,
        model=settings.eval_judge_model,
    )
    records: list[GoldRunRecord] = []
    store = LocalArtifactStore()
    for case in selected_cases:
        for current_mode in modes:
            for repetition in range(1, repetitions + 1):
                output_id = blind_id(seed, case.id, current_mode, repetition)
                path = Path("reports/gold/runs") / f"{output_id}.json"
                if path.exists():
                    if resume:
                        records.append(
                            GoldRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
                        )
                        console.print(f"Reused {case.id} / {current_mode} / {repetition}")
                        continue
                    raise typer.BadParameter(f"Immutable run already exists: {path}")
                started = perf_counter()
                if current_mode == "baseline":
                    state = run_frozen_baseline(gold, case, system_client)
                else:
                    state = run_frozen_multi(gold, case, system_client, settings)
                latency_seconds = perf_counter() - started
                judgment, judge_usage = judge_answer(
                    gold, case, state.final_answer or "", output_id, judge_client
                )
                record = new_run_record(
                    dataset_path=dataset,
                    case=case,
                    mode=current_mode,  # type: ignore[arg-type]
                    repetition=repetition,
                    state=state,
                    judgment=judgment,
                    system_model=system_model,
                    judge_model=settings.eval_judge_model,
                    seed=seed,
                    latency_seconds=latency_seconds,
                    judge_usage=judge_usage,
                )
                store.write_text(f"gold/runs/{output_id}.json", record.model_dump_json(indent=2))
                records.append(record)
                console.print(f"Completed {case.id} / {current_mode} / {repetition}")
    report = render_gold_report(records, seed=seed)
    report_path = store.write_text("gold/report.md", report)
    console.print(report)
    console.print(f"Saved {len(records)} immutable runs and {report_path}")


def _load_run_records(path: Path) -> list[GoldRunRecord]:
    return [
        GoldRunRecord.model_validate_json(item.read_text(encoding="utf-8"))
        for item in path.glob("*.json")
    ]


@app.command("export-review-packet")
def export_review_packet_command(
    dataset: Annotated[Path, typer.Option()] = Path("datasets/gold_research_eval.json"),
    runs: Annotated[Path, typer.Option()] = Path("reports/gold/runs"),
    output: Annotated[Path, typer.Option()] = Path(
        "reports/gold/annotations/human_review_packet.jsonl"
    ),
    seed: Annotated[int | None, typer.Option(help="Only export runs with this seed")] = None,
) -> None:
    """Export repetition-one answers without system identities for human labeling."""

    gold = _load_gold_or_exit(dataset, require_approved=True)
    cases = {case.id: case for case in gold.cases}
    records = sorted(
        (
            record
            for record in _load_run_records(runs)
            if record.repetition == 1 and (seed is None or record.seed == seed)
        ),
        key=lambda item: item.blind_output_id,
    )
    rows = []
    for record in records:
        case = cases[record.case_id]
        rows.append(
            {
                "blind_output_id": record.blind_output_id,
                "question": case.query,
                "evidence": [
                    item.model_dump(mode="json") for item in evidence_for_case(gold, case)
                ],
                "rubric": [
                    {"item_type": kind, **claim.model_dump()} for kind, claim in rubric(case)
                ],
                "answer": record.answer,
                "human_review": {
                    "reviewer_id": "FILL_ME",
                    "evaluation": {
                        "blind_output_id": record.blind_output_id,
                        "decisions": [
                            {
                                "item_id": claim.id,
                                "item_type": kind,
                                "verdict": "FILL_ME",
                                "citation_status": "FILL_ME",
                            }
                            for kind, claim in rubric(case)
                        ],
                        "extra_claims": [],
                        "notes": "",
                    },
                },
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    console.print(f"Exported {len(rows)} blinded outputs to {output}")


@app.command("import-human-labels")
def import_human_labels_command(
    labels: Annotated[Path, typer.Option(help="JSONL HumanReview records")],
    dataset: Annotated[Path, typer.Option()] = Path("datasets/gold_research_eval.json"),
    runs: Annotated[Path, typer.Option()] = Path("reports/gold/runs"),
    seed: Annotated[int | None, typer.Option(help="Only calibrate runs with this seed")] = None,
) -> None:
    """Validate human labels and report agreement with the model judge."""

    gold = _load_gold_or_exit(dataset, require_approved=True)
    cases = {case.id: case for case in gold.cases}
    records = [record for record in _load_run_records(runs) if seed is None or record.seed == seed]
    if not records:
        raise typer.BadParameter("No run records matched the requested seed")
    record_by_id = {item.blind_output_id: item for item in records}
    reviews = []
    unscored_extra_claims = 0
    for line in labels.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        review_payload = payload.get("human_review", payload)
        evaluation_payload = review_payload.get("evaluation", {})
        extras = evaluation_payload.get("extra_claims", [])
        text_extras = [item for item in extras if isinstance(item, str)]
        if text_extras:
            unscored_extra_claims += len(text_extras)
            evaluation_payload["extra_claims"] = [
                item for item in extras if not isinstance(item, str)
            ]
        reviews.append(HumanReview.model_validate(review_payload))
    for review in reviews:
        record = record_by_id.get(review.evaluation.blind_output_id)
        if not record:
            raise typer.BadParameter(f"Unknown output id: {review.evaluation.blind_output_id}")
        validate_judgment(cases[record.case_id], review.evaluation, record.blind_output_id)
    summary = calibration(
        [record.judgment for record in records], [review.evaluation for review in reviews]
    )
    if unscored_extra_claims:
        console.print(
            f"Excluded {unscored_extra_claims} text-only extra claims from agreement; "
            "calibration compares fixed rubric verdicts only."
        )
    path = LocalArtifactStore().write_text(
        "gold/calibration.json", summary.model_dump_json(indent=2)
    )
    trust_label = "TRUSTED" if summary.trusted else "PROVISIONAL"
    calibrated_report = render_gold_report(records, seed=records[0].seed)
    calibrated_report += (
        "\n## Human calibration\n\n"
        f"Status: **{trust_label}**. Exact agreement: **{summary.exact_agreement:.1%}**; "
        f"Cohen's kappa: **{summary.cohen_kappa:.3f}** across "
        f"**{summary.comparisons}** claim decisions.\n"
    )
    LocalArtifactStore().write_text("gold/calibrated_report.md", calibrated_report)
    console.print(summary.model_dump_json(indent=2))
    console.print(f"Saved calibration to {path}")


if __name__ == "__main__":
    app()
