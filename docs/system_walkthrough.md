# System Walkthrough

## Read the system in this order

1. `core/state.py` defines the shared folder passed between agents.
2. `agents/supervisor.py` shows the complete routing policy.
3. `agents/researcher.py`, `analyst.py`, and `writer.py` show each handoff.
4. `graph/workflow.py` connects those classes as LangGraph nodes and edges.
5. `evaluation/benchmark.py` shows every metric formula.

## What happens during one run

The CLI creates a `ResearchState` containing the validated query. The Supervisor inspects which
artifact is missing and records its decision. The Researcher invokes OpenAI web search (so Tavily
is not required), stores the returned URLs and notes, then hands state back. The Analyst receives
only those notes and identifies support, disagreement, and uncertainty. The Writer receives both
artifacts plus a numbered source list and creates the cited final answer. A final Supervisor pass
routes to `done`.

Web search is required, not merely offered to the model. Only URLs returned in provider citation
metadata enter `state.sources`; plausible-looking URLs written only as text are not trusted. A run
with no verified source is marked as an error rather than silently presented as grounded research.

Each worker has a deliberately narrow prompt. Agents do not talk directly: they communicate only
by reading and writing named `ResearchState` fields. This makes the handoffs observable and keeps
the graph independent of provider-specific code.

## Inspecting a run

```powershell
.\.venv\Scripts\python.exe -m multi_agent_research_lab.cli multi-agent `
  --query "When should a team use a multi-agent LLM workflow?"
```

In the JSON output, inspect:

- `route_history`: expected sequence is researcher, analyst, writer, done.
- `research_notes`, `analysis_notes`, `final_answer`: the three handoff artifacts.
- `agent_results`: output and token metadata per worker.
- `trace`: routing decisions and wall-clock duration per node.
- `errors`: guardrail or provider failures.

`OPENAI_REQUEST_TIMEOUT_SECONDS` limits each blocking API call. `TIMEOUT_SECONDS` is checked
between graph nodes; it cannot interrupt a synchronous provider call already in progress.

With `LANGSMITH_API_KEY` configured, tracing is enabled automatically under
`LANGSMITH_PROJECT`. LangGraph nodes and wrapped OpenAI calls appear as nested runs. Local trace
events remain available when external tracing is off.

## Understanding evaluation

Run `make benchmark` (or the equivalent Python command on Windows). It executes the same query in
two modes and writes `reports/benchmark_report.md`.

- **Latency** is elapsed wall-clock time. Lower is faster.
- **Input/output tokens** measure model usage. Role separation usually increases both.
- **Estimated cost** applies configured model token rates. It excludes web-search tool fees.
- **Citation coverage** is the fraction of returned sources referenced by `[n]` or URL.
- **Failure rate** is 100% when a run has errors or no answer, otherwise 0% for that run.
- **Quality proxy (0–10)** combines answer presence (2), depth up to 250 words (3), citation
  coverage (3), and source diversity up to three sources (2).

The quality score is intentionally simple and reproducible. It does not measure factual
correctness. Use the peer-review rubric or a labeled evaluation dataset for that.

## Synthetic dataset evaluation

`datasets/mock_research_eval.jsonl` provides six non-authoritative cases with weighted expected
concepts and manual-review points. `evaluate-dataset` scores concept coverage in addition to the
structural metrics. It saves the full query, expectations, final state, and scores for each case in
`reports/runs/<mode>/`; those raw provider outputs are intentionally ignored by Git.

Start with `--limit 1` because baseline mode makes one paid model/search call per case and
multi-agent mode makes three. Use both modes on the same case set for a paired comparison.

## Useful experiments

Change one thing at a time: lower `MAX_ITERATIONS`, remove a source, alter an agent prompt, or run
the same query in both modes. Compare the final state and trace to see where behavior changes.

## Confidence ladder

1. `evaluate-dataset` is a fast smoke test using transparent phrase matching.
2. `validate-gold-dataset` checks frozen-evidence integrity and human approval provenance.
3. `evaluate-gold` compares both modes on identical evidence over repeated runs and stores
   claim-level judgments under `reports/gold/runs/`.
4. `export-review-packet` removes system identities from first-repetition outputs.
5. `import-human-labels` measures exact agreement and Cohen's kappa. Below the 80%/0.70 gates,
   automated aggregates stay provisional.

This separates three questions often conflated: whether the software ran, whether an answer
matched a drafted expectation, and whether reviewed evidence actually supports its claims.
