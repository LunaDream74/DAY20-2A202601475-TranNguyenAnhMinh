# Synthetic Evaluation Dataset

`mock_research_eval.jsonl` contains six hand-authored cases for learning and pipeline testing. It
is not an official benchmark and its `reference_points` are not authoritative ground truth.

Each JSONL row includes:

- `query`, category, difficulty, and intended audience;
- weighted `concepts` with accepted aliases for deterministic automatic scoring;
- `reference_points` for manual review; and
- `evaluation_notes` describing limitations.

Concept coverage is a case-insensitive phrase-presence check. It is useful for catching omissions,
but it cannot determine whether a concept was explained correctly. Review the complete saved state
and cited evidence before accepting a result.

Run one inexpensive case first:

```powershell
.\.venv\Scripts\python.exe -m multi_agent_research_lab.cli evaluate-dataset `
  --mode baseline --limit 1
```

Full dataset runs make paid API and web-search calls. Raw per-case states are saved under
`reports/runs/` for local audit and are excluded from Git because they may contain provider output.

## Draft gold set

`gold_research_eval.json` contains 12 frozen-evidence cases and atomic claim rubrics. Unlike the
mock keyword set, it can measure factual precision, required-claim recall, grounded F1,
contradictions, citations, caveats, and forbidden claims. It is deliberately marked
`pending_human`; follow [GOLD_REVIEW.md](GOLD_REVIEW.md) before running it. This prevents drafted
expectations or a model judge from being mislabeled as ground truth.
