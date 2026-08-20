# Multi-Agent System Design

## Problem

Answer open-ended research questions with source-backed conclusions while keeping every handoff
visible enough to debug and evaluate.

## Why multi-agent?

A single-agent baseline searches, evaluates, and writes in one context. It is fast and cheap, but
its intermediate reasoning cannot be inspected. The multi-agent version deliberately separates
evidence collection, analysis, and communication. This costs more calls but makes failures easier
to locate.

## Agent roles

| Agent | Responsibility | Reads | Writes | Main failure mode |
|---|---|---|---|---|
| Supervisor | Select the next missing artifact | Entire state | `next_agent`, route trace | Looping or early stop |
| Researcher | Search and collect cited evidence | Request | `sources`, `research_notes` | Weak/missing sources |
| Analyst | Compare evidence and flag uncertainty | Research notes | `analysis_notes` | Unsupported inference |
| Writer | Answer for the requested audience | Research + analysis | `final_answer` | Citation mismatch |

The Supervisor is deterministic rather than LLM-based. Its routing rules can therefore be read,
tested, and explained directly.

## Shared state and routing

`ResearchState` is the handoff contract. Besides the artifacts above, it stores route history,
local trace events, errors, iterations, tokens, and estimated cost.

```text
START -> Supervisor -> Researcher -> Supervisor -> Analyst
      -> Supervisor -> Writer -> Supervisor -> END
```

The Supervisor checks for `research_notes`, then `analysis_notes`, then `final_answer`. The graph
stops when the answer exists or `MAX_ITERATIONS` is reached.

## Guardrails

- Max iterations: defaults to 6 supervisor decisions.
- Timeout: provider calls default to 30 seconds; the workflow checks its 60-second deadline
  between nodes.
- Retry: the OpenAI SDK retries transient failures once.
- Fallback: workflow errors are captured in state and return a safe diagnostic answer.
- Validation: Pydantic validates inputs; each worker checks required upstream artifacts.

## Benchmark plan

Run the same query through `single-agent` and `multi-agent`. Compare latency, input/output tokens,
estimated model-token cost, citation coverage, failure rate, and the documented structural-quality
proxy. Manually review correctness because structural metrics cannot establish factuality.
