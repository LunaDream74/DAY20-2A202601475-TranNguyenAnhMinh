# Judge–Human Disagreement Analysis

## Conclusion

GPT-4o-mini is not calibrated for this dataset. Across 174 fixed-rubric decisions,
exact verdict agreement was 77.6% and Cohen's kappa was 0.564, below the 80% and
0.70 trust gates. The current model-judged system metrics therefore remain
provisional.

## Where the Judge Disagreed

There were 39 verdict disagreements. The dominant error was over-crediting content:

| Judge verdict | Human verdict | Count |
|---|---|---:|
| supported | absent | 22 |
| absent | supported | 10 |
| supported | contradicted | 4 |
| other transitions | mixed | 3 |

Agreement was strongest for forbidden claims (96.7%) and required claims (79.5%),
but weak for optional claims (67.9%) and caveats (60.7%). The largest concentrations
were `application_privacy_review` (10 disagreements),
`application_sensitive_data` (6), and `retrieval_global_questions` (5).

Baseline verdict agreement was 79.3%; multi-agent agreement was 75.9%. Multi-agent
answers were longer on average (1,480 versus 1,175 characters), which increases
paraphrase and claim-alignment difficulty.

## Root Causes

1. **Compound rubric items.** Several “atomic” items actually require many elements.
   For example, the privacy inventory item combines data elements, actions, purposes,
   people, components, owners, and processors. The judge awarded `supported` for
   partial overlap while the human reviewer required the complete claim.
2. **Answer-presence overreach.** The judge sometimes credited facts available in the
   evidence but absent from the answer, especially the voluntary-status caveats.
3. **Extra-claim under-detection.** The human noted 38 extra claims. The judge emitted
   only four across all baseline answers and none across multi-agent answers. This can
   inflate judge-derived factual precision because unsupported extras never enter its
   denominator.
4. **Citation disagreement is mostly secondary.** Citation-status agreement was 0%
   when verdicts differed, but 94.9% when both reviewers marked a claim supported.
   The primary problem is deciding whether a claim is present, not evaluating a
   mutually recognized citation.

## Human-Labeled Rubric Metrics

These exclude the 38 text-only extra claims and must not be interpreted as full
factual precision.

| Metric | Baseline | Multi-agent |
|---|---:|---:|
| Required-claim recall | 81.8% | 70.5% |
| Required-claim citation recall | 79.5% | 65.9% |
| Caveat coverage | 50.0% | 50.0% |
| Rubric-only citation precision | 95.7% | 85.1% |
| Forbidden-claim violations | 0 | 0 |

## Confidence-Recovery Plan

1. Split compound rubric items into independently scorable claims and explicitly state
   that every clause must appear for `supported`.
2. Require the judge to return an exact answer span and evidence IDs for every
   non-absent decision; reject unsupported spans during validation.
3. Add a separate claim-extraction pass before rubric alignment so extra claims cannot
   silently disappear. Human-label a small sample of extracted extras for precision
   and recall.
4. Rejudge the existing 24 answers under a versioned prompt and rubric. This requires
   judge calls only—no candidate-system rerun.
5. Repeat calibration. Run additional candidate repetitions only after agreement
   reaches both trust gates.
