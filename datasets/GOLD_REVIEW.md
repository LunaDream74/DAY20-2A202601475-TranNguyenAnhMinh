# Gold Dataset Review

`gold_research_eval.json` is a **draft lab gold set**, not an official benchmark. Its source
text is a concise reviewer-written summary, not a quotation.

Before running `evaluate-gold`, a human reviewer must open every source URL and check the title,
summary, required/optional/forbidden claims, caveat, and evidence links. Then change the case's
`status` to `approved` and add a stable `human_reviewer_id` plus an ISO-8601 `reviewed_at` value.
Do not approve a case with a dead source, ambiguous rubric, or time-sensitive claim.

Run `malab validate-gold-dataset --require-approved` after review. The loader verifies hashes,
unique IDs, references, and approval provenance. A changed evidence summary requires replacing
its SHA-256 hash; this makes silent edits detectable.

After evaluation, use `malab export-review-packet`, label all repetition-one outputs without
looking up their system identity, and import JSONL `HumanReview` records with
`malab import-human-labels --labels <path>`. Treat model-judge aggregates as trusted only when
exact agreement is at least 80% and Cohen's kappa is at least 0.70.
