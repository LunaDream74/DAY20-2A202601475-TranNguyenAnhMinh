# Human-Adjudicated Gold Evaluation

> This is a lab-created gold set, not an official benchmark. Results remain provisional until judge calibration passes 80% exact agreement and Cohen's kappa 0.70.

| Metric | Baseline | Multi-agent | Paired delta (95% bootstrap CI) |
|---|---:|---:|---:|
| grounded_f1 | 94.8% | 88.9% | -5.9% (-24.2%, +5.6%) |
| factual_precision | 98.6% | 91.7% | -6.9% (-25.0%, +4.2%) |
| required_claim_recall | 93.1% | 87.5% | -5.6% (-25.0%, +8.3%) |
| contradiction_rate | 1.4% | 0.0% | -1.4% (-4.2%, +0.0%) |
| citation_precision | 100.0% | 91.7% | -8.3% (-25.0%, +0.0%) |
| citation_recall | 90.3% | 87.5% | -2.8% (-22.2%, +11.1%) |
| caveat_coverage | 58.3% | 66.7% | +8.3% (-16.7%, +33.3%) |

## Operational metrics

| System | Mean latency | Total system tokens | Total judge tokens | Estimated total cost |
|---|---:|---:|---:|---:|
| baseline | 22.93s | 11032 | 15044 | $0.0035 |
| multi-agent | 44.49s | 25182 | 15590 | $0.0035 |

A system is not declared superior unless the paired interval excludes zero. Inspect claim-level judgments and human calibration before interpreting aggregates.
