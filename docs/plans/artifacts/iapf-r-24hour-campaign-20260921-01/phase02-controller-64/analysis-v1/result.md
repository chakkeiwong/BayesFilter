# Independent R campaign phase result

Phase: phase02-controller-64. Fixed-count CPU reference analysis.

| Data | d | Method | Complete | Mean ratio (95% interval) | SD (95% interval) | Mean N | Practical | Heuristic | Literal pattern |
|---|---:|---|---:|---|---|---:|---|---|---|
| 89600080 | 80 | qr | 64/64 | 1.0294 [0.9899, 1.071] | 0.17031 [0.1411, 0.196] | 1953.1 | False | veto | False |
| 89600080 | 80 | short_qr | 64/64 | 0.99611 [0.9324, 1.062] | 0.27112 [0.2119, 0.3251] | 1078.1 | True | passed observed screen | True |
| 89600080 | 80 | bpf | 64/64 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |
| 89600080 | 80 | fully_adapted | 64/64 | 0.62559 [0.2388, 1.168] | 1.9437 [0.3811, 3.056] | 5000 | N/A | comparator | N/A |
| 89600080 | 80 | sis | 64/64 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |
| 89700080 | 80 | qr | 64/64 | 0.98385 [0.9415, 1.027] | 0.17637 [0.1412, 0.2054] | 2000 | False | veto | False |
| 89700080 | 80 | short_qr | 64/64 | 1.0158 [0.9509, 1.081] | 0.2578 [0.2014, 0.3123] | 1062.5 | True | veto | True |
| 89700080 | 80 | bpf | 64/64 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |
| 89700080 | 80 | fully_adapted | 64/64 | 0.40915 [0.233, 0.6153] | 0.79172 [0.3739, 1.097] | 5000 | N/A | comparator | N/A |
| 89700080 | 80 | sis | 64/64 | 0 [0, 0] | 0 [0, 0] | 10000 | N/A | comparator | N/A |

A heuristic veto means an observed conditional loss to a simple comparator; it does not
establish population inferiority when the paired interval includes zero. Intervals are
pointwise, conditional on each fixed dataset, and are not simultaneous or cross-dataset inference.
Finite log ratios with exponentiation underflow are counted in summary.json. A displayed zero
sample SD for an underflowed baseline is not zero theoretical variance.

| Data | d | Method | Observed / paper SD | Observed / paper mean N | Observed / paper resampling | Whole-run seconds |
|---|---:|---|---|---|---|---:|
| 89600080 | 80 | qr | 0.17031 / 0.35 | 1953.1 / 1142 | 37.219 / 71.88 | 28.963 |
| 89600080 | 80 | short_qr | 0.27112 / 0.35 | 1078.1 / 1142 | 37.391 / 71.88 | 22.071 |
| 89600080 | 80 | bpf | 0 / unreported | 10000 / unreported | 99 / unreported | 6.5104 |
| 89600080 | 80 | fully_adapted | 1.9437 / unreported | 5000 / unreported | 99 / unreported | 3.6976 |
| 89600080 | 80 | sis | 0 / unreported | 10000 / unreported | 0 / unreported | 7.1732 |
| 89700080 | 80 | qr | 0.17637 / 0.35 | 2000 / 1142 | 37.438 / 71.88 | 23.65 |
| 89700080 | 80 | short_qr | 0.2578 / 0.35 | 1062.5 / 1142 | 37.078 / 71.88 | 18.117 |
| 89700080 | 80 | bpf | 0 / unreported | 10000 / unreported | 99 / unreported | 5.3102 |
| 89700080 | 80 | fully_adapted | 0.79172 / unreported | 5000 / unreported | 99 / unreported | 3.0928 |
| 89700080 | 80 | sis | 0 / unreported | 10000 / unreported | 0 / unreported | 5.9611 |

Paper values come from the preserved GJL first-study tables. Different datasets and
QR's different fitting objective prevent literal equivalence claims. Whole-run timing
includes learning and final evaluation and is not matched-cost or cross-machine evidence.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| 89600080 qr: not promoted | False | heuristic=veto; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89600080 short_qr: passes reference screen | True | heuristic=passed observed screen; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89700080 qr: not promoted | False | heuristic=veto; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |
| 89700080 short_qr: not promoted | True | heuristic=veto; fits=True; tails=True | Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Exact record identity, source hashes, seeds, fit/tail and missing-pair status are preserved per cell. |
| Statistically supported ranking | No global ranking. Paired pointwise intervals in summary.json support only their specified contrast and fixed data. |
| Descriptive-only differences | Raw means, extreme ratios, resampling counts and unmatched-cost timing do not establish superiority. |
| Default-readiness | No algorithm or production default changes. QR differs from paper Equation15. |
| Next evidence needed | Complete the remaining frozen campaign; exact original settings/data and multivariate full-filter R/TF parity remain open. |

Post-run skeptical review: the strongest alternative explanation is data-specific Monte Carlo
variation or unobserved rare large weights. Larger independent fixed-count runs could overturn
a small-sample screen. The weakest evidence is literal matching to unpublished original data
and implementation choices. These results do not prove correctness or failure of LEDH, KDM or HMC.
